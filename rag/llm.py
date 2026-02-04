from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from operator import itemgetter
from typing import Dict, List
from functools import lru_cache 

from utills.dictionary import get_dictionary_bundle, normalize_query
from utills.deictic import has_deictic_expression


def get_document_list() -> RecursiveCharacterTextSplitter:
    """마크다운/텍스트 문서를 쪼개서 Document 리스트로 만든다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, #임베딩 성능을 위해 쪼개기
        chunk_overlap=200
    )

    #TextLoader → Document → Chroma → Retriever → RAG → Chain
    #document로 관리해야 용이
    loader = TextLoader("./edu_markdown.txt", encoding="utf-8")
    docs = loader.load() # str To document
    return text_splitter.split_documents(docs)


def _format_docs(docs:List[Document]) -> str:
    """
    여러 Document 객체의 page_content를
    두 줄 개행("\n\n") 으로 연결하여 하나의 문자열로 반환한다.    
    """
    return "\n\n".join(d.page_content for d in docs)

@lru_cache(maxsize=1)
def get_llm(model='exaone3.5:2.4b'):
    """LLM 모델을 가져온다. (ollma로 로컬 llm 실행 / LG모델 사용, 한국어에 특화 (1.6 GB))
    """
    #llm = ChatOpenAI(model=model, temperature=0.2) : GPT 사용시
    llm = ChatOllama(
        model=model, 
        base_url="http://127.0.0.1:11434",
        temperature=0
        )

    return llm


@lru_cache(maxsize=4)
def get_embeddings(model='intfloat/multilingual-e5-large-instruct') -> HuggingFaceEmbeddings:
    """임베딩 모델을 가져온다."""
    return HuggingFaceEmbeddings(model_name=model)


# ----------------------------
# Vector DB (Ingest vs Load)
# ----------------------------
@lru_cache(maxsize=1)
def build_database():
    """
    Chroma를 현재 존재하는 디렉토리에 빌드한다.
    """
    collection_name = 'Chroma_edu'

    database = Chroma.from_documents(
        documents=get_document_list(),
        embedding_function=get_embeddings(),
        collection_name=collection_name,
        persist_directory='./chroma_huggingface'
    )

    return database

@lru_cache(maxsize=1)
def load_db():
    """
    빌드된 Chroma database를 가져온다.
    """
    database = Chroma(
        collection_name='Chroma_edu',
        persist_directory="./chroma_huggingface",
        embedding_function=get_embeddings(),
    )

    return database

# ----------------------------
# Build retriever (HF Embeddings + Chroma)
# ----------------------------
def get_retriever():  # 
    """load된 db를 가지고 retriver을 생성한다."""
    vectorstore = load_db()
    return vectorstore.as_retriever(search_kwargs={"k":3})

#session store 만들어두기
_STORE: Dict[str, ChatMessageHistory] = {}
# private 변수명 지정시 _ 를 붙임, : 타입 명시


# ----------------------------
# Session history store
# ----------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """history_aware를 추가하기 위해 sessionId의 history를 저장 | session 메모리에 적재"""
    if session_id not in _STORE:        
        _STORE[session_id] = ChatMessageHistory()
    return _STORE[session_id]

# ----------------------------
# 기본 답변 프롬프트 생성 
# ----------------------------
@lru_cache(maxsize=1)
def get_answer_prompt():
    return ChatPromptTemplate.from_messages([

        ("system",
        """
            추가 규칙:
            - 답변은 반드시 아래 형식만 사용한다.
            - 형식 밖의 문장은 절대 출력하지 않는다.
            - Context에 없는 내용(상식/추론/추가 설명/예시/주의사항)은 금지.
            - 답변 섹션의 모든 문장은 "근거 인용" 섹션에서 직접 뒷받침되어야 한다.
            - 답변은 한국어를 사용하여 출력한다.
            - 근거 인용이 불가능하면 아래 문장만 출력한다: "해당 질문은 제공된 문서 범위에 포함되지 않습니다.
            문서 관련 질문을 입력해 주세요"
            - [답변] 

            - [근거 인용]
            - (1~3문장 이내로 간단히 작성)
            - 각 인용문은 반드시 한 줄에 하나씩 출력한다.
            - 각 인용문 사이에는 줄바꿈(\\n)을 넣는다.
            - 한 줄에 두 개 이상의 인용문을 작성하지 않는다.
            - "Context에서 그대로 복사한 문장" 형태로 작성한다.,

            추가 규칙:
            - 질문이 Context와 무관하거나 정보 제공 범위를 벗어나면,
            아래 문장만 출력한다:

            "해당 질문은 제공된 문서 범위에 포함되지 않습니다.
            문서 관련 질문을 입력해 주세요."

                    ("human",
                  
            Question:
            {question}

            Context:
            {context}

            Answer (with quotes from Context):
                    """.strip()
                    )
    ])

# ----------------------------
# history 답변 프롬프트 생성
# ----------------------------
def get_rewrite_prompt() -> ChatPromptTemplate:
    """
    history prompt를 생성한다.
    system : 무엇을 해야할지 역할
    """
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, which may reference context in the chat history, "
                "rewrite the question into a standalone question that can be understood without the chat history. "
                "Do NOT answer the question. Return only the standalone question, with no extra text.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return rewrite_prompt


def check_deictic(query:str):
    """지시어 포함 체크 후 사용할 retriever를 확인"""
    deictic_yn = has_deictic_expression(query)

    print("deictic_yn :", deictic_yn)

    return get_history_rag_chian() if deictic_yn else get_init_rag_chain() 

# chain으로 분기 (history, 최초 chain으로 분기)
# ----------------------------
# 기존 chain 
# ----------------------------
def get_init_rag_chain():
    """일반 rag_chain을 생성한다"""
    retriever = get_retriever()
    llm = get_llm()

    print("retriever :",retriever)

    answer_prompt = get_answer_prompt()
    parser = StrOutputParser() # ai_msg TO text

    #input : {"input": "", "chat_history" " [...]"}
    #python dict로 만드는 건가
    pre = {
        "question": itemgetter("input"),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        # Use invoke() explicitly to be robust across retriever implementations
        "docs": RunnableLambda(lambda x: retriever.invoke(x["input"])),

    }

    print("pre contents:", retriever.invoke("메가웨어 설치 순서 알려줘"))
    
    to_prompt_vars = {
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        "context": itemgetter("docs") | RunnableLambda(_format_docs),
    }

    pre_r = RunnableMap(pre) #이 dict는 데이터를 의미 X
                             #입력을 받아 각 key별 Runnable을 실행해서 결과 dict를 만드는 RunnableMap이라고 명시
    tp_r  = RunnableMap(to_prompt_vars)
  
    chain = pre_r | tp_r | answer_prompt | llm | parser
    return chain


def get_history_rag_chian():
    """history rag_chain을 생성한다"""
    #llm 먼저 호출
    llm = get_llm()
    retriever = get_retriever()
    parser = StrOutputParser()

    rewrite_prompt = get_rewrite_prompt()
    answer_prompt = get_answer_prompt()
    rewrite_chain = rewrite_prompt | llm | parser

    pre = {
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        "original_question": itemgetter("input"),
        "standalone_question": rewrite_chain,
    }

    retrieve = {
        "chat_history": itemgetter("chat_history"),
        # 원 질문은 그대로 사용 (원하시면 standalone_question으로 바꿔도 됨)
        "question": itemgetter("original_question"),
        "docs": RunnableLambda(lambda x: retriever.invoke(x["standalone_question"])),
    }

    to_prompt_vars = {
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        "context": itemgetter("docs") | RunnableLambda(_format_docs),      
    }

    pre_r = RunnableMap(pre)

    retrieve_r = RunnableMap(retrieve)

    to_prompt_vars_r = RunnableMap(to_prompt_vars)

    base = pre_r | retrieve_r | to_prompt_vars_r | answer_prompt | llm

    chain_with_history = RunnableWithMessageHistory(
            base,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        
    )

    return chain_with_history | parser


def get_ai_response(user_message):
    """사용자 질문을 받아 RAG 기반 답변을 생성한다."""
    bundle = get_dictionary_bundle() # bundle 가져오기
    query = normalize_query(user_message, bundle) # 신규 질문 작성

    rag_chain = check_deictic(query)

   # prompt = get_retriever_prompt()
   # combine_docs_chain = create_stuff_documents_chain(get_llm(), prompt)
  #  rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    ai_response = rag_chain.stream(
        {
            "input": query, "chat_history": []
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response