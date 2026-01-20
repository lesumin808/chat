from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

from utills.dictionary import get_dictionary_bundle
from utills.dictionary import normalize_query
from utills.deictic import has_deictic_expression
#from langchain_ollama import ChatOllama
from functools import lru_cache 

def get_document_list() -> TextLoader:
    """마크다운/텍스트 문서를 쪼개서 Document 리스트로 만든다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, #임베딩 성능을 위해 쪼개기
        chunk_overlap=200
    )

    loader = TextLoader("./edu_markdown.txt", encoding="utf-8")
    document_list = loader.load_and_split(text_splitter)

    return document_list


@lru_cache(maxsize=4)
def get_llm(model='gpt-4.1-mini'):
    """LLM 모델을 가져온다."""
    #llm = ChatOllama(model="llama3.2:latest", base_url="http://127.0.0.1:11434")
    llm = ChatOpenAI(model=model, temperature=0.2)
    return llm


@lru_cache(maxsize=4)
def get_embeddings(model='intfloat/multilingual-e5-large-instruct') -> HuggingFaceEmbeddings:
    """임베딩 모델을 가져온다."""
    return HuggingFaceEmbeddings(model_name=model)


@lru_cache
def get_retriever_prompt():
    """retrieval QA prompt를 가져온다."""
    return hub.pull("langchain-ai/retrieval-qa-chat")


@lru_cache(maxsize=1)
def get_database():
    """
    Chroma DB를 준비한다.
    - 지금은 from_documents로 생성(간단 버전)
    - 운영에선 ingest(생성)과 load(로드)를 분리 추천
    """
    collection_name = 'Chroma_edu'

    database = Chroma.from_documents(
        documents=get_document_list(),
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory='./chroma_huggingface'
    )

    return database

#session store 만들어두기
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """history_aware를 추가하기 위해 sessionId의 history를 저장 | session 메모리에 적재"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_history_retriver():
    """histroyr retriver 생성"""
    retriever = get_retriever()
    llm = get_llm()

    # history-aware란? 이 질문이 이전 대화 맥락을 참고하고 있는가?
    # => 질문을 독립 질문으로 재작성하는 것
    # => 이전 대화를 참고해서 지금 질문을 혼자서도 이해 가능한 질문으로 바꿔라.
    # LLM을 이용한 질문 의미 복원 기술
    # 질문 재작성 => 대화 맥락이 섞인 질문을 “독립 질문”으로 변환
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "“Return only the standalone question, with no extra text"
    )

    # create_history_aware_retriever
    # 사용자 질문, ai 메세지를 추가
    # 검색준비 상태
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"), #사용자가 이런 input을 넣으면 새롭게 질문을 만드는 것
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    return history_aware_retriever


def get_retriever():
    """Retriever를 준비한다."""
    database = get_database()
    return database.as_retriever(search_kwargs={"k": 3})


def check_deictic(query:str):
    """지시어 포함 체크 후 사용할 retriever를 확인"""
    deictic_yn = has_deictic_expression(query)

    if deictic_yn:
        return get_history_rag_chian()
    else:
        return get_init_rag_chain()
    

# chain으로 분기 (history, 최초 chain으로 분기)
def get_init_rag_chain():
    """일반 rag_chain을 생성한다"""
    combine_docs_chain = create_stuff_documents_chain(get_llm(), get_retriever_prompt())
    return create_retrieval_chain(get_retriever(), combine_docs_chain).pick('answer')


def get_history_rag_chian():
    """history rag_chain을 생성한다"""
    #llm 먼저 호출
    combine_docs_chain = create_stuff_documents_chain(get_llm(), get_retriever_prompt())
    history_rag_chain = create_retrieval_chain(get_history_retriver(), combine_docs_chain)
    return RunnableWithMessageHistory(
            history_rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
    ).pick('answer')  # ui출력방식을 설정

def get_ai_response(user_message):
    """사용자 질문을 받아 RAG 기반 답변을 생성한다."""
    bundle = get_dictionary_bundle() # bundle 가져오기
    query = normalize_query(user_message, bundle) # 신규 질문 작성

    print(query)

    rag_chain = check_deictic(query)

   # prompt = get_retriever_prompt()
   # combine_docs_chain = create_stuff_documents_chain(get_llm(), prompt)
  #  rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    ai_response = rag_chain.stream(
        {
            "input": query
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response