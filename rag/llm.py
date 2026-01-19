from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from utill.dictionary import get_dictionary_bundle
from utill.dictionary import normalize_query
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
    return ChatOpenAI(model=model, temperature=0.2)


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


def get_retriever():
    """Retriever를 준비한다."""
    database = get_database()
    return database.as_retriever(search_kwargs={"k": 1})


def get_ai_response(user_message):
    """사용자 질문을 받아 RAG 기반 답변을 생성한다."""
    bundle = get_dictionary_bundle() # bundle 가져오기
    query = normalize_query(user_message, bundle) # 신규 질문 작성


    retriever = get_retriever()

    prompt = get_retriever_prompt()
    combine_docs_chain = create_stuff_documents_chain(get_llm(), prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = rag_chain.invoke({"input": query})

    return response["answer"]