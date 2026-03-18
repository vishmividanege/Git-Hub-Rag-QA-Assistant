# utils/vector_store.py
import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, VECTORSTORE_DIR, EMBEDDING_MODEL_NAME

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

def create_vector_store(documents, repo_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    persist_dir = os.path.join(VECTORSTORE_DIR, repo_id)
    
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir
    )

    return db

def load_vector_store(repo_id):
    embeddings = get_embeddings()
    persist_dir = os.path.join(VECTORSTORE_DIR, repo_id)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
