# utils/vector_store.py
import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, DB_PATH

def create_vector_store(documents, repo_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_dir = os.path.join(DB_PATH, repo_id)
    
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir
    )

    return db

def load_vector_store(repo_id):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = os.path.join(DB_PATH, repo_id)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
