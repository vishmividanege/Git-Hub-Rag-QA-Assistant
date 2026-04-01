import os
import time
import shutil
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from utils.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, VECTORSTORE_DIR, 
    EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_DELAY
)

def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=api_key)

def create_vector_store(documents, repo_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    persist_dir = os.path.join(VECTORSTORE_DIR, repo_id)
    

    if os.path.exists(persist_dir):
        print(f"Forcing clean start for {repo_id} to update metric/schema...")
        try:
            shutil.rmtree(persist_dir)
            print("Existing vector store removed successfully.")
        except Exception as e:
            print(f"Warning: Could not remove directory: {e}")
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        return None

    print(f"Starting vector store creation for {repo_id} with {total_chunks} chunks.")
    
    
    first_batch = chunks[:EMBEDDING_BATCH_SIZE]
    db = Chroma.from_documents(
        first_batch,
        embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    
    for i in range(EMBEDDING_BATCH_SIZE, total_chunks, EMBEDDING_BATCH_SIZE):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
        print(f"Embedding batch {i} to {min(i + EMBEDDING_BATCH_SIZE, total_chunks)}...")
        time.sleep(EMBEDDING_BATCH_DELAY)
        db.add_documents(batch)

    print(f"Successfully created vector store for {repo_id}")
    return db

def load_vector_store(repo_id):
    embeddings = get_embeddings()
    persist_dir = os.path.join(VECTORSTORE_DIR, repo_id)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
