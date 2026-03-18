import os
import time
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, VECTORSTORE_DIR, 
    EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_DELAY
)

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
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        return None

    print(f"Starting vector store creation for {repo_id} with {total_chunks} chunks.")
    
    # Process first batch to initialize Chroma
    first_batch = chunks[:EMBEDDING_BATCH_SIZE]
    db = Chroma.from_documents(
        first_batch,
        embeddings,
        persist_directory=persist_dir
    )

    # Process remaining chunks in batches with delay
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
        embedding_function=embeddings
    )
