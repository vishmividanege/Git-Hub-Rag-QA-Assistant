
from utils.config import TOP_K, SEARCH_TYPE
from utils.vector_store import load_vector_store
from utils.llm import get_llm
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma

def get_qa_chain(repo_id):
    db = load_vector_store(repo_id)
    # Map "cosine similarity" to "similarity" for LangChain's retriever
    retriever_type = "similarity" if SEARCH_TYPE == "cosine similarity" else SEARCH_TYPE
    
    retriever = db.as_retriever(
        search_type=retriever_type,
        search_kwargs={"k": TOP_K}
    )

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=True
    )

    return qa

def ask_question(query, repo_id):
    db = load_vector_store(repo_id)
    
    # Manual retrieval to get documentation and scores
    if SEARCH_TYPE == "mmr":
        source_docs = db.max_marginal_relevance_search(query, k=TOP_K)
        # Fetch similarity scores for a larger pool to match against MMR docs
        candidates_with_scores = db.similarity_search_with_score(query, k=100)
        score_map = {doc.page_content: score for doc, score in candidates_with_scores}
        
        for doc in source_docs:
            doc.metadata["score"] = score_map.get(doc.page_content, "N/A")
    else:
        # For similarity, we can get scores directly
        docs_and_scores = db.similarity_search_with_score(query, k=TOP_K)
        source_docs = [doc for doc, score in docs_and_scores]
        for i, (doc, score) in enumerate(docs_and_scores):
            doc.metadata["score"] = score

    qa = get_qa_chain(repo_id)
    result = qa.invoke({"query": query})
    answer = result["result"]
    
    # Display relevant chunks in terminal
    print("\n" + "="*60)
    print(f"TOP {len(source_docs)} RELEVANT CHUNKS ({SEARCH_TYPE.upper()}):")
    print("="*60)
    for i, doc in enumerate(source_docs):
        score_val = doc.metadata.get('score', 'N/A')
        score_str = f" | MMR Score: {score_val:.4f}" if isinstance(score_val, float) else ""
        print(f"\n--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}{score_str}) ---")
        print(doc.page_content)
    print("="*60 + "\n")

    sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]

    return answer, list(set(sources))
