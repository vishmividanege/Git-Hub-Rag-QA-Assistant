
from utils.config import TOP_K, SEARCH_TYPE
from utils.vector_store import load_vector_store
from utils.llm import get_llm
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# prompt template for better RAG responses ---------------------------------------------------------
prompt_template = """You are the GitHub RAG Assistant. Use the following pieces of context to answer the user's question.
If the context contains lists of features, requirements, or steps, please present them clearly using markdown bullet points.
If you don't know the answer or the context doesn't provide enough information, honestly state that you don't know.
Be thorough and provide full code implementations if requested.

Context:
{context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
################################################################################

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
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa

def ask_question(query, repo_id):
    db = load_vector_store(repo_id)
    
    # Manual retrieval to get documentation and scores
    if SEARCH_TYPE == "mmr":
        source_docs = db.max_marginal_relevance_search(query, k=TOP_K)
        # Fetch similarity scores for a larger pool to match against MMR docs
        candidates_with_scores = db.similarity_search_with_score(query, k=100)
        # Convert distance to similarity score (1 - distance)
        score_map = {doc.page_content: max(0, 1 - score) for doc, score in candidates_with_scores}
        
        for doc in source_docs:
            doc.metadata["score"] = score_map.get(doc.page_content, "N/A")
    else:
        # For similarity, we can get scores directly
        docs_and_scores = db.similarity_search_with_score(query, k=TOP_K)
        # Sort by score (distance) just in case, though Chroma usually returns them ordered
        docs_and_scores.sort(key=lambda x: x[1])
        source_docs = [doc for doc, score in docs_and_scores]
        for i, (doc, score) in enumerate(docs_and_scores):
            # Convert distance to similarity score (1 - distance)
            similarity = max(0, 1 - score)
            doc.metadata["score"] = similarity

    qa = get_qa_chain(repo_id)
    result = qa.invoke({"query": query})
    answer = result["result"]
    
    # Display relevant chunks in terminal
    print("\n" + "="*60)
    print(f"TOP {len(source_docs)} RELEVANT CHUNKS ({SEARCH_TYPE.upper()}):")
    print("="*60)
    for i, doc in enumerate(source_docs):
        score_val = doc.metadata.get('score', 'N/A')
        # Use "Similarity Score" as requested by the user
        score_str = f" | Similarity Score: {score_val:.4f}" if isinstance(score_val, float) else ""
        print(f"\n--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}{score_str}) ---")
        print(doc.page_content)
    print("="*60 + "\n")

    sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]

    return answer, list(set(sources))
