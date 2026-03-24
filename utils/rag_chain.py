
from utils.config import TOP_K
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
    retriever = db.as_retriever(
        search_type="similarity",
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
    
    # Retrieve docs with similarity scores
    docs_and_scores = db.similarity_search_with_score(query, k=TOP_K)
    # Sort by distance (ascending = most similar first)
    docs_and_scores.sort(key=lambda x: x[1])
    source_docs = [doc for doc, score in docs_and_scores]
    for doc, score in docs_and_scores:
        # Convert distance to similarity score (1 - distance)
        doc.metadata["score"] = max(0, 1 - score)

    qa = get_qa_chain(repo_id)
    result = qa.invoke({"query": query})
    answer = result["result"]
    
    # Display relevant chunks in terminal
    print("\n" + "="*60)
    print(f"TOP {len(source_docs)} RELEVANT CHUNKS (SIMILARITY):")
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
