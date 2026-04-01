from utils.config import TOP_K
from utils.vector_store import load_vector_store
from utils.llm import get_llm
from langchain_classic.chains import ConversationalRetrievalChain
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

# Template for condensing history + new question into a standalone question
condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
################################################################################

def get_qa_chain(repo_id):
    db = load_vector_store(repo_id)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT
    )

    return qa

def ask_question(query, repo_id, chat_history_dicts=None):
    db = load_vector_store(repo_id)
    
    # Format chat history for LangChain [(human, ai), ...]
    chat_history = []
    if chat_history_dicts:
        # We need to pair them up or handle them as messages
        # Simple approach: assume they are in order and user/ai alternate
        # Actually, ConversationalRetrievalChain expects a list of tuples
        curr_user = None
        for msg in chat_history_dicts:
            if msg['role'] == 'user':
                curr_user = msg['content']
            elif msg['role'] == 'ai' and curr_user is not None:
                chat_history.append((curr_user, msg['content']))
                curr_user = None

    # Retrieve docs with similarity scores (for debugging output)
    docs_and_scores = db.similarity_search_with_score(query, k=TOP_K)
    # Sort by distance (ascending = most similar first)
    docs_and_scores.sort(key=lambda x: x[1])
    
    source_docs = []
    for doc, score in docs_and_scores:
        # Convert distance to similarity score (1 - distance for cosine)
        try:
            dist = float(score)
            doc.metadata["distance"] = dist
            doc.metadata["score"] = max(0, 1 - dist)
        except (ValueError, TypeError):
            doc.metadata["score"] = 0.0
        source_docs.append(doc)

    qa = get_qa_chain(repo_id)
    result = qa.invoke({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    
    # Display relevant chunks in terminal
    print("\n" + "="*60)
    print(f"TOP {len(source_docs)} RELEVANT CHUNKS (SIMILARITY):")
    print("="*60)
    for i, doc in enumerate(source_docs):
        score_val = doc.metadata.get('score')
        dist_val = doc.metadata.get('distance', 0.0)
        # Use "Similarity Score" and include Raw distance for debugging
        score_str = f" | Score: {score_val:.4f} (Dist: {dist_val:.4f})" if score_val is not None else ""
        print(f"\n--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}{score_str}) ---")
        print(doc.page_content)
    print("="*60 + "\n")

    sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]

    return answer, list(set(sources))
