from utils.config import TOP_K
from utils.vector_store import load_vector_store
from utils.llm import get_llm
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage



contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


qa_system_prompt = """You are the GitHub RAG Assistant. Use the following pieces of context to answer the user's question.
If the context contains lists of features, requirements, or steps, please present them clearly using markdown bullet points.
If you don't know the answer or the context doesn't provide enough information, honestly state that you don't know.
Be thorough and provide full code implementations if requested.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def get_qa_chain(repo_id):
    llm = get_llm()
    db = load_vector_store(repo_id)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

   
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def ask_question(query, repo_id, chat_history_dicts=None):
    
    chat_history = []
    if chat_history_dicts:
        for msg in chat_history_dicts:
            if msg['role'] == 'user':
                chat_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'ai':
                chat_history.append(AIMessage(content=msg['content']))


    rag_chain = get_qa_chain(repo_id)
    
    
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    answer = result["answer"]
    source_docs = result.get("context", [])
    

    db = load_vector_store(repo_id)
    
    docs_with_scores = db.similarity_search_with_score(query, k=len(source_docs))

    
    print("\n" + "="*60)
    print(f"TOP {len(source_docs)} RELEVANT CHUNKS (SIMILARITY):")
    print("="*60)
    
   
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\n--- Chunk {i+1} (Score: {score:.4f}, Source: {doc.metadata.get('source', 'Unknown')}) ---")
        print(doc.page_content)
    print("="*60 + "\n")

    sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]

    return answer, list(set(sources))
