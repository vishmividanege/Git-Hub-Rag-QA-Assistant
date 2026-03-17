# utils/rag_chain.py
from langchain_classic.chains import RetrievalQA
from utils.vector_store import load_vector_store
from utils.llm import get_llm

def get_qa_chain(repo_id):
    db = load_vector_store(repo_id)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        return_source_documents=True
    )

    return qa

def ask_question(query, repo_id):
    qa = get_qa_chain(repo_id)
    result = qa({"query": query})

    answer = result["result"]
    sources = [doc.metadata["source"] for doc in result["source_documents"]]

    return answer, list(set(sources))
