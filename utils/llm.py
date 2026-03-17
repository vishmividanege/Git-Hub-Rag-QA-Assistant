from langchain_community.llms import Ollama

def get_llm(model_name="llama3"):
    try:
        return Ollama(model=model_name)
    except Exception:
        # Fallback to a common model if llama3 fails or isn't present
        return Ollama(model="mistral")
