import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

from utils.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS

def get_llm(model_name=LLM_MODEL_NAME):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in a .env file.")
    
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_NEW_TOKENS
    )
