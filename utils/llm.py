import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

from utils.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS

def get_llm(model_name=LLM_MODEL_NAME):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the sidebar or a .env file.")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_NEW_TOKENS,
        convert_system_message_to_human=True
    )
