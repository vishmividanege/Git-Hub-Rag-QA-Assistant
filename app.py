import streamlit as st
import os
from utils.github_loader import clone_repo, get_repo_id
from utils.file_loader import load_files
from utils.vector_store import create_vector_store
from utils.rag_chain import ask_question

# --- Page Config ---
st.set_page_config(
    page_title="GitHub RAG Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        opacity: 0.9;
    }
    
    .stTextInput>div>div>input {
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .source-box {
        background-color: rgba(51, 65, 85, 0.4);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    
    .header-text {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='color: #60a5fa;'>Settings</h1>", unsafe_allow_html=True)
    
    # Google API Key Handling
    env_key = os.getenv("GOOGLE_API_KEY")
    api_key = st.text_input(
        "Enter Google API Key", 
        value=env_key if env_key else "",
        type="password", 
        help="Get your key from https://aistudio.google.com/app/apikey. If you set it in .env, it will appear here."
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key is Active!")
    else:
        st.warning("Please enter your Google API Key or set it in the .env file.")
    
    st.info("The assistant is now powered by Google Gemini 1.5 Flash.")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

# --- Main UI ---
st.markdown("<div class='header-text'>GitHub RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Chat with any GitHub repository using Google Gemini.</p>", unsafe_allow_html=True)

# Repository Input Section
col1, col2 = st.columns([3, 1])
with col1:
    repo_url = st.text_input("Enter GitHub Repository URL", placeholder="https://github.com/username/repo")
with col2:
    process_btn = st.button("Deep Scan Repo", use_container_width=True)

if process_btn:
    if repo_url:
        try:
            repo_id = get_repo_id(repo_url)
            with st.status("Analyzing Repository...", expanded=True) as status:
                st.write("Cloning repository...")
                path = clone_repo(repo_url)
                
                st.write("Parsing files...")
                docs = load_files(path)
                
                if not docs:
                    st.error("No supported files found in the repository.")
                    status.update(label="Analysis Failed", state="error")
                else:
                    st.write(f"Indexing {len(docs)} documents...")
                    create_vector_store(docs, repo_id)
                    st.session_state['repo_id'] = repo_id
                    status.update(label="Intelligence Ready!", state="complete", expanded=False)
                    st.toast("Repository fully indexed and ready for queries!")
        except Exception as e:
            st.error(f"Error processing repository: {e}")
    else:
        st.warning("Please enter a valid GitHub repository URL.")

# Chat Section
st.divider()

if 'repo_id' in st.session_state:
    st.markdown("### Ask your question")
    question = st.text_input("E.g., 'How does the authentication flow work?' or 'Explain the project structure.'", key="query_input")
    
    if st.button("Ask Intelligence"):
        if question:
            with st.spinner("AI is thinking..."):
                try:
                    answer, sources = ask_question(question, st.session_state['repo_id'])
                    
                    st.markdown("---")
                    st.markdown("#### Intelligence Response")
                    st.write(answer)
                    
                    if sources:
                        with st.expander("View Grounding Sources"):
                            for s in sources:
                                st.code(s)
                except Exception as e:
                    st.error(f"An error occurred during query: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("👋 Enter a GitHub URL above and click 'Deep Scan Repo' to start chatting.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Built with ❤️ using LangChain & Streamlit</p>", unsafe_allow_html=True)