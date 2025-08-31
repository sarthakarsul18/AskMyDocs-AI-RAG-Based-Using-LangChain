import streamlit as st
import requests
from rag_backend import EasyRAG

# ===== Page Setup =====
st.set_page_config(page_title="AskMyDocs AI", layout="wide")

# ===== Initialize =====
if 'rag' not in st.session_state:
    st.session_state.rag = EasyRAG()
if 'chat' not in st.session_state:
    st.session_state.chat = []
if 'loaded' not in st.session_state:
    st.session_state.loaded = False


st.sidebar.title("Setup")
api_key = st.sidebar.text_input("HuggingFace API Key", type="password")

def verify_hf_key(key: str) -> bool:
    """Check if Hugging Face API key is valid"""
    try:
        resp = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {key}"}
        )
        return resp.status_code == 200
    except Exception:
        return False

if api_key:
    if verify_hf_key(api_key):
        st.session_state.rag.set_api_key(api_key)
        st.sidebar.success("✅ API key set successfully!")
    else:
        st.sidebar.error("❌ Invalid API key. Please try again.")

files = st.sidebar.file_uploader("Upload PDF or TXT", type=['pdf','txt'], accept_multiple_files=True)

if files and api_key:
    if st.sidebar.button("Process Files"):
        success, msg = st.session_state.rag.process_documents(files)
        st.session_state.loaded = success
        st.sidebar.success(msg if success else "Failed")

# ===== Main =====
st.title("🤖 AskMyDocs AI")
st.write("Upload files and ask AI questions!")


with st.expander("ℹ️ About AskMyDocs AI"):
    st.markdown("""
---
### 🌟 About AskMyDocs AI
Built with passion by **Sarthak Arsul** 🧑‍💻  

This bot brings your documents to life using **Retrieval-Augmented Generation (RAG)**.  
It combines:  
- 🧠 **Hugging Face LLMs** for intelligent answers  
- 📚 **FAISS Vector DB** for fast document search  
- 🎨 **Streamlit** for a clean and interactive interface  

Just upload a PDF or TXT file, ask questions in plain English, and watch your data talk back to you! 🚀  

---
""")

# Question input
if st.session_state.loaded:
    question = st.text_input("Ask a question:")
    if st.button("Ask") and question:
        answer = st.session_state.rag.ask_question(question)
        st.session_state.chat.append({"q": question, "a": answer})

# Show chat (last 5)
for chat in reversed(st.session_state.chat[-5:]):
    st.markdown(f"**You:** {chat['q']}")
    st.markdown(f"**AI:** {chat['a']}")
    st.markdown("---")


if not api_key:
    st.info("Enter API key in sidebar")
elif not st.session_state.loaded:
    st.info("Upload and process documents first")
