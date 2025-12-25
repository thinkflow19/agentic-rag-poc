"""Streamlit chat UI for the agentic RAG agent."""

import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from app.core.agent.agent import create_agent_executor
from app.core.ingestion.ingest import ingest_from_url, process_document
from app.core.tools.document_search import clear_retriever_cache
from app.config import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    with st.spinner("Initializing agent..."):
        try:
            st.session_state.agent = create_agent_executor()
            st.session_state.agent_ready = True
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.session_state.agent_ready = False

st.title("🤖 Agentic RAG Chat")
st.markdown("""
Ask questions about the ingested documents or general knowledge.
The agent will search documents and the internet to provide grounded answers with citations.
""")

with st.sidebar:
    st.markdown("### 🤖 Agentic RAG")
    st.markdown("---")
    
    st.markdown("**Tools:**")
    st.markdown("- 📄 Document Search")
    st.markdown("- 🌐 Internet Search")
    
    st.markdown("---")
    
    st.markdown("### 📚 Add Documents")
    st.caption("⚠️ Note: Adding a new document will replace the existing index.")
    
    tab1, tab2 = st.tabs(["📄 Upload PDF", "🔗 From URL"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF file to index and search"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Index Document", use_container_width=True, type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_path = Path(tmp_file.name)
                            tmp_file.write(uploaded_file.getbuffer())
                        
                        try:
                            artifacts_dir = Config.ARTIFACTS_DIR
                            result = process_document(tmp_path, artifacts_dir)
                            
                            st.success(f"✅ Document indexed successfully!")
                            st.info(f"""
                            - **Pages**: {result['pages']}
                            - **Chunks**: {result['chunks']}
                            - **Method**: {result['extraction_method']}
                            """)
                            
                            clear_retriever_cache()
                            st.success("🔄 Retriever cache cleared. New document will be available for queries.")
                            
                        finally:
                            if tmp_path.exists():
                                tmp_path.unlink()
                                
                    except Exception as e:
                        st.error(f"❌ Error indexing document: {str(e)}")
                        st.exception(e)
    
    with tab2:
        url_input = st.text_input(
            "PDF URL",
            placeholder="https://example.com/document.pdf",
            help="Enter a URL to a PDF document"
        )
        
        if url_input:
            if st.button("🔗 Index from URL", use_container_width=True, type="primary"):
                with st.spinner("Downloading and processing document..."):
                    try:
                        artifacts_dir = Config.ARTIFACTS_DIR
                        result = ingest_from_url(url_input, artifacts_dir)
                        
                        st.success(f"✅ Document indexed successfully!")
                        st.info(f"""
                        - **Pages**: {result['pages']}
                        - **Chunks**: {result['chunks']}
                        - **Method**: {result['extraction_method']}
                        """)
                        
                        clear_retriever_cache()
                        st.success("🔄 Retriever cache cleared. New document will be available for queries.")
                        
                    except Exception as e:
                        st.error(f"❌ Error indexing document: {str(e)}")
                        st.exception(e)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if (message["role"] == "assistant" 
            and "tools_used" in message 
            and message["tools_used"] is not None 
            and len(message["tools_used"]) > 0):
            with st.expander("🔍 Tools Used"):
                for tool in message["tools_used"]:
                    st.code(f"{tool['name']}({tool.get('args', {})})", language="python")

if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.get("agent_ready", False):
        st.error("Agent is not ready. Please check the error above.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                result = st.session_state.agent({
                    "input": prompt,
                    "chat_history": chat_history
                })
                response = result.get("output", "No response generated")
                
                st.markdown(response)
                
                tools_used = []
                if "p." in response or "page" in response.lower():
                    tools_used.append({"name": "document_search", "args": {"query": prompt}})
                if "http" in response.lower() or "source:" in response.lower():
                    tools_used.append({"name": "internet_search", "args": {"query": prompt}})
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "tools_used": tools_used if tools_used else None
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

