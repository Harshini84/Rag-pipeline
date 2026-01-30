import os
import tempfile
import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.config import DOCS_PATH


st.set_page_config(page_title="RAG Chat", layout="wide")
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("RAG Pipeline Chat")

with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload files (pdf/docx/txt/csv/xls/xlsx)", type=['pdf', 'docx', 'txt', 'csv', 'xls', 'xlsx'], accept_multiple_files=True)
    if st.button("Ingest Documents"):
        with st.spinner("Loading..."):
            try:
                # save uploaded files to a temporary directory if any
                tmp_dir = None
                if uploaded_files:
                    st.info(f"Uploading {len(uploaded_files)} files: {', '.join([f.name for f in uploaded_files])}")
                    tmp_dir = tempfile.mkdtemp(prefix="st_upload_")
                    for up in uploaded_files:
                        save_path = os.path.join(tmp_dir, up.name)
                        with open(save_path, "wb") as f:
                            f.write(up.getbuffer())
                    st.info(f"Saved to: {tmp_dir}")
                else:
                    st.warning("No files uploaded - using default directory")

                st.session_state.pipeline = RAGPipeline()
                if tmp_dir:
                    result = st.session_state.pipeline.ingest_documents(directory=tmp_dir)
                else:
                    result = st.session_state.pipeline.ingest_documents()

                num_chunks = len(st.session_state.pipeline.vector_store.texts)
                sources = result.get('sources', [])
                st.success(f" Loaded {num_chunks} chunks from {len(sources)} documents")
                if sources:
                    st.write("**Documents:**")
                    for s in sources:
                        st.write(f"- {os.path.basename(s)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Add chunk visibility toggle
    st.divider()
    show_chunks = st.checkbox("Show retrieved chunks", value=False, help="Display the actual text chunks used to generate answers")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    if st.session_state.pipeline is None:
        st.error("Please ingest documents first")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                result = st.session_state.pipeline.query(prompt, show_chunks=show_chunks)
                st.markdown(result["answer"])
                st.caption(f" {result['source_used']} chunks")

                # Show retrieved chunks if enabled
                if result.get('chunks'):
                    with st.expander(" View Retrieved Chunks"):
                        for chunk in result['chunks']:
                            st.markdown(f"**Chunk {chunk['index']}** (from {chunk['source']})")
                            st.text(chunk['text'])
                            st.divider()

                st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
