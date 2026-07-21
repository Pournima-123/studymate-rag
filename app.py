"""
app.py
------
Streamlit front-end for StudyMate: upload lecture PDFs, ask questions,
get cited answers grounded in your own notes.

Run with: streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.ingest import build_chunks_from_pdf
from src.retriever import Retriever
from src.qa import answer_question

st.set_page_config(page_title="StudyMate — RAG Study Assistant", page_icon="📚")

st.title("📚 StudyMate")
st.caption("Upload your lecture notes. Ask questions. Get answers cited back to the exact page.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

uploaded_files = st.file_uploader(
    "Upload one or more lecture PDFs", type=["pdf"], accept_multiple_files=True
)

if st.button("Process notes", disabled=not uploaded_files):
    all_chunks = []
    with st.spinner("Extracting and chunking text..."):
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = Path(tmp.name)
            all_chunks.extend(build_chunks_from_pdf(tmp_path))

    st.session_state.chunks = all_chunks
    st.session_state.retriever = Retriever(all_chunks)
    st.success(f"Processed {len(uploaded_files)} file(s) into {len(all_chunks)} searchable chunks.")

st.divider()

question = st.text_input("Ask a question about your notes")

if st.button("Get answer", disabled=not question or st.session_state.retriever is None):
    with st.spinner("Searching notes and generating answer..."):
        results = st.session_state.retriever.search(question, top_k=4)
        answer = answer_question(question, results)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Show retrieved source excerpts"):
        for r in results:
            st.markdown(f"**{r.chunk.source_file} — page {r.chunk.page_number}** (relevance: {r.score:.2f})")
            st.write(r.chunk.text[:400] + "...")
            st.markdown("---")

if st.session_state.retriever is None:
    st.info("Upload and process your lecture notes above to get started.")
