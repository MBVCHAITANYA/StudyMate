import streamlit as st
from pdf_parser import extract_text_from_pdfs
from chunker import chunk_text
from embedder import embed_chunks
from retriever import build_faiss_index, retrieve_top_k
from llm import generate_answer
import numpy as np

st.set_page_config(page_title="📚 StudyMate", layout="wide")

st.markdown("<h1 style='text-align: center;'>StudyMate - Your AI Study Partner</h1>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    raw_texts = extract_text_from_pdfs(uploaded_files)
    all_chunks = []
    for text in raw_texts:
        all_chunks.extend(chunk_text(text))

    embeddings = embed_chunks(all_chunks)
    faiss_index = build_faiss_index(np.array(embeddings))
    st.success("PDFs uploaded and processed! Ask away!")

    query = st.text_input("Ask a question based on your documents:")
    if st.button("Get Answer") and query:
        query_embedding = embed_chunks([query])
        context = retrieve_top_k(faiss_index, np.array(query_embedding), all_chunks)
        answer = generate_answer(query, context)
        st.markdown("### 📥 Answer")
        st.markdown(answer)
        with st.expander("Referenced Context"):
            for para in context:
                st.markdown(f"> {para}")
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []
        st.session_state.qa_history.append((query, answer))

if st.button("Download Q&A History"):
    history = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.qa_history])
    st.download_button("Click to download", data=history, file_name="qa_log.txt")
