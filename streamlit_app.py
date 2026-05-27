import streamlit as st

st.set_page_config(
    page_title="Hackathon RAG",
    layout="wide"
)

st.title("🚀 Placement Intelligence RAG")

query = st.text_input(
    "Ask a question"
)

if query:
    st.success(
        f"You asked: {query}"
    )