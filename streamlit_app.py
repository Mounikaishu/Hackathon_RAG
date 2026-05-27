import streamlit as st
import requests

# Backend API URL (Render)
API_URL = "https://hackathon-rag.onrender.com/query"

st.set_page_config(
    page_title="Placement Intelligence RAG",
    layout="wide"
)

st.title("🚀 Placement Intelligence RAG")

query = st.text_input("Ask a placement question")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                API_URL,
                json={"query": query},
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("success"):
                    st.success("Response generated")
                    st.write(data.get("response"))
                else:
                    st.error("Backend returned an error")
                    st.write(data)

            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error(f"Connection failed: {str(e)}")