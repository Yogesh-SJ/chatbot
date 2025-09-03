# app.py
import streamlit as st
from dotenv import load_dotenv
from langchain_helper import initialize_qa_chain, build_vector_database

load_dotenv()  # loads GOOGLE_API_KEY from .env

st.set_page_config(layout="wide")
st.title("Chatbot")

col_left, col_right = st.columns(2)

with col_left:
    if st.button("Create Knowledgebase"):
        try:
            build_vector_database()  # builds from faq_data.csv
            st.success("Vector DB built and saved.")
        except Exception as e:
            st.error(f"Failed to build DB: {e}")

with col_right:
    user_question = st.text_input("Ask a question:")

    if user_question:
        qa_chain = initialize_qa_chain()
        result = qa_chain.invoke({"query": user_question})  # <- important: pass dict with "query"

        st.subheader("Answer:")
        st.write(result["result"])

        with st.expander("Sources (top matches)"):
            for i, doc in enumerate(result.get("source_documents", []), start=1):
                st.markdown(f"**Match {i}**")
                st.text(doc.page_content)
