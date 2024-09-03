import streamlit as st
from langchain_helper import initialize_qa_chain, build_vector_database

# Set page layout to wide
st.set_page_config(layout="wide")

# Title at the top
st.title("Q&A Application 🌱")

# Create two columns
col_left, col_right = st.columns(2)

# In the first column, add the button to create the knowledge base
with col_left:
    initialize_btn = st.button("Create Knowledgebase")
    if initialize_btn:
        build_vector_database()

# In the second column, add the text input and display the answer
with col_right:
    user_question = st.text_input("Ask a question:")

    if user_question:
        qa_chain = initialize_qa_chain()
        response = qa_chain(user_question)

        st.subheader("Answer:")
        st.write(response["result"])
