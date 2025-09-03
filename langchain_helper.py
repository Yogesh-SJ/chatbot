# langchain_helper.py
import os
import pandas as pd
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS

load_dotenv()  

DB_PATH = "index_storage"
CSV_PATH = "faq_data.csv"   


def _embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def _llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)


def build_vector_database(csv_path: str = CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Please create it with 'prompt' and 'response' columns."
        )

    df = pd.read_csv(csv_path, encoding="cp1252")  

    required_cols = {"prompt", "response"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: 'prompt' and 'response'")

   
    docs = [
        Document(
            page_content=f"prompt: {row['prompt']}\nresponse: {row['response']}",
            metadata={"row": int(i)}
        )
        for i, row in df.iterrows()
    ]

    vectordb = FAISS.from_documents(docs, embedding=_embeddings())
    vectordb.save_local(DB_PATH)


def initialize_qa_chain():
    
    vectordb = FAISS.load_local(
        DB_PATH,
        _embeddings(),
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 5}
    )

    prompt_structure = """You are a helpful assistant that must answer strictly from the provided CONTEXT.
Use as much content as possible from each item's 'response' section verbatim when appropriate.
If the answer is not in the CONTEXT, reply exactly with: I don't know.

CONTEXT:
{context}

QUESTION:
{question}
"""
    qa_prompt = PromptTemplate(
        template=prompt_structure,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return chain
