from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables (e.g., API keys)

# Initialize the LLM model
language_model = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize embeddings
model_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
db_path = "index_storage"

def build_vector_database():
    # Load data from a CSV file
    data_loader = CSVLoader(file_path='faq_data.csv', source_column="prompt")
    document_data = data_loader.load()

    # Create a vector database using FAISS
    vector_database = FAISS.from_documents(documents=document_data, embedding=model_embeddings)

    # Save the vector database locally
    vector_database.save_local(db_path)

def initialize_qa_chain():
    # Load the vector database
    loaded_vector_database = FAISS.load_local(db_path, model_embeddings)

    # Create a retriever for querying the database
    context_retriever = loaded_vector_database.as_retriever(score_threshold=0.7)

    prompt_structure = """Given the following context and a question, generate an answer based on this context only.
    Try to use as much content as possible from the "response" section in the context without significant changes.
    If the answer is not found in the context, state "I don't know." Avoid making up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    qa_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])

    # Create a Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=language_model,
        chain_type="stuff",
        retriever=context_retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa_chain

if __name__ == "__main__":
    build_vector_database()
    qa_chain = initialize_qa_chain()
    print(qa_chain("Do you offer a JavaScript course?"))
