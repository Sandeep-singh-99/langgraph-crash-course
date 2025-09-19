import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "store", "Profile.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    if not os.path.exists(persistent_directory):
        st.write("Creating vector store from PDF...")
        loader = PyMuPDFLoader(file_path=file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_directory)
        st.write("Vector store created successfully.")
    else:
        st.write("Loading existing vector store...")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Query
    text_input = st.text_input("Ask a question about the PDF:")
    if text_input:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(text_input)
        st.write("---- Relevant Documents ----")
        for i, doc in enumerate(relevant_docs, 1):
            st.write(f"Document {i}: {doc.page_content}")

except Exception as e:
    import traceback
    st.error(f"Error: {str(e)}")
    traceback.print_exc()
