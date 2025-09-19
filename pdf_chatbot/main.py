import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "store", "Profile.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    st.write("Persistent directory does not exist. Please create it first.")

    if not os.path.exists(file_path):
        st.write(f"File not found at {file_path}. Please ensure the PDF file is in the correct location.")

    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents=documents)

    st.write("----Document Chunks Information----")
    st.write(f"Total number of chunks: {len(docs)}")
    st.write(f"First chunk {docs[0].page_content}...")

    st.write("----Creating Vector Store----")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.write("Embeddings created Successfully")

    st.write("Creating Chroma Vector Store...")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persistent_directory)
    st.write("Chroma Vector Store created successfully.")
else:
    st.write("Persistent directory already exists. Skipping document loading and vector store creation.")
