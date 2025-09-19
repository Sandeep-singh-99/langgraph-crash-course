import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")

# Paths
current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# HuggingFace embeddings (same as used during indexing)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",  model_kwargs={"device": "cpu"})

# Load the existing database
if not os.path.exists(persistent_directory):
    st.error("❌ Chroma DB not found. Please run the PDF indexing script first.")
else:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Input box for questions
    text_input = st.text_input("Ask a question about the indexed PDFs:")

    if text_input:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(text_input)

        st.write("---- Relevant Documents ----")
        for i, doc in enumerate(relevant_docs, 1):
            st.write(f"**Document {i}:** {doc.page_content}")
            if doc.metadata:
                st.caption(f"Metadata: {doc.metadata}")
