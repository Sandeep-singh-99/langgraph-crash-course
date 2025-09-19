import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

if not os.path.exists(persistent_directory):
    st.write("Persistent directory does not exist. Please create it first.")

text_input = st.text_input("Ask a question about the PDF document:")

retriever = db.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

relevant_docs = retriever.invoke(text_input)

st.write("---- Relevant Document ----")
for i, doc in enumerate(relevant_docs, 1):
    st.write(f"Document {i}: {doc.page_content}")
    if doc.metadata:
        st.write(f"Metadata: {doc.metadata}")