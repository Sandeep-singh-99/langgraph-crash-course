from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("🤗 HuggingFace LLM Test")

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",  # small free model
    task="text-generation",
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.03,
    provider="hf-inference"  # official Hugging Face Inference API
)

prompt = st.text_input("Ask something:")

if prompt:
    with st.spinner("Generating response..."):
        response = llm.predict(prompt)
        st.write("LLM Response:", response)
