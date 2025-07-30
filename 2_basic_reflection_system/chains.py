from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "System",
            "You are a twitter bot that posts tweets about LangGraph, a framework for building LLM applications."
            "Generate a best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous post."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)