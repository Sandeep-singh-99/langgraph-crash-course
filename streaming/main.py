import time
import streamlit as st
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

# ----- Define State -----
class State(TypedDict):
    messages: Annotated[list, add_messages]
    requires_human_review: bool

# ----- Initialize LLM -----
# Use flash for dev (higher limits), pro for production
MODEL_NAME = "gemini-1.5-flash"  # change to gemini-2.5-pro if needed
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0)

# ----- Search Tool -----
search_tool = TavilySearchResults(search_depth="basic")
tools = [search_tool]

# ----- Agent -----
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

# ----- Nodes -----
def chatbot(state: State) -> State:
    if state.get("requires_human_review", False):
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            context = f"Previous feedback: {state['messages'][-2].content}\nHuman feedback: {last_message.content}"
            response = agent.invoke(context)
        else:
            response = agent.invoke(state["messages"])
    else:
        response = agent.invoke(state["messages"])

    ai_message = AIMessage(content=response["output"], role="assistant")
    requires_review = len(response["output"]) > 500

    return {
        "messages": state["messages"] + [ai_message],
        "requires_human_review": requires_review
    }

def human_review(state: State) -> State:
    return state

# ----- Build Graph -----
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human_review", human_review)

def route_to_review(state: State):
    return "human_review" if state["requires_human_review"] else END

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    route_to_review,
    {"human_review": "human_review", END: END}
)
graph_builder.add_edge("human_review", "chatbot")

# ----- Memory -----
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# ----- Streamlit App -----
st.set_page_config(page_title="LangGraph Streaming Chatbot", page_icon="🤖")
st.title("💬 LangGraph + Gemini Chatbot with Streaming")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_thread_1"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Streaming AI output with retry
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        retry_attempts = 3
        delay = 5  # start wait in seconds

        for attempt in range(retry_attempts):
            try:
                for chunk in graph.stream(
                    {
                        "messages": st.session_state.messages,
                        "requires_human_review": False
                    },
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                ):
                    if "messages" in chunk:
                        latest_ai_message = chunk["messages"][-1]["content"]
                        if isinstance(latest_ai_message, str):
                            full_response = latest_ai_message
                            message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                break  # success → exit retry loop

            except ResourceExhausted:
                if attempt < retry_attempts - 1:
                    st.warning(f"⚠️ Rate limit hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                else:
                    st.error("❌ Rate limit exceeded. Please wait a minute and try again.")
