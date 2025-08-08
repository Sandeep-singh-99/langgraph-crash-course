from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver


# Load environment variables
load_dotenv()

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

search_tool = TavilySearchResults(search_depth="basic")

tools = [search_tool]

agents = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)

# Define the chatbot node
def chatbot(state: State) -> State:
    response = agents.invoke(state["messages"])
    ai_message = AIMessage(content=response["output"], role="assistant")
    # Return the updated state with the new message
    return {"messages": state["messages"] + [ai_message]}


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)

thread_id = 'user_thread_1'

# Run the graph with an initial message
result = graph.invoke(
    {"messages": [{"role": "user", "content": "How to find and apply remote internships in Full Stack Development?"}]},
    config={"configurable": {"thread_id": thread_id}}
)

# Print response
print(result["messages"][-1].content)


# Run another query in the same thread to demonstrate memory

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Can you give more details about the skills needed?"}]},
    config={"configurable": {"thread_id": thread_id}}
)


# Print the new response
print("\n Follow up response.")
print(result["messages"][-1].content)




