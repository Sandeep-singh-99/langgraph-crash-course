from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph
graph_builder = StateGraph(State)

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Initialize the search tool
search_tool = TavilySearchResults(max_results=2)  # Updated class

# List of tools
tools = [search_tool]

# Define a ReAct-compatible prompt
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the provided tools to answer queries accurately.

Tools available: {tool_names}

{tools}

Human: {input}
Assistant: {agent_scratchpad}"""
)

# Create the ReAct agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Define the chatbot node
def chatbot(state: State):
    # Extract the latest user message
    user_message = state["messages"][-1].content
    # Invoke the agent with the user message
    response = agent_executor.invoke({"input": user_message})
    # Return the agent's response as a message
    return {"messages": [{"role": "assistant", "content": response["output"]}]}

# Add the chatbot node and edges
graph_builder.add_node("Chatbot", chatbot)
graph_builder.add_edge(START, "Chatbot")
graph_builder.add_edge("Chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# Function to run the chatbot interactively
def run_chatbot(input_message: str):
    initial_state = {"messages": [{"role": "user", "content": input_message}]}
    result = graph.invoke(initial_state)
    return result["messages"][-1]["content"]

# Example usage
if __name__ == "__main__":
    user_input = "What is the weather today?"
    print(f"User: {user_input}")
    response = run_chatbot(user_input)
    print(f"Chatbot: {response}")