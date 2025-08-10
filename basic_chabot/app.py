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

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    requires_human_review: bool

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# Initialize search tool
search_tool = TavilySearchResults(search_depth="basic")
tools = [search_tool]

# Initialize agent
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)

# Define the chatbot node
def chatbot(state: State) -> State:
    # check if human review was required and include feedback if present
    if state.get("requires_human_review", False):
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            # Human provided feedback, include it in the context
            context = f"Previous feedback: {state['messages'][-2].content}\nHuman feedback: {last_message.content}"
            response = agent.invoke(context)
        else:
            response = agent.invoke(state["messages"])
    else:
        response = agent.invoke(state["messages"])
    
    ai_message = AIMessage(content=response["output"], role="assistant")
    # Decide if human review is needed
    requires_review = len(response["output"]) > 500 # Example: Flag long responses for review

    return {
        "messages": state["messages"] + [ai_message],
        "requires_human_review": requires_review
    }

def human_review(state: State) -> State:
    # This node pauses for human input; in a real application, this could notify a human
    print("Human review required for the following response: ")
    print(state["messages"][-1].content)
    # Simulate human feedback (in practice, this would come from an external input)
    human_feedback = input("Please provide feedback or type 'approve' to continue: " )
    if human_feedback.lower() == "approve":
        return {
            "messages": state["messages"],
            "requires_human_review": False
        }
    else:
        return {
            "messages": state["messages"] + [HumanMessage(content=human_feedback, role="user")],
            "requires_human_review": True
        }

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human_review", human_review)

# Define conditional edges
def route_to_review(state: State):
    return "human_review" if state["requires_human_review"] else END

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_to_review, {"human_review": "human_review", END: END})
graph_builder.add_edge("human_review", "chatbot")

# Initialize memory
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Define thread ID for conversation persistence
thread_id = "user_thread_1"

# Run the graph with an initial message
result = graph.invoke(
    {
        "messages": [{"role": "user", "content": "How to find and apply remote internships in Full Stack Development?"}],
        "requires_human_review": False
    },
    config={"configurable": {"thread_id": thread_id}}
)

# print initial response
print("Initial response:")
print(result["messages"][-1].content)

# If human review is required, handle it
if result["requires_human_review"]:
    result = graph.invoke(
        result,  # Pass the current state
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\nResponse after human review:")
    print(result["messages"][-1].content)


# Run a follow-up query in the same thread to demonstrate memory
result = graph.invoke(
    {
        "messages": [{"role": "user", "content": "Can you give more details about the skills needed?"}],
        "requires_human_review": False
    },
    config={"configurable": {"thread_id": thread_id}}
)

# Print follow-up response
print("\nFollow-up response:")
print(result["messages"][-1].content)

# If human review is required for follow-up
if result["requires_human_review"]:
    result = graph.invoke(
        result,
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\nFollow-up response after human review:")
    print(result["messages"][-1].content)