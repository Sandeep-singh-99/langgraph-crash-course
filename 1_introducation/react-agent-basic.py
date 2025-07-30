from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0);

search_tool = TavilySearchResults(search_depth="basic")

tools = [search_tool]

agents = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agents.invoke("how to find and apply remote internships in Full Stack Development?")
