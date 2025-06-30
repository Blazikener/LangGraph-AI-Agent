from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
import os

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
print(f"API key exists: {bool(groq_key)}")
if not groq_key:
    print("ERROR: GROQ_API_KEY not found in .env file or environment variables")
    exit(1)

# Verify API key exists
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("ERROR: GROQ_API_KEY environment variable not set")
    print("Please create a .env file with GROQ_API_KEY=your_api_key")
    exit(1)

# Initialize chat model
os.environ["GROQ_API_KEY"] = groq_key  # Set in environment

llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.5
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("Message: ")
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(state["messages"][-1].content)