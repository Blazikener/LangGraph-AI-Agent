from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Clean up API key formatting
possible_keys = [k for k in os.environ if "GROQ_API" in k.upper()]
for key in possible_keys:
    if key != key.strip():
        clean_name = key.strip()
        clean_value = os.environ[key].strip()
        os.environ[clean_name] = clean_value
        del os.environ[key]
    elif os.environ[key] != os.environ[key].strip():
        os.environ[key] = os.environ[key].strip()

# Initialize language model
llm = init_chat_model("groq:llama-3.3-70b-versatile")

# Define conversation state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the conversation graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": llm.invoke(state["messages"])})
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# Run the chatbot
user_input = input("Enter a message: ")
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
print(state["messages"][-1].content)
