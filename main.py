from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

# --- COMPREHENSIVE FIX ---
# 1. Find all environment variables that look like API keys
possible_keys = [k for k in os.environ if "GROQ_API" in k.upper()]

# 2. Clean up any keys with spaces in name or value
for key in possible_keys:
    if key != key.strip():  # If key name has spaces
        clean_name = key.strip()
        clean_value = os.environ[key].strip()
        os.environ[clean_name] = clean_value
        del os.environ[key]
    elif os.environ[key] != os.environ[key].strip():  # If value has spaces
        os.environ[key] = os.environ[key].strip()
# --- END FIX ---

llm = init_chat_model("groq:llama-3.3-70b-versatile")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

user_input = input("Enter a message: ")
state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print(state["messages"][-1].content)

# DEBUG: Show all Groq-related environment variables
print("\nEnvironment Variables:")
for key in os.environ:
    if "GROQ" in key or "API" in key:
        value = os.environ[key]
        masked = value[:4] + '*'*(len(value)-8) + value[-4:]
        print(f"{key} = '{masked}'")

# Verify key exists
if "GROQ_API_KEY" not in os.environ:
    print("\nERROR: GROQ_API_KEY not found in environment!")
    exit(1)

# Test API key directly
import requests
try:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    response = requests.post(url, headers=headers, json={"model": "llama3-70b-8192", "messages": [{"role": "user", "content": "Hello"}]})
    print(f"\nAPI Test Status: {response.status_code}")
    if response.status_code != 200:
        print(f"API Test Failed: {response.text}")
        exit(1)
except Exception as e:
    print(f"\nAPI Test Error: {e}")
    exit(1)