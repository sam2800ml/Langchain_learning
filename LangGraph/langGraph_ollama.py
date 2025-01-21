from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
import os





tool = TavilySearchResults(max_results=1)
model = ChatOllama(model = "llama3.2:1b")

model_with_tools = model.bind_tools([tool])

response = model_with_tools("Whats a BCI")

print(response)