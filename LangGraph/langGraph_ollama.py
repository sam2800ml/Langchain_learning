from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
import os


os.environ["TAVILY_API_KEY"] = "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"


tool = TavilySearchResults(max_results=1)
model = ChatOllama(model = "llama3.2:1b")

model_with_tools = model.bind_tools([tool])

response = model_with_tools("Whats a BCI")

print(response)