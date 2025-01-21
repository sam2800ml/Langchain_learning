from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.tools.tavily_search import TavilySearchResults
import os
os.environ["TAVILY_API_KEY"] = "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"

tool = TavilySearchResults(max_results=1)
tools = [tool]
#tavily = tool.invoke("Whats a node in langgraph")
#print(tavily)

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model_with_tools = model.bind_tools(tools)

def chatbot(state: State):
    # Ensure there is a user message to process
    if not state["messages"]:
        return {"messages": [{"role": "assistant", "content": "Please enter a message to start chatting."}]}
    
    # Handle HumanMessage objects properly
    last_message = state["messages"][-1]
    if hasattr(last_message, "content"):
        user_message = last_message.content  # Access the `content` attribute
    else:
        user_message = str(last_message)  # Fallback if not a HumanMessage
    
    # If the user message is a list, join its contents
    if isinstance(user_message, list):
        prompt_text = " ".join(msg.content if hasattr(msg, "content") else str(msg) for msg in user_message)
    else:
        prompt_text = str(user_message)
    
    # Tokenize user input
    inputs = tokenizer.encode(prompt_text, return_tensors="pt")
    
    # Generate a response using the model
    outputs = model.generate(inputs, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Append the response to the message history
    state["messages"].append({"role": "assistant", "content": response})
    return {"messages": state["messages"]}




graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def stream_graph_update(user_input: str):
    for event in graph.stream({"messages":[{"role":"user", "content":user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye")
            break
            
        stream_graph_update(user_input)
    except:
        user_input = "What do you want to know about Langgraph"
        print("User: " + user_input)
        stream_graph_update(user_input)
        break
