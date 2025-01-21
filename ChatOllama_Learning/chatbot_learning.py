from typing import Annotated # Used to add metadata to types
from typing_extensions import TypedDict # dictionaries with a fixed set of keys 

from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages # Add messages to the graph
from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama # integration with ollama
from dotenv import load_dotenv
import os 
from langchain_community.tools.tavily_search import TavilySearchResults # tool to perform search queries
from langchain_core.messages import ToolMessage

import json

load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = api_key

tool = TavilySearchResults(max_results=1)
tools = [tool] # create a list of tools

memory = MemorySaver()

class BasicToolNode():
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools} # recieve a list of tools and convert it ina dictionary of tools 
    
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found")
        outputs = []
        for tool_call in message.tool_calls:  # For each tool_call in the message, invokes it and append the result as a ToolMessage in output
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name = tool_call["name"],
                    tool_call_id = tool_call["id"]
                )
            )
        return {"messages": outputs}
        



class State(TypedDict):
    messages : Annotated[list, add_messages]



graph_builder = StateGraph(State)

llm = ChatOllama(
      model="llama3.2:1b"
).bind_tools(tools)

def chatbot(state: State):
    return {"messages":[llm.invoke(state["messages"])]}


def route_tools(
    state: State,
): 
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Define the tools node
basic_tool_node = BasicToolNode(tools)

graph_builder.add_node("tools", basic_tool_node)

# Define other nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END}  # Route to "tools" or END
)

graph_builder.add_edge("tools", "chatbot")  # Loop back to chatbot
graph_builder.add_edge(START, "chatbot")   # Start with chatbot

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user","content": user_input }]}, config=config):
        for value in event.values():
            print("assistant", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit"]:
            print("Goodbye: ")
            break
        stream_graph_updates(user_input)
        snapshot = graph.get_state(config)
        
    except:
        break
