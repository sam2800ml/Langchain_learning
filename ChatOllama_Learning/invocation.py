from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage


llm = ChatOllama(
    model="llama3.2:1b"
)

messages = [
    (
        "system",
        "You are a helpful assistant that translate to french",    
    ),
    ("human","I love programming")
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)

