from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.8
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates to {input_language} to {Output_Language}",
        ),
        ("human","{input}")
    ]
)

chain = prompt | llm 

print(chain)
response = chain.invoke(
    {
        "input_language":"English",
        "Output_Language":"french",
        "input":"Yo amo la programacion",
    }
)

print(response.content)