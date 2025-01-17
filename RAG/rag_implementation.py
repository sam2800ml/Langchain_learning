from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub 
from typing_extensions import List, TypedDict
import os


class State(TypedDict):
    question:str
    context : List[Document]
    answer: str

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model




def load_Documents(path):
    all_docs = []
    for document in os.listdir(path):
        print(document)
        pdf_path = os.path.join(path,document)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(docs[0].metadata)
        all_docs.extend(docs)
    return all_docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index = True
    )

    all_splits = text_splitter.split_documents(docs)

    print(len(all_splits))
    return all_splits

def embedding(splits):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = Chroma(embedding_function=embedding)
    ids = vector_store.add_documents(documents=splits)
    
    return vector_store

def create_retrieve_fn(vector_store):
    def retrieve(state: State):
        retriever_docs = vector_store.similarity_search(state["question"], k=10)
        return {"context": retriever_docs}
    return retrieve
def create_generate_fn(prompt,tokenizer, model):
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question":state["question"], "context":docs_content})
        inputs = tokenizer(messages, return_tensors = "pt", truncation=True, padding=True)
        output = model.generate(**inputs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"answer": response}
    return generate




data_path = "data"
model_name = "ibm-granite/granite-3.1-2b-instruct"
tokenizer_model, model_model = load_model(model_name)
prompt_query = hub.pull("rlm/rag-prompt")
documents = load_Documents(data_path)
split = split_text(documents)
embeddings = embedding(split)
retrieve_fn = create_retrieve_fn(embeddings)
generate_fn = create_generate_fn(prompt=prompt_query, tokenizer=tokenizer_model, model=model_model)
graph_builder = StateGraph(State).add_sequence([retrieve_fn, generate_fn])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
result = graph.invoke({"question": "What is a neuron?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')