from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_community.document_loaders import PyPDFLoader
import os
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
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":10},
    )
    print("-"*100)
    batch = retriever.batch(
        [
            "Whats a neuron",
            "whats a soma"
        ]
    )
    print(f"Batch: {batch}")
path = "data"
documents = load_Documents(path)
split = split_text(documents)
embeddings = embedding(split)