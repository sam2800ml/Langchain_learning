from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub 
from typing_extensions import List, TypedDict
import hashlib
import os
from langchain.prompts import PromptTemplate
import torch
from tqdm import tqdm
from IPython.display import Image, display
from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

class State(TypedDict):
    question:str
    context : List[Document]
    answer: str



# This function calculates the hash of the file
def calculate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# This function checks if the PDF has already been processed
def is_pdf_processed(pdf_hash, processed_pdfs_path="processed_pdfs.txt"):
    if not os.path.exists(processed_pdfs_path):
        return False  # No previous processed files
    with open(processed_pdfs_path, "r") as file:
        processed_hashes = file.readlines()
    return pdf_hash + "\n" in processed_hashes

# This function saves the processed PDF hash to the file
def mark_pdf_as_processed(pdf_hash, processed_pdfs_path="processed_pdfs.txt"):
    with open(processed_pdfs_path, "a") as file:
        file.write(pdf_hash + "\n")


def load_model(model_name):
    print("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model




def load_Documents(path, processed_pdfs_path="processed_pdfs.txt"):
    print("Loading the documents")
    all_docs = []
    for document in os.listdir(path):
        print(document)
        pdf_path = os.path.join(path, document)
        
        # Calculate hash of the PDF
        pdf_hash = calculate_file_hash(pdf_path)
        
        # Skip the PDF if it's already processed
        if is_pdf_processed(pdf_hash, processed_pdfs_path):
            print(f"PDF {document} has already been processed. Skipping.")
            continue

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(docs[0].metadata)
        all_docs.extend(docs)
        
        # Mark this PDF as processed
        mark_pdf_as_processed(pdf_hash, processed_pdfs_path)

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

def embedding(splits, batch_size=64 ):
    print("Loading the embeddings")
    
    device = "cuda" if torch.cuda.is_available() else "mps"
    embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={"device": device})  
    vector_store = Chroma(embedding_function=embedding, persist_directory="DB",collection_name="chat",)
    for i in tqdm(range(0, len(splits), batch_size), desc="Embedding Chunks"):
        batch = splits[i:i + batch_size]
        vector_store.add_documents(documents=batch)
    
    return vector_store

def create_retrieve_fn(vector_store):
    def retrieve(state: State):
        print("Loading the retrieve")

        retriever_docs = vector_store.similarity_search(state["question"], k=5)
        return {"context": retriever_docs}
    return retrieve

def create_generate_fn(prompt, tokenizer, model):
    def generate(state: State):
        if not state.get("context") or not state.get("question"):
            raise ValueError("State must contain 'context' and 'question'.")
        
        print("Loading the generator...")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        
        # Prepare the prompt text
        if isinstance(messages, list):
            prompt_text = " ".join(msg.content for msg in messages)
        else:
            prompt_text = str(messages)
        
        # Tokenize input
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings - 256,
            padding=True
        )
        
        # Generate response
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"answer": response}
    
    return generate





data_path = "data"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_model, model_model = load_model(model_name)
custom_prompt_template = """
Answer the question below based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

# Create a new PromptTemplate instance
prompt_query = PromptTemplate(template=custom_prompt_template, input_variables=["question", "context"])

documents = load_Documents(data_path)
split = split_text(documents)
embeddings = embedding(split)
retrieve_fn = create_retrieve_fn(embeddings)
generate_fn = create_generate_fn(prompt=prompt_query, tokenizer=tokenizer_model, model=model_model)


graph_builder = StateGraph(State).add_sequence([retrieve_fn, generate_fn])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "Whats a bci"})
print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
display(Image(graph.get_graph().draw_mermaid_png()))