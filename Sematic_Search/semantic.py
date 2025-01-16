from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

documents_pdf = "Sematic_Search/data/NEUROSCIENCE Exploring the Brain, Fourth Edition (Mark F. Bear, Ph.D., Barry W. Connors, Ph.D. etc.) (Z-Library).pdf"
loader = PyPDFLoader(documents_pdf)
docs = loader.load()
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200, 
    add_start_index = True
)

all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}")
print(vector_1[:10])

vector_store = Chroma(embedding_function=embeddings)

ids = vector_store.add_documents(documents=all_splits)

# this documents based on similarity to a string query
result = vector_store.similarity_search(
    "whats neuroscience"
)
print(f"String: {result[0]}")
print("-"*100)
result2 = vector_store.asimilarity_search("whats neuroscience")
print(f"Result base on async: {result2}")
print("-"*100)
result_embedding = embeddings.embed_query("whats neuroscience")
result3 = vector_store.similarity_search_by_vector(result_embedding)
print(f"result by embedding: {result3}")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1},
)
print("-"*100)
batch = retriever.batch(
    [
        "Whats a neuron",
        "whats a soma"
    ]
)
print(f"Batch: {batch}")