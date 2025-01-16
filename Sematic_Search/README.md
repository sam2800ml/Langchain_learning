# Semantic Search Engine 

In this we are going to be following the tutorial in langchain in how to create a semantic search engine for Pdf documents, so we can learn how to retrieve the information from the pdfs to pass it to the model 

the most important concepts in here is:
- Documents and documental loaders
- Text splitters
- Embeddings
- Vector stores and retrieves

## Documents and Document Loaders
The document and document loaders are an essencial part of the rag implementation, this isi because this can process all the documents that the model its going to be recieving, the document loader functions with pypdfloader in which we can load a whole file with multiple pdfs, and we can obtained different information of them as the metadata and id.
we can generate sample documents using Document, and this can function as an example of the relations between some documents and others


## Text splitters
The splitters are also a very important step for our RAG implementation, when we are about to use our model, we dont pase the whole book to the model, this can take time to process everything, thats why we use the text splitters, we can split the text into sections that we can define, and these sections will be called chunks, we can define also an overlap the way that it works is a overlap in case that a text is being cut off in a essential part

## Embeddings
Vector search is a common way to store and search over unstructured data, the idea is to store a numeric vector that are associated with the text. when we recieve the query from our user, we are going to the same, and then perform vector similarity to identify related text

## Vector stores

Vector store objects contains methods for adding text and Document objects, and then querying them using similarity metrics, for this vector store we can use different methods we can use the one integrate in langchain or we can use external by third parties like postgress, pinecone, fais, chroma, all of this are already integrate it, and will be easier.
>
To make out vector store work better we have to pass it the embedding model, because it needs to know the model to be able to store all the information, also we can pass after out chunks so it can be store, and then we can start querying the documents, we can use Synchronously and asynchronously, we can also use the similarity scores, and by similarity maximum marginal relevance
