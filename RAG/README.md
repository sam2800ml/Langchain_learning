# Retrieval Augmented Generation (RAG)
Has two main components:
- Indexing -> A pipeline for ingesting data from a source and indexing it
- Retrieval and Generation -> Takes the user query at run time and retrieves the relevant data from the index, then passes that to the model

## Indexing
- Load: First we load the data using the Document loaders
- Split: Then we split the documents that we load into smaller chunks 
- Store: We convert the chunks in to embeddings vector that is being store and its easier to query for similarity

## Retrieval and Generation

- Retrieve: given the user input, we first convert that to an embedding that the is used to retrieve relevant splits base on the similarity of the embeddings
- Generate: The model then gives an answer base on the query of the user and the information retrieve in the previous step
