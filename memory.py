import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

class EmbeddingMemory:
    def __init__(self, name='memory1', top_k=3):
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory='.chromadb',
        ))
        # client.reset()
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("openai_api_key"),
            model_name="text-embedding-ada-002"
        )
        collection = client.get_or_create_collection(
            name=name, embedding_function=openai_ef)
        self.collection = collection
        self.top_k = top_k
        self.index = collection.count()

    def add(self, documents):
        ids = [str(self.index+idx) for idx in range(len(documents))]
        self.index += len(ids)
        self.collection.add(documents=documents, ids=ids)

    def query(self, query_text):
        documents = []
        count = self.collection.count()
        if count < self.top_k:
            results = self.collection.peek()
            documents = results['documents']
        else:
            results = self.collection.query(query_texts=[query_text], n_results=self.top_k)
            documents = results['documents'][0]
        return documents
