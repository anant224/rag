from langchain_chroma import Chroma
from pathlib import Path


class Vectorizer:

    def __init__(self, CHROMA_DIR, embedding):
        self.CHROMA_DIR = CHROMA_DIR
        self.embedding = embedding

    # Use this the FIRST time -> creates the DB from your place documents
    def build_vector(self, documents):
        vector_db = Chroma.from_documents(
            documents=documents,
            persist_directory=self.CHROMA_DIR,
            embedding=self.embedding,
        )
        print("Vector store created")
        return vector_db

    # Use this LATER -> loads the DB that already exists on disk
    def load_vector_store(self):
        vector_db = Chroma(
            persist_directory=self.CHROMA_DIR,
            embedding_function=self.embedding,
        )
        print("Vector store loaded")
        return vector_db

    # Turns the DB into a retriever (top-k similarity search)
    def create_retriever(self, vector_store, k=3):
        return vector_store.as_retriever(
            search_kwargs={"k": k}
        )