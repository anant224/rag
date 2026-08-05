from langchain_chroma import Chroma
from pathlib import Path


class Vectorizer:

    def __init__(self, CHROMA_DIR, embedding):
        self.CHROMA_DIR = CHROMA_DIR
        self.embedding = embedding

    # FIRST time -> creates DB from place documents
    def build_vector(self, documents):
        vector_db = Chroma.from_documents(
            documents=documents,
            persist_directory=self.CHROMA_DIR,
            embedding=self.embedding,
        )
        print("Vector store created")
        return vector_db

    # LATER runs -> loads the saved DB
    def load_vector_store(self):
        vector_db = Chroma(
            persist_directory=self.CHROMA_DIR,
            embedding_function=self.embedding,
        )
        print("Vector store loaded")
        return vector_db

    # retriever WITH city filter -> only that city's places come back
    def create_retriever(self, vector_store, city=None, k=5):
        search_kwargs = {"k": k}
        if city:
            search_kwargs["filter"] = {"city": city}
        return vector_store.as_retriever(search_kwargs=search_kwargs)