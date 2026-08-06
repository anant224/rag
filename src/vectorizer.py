"""
vectorizer.py
-------------
Builds / loads TWO Chroma collections inside the same DB folder:

  - "places" collection -> place records  (used by the itinerary)
  - "blogs"  collection -> city blogs     (used by Ask-a-question)

Both use the SAME HuggingFace embedding model.
The DB is built ONCE. On later runs it is just loaded (fast).
"""

from pathlib import Path
from langchain_chroma import Chroma


class Vectorizer:

    def __init__(self, chroma_dir, embedding):
        self.chroma_dir = chroma_dir
        self.embedding = embedding

    # ---------- BUILD (first run) ----------
    def build_vector(self, place_docs, blog_docs):
        places = Chroma.from_documents(
            documents=place_docs,
            persist_directory=self.chroma_dir,
            embedding=self.embedding,
            collection_name="places",
        )
        blogs = Chroma.from_documents(
            documents=blog_docs,
            persist_directory=self.chroma_dir,
            embedding=self.embedding,
            collection_name="blogs",
        )
        print("Vector store created (places + blogs)")
        return places, blogs

    # ---------- LOAD (later runs) ----------
    def load_vector_store(self):
        places = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embedding,
            collection_name="places",
        )
        blogs = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embedding,
            collection_name="blogs",
        )
        print("Vector store loaded (places + blogs)")
        return places, blogs

    # ---------- does the DB already exist? ----------
    def exists(self):
        return Path(self.chroma_dir).exists()

    # ---------- retriever for PLACES, filtered by city ----------
    def place_retriever(self, places_db, city, k=6):
        return places_db.as_retriever(
            search_kwargs={"k": k, "filter": {"city": city}}
        )

    # ---------- retriever for BLOGS (semantic search, no filter) ----------
    def blog_retriever(self, blogs_db, k=3):
        return blogs_db.as_retriever(search_kwargs={"k": k})
