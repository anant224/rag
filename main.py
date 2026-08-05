import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   # reads .env (GEMINI_API_KEY)

from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import DocumentLoader
from src.vectorizer import Vectorizer
from src.retriever import Retriever
from src.generator import Generator


# ---------- CONFIG ----------
DATA_FOLDER = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ---------- HELPER: pack user inputs into one clean question ----------
def build_trip_request(city:str, days, budget, travelers, travel_types):
    types = ", ".join(travel_types)
    return (
        f"Destination: {city} | Duration: {days} days | Budget: {budget} | "
        f"Travelers: {travelers} | Travel types: {types}"
    )


def main():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

    # 1. Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Vectorizer
    vectorizer = Vectorizer(CHROMA_DIR=CHROMA_DIR, embedding=embedding_model)

    # 3. Build DB once; load it on later runs
    if Path(CHROMA_DIR).exists():
        print("Existing vector DB found -> loading it")
        vector_db = vectorizer.load_vector_store()
    else:
        print("No vector DB found -> building it once")
        loader = DocumentLoader(DATA_FOLDER)
        documents = loader.load_documents()
        vector_db = vectorizer.build_vector(documents)

    # ---------- USER INPUTS ----------
    
    city="Mumbai"
    question = build_trip_request(
        city=city,
        days=5,
        budget="luxury",
        travelers=7,
        travel_types=["beaches", "bollywood", "food", "adventure", "old mumbai"],
    )

    # 4. Retriever -> filtered by city so ONLY that city's places return
    base_retriever = vectorizer.create_retriever(vector_db, city=city, k=5)
    retriever = Retriever(base_retriever)

    # 5. Generator
    generator = Generator(model_name=GEMINI_MODEL, api_key=GEMINI_API_KEY)

    # 6. Retrieve relevant places
    rag_context = retriever.search_documents(question)

    # 7. Build prompt + generate
    prompt = generator.build_prompt(
        question=question,
        memory_context="",
        rag_context=rag_context,
    )
    itinerary = generator.generate(prompt)

    # 8. Show it
    print("\n================ YOUR ITINERARY ================\n")
    print(itinerary)


if __name__ == "__main__":
    main()