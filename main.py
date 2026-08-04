import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   

from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import DocumentLoader
from src.vectorizer import Vectorizer
from src.retriever import Retriever
from src.generator import Generator



DATA_FOLDER = "data"                 
CHROMA_DIR = "chroma_db"             
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ---------- HELPER: turn user inputs into one clean question ----------
def build_trip_request(city, days, budget, travelers, travel_types):
    types = ", ".join(travel_types)   # travel_types is a list
    return (
        f"Destination: {city} | Duration: {days} days | Budget: {budget} | "
        f"Travelers: {travelers} | Travel types: {types}"
    )


def main():
    # Safety check for the API key
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

    # 1. Embedding model (used for both building and searching)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 2. Vectorizer setup
    vectorizer = Vectorizer(CHROMA_DIR=CHROMA_DIR, embedding=embedding_model)

    # 3. Build the DB only once; load it on later runs
    if Path(CHROMA_DIR).exists():
        print("Existing vector DB found -> loading it")
        vector_db = vectorizer.load_vector_store()
    else:
        print("No vector DB found -> building it once")
        loader = DocumentLoader(DATA_FOLDER)
        documents = loader.load_documents()
        vector_db = vectorizer.build_vector(documents)

    # 4. Retriever
    base_retriever = vectorizer.create_retriever(vector_db, k=5)
    retriever = Retriever(base_retriever)

    # 5. Generator (Gemini)
    generator = Generator(model_name=GEMINI_MODEL, api_key=GEMINI_API_KEY)

    # ---------- USER INPUTS ----------
    question = build_trip_request(
        city="Amritsar",
        days=2,
        budget="moderate",
        travelers=4,
        travel_types=["religious", "historical", "food"],
    )

    # 6. Retrieve relevant places
    rag_context = retriever.search_documents(question)

    # 7. Build prompt + generate itinerary
    prompt = generator.build_prompt(
        question=question,
        memory_context="",        # no chat history for now
        rag_context=rag_context,
    )
    itinerary = generator.generate(prompt)

    # 8. Display in the terminal (plain print)
    print("\n================ YOUR ITINERARY ================\n")
    print(itinerary)


if __name__ == "__main__":
    main()