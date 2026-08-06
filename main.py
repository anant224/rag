import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()   # reads .env (GEMINI_API_KEY)

from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import DocumentLoader
from src.vectorizer import Vectorizer
from src.retriever import Retriever
from src.generator import Generator
from src.budget import BudgetEstimator


DATA_FOLDER = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def build_trip_request(city, days, budget, travelers, travel_types):
    types = ", ".join(travel_types)
    return (
        f"Destination: {city} | Duration: {days} days | Budget: {budget} | "
        f"Travelers: {travelers} | Travel types: {types}"
    )


def main():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

    # 1. Embedding + Vectorizer
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorizer = Vectorizer(CHROMA_DIR=CHROMA_DIR, embedding=embedding_model)

    # 2. Build DB once; load after
    if Path(CHROMA_DIR).exists():
        print("Existing vector DB found -> loading it")
        vector_db = vectorizer.load_vector_store()
    else:
        print("No vector DB found -> building it once")
        loader = DocumentLoader(DATA_FOLDER)
        documents = loader.load_documents()
        vector_db = vectorizer.build_vector(documents)

    # 3. Generator + Budget estimator
    generator = Generator(model_name=GEMINI_MODEL, api_key=GEMINI_API_KEY)
    budget_estimator = BudgetEstimator()

    # ---------- USER INPUTS ----------
    city = "Ooty"
    days = 2
    budget = "budget"
    travelers = 6
    travel_types = ["mountains", "historical", "food", "adventure"]

    question = build_trip_request(city, days, budget, travelers, travel_types)

    
    base_retriever = vectorizer.create_retriever(vector_db, city=city, k=5)
    retriever = Retriever(base_retriever)
    rag_context = retriever.search_documents(question)

    itinerary = generator.generate(generator.build_prompt(question, rag_context))

    b = budget_estimator.estimate(city, days, travelers, budget)
    budget_text = budget_estimator.format_budget(city, days, travelers, budget, b)

  
    print("\n============== YOUR ITINERARY ==============\n")
    print(itinerary)

    print("\n\n============== BUDGET ESTIMATION ==============\n")
    print(budget_text)


if __name__ == "__main__":
    main()