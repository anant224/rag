"""
tools.py
--------
🧰 THE LANGCHAIN TOOLS LAYER.

Here we register our 4 skills as real LangChain @tool objects:

    1. recommend_places   -> suggest cities
    2. plan_itinerary     -> build a day-by-day plan
    3. estimate_budget    -> estimate the trip cost
    4. answer_question    -> answer a general travel question

WHY THIS FILE EXISTS (for the SME):
  A LangChain "tool" is a skill wrapped with the @tool decorator so it can be
  called in a standard way. Our LangGraph nodes call these tools. This is NOT
  an autonomous agent -> the routing is controlled by LangGraph on purpose
  (predictable + safe, no infinite loops).

HOW IT WORKS (beginner note):
  Tools are simple functions, but our skills need the loaded objects
  (recommender, generator, budget, etc.). So we keep those objects in a small
  box called SERVICES, filled once at startup by init_tools(...). The tools
  then read from that box.
"""

from langchain_core.tools import tool

# a simple "box" that holds the ready-to-use services (filled at startup)
SERVICES = {}


def init_tools(recommender, city_index, generator, budget, qa,
               vectorizer, places_db):
    """Called once from app.py to give the tools everything they need."""
    SERVICES["recommender"] = recommender
    SERVICES["city_index"] = city_index
    SERVICES["generator"] = generator
    SERVICES["budget"] = budget
    SERVICES["qa"] = qa
    SERVICES["vectorizer"] = vectorizer
    SERVICES["places_db"] = places_db
    print("LangChain tools ready 🧰")


# ============================================================
#  TOOL 1 — recommend cities
# ============================================================
@tool
def recommend_places(categories: list, region: str = "") -> list:
    """Recommend the top 3 cities that match the given travel categories
    (like mountain, beaches, religious) and an optional region of India."""
    recommender = SERVICES["recommender"]
    return recommender.recommend(categories, region)


# ============================================================
#  TOOL 2 — plan itinerary  (RAG: Chroma places + Gemini)
# ============================================================
@tool
def plan_itinerary(city: str, days: int, travelers: int,
                   budget: str, categories: list) -> str:
    """Build a day-by-day travel itinerary for a city, using the number of
    days, travelers, budget level, and the chosen travel categories."""
    vectorizer = SERVICES["vectorizer"]
    places_db = SERVICES["places_db"]
    generator = SERVICES["generator"]
    city_index = SERVICES["city_index"]

    cats = categories or ["sightseeing"]
    question = (
        f"Destination: {city} | Duration: {days} days | Budget: {budget} | "
        f"Travelers: {travelers} | Travel types: {', '.join(cats)}"
    )

    # search the "places" collection, filtered to this city
    base_retriever = vectorizer.place_retriever(places_db, city, k=6)
    from src.retriever import Retriever
    rag_context = Retriever(base_retriever).search_documents(question)

    city_intro = city_index.blog(city)
    prompt = generator.build_prompt(question, rag_context, city_intro)
    return generator.generate(prompt)


# ============================================================
#  TOOL 3 — estimate budget  (plain math)
# ============================================================
@tool
def estimate_budget(city: str, days: int, travelers: int, budget: str) -> str:
    """Estimate the total trip budget for a city given days, travelers,
    and budget level (economy / moderate / luxury)."""
    budget_est = SERVICES["budget"]
    b = budget_est.estimate(city, days, travelers, budget)
    return budget_est.format_budget(city, days, travelers, budget, b)


# ============================================================
#  TOOL 4 — answer a travel question  (RAG: Chroma blogs + Gemini)
# ============================================================
@tool
def answer_question(question: str) -> str:
    """Answer a general travel question using the city blog information."""
    qa = SERVICES["qa"]
    return qa.answer(question)
