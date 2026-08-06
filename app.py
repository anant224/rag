"""
app.py
------
FastAPI backend for QuanTrip AI.

On startup it:
  1. loads the data (places + blogs)
  2. builds / loads the two Chroma collections
  3. creates all the helpers (generator, budget, recommender, qa, ...)
  4. fills the LangChain tools box (init_tools)
  5. builds the LangGraph brain (ChatBrain)

The /chat endpoint runs the brain and remembers state per session_id.
The static/ folder (the chat UI) is served at "/".
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import DocumentLoader
from src.vectorizer import Vectorizer
from src.generator import Generator
from src.budget import BudgetEstimator
from src.city_index import CityIndex
from src.recommender import Recommender
from src.qa import QA
from src.tools import init_tools
from src.graph import ChatBrain


# ---------- CONFIG ----------
DATA_FOLDER = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


app = FastAPI(title="QuanTrip AI")

# simple in-memory memory:  session_id -> conversation state
SESSIONS = {}
BRAIN = None


@app.on_event("startup")
def setup():
    global BRAIN
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

    # 1. embedding + vectorizer
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorizer = Vectorizer(CHROMA_DIR, embedding)

    # 2. load data (needed for the city index either way)
    loader = DocumentLoader(DATA_FOLDER)
    place_docs, blog_docs = loader.load_documents()

    # 3. build the DB once, load it afterwards
    if vectorizer.exists():
        places_db, blogs_db = vectorizer.load_vector_store()
    else:
        places_db, blogs_db = vectorizer.build_vector(place_docs, blog_docs)

    # 4. helpers
    generator = Generator(GEMINI_MODEL, GEMINI_API_KEY)
    budget = BudgetEstimator()
    city_index = CityIndex(place_docs, blog_docs)
    recommender = Recommender(city_index, generator)
    qa = QA(vectorizer.blog_retriever(blogs_db, k=3), generator)

    # 5. give the LangChain tools everything they need
    init_tools(recommender, city_index, generator, budget,
               qa, vectorizer, places_db)

    # 6. build the LangGraph brain
    BRAIN = ChatBrain(recommender, city_index, generator)
    print("QuanTrip AI is ready ✅")


# ---------- request shape ----------
class ChatRequest(BaseModel):
    session_id: str
    message: str
    intent: str = ""      # optional: set when a starter button is clicked


@app.post("/chat")
def chat(req: ChatRequest):
    state = SESSIONS.get(req.session_id, {})
    state["user_message"] = req.message
    # a button click sets the intent (only when NOT mid-flow)
    if req.intent and not state.get("stage"):
        state["intent"] = req.intent

    state = BRAIN.handle(state)          # run the LangGraph brain
    SESSIONS[req.session_id] = state     # remember for the next message
    return {"reply": state.get("bot_response", "Sorry, I didn't get that.")}


@app.post("/reset")
def reset(req: ChatRequest):
    SESSIONS.pop(req.session_id, None)
    return {"reply": "New chat started."}


# ---------- serve the frontend (keep LAST) ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
