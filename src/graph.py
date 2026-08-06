"""
graph.py
--------
🟩 THE LANGGRAPH LAYER (the routing brain).

One user message = one graph run. The "state" (what we know + what we are
waiting for) is passed in, updated, and returned. app.py stores it per
session_id -> that is how the bot remembers the conversation.

FLOW:
    router --> recommend
           |-> itinerary
           |-> budget
           |-> qa

HUMAN-IN-THE-LOOP:
  If info is missing (city / days / travelers / budget / trip type) a node
  ASKS for it and sets a "stage". loop_count caps the asks at 2, after which
  missing info is auto-filled so the chat can never loop forever.
"""

import re
from typing import TypedDict
from langgraph.graph import StateGraph, END

# our LangChain tools
from src.tools import (
    recommend_places,
    plan_itinerary,
    estimate_budget,
    answer_question,
)


# ---------- the shared state that flows through the graph ----------
class ChatState(TypedDict, total=False):
    user_message: str
    intent: str            # recommend / itinerary / budget / qa
    stage: str             # what we are waiting for
    categories: list
    region: str
    recommended_cities: list
    city: str
    days: int
    travelers: int
    budget_level: str
    loop_count: int
    bot_response: str      # the reply we send back


# ---------- small text helpers ----------
def parse_trip_details(text):
    """Pull days, travelers, budget level out of a free sentence."""
    t = text.lower()

    days = None
    m = re.search(r"(\d+)\s*day", t)
    if m:
        days = int(m.group(1))

    travelers = None
    m = re.search(r"(\d+)\s*(people|person|traveler|travellers|travelers|pax|adults?)", t)
    if m:
        travelers = int(m.group(1))

    budget = None
    for b in ["economy", "moderate", "luxury"]:
        if b in t:
            budget = b
            break

    # fallback: two lone numbers -> first = days, second = travelers
    if days is None or travelers is None:
        nums = re.findall(r"\d+", t)
        if len(nums) >= 2:
            days = days or int(nums[0])
            travelers = travelers or int(nums[1])
        elif len(nums) == 1 and days is None:
            days = int(nums[0])

    return days, travelers, budget


def is_yes(text):
    return any(w in text.lower() for w in ["yes", "yeah", "yep", "sure", "ok", "please", "haan"])


# ===========================================================
#  ChatBrain: builds the LangGraph and holds the services
# ===========================================================
class ChatBrain:

    def __init__(self, recommender, city_index, generator):
        # we only need these three directly inside the nodes;
        # the heavy work (itinerary/budget/qa) is done by the tools.
        self.recommender = recommender
        self.city_index = city_index
        self.generator = generator
        self.graph = self._build_graph()

    # ---------------- build the graph ----------------
    def _build_graph(self):
        g = StateGraph(ChatState)

        g.add_node("router", self.router_node)
        g.add_node("recommend", self.recommend_node)
        g.add_node("itinerary", self.itinerary_node)
        g.add_node("budget", self.budget_node)
        g.add_node("qa", self.qa_node)

        g.set_entry_point("router")

        g.add_conditional_edges(
            "router",
            self.route_decision,
            {"recommend": "recommend", "itinerary": "itinerary",
             "budget": "budget", "qa": "qa"},
        )

        for node in ["recommend", "itinerary", "budget", "qa"]:
            g.add_edge(node, END)

        return g.compile()

    # ---------------- ROUTER ----------------
    def router_node(self, state: ChatState):
        # if we are mid-flow, keep the current intent
        if state.get("stage"):
            return state
        # if a button already set the intent, trust it
        if state.get("intent") in ["recommend", "itinerary", "budget", "qa"]:
            return state
        # otherwise ask the LLM to classify
        state["intent"] = self._classify(state.get("user_message", ""))
        return state

    def route_decision(self, state: ChatState):
        stage = state.get("stage", "")
        if stage.startswith("rec_"):
            return "recommend"
        if stage.startswith("itin_"):
            return "itinerary"
        if stage.startswith("bud_"):
            return "budget"
        return state.get("intent", "qa")

    def _classify(self, msg):
        prompt = f"""Classify the user's travel message into ONE word.

            - recommend : ONLY when they ask which CITY or DESTINATION to travel to
              (e.g. "suggest a place for holidays", "where should I go").
              Do NOT use this for food spots, things to do, or places WITHIN a city.
            - itinerary : they want a trip plan / day-by-day schedule for a known city.
            - budget    : they want a cost / budget estimate.
            - qa        : ANY other travel question, including food, packing, weather,
              best time to visit, attractions, tips, or things to do in a city.

            Message: "{msg}"
            Answer with only one word (recommend/itinerary/budget/qa):"""
        out = self.generator.generate(prompt).strip().lower()
        for key in ["recommend", "itinerary", "budget", "qa"]:
            if key in out:
                return key
        return "qa"

    # ---------------- RECOMMEND ----------------
    def recommend_node(self, state: ChatState):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")
        loop = state.get("loop_count", 0)

        # user is picking one of the 3 suggested cities
        if stage == "rec_await_city_choice":
            city = self._match_city(msg, state.get("recommended_cities", []))
            if city:
                state["city"] = city
                state["stage"] = "itin_await_details"
                state["intent"] = "itinerary"
                state["loop_count"] = 0
                state["bot_response"] = (
                    f"Great choice — {city}! ✨\n\n"
                    "To build your itinerary, tell me:\n"
                    "• How many days?\n• How many travelers?\n"
                    "• Budget level (economy / moderate / luxury)?"
                )
            else:
                state["bot_response"] = "Please pick one of the suggested cities by typing its name. 🙂"
            return state

        # fresh recommend OR an answer to "what kind of trip?"
        cats, region = self.recommender.extract_preferences(msg)
        if region:
            state["region"] = region
        if cats:
            state["categories"] = cats

        # not enough info yet -> ask once (respect the 2-loop cap)
        if not state.get("categories") and loop < 2:
            state["stage"] = "rec_await_categories"
            state["loop_count"] = loop + 1
            state["intent"] = "recommend"
            state["bot_response"] = (
                "I'd love to help! What kind of trip do you enjoy? 🌍\n"
                "(mountains, beaches, religious, heritage, adventure, rivers, party)\n"
                "You can also tell me a region of India if you have one in mind."
            )
            return state

        # 2 loops done and still nothing -> auto-fill
        if not state.get("categories"):
            state["categories"] = self.recommender.auto_fill_categories()

        # 🧰 call the LangChain TOOL
        cities = recommend_places.invoke({
            "categories": state["categories"],
            "region": state.get("region", ""),
        })
        state["recommended_cities"] = cities
        state["stage"] = "rec_await_city_choice"
        state["bot_response"] = self.recommender.format_recommendations(cities)
        return state

    # ---------------- ITINERARY ----------------
    def itinerary_node(self, state: ChatState):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")
        loop = state.get("loop_count", 0)

        # we don't know the city yet -> ask
        if not state.get("city") and stage != "itin_await_city":
            state["stage"] = "itin_await_city"
            state["intent"] = "itinerary"
            state["bot_response"] = "Which city would you like a trip plan for? 🗺️"
            return state

        if stage == "itin_await_city":
            state["city"] = self._extract_city(msg) or msg.strip().title()
            state["stage"] = "itin_await_details"
            state["bot_response"] = (
                f"{state['city']} it is! How many days, how many travelers, "
                "and which budget (economy / moderate / luxury)?"
            )
            return state

        # gather days / travelers / budget
        if stage == "itin_await_details" or not self._has_details(state):
            days, travelers, budget = parse_trip_details(msg)
            state["days"] = state.get("days") or days
            state["travelers"] = state.get("travelers") or travelers
            state["budget_level"] = state.get("budget_level") or budget

            if not self._has_details(state):
                if loop < 2:
                    state["stage"] = "itin_await_details"
                    state["loop_count"] = loop + 1
                    state["bot_response"] = (
                        "Got it! Please share the missing bits — number of days, "
                        "number of travelers, and budget (economy / moderate / luxury)."
                    )
                    return state
                # auto-fill after 2 tries
                state["days"] = state.get("days") or 3
                state["travelers"] = state.get("travelers") or 2
                state["budget_level"] = state.get("budget_level") or "moderate"

        # 🧰 call the LangChain TOOL to build the itinerary
        itinerary = plan_itinerary.invoke({
            "city": state["city"],
            "days": state["days"],
            "travelers": state["travelers"],
            "budget": state["budget_level"],
            "categories": state.get("categories") or [],
        })
        # use a "bud_" stage so the router sends "yes" to the BUDGET node
        state["stage"] = "bud_await_budget_offer"
        state["intent"] = "budget"
        state["loop_count"] = 0
        state["bot_response"] = itinerary + "\n\nWould you like a budget estimate for this trip? (yes / no)"
        return state

    # ---------------- BUDGET ----------------
    def budget_node(self, state: ChatState):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")
        loop = state.get("loop_count", 0)

        # was offered after an itinerary
        if stage == "bud_await_budget_offer":
            if is_yes(msg):
                return self._do_budget(state)
            state["stage"] = ""
            state["intent"] = ""          # ⭐ FIX 2: reset intent so next message is fresh
            state["bot_response"] = "No problem! Enjoy your trip. 🌟 Ask me anything else anytime."
            return state

        # standalone budget: gather city + details
        days, travelers, budget = parse_trip_details(msg)
        city = self._extract_city(msg)
        state["city"] = state.get("city") or city
        state["days"] = state.get("days") or days
        state["travelers"] = state.get("travelers") or travelers
        state["budget_level"] = state.get("budget_level") or budget

        if not (state.get("city") and self._has_details(state)):
            if loop < 2:
                state["stage"] = "bud_await_details"
                state["loop_count"] = loop + 1
                state["intent"] = "budget"
                state["bot_response"] = (
                    "Sure! For a budget estimate, tell me the city, number of days, "
                    "number of travelers, and budget level (economy / moderate / luxury)."
                )
                return state
            state["city"] = state.get("city") or "Manali"
            state["days"] = state.get("days") or 3
            state["travelers"] = state.get("travelers") or 2
            state["budget_level"] = state.get("budget_level") or "moderate"

        return self._do_budget(state)

    def _do_budget(self, state):
        # 🧰 call the LangChain TOOL
        text = estimate_budget.invoke({
            "city": state["city"],
            "days": state["days"],
            "travelers": state["travelers"],
            "budget": state["budget_level"],
        })
        state["stage"] = ""
        state["intent"] = ""              # ⭐ FIX 1: reset intent after budget is done
        state["bot_response"] = text
        return state

    # ---------------- QA ----------------
    def qa_node(self, state: ChatState):
        # 🧰 call the LangChain TOOL
        state["bot_response"] = answer_question.invoke({
            "question": state.get("user_message", ""),
        })
        state["stage"] = ""
        state["intent"] = ""              # ⭐ FIX 3: reset intent after answering
        return state

    # ---------------- small helpers ----------------
    def _has_details(self, state):
        return bool(state.get("days") and state.get("travelers") and state.get("budget_level"))

    def _match_city(self, msg, options):
        for c in options:
            if c.lower() in msg.lower():
                return c
        return None

    def _extract_city(self, msg):
        for c in self.city_index.all_cities():
            if c.lower() in msg.lower():
                return c
        return None

    # ---------------- public entry used by app.py ----------------
    def handle(self, state: ChatState):
        return self.graph.invoke(state)