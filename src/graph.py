"""
graph.py  — the LangGraph "brain" (routing + conversation flow)

Beginner idea:
  * One user message = one run of the graph.
  * We keep a small "state" dict that REMEMBERS the conversation
    (city, days, travelers, budget, last_city...). app.py stores it per
    session_id, so the bot remembers you until you refresh / start New Chat.
  * The router picks ONE of 5 skills: recommend / itinerary / budget / qa / offtopic
"""

import re
from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.tools import (
    recommend_places,
    plan_itinerary,
    estimate_budget,
    answer_question,
)


# ---------- the memory that flows through the graph ----------
class ChatState(TypedDict, total=False):
    user_message: str
    intent: str
    stage: str
    categories: list
    region: str
    recommended_cities: list
    city: str
    last_city: str          # remembers the city we last talked about
    days: int
    travelers: int
    budget_level: str
    loop_count: int
    bot_response: str


# ---------- read trip details out of a sentence ----------
def parse_details(text):
    t = text.lower()

    days = None
    m = re.search(r"(\d+)\s*day", t)
    if m:
        days = int(m.group(1))

    travelers = None
    m = re.search(r"(\d+)\s*(people|person|persons|traveler|travelers|traveller|travellers|adult|adults|member|members|pax)", t)
    if m:
        travelers = int(m.group(1))

    # budget words -> low / medium / high (old words still accepted)
    budget = None
    words = {"low": "low", "cheap": "low", "economy": "low",
             "medium": "medium", "moderate": "medium", "mid": "medium",
             "high": "high", "luxury": "high", "premium": "high"}
    for word, tier in words.items():
        if word in t:
            budget = tier
            break

    # if only bare numbers were given, guess: first=days, second=travelers
    if days is None or travelers is None:
        nums = re.findall(r"\d+", t)
        if len(nums) >= 2:
            days = days or int(nums[0])
            travelers = travelers or int(nums[1])
        elif len(nums) == 1 and days is None:
            days = int(nums[0])

    return days, travelers, budget


def is_yes(text):
    return any(w in text.lower() for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "haan"])


# ===========================================================
class ChatBrain:

    def __init__(self, recommender, city_index, generator):
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
        g.add_node("offtopic", self.offtopic_node)

        g.set_entry_point("router")
        g.add_conditional_edges("router", self.route, {
            "recommend": "recommend", "itinerary": "itinerary",
            "budget": "budget", "qa": "qa", "offtopic": "offtopic",
        })
        for n in ["recommend", "itinerary", "budget", "qa", "offtopic"]:
            g.add_edge(n, END)
        return g.compile()

    # ---------------- ROUTER ----------------
    def router_node(self, state):
        if state.get("stage"):                       # already mid-conversation
            return state
        if state.get("intent") in ["recommend", "itinerary", "budget", "qa"]:
            return state                             # a button was clicked
        state["intent"] = self._classify(state.get("user_message", ""))
        return state

    def route(self, state):
        stage = state.get("stage", "")
        if stage.startswith("rec_"):
            return "recommend"
        if stage.startswith("itin_"):
            return "itinerary"
        if stage.startswith("bud_"):
            return "budget"
        return state.get("intent", "qa")

    def _classify(self, msg):
        prompt = f"""You are the router of a TRAVEL assistant.
Put the message in ONE of these buckets:
- recommend : wants a city / destination suggestion
- itinerary : wants a trip plan for a city
- budget    : wants a trip cost estimate
- qa        : any travel question (food, weather, attractions, packing, best time...)
- offtopic  : NOT about travel at all (maths, coding, jokes, general chit-chat...)

Message: "{msg}"
Answer with only one word:"""
        out = self.generator.generate(prompt).strip().lower()
        for k in ["recommend", "itinerary", "budget", "qa", "offtopic"]:
            if k in out:
                return k
        return "qa"

    # ---------------- OFF-TOPIC (politely decline) ----------------
    def offtopic_node(self, state):
        state.update(stage="", intent="")
        state["bot_response"] = (
            "Ah, that's a little outside my world! 🙂 I'm your travel buddy, so I help with "
            "destinations, trip plans, budgets and travel questions.\n"
            "Try me with something like \"suggest a mountain trip\" or \"plan 3 days in Goa\"."
        )
        return state

    # ---------------- RECOMMEND ----------------
    def recommend_node(self, state):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")
        loop = state.get("loop_count", 0)

        # user is choosing one of the 3 suggested cities
        if stage == "rec_await_city_choice":
            city = self._match(msg, state.get("recommended_cities", []))
            if city:
                state.update(city=city, last_city=city, stage="itin_await_details",
                             intent="itinerary", loop_count=0)
                self._absorb(state, msg)
                state["bot_response"] = self._ask_missing(state, f"Great pick — {city}! 🎉")
            else:
                state["bot_response"] = "Just type one of the cities I suggested and we'll take it from there. 🙂"
            return state

        # figure out what kind of trip they like
        cats, region = self.recommender.extract_preferences(msg)
        if region:
            state["region"] = region
        if cats:
            state["categories"] = cats

        if not state.get("categories") and loop < 2:
            state.update(stage="rec_await_categories", loop_count=loop + 1, intent="recommend")
            state["bot_response"] = (
                "Love it! What kind of trip are you in the mood for? 🌍\n"
                "(mountains, beaches, temples, history, adventure, rivers, or nightlife)"
            )
            return state

        if not state.get("categories"):
            state["categories"] = self.recommender.auto_fill_categories()

        cities = recommend_places.invoke({
            "categories": state["categories"], "region": state.get("region", ""),
        })
        state.update(recommended_cities=cities, stage="rec_await_city_choice")
        state["bot_response"] = self.recommender.format_recommendations(cities)
        return state

    # ---------------- ITINERARY ----------------
    def itinerary_node(self, state):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")
        loop = state.get("loop_count", 0)

        # did the user name a city in THIS message?
        mentioned = self._mentions_a_city(msg)
        if mentioned == "unknown":
            # they named a city we don't have (e.g. Lahore) -> decline politely
            state.update(stage="", intent="")
            state["bot_response"] = self._unknown_city_msg()
            return state
        if mentioned:
            # a known city -> use it (overrides any old remembered city)
            state.update(city=mentioned, last_city=mentioned)

        # no city yet -> ask for it
        if not state.get("city") and stage != "itin_await_city":
            state.update(stage="itin_await_city", intent="itinerary")
            state["bot_response"] = "Awesome! Which city are you thinking of? 🗺️"
            return state

        # they typed a city -> check it exists in our data
        if stage == "itin_await_city":
            city = self._extract_city(msg)
            if not city:
                state["bot_response"] = self._unknown_city_msg()
                return state
            state.update(city=city, last_city=city, stage="itin_await_details")
            self._absorb(state, msg)                       # grab any details they gave too
            state["bot_response"] = self._ask_missing(state, f"{city} — lovely choice! 🎉")
            return state

        # collect days / travelers / budget (only ask for what's missing)
        if stage == "itin_await_details" or not self._have_all(state):
            self._absorb(state, msg)
            if not self._have_all(state):
                if loop < 2:
                    state.update(stage="itin_await_details", loop_count=loop + 1)
                    state["bot_response"] = self._ask_missing(state)
                    return state
                state.setdefault("days", 3)
                state.setdefault("travelers", 2)
                state.setdefault("budget_level", "medium")

        itinerary = plan_itinerary.invoke({
            "city": state["city"], "days": state["days"], "travelers": state["travelers"],
            "budget": state["budget_level"], "categories": state.get("categories") or [],
        })
        state.update(stage="bud_offer", intent="budget", loop_count=0, last_city=state["city"])
        state["bot_response"] = itinerary + "\n\nWant me to estimate a budget for this trip too? (yes / no)"
        return state

    # ---------------- BUDGET ----------------
    def budget_node(self, state):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")
        loop = state.get("loop_count", 0)

        # we offered a budget after the itinerary
        if stage == "bud_offer":
            if is_yes(msg):
                return self._do_budget(state)
            state.update(stage="", intent="")
            state["bot_response"] = "No worries — have an amazing trip! 🌟 I'm here whenever you need me."
            return state

        # standalone budget request -> read city + details from the message
        city = self._extract_city(msg)
        if city:
            state.update(city=city, last_city=city)
        self._absorb(state, msg)

        # they named a city we don't have
        if state.get("city") and not self._known(state["city"]):
            state.update(stage="", intent="")
            state["bot_response"] = self._unknown_city_msg()
            return state

        # still missing city or details -> ask only for what's missing (max 2 times)
        if not state.get("city") or not self._have_all(state):
            if loop < 2:
                bits = []
                if not state.get("city"):
                    bits.append("which city")
                if not state.get("days"):
                    bits.append("how many days")
                if not state.get("travelers"):
                    bits.append("how many travelers")
                if not state.get("budget_level"):
                    bits.append("budget (low / medium / high)")
                state.update(stage="bud_await", loop_count=loop + 1, intent="budget")
                state["bot_response"] = "Happy to estimate! Could you tell me " + self._join(bits) + "?"
                return state
            # after 2 tries, fill sensible defaults (but a city is still required)
            state.setdefault("days", 3)
            state.setdefault("travelers", 2)
            state.setdefault("budget_level", "medium")
            if not state.get("city"):
                state.update(stage="", intent="")
                state["bot_response"] = "I still need a city for the estimate. " + self._unknown_city_msg()
                return state

        return self._do_budget(state)

    def _do_budget(self, state):
        text = estimate_budget.invoke({
            "city": state["city"], "days": state["days"],
            "travelers": state["travelers"], "budget": state["budget_level"],
        })
        state.update(stage="", intent="")
        state["bot_response"] = text
        return state

    # ---------------- QA (remembers the last city) ----------------
    def qa_node(self, state):
        msg = state.get("user_message", "")
        city = self._extract_city(msg)
        if city:
            state["last_city"] = city               # remember it for follow-ups
            question = msg
        elif state.get("last_city"):
            # follow-up like "best place to eat there" -> attach remembered city
            question = f"{msg} (the user is asking about {state['last_city']})"
        else:
            question = msg
        state["bot_response"] = answer_question.invoke({"question": question})
        state.update(stage="", intent="")
        return state

    # ---------------- small helpers ----------------
    def _absorb(self, state, msg):
        d, t, b = parse_details(msg)
        if d and not state.get("days"):
            state["days"] = d
        if t and not state.get("travelers"):
            state["travelers"] = t
        if b and not state.get("budget_level"):
            state["budget_level"] = b

    def _have_all(self, state):
        return bool(state.get("days") and state.get("travelers") and state.get("budget_level"))

    def _ask_missing(self, state, opening=""):
        need = []
        if not state.get("days"):
            need.append("how many days")
        if not state.get("travelers"):
            need.append("how many travelers")
        if not state.get("budget_level"):
            need.append("your budget (low / medium / high)")
        text = "Could you tell me " + self._join(need) + "?"
        return (opening + "\n\n" + text) if opening else text

    def _join(self, items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return items[0] + " and " + items[1]
        return ", ".join(items[:-1]) + ", and " + items[-1]

    def _unknown_city_msg(self):
        sample = ", ".join(self.city_index.all_cities()[:6])
        return (
            "Ah, we're still adding that city to our travel database — it should be there soon! 🙏\n"
            f"For now I can plan a wonderful trip to places like: {sample}.\n"
            "Which one would you like?"
        )

    def _match(self, msg, options):
        for c in options:
            if c.lower() in msg.lower():
                return c
        return None

    def _extract_city(self, msg):
        for c in self.city_index.all_cities():
            if c.lower() in msg.lower():
                return c
        return None

    def _mentions_a_city(self, msg):
        """Return a known city if named, 'unknown' if the user clearly asks
        for a trip to some place we don't have, or None otherwise."""
        # is a KNOWN city named?
        known = self._extract_city(msg)
        if known:
            return known
        # do they clearly want a trip for a place, but it's not in our data?
        trip_words = ["itinerary", "trip", "plan", "visit", "go to", " for "]
        if any(w in msg.lower() for w in trip_words):
            # a capitalized place-like word we don't recognize?
            names = [w.strip(",.?!") for w in msg.split() if w[:1].isupper()]
            if names:
                return "unknown"
        return None

    def _known(self, city):
        return bool(city) and city.lower() in [c.lower() for c in self.city_index.all_cities()]

    # ---------------- public entry (used by app.py) ----------------
    def handle(self, state):
        return self.graph.invoke(state)
