"""
graph.py  — the LangGraph "brain" (routing + conversation flow)

Rules:
  * Mid-task, we deterministically check if the message ANSWERS the current
    question. If yes -> continue. If it's a new request/question -> switch.
  * Chit-chat ("hey") inside a flow re-asks the question (does not abandon it).
  * A NEW city clears the previous trip's days/travelers/budget (no stale data).
  * Numbers are VALIDATED (no negative / zero / huge days or travelers) and a
    final safety guard means a bad number can NEVER generate a trip.
  * We never assume inputs; we keep asking only for what's missing (no limit).
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
    last_city: str
    days: int
    travelers: int
    budget_level: str
    bot_response: str


# ---------- read numbers/budget out of a sentence ----------
def parse_details(text):
    t = text.lower()

    days = None
    m = re.search(r"(-?\d+)\s*day", t)
    if m:
        days = int(m.group(1))

    travelers = None
    m = re.search(r"(-?\d+)\s*(people|person|persons|traveler|travelers|traveller|travellers|adult|adults|member|members|pax)", t)
    if m:
        travelers = int(m.group(1))

    budget = None
    words = {"low": "low", "cheap": "low", "economy": "low",
             "medium": "medium", "moderate": "medium", "mid": "medium",
             "high": "high", "luxury": "high", "premium": "high", "good": "high"}
    for word, tier in words.items():
        if word in t:
            budget = tier
            break

    if days is None and travelers is None:
        nums = re.findall(r"-?\d+", t)
        if len(nums) >= 2:
            days = int(nums[0])
            travelers = int(nums[1])
        elif len(nums) == 1:
            days = int(nums[0])

    return days, travelers, budget


def is_yes(text):
    return any(w in text.lower() for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "haan", "yup", "go ahead"])


def is_plain_no(text):
    return text.lower().strip().strip(".!") in ("no", "nope", "no thanks", "no thank you", "not now", "nah")


def looks_like_question(text):
    t = text.lower().strip()
    if t.endswith("?"):
        return True
    starters = ("where ", "what ", "how ", "which ", "can ", "could ", "is ",
                "are ", "do ", "does ", "any ", "when ", "who ", "why ", "tell me")
    return t.startswith(starters)


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

    # ================= ROUTER =================
    def router_node(self, state):
        msg = state.get("user_message", "")
        stage = state.get("stage", "")

        if not stage and state.get("intent") in ["recommend", "itinerary", "budget", "qa"]:
            return state

        if stage and self._answers_current(state, msg):
            return state

        if stage:
            self._reset_task(state)
        state["intent"] = self._classify(msg)
        return state

    def route(self, state):
        if state.get("intent") == "offtopic":
            return "offtopic"
        stage = state.get("stage", "")
        if stage.startswith("rec_"):
            return "recommend"
        if stage.startswith("itin_"):
            return "itinerary"
        if stage.startswith("bud_"):
            return "budget"
        return state.get("intent", "qa")

    # ---- deterministic: does the message answer the current question? ----
    def _answers_current(self, state, msg):
        stage = state.get("stage", "")

        if stage == "rec_await_categories":
            cats, _ = self.recommender.extract_preferences(msg)
            if cats:
                return True
            return not self._is_new_request(msg)

        if stage == "rec_await_city_choice":
            if self._find_known_city(msg) or self._match(msg, state.get("recommended_cities", [])):
                return True
            cats, _ = self.recommender.extract_preferences(msg)
            if cats:
                return True
            return not self._is_new_request(msg)

        if stage == "itin_await_city":
            if self._find_known_city(msg):
                return True
            if self._is_new_request(msg):
                return False
            return not looks_like_question(msg) and len(msg.split()) <= 4

        if stage in ("itin_await_details", "bud_await"):
            if looks_like_question(msg) and not self._find_known_city(msg):
                d0, t0, b0 = parse_details(msg)
                if d0 is None and t0 is None and not b0:
                    return False        # it's a real question, switch to QA
            d, t, b = parse_details(msg)
            if d is not None or t is not None or b:
                return True
            return bool(re.fullmatch(r"-?\d+", msg.strip()))

        if stage == "bud_offer":
            low = msg.lower()
            if is_yes(msg) or any(w in low for w in ["budget", "cost", "estimate", "how much"]):
                return True
            if is_plain_no(msg):
                return True
            return False

        return False

    def _is_new_request(self, msg):
        low = msg.lower().strip()
        if low in ("hi", "hey", "hello", "yo", "hmm", "ok", "okay", "cool", "thanks"):
            return False
        if looks_like_question(msg):
            return True
        if any(w in low for w in ["plan", "itinerary", "budget", "cost", "how much", "recommend", "suggest"]):
            return True
        return False

    def _classify(self, msg):
        prompt = f"""You are the intent router for a friendly TRAVEL assistant.

The user said: "{msg}"

Choose exactly ONE label:
- recommend : wants a CITY / DESTINATION suggestion to travel to
              ("where should I go", "suggest a place for holidays", "plan a trip for me")
- itinerary : wants a day-by-day trip PLAN for a place ("plan 3 days in Goa")
- budget    : wants a trip COST estimate ("how much for a Goa trip")
- qa        : ANY travel question or info request, INCLUDING flights, hotels,
              restaurants, nearby places, weather, packing, best time, safety,
              or "tell me about <place>"
- offtopic  : NOT about travel at all (maths, coding, jokes, greetings like "hi")

RULES:
- Questions about flights, hotels, food, or "nearby city" are always "qa".
- A plain greeting like "hi"/"hey" with no travel request is "offtopic".
- If unsure, prefer "qa".

Answer with only ONE word:"""
        out = self.generator.generate(prompt).strip().lower()
        for k in ["recommend", "itinerary", "budget", "qa", "offtopic"]:
            if k in out:
                return k
        return "qa"

    # ================= OFF-TOPIC =================
    def offtopic_node(self, state):
        state.update(stage="", intent="")
        state["bot_response"] = (
            "I'm your travel buddy, so I stick to travel! 🙂 I can recommend destinations, "
            "plan itineraries, estimate budgets, or answer travel questions.\n"
            "Try me with something like \"suggest a mountain trip\" or \"plan 3 days in Goa\"."
        )
        return state

    # ================= RECOMMEND =================
    def recommend_node(self, state):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")

        if stage == "rec_await_city_choice":
            city = self._match(msg, state.get("recommended_cities", [])) or self._find_known_city(msg)
            if city and self._known(city):
                self._set_city(state, city)
                state.update(stage="itin_await_details", intent="itinerary")
                err = self._absorb_validated(state, msg)
                if err:
                    state["bot_response"] = err
                    return state
                if self._have_all(state):
                    return self._generate_itinerary(state)
                state["bot_response"] = self._ask_missing(state, self._city_ack(city))
                return state

            cats, region = self.recommender.extract_preferences(msg)
            if cats:
                state["categories"] = cats
                if region:
                    state["region"] = region
                return self._make_recommendations(state)

            state["bot_response"] = (
                "Just tell me which of those cities you'd like, or a different kind of trip "
                "(mountains, beaches, temples, history, adventure, rivers, nightlife). 🙂"
            )
            return state

        cats, region = self.recommender.extract_preferences(msg)
        if region:
            state["region"] = region
        if cats:
            state["categories"] = cats

        if not state.get("categories"):
            state.update(stage="rec_await_categories", intent="recommend")
            state["bot_response"] = (
                "Sure! Just tell me what kind of trip you're in the mood for. 🌍\n"
                "(mountains, beaches, temples, history, adventure, rivers, or nightlife)"
            )
            return state

        return self._make_recommendations(state)

    def _make_recommendations(self, state):
        cities = recommend_places.invoke({
            "categories": state["categories"], "region": state.get("region", ""),
        })
        state.update(recommended_cities=cities, stage="rec_await_city_choice", intent="recommend")
        state["bot_response"] = self.recommender.format_recommendations(cities)
        return state

    # ================= ITINERARY =================
    def itinerary_node(self, state):
        msg = state.get("user_message", "")

    def itinerary_node(self, state):
        msg = state.get("user_message", "")

        # fresh "plan a trip" request (not mid-details) -> clear old trip data
        if state.get("stage", "") not in ("itin_await_city", "itin_await_details"):
            for k in ["days", "travelers", "budget_level"]:
                state.pop(k, None)

        dest = self._destination(state, msg)

        dest = self._destination(state, msg)
        if isinstance(dest, tuple):
            self._reset_task(state)
            state["bot_response"] = self._unknown_city_msg()
            return state
        if dest:
            self._set_city(state, dest)

        if not state.get("city"):
            state.update(stage="itin_await_city", intent="itinerary")
            state["bot_response"] = "Awesome! Which city would you like to explore? 🗺️"
            return state

        err = self._absorb_validated(state, msg)
        if err:
            state.update(stage="itin_await_details", intent="itinerary")
            state["bot_response"] = err
            return state

        if not self._have_all(state):
            ack = self._city_ack(state["city"]) if dest else ""
            state.update(stage="itin_await_details", intent="itinerary")
            state["bot_response"] = self._ask_missing(state, ack)
            return state

        return self._generate_itinerary(state)

    def _generate_itinerary(self, state):
        # SAFETY GUARD: never generate with invalid numbers
        if not self._valid(state.get("days"), 1, 30):
            state.pop("days", None)
        if not self._valid(state.get("travelers"), 1, 50):
            state.pop("travelers", None)
        if not self._have_all(state):
            state.update(stage="itin_await_details", intent="itinerary")
            state["bot_response"] = self._ask_missing(state, "Let's just confirm the details 🙂")
            return state

        itinerary = plan_itinerary.invoke({
            "city": state["city"], "days": state["days"], "travelers": state["travelers"],
            "budget": state["budget_level"], "categories": state.get("categories") or [],
        })
        state.update(stage="bud_offer", intent="budget", last_city=state["city"])
        state["bot_response"] = itinerary + "\n\nWant me to estimate a budget for this trip too? (yes / no)"
        return state

    # ================= BUDGET =================
    def budget_node(self, state):
        stage = state.get("stage", "")
        msg = state.get("user_message", "")

        if stage == "bud_offer":
            low = msg.lower()
            wants = is_yes(msg) or any(w in low for w in ["budget", "cost", "estimate", "how much"])
            if wants:
                return self._do_budget(state)
            state.update(stage="", intent="")
            state["bot_response"] = "No worries — have an amazing trip! 🌟 I'm here whenever you need me."
            return state
        
                

        # fresh standalone budget request -> clear old trip data
        if stage not in ("bud_await", "bud_offer"):
            for k in ["days", "travelers", "budget_level"]:
                state.pop(k, None)


        dest = self._destination(state, msg)
        if isinstance(dest, tuple):
            self._reset_task(state)
            state["bot_response"] = self._unknown_city_msg()
            return state
        if dest:
            self._set_city(state, dest)

        if not state.get("city"):
            state.update(stage="bud_await", intent="budget")
            state["bot_response"] = "Sure! Which city is the trip for? 🙂"
            return state

        err = self._absorb_validated(state, msg)
        if err:
            state.update(stage="bud_await", intent="budget")
            state["bot_response"] = err
            return state

        if not self._have_all(state):
            state.update(stage="bud_await", intent="budget")
            state["bot_response"] = self._ask_missing(state)
            return state

        return self._do_budget(state)

    def _do_budget(self, state):
        # SAFETY GUARD: never estimate with invalid numbers
        if not self._valid(state.get("days"), 1, 30):
            state.pop("days", None)
        if not self._valid(state.get("travelers"), 1, 50):
            state.pop("travelers", None)
        if not self._have_all(state):
            state.update(stage="bud_await", intent="budget")
            state["bot_response"] = self._ask_missing(state, "Let's just confirm the details 🙂")
            return state

        text = estimate_budget.invoke({
            "city": state["city"], "days": state["days"],
            "travelers": state["travelers"], "budget": state["budget_level"],
        })
        state.update(stage="", intent="")
        state["bot_response"] = text
        return state

    # ================= QA =================
    def qa_node(self, state):
        msg = state.get("user_message", "")

        if self._is_generic_opener(msg):
            state.update(stage="", intent="")
            state["bot_response"] = (
                "Of course! 😊 What would you like to know? You can ask me about a "
                "destination, the best time to visit, local food, packing tips, and more."
            )
            return state

        city = self._find_known_city(msg)
        if city:
            state["last_city"] = city
            question = msg
        elif state.get("last_city") and any(w in f" {msg.lower()} " for w in
                                            [" there ", " here ", " that place ", " same place ",
                                             " this place ", " it ", " its "]):
            question = f"{msg} (the user is referring to {state['last_city']})"
        else:
            question = msg

        state["bot_response"] = answer_question.invoke({"question": question})
        state.update(stage="", intent="")
        return state

    # ================= input reading + validation =================
    def _absorb_validated(self, state, msg):
        text = msg.strip()

        # lone number answers the ONE numeric field still missing
        m = re.fullmatch(r"-?\d+", text)
        if m:
            n = int(m.group())
            missing = [f for f in ["days", "travelers"] if not state.get(f)]
            if len(missing) == 1:
                return self._set_number(state, missing[0], n)
            if "days" in missing:
                return self._set_number(state, "days", n)

        d, t, b = parse_details(msg)

        # validate even if a value is already set (so "-3 days" is always caught)
        if d is not None:
            err = self._set_number(state, "days", d, force=True)
            if err:
                return err
        if t is not None:
            err = self._set_number(state, "travelers", t, force=True)
            if err:
                return err
        if b and not state.get("budget_level"):
            state["budget_level"] = b
        return None

    def _set_number(self, state, field, n, force=False):
        if field == "days":
            if n <= 0:
                return "Hmm, the number of days can't be zero or negative. 😊 How many days are you planning?"
            if n > 30:
                return "That's a really long trip! Please pick between 1 and 30 days. How many days?"
            if force or not state.get("days"):
                state["days"] = n
        else:
            if n <= 0:
                return "The number of travelers should be at least 1. 🙂 How many people are traveling?"
            if n > 50:
                return "That's a big group! Please enter up to 50 travelers. How many people are traveling?"
            if force or not state.get("travelers"):
                state["travelers"] = n
        return None

    def _valid(self, n, lo, hi):
        return isinstance(n, int) and lo <= n <= hi

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

    def _city_ack(self, city):
        return f"{city} — lovely choice! 🎉"

    def _unknown_city_msg(self):
        sample = ", ".join(self.city_index.all_cities()[:6])
        return (
            "Ah, we're still adding that city to our travel database — it should be there soon! 🙏\n"
            f"For now I can plan a wonderful trip to places like: {sample}.\n"
            "Which one would you like?"
        )

    def _is_generic_opener(self, msg):
        t = msg.lower()
        return ("question" in t and len(t.split()) <= 6 and not self._find_known_city(msg))

    # ================= city detection =================
    def _set_city(self, state, city):
        """Set the city. If DIFFERENT from before, clear old trip details."""
        if city != state.get("city"):
            for k in ["days", "travelers", "budget_level"]:
                state.pop(k, None)
        state.update(city=city, last_city=city)

    def _destination(self, state, msg):
        known = self._find_known_city(msg)
        if known:
            return known
        if not state.get("city"):
            name = self._llm_place(msg)
            if name and name.upper() != "NONE":
                return ("unknown", name)
        return None

    def _llm_place(self, msg):
        prompt = f"""Read the user's message and find if they name a SPECIFIC city
or town they want to travel to or plan a trip for.

User message: "{msg}"

Rules:
- If a specific city/town is named, reply with ONLY that place's name.
- If NO specific city is named (e.g. "plan a trip for me", "suggest a place"),
  reply with exactly: NONE
- Reply with just the place name or NONE.

Answer:"""
        out = self.generator.generate(prompt).strip()
        out = out.splitlines()[0].strip().strip(".,!?\"'")
        if not out or len(out.split()) > 3:
            return "NONE"
        return out

    def _match(self, msg, options):
        for c in options:
            if c.lower() in msg.lower():
                return c
        return None

    def _find_known_city(self, msg):
        low = msg.lower()
        for c in self.city_index.all_cities():
            if c.lower() in low:
                return c
        return None

    def _known(self, city):
        return bool(city) and city.lower() in [c.lower() for c in self.city_index.all_cities()]

    def _reset_task(self, state):
        for k in ["stage", "categories", "region", "recommended_cities",
                  "city", "days", "travelers", "budget_level"]:
            state.pop(k, None)

    # ---------------- public entry (used by app.py) ----------------
    def handle(self, state):
        return self.graph.invoke(state)
