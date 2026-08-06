"""
recommender.py
--------------
The "Recommend a place" helper. Two jobs:

  1. extract_preferences() -> uses the LLM to read a free-text query and
     pull out the user's CATEGORIES (travel types) and optional REGION.

  2. recommend()           -> plain-Python matching on the CityIndex:
        - filter by region (only if the user gave one)
        - score each city = how many chosen categories it covers
        - KEEP ONLY cities that actually match (no zero-match padding)
        - sort by (match score -> popularity)
        - return the top 3 cities

The matching is plain Python (NOT RAG) -> reliable and easy to explain.
"""

import json
import random

# the final category list you decided on
VALID_CATEGORIES = [
    "religious", "mountain", "beaches", "heritage",
    "adventure", "rivers", "party",
]

VALID_REGIONS = [
    "north india", "south india", "east india",
    "west india", "central india",
]


class Recommender:

    def __init__(self, city_index, generator):
        self.city_index = city_index
        self.generator = generator   # used only for the extraction step

    # ---------- 1. pull categories + region from free text ----------
    def extract_preferences(self, text: str):
        prompt = f"""Read the user's travel message and extract their preferences.

User message: "{text}"

Return ONLY a JSON object with two keys:
- "categories": a list from {VALID_CATEGORIES} that match the user's interests (can be empty)
- "region": one of {VALID_REGIONS} if clearly mentioned, else ""

Rules:
- Map temple/spiritual -> religious, hills/snow/trek/mountains -> mountain, sea/beach -> beaches,
  fort/museum/historic -> heritage, trekking/rafting/sports -> adventure,
  river/lake -> rivers, pub/nightlife/club -> party.
- Return valid JSON only.

JSON:"""
        raw = self.generator.generate(prompt).strip()
        return self._safe_parse(raw)

    def _safe_parse(self, raw: str):
        # remove code fences if the model added them
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        cats = [c.lower() for c in data.get("categories", []) if c.lower() in VALID_CATEGORIES]
        region = (data.get("region") or "").lower()
        if region not in VALID_REGIONS:
            region = ""
        return cats, region

    # ---------- 2. match + rank -> top 3 cities ----------
    def recommend(self, categories, region="", top_n=3):
        # if still no categories, pick a sensible default
        if not categories:
            categories = ["heritage"]

        scored = []
        for city in self.city_index.all_cities():
            info = self.city_index.get(city)

            # region filter only if the user gave one
            if region and info["region"].lower() != region:
                continue

            overlap = len(set(categories) & info["categories"])
            scored.append((overlap, info["popularity"], city))

        # if the region filter removed everyone, retry without region
        if not scored:
            for city in self.city_index.all_cities():
                info = self.city_index.get(city)
                overlap = len(set(categories) & info["categories"])
                scored.append((overlap, info["popularity"], city))

        # ⭐ KEEP ONLY cities that actually match at least one category
        #    (this stops non-matching popular cities from padding the list)
        matches = [s for s in scored if s[0] > 0]
        if matches:
            scored = matches

        # sort by matches, then popularity (both high -> first)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [city for _, _, city in scored[:top_n]]

    # ---------- build a friendly text with short blurbs ----------
    def format_recommendations(self, cities):
        lines = ["Here are 3 great picks for you:\n"]
        for i, city in enumerate(cities, start=1):
            blog = self.city_index.blog(city)
            blurb = blog.split(".")[0] + "." if blog else "A wonderful destination."
            lines.append(f"{i}. {city} — {blurb}")
        lines.append("\nWhich one would you like to explore? (type the city name)")
        return "\n".join(lines)

    # ---------- auto-fill after 2 loops ----------
    def auto_fill_categories(self):
        return [random.choice(VALID_CATEGORIES)]
