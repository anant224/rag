"""
budget.py
---------
Rate-based budget estimator (data-independent, always works).
Budget tiers are now: low / medium / high  (easier for users than
economy/moderate/luxury). We still accept the old words as synonyms.
"""


class BudgetEstimator:

    # per-person-per-day rates (INR) by budget tier
    RATES = {
        "low":    {"hotel": 800,  "food": 500,  "travel": 300,  "activities": 300},
        "medium": {"hotel": 2000, "food": 1000, "travel": 600,  "activities": 700},
        "high":   {"hotel": 5000, "food": 2500, "travel": 1500, "activities": 1500},
    }

    # only list cities that differ from baseline (1.0). Unlisted cities = 1.0.
    CITY_FACTOR = {
        "Manali": 1.2,
        "Shimla": 1.2,
        "Goa": 1.3,
        "Mumbai": 1.3,
        "Amritsar": 1.0,
    }

    # accept old words too, so nothing breaks
    SYNONYMS = {
        "economy": "low", "cheap": "low", "budget": "low",
        "moderate": "medium", "mid": "medium",
        "luxury": "high", "premium": "high",
    }

    def _normalize(self, budget):
        b = (budget or "medium").lower()
        b = self.SYNONYMS.get(b, b)     # map synonyms
        if b not in self.RATES:
            b = "medium"
        return b

    def get_factor(self, city):
        return self.CITY_FACTOR.get(city, 1.0)

    def estimate(self, city, days, travelers, budget):
        budget = self._normalize(budget)
        r = self.RATES[budget]
        factor = self.get_factor(city)
        nights = max(1, days - 1)

        hotel      = r["hotel"]  * factor * travelers * nights
        travel     = r["travel"] * factor * travelers * days
        food       = r["food"]   * travelers * days
        activities = r["activities"] * travelers * days

        subtotal = hotel + travel + food + activities
        misc = subtotal * 0.10
        total = subtotal + misc

        return {
            "hotel": round(hotel),
            "travel": round(travel),
            "food": round(food),
            "activities": round(activities),
            "misc": round(misc),
            "total": round(total),
            "tier": budget,
        }

    def format_budget(self, city, days, travelers, budget, b):
        low = round(b["total"] * 0.9)
        high = round(b["total"] * 1.1)
        return (
            f"💰 Here's an approximate budget for your {b['tier']}-tier trip to {city} "
            f"({travelers} travelers, {days} days):\n"
            f"- Hotel:          Rs {b['hotel']}\n"
            f"- Food:           Rs {b['food']}\n"
            f"- Local Travel:   Rs {b['travel']}\n"
            f"- Activities:     Rs {b['activities']}\n"
            f"- Miscellaneous:  Rs {b['misc']}\n"
            f"------------------------------------\n"
            f"- Total (approx): Rs {low} - Rs {high}\n\n"
            f"Just a ballpark to help you plan — it doesn't include flights. 🙂"
        )
