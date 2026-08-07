"""
budget.py
---------
Rate-based budget estimator (data-independent, always works).
Per-person-per-day rates by budget tier, scaled by travelers, days,
and a small per-city factor for hotel + local travel.
"""


class BudgetEstimator:

    # per-person-per-day rates (INR) -> this IS the economy/moderate/luxury factor
    RATES = {
        "economy":  {"hotel": 600,  "food": 500,  "travel": 300,  "activities": 300},
        "moderate": {"hotel": 1500, "food": 1000, "travel": 500,  "activities": 700},
        "luxury":   {"hotel": 4000, "food": 2000, "travel": 1200, "activities": 1300},
    }

    # only list cities that differ from baseline (1.0). Unlisted cities = 1.0.
    CITY_FACTOR = {
        "Manali": 1.2,
        "Shimla": 1.2,
        "Goa": 1.3,
        "Mumbai": 1.3,
        "Amritsar": 1.0,
    }

    def get_factor(self, city):
        return self.CITY_FACTOR.get(city, 1.0)

    def estimate(self, city, days, travelers, budget):
        budget = (budget or "moderate").lower()
        r = self.RATES.get(budget, self.RATES["moderate"])
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
        }

    def format_budget(self, city, days, travelers, budget, b):
        low = round(b["total"] * 0.9)
        high = round(b["total"] * 1.1)
        return (
            f"💰 Estimated Budget for {city} ({travelers} travelers, {days} days, {budget})\n"
            f"- Hotel:            Rs {b['hotel']}\n"
            f"- Food:             Rs {b['food']}\n"
            f"- Local Travel:     Rs {b['travel']}\n"
            f"- Activities:       Rs {b['activities']}\n"
            f"- Miscellaneous:    Rs {b['misc']}\n"
            f"------------------------------------\n"
            f"- Total (approx):   Rs {low} - Rs {high}\n"
            f"(Approximate estimate, excludes flights.)"
        )
