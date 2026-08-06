class BudgetEstimator:

    # per-person-per-day rates (INR) -> this IS your economy/moderate/luxury factor
    RATES = {
        "economy":  {"hotel": 800,  "food": 500,  "travel": 300,  "activities": 300},
        "moderate": {"hotel": 2000, "food": 1000, "travel": 600,  "activities": 700},
        "luxury":   {"hotel": 5000, "food": 2500, "travel": 1500, "activities": 1500},
    }

    # Only list cities that differ from baseline (1.0). Any city not here = 1.0.
    CITY_FACTOR = {
        "Manali": 1.2,
        "Shimla": 1.2,
        "Goa": 1.3,
        "Mumbai": 1.3,
        "Amritsar": 1.0,
        # add more as you like; unlisted cities default to 1.0 automatically
    }

    def get_factor(self, city):
        return self.CITY_FACTOR.get(city, 1.0)   # safe default for any city

    # ---------- main calculation ----------
    def estimate(self, city, days, travelers, budget):
        budget = budget.lower()
        r = self.RATES.get(budget, self.RATES["moderate"])
        factor = self.get_factor(city)
        nights = max(1, days - 1)   # a 3-day trip = 2 hotel nights

        # city factor applied to hotel + travel (these vary most by city)
        hotel      = r["hotel"]  * factor * travelers * nights
        travel     = r["travel"] * factor * travelers * days
        food       = r["food"]   * travelers * days
        activities = r["activities"] * travelers * days

        subtotal = hotel + travel + food + activities
        misc = subtotal * 0.10          # 10% buffer for shopping/tips/extras
        total = subtotal + misc

        return {
            "hotel": round(hotel),
            "travel": round(travel),
            "food": round(food),
            "activities": round(activities),
            "misc": round(misc),
            "total": round(total),
        }

    # ---------- clean text block to display ----------
    def format_budget(self, city, days, travelers, budget, b):
        low = round(b["total"] * 0.9)
        high = round(b["total"] * 1.1)
        return (
            f"Estimated Budget (for {travelers} travelers, {days} days, {budget}):\n"
            f"- Hotel .............. Rs {b['hotel']}\n"
            f"- Food ............... Rs {b['food']}\n"
            f"- Local Travel ....... Rs {b['travel']}\n"
            f"- Activities ......... Rs {b['activities']}\n"
            f"- Miscellaneous (10%). Rs {b['misc']}\n"
            f"----------------------------------------\n"
            f"- Total (approx) ..... Rs {low} - Rs {high}\n"
            f"(Approximate estimate, excludes flights.)"
        )