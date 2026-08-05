from langchain_google_genai import ChatGoogleGenerativeAI


class Generator:

    def __init__(self, model_name: str, api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content[0]["text"]

    def build_prompt(self, question: str, memory_context: str, rag_context: str) -> str:
        return f"""You are a smart travel itinerary planner. Create a clear, realistic, day-by-day itinerary using ONLY the places in the Retrieved Context below. Do not invent places, prices, or facts that are not present in the Retrieved Context.

Conversation History:
{memory_context}

Retrieved Context (available places and details):
{rag_context}

User's Trip Request:
{question}

The request may include: Destination (city), Duration (days), Budget level (economy/moderate/luxury), Traveler count, and Travel types (one or more: religious, historical, shopping, food, adventure, etc.).

Instructions:
1. Use ONLY places from the Retrieved Context. Never add places not listed there.
2. Match places to the user's travel types. If multiple types are given, include a good mix covering each.
3. Spread places across the exact number of days. Group nearby places and keep a logical order (use timings/best_time when available).
4. Respect budget (economy = free/low-cost, moderate = balanced, luxury = premium/higher-rated).
5. Consider traveler count (family/couple/group-friendly) when the context mentions it.
6. For each day use a simple Morning / Afternoon / Evening layout with the place name, why it fits, best time, cost, and a nearby food option from the context.
7. If there aren't enough places for all days, say so honestly - do not fill with made-up places.
8. Keep it clean and easy to follow with day-wise headings.

Now generate the itinerary:"""