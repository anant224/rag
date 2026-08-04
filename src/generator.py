from langchain_google_genai import ChatGoogleGenerativeAI


class Generator:

    def __init__(self, model_name: str, api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,          # 0 = factual, consistent answers (good for RAG)
            google_api_key=api_key,
        )

    # Sends the final prompt to Gemini and returns the answer text
    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    # Builds the full prompt: rules + history + retrieved places + question
    def build_prompt(self, question: str, memory_context: str, rag_context: str) -> str:

        return f"""You are a smart travel itinerary planner. Your job is to create a clear, realistic, day-by-day travel itinerary using ONLY the places found in the Retrieved Context below. You must not invent places, prices, or facts that are not present in the Retrieved Context.

Conversation History:
{memory_context}

Retrieved Context (available places and their details):
{rag_context}

User's Trip Request:
{question}

The user's request may include:
- Destination (city name)
- Duration (number of days)
- Budget level (economy, moderate, or luxury)
- Traveler count (number of people)
- Travel types (one or more: religious, historical, shopping, food, sightseeing, etc.)

Instructions:
1. Build the itinerary using ONLY places from the Retrieved Context. Never add places that are not listed there.
2. Match the places to the user's chosen travel types. If the user selects multiple types (e.g., religious + food), include a good mix of places covering each selected type.
3. Spread the plan across the exact number of days given. Balance each day so it is not overloaded — group nearby places together and keep a logical order (use timings, best_time, and nearby info from the context when available).
4. Respect the budget level:
   - economy: prefer free or low-cost places and cheaper food options.
   - moderate: balance cost and experience.
   - luxury: prioritize premium/higher-rated experiences and dining.
5. Keep the traveler count in mind (e.g., family-friendly, group-suitable, or couple-friendly places) when the context provides such details.
6. For each day, present a simple schedule (Morning / Afternoon / Evening) with the place name, why it fits, ideal time to visit, and any relevant detail (entry cost, food nearby) taken from the context.
7. Suggest nearby food/restaurants from the context when available, matching the budget level.
8. If the Retrieved Context does not have enough places to fill all the days, honestly say so and build the best itinerary possible with what is available — do not fill gaps with made-up places.
9. Keep the tone friendly, clear, and easy to follow. Use day-wise headings.
10. If source/file names are present in the context, you may cite them briefly at the end.

Now generate the itinerary:"""
    