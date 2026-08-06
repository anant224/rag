from langchain_google_genai import ChatGoogleGenerativeAI


class Generator:

    def __init__(self, model_name: str, api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    def build_prompt(self, question, rag_context):
        return f"""You are a smart travel itinerary planner. Create an engaging, realistic, day-by-day itinerary using ONLY the places in the Retrieved Context. Do not invent places, food, or facts that are not present in the context.

Retrieved Context (available places, with nearby spots and food options):
{rag_context}

User's Trip Request:
{question}

Instructions:
1. Use ONLY places from the Retrieved Context. Never add places not listed there.
2. Match places to the user's travel types and spread them across the exact number of days.
3. For each day, write a short catchy TITLE that captures the theme of that day.
4. Under each day, write descriptive bullet points in full friendly sentences (not just place names). Naturally group activities into morning, afternoon and evening flow WITHOUT writing exact clock times.
5. In the bullets, also mention what the traveler can DO nearby (using the Nearby info from the context) and suggest a FOOD option or local dish (using the Local food info from the context).
6. End each day with a "Highlight of the day:" line and a short helpful "Note:" line.
7. Keep the tone warm and engaging, like a travel brochure. Bold the important place names.

FORMAT (follow this style exactly):

Day 1 | <Catchy Day Title>
------------------------------------------------
- <Full descriptive sentence about the first activity, mentioning the place in bold.>
- <Another sentence covering a nearby spot they can explore.>
- <A sentence suggesting a food option or local dish to try.>
Highlight of the day: <short highlight>
Note: <short practical note>

Day 2 | <Catchy Day Title>
------------------------------------------------
- ...
Highlight of the day: ...
Note: ...

(Repeat for every day. Keep it clean, warm, and easy to read.)

Now generate the itinerary:"""