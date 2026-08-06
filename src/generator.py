"""
generator.py
------------
Talks to Gemini. Two jobs:
  1. generate()      -> send any prompt, get the text answer back
  2. build_prompt()  -> builds the brochure-style itinerary prompt
"""

from langchain_google_genai import ChatGoogleGenerativeAI


class Generator:

    def __init__(self, model_name: str, api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=api_key,
        )

    def generate(self, prompt: str) -> str:
            result = self.model.invoke(prompt)
            content = result.content

            # Gemini sometimes returns content as a LIST of parts -> join into text
            if isinstance(content, list):
                text = ""
                for part in content:
                    if isinstance(part, str):
                        text += part
                    elif isinstance(part, dict):
                        text += part.get("text", "")
                return text

            return content
    

    def build_prompt(self, question, rag_context, city_intro=""):
        return f"""You are a smart travel itinerary planner. Create an engaging, realistic, day-by-day itinerary using ONLY the places in the Retrieved Context. Do not invent places, food, or facts.

City Intro (for a warm opening line):
{city_intro}

Retrieved Context (available places, with nearby spots and food options):
{rag_context}

User's Trip Request:
{question}

Instructions:
1. Use ONLY places from the Retrieved Context. Never add places not listed there.
2. Match places to the user's travel types and spread them across the exact number of days.
3. For each day, write a short catchy TITLE that captures the theme of the day.
4. Under each day, write friendly full-sentence bullets. Group the flow into morning, afternoon and evening WITHOUT exact clock times.
5. In the bullets, mention what the traveler can DO nearby and suggest a FOOD option / local dish from the context.
6. End each day with a "Highlight of the day:" line and a short "Note:" line.
7. Warm, brochure-like tone. Bold the important place names.

FORMAT:

Day 1 | <Catchy Day Title>
------------------------------------------------
- <descriptive sentence with the place in bold>
- <nearby spot they can explore>
- <a food option / local dish to try>
Highlight of the day: <short highlight>
Note: <short note>

(Repeat for each day.)

Now generate the itinerary:"""
