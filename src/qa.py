"""
qa.py  — "Ask a travel question"

Searches the BLOG vectors in Chroma for the most relevant city info,
then lets Gemini answer in a warm, natural way. If nothing relevant is
found, it still answers helpfully as a friendly travel guide.
"""


class QA:

    def __init__(self, blog_retriever, generator):
        self.blog_retriever = blog_retriever   # searches the "blogs" collection
        self.generator = generator

    def answer(self, question: str) -> str:
        docs = self.blog_retriever.invoke(question)

        if docs:
            context = "\n\n".join(
                f"City: {d.metadata.get('city')}\n{d.page_content}" for d in docs
            )
            prompt = f"""You are a warm, friendly travel guide. Answer the traveler's
question in a natural, conversational way, using MAINLY the City Info below.
If the info doesn't fully cover it, you may add general travel knowledge.
Keep it helpful and easy to read (a short, friendly paragraph).

City Info:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""You are a warm, friendly travel guide. Answer this travel
question in a natural, helpful way (a short, friendly paragraph):

Question: {question}

Answer:"""

        return self.generator.generate(prompt)
