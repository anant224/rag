"""
qa.py
-----
The "Ask a travel question" helper.

RAG-first: semantically search the BLOGS collection in Chroma for the most
relevant city overviews, then let the LLM answer freely using them.
If nothing relevant is found, fall back to a general LLM answer.
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
            prompt = f"""You are a friendly travel assistant. Answer the user's question
using MAINLY the City Information below. If it does not fully cover the question,
you may add general travel knowledge, but prefer the given info.

City Information:
{context}

Question: {question}

Give a helpful, natural answer:"""
        else:
            # fallback: no relevant blog found
            prompt = f"""You are a friendly travel assistant. Answer this travel
question helpfully and concisely:

Question: {question}

Answer:"""

        return self.generator.generate(prompt)
