"""
retriever.py
------------
Takes the place Documents that Chroma returns and turns them into clean
text (with nearby spots + local food) that the itinerary LLM can read.
"""


class Retriever:

    def __init__(self, retriever):
        self.retriever = retriever

    def search_documents(self, question: str) -> str:
        documents = self.retriever.invoke(question)
        return self.format_context(documents)

    def format_context(self, documents: list) -> str:
        if not documents:
            return ""

        blocks = []
        for i, doc in enumerate(documents, start=1):
            m = doc.metadata
            nearby = (m.get("nearby_places") or "").replace(",", ", ")
            food = (m.get("local_food") or "").replace(",", ", ")
            block = (
                f"[{i}] {m.get('place', 'unknown')} ({m.get('city', '')})\n"
                f"About: {doc.page_content}\n"
                f"Nearby: {nearby}\n"
                f"Local food: {food}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)
