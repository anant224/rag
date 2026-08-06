class Retriever:

    def __init__(self, retriever):
        self.retriever = retriever

    def search_documents(self, question: str) -> str:
        documents = self.retriever.invoke(question)
        return self.format_context(documents)

    def format_context(self, documents: list) -> str:
        if not documents:
            raise Exception("No relevant documents found.")

        blocks = []
        for i, doc in enumerate(documents, start=1):
            m = doc.metadata
            nearby = ", ".join(m.get("nearby_places") or [])
            food = ", ".join(m.get("local_food") or [])
            block = (
                f"[{i}] {m.get('place', 'unknown')} ({m.get('city', '')})\n"
                f"About: {doc.page_content}\n"
                f"Nearby: {nearby}\n"
                f"Local food: {food}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)