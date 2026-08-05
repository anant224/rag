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
            place = doc.metadata.get("place", "unknown")
            city = doc.metadata.get("city", "unknown")
            blocks.append(f"[{i}] ({place}, {city})\n{doc.page_content}")

        return "\n\n".join(blocks)