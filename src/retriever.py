class Retriever:

    def __init__(self, retriever):
        self.retriever = retriever

    # Takes the user's question -> finds matching places -> returns them as text
    def search_documents(self, question: str) -> str:
        documents = self.retriever.invoke(question)
        return self.format_context(documents)

    # Turns the retrieved Documents into a clean text block for the LLM
    def format_context(self, documents: list) -> str:
        if not documents:
            raise Exception("No relevant documents found.")

        blocks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")

        return "\n\n".join(blocks)