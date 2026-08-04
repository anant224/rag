from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, JSONLoader


class DocumentLoader:

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load_documents(self):
        documents = []

        try:
            print(f"Loading from: {self.folder_path}")

            documents += DirectoryLoader(
                self.folder_path,
                glob="**/*.jsonl",         # loads ALL .jsonl files (your 50-60 files)
                loader_cls=JSONLoader,
                loader_kwargs={
                    "jq_schema": ".",       # take the whole JSON object on each line
                    "text_content": False,  # allow non-string JSON values to load
                    "json_lines": True,     # IMPORTANT: one JSON object per line
                },
                show_progress=True,         # nice progress bar for 60 files
            ).load()

            print(f"Loaded {len(documents)} place records")

        except Exception as e:
            print(f"Error loading {self.folder_path}: {e}")
            return

        return documents