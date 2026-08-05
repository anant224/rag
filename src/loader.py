from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, JSONLoader


class DocumentLoader:

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def _metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["city"] = record.get("city")
        metadata["place"] = record.get("place")
        return metadata

    def load_documents(self):
        documents = []

        try:
            print(f"Loading from: {self.folder_path}")

            documents += DirectoryLoader(
                self.folder_path,
                glob="**/*.jsonl",              
                loader_cls=JSONLoader,
                loader_kwargs={
                    "jq_schema": ".",           
                    "content_key": "description",  
                    "json_lines": True,        
                    "metadata_func": self._metadata_func,
                },
                show_progress=True,
            ).load()

            print(f"Loaded {len(documents)} place records")

        except Exception as e:
            print(f"Error loading {self.folder_path}: {e}")
            return

        return documents