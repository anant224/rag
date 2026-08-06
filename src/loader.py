"""
loader.py
---------
Reads all .jsonl files from the data folder and splits each line into
TWO kinds of records:

  1. PLACE record  -> has a "place" key  (Golden Temple, Hadimba Temple ...)
  2. BLOG  record  -> has a "blog"  key  (one city overview per city)

Place records -> used for the itinerary (the "places" Chroma collection)
Blog  records -> used for Ask-a-question (the "blogs"  Chroma collection)

NOTE: some records (like the blog) may be spread across MULTIPLE lines
(pretty-printed JSON). So we keep adding lines to a buffer until one full
JSON object is complete -> this handles both single-line and multi-line data.
"""

import json
from pathlib import Path
from langchain_core.documents import Document


class DocumentLoader:

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def load_documents(self):
        place_docs = []   # list of place Documents
        blog_docs = []    # list of blog Documents

        print(f"Loading from: {self.folder_path}")

        # go through every .jsonl file (all your city files)
        for file_path in self.folder_path.glob("**/*.jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                buffer = ""              # collects lines until one JSON object is complete
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    buffer += line       # add this line to the buffer
                    try:
                        record = json.loads(buffer)   # did we get a full object?
                    except json.JSONDecodeError:
                        continue          # not complete yet -> keep adding lines

                    # success -> we have a complete record, reset the buffer
                    buffer = ""
                    if "blog" in record:
                        blog_docs.append(self._make_blog_doc(record))
                    elif "place" in record:
                        place_docs.append(self._make_place_doc(record))

        print(f"Loaded {len(place_docs)} places and {len(blog_docs)} blogs")
        return place_docs, blog_docs

    # ---------- build a PLACE document ----------
    def _make_place_doc(self, r):
        text = (
            f"Place: {r.get('place')}\n"
            f"City: {r.get('city')}\n"
            f"About: {r.get('description', '')}"
        )
        # Chroma metadata must be simple, so lists become comma strings
        metadata = {
            "type": "place",
            "city": r.get("city"),
            "place": r.get("place"),
            "region": r.get("region_in_india"),
            "categories": ",".join(r.get("categories") or []),
            "nearby_places": ",".join(r.get("nearby_places") or []),
            "local_food": ",".join(r.get("local_food_specialties") or []),
            "popularity": r.get("recommended_stay_min", 0),
        }
        return Document(page_content=text, metadata=metadata)

    # ---------- build a BLOG document ----------
    def _make_blog_doc(self, r):
        metadata = {"type": "blog", "city": r.get("city")}
        return Document(page_content=r.get("blog", ""), metadata=metadata)