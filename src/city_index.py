"""
city_index.py
-------------
Builds a small dictionary about every city, used by the RECOMMENDER.
It is built ONCE from the place + blog documents.

For each city we store:
  - categories : all categories across that city's places (a set)
  - region     : e.g. "North India"
  - popularity : a simple score (number of places + stay minutes)
  - blog       : the city overview paragraph (from the blog record)
"""

from collections import defaultdict


class CityIndex:

    def __init__(self, place_docs, blog_docs):
        self.index = self._build(place_docs, blog_docs)

    def _build(self, place_docs, blog_docs):
        cats = defaultdict(set)     # city -> set of categories
        region = {}                 # city -> region
        pop = defaultdict(float)    # city -> popularity number
        count = defaultdict(int)    # city -> number of places

        for doc in place_docs:
            m = doc.metadata
            city = m.get("city")
            if not city:
                continue
            for c in (m.get("categories") or "").split(","):
                c = c.strip().lower()
                if c:
                    cats[city].add(c)
            if m.get("region"):
                region[city] = m.get("region")
            pop[city] += float(m.get("popularity") or 0)
            count[city] += 1

        # city -> blog text
        blogs = {d.metadata.get("city"): d.page_content for d in blog_docs}

        index = {}
        for city in cats:
            index[city] = {
                "categories": cats[city],
                "region": region.get(city, ""),
                "popularity": count[city] * 10 + pop[city] / 100.0,
                "blog": blogs.get(city, ""),
            }
        return index

    def all_cities(self):
        return list(self.index.keys())

    def get(self, city):
        return self.index.get(city)

    def blog(self, city):
        info = self.index.get(city)
        return info["blog"] if info else ""
