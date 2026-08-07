"""
city_index.py
-------------
Builds a dictionary about every city for the RECOMMENDER.

For each city we store:
  - category_counts : {category: how many places have it}  <-- the key change
  - categories      : the set of categories (kept for reference)
  - region          : e.g. "North India"
  - popularity      : simple score (number of places + stay minutes)
  - blog            : the city overview paragraph
"""

from collections import defaultdict


class CityIndex:

    def __init__(self, place_docs, blog_docs):
        self.index = self._build(place_docs, blog_docs)

    def _build(self, place_docs, blog_docs):
        # city -> {category: count of places with that category}
        cat_counts = defaultdict(lambda: defaultdict(int))
        region = {}
        pop = defaultdict(float)
        count = defaultdict(int)

        for doc in place_docs:
            m = doc.metadata
            city = m.get("city")
            if not city:
                continue
            for c in (m.get("categories") or "").split(","):
                c = c.strip().lower()
                if c:
                    cat_counts[city][c] += 1     # count how many places have it
            if m.get("region"):
                region[city] = m.get("region")
            pop[city] += float(m.get("popularity") or 0)
            count[city] += 1

        # city -> blog text
        blogs = {d.metadata.get("city"): d.page_content for d in blog_docs}

        index = {}
        for city in cat_counts:
            index[city] = {
                "category_counts": dict(cat_counts[city]),
                "categories": set(cat_counts[city].keys()),
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