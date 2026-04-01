from typing import List
from app.models import BookResult


class ResultsSynthesizer:
    @staticmethod
    def merge_and_sort(results: List[BookResult]) -> List[BookResult]:
        """Deduplicate by ISBN and sort by best price descending."""
        if not results:
            return []

        # Dedup by ISBN, keeping the one with higher confidence/prices
        unique_results: dict[str, BookResult] = {}
        for res in results:
            if res.isbn not in unique_results:
                unique_results[res.isbn] = res
            else:
                existing = unique_results[res.isbn]
                existing.price_momox = max(existing.price_momox, res.price_momox)
                existing.price_recyclivre = max(existing.price_recyclivre, res.price_recyclivre)
                existing.confidence_score = max(existing.confidence_score, res.confidence_score)

        sorted_list = list(unique_results.values())
        sorted_list.sort(key=lambda x: x.max_price, reverse=True)
        return sorted_list

    @staticmethod
    def to_csv_rows(results: List[BookResult]) -> list[dict]:
        """Convert results to a list of dicts for CSV export (no pandas needed)."""
        return [
            {
                "Title": r.title,
                "ISBN": r.isbn,
                "Author": r.author or "Unknown",
                "Momox (€)": r.price_momox,
                "RecycLivre (€)": r.price_recyclivre,
                "Best Price (€)": r.max_price,
                "Confidence": f"{r.confidence_score:.2f}"
            }
            for r in results
        ]


# Singleton instance
synthesizer = ResultsSynthesizer()
