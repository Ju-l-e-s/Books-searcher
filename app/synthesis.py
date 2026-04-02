from typing import List
from app.models import SpineResult, BookCandidate


class ResultsSynthesizer:
    @staticmethod
    def merge_and_sort_spines(spines: List[SpineResult]) -> List[SpineResult]:
        """Deduplicate by top ISBN across spines and sort by best candidate's max price."""
        if not spines:
            return []
        
        # Sort candidates within each spine by price descending
        for spine in spines:
            spine.candidates.sort(key=lambda x: x.max_price, reverse=True)
            
        # Sort spines by their best candidate's price
        spines.sort(key=lambda x: x.candidates[0].max_price if x.candidates else 0, reverse=True)
        return spines

    @staticmethod
    def to_csv_rows(spines: List[SpineResult]) -> list[dict]:
        """Convert spines to a list of dicts for CSV export (using top candidate)."""
        rows = []
        for s in spines:
            if not s.candidates: continue
            r = s.candidates[0]
            rows.append({
                "Title": r.title,
                "ISBN": r.isbn,
                "Author": r.author or "Unknown",
                "Momox (€)": r.price_momox,
                "RecycLivre (€)": r.price_recyclivre,
                "Best Price (€)": r.max_price,
                "Confidence": f"{r.confidence_score:.2f}"
            })
        return rows


# Singleton instance
synthesizer = ResultsSynthesizer()
