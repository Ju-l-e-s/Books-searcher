import pytest
from app.synthesis import synthesizer
from app.models import SpineResult, BookCandidate


class TestMergeAndSortSpines:
    def test_sorts_candidates_within_spines(self):
        spines = [
            SpineResult(
                original_text="Text 1",
                candidates=[
                    BookCandidate(title="Cheap", isbn="1", price_momox=1.0),
                    BookCandidate(title="Expensive", isbn="2", price_momox=10.0),
                ]
            )
        ]
        sorted_spines = synthesizer.merge_and_sort_spines(spines)
        assert sorted_spines[0].candidates[0].title == "Expensive"
        assert sorted_spines[0].candidates[1].title == "Cheap"

    def test_sorts_spines_by_best_candidate_price(self):
        spines = [
            SpineResult(
                original_text="Spine Mid",
                candidates=[BookCandidate(title="Mid", isbn="2", price_momox=5.0)]
            ),
            SpineResult(
                original_text="Spine Cheap",
                candidates=[BookCandidate(title="Cheap", isbn="1", price_momox=1.0)]
            ),
            SpineResult(
                original_text="Spine Expensive",
                candidates=[BookCandidate(title="Expensive", isbn="3", price_momox=15.0)]
            ),
        ]
        sorted_spines = synthesizer.merge_and_sort_spines(spines)
        assert sorted_spines[0].candidates[0].title == "Expensive"
        assert sorted_spines[1].candidates[0].title == "Mid"
        assert sorted_spines[2].candidates[0].title == "Cheap"

    def test_empty_input_returns_empty(self):
        assert synthesizer.merge_and_sort_spines([]) == []


class TestBookCandidateModel:
    def test_max_price_computed_field(self):
        book = BookCandidate(title="T", isbn="X", price_momox=10.0, price_recyclivre=15.0)
        assert book.max_price == 15.0

    def test_default_values(self):
        book = BookCandidate(title="T", isbn="X")
        assert book.price_momox == 0.0
        assert book.price_recyclivre == 0.0
        assert book.confidence_score == 0.0
        assert book.author is None
        assert book.max_price == 0.0


class TestCsvRows:
    def test_to_csv_rows(self):
        spines = [
            SpineResult(
                original_text="Text",
                candidates=[
                    BookCandidate(title="Book A", isbn="123", author="Author 1",
                                  price_momox=10.0, price_recyclivre=5.0, confidence_score=0.85)
                ]
            )
        ]
        rows = synthesizer.to_csv_rows(spines)
        assert len(rows) == 1
        assert rows[0]["Title"] == "Book A"
        assert rows[0]["Best Price (€)"] == 10.0
        assert rows[0]["Confidence"] == "0.85"

    def test_to_csv_rows_unknown_author(self):
        spines = [
            SpineResult(
                original_text="Text",
                candidates=[BookCandidate(title="No Author", isbn="999")]
            )
        ]
        rows = synthesizer.to_csv_rows(spines)
        assert rows[0]["Author"] == "Unknown"
