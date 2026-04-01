import pytest
from app.synthesis import synthesizer
from app.models import BookResult


class TestMergeAndSort:
    def test_deduplicates_by_isbn_and_keeps_best_prices(self):
        results = [
            BookResult(title="Book A", isbn="123", price_momox=10.0, price_recyclivre=5.0),
            BookResult(title="Book A", isbn="123", price_momox=12.0, price_recyclivre=0.0),
            BookResult(title="Book B", isbn="456", price_momox=20.0, price_recyclivre=25.0),
        ]
        merged = synthesizer.merge_and_sort(results)

        assert len(merged) == 2
        # Book B should be first because 25 > 12
        assert merged[0].isbn == "456"
        assert merged[0].price_recyclivre == 25.0
        # Book A should have merged best prices
        assert merged[1].isbn == "123"
        assert merged[1].price_momox == 12.0
        assert merged[1].price_recyclivre == 5.0

    def test_empty_input_returns_empty(self):
        assert synthesizer.merge_and_sort([]) == []

    def test_single_book(self):
        results = [BookResult(title="Solo", isbn="789", price_momox=3.0)]
        merged = synthesizer.merge_and_sort(results)
        assert len(merged) == 1
        assert merged[0].title == "Solo"

    def test_sorts_by_max_price_descending(self):
        results = [
            BookResult(title="Cheap", isbn="001", price_momox=1.0, price_recyclivre=2.0),
            BookResult(title="Mid", isbn="002", price_momox=5.0, price_recyclivre=3.0),
            BookResult(title="Expensive", isbn="003", price_momox=0.5, price_recyclivre=15.0),
        ]
        merged = synthesizer.merge_and_sort(results)
        assert [b.isbn for b in merged] == ["003", "002", "001"]

    def test_confidence_keeps_highest(self):
        results = [
            BookResult(title="Book", isbn="111", confidence_score=0.5),
            BookResult(title="Book", isbn="111", confidence_score=0.9),
        ]
        merged = synthesizer.merge_and_sort(results)
        assert len(merged) == 1
        assert merged[0].confidence_score == 0.9


class TestBookResultModel:
    def test_max_price_computed_field(self):
        book = BookResult(title="T", isbn="X", price_momox=10.0, price_recyclivre=15.0)
        assert book.max_price == 15.0

    def test_max_price_serialized_in_json(self):
        book = BookResult(title="T", isbn="X", price_momox=10.0, price_recyclivre=5.0)
        data = book.model_dump()
        assert "max_price" in data
        assert data["max_price"] == 10.0

    def test_default_values(self):
        book = BookResult(title="T", isbn="X")
        assert book.price_momox == 0.0
        assert book.price_recyclivre == 0.0
        assert book.confidence_score == 0.0
        assert book.author is None
        assert book.max_price == 0.0

    def test_mutable_prices(self):
        """Ensure Pydantic model allows mutation (frozen=False)."""
        book = BookResult(title="T", isbn="X", price_momox=1.0)
        book.price_momox = 5.0
        assert book.price_momox == 5.0


class TestCsvRows:
    def test_to_csv_rows(self):
        results = [
            BookResult(title="Book A", isbn="123", author="Author 1",
                       price_momox=10.0, price_recyclivre=5.0, confidence_score=0.85)
        ]
        rows = synthesizer.to_csv_rows(results)
        assert len(rows) == 1
        assert rows[0]["Title"] == "Book A"
        assert rows[0]["Best Price (€)"] == 10.0
        assert rows[0]["Confidence"] == "0.85"

    def test_to_csv_rows_unknown_author(self):
        results = [BookResult(title="No Author", isbn="999")]
        rows = synthesizer.to_csv_rows(results)
        assert rows[0]["Author"] == "Unknown"
