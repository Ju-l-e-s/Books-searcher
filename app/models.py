from pydantic import BaseModel, computed_field
from typing import List, Optional


class BookResult(BaseModel):
    model_config = {"frozen": False}

    title: str
    isbn: str
    author: Optional[str] = None
    price_momox: float = 0.0
    price_recyclivre: float = 0.0
    confidence_score: float = 0.0

    @computed_field
    @property
    def max_price(self) -> float:
        """Best available price — now properly serialized in JSON responses."""
        return max(self.price_momox, self.price_recyclivre)


class AnalysisResponse(BaseModel):
    results: List[BookResult]


class OCRItem(BaseModel):
    text: str
    bbox_ratio: Optional[float] = None


class AnalysisRequest(BaseModel):
    items: List[OCRItem]
