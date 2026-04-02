from pydantic import BaseModel, computed_field
from typing import List, Optional


class BookDimensions(BaseModel):
    height: Optional[float] = None
    width: Optional[float] = None
    thickness: Optional[float] = None

    def __str__(self) -> str:
        if not self.height: return "Unknown"
        return f"{self.height}x{self.width or '?'}x{self.thickness or '?'} cm"


class BookCandidate(BaseModel):
    model_config = {"frozen": False}

    title: str
    isbn: str
    author: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[str] = None
    price_momox: float = 0.0
    price_recyclivre: float = 0.0
    confidence_score: float = 0.0
    cover_url: Optional[str] = None
    dimensions: Optional[BookDimensions] = None

    @computed_field
    @property
    def max_price(self) -> float:
        return max(self.price_momox, self.price_recyclivre)


class SpineResult(BaseModel):
    original_text: str
    bbox_ratio: Optional[float] = None
    candidates: List[BookCandidate]


class AnalysisResponse(BaseModel):
    results: List[SpineResult]


class OCRItem(BaseModel):
    text: str
    bbox_ratio: Optional[float] = None


class AnalysisRequest(BaseModel):
    items: List[OCRItem]
