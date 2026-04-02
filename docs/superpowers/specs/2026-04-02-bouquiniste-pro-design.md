# Design Spec: Bouquiniste Pro (Multi-Edition & Dimensions)

**Date:** 2026-04-02  
**Feature:** Multi-candidate selection per spine, dimension triangulation, automatic cover display, and advanced export (CSV/Excel).

---

## 1. Goal
Provide a professional tool ("Bouquiniste Pro") that allows users to precisely identify the correct book edition from multiple candidates, using visual confirmation (covers) and physical dimensions, while supporting flexible export options.

## 2. Architecture & Data Flow
The system will shift from a 1:1 (Spine -> Book) relationship to a 1:N relationship (Spine -> Candidates).

### Backend (Python/FastAPI)
- **`app/models.py`**:
    - `BookCandidate`: A model representing one potential edition.
        - `isbn`, `title`, `author`, `publisher`, `year`, `confidence_score`.
        - `dimensions`: `{ "height": float, "width": float, "thickness": float }`.
        - `cover_url`: Link to the cover image.
        - `prices`: `{ "momox": float, "recyclivre": float }`.
    - `SpineResult`: A model representing one detected spine from the image.
        - `original_text`: The OCR string.
        - `bbox_ratio`: Geometric ratio from YOLO.
        - `candidates`: List of 1 to 3 `BookCandidate` objects, sorted by score.
- **`app/isbn_resolver.py`**:
    - Modify `resolve()` to return the Top 3 candidates instead of Top 1.
    - Extract `dimensions` (height/width/thickness) and `cover_url` from API responses (Google Books / Open Library).
    - Use dimensions as an internal scoring signal (e.g., if photo ratio matches candidate dimensions, boost score).
- **`app/main.py`**:
    - Update endpoints (`/analyze-shelf` and `/analyze-shelf-texts`) to return a `List[SpineResult]`.

### Frontend (HTML/JS)
- **Table Structure**:
    - **Cover Column**: Automatically displays `<img>` for the selected candidate.
    - **Edition Column**: Contains a `<select>` menu. Changing the selection updates:
        - Cover Image
        - Dimensions
        - Prices (Momox/RecycLivre)
    - **Dimensions Column**: Displays `H x W x T` (e.g., `18.0 x 11.5 x 2.0 cm`).
    - **Selection Column**: Checkbox for targeted export.
- **Export Logic**:
    - Support **CSV** and **Excel (.xlsx)**.
    - Options: "Export All" or "Export Selected".
    - Use `SheetJS` (xlsx.full.min.js) for browser-side Excel generation.

## 3. UI/UX
- **Visual Feedback**: Covers load automatically as soon as results appear.
- **Interaction**: The row "Active State" is driven by the `<select>` index.
- **Precision Proof**: Dimensions are clearly visible, proving the tool's accuracy.

## 4. Testing Strategy
- **Unit Tests**:
    - Verify `ISBNResolver` correctly extracts dimensions and cover URLs.
    - Verify `ISBNResolver` returns exactly 3 candidates (when available).
- **Frontend Tests**:
    - Manual verification of `<select>` triggering row updates.
    - Verification of Excel/CSV download contents.

---
**Spec self-review:**
- No placeholders like TBD.
- Internal consistency: Model names (`BookCandidate`, `SpineResult`) match across files.
- Scope: Focused on multi-edition and export.
- Ambiguity: Explicitly mentions `SheetJS` for export.
