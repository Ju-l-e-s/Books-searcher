# Books-searcher Architecture

## Stack
- FastAPI + Uvicorn
- YOLO (ultralytics) + EasyOCR pour detection texte
- Open Library API pour ISBN lookup
- Playwright pour scrap Momox + RecycLivre

## Data Flow
1. Photo upload → YOLO detection → EasyOCR extraction
2. ISBN → Open Library API → metadata livre
3. Prix → scrapers Momox/RecycLivre
4. Synthèse → frontend JSON

## TODO
- [ ] ISBN resolver completion (en cours)
- [ ] Scraper Momox/RecycLivre
- [ ] Frontend refinement

