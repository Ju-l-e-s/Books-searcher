# BookShelf Sniper 📚🎯

**BookShelf Sniper** est une application web intelligente qui permet de transformer une simple photo de tranches de livres en un tableau comparatif des meilleurs prix de rachat.

## 🚀 Fonctionnalités

- **Détection IA** : Utilise YOLO pour segmenter précisément les tranches de livres sur une photo.
- **OCR Haute Précision** : Extraction du texte via EasyOCR pour identifier les titres et auteurs.
- **Résolution ISBN** : Interroge les API Google Books et Open Library pour retrouver l'ISBN exact.
- **Comparateur de Prix** : Scrape en temps réel les prix de rachat sur **Momox** et **RecycLivre**.
- **Interface Moderne** : UI réactive avec zone de drop, prévisualisation et export des résultats (CSV/JSON).

## 🛠️ Installation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/Ju-l-e-s/Books-searcher.git
   cd Books-searcher
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -e .
   playwright install chromium
   ```

3. **Lancer l'application** :
   ```bash
   uvicorn app.main:app --reload
   ```

## 🧪 Tests

Pour lancer la suite de tests :
```bash
pytest
```

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
