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

## 🔑 Configuration (Quota Google Books)

Si vous recevez des erreurs `429 Quota Exceeded`, vous devez utiliser une clé API Google Books :
1. Créez un projet sur la [Google Cloud Console](https://console.cloud.google.com/).
2. Activez la **Books API**.
3. Créez des identifiants (Clé API).
4. Créez un fichier `.env` à la racine du projet :
   ```env
   GOOGLE_BOOKS_API_KEY=votre_cle_ici
   ```
5. Relancez l'application en exportant la variable ou en utilisant un outil comme `dotenv`.

## 🧪 Tests

Pour lancer la suite de tests :
```bash
pytest
```

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
