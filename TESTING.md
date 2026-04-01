# Procédure de Test — BookShelf Sniper

Suivez ces étapes pour vérifier que le système fonctionne correctement.

## 1. Installation de l'Environnement

Dans votre terminal, exécutez les commandes suivantes depuis la racine du projet (`/Users/jules/Desktop/SearchingBooks`) :

```bash
# Installation des dépendances Python
pip install -e .

# Installation des navigateurs pour Playwright (nécessaire pour le scraping)
playwright install chromium
```

## 2. Lancement du Serveur

Démarrez le serveur de développement FastAPI :

```bash
uvicorn app.main:app --reload
```

Le serveur sera accessible à l'adresse suivante : [http://localhost:8000](http://localhost:8000)

## 3. Test de l'Interface Utilisateur (Recommandé)

1. Ouvrez votre navigateur sur **`http://localhost:8000`**.
2. Vous devriez voir une interface moderne avec un logo "BookShelf Sniper".
3. **Action** : Glissez-déposez une photo de tranches de livres (Format JPG/PNG) sur la zone en pointillés.
4. **Attente** : Un indicateur "Analyse en cours..." apparaît. (Le premier appel peut prendre 30-60s pour charger les modèles IA).
5. **Résultat** : Un tableau s'affiche avec les colonnes `Titre`, `ISBN`, `Momox`, `RecycLivre` et `Meilleur Prix`. Les prix sont en Euros (€).

## 4. Test Rapide en Ligne de Commande (API)

Si vous préférez tester via `curl` avec un fichier image local (ex: `photo.jpg`) :

```bash
curl -X POST "http://localhost:8000/analyze-shelf" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@photo.jpg"
```

## Points de Vérification

- [ ] Les logs du terminal affichent "Downloading yolov8n.pt" au premier lancement.
- [ ] Le tableau de résultats contient bien des prix de rachat valides (> 0€).
- [ ] Les boutons "Exporter CSV" et "Exporter JSON" téléchargent bien un fichier structuré.
- [ ] En cas d'erreur de segmentation, l'interface affiche un message clair.
