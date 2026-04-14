# Chatbot Marketing NLP (Python + Gemini)

Chatbot conversationnel en Python qui:
- Analyse automatiquement des CSV de datawarehouse
- Repond aux questions en langage naturel
- Propose des recommandations marketing actionnables
- Peut enrichir ses reponses via Gemini (optionnel)

## Fonctionnalites

- Analyse des donnees produits/ventes/campagnes depuis `datawarehouse/`
- Detection automatique de tous les fichiers CSV du dossier et profilage (lignes, colonnes, taux de valeurs manquantes)
- Detection d'opportunites marketing (produits peu vendus et accessibles)
- Requetes conversationnelles en francais (NLP avec spaCy)
- Recommandations strategie marketing adaptees au contexte
- Mode hybride:
  - moteur analytique local (fiable, base donnees)
  - Gemini (style naturel, reformulation intelligente)

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configuration

1. Copier `.env.example` vers `.env`
2. Remplir la cle API Gemini

Variables:
- `GEMINI_API_KEY`: cle API Gemini
- `GEMINI_MODEL`: par defaut `gemini-1.5-flash`
- `DATA_DIR`: dossier des CSV (par defaut `./datawarehouse`)

## Lancement

```bash
python main.py
```

## Lancement web (Flask)

```bash
python web_app.py
```

Puis ouvrir: http://127.0.0.1:5000

## Exemples de questions

- `Quel produit dois-je promouvoir avec un petit budget ?`
- `Montre le top 10 des produits`
- `Donne-moi une synthese KPI`
- `Quels fichiers CSV sont charges ?`
- `table csv: dim_product`
- `Trouver produit: plateau`
- `Recommande une strategie marketing pour augmenter les ventes`

## Notes techniques

- Le bot gere les formats numeriques heterogenes (virgules/points, devise)
- Si Gemini est indisponible, le bot continue avec des reponses analytiques locales
- Le systeme fonctionne meme si certaines colonnes sont absentes (avec fallback)
