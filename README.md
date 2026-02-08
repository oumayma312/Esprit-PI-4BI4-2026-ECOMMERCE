# Projet de Scraping de Factures PDF

Ce projet télécharge et analyse les factures électroniques depuis le dataset Mendeley.

## Dataset Source
- **Nom**: Samples of electronic invoices
- **URL**: https://data.mendeley.com/datasets/tnj49gpmtz/2
- **Contenu**: 3000 factures PDF (1000 valides, 1000 avec IBAN coloré, 1000 avec char-spacing)
- **Taille**: 129.59 MB

## Scripts Disponibles

### 1. `download_invoices.py`
Télécharge et extrait toutes les factures PDF depuis Mendeley.

**Usage:**
```bash
python download_invoices.py
```

**Sortie:**
- Dossier `invoices_pdf/` contenant 3000 factures PDF

### 2. `create_excel_from_pdfs.py`
Génère un fichier Excel avec les métadonnées de toutes les factures.

**Usage:**
```bash
python create_excel_from_pdfs.py
```

**Sortie:**
- `mendeley_factures.xlsx` avec les colonnes:
  - Invoice_ID
  - Filename
  - Dataset_Type
  - Charspace_Coefficient
  - Color_Blue_Value
  - File_Size_KB
  - Local_Path
  - Status

### 3. `scraping.py`
Script alternatif pour scraper le site Mendeley (méthode par liens directs).

## Installation

```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install requests beautifulsoup4 pandas openpyxl
```

## Workflow Complet

1. Télécharger les factures:
```bash
python download_invoices.py
```

2. Générer le fichier Excel:
```bash
python create_excel_from_pdfs.py
```

## Structure des Données

Les factures sont organisées en 3 catégories:
- **Valid Information**: Factures avec informations valides standard
- **Colored IBAN**: IBAN avec fond coloré (RGB 255,255,240 à 255,255,254)
- **Char-spaced IBAN**: IBAN avec espacement modifié (coefficient 0.001 à 1)

## Nomenclature des Fichiers

```
invoice_<id>(_charspace_<coefficient>)(_color_B_<blue_value>).pdf
```

## Licence
Dataset sous licence CC BY 4.0

## Auteurs du Dataset
- Kozłowski, Marek
- Weichbroth, Paweł
- Politechnika Gdanska
