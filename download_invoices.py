"""
Script pour télécharger les factures PDF depuis Mendeley Data
Dataset: Samples of electronic invoices (1000 factures)
URL: https://data.mendeley.com/datasets/tnj49gpmtz/2
"""

import requests
import os
from zipfile import ZipFile

# Configuration
DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/tnj49gpmtz/download/2"
OUTPUT_DIR = "invoices_pdf"
ZIP_FILE = "invoices.zip"

def download_file(url, filename):
    """Télécharge un fichier depuis une URL"""
    print(f"Téléchargement depuis {url}...")
    print("Cela peut prendre quelques minutes (121 MB)...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"\rProgression: {progress:.1f}%", end='', flush=True)
    
    print(f"\n✓ Téléchargement terminé: {filename}")
    return filename

def extract_zip(zip_path, extract_to):
    """Extrait un fichier ZIP"""
    print(f"\nExtraction des fichiers PDF vers {extract_to}...")
    
    os.makedirs(extract_to, exist_ok=True)
    
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.pdf')]
        print(f"✓ {len(pdf_files)} fichiers PDF extraits")
    
    return extract_to

def list_pdf_files(directory):
    """Liste tous les fichiers PDF"""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def main():
    """Fonction principale"""
    print("=" * 60)
    print("Téléchargement des factures PDF depuis Mendeley Data")
    print("=" * 60)
    
    try:
        # Télécharger le fichier ZIP
        zip_file_path = download_file(DOWNLOAD_URL, ZIP_FILE)
        
        # Extraire les fichiers PDF
        extract_dir = extract_zip(zip_file_path, OUTPUT_DIR)
        
        # Lister les fichiers PDF extraits
        pdf_files = list_pdf_files(OUTPUT_DIR)
        
        print(f"\n{'=' * 60}")
        print(f"✓ SUCCÈS!")
        print(f"{'=' * 60}")
        print(f"Nombre total de factures PDF: {len(pdf_files)}")
        print(f"Emplacement: {os.path.abspath(OUTPUT_DIR)}")
        
        print(f"\nExemples de fichiers:")
        for pdf_file in pdf_files[:10]:
            print(f"  - {os.path.basename(pdf_file)}")
        
        if len(pdf_files) > 10:
            print(f"  ... et {len(pdf_files) - 10} autres fichiers")
        
        # Nettoyer le fichier ZIP
        print(f"\nNettoyage du fichier ZIP temporaire...")
        os.remove(zip_file_path)
        print("✓ Nettoyage terminé")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Erreur lors du téléchargement: {e}")
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
