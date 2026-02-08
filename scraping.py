"""
Script d'extraction des factures PDF depuis Mendeley Dataset
Theme: Factures électroniques fictives
Télécharge tous les PDFs du dataset tnj49gpmtz/2 vers Excel
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime

def download_mendeley_invoices(output_dir='mendeley_invoices', base_url="https://data.mendeley.com/datasets/tnj49gpmtz/2"):
    """
    Télécharge toutes les factures PDF du dataset Mendeley
    
    Args:
        output_dir: Dossier de destination
        base_url: URL de la page dataset
        
    Returns:
        DataFrame avec métadonnées des factures téléchargées
    """
    
    # Créer dossier si inexistant
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nRecherche des liens PDF sur Mendeley...")
    
    # Headers pour éviter blocage
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    # Récupérer page principale
    response = session.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    invoices_data = []
    theme = "Factures fictives Mendeley"
    downloaded_count = 0
    
    print("✓ Page dataset chargée")
    
    # Chercher liens de téléchargement (ZIP ou PDFs directs)
    download_links = soup.find_all('a', href=re.compile(r'(files|download|\.zip|\.pdf)', re.I))
    
    print(f"✓ {len(download_links)} lien(s) de téléchargement trouvé(s)")
    
    for idx, link in enumerate(download_links, 1):
        print(f"\n  → Traitement lien {idx}/{len(download_links)}...")
        
        href = link.get('href')
        if not href:
            continue
            
        full_url = urljoin(base_url, href)
        filename = os.path.basename(urlparse(full_url).path)
        
        invoice_info = {
            'Theme': theme,
            'Status': 'Téléchargée',
            'Source_URL': full_url,
            'Filename': filename,
            'File_Type': 'PDF' if filename.lower().endswith('.pdf') else 'ZIP',
            'Download_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Local_Path': '',
            'File_Size': 0,
            'Invoice_ID': '',
            'PDF_Image_Status': 'PDF téléchargé (extraction texte possible)'
        }
        
        # Extraire ID facture du nom si possible
        id_match = re.search(r'invoice_(\d+)', filename.lower())
        if id_match:
            invoice_info['Invoice_ID'] = id_match.group(1)
            print(f"    • ID Facture: {invoice_info['Invoice_ID']}")
        
        try:
            print(f"    • Téléchargement: {filename[:50]}...")
            
            # Télécharger fichier
            dl_response = session.get(full_url, stream=True, timeout=30)
            dl_response.raise_for_status()
            
            # Chemin local
            local_path = os.path.join(output_dir, filename)
            with open(local_path, 'wb') as f:
                for chunk in dl_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Infos fichier
            file_size = os.path.getsize(local_path)
            invoice_info['Local_Path'] = local_path
            invoice_info['File_Size'] = f"{file_size/1024:.1f} KB"
            
            downloaded_count += 1
            print(f"    ✓ Téléchargé: {filename} ({invoice_info['File_Size']})")
            
        except Exception as e:
            invoice_info['Status'] = f'Erreur: {str(e)[:50]}'
            print(f"    ✗ Erreur: {e}")
        
        invoices_data.append(invoice_info)
        time.sleep(1)  # Pause anti-ban
    
    print(f"\n✓ Total: {downloaded_count}/{len(download_links)} fichier(s) téléchargé(s)")
    return pd.DataFrame(invoices_data)


def extract_pdf_text(pdf_path):
    """
    Extrait texte basique d'un PDF (bonus pour analyse)
    Nécessite: pip install PyPDF2
    """
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text[:500]  # Premier 500 chars
    except:
        return "Extraction nécessiter PyPDF2"


def save_to_excel(df, output_file='mendeley_invoices.xlsx'):
    """
    Sauvegarde Excel (identique structure originale)
    """
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Invoices Data', index=False)
            worksheet = writer.sheets['Invoices Data']
            
            for idx, col in enumerate(df.columns):
                max_length = min(
                    max(df[col].astype(str).apply(len).max(), len(col)), 100
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        print(f"\n✓ Fichier Excel créé: {output_file}")
        print(f"✓ {len(df)} factures listées")
        
    except Exception as e:
        print(f"✗ Excel erreur: {e}")
        csv_file = output_file.replace('.xlsx', '.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSV créé: {csv_file}")


def main():
    """Fonction principale (structure identique FB script)"""
    print("=" * 70)
    print("EXTRACTION FACTURES PDF Mendeley Dataset")
    print("Dataset: tnj49gpmtz/2 - 1000+ factures fictives")
    print("=" * 70)
    
    output_dir = 'mendeley_invoices'
    output_excel = 'mendeley_factures.xlsx'
    
    print(f"\n→ Dossier sortie: {output_dir}")
    print(f"→ Excel sortie: {output_excel}")
    
    try:
        # Télécharger factures
        df = download_mendeley_invoices(output_dir)
        
        if len(df) > 0:
            print("\n" + "=" * 90)
            print("DÉTAIL COMPLET DES FACTURES TÉLÉCHARGÉES")
            print("=" * 90)
            
            for idx, row in df.iterrows():
                print(f"\n{'█' * 90}")
                print(f"FACTURE #{idx + 1}")
                print(f"{'█' * 90}")
                
                for column, value in row.items():
                    if pd.isna(value) or value == '':
                        display_value = "[Non disponible]"
                    else:
                        display_value = str(value)
                        if len(display_value) > 80:
                            display_value = display_value[:80] + "..."
                    
                    print(f"  ├─ {column:25s}: {display_value}")
                
                print(f"{'─' * 90}")
        
        print("\n" + "=" * 70)
        print("RÉSUMÉ DES DONNÉES EXTRAITES:")
        print("=" * 70)
        print(df.to_string(index=False))
        
        # Sauvegarder Excel
        save_to_excel(df, output_excel)
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
