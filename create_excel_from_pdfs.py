"""
Créer un fichier Excel avec les métadonnées de toutes les factures PDF téléchargées
"""

import os
import pandas as pd
from datetime import datetime
import re

def scan_pdf_directory(base_dir='invoices_pdf'):
    """
    Scanne le dossier pour trouver tous les PDFs et extraire leurs métadonnées
    """
    print(f"Scanning directory: {base_dir}")
    
    invoices_data = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                file_size = os.path.getsize(full_path)
                
                # Extraire le type de dataset depuis le chemin
                if 'valid information' in root:
                    dataset_type = 'Valid Information'
                elif 'colored IBAN' in root:
                    dataset_type = 'Colored IBAN'
                elif 'char-spaced' in root or 'charspace' in root:
                    dataset_type = 'Char-spaced IBAN'
                else:
                    dataset_type = 'Unknown'
                
                # Extraire l'ID de la facture
                invoice_id = ''
                id_match = re.search(r'invoice_(\d+)', file)
                if id_match:
                    invoice_id = id_match.group(1)
                
                # Extraire les paramètres spéciaux (charspace, color)
                charspace_match = re.search(r'charspace_(\d+)', file)
                color_match = re.search(r'color_B_(\d+)', file)
                
                charspace_value = charspace_match.group(1) if charspace_match else ''
                color_value = color_match.group(1) if color_match else ''
                
                invoice_info = {
                    'Invoice_ID': invoice_id,
                    'Filename': file,
                    'Dataset_Type': dataset_type,
                    'Charspace_Coefficient': charspace_value,
                    'Color_Blue_Value': color_value,
                    'File_Size_KB': f"{file_size/1024:.2f}",
                    'File_Size_Bytes': file_size,
                    'Local_Path': full_path,
                    'Relative_Path': os.path.relpath(full_path, base_dir),
                    'Status': 'Downloaded',
                    'Download_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Source': 'Mendeley Data - tnj49gpmtz/2',
                    'Theme': 'Electronic Invoices - Fraud Detection Dataset'
                }
                
                invoices_data.append(invoice_info)
    
    return pd.DataFrame(invoices_data)

def save_to_excel(df, output_file='mendeley_factures.xlsx'):
    """
    Sauvegarde les données dans un fichier Excel formaté
    """
    try:
        # Trier par Invoice_ID
        df['Invoice_ID_Int'] = pd.to_numeric(df['Invoice_ID'], errors='coerce')
        df = df.sort_values('Invoice_ID_Int').drop('Invoice_ID_Int', axis=1)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Invoices Data', index=False)
            
            worksheet = writer.sheets['Invoices Data']
            
            # Ajuster la largeur des colonnes
            for idx, col in enumerate(df.columns):
                max_length = min(
                    max(df[col].astype(str).apply(len).max(), len(col)), 100
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        print(f"\n✓ Fichier Excel créé: {output_file}")
        print(f"✓ {len(df)} factures listées")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur Excel: {e}")
        csv_file = output_file.replace('.xlsx', '.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSV créé à la place: {csv_file}")
        return False

def print_summary(df):
    """
    Affiche un résumé des données
    """
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES FACTURES PDF")
    print("=" * 70)
    print(f"Nombre total de factures: {len(df)}")
    
    print("\nRépartition par type de dataset:")
    print(df['Dataset_Type'].value_counts().to_string())
    
    print("\nTaille totale des fichiers:")
    total_size_mb = df['File_Size_Bytes'].sum() / (1024 * 1024)
    print(f"{total_size_mb:.2f} MB")
    
    print("\nExemples de factures (10 premières):")
    print(df[['Invoice_ID', 'Filename', 'Dataset_Type', 'File_Size_KB']].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)

def main():
    """Fonction principale"""
    print("=" * 70)
    print("GÉNÉRATION EXCEL - FACTURES PDF MENDELEY")
    print("=" * 70)
    
    base_dir = 'invoices_pdf'
    output_excel = 'mendeley_factures.xlsx'
    
    if not os.path.exists(base_dir):
        print(f"✗ Erreur: Le dossier {base_dir} n'existe pas!")
        print("Exécutez d'abord download_invoices.py pour télécharger les PDFs")
        return
    
    print(f"\n→ Dossier source: {base_dir}")
    print(f"→ Fichier Excel: {output_excel}")
    
    try:
        # Scanner les PDFs
        print(f"\nRecherche des fichiers PDF...")
        df = scan_pdf_directory(base_dir)
        
        if len(df) == 0:
            print("✗ Aucun fichier PDF trouvé!")
            return
        
        print(f"✓ {len(df)} fichiers PDF trouvés")
        
        # Afficher le résumé
        print_summary(df)
        
        # Sauvegarder dans Excel
        save_to_excel(df, output_excel)
        
        print(f"\n✓ TERMINÉ avec succès!")
        print(f"✓ Fichier Excel: {os.path.abspath(output_excel)}")
        
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
