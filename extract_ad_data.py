"""
Script d'extraction des données de Facebook Ad Library vers Excel
Theme: Produits artisanaux
"""

from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

def extract_ad_data(html_file_path):
    """
    Extrait les données des publicités depuis le fichier HTML
    
    Args:
        html_file_path: Chemin vers le fichier HTML Ad Library
        
    Returns:
        DataFrame pandas avec les données extraites
    """
    
    # Lire le fichier HTML
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Parser le HTML avec BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Liste pour stocker toutes les annonces
    ads_data = []
    
    # Définir le thème
    theme = "produits artisanaux"
    
    print("\nRecherche des annonces dans le fichier HTML...")
    
    # Nouvelle approche: chercher tous les blocs contenant "Started running on"
    # Car chaque annonce a une date unique
    date_elements = soup.find_all('span', string=re.compile(r'Started running on'))
    
    print(f"✓ {len(date_elements)} annonce(s) trouvée(s) avec date")
    print(f"✓ {len(date_elements)} annonce(s) trouvée(s) avec date")
    
    # Pour chaque élément de date, extraire toute l'annonce
    for idx, date_elem in enumerate(date_elements, 1):
        print(f"\n  → Extraction de l'annonce {idx}...")
        
        ad_info = {
            'Theme': theme,
            'Status': 'Active ads',
            'Library ID': '',
            'Starting Ads Date': '',
            'Nom de la société': '',
            'Description produit': '',
            'URL image ou video': '',
            'URL lien publicité': '',
            'Texte du lien': '',
            'Sous-titre du lien': ''
        }
        
        # Extraire la date depuis cet élément
        date_text = date_elem.get_text()
        match = re.search(r'Started running on (.+)', date_text)
        if match:
            ad_info['Starting Ads Date'] = match.group(1)
            print(f"    • Date: {ad_info['Starting Ads Date']}")
        
        # Remonter pour trouver le conteneur de l'annonce
        # On remonte jusqu'à trouver le parent contenant Library ID et TOUS les autres éléments
        container = date_elem
        for level in range(30):
            container = container.parent
            if container is None:
                break
            
            # Chercher le Library ID dans ce conteneur
            lib_id_elem = container.find('span', string=re.compile(r'Library ID:\s*\d+'))
            if lib_id_elem:
                lib_id_text = lib_id_elem.get_text()
                match = re.search(r'Library ID:\s*(\d+)', lib_id_text)
                if match:
                    ad_info['Library ID'] = match.group(1)
                    print(f"    • Library ID: {ad_info['Library ID']}")
                
                # Vérifier si ce conteneur a aussi des images (signe qu'on a le bon conteneur)
                has_image = container.find('img') is not None
                has_description = container.find('div', style=re.compile(r'white-space:\s*pre-wrap')) is not None
                
                # Si on a l'image ET la description, c'est le bon conteneur
                if has_image and has_description:
                    print(f"    ✓ Conteneur complet trouvé au niveau {level}")
                    break
                # Sinon on continue à remonter
        
        if container and ad_info['Library ID']:
            # Chercher le nom de la société dans l'attribut alt de l'image
            img_elem = container.find('img', alt=True)
            if img_elem and img_elem.get('alt'):
                alt_text = img_elem.get('alt').strip()
                # Vérifier que ce n'est pas juste un texte vide ou générique
                if alt_text and len(alt_text) > 2 and alt_text.lower() not in ['image', 'photo', 'ad']:
                    ad_info['Nom de la société'] = alt_text
                    print(f"    • Société (depuis alt): {alt_text}")
            
            # Si pas trouvé dans alt, chercher dans les liens avec la classe spécifique
            if not ad_info['Nom de la société']:
                # Chercher un lien avec le nom de la page Facebook
                page_links = container.find_all('a', href=re.compile(r'facebook\.com/[^/]+/?$'))
                for link in page_links:
                    span = link.find('span', class_=re.compile(r'x8t9es0.*x117nqv4.*xeuugli'))
                    if span:
                        company_name = span.get_text().strip()
                        if company_name and len(company_name) > 2 and 'Sponsored' not in company_name:
                            ad_info['Nom de la société'] = company_name
                            print(f"    • Société (depuis lien): {company_name}")
                            break
            
            # Chercher la description dans le div avec white-space: pre-wrap
            desc_div = container.find('div', style=re.compile(r'white-space:\s*pre-wrap'))
            if desc_div:
                # Extraire le texte en préservant les sauts de ligne
                desc_text = ''
                for content in desc_div.descendants:
                    if isinstance(content, str):
                        desc_text += content
                    elif hasattr(content, 'name') and content.name == 'br':
                        desc_text += '\n'
                
                desc_text = desc_text.strip()
                # Vérifier que c'est une vraie description
                if desc_text and len(desc_text) > 10 and 'Sponsored' not in desc_text:
                    ad_info['Description produit'] = desc_text
                    preview = desc_text[:100].replace('\n', ' ')
                    print(f"    • Description: {preview}..." if len(desc_text) > 100 else f"    • Description: {preview}")
            
            # Chercher l'image principale avec alt
            if not ad_info['URL image ou video']:
                img_with_alt = container.find('img', alt=True)
                if img_with_alt:
                    img_src = img_with_alt.get('src', '')
                    if img_src:
                        ad_info['URL image ou video'] = img_src
                        print(f"    • Image trouvée: {img_src.split('/')[-1][:50]}")
            
            # Chercher le lien principal de la page Facebook
            fb_link = container.find('a', href=re.compile(r'https?://www\.facebook\.com/[^/]+/?$'))
            if fb_link:
                link_url = fb_link.get('href', '')
                if link_url:
                    ad_info['URL lien publicité'] = link_url
                    print(f"    • Lien FB: {link_url}")
            
            # Si pas de lien FB direct, chercher n'importe quel lien http
            if not ad_info['URL lien publicité']:
                any_link = container.find('a', href=re.compile(r'https?://'))
                if any_link:
                    link_url = any_link.get('href', '')
                    if link_url and 'facebook.com/ads/library' not in link_url:
                        ad_info['URL lien publicité'] = link_url
                        print(f"    • Lien: {link_url[:60]}...")
        
        # Ajouter l'annonce à la liste
        ads_data.append(ad_info)
    
    # Si aucune annonce n'a été trouvée avec Library ID, essayer une méthode alternative
    if len(ads_data) == 0:
        print("\nAucune annonce trouvée avec Library ID. Recherche alternative...")
        
        # Chercher toutes les images qui pourraient être des annonces
        all_images = soup.find_all('img', alt='')
        
        for img in all_images:
            # Remonter pour trouver le conteneur potentiel
            container = img
            for _ in range(10):
                container = container.parent
                if container is None:
                    break
            
            if container:
                # Vérifier s'il y a du contenu texte substantiel
                text_content = container.get_text()
                if len(text_content) > 100:  # Au moins 100 caractères
                    ad_info = {
                        'Theme': theme,
                        'Status': 'Active ads',
                        'Library ID': '',
                        'Starting Ads Date': '',
                        'Nom de la société': '',
                        'Description produit': text_content[:500],  # Limiter à 500 caractères
                        'URL image ou video': img.get('src', '')
                    }
                    ads_data.append(ad_info)
    
    print(f"\n✓ Total: {len(ads_data)} annonce(s) extraite(s)")
    
    # Créer un DataFrame
    df = pd.DataFrame(ads_data)
    
    return df


def save_to_excel(df, output_file='facebook_ads_data.xlsx'):
    """
    Sauvegarde le DataFrame dans un fichier Excel
    
    Args:
        df: DataFrame pandas contenant les données
        output_file: Nom du fichier Excel de sortie
    """
    try:
        # Créer un writer Excel avec xlsxwriter engine
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Ads Data', index=False)
            
            # Obtenir le worksheet pour formater
            workbook = writer.book
            worksheet = writer.sheets['Ads Data']
            
            # Auto-ajuster la largeur des colonnes
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                # Limiter la largeur maximale
                max_length = min(max_length, 100)
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        print(f"\n✓ Fichier Excel créé avec succès: {output_file}")
        print(f"✓ Nombre d'annonces extraites: {len(df)}")
        
    except Exception as e:
        print(f"\n✗ Erreur lors de la création du fichier Excel: {e}")
        print("Tentative de sauvegarde en CSV...")
        
        # Alternative: sauvegarder en CSV si Excel échoue
        csv_file = output_file.replace('.xlsx', '.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"✓ Fichier CSV créé: {csv_file}")


def main():
    """Fonction principale"""
    print("=" * 60)
    print("Extraction des données Facebook Ad Library")
    print("Theme: Produits artisanaux")
    print("=" * 60)
    
    # Chemin du fichier HTML
    html_file = r'c:\Users\MSI\Desktop\Scraping_adlibrary\Ad Library.html'
    
    # Fichier de sortie Excel
    output_excel = r'c:\Users\MSI\Desktop\Scraping_adlibrary\facebook_ads_artisanaux.xlsx'
    
    print(f"\n→ Lecture du fichier: {html_file}")
    
    try:
        # Extraire les données
        df = extract_ad_data(html_file)
        
        # Afficher chaque annonce en détail avec une boucle foreach
        if len(df) > 0:
            print("\n" + "=" * 80)
            print("DÉTAIL COMPLET DE CHAQUE ANNONCE")
            print("=" * 80)
            
            for idx, row in df.iterrows():
                print(f"\n{'█' * 80}")
                print(f"ANNONCE #{idx + 1}")
                print(f"{'█' * 80}")
                
                for column, value in row.items():
                    # Formater l'affichage selon le type de données
                    if pd.isna(value) or value == '':
                        display_value = "[Non disponible]"
                    else:
                        display_value = str(value)
                        # Limiter l'affichage pour les URLs et textes longs
                        if len(display_value) > 100 and column in ['URL image ou video', 'URL lien publicité', 'Description produit']:
                            display_value = display_value[:100] + "..."
                    
                    print(f"  ├─ {column:30s}: {display_value}")
                
                print(f"{'─' * 80}")
        else:
            print("\n✗ Aucune annonce trouvée dans le fichier HTML!")
        
        # Afficher un aperçu du tableau
        print("\n" + "=" * 60)
        print("RÉSUMÉ DES DONNÉES EXTRAITES:")
        print("=" * 60)
        print(df.to_string())
        
        # Sauvegarder dans Excel
        print("\n" + "=" * 60)
        save_to_excel(df, output_excel)
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\n✗ Erreur: Le fichier '{html_file}' n'existe pas!")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
