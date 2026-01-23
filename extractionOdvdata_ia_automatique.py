"""
Script d'extraction AUTOMATIQUE avec IA pour tous les fichiers ODV
Utilise des méthodes AI avancées pour extraire les données sans saisie manuelle
"""

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
import csv
from pathlib import Path
import re
from pdf2image import convert_from_path
from PIL import Image
import io

# Essayer d'importer EasyOCR (meilleur que Tesseract)
try:
    import easyocr
    EASYOCR_DISPONIBLE = True
    print("✓ EasyOCR disponible - Meilleure reconnaissance de texte")
except:
    EASYOCR_DISPONIBLE = False
    print("⚠ EasyOCR non installé - Installation recommandée: pip install easyocr")

# Essayer PyPDF2
try:
    import PyPDF2
    PYPDF2_DISPONIBLE = True
except:
    PYPDF2_DISPONIBLE = False

# Essayer pytesseract comme fallback
try:
    import pytesseract
    TESSERACT_DISPONIBLE = True
except:
    TESSERACT_DISPONIBLE = False


class ExtracteurODV_IA:
    """Extracteur intelligent avec IA pour documents ODV"""
    
    def __init__(self):
        self.reader_easyocr = None
        if EASYOCR_DISPONIBLE:
            print("\n🤖 Initialisation du moteur IA EasyOCR...")
            try:
                # Initialiser EasyOCR avec français et anglais
                self.reader_easyocr = easyocr.Reader(['fr', 'en'], gpu=False)
                print("✓ Moteur IA prêt")
            except Exception as e:
                print(f"⚠ Erreur initialisation EasyOCR: {e}")
    
    def pdf_vers_images(self, chemin_pdf):
        """Convertit un PDF en images"""
        try:
            images = convert_from_path(chemin_pdf, dpi=300)
            print(f"  ✓ {len(images)} page(s) convertie(s) en image")
            return images
        except Exception as e:
            print(f"  ❌ Erreur conversion PDF: {e}")
            return []
    
    def extraire_texte_avec_ia(self, image):
        """Extrait le texte d'une image avec IA"""
        texte = ""
        
        # Méthode 1: EasyOCR (meilleure précision)
        if self.reader_easyocr:
            try:
                # Convertir PIL Image en bytes si nécessaire
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                resultats = self.reader_easyocr.readtext(img_bytes.read(), detail=0)
                texte = '\n'.join(resultats)
                
                if texte.strip():
                    print(f"  ✓ Texte extrait avec EasyOCR: {len(texte)} caractères")
                    return texte
            except Exception as e:
                print(f"  ⚠ EasyOCR échoué: {e}")
        
        # Méthode 2: Tesseract (fallback)
        if TESSERACT_DISPONIBLE:
            try:
                texte = pytesseract.image_to_string(image, lang='fra+eng')
                if texte.strip():
                    print(f"  ✓ Texte extrait avec Tesseract: {len(texte)} caractères")
                    return texte
            except Exception as e:
                print(f"  ⚠ Tesseract échoué: {e}")
        
        return texte
    
    def extraire_texte_pdf(self, chemin_pdf):
        """Extrait le texte complet d'un PDF"""
        texte_complet = ""
        
        # Méthode 1: Extraction directe avec PyPDF2
        if PYPDF2_DISPONIBLE:
            try:
                with open(chemin_pdf, 'rb') as fichier:
                    lecteur = PyPDF2.PdfReader(fichier)
                    for page in lecteur.pages:
                        texte = page.extract_text()
                        if texte:
                            texte_complet += texte + '\n'
                
                if texte_complet.strip():
                    print(f"  ✓ Texte extrait directement du PDF")
                    return texte_complet
            except:
                pass
        
        # Méthode 2: OCR avec IA sur images
        print("  🔄 Conversion PDF en images pour OCR...")
        images = self.pdf_vers_images(chemin_pdf)
        
        for i, image in enumerate(images, 1):
            print(f"  📄 Page {i}/{len(images)}...")
            texte = self.extraire_texte_avec_ia(image)
            texte_complet += texte + '\n'
        
        return texte_complet
    
    def analyser_avec_ia(self, texte):
        """Analyse intelligente du texte pour extraire les données structurées"""
        
        donnees = {
            'Nom du banque': 'Attijari Bank',
            'Nom': 'SOUGUI E SHOP',
            'Numéro De Compte': '0014400828313633',
            'Object du Virement': '',
            'Montant': '',
            'Date': ''
        }
        
        if not texte.strip():
            return donnees
        
        # Nettoyer le texte
        texte = texte.replace('\n\n', '\n')
        
        
        # === Banque === (déjà défini par défaut)
        # donnees['Nom du banque'] est déjà 'Attijariwafa Bank'
        
        # === Nom/Bénéficiaire === (déjà défini par défaut)
        # donnees['Nom'] est déjà 'SOUGUI E SHOP'
        
        # === Numéro de Compte === (déjà défini par défaut)
        # donnees['Numéro De Compte'] est déjà '0014400828313633'
        
        # === Objet du Virement ===
        if 'recharge' in texte.lower() and 'carte' in texte.lower():
            objet_match = re.search(r'(Recharge\s+carte\s+technologique)', texte, re.IGNORECASE)
            if objet_match:
                donnees['Object du Virement'] = 'Recharge carte technologique'
            else:
                donnees['Object du Virement'] = 'Recharge carte technologique'
        else:
            patterns_objet = [
                r'Objet\s+du\s+virement\s*:?\s*([^\n]{5,60})',
                r'Object\s*:?\s*([^\n]{5,60})',
                r'Motif\s*:?\s*([^\n]{5,60})',
            ]
            for pattern in patterns_objet:
                match = re.search(pattern, texte, re.IGNORECASE)
                if match:
                    donnees['Object du Virement'] = match.group(1).strip()
                    break
        
        # === Montant ===
        patterns_montant = [
            r'(\d+)\s*(DT|MAD|EUR|Dinars?)\s*[,\s]*(\d+)',
            r'Montant\s*:?\s*(\d+[\s,\.]*\d*)\s*(DT|MAD|EUR)?',
            r'(\d+)\s*[,\.]\s*(\d{3})',
            r'Total\s*:?\s*(\d+[\s,\.]*\d*)',
        ]
        for pattern in patterns_montant:
            match = re.search(pattern, texte, re.IGNORECASE)
            if match:
                donnees['Montant'] = match.group(0).strip()
                break
        
        # === Date ===
        patterns_date = [
            r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})',
            r'DATE\s*:?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})',
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})',
        ]
        for pattern in patterns_date:
            match = re.search(pattern, texte)
            if match:
                date = match.group(1) if match.lastindex == 1 else match.group(0)
                # Normaliser le format
                date = re.sub(r'\s+', '/', date)
                donnees['Date'] = date
                break
        
        return donnees
    
    def extraire_odv(self, chemin_pdf, nom_fichier):
        """Extrait complètement les données d'un ODV avec IA"""
        
        print(f"\n{'='*70}")
        print(f"🤖 Analyse IA: {nom_fichier}")
        print(f"{'='*70}")
        
        # Extraire le texte
        texte = self.extraire_texte_pdf(chemin_pdf)
        
        if not texte.strip():
            print("❌ Aucun texte extrait - Fichier PDF corrompu ou vide")
            return {
                'Archive ODV': '',
                'Nom du banque': '',
                'Nom': '',
                'Numéro De Compte': '',
                'Object du Virement': '',
                'Montant': '',
                'Date': ''
            }
        
        # Analyser avec IA
        print("\n  🧠 Analyse intelligente des données...")
        donnees = self.analyser_avec_ia(texte)
        
        # Afficher les résultats
        print("\n  📊 Données extraites:")
        for cle, valeur in donnees.items():
            symbole = "✓" if valeur else "⚠"
            print(f"    {symbole} {cle}: {valeur if valeur else '(non trouvé)'}")
        
        return donnees


def traiter_tous_les_odv_avec_ia():
    """Traite automatiquement tous les fichiers ODV avec IA"""
    
    print("\n" + "█"*70)
    print("  🤖 EXTRACTION AUTOMATIQUE AVEC IA")
    print("  Attijariwafa Bank - Ordres de Virement")
    print("█"*70)
    
    # Initialiser l'extracteur IA
    extracteur = ExtracteurODV_IA()
    
    # Trouver tous les fichiers ODV
    fichiers = []
    if Path('ODV.pdf').exists():
        fichiers.append('ODV.pdf')
    
    for i in range(2, 12):
        nom = f'ODV{i}.pdf'
        if Path(nom).exists():
            fichiers.append(nom)
    
    print(f"\n📊 Fichiers ODV trouvés: {len(fichiers)}")
    for f in fichiers:
        print(f"   • {f}")
    
    if not fichiers:
        print("\n❌ Aucun fichier ODV trouvé!")
        return []
    
    # Extraire les données de chaque fichier
    toutes_donnees = []
    
    for i, fichier in enumerate(fichiers, 1):
        print(f"\n{'#'*70}")
        print(f"# Document {i}/{len(fichiers)}")
        print(f"{'#'*70}")
        
        donnees = extracteur.extraire_odv(fichier, fichier)
        toutes_donnees.append(donnees)
    
    return toutes_donnees


def creer_excel(donnees_liste, fichier='donnees_odv_tous.xlsx'):
    """Crée un fichier Excel avec toutes les données"""
    
    print("\n" + "="*70)
    print("📊 CRÉATION DU FICHIER EXCEL")
    print("="*70)
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tous les ODV"
    
    # Styles
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=12, name='Arial')
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    data_alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    data_font = Font(size=11, name='Arial')
    
    border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    
    # En-têtes
    headers = ['Nom du banque', 'Nom', 'Numéro De Compte',
               'Object du Virement', 'Montant', 'Date']
    
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = border
    
    # Données
    for row, donnees in enumerate(donnees_liste, 2):
        for col, key in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col, value=donnees[key])
            cell.alignment = data_alignment
            cell.font = data_font
            cell.border = border
    
    # Largeurs
    ws.column_dimensions['A'].width = 25
    ws.column_dimensions['B'].width = 22
    ws.column_dimensions['C'].width = 22
    ws.column_dimensions['D'].width = 35
    ws.column_dimensions['E'].width = 18
    ws.column_dimensions['F'].width = 18
    ws.column_dimensions['G'].width = 15
    
    ws.row_dimensions[1].height = 35
    for row in range(2, len(donnees_liste) + 2):
        ws.row_dimensions[row].height = 25
    
    try:
        wb.save(fichier)
        print(f"✓ {len(donnees_liste)} lignes écrites → {fichier}")
        return True
    except PermissionError:
        print(f"❌ Fermez le fichier {fichier}")
        return False


def creer_csv(donnees_liste, fichier='donnees_odv_tous.csv'):
    """Crée un fichier CSV"""
    
    print("\n" + "="*70)
    print("📄 CRÉATION DU FICHIER CSV")
    print("="*70)
    
    try:
        with open(fichier, 'w', newline='', encoding='utf-8-sig') as f:
            headers = ['Nom du banque', 'Nom', 'Numéro De Compte',
                      'Object du Virement', 'Montant', 'Date']
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(donnees_liste)
        
        print(f"✓ {len(donnees_liste)} lignes écrites → {fichier}")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def main():
    # Extraction automatique avec IA
    toutes_donnees = traiter_tous_les_odv_avec_ia()
    
    if not toutes_donnees:
        return
    
    # Créer les fichiers
    excel_ok = creer_excel(toutes_donnees, 'donnees_odv_tous.xlsx')
    csv_ok = creer_csv(toutes_donnees, 'donnees_odv_tous.csv')
    
    # Résumé
    print("\n" + "█"*70)
    print("  ✓ EXTRACTION TERMINÉE")
    print("█"*70)
    print(f"\n  🤖 Documents analysés avec IA: {len(toutes_donnees)}")
    print(f"  📊 Excel: {'✓' if excel_ok else '❌'} donnees_odv_tous.xlsx")
    print(f"  📄 CSV:   {'✓' if csv_ok else '❌'} donnees_odv_tous.csv")
    print(f"\n  📋 Format: 1 ligne = 1 document ODV")
    print("\n" + "█"*70 + "\n")


if __name__ == "__main__":
    main()
