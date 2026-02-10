""" 
Flask Web Application for Exploratory Data Analysis
Application web professionnelle pour l'analyse exploratoire de données (EDA)

Fonctionnalités:
- Upload et analyse de fichiers CSV/Excel
- Visualisations interactives (histogrammes, boxplots, corrélations)
- Analyse PCA (Analyse en Composantes Principales)
- Génération de rapports PDF et HTML
- Support multi-feuilles pour fichiers Excel
- Détection automatique des outliers et valeurs manquantes
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import io
import base64
import os
import json
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime

# ============================================================================
# CONFIGURATION DE L'APPLICATION FLASK
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Dossier pour stocker les fichiers uploadés
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Taille max: 16MB
app.secret_key = 'eda_secret_key_2026'  # Clé secrète pour les sessions

# S'assurer que le dossier uploads existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration des styles de graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# VARIABLES GLOBALES POUR LA GESTION DES DONNÉES
# ============================================================================

current_analyzer = None  # Instance actuelle de DataAnalyzer
current_filename = None  # Nom du fichier actuellement analysé
all_sheets = {}  # Dictionnaire: {nom_feuille: DataFrame} pour fichiers Excel multi-feuilles
current_sheet_name = None  # Nom de la feuille actuellement sélectionnée

class DataAnalyzer:
    """
    Classe d'analyse de données pour Flask
    Version améliorée avec support pour application web
    
    Responsabilités:
    - Nettoyage et préparation des données
    - Calcul des statistiques descriptives
    - Détection des valeurs manquantes et outliers
    - Génération de corrélations (Pearson, Cramér's V, η²)
    - Analyse PCA complète
    - Génération de recommandations
    """
    
    def __init__(self, df):
        """
        Initialise l'analyseur avec un DataFrame
        
        Args:
            df (DataFrame): DataFrame pandas à analyser
        """
        self.df = df.copy()  # Copie pour ne pas modifier l'original
        self.numeric_cols = []  # Liste des colonnes numériques
        self.categorical_cols = []  # Liste des colonnes catégorielles
        self.recommendations = []  # Recommandations générées
        self.clean_data()  # Nettoyage automatique lors de l'initialisation
        
    def clean_data(self):
        """
        Nettoie et prépare les données pour l'analyse web
        
        Opérations effectuées:
        1. Conversion des noms de colonnes en strings
        2. Identification et conversion des colonnes numériques
        3. Conversion des dates
        4. Nettoyage des colonnes textuelles (suppression espaces, guillemets, valeurs vides)
        5. Classification automatique: numériques vs catégorielles
        """
        # Convertir tous les noms de colonnes en strings pour éviter AttributeError
        self.df.columns = [str(col) for col in self.df.columns]
        
        # Patterns pour identifier les colonnes financères/numériques courantes
        numeric_patterns = ['pu ht', 'quantite', 'prix total ht', 'montant total ht', 
                          'total ht', 'total ttc', 'timbre']
        
        # Convertir les colonnes numériques potentielles
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in numeric_patterns):
                try:
                    # Format français: remplacer virgule par point, supprimer guillemets
                    self.df[col] = self.df[col].astype(str).str.replace('"', '').str.replace(',', '.')
                    # Convertir en numérique (errors='coerce' = NaN pour valeurs invalides)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass  # Ignorer en cas d'échec
        
        # Conversion des colonnes de dates
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            except:
                pass
        
        # Nettoyage des colonnes textuelles avec préservation des NaN
        for col in self.df.select_dtypes(include=['object']).columns:
            # Remplacer les différentes représentations de valeurs vides par NaN
            self.df[col] = self.df[col].replace(['', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
            # Nettoyer uniquement les valeurs non-nulles
            self.df[col] = self.df[col].apply(lambda x: x.strip().replace('"', '') if pd.notna(x) and isinstance(x, str) else x)
            # Remplacer les chaînes vides (après strip) par NaN
            self.df[col] = self.df[col].replace('', np.nan)
        
        # Détection automatique des types de colonnes
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove date column from categorical if it exists
        if 'date' in self.categorical_cols:
            self.categorical_cols.remove('date')
    
    def get_overview_data(self):
        """Get dataset overview"""
        return {
            'total_rows': int(len(self.df)),
            'total_columns': int(len(self.df.columns)),
            'numeric_features': int(len(self.numeric_cols)),
            'categorical_features': int(len(self.categorical_cols)),
            'head': self.df.head(10).to_html(classes='table table-striped', index=False)
        }
    
    def get_missing_values(self):
        """Analyze missing values for all columns (numeric and categorical)"""
        missing_counts = []
        
        for col in self.df.columns:
            # Count NaN/None values
            null_count = self.df[col].isnull().sum()
            
            # For object columns, also count empty strings
            if self.df[col].dtype == 'object':
                empty_string_count = (self.df[col] == '').sum()
                null_count += empty_string_count
            
            missing_counts.append(null_count)
        
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': missing_counts,
            'Missing %': [(count / len(self.df) * 100) for count in missing_counts]
        })
        missing['Missing %'] = missing['Missing %'].round(2)
        
        return missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    def get_statistics(self):
        """Get descriptive statistics"""
        if len(self.numeric_cols) > 0:
            stats_df = self.df[self.numeric_cols].describe().T
            stats_df['median'] = self.df[self.numeric_cols].median()
            stats_df['variance'] = self.df[self.numeric_cols].var()
            stats_df['skewness'] = self.df[self.numeric_cols].skew()
            stats_df['kurtosis'] = self.df[self.numeric_cols].kurtosis()
            return stats_df
        return None
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        outliers_info = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            if len(outliers) > 0:
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(self.df)) * 100,
                    'lower_bound': lower,
                    'upper_bound': upper
                }
        return outliers_info
    
    def perform_pca(self):
        """Perform enhanced PCA analysis with comprehensive visualizations"""
        if len(self.numeric_cols) < 2:
            return None
        
        df_numeric = self.df[self.numeric_cols].dropna()
        if len(df_numeric) < 3:
            return None
        
        # Limiter le nombre de lignes pour accélérer le calcul
        max_samples = 5000
        if len(df_numeric) > max_samples:
            df_numeric = df_numeric.sample(n=max_samples, random_state=42)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        n_components = min(len(self.numeric_cols), len(df_numeric))
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pca_cols)
        
        # Calculate loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        return {
            'pca': pca,
            'pca_df': pca_df,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'n_components': n_components,
            'feature_names': self.numeric_cols,
            'eigenvalues': pca.explained_variance_.tolist(),
            'loadings': loadings.tolist(),
            'kaiser_components': int(sum(pca.explained_variance_ > 1))
        }
    
    def generate_recommendations(self):
        """Generate actionable recommendations (simplified, no remove recommendations)"""
        recommendations = []
        
        missing = self.get_missing_values()
        for idx, row in missing.iterrows():
            if row['Missing %'] > 5:
                recommendations.append({
                    'category': 'Data Imputation',
                    'priority': 'High' if row['Missing %'] > 30 else 'Medium',
                    'action': f"Handle missing values in '{row['Column']}'",
                    'reason': f"{row['Missing %']:.1f}% missing"
                })
        
        outliers = self.detect_outliers()
        for col, info in outliers.items():
            if info['percentage'] > 5:
                recommendations.append({
                    'category': 'Outlier Analysis',
                    'priority': 'Medium',
                    'action': f"Analyze outliers in '{col}'",
                    'reason': f"{info['percentage']:.1f}% outliers"
                })
        
        stats = self.get_statistics()
        if stats is not None:
            for col in self.numeric_cols:
                skew = stats.loc[col, 'skewness'] if col in stats.index else 0
                if abs(skew) > 2:
                    recommendations.append({
                        'category': 'Data Transformation',
                        'priority': 'Medium',
                        'action': f"Apply transformation to '{col}'",
                        'reason': f"High skewness"
                    })
        
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        recommendations.append({
                            'category': 'Multicollinearity',
                            'priority': 'Medium',
                            'action': f"Review '{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}'",
                            'reason': f"High correlation ({corr_matrix.iloc[i, j]:.2f})"
                        })
                        break
        
        if len(self.numeric_cols) > 1:
            scales = [self.df[col].std() for col in self.numeric_cols if self.df[col].std() > 0]
            if scales and max(scales) / min(scales) > 100:
                recommendations.append({
                    'category': 'Feature Scaling',
                    'priority': 'High',
                    'action': "Apply standardization/normalization",
                    'reason': "Features have different scales"
                })
        
        return recommendations
    
    def calculate_cramers_v_matrix(self):
        """Calculate Cramér's V matrix for categorical variables"""
        try:
            cats = self.categorical_cols[:10]
            n_cats = len(cats)
            
            if n_cats < 2:
                return None
            
            cramers_matrix = np.zeros((n_cats, n_cats))
            
            for i, cat1 in enumerate(cats):
                for j, cat2 in enumerate(cats):
                    if i == j:
                        cramers_matrix[i, j] = 1.0
                    elif i < j:
                        try:
                            contingency_table = pd.crosstab(self.df[cat1], self.df[cat2])
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            n = contingency_table.sum().sum()
                            min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                            if min_dim > 0:
                                cramers_v = np.sqrt(chi2 / (n * min_dim))
                                cramers_matrix[i, j] = cramers_v
                                cramers_matrix[j, i] = cramers_v
                        except:
                            cramers_matrix[i, j] = 0
                            cramers_matrix[j, i] = 0
            
            return pd.DataFrame(cramers_matrix, index=cats, columns=cats)
        except Exception as e:
            print(f"Error calculating Cramér's V: {e}")
            return None
    
    def calculate_correlation_ratio_matrix(self):
        """Calculate correlation ratio (η²) between quantitative and qualitative variables"""
        try:
            num_cols = self.numeric_cols[:10]
            cat_cols = self.categorical_cols[:10]
            
            if len(num_cols) == 0 or len(cat_cols) == 0:
                return None
            
            corr_ratio_matrix = np.zeros((len(num_cols), len(cat_cols)))
            
            for i, num_col in enumerate(num_cols):
                for j, cat_col in enumerate(cat_cols):
                    try:
                        categories = self.df[cat_col].dropna().unique()
                        
                        if len(categories) > 50:
                            continue
                        
                        groups = []
                        for cat in categories:
                            group_data = self.df[self.df[cat_col] == cat][num_col].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data.values)
                        
                        if len(groups) < 2:
                            corr_ratio_matrix[i, j] = 0
                            continue
                        
                        all_data = self.df[num_col].dropna()
                        overall_mean = all_data.mean()
                        ss_total = np.sum((all_data - overall_mean) ** 2)
                        
                        if ss_total == 0:
                            corr_ratio_matrix[i, j] = 0
                            continue
                        
                        ss_between = 0
                        for group in groups:
                            if len(group) > 0:
                                group_mean = np.mean(group)
                                ss_between += len(group) * (group_mean - overall_mean) ** 2
                        
                        eta_squared = ss_between / ss_total
                        corr_ratio_matrix[i, j] = min(eta_squared, 1.0)
                        
                    except Exception as e:
                        corr_ratio_matrix[i, j] = 0
            
            return pd.DataFrame(corr_ratio_matrix, index=num_cols, columns=cat_cols)
        except Exception as e:
            print(f"Error calculating correlation ratio: {e}")
            return None

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def fig_to_base64(fig):
    """
    Convertit une figure matplotlib en chaîne base64 pour affichage web
    
    Cette fonction permet d'intégrer des graphiques matplotlib directement
    dans les pages HTML via des balises <img src="data:image/png;base64,...">
    
    Args:
        fig (matplotlib.figure.Figure): Figure matplotlib à convertir
        
    Returns:
        str: Image encodée en base64 (format UTF-8)
    """
    # Créer un buffer en mémoire pour stocker l'image
    buf = io.BytesIO()
    # Sauvegarder la figure dans le buffer (format PNG, haute qualité)
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    # Retourner au début du buffer pour lecture
    buf.seek(0)
    # Encoder en base64 et décoder en UTF-8 pour usage web
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # Fermer la figure pour libérer la mémoire
    plt.close(fig)
    return img_base64

def load_data(file):
    """
    Charge les données depuis un fichier uploadé avec support multi-format
    
    Formats supportés:
    - CSV/TXT/TSV/DAT: Teste automatiquement différents séparateurs (;, ,, \t, |)
                        et encodages (UTF-8, Latin-1, ISO-8859-1, CP1252)
    - Excel: Support multi-feuilles (.xlsx, .xls, .xlsm, .xlsb)
             Teste différents engines (openpyxl, xlrd)
    
    Args:
        file (FileStorage): Fichier uploadé via Flask request.files
        
    Returns:
        DataFrame: Pour fichiers à feuille unique (CSV ou Excel 1 sheet)
        dict: Pour fichiers Excel multi-feuilles {sheet_name: DataFrame}
        None: Si le chargement échoue
    """
    try:
        filename_lower = file.filename.lower()
        
        # ===== FICHIERS CSV/TXT/TSV/DAT =====
        if filename_lower.endswith(('.csv', '.txt', '.tsv', '.dat')):
            # Liste des encodages à tester (du plus courant au plus rare)
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            # Liste des séparateurs à tester
            separators = [';', ',', '\t', '|']
            
            # Essayer toutes les combinaisons encodage x séparateur
            for encoding in encodings:
                for sep in separators:
                    try:
                        file.seek(0)  # Retour au début du fichier
                        df = pd.read_csv(file, sep=sep, encoding=encoding, on_bad_lines='skip')
                        # Vérifier que le DataFrame est valide (>1 colonne et >0 lignes)
                        if len(df.columns) > 1 and len(df) > 0:
                            print(f"Successfully loaded with encoding={encoding}, separator={sep}")
                            return df
                    except Exception as e:
                        continue  # Essayer la combinaison suivante
            
            # Fallback: Laisser pandas détecter automatiquement
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding='utf-8')
                return df
            except:
                file.seek(0)
                df = pd.read_csv(file)
                return df
        
        # ===== FICHIERS EXCEL =====
        elif filename_lower.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            file.seek(0)
            # Essayer différents engines Excel (compatibilité maximale)
            engines = ['openpyxl', 'xlrd', None]
            for engine in engines:
                try:
                    file.seek(0)
                    # Lire TOUTES les feuilles du fichier Excel
                    # sheet_name=None retourne dict {sheet_name: DataFrame}
                    excel_data = pd.read_excel(file, engine=engine, sheet_name=None)
                    
                    if excel_data and len(excel_data) > 0:
                        print(f"Successfully loaded Excel file with engine={engine}")
                        print(f"Found {len(excel_data)} sheet(s): {list(excel_data.keys())}")
                        
                        # Filtrer les feuilles vides
                        filtered_sheets = {name: df for name, df in excel_data.items() if len(df) > 0}
                        
                        if len(filtered_sheets) == 1:
                            # Une seule feuille: retourner DataFrame directement
                            return list(filtered_sheets.values())[0]
                        elif len(filtered_sheets) > 1:
                            # Plusieurs feuilles: retourner dictionnaire
                            return filtered_sheets
                        
                except Exception as e:
                    print(f"Failed with engine {engine}: {e}")
                    continue  # Essayer le prochain engine
            return None
        
        else:
            # Format de fichier non supporté
            print(f"Unsupported file format: {file.filename}")
            return None
            
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# ROUTES FLASK - PAGES PRINCIPALES
# ============================================================================

@app.route('/')
def index():
    """
    Page d'accueil de l'application
    
    Fonctionnalités:
    - Affiche l'interface d'upload de fichiers
    - Charge automatiquement 'devis.csv' s'il existe
    - Affiche les informations du dataset si chargé
    
    Returns:
        Template HTML: index.html avec statut des données
    """
    global current_analyzer, current_filename
    
    # Tentative de chargement automatique du fichier par défaut
    if current_analyzer is None:
        try:
            df = pd.read_csv('devis.csv', sep=';', encoding='utf-8')
            current_analyzer = DataAnalyzer(df)
            current_filename = 'devis.csv'
        except:
            pass  # Ignorer si le fichier n'existe pas
    
    has_data = current_analyzer is not None
    return render_template('index.html', has_data=has_data, filename=current_filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Gère l'upload des fichiers de données
    
    Formats supportés:
    - CSV (.csv, .txt, .tsv, .dat)
    - Excel (.xlsx, .xls, .xlsm, .xlsb)
    
    Fonctionnalités:
    - Validation du type de fichier
    - Support multi-feuilles pour Excel
    - Sélection automatique de la première feuille
    - Création automatique d'une instance DataAnalyzer
    
    Returns:
        JSON: Statut de succès/échec avec nom du fichier et liste des feuilles
    """
    global current_analyzer, current_filename, all_sheets, current_sheet_name
    
    try:
        # Vérifier qu'un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Valider l'extension du fichier
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.xlsm', '.xlsb', '.txt', '.tsv', '.dat'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid file type. Allowed: CSV, Excel (xlsx, xls, xlsm, xlsb), TXT, TSV, DAT'}), 400
        
        # Charger les données (fonction gère CSV et Excel)
        data = load_data(file)
        if data is None:
            return jsonify({'error': 'Failed to load file. Please ensure it is a valid CSV or Excel file with data.'}), 400
        
        # Sécuriser le nom du fichier (prévenir injections)
        current_filename = secure_filename(file.filename)
        
        # Vérifier si plusieurs feuilles (Excel) ou une seule (CSV)
        if isinstance(data, dict):
            # Cas Excel multi-feuilles: data = {sheet_name: DataFrame}
            all_sheets = data
            sheet_names = list(all_sheets.keys())
            current_sheet_name = sheet_names[0]  # Sélectionner première feuille par défaut
            current_analyzer = DataAnalyzer(all_sheets[current_sheet_name])
            
            return jsonify({
                'success': True, 
                'filename': current_filename,
                'sheets': sheet_names,
                'current_sheet': current_sheet_name
            })
        else:
            # Cas fichier simple (CSV): data = DataFrame
            all_sheets = {}
            current_sheet_name = None
            if len(data) == 0:
                return jsonify({'error': 'File contains no data'}), 400
            current_analyzer = DataAnalyzer(data)
            
            return jsonify({
                'success': True, 
                'filename': current_filename,
                'sheets': [],
                'current_sheet': None
            })
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/switch_sheet', methods=['POST'])
def switch_sheet():
    """Switch to a different sheet"""
    global current_analyzer, current_sheet_name, all_sheets
    
    try:
        data = request.get_json()
        sheet_name = data.get('sheet_name')
        
        if not all_sheets:
            return jsonify({'error': 'No multi-sheet file loaded'}), 400
        
        if sheet_name not in all_sheets:
            return jsonify({'error': f'Sheet "{sheet_name}" not found'}), 404
        
        current_sheet_name = sheet_name
        current_analyzer = DataAnalyzer(all_sheets[sheet_name])
        
        return jsonify({
            'success': True,
            'current_sheet': current_sheet_name,
            'rows': len(all_sheets[sheet_name]),
            'columns': len(all_sheets[sheet_name].columns)
        })
    except Exception as e:
        print(f"Switch sheet error: {str(e)}")
        return jsonify({'error': f'Failed to switch sheet: {str(e)}'}), 500

@app.route('/get_sheets')
def get_sheets():
    """Get list of available sheets"""
    global all_sheets, current_sheet_name
    
    if all_sheets:
        return jsonify({
            'sheets': list(all_sheets.keys()),
            'current_sheet': current_sheet_name
        })
    else:
        return jsonify({
            'sheets': [],
            'current_sheet': None
        })

@app.route('/visualizations')
def visualizations():
    """Visualizations page"""
    if current_analyzer is None:
        return render_template('error.html', message='No data loaded. Please upload a file first from the home page.')
    
    return render_template('visualizations.html', filename=current_filename)

@app.route('/code')
def code_page():
    """Code page"""
    try:
        with open('eda_analysis.py', 'r', encoding='utf-8') as f:
            code_content = f.read()
        return render_template('code.html', code=code_content)
    except Exception as e:
        print(f"Code page error: {str(e)}")
        return render_template('error.html', message='Could not load code file. Please ensure eda_analysis.py exists.')

@app.route('/recommendations')
def recommendations():
    """Recommendations page"""
    if current_analyzer is None:
        return render_template('error.html', message='No data loaded. Please upload a file first from the home page.')
    
    return render_template('recommendations.html', filename=current_filename)

@app.route('/reports')
def reports():
    """Reports page with detailed analysis"""
    if current_analyzer is None:
        return render_template('error.html', message='No data loaded. Please upload a file.')
    
    return render_template('reports.html', filename=current_filename)

@app.route('/api/overview')
def api_overview():
    """API endpoint for overview data"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    overview = current_analyzer.get_overview_data()
    return jsonify(overview)

@app.route('/api/missing_values')
def api_missing_values():
    """API endpoint for missing values"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    missing = current_analyzer.get_missing_values()
    
    if len(missing) > 0:
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        missing.plot(x='Column', y='Missing %', kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Missing Percentage (%)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        img = fig_to_base64(fig)
        
        return jsonify({
            'has_missing': True,
            'data': missing.to_dict('records'),
            'chart': img
        })
    else:
        return jsonify({'has_missing': False})

@app.route('/api/statistics')
def api_statistics():
    """API endpoint for statistics"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    stats = current_analyzer.get_statistics()
    if stats is not None:
        # Add interpretation HTML
        interpretation_html = '''
        <div class="interpretation-box mt-4">
            <strong>📊 Statistical Summary Interpretation:</strong><br><br>
            <strong>Key Metrics Explained:</strong>
            <ul class="mt-2">
                <li><strong>Mean vs Median:</strong> Large differences indicate skewed distributions or outliers</li>
                <li><strong>Std (Standard Deviation):</strong> Measures spread - higher values mean more variability</li>
                <li><strong>Skewness:</strong>
                    <ul>
                        <li>≈ 0: Symmetric distribution (normal-like)</li>
                        <li>> 0: Right-skewed (long tail on right, mean > median)</li>
                        <li>< 0: Left-skewed (long tail on left, mean < median)</li>
                        <li>|skewness| > 1: Highly skewed, consider transformations (log, sqrt, Box-Cox)</li>
                    </ul>
                </li>
                <li><strong>Kurtosis:</strong>
                    <ul>
                        <li>≈ 0: Normal-like tails</li>
                        <li>> 0: Heavy tails, more outliers than normal distribution</li>
                        <li>< 0: Light tails, fewer outliers</li>
                    </ul>
                </li>
            </ul>
            <br>
            <strong>What to look for:</strong>
            <ul class="mt-2 mb-0">
                <li>🔍 <strong>Scale differences:</strong> Features with vastly different ranges (e.g., 0-1 vs 0-10000) need normalization/standardization</li>
                <li>🔍 <strong>Zero variance:</strong> Check min = max (constant features should be removed)</li>
                <li>🔍 <strong>Outliers:</strong> Large difference between 75th percentile and max suggests outliers</li>
                <li>🔍 <strong>Distribution shape:</strong> High skewness/kurtosis may benefit from transformations before modeling</li>
            </ul>
        </div>
        '''
        
        return jsonify({
            'html': stats.to_html(classes='table table-striped', float_format='%.2f') + interpretation_html
        })
    return jsonify({'html': None})

@app.route('/api/distributions')
def api_distributions():
    """API endpoint for distribution charts - ALL numeric columns"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    distributions = []
    # Show ALL numeric columns (no limit)
    for col in current_analyzer.numeric_cols:
        try:
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(current_analyzer.df[col].dropna(), bins=30, edgecolor='black', color='skyblue')
            ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(alpha=0.3)
            hist_img = fig_to_base64(fig)
            
            # Statistics
            mean_val = current_analyzer.df[col].mean()
            median_val = current_analyzer.df[col].median()
            std_val = current_analyzer.df[col].std()
            skew_val = current_analyzer.df[col].skew()
            
            distributions.append({
                'column': col,
                'histogram': hist_img,
                'mean': f"{mean_val:.2f}",
                'median': f"{median_val:.2f}",
                'std': f"{std_val:.2f}",
                'skewness': f"{skew_val:.2f}"
            })
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            continue
    
    return jsonify(distributions)

@app.route('/api/boxplots')
def api_boxplots():
    """API endpoint for boxplots - ALL numeric columns"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    outliers_info = current_analyzer.detect_outliers()
    boxplots = []
    
    # Show ALL numeric columns (no limit)
    for col in current_analyzer.numeric_cols:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            current_analyzer.df.boxplot(column=col, ax=ax, patch_artist=True,
                                       boxprops=dict(facecolor='lightblue'),
                                       medianprops=dict(color='red', linewidth=2))
            ax.set_title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
            ax.set_ylabel(col, fontsize=12)
            ax.grid(alpha=0.3)
            img = fig_to_base64(fig)
            
            has_outliers = col in outliers_info
            outlier_data = outliers_info.get(col, {})
            
            boxplots.append({
                'column': col,
                'chart': img,
                'has_outliers': has_outliers,
                'outlier_count': outlier_data.get('count', 0),
                'outlier_percentage': f"{outlier_data.get('percentage', 0):.2f}",
                'lower_bound': f"{outlier_data.get('lower_bound', 0):.2f}",
                'upper_bound': f"{outlier_data.get('upper_bound', 0):.2f}"
            })
        except Exception as e:
            print(f"Error processing boxplot for column {col}: {e}")
            continue
    
    return jsonify(boxplots)

# ============================================================================
# ROUTES API - ENDPOINTS POUR LES VISUALISATIONS
# ============================================================================

@app.route('/api/correlation')
def api_correlation():
    """
    API endpoint pour la matrice de corrélation des variables quantitatives
    
    Utilise la corrélation de Pearson pour mesurer les relations linéaires
    entre variables numériques
    
    Returns:
        JSON: Heatmap de corrélation encodée en base64
    """
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if len(current_analyzer.numeric_cols) > 1:
        # Calculer la matrice de corrélation de Pearson
        corr_matrix = current_analyzer.df[current_analyzer.numeric_cols].corr()
        
        # Générer le heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Matrix - Quantitative Variables', fontsize=16, fontweight='bold')
        img = fig_to_base64(fig)
        
        return jsonify({'chart': img})
    
    return jsonify({'chart': None})

@app.route('/api/correlation_qualitative')
def api_correlation_qualitative():
    """
    API endpoint pour la matrice de corrélation des variables qualitatives
    
    Utilise Cramér's V (test Chi-carré) pour mesurer l'association entre
    variables catégorielles. Valeurs entre 0 (aucune association) et 1 (association parfaite)
    
    Returns:
        JSON: Heatmap Cramér's V encodé en base64
    """
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if len(current_analyzer.categorical_cols) > 1:
        # Calculer la matrice de Cramér's V
        cramers_matrix = current_analyzer.calculate_cramers_v_matrix()
        
        if cramers_matrix is not None:
            # Générer le heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cramers_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                       vmin=0, vmax=1, square=True, linewidths=1, ax=ax)
            ax.set_title("Correlation Matrix - Qualitative Variables (Cramér's V)", 
                       fontsize=16, fontweight='bold')
            img = fig_to_base64(fig)
            
            return jsonify({'chart': img})
    
    return jsonify({'chart': None})

@app.route('/api/correlation_mixed')
def api_correlation_mixed():
    """
    API endpoint pour la corrélation mixte (quantitatif vs qualitatif)
    
    Utilise le Rapport de Corrélation (η² / eta-squared) basé sur l'ANOVA
    pour mesurer la relation entre variables numériques et catégorielles
    
    Returns:
        JSON: Heatmap de rapport de corrélation encodé en base64
    """
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if len(current_analyzer.numeric_cols) > 0 and len(current_analyzer.categorical_cols) > 0:
        # Calculer la matrice de rapport de corrélation
        corr_ratio_matrix = current_analyzer.calculate_correlation_ratio_matrix()
        
        if corr_ratio_matrix is not None:
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr_ratio_matrix, annot=True, fmt='.2f', cmap='viridis',
                       vmin=0, vmax=1, square=True, linewidths=1, ax=ax)
            ax.set_title('Correlation Matrix - Quantitative vs Qualitative Variables (η²)', 
                       fontsize=16, fontweight='bold')
            ax.set_xlabel('Qualitative Variables', fontsize=12)
            ax.set_ylabel('Quantitative Variables', fontsize=12)
            img = fig_to_base64(fig)
            
            return jsonify({'chart': img})
    
    return jsonify({'chart': None})

@app.route('/api/categorical')
def api_categorical():
    """API endpoint for categorical charts - ALL categorical columns"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    categorical_charts = []
    
    # Show ALL categorical columns (no limit)
    for col in current_analyzer.categorical_cols:
        try:
            value_counts = current_analyzer.df[col].value_counts().head(10)
            
            if len(value_counts) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                value_counts.plot(kind='bar', color='teal', edgecolor='black', ax=ax)
                ax.set_title(f'Top 10 Values in {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                img = fig_to_base64(fig)
                
                categorical_charts.append({
                    'column': col,
                    'chart': img,
                    'unique_values': int(current_analyzer.df[col].nunique()),
                    'most_common': str(value_counts.index[0]),
                    'frequency': int(value_counts.values[0])
                })
        except Exception as e:
            print(f"Error processing categorical column {col}: {e}")
            continue
    
    return jsonify(categorical_charts)

@app.route('/api/pca')
def api_pca():
    """API endpoint for PCA analysis"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded', 'available': False}), 400
    
    try:
        print("Starting PCA analysis...")
        pca_results = current_analyzer.perform_pca()
        
        if pca_results is None:
            print("PCA results is None - not enough numeric columns or data")
            return jsonify({'available': False, 'message': 'Insufficient numeric data for PCA'})
        
        print(f"PCA calculated with {pca_results['n_components']} components")
        charts = {}
        
        # Scree Plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            components = range(1, pca_results['n_components'] + 1)
            ax.bar(components, pca_results['explained_variance'], color='steelblue', edgecolor='black', alpha=0.7)
            ax.plot(components, pca_results['explained_variance'], 'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Principal Component', fontsize=12)
            ax.set_ylabel('Explained Variance Ratio', fontsize=12)
            ax.set_title('Scree Plot - Variance Explained by Each Component', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            for i, v in enumerate(pca_results['explained_variance']):
                ax.text(i + 1, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold', fontsize=10)
            charts['scree'] = fig_to_base64(fig)
            print("Scree plot created")
        except Exception as e:
            print(f"Error creating scree plot: {e}")
            raise
        
        # Cumulative variance
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(components, pca_results['cumulative_variance'], 'go-', linewidth=2, markersize=8)
            ax.fill_between(components, pca_results['cumulative_variance'], alpha=0.3, color='green')
            ax.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% Threshold')
            ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95% Threshold')
            ax.set_xlabel('Number of Components', fontsize=12)
            ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
            ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            for i, v in enumerate(pca_results['cumulative_variance']):
                ax.text(i + 1, v + 0.02, f'{v:.1%}', ha='center', fontsize=9)
            charts['cumulative'] = fig_to_base64(fig)
            print("Cumulative variance plot created")
        except Exception as e:
            print(f"Error creating cumulative variance plot: {e}")
            raise
        
        # 2D PCA
        if pca_results['n_components'] >= 2:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                pca_df = pca_results['pca_df']
                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'],
                                   c=range(len(pca_df)), cmap='plasma',
                                   s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
                ax.set_xlabel(f"PC1 ({pca_results['explained_variance'][0]:.1%})", fontsize=12)
                ax.set_ylabel(f"PC2 ({pca_results['explained_variance'][1]:.1%})", fontsize=12)
                ax.set_title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Sample Index')
                charts['pca_2d'] = fig_to_base64(fig)
                print("2D PCA plot created")
            except Exception as e:
                print(f"Error creating 2D PCA plot: {e}")
        
        # Biplot
        if pca_results['n_components'] >= 2:
            try:
                fig, ax = plt.subplots(figsize=(12, 10))
                pca_df = pca_results['pca_df']
                loadings = np.array(pca_results['loadings'])
                
                # Plot observations
                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'],
                                   c=range(len(pca_df)), cmap='viridis',
                                   s=50, alpha=0.5, edgecolors='black', linewidth=0.5)
                
                # Plot variable vectors  
                scale_factor = 3
                feature_names = pca_results['feature_names']
                for i, var in enumerate(feature_names):
                    ax.arrow(0, 0,
                            loadings[i, 0] * scale_factor,
                            loadings[i, 1] * scale_factor,
                            head_width=0.1, head_length=0.1,
                            fc='red', ec='red', alpha=0.8, linewidth=2)
                    ax.text(loadings[i, 0] * scale_factor * 1.15,
                           loadings[i, 1] * scale_factor * 1.15,
                           var, fontsize=9, fontweight='bold',
                           ha='center', color='darkred')
                
                ax.set_xlabel(f"PC1 ({pca_results['explained_variance'][0]:.1%})", fontsize=12)
                ax.set_ylabel(f"PC2 ({pca_results['explained_variance'][1]:.1%})", fontsize=12)
                ax.set_title('PCA Biplot - Observations and Variables', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                plt.colorbar(scatter, ax=ax, label='Sample Index')
                charts['biplot'] = fig_to_base64(fig)
                print("Biplot created")
            except Exception as e:
                print(f"Error creating biplot: {e}")
        
        # Correlation Circle
        if pca_results['n_components'] >= 2:
            try:
                fig, ax = plt.subplots(figsize=(10, 10))
                loadings = np.array(pca_results['loadings'])
                
                # Draw circle
                circle = plt.Circle((0, 0), 1, color='navy', fill=False, linewidth=2)
                ax.add_patch(circle)
                
                # Plot correlations
                corr_with_pc = loadings[:, :2]
                feature_names = pca_results['feature_names']
                for i, var in enumerate(feature_names):
                    ax.arrow(0, 0, corr_with_pc[i, 0], corr_with_pc[i, 1],
                            head_width=0.05, head_length=0.05,
                            fc='blue', ec='blue', alpha=0.8, linewidth=2)
                    ax.text(corr_with_pc[i, 0] * 1.1, corr_with_pc[i, 1] * 1.1,
                           var, fontsize=9, fontweight='bold', ha='center')
                
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_xlabel(f"PC1 ({pca_results['explained_variance'][0]:.1%})", fontsize=12)
                ax.set_ylabel(f"PC2 ({pca_results['explained_variance'][1]:.1%})", fontsize=12)
                ax.set_title('Correlation Circle - Variable Contributions', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')
                charts['correlation_circle'] = fig_to_base64(fig)
                print("Correlation circle created")
            except Exception as e:
                print(f"Error creating correlation circle: {e}")
        
        # Variable Contributions
        try:
            # Show all components (up to 10 to avoid excessively large plots)
            num_pcs = min(10, pca_results['n_components'])
            
            # Adjust layout based on number of components
            if num_pcs <= 3:
                ncols = num_pcs
                nrows = 1
                figsize = (6*ncols, 6)
            elif num_pcs <= 6:
                ncols = 3
                nrows = 2
                figsize = (18, 12)
            else:
                ncols = 4
                nrows = (num_pcs + 3) // 4
                figsize = (24, 6*nrows)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            components = np.array(pca_results['components'])
            feature_names = pca_results['feature_names']
            
            for idx in range(num_pcs):
                contrib = (components[idx]**2) / (components[idx]**2).sum() * 100
                sorted_idx = np.argsort(contrib)[::-1]
                
                axes[idx].barh(np.array(feature_names)[sorted_idx],
                              contrib[sorted_idx], color='teal', edgecolor='black')
                axes[idx].set_xlabel('Contribution (%)', fontsize=11)
                axes[idx].set_title(f'PC{idx+1} Variable Contributions\n({pca_results["explained_variance"][idx]:.1%} variance)',
                                  fontsize=12, fontweight='bold')
                axes[idx].grid(alpha=0.3, axis='x')
                
                # Highlight top contributors
                for i, val in enumerate(contrib[sorted_idx][:3]):
                    axes[idx].text(val + 1, i, f'{val:.1f}%', va='center',
                                  fontweight='bold', fontsize=9)
            
            # Hide unused subplots
            for idx in range(num_pcs, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            charts['contributions'] = fig_to_base64(fig)
            print("Contributions plot created")
        except Exception as e:
            print(f"Error creating contributions plot: {e}")
        
        # Eigenvalues (Kaiser criterion)
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            eigenvalues = pca_results['eigenvalues']
            components_range = range(1, len(eigenvalues) + 1)
            ax.bar(components_range, eigenvalues, color='coral', edgecolor='black', alpha=0.7)
            ax.axhline(y=1, color='red', linestyle='--', linewidth=2,
                      label='Kaiser criterion (eigenvalue = 1)')
            ax.set_xlabel('Principal Component', fontsize=12)
            ax.set_ylabel('Eigenvalue', fontsize=12)
            ax.set_title('Eigenvalues by Component (Kaiser Criterion)', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            for i, v in enumerate(eigenvalues):
                ax.text(i + 1, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)
            charts['eigenvalues'] = fig_to_base64(fig)
            print("Eigenvalues plot created")
        except Exception as e:
            print(f"Error creating eigenvalues plot: {e}")
        
        # Loadings heatmap
        try:
            fig, ax = plt.subplots(figsize=(12, max(6, len(feature_names) * 0.4)))
            components_df = pd.DataFrame(
                np.array(pca_results['components']).T,
                columns=[f'PC{i+1}' for i in range(pca_results['n_components'])],
                index=feature_names
            )
            sns.heatmap(components_df, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, ax=ax, cbar_kws={'label': 'Loading Value'},
                       linewidths=0.5, linecolor='gray')
            ax.set_title('PCA Component Loadings (Eigenvectors)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Principal Components', fontsize=12)
            ax.set_ylabel('Original Features', fontsize=12)
            charts['loadings'] = fig_to_base64(fig)
            print("Loadings heatmap created")
        except Exception as e:
            print(f"Error creating loadings heatmap: {e}")
        
        print("PCA analysis completed successfully")
        return jsonify({
            'available': True,
            'n_components': pca_results['n_components'],
            'explained_variance': pca_results['explained_variance'],
            'cumulative_variance': pca_results['cumulative_variance'],
            'kaiser_components': pca_results['kaiser_components'],
            'eigenvalues': pca_results['eigenvalues'],
            'charts': charts
        })
    except Exception as e:
        print(f"Error in PCA endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'available': False, 
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint for recommendations"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    recommendations = current_analyzer.generate_recommendations()
    
    high = [r for r in recommendations if r['priority'] == 'High']
    medium = [r for r in recommendations if r['priority'] == 'Medium']
    low = [r for r in recommendations if r['priority'] == 'Low']
    
    return jsonify({
        'high': high,
        'medium': medium,
        'low': low,
        'total': len(recommendations)
    })

@app.route('/download/code')
def download_code():
    """Download Python code"""
    return send_file('eda_analysis.py', as_attachment=True, download_name='eda_analysis.py')

@app.route('/download/report-pdf')
def download_report_pdf():
    """Generate and download comprehensive PDF report"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=50, leftMargin=50,
                              topMargin=50, bottomMargin=50)
        
        # Container for PDF elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1976d2'),
            spaceAfter=12,
            spaceBefore=12
        )
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#424242'),
            spaceAfter=8
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        code_style = ParagraphStyle(
            'Code',
            parent=styles['Code'],
            fontSize=8,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=10,
            backColor=colors.HexColor('#f5f5f5'),
            borderColor=colors.HexColor('#e0e0e0'),
            borderWidth=1,
            borderPadding=10
        )
        
        # Title Page
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph("Exploratory Data Analysis", title_style))
        elements.append(Paragraph("Comprehensive Report", title_style))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Dataset: {current_filename}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", styles['Normal']))
        elements.append(PageBreak())
        
        # ========== SECTION 1: MISSING VALUES ==========
        elements.append(Paragraph("1. Missing Values Analysis", heading_style))
        elements.append(Spacer(1, 12))
        
        # Purpose and explanation
        elements.append(Paragraph("<b>Purpose</b>", subheading_style))
        elements.append(Paragraph(
            "Missing values in datasets can significantly impact model performance and statistical analysis. "
            "This section identifies columns with missing data and provides comprehensive strategies for handling them effectively.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Missing values data
        missing = current_analyzer.get_missing_values()
        if len(missing) > 0:
            elements.append(Paragraph("<b>Missing Values in Your Dataset</b>", subheading_style))
            
            # Create table
            missing_data = [['Column', 'Missing Count', 'Missing %']]
            for idx, row in missing.iterrows():
                missing_data.append([
                    str(row['Column']),
                    str(int(row['Missing Count'])),
                    f"{row['Missing %']:.2f}%"
                ])
            
            missing_table = Table(missing_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(missing_table)
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("No missing values detected in the dataset.", normal_style))
            elements.append(Spacer(1, 12))
        
        # Solutions
        elements.append(Paragraph("<b>Proposed Solutions</b>", subheading_style))
        elements.append(Paragraph(
            "<b>Solution 1: Statistical Imputation</b> - Replace missing values with mean, median, or mode. "
            "Best for numeric features with 5-30% missing values.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>Solution 2: Advanced Imputation</b> - Use KNN or regression-based imputation for better accuracy.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Code snippet
        elements.append(Paragraph("<b>Implementation Code</b>", subheading_style))
        code_text = """# Statistical imputation
df['column'].fillna(df['column'].mean(), inplace=True)  # Mean
df['column'].fillna(df['column'].median(), inplace=True)  # Median

# Advanced imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df[numeric_cols])"""
        elements.append(Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
        elements.append(PageBreak())
        
        # ========== SECTION 2: OUTLIER ANALYSIS ==========
        elements.append(Paragraph("2. Outlier Analysis", heading_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("<b>Purpose</b>", subheading_style))
        elements.append(Paragraph(
            "Outliers are data points that significantly deviate from other observations. "
            "They can indicate errors, anomalies, or valuable insights depending on the context.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Outlier detection
        outliers = current_analyzer.detect_outliers()
        if len(outliers) > 0:
            elements.append(Paragraph("<b>Outliers Detected in Your Dataset</b>", subheading_style))
            
            outlier_data = [['Column', 'Outlier Count', 'Percentage', 'Lower Bound', 'Upper Bound']]
            for col, info in outliers.items():
                outlier_data.append([
                    str(col),
                    str(info['count']),
                    f"{info['percentage']:.2f}%",
                    f"{info['lower_bound']:.2f}",
                    f"{info['upper_bound']:.2f}"
                ])
            
            outlier_table = Table(outlier_data, colWidths=[1.8*inch, 1.2*inch, 1*inch, 1.2*inch, 1.2*inch])
            outlier_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(outlier_table)
            elements.append(Spacer(1, 12))
            
            # Recommended actions for each column
            elements.append(Paragraph("<b>Recommended Actions by Column</b>", subheading_style))
            elements.append(Spacer(1, 6))
            
            for col, info in outliers.items():
                percentage = info['percentage']
                count = info['count']
                
                # Determine recommendation based on outlier percentage
                if percentage < 1:
                    recommendation = (
                        f"<b>{col}</b> ({percentage:.2f}% outliers - {count} values):<br/>"
                        f"<b>Recommended Action:</b> <i>Remove outliers</i><br/>"
                        f"• Very low percentage of outliers - safe to remove<br/>"
                        f"• Use boolean indexing to filter out values outside [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]<br/>"
                        f"• Minimal impact on dataset size"
                    )
                    color = colors.green
                elif percentage < 5:
                    recommendation = (
                        f"<b>{col}</b> ({percentage:.2f}% outliers - {count} values):<br/>"
                        f"<b>Recommended Action:</b> <i>Winsorization (Capping)</i><br/>"
                        f"• Moderate percentage - replace extreme values with boundary values<br/>"
                        f"• Cap values below {info['lower_bound']:.2f} and above {info['upper_bound']:.2f}<br/>"
                        f"• Preserves dataset size while reducing extreme values impact"
                    )
                    color = colors.orange
                elif percentage < 10:
                    recommendation = (
                        f"<b>{col}</b> ({percentage:.2f}% outliers - {count} values):<br/>"
                        f"<b>Recommended Action:</b> <i>Log Transformation</i><br/>"
                        f"• High percentage - transformation is safer than removal<br/>"
                        f"• Apply log(x+1) or Box-Cox transformation to compress range<br/>"
                        f"• Reduces impact of outliers while preserving all data"
                    )
                    color = colors.red
                else:
                    recommendation = (
                        f"<b>{col}</b> ({percentage:.2f}% outliers - {count} values):<br/>"
                        f"<b>Recommended Action:</b> <i>Investigation Required</i><br/>"
                        f"• Very high percentage - may indicate data distribution issues<br/>"
                        f"• Investigate if outliers are legitimate or data collection errors<br/>"
                        f"• Consider robust scaling or domain-specific handling"
                    )
                    color = colors.HexColor('#8B0000')
                
                # Create colored box for each recommendation
                rec_style = ParagraphStyle(
                    'Recommendation',
                    parent=normal_style,
                    fontSize=9,
                    textColor=colors.black,
                    leftIndent=10,
                    rightIndent=10,
                    spaceBefore=6,
                    spaceAfter=6
                )
                
                rec_data = [[Paragraph(recommendation, rec_style)]]
                rec_table = Table(rec_data, colWidths=[6.5*inch])
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), color),
                    ('ALPHA', (0, 0), (-1, -1), 0.15),
                    ('BOX', (0, 0), (-1, -1), 2, color),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ]))
                elements.append(rec_table)
                elements.append(Spacer(1, 8))
            
            elements.append(Spacer(1, 6))
            
            # Add boxplot visualizations
            elements.append(Paragraph("<b>Visualizations</b>", subheading_style))
            for col in list(outliers.keys())[:3]:  # Limit to first 3 columns
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.boxplot(current_analyzer.df[col].dropna())
                    ax.set_ylabel(col)
                    ax.set_title(f'Box Plot: {col}')
                    ax.grid(alpha=0.3)
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    plt.close(fig)
                    
                    img = Image(img_buffer, width=4*inch, height=2*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 6))
                except:
                    pass
        else:
            elements.append(Paragraph("No significant outliers detected in the dataset.", normal_style))
        
        elements.append(Spacer(1, 12))
        
        # General Solutions
        elements.append(Paragraph("<b>General Solutions for Outlier Treatment</b>", subheading_style))
        elements.append(Paragraph(
            "<b>Solution 1: Investigation</b> - Analyze outliers to determine if they're legitimate or errors.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>Solution 2: Transformation</b> - Apply log or Box-Cox transformations to compress outliers.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>Solution 3: Capping/Winsorization</b> - Replace extreme values with percentile values.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>Solution 4: Robust Scaling</b> - Use robust scaler that is less sensitive to outliers.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Code snippet
        elements.append(Paragraph("<b>Implementation Code</b>", subheading_style))
        code_text = """# IQR method for outlier detection
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]

# Method 1: Remove outliers (<1% outliers)
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Method 2: Capping/Winsorization (1-5% outliers)
from scipy.stats.mstats import winsorize
df['column_winsorized'] = winsorize(df['column'], limits=[0.05, 0.05])

# Method 3: Log Transformation (5-10% outliers)
df['column_log'] = np.log1p(df['column'])

# Method 4: Robust Scaling (>10% outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df['column_scaled'] = scaler.fit_transform(df[['column']])"""
        elements.append(Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
        elements.append(PageBreak())
        
        # ========== SECTION 3: DATA TRANSFORMATION ==========
        elements.append(Paragraph("3. Data Transformation", heading_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("<b>Purpose</b>", subheading_style))
        elements.append(Paragraph(
            "Data transformation adjusts variable distributions to improve model performance, "
            "satisfy statistical assumptions, and enhance the interpretability of results.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Skewness analysis
        stats_df = current_analyzer.get_statistics()
        if stats_df is not None and 'skewness' in stats_df.columns:
            elements.append(Paragraph("<b>Skewness Analysis</b>", subheading_style))
            elements.append(Paragraph(
                "Features with high skewness (>2 or <-2) may benefit from transformation:",
                normal_style
            ))
            elements.append(Spacer(1, 6))
            
            skew_data = [['Feature', 'Skewness', 'Recommendation']]
            for idx, row in stats_df.iterrows():
                skew = row['skewness']
                if abs(skew) > 0.5:  # Show features with any skewness
                    recommendation = "Transform" if abs(skew) > 2 else "Monitor"
                    skew_data.append([
                        str(idx),
                        f"{skew:.2f}",
                        recommendation
                    ])
            
            if len(skew_data) > 1:
                skew_table = Table(skew_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
                skew_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(skew_table)
                elements.append(Spacer(1, 12))
        
        # Methods and code
        elements.append(Paragraph("<b>Transformation Methods</b>", subheading_style))
        elements.append(Paragraph(
            "<b>Log Transformation:</b> Best for right-skewed data (income, prices, population).",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>Square Root:</b> For moderately skewed count data.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>Box-Cox:</b> Automatically finds optimal transformation parameter.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Decision Guide
        elements.append(Paragraph("<b>Which Transformation Should You Choose?</b>", subheading_style))
        elements.append(Paragraph(
            "<b>Decision Criteria Based on Data Characteristics:</b>",
            normal_style
        ))
        elements.append(Spacer(1, 6))
        
        decision_data = [
            ['Data Characteristic', 'Recommended Transformation', 'Why?'],
            ['Skewness: 0.5 to 1.0\n(Mild skew)', 'Square Root Transformation', 'Gentle transformation;\npreserves relationships'],
            ['Skewness: 1.0 to 2.0\n(Moderate skew)', 'Log Transformation\n(log or log1p)', 'Effective for most cases;\nhandles zeros with log1p'],
            ['Skewness > 2.0\n(High skew)', 'Box-Cox Transformation', 'Automatically finds optimal;\nstrongest correction'],
            ['Data with zeros', 'Log1p or Square Root', 'log(x+1) or sqrt handle zeros;\nlog(x) would fail'],
            ['Negative values present', 'Yeo-Johnson Transform', 'Variant of Box-Cox;\nworks with negatives'],
            ['Count data (integers)', 'Square Root or Log1p', 'Natural for Poisson data;\npreserves count nature'],
            ['Financial/Price data', 'Log Transformation', 'Standard in finance;\nlog-returns interpretable'],
            ['Unclear distribution', 'Box-Cox (try first)', 'Automated optimization;\nthen validate results']
        ]
        
        decision_table = Table(decision_data, colWidths=[2*inch, 2*inch, 2*inch])
        decision_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
        ]))
        elements.append(decision_table)
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("<b>⚠️ Important Validation Steps:</b>", subheading_style))
        elements.append(Paragraph(
            "• Always visualize distribution BEFORE and AFTER transformation<br/>"
            "• Check if skewness improved (closer to 0 is better)<br/>"
            "• Verify model performance improved (R², RMSE, etc.)<br/>"
            "• Remember to inverse-transform predictions for interpretation<br/>"
            "• Apply same transformation to test/production data",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Code snippet
        elements.append(Paragraph("<b>Implementation Code</b>", subheading_style))
        code_text = """# Log transformation
df['column_log'] = np.log1p(df['column'])  # log1p handles zeros

# Square root transformation
df['column_sqrt'] = np.sqrt(df['column'])

# Box-Cox transformation (positive values only)
from scipy.stats import boxcox
df['column_boxcox'], lambda_param = boxcox(df['column'])
print(f'Optimal lambda: {lambda_param}')

# Yeo-Johnson transformation (works with negative values)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['column_transformed'] = pt.fit_transform(df[['column']])"""
        elements.append(Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
        elements.append(PageBreak())
        
        # ========== SECTION 4: MULTICOLLINEARITY ==========
        elements.append(Paragraph("4. Multicollinearity Analysis", heading_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("<b>Purpose</b>", subheading_style))
        elements.append(Paragraph(
            "Multicollinearity occurs when independent variables are highly correlated. "
            "This analysis identifies these relationships and provides strategies to address redundancy.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Correlation analysis
        if len(current_analyzer.numeric_cols) >= 2:
            elements.append(Paragraph("<b>Correlation Analysis</b>", subheading_style))
            
            # Generate correlation matrix visualization
            try:
                corr_matrix = current_analyzer.df[current_analyzer.numeric_cols].corr()
                
                # Find high correlations
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr_pairs.append([
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                f"{corr_matrix.iloc[i, j]:.3f}"
                            ])
                
                if high_corr_pairs:
                    elements.append(Paragraph("High correlation pairs detected (|r| > 0.7):", normal_style))
                    elements.append(Spacer(1, 6))
                    
                    corr_data = [['Feature 1', 'Feature 2', 'Correlation']]
                    corr_data.extend(high_corr_pairs)
                    
                    corr_table = Table(corr_data, colWidths=[2*inch, 2*inch, 1.5*inch])
                    corr_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(corr_table)
                    elements.append(Spacer(1, 12))
                    
                    # Add correlation heatmap
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                              center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                              ax=ax)
                    ax.set_title('Correlation Matrix')
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    plt.close(fig)
                    
                    img = Image(img_buffer, width=5*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                else:
                    elements.append(Paragraph("No high correlation pairs detected (all |r| < 0.7).", normal_style))
                    elements.append(Spacer(1, 12))
            except:
                pass
        
        # Qualitative Variables Correlation (Cramér's V)
        if len(current_analyzer.categorical_cols) >= 2:
            elements.append(Paragraph("<b>Qualitative Variables Correlation (Cramér's V)</b>", subheading_style))
            elements.append(Paragraph(
                "Cramér's V measures association between categorical variables. "
                "Values range from 0 (no association) to 1 (perfect association).",
                normal_style
            ))
            elements.append(Spacer(1, 6))
            
            try:
                cramers_matrix = current_analyzer.calculate_cramers_v_matrix()
                
                if cramers_matrix is not None:
                    # Find high associations
                    high_assoc_pairs = []
                    for i in range(len(cramers_matrix.columns)):
                        for j in range(i+1, len(cramers_matrix.columns)):
                            if cramers_matrix.iloc[i, j] > 0.5:
                                high_assoc_pairs.append([
                                    cramers_matrix.columns[i],
                                    cramers_matrix.columns[j],
                                    f"{cramers_matrix.iloc[i, j]:.3f}"
                                ])
                    
                    if high_assoc_pairs:
                        elements.append(Paragraph("Strong associations detected (V > 0.5):", normal_style))
                        elements.append(Spacer(1, 6))
                        
                        assoc_data = [['Feature 1', 'Feature 2', "Cramér's V"]]
                        assoc_data.extend(high_assoc_pairs[:10])  # Limit to 10
                        
                        assoc_table = Table(assoc_data, colWidths=[2*inch, 2*inch, 1.5*inch])
                        assoc_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff9800')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        elements.append(assoc_table)
                        elements.append(Spacer(1, 12))
                        
                        # Add heatmap
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cramers_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                                  vmin=0, vmax=1, square=True, linewidths=1, 
                                  cbar_kws={"shrink": 0.8}, ax=ax)
                        ax.set_title("Cramér's V Matrix")
                        
                        img_buffer = io.BytesIO()
                        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                        img_buffer.seek(0)
                        plt.close(fig)
                        
                        img = Image(img_buffer, width=5*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 12))
                    else:
                        elements.append(Paragraph("No strong associations detected (all V < 0.5).", normal_style))
                        elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error in Cramér's V analysis: {e}")
                pass
        
        # Mixed Correlation (Quantitative vs Qualitative)
        if len(current_analyzer.numeric_cols) > 0 and len(current_analyzer.categorical_cols) > 0:
            elements.append(Paragraph("<b>Mixed Correlation - Quantitative vs Qualitative (η²)</b>", subheading_style))
            elements.append(Paragraph(
                "Correlation ratio (η²) measures how well categorical variables explain variance in numeric variables. "
                "Values range from 0 (no effect) to 1 (perfect explanation).",
                normal_style
            ))
            elements.append(Spacer(1, 6))
            
            try:
                corr_ratio_matrix = current_analyzer.calculate_correlation_ratio_matrix()
                
                if corr_ratio_matrix is not None:
                    # Find high correlation ratios
                    high_eta_pairs = []
                    for i, num_col in enumerate(corr_ratio_matrix.index):
                        for j, cat_col in enumerate(corr_ratio_matrix.columns):
                            if corr_ratio_matrix.iloc[i, j] > 0.14:  # Large effect size
                                high_eta_pairs.append([
                                    num_col,
                                    cat_col,
                                    f"{corr_ratio_matrix.iloc[i, j]:.3f}"
                                ])
                    
                    if high_eta_pairs:
                        elements.append(Paragraph("Strong effects detected (η² > 0.14):", normal_style))
                        elements.append(Spacer(1, 6))
                        
                        eta_data = [['Numeric Feature', 'Categorical Feature', 'η²']]
                        eta_data.extend(high_eta_pairs[:10])  # Limit to 10
                        
                        eta_table = Table(eta_data, colWidths=[2*inch, 2*inch, 1.5*inch])
                        eta_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4caf50')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        elements.append(eta_table)
                        elements.append(Spacer(1, 12))
                        
                        # Add heatmap
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sns.heatmap(corr_ratio_matrix, annot=True, fmt='.2f', cmap='viridis',
                                  vmin=0, vmax=1, square=True, linewidths=1, 
                                  cbar_kws={"shrink": 0.8}, ax=ax)
                        ax.set_title('Correlation Ratio (η²)')
                        ax.set_xlabel('Categorical Variables')
                        ax.set_ylabel('Numeric Variables')
                        
                        img_buffer = io.BytesIO()
                        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                        img_buffer.seek(0)
                        plt.close(fig)
                        
                        img = Image(img_buffer, width=5.5*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 12))
                    else:
                        elements.append(Paragraph("No strong effects detected (all η² < 0.14).", normal_style))
                        elements.append(Spacer(1, 12))
            except Exception as e:
                print(f"Error in correlation ratio analysis: {e}")
                pass
        
        # Solutions
        elements.append(Paragraph("<b>Proposed Solutions</b>", subheading_style))
        elements.append(Paragraph(
            "<b>For Quantitative Variables:</b>",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Feature Selection</b> - Remove one feature from highly correlated pairs (|r| > 0.9)",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>PCA</b> - Transform correlated features into uncorrelated principal components",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Regularization</b> - Use Ridge or Lasso regression to handle multicollinearity",
            normal_style
        ))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            "<b>For Qualitative Variables:</b>",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Merge Categories</b> - Combine highly associated categorical features (V > 0.7)",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Feature Engineering</b> - Create interaction terms or composite features",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Dimensionality Reduction</b> - Use Multiple Correspondence Analysis (MCA)",
            normal_style
        ))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            "<b>For Mixed Variables:</b>",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Group-based Models</b> - Use categorical features for stratification (η² > 0.14)",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Interaction Features</b> - Create numeric × categorical interaction terms",
            normal_style
        ))
        elements.append(Paragraph(
            "• <b>Target Encoding</b> - Encode categories based on numeric target statistics",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Code snippet
        elements.append(Paragraph("<b>Implementation Code</b>", subheading_style))
        code_text = """# 1. Quantitative Correlation (Pearson)
corr_matrix = df[numeric_cols].corr()
high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix.abs() < 1.0)

# VIF calculation for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]

# 2. Qualitative Correlation (Cramér's V)
from scipy.stats import chi2_contingency
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# Calculate for all categorical pairs
cat_corr = pd.DataFrame()
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cat_corr.loc[col1, col2] = cramers_v(df[col1], df[col2])

# 3. Mixed Correlation (η²)
def correlation_ratio(categories, values):
    categories = pd.Series(categories)
    values = pd.Series(values)
    overall_mean = values.mean()
    ss_total = ((values - overall_mean) ** 2).sum()
    
    groups = values.groupby(categories)
    ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 
                     for _, g in groups)
    return ss_between / ss_total

# Calculate for numeric vs categorical
mixed_corr = pd.DataFrame()
for num_col in numeric_cols:
    for cat_col in categorical_cols:
        mixed_corr.loc[num_col, cat_col] = correlation_ratio(
            df[cat_col], df[num_col]
        )"""
        elements.append(Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
        elements.append(PageBreak())
        
        # ========== SECTION 5: FEATURE SCALING ==========
        elements.append(Paragraph("5. Feature Scaling", heading_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("<b>Purpose</b>", subheading_style))
        elements.append(Paragraph(
            "Feature scaling ensures all features contribute equally to model training by normalizing their ranges. "
            "This is crucial for distance-based algorithms and gradient descent optimization.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Scale analysis
        if stats_df is not None and 'std' in stats_df.columns:
            elements.append(Paragraph("<b>Scale Analysis</b>", subheading_style))
            elements.append(Paragraph(
                "Comparison of feature scales (standard deviation):",
                normal_style
            ))
            elements.append(Spacer(1, 6))
            
            scale_data = [['Feature', 'Mean', 'Std Dev', 'Min', 'Max']]
            for idx, row in stats_df.iterrows():
                scale_data.append([
                    str(idx)[:20],  # Truncate long names
                    f"{row['mean']:.2f}",
                    f"{row['std']:.2f}",
                    f"{row['min']:.2f}",
                    f"{row['max']:.2f}"
                ])
            
            scale_table = Table(scale_data, colWidths=[1.8*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            scale_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 8)
            ]))
            elements.append(scale_table)
            elements.append(Spacer(1, 12))
        
        # Methods
        elements.append(Paragraph("<b>Scaling Methods</b>", subheading_style))
        elements.append(Paragraph(
            "<b>StandardScaler (Z-score):</b> Transforms to mean=0, std=1. Best for normally distributed data.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>MinMaxScaler:</b> Scales to [0, 1] range. Best when you need bounded values.",
            normal_style
        ))
        elements.append(Paragraph(
            "<b>RobustScaler:</b> Uses median and IQR. Best when outliers are present.",
            normal_style
        ))
        elements.append(Spacer(1, 12))
        
        # Code snippet
        elements.append(Paragraph("<b>Implementation Code</b>", subheading_style))
        code_text = """# StandardScaler (Z-score normalization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

# RobustScaler (robust to outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])"""
        elements.append(Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        # Send file
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'EDA_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
