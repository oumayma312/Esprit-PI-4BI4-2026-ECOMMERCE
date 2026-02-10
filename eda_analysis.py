"""
Comprehensive Exploratory Data Analysis (EDA)
This script performs complete data analysis including:
- Data loading and overview
- Missing values analysis
- Descriptive statistics
- Visualizations (histograms, boxplots, bar charts)
- Outlier detection
- PCA analysis with 3D visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class DataAnalyzer:
    """
    Classe principale pour effectuer une Analyse Exploratoire de Données (EDA) complète
    
    Cette classe gère:
    - Le chargement et nettoyage des données
    - L'analyse des valeurs manquantes
    - Les statistiques descriptives
    - La détection des outliers
    - Les visualisations (histogrammes, boxplots, corrélations)
    - L'analyse en composantes principales (PCA)
    """
    
    def __init__(self, file_path):
        """
        Initialise l'analyseur avec un fichier de données
        
        Args:
            file_path (str): Chemin vers le fichier CSV à analyser
        """
        self.file_path = file_path
        self.df = None  # DataFrame pandas contenant les données
        self.numeric_cols = []  # Liste des colonnes numériques
        self.categorical_cols = []  # Liste des colonnes catégorielles
        self.recommendations = []  # Liste des recommandations générées
        
    def load_data(self):
        """
        Charge les données depuis un fichier CSV
        
        Essaie différents encodages (UTF-8, Latin-1) et séparateurs (';', ',')
        pour maximiser la compatibilité avec différents formats de fichiers
        
        Returns:
            bool: True si le chargement réussit, False sinon
        """
        try:
            # Tentative 1: Chargement avec séparateur ';' et encodage UTF-8
            try:
                self.df = pd.read_csv(self.file_path, sep=';', encoding='utf-8')
            except:
                # Tentative 2: Chargement avec encodage Latin-1 (pour fichiers européens)
                self.df = pd.read_csv(self.file_path, encoding='latin-1')
            
            print(f"✓ Data loaded successfully!")
            print(f"  Dataset dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def clean_data(self):
        """
        Nettoie et prépare les données pour l'analyse
        
        Effectue les opérations suivantes:
        - Identifie et convertit les colonnes numériques (enlève guillemets, virgules)
        - Convertit les colonnes de dates au format datetime
        - Nettoie les colonnes textuelles (supprime guillemets et espaces)
        - Classifie les colonnes en numériques et catégorielles
        """
        # Patterns pour identifier les colonnes numériques courantes
        numeric_patterns = ['pu ht', 'quantite', 'prix total ht', 'montant total ht', 
                          'total ht', 'total ttc', 'timbre']
        
        # Convertir les colonnes numériques identifiées par pattern
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in numeric_patterns):
                try:
                    # Supprimer les guillemets et remplacer virgules par points (format français)
                    self.df[col] = self.df[col].astype(str).str.replace('"', '').str.replace(',', '.')
                    # Convertir en numérique (NaN pour valeurs invalides)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass
        
        # Convertir la colonne date si elle existe
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            except:
                pass
        
        # Nettoyer toutes les colonnes textuelles
        for col in self.df.select_dtypes(include=['object']).columns:
            # Supprimer guillemets et espaces en début/fin
            self.df[col] = self.df[col].astype(str).str.replace('"', '').str.strip()
        
        # Identifier automatiquement les types de colonnes
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclure 'date' des colonnes catégorielles (traitement spécial pour dates)
        if 'date' in self.categorical_cols:
            self.categorical_cols.remove('date')
    
    def get_overview(self):
        """Get dataset overview"""
        overview = {
            'first_rows': self.df.head(10),
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes,
            'info': self.df.info()
        }
        return overview
    
    def analyze_missing_values(self):
        """
        Analyse les valeurs manquantes dans le dataset
        
        Calcule pour chaque colonne:
        - Le nombre de valeurs manquantes
        - Le pourcentage de valeurs manquantes
        
        Returns:
            DataFrame: Tableau des colonnes avec valeurs manquantes, trié par gravité
        """
        # Créer un DataFrame avec les statistiques de valeurs manquantes
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        # Filtrer uniquement les colonnes avec valeurs manquantes et trier
        missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        # Generate recommendations
        for idx, row in missing.iterrows():
            if row['Missing Percentage'] > 50:
                self.recommendations.append({
                    'type': 'column_removal',
                    'column': row['Column'],
                    'reason': f"High missing data: {row['Missing Percentage']:.1f}%",
                    'action': f"Consider removing column '{row['Column']}' due to {row['Missing Percentage']:.1f}% missing values"
                })
            elif row['Missing Percentage'] > 5:
                self.recommendations.append({
                    'type': 'imputation',
                    'column': row['Column'],
                    'reason': f"Moderate missing data: {row['Missing Percentage']:.1f}%",
                    'action': f"Consider imputing missing values in '{row['Column']}' using median/mode or dropping rows"
                })
        
        return missing
    
    def get_statistics(self):
        """Get descriptive statistics"""
        if len(self.numeric_cols) > 0:
            stats_df = self.df[self.numeric_cols].describe().T
            stats_df['median'] = self.df[self.numeric_cols].median()
            stats_df['variance'] = self.df[self.numeric_cols].var()
            stats_df = stats_df[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'variance']]
            return stats_df
        return None
    
    def detect_outliers(self):
        """
        Détecte les outliers (valeurs aberrantes) en utilisant la méthode IQR
        
        Méthode IQR (Interquartile Range):
        - Q1 = Premier quartile (25%)
        - Q3 = Troisième quartile (75%)
        - IQR = Q3 - Q1
        - Outliers: valeurs < Q1 - 1.5*IQR ou > Q3 + 1.5*IQR
        
        Returns:
            dict: Dictionnaire avec info sur outliers pour chaque colonne
        """
        outliers_info = {}
        
        # Analyser chaque colonne numérique
        for col in self.numeric_cols:
            # Calculer les quartiles
            Q1 = self.df[col].quantile(0.25)  # 25ème percentile
            Q3 = self.df[col].quantile(0.75)  # 75ème percentile
            IQR = Q3 - Q1  # Écart interquartile
            
            # Définir les limites de détection des outliers
            lower_bound = Q1 - 1.5 * IQR  # Limite inférieure
            upper_bound = Q3 + 1.5 * IQR  # Limite supérieure
            
            # Identifier les outliers (valeurs hors des limites)
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers_count = len(outliers)
            outliers_pct = (outliers_count / len(self.df)) * 100
            
            if outliers_count > 0:
                outliers_info[col] = {
                    'count': outliers_count,
                    'percentage': outliers_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                if outliers_pct > 10:
                    self.recommendations.append({
                        'type': 'outliers',
                        'column': col,
                        'reason': f"High outlier rate: {outliers_pct:.1f}%",
                        'action': f"Review outliers in '{col}' - {outliers_count} values ({outliers_pct:.1f}%) outside normal range"
                    })
        
        return outliers_info
    
    def create_visualizations(self):
        """Create all visualizations"""
        visualizations = {}
        
        # 1. Missing values heatmap
        if self.df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Heatmap', fontsize=16, fontweight='bold')
            ax.set_xlabel('Columns', fontsize=12)
            visualizations['missing_heatmap'] = fig
            plt.close()
        
        # 2. Histograms for numeric columns
        for col in self.numeric_cols[:6]:  # Limit to first 6 numeric columns
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df[col].hist(bins=30, edgecolor='black', color='skyblue', ax=ax)
            ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(alpha=0.3)
            visualizations[f'hist_{col}'] = fig
            plt.close()
        
        # 3. Boxplots for numeric columns
        for col in self.numeric_cols[:6]:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df.boxplot(column=col, ax=ax, patch_artist=True,
                           boxprops=dict(facecolor='lightblue'),
                           medianprops=dict(color='red', linewidth=2))
            ax.set_title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
            ax.set_ylabel(col, fontsize=12)
            ax.grid(alpha=0.3)
            visualizations[f'box_{col}'] = fig
            plt.close()
        
        # 4. Correlation heatmap - Quantitative variables
        if len(self.numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = self.df[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, ax=ax)
            ax.set_title('Correlation Matrix - Quantitative Variables', fontsize=16, fontweight='bold')
            visualizations['correlation_quantitative'] = fig
            plt.close()
        
        # 4b. Correlation heatmap - Qualitative variables (Cramér's V)
        if len(self.categorical_cols) > 1:
            cramers_matrix = self.calculate_cramers_v_matrix()
            if cramers_matrix is not None:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cramers_matrix, annot=True, fmt='.2f', cmap='YlOrRd', 
                           vmin=0, vmax=1, square=True, linewidths=1, ax=ax)
                ax.set_title("Correlation Matrix - Qualitative Variables (Cramér's V)", 
                           fontsize=16, fontweight='bold')
                visualizations['correlation_qualitative'] = fig
                plt.close()
        
        # 4c. Correlation heatmap - Quantitative vs Qualitative (Correlation Ratio)
        if len(self.numeric_cols) > 0 and len(self.categorical_cols) > 0:
            corr_ratio_matrix = self.calculate_correlation_ratio_matrix()
            if corr_ratio_matrix is not None:
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(corr_ratio_matrix, annot=True, fmt='.2f', cmap='viridis', 
                           vmin=0, vmax=1, square=True, linewidths=1, ax=ax)
                ax.set_title('Correlation Matrix - Quantitative vs Qualitative Variables (η²)', 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('Qualitative Variables', fontsize=12)
                ax.set_ylabel('Quantitative Variables', fontsize=12)
                visualizations['correlation_mixed'] = fig
                plt.close()
        
        # 5. Bar charts for categorical columns
        for col in self.categorical_cols[:5]:  # Limit to first 5 categorical columns
            value_counts = self.df[col].value_counts().head(10)
            if len(value_counts) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                value_counts.plot(kind='bar', color='teal', edgecolor='black', ax=ax)
                ax.set_title(f'Top 10 Values in {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                visualizations[f'bar_{col}'] = fig
                plt.close()
        
        return visualizations
    
    def calculate_cramers_v_matrix(self):
        """
        Calcule la matrice de corrélation de Cramér's V pour les variables catégorielles
        
        Cramér's V mesure l'association entre deux variables catégorielles:
        - Valeurs entre 0 et 1
        - 0 = aucune association
        - 1 = association parfaite
        - Basé sur le test du Chi-carré
        
        Returns:
            DataFrame: Matrice de corrélation Cramér's V (ou None si impossible)
        """
        try:
            # Limiter aux 10 premières colonnes catégorielles pour éviter surcharge
            cats = self.categorical_cols[:10]
            n_cats = len(cats)
            
            # Vérifier qu'il y a au moins 2 variables catégorielles
            if n_cats < 2:
                return None
            
            # Initialiser la matrice de corrélation
            cramers_matrix = np.zeros((n_cats, n_cats))
            
            # Calculer Cramér's V pour chaque paire de variables
            for i, cat1 in enumerate(cats):
                for j, cat2 in enumerate(cats):
                    if i == j:
                        # Diagonale: corrélation parfaite avec soi-même
                        cramers_matrix[i, j] = 1.0
                    elif i < j:
                        # Calculer Cramér's V pour la paire (cat1, cat2)
                        try:
                            # Créer la table de contingence
                            contingency_table = pd.crosstab(self.df[cat1], self.df[cat2])
                            # Test du Chi-carré
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            n = contingency_table.sum().sum()  # Nombre total d'observations
                            min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                            if min_dim > 0:
                                # Formule de Cramér's V
                                cramers_v = np.sqrt(chi2 / (n * min_dim))
                                # Matrice symétrique: remplir les deux côtés
                                cramers_matrix[i, j] = cramers_v
                                cramers_matrix[j, i] = cramers_v
                        except:
                            # En cas d'erreur, mettre 0 (aucune corrélation)
                            cramers_matrix[i, j] = 0
                            cramers_matrix[j, i] = 0
            
            return pd.DataFrame(cramers_matrix, index=cats, columns=cats)
        except Exception as e:
            print(f"Error calculating Cramér's V: {e}")
            return None
    
    def calculate_correlation_ratio_matrix(self):
        """
        Calcule la matrice de rapport de corrélation (η²) entre variables quantitatives et qualitatives
        
        Le rapport de corrélation (eta-squared) mesure la relation entre:
        - Variable quantitative (dépendante): numérique continue
        - Variable qualitative (indépendante): catégorielle
        
        Méthode basée sur l'ANOVA:
        - η² = SS_between / SS_total
        - Valeurs entre 0 (aucune relation) et 1 (relation parfaite)
        - Interprétation:
          * η² < 0.06: effet faible
          * 0.06 ≤ η² < 0.14: effet moyen
          * η² ≥ 0.14: effet fort
        
        Returns:
            DataFrame: Matrice de rapport de corrélation (lignes=numériques, colonnes=catégorielles)
        """
        try:
            # Limiter à 10 colonnes de chaque type pour éviter surcharge
            num_cols = self.numeric_cols[:10]
            cat_cols = self.categorical_cols[:10]
            
            # Vérifier qu'il y a des colonnes des deux types
            if len(num_cols) == 0 or len(cat_cols) == 0:
                return None
            
            # Initialiser la matrice de résultats
            corr_ratio_matrix = np.zeros((len(num_cols), len(cat_cols)))
            
            # Calculer η² pour chaque paire (numérique, catégorielle)
            for i, num_col in enumerate(num_cols):
                for j, cat_col in enumerate(cat_cols):
                    try:
                        # Obtenir les catégories uniques de la variable qualitative
                        categories = self.df[cat_col].dropna().unique()
                        
                        # Limiter le nombre de catégories (trop de catégories = calcul instable)
                        if len(categories) > 50:
                            continue
                        
                        # Créer les groupes pour chaque catégorie
                        groups = []
                        for cat in categories:
                            group_data = self.df[self.df[cat_col] == cat][num_col].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data.values)
                        
                        # Vérifier qu'il y a au moins 2 groupes pour ANOVA
                        if len(groups) < 2:
                            corr_ratio_matrix[i, j] = 0
                            continue
                        
                        # Calculer la moyenne générale
                        all_data = self.df[num_col].dropna()
                        overall_mean = all_data.mean()
                        
                        # Somme des carrés totale (SS_total)
                        ss_total = np.sum((all_data - overall_mean) ** 2)
                        
                        if ss_total == 0:
                            corr_ratio_matrix[i, j] = 0
                            continue
                        
                        # Somme des carrés inter-groupes (SS_between)
                        ss_between = 0
                        for group in groups:
                            if len(group) > 0:
                                group_mean = np.mean(group)
                                # Contribution de ce groupe à la variance inter-groupes
                                ss_between += len(group) * (group_mean - overall_mean) ** 2
                        
                        # Calculer le rapport de corrélation η²
                        eta_squared = ss_between / ss_total
                        # S'assurer que la valeur est entre 0 et 1
                        corr_ratio_matrix[i, j] = min(eta_squared, 1.0)
                        
                    except Exception as e:
                        # En cas d'erreur, mettre 0
                        corr_ratio_matrix[i, j] = 0
            
            # Retourner sous forme de DataFrame avec noms de colonnes
            return pd.DataFrame(corr_ratio_matrix, index=num_cols, columns=cat_cols)
        except Exception as e:
            print(f"Error calculating correlation ratio: {e}")
            return None
    
    def perform_pca(self):
        """
        Effectue l'Analyse en Composantes Principales (PCA) avec visualisations complètes
        
        La PCA est une technique de réduction de dimensionnalité qui:
        - Transforme les variables corrélées en composantes non-corrélées
        - Maximise la variance expliquée par chaque composante
        - Permet d'identifier les variables les plus importantes
        
        Étapes:
        1. Standardisation des données (moyenne=0, écart-type=1)
        2. Calcul des composantes principales
        3. Analyse de la variance expliquée
        4. Génération de 9 visualisations détaillées
        
        Returns:
            dict: Résultats PCA avec:
                - explained_variance: Part de variance expliquée par chaque PC
                - cumulative_variance: Variance cumulée
                - components: Vecteurs propres (loadings)
                - pca_df: DataFrame avec les nouvelles coordonnées
                - eigenvalues: Valeurs propres
                - visualisations: Dict de 9 graphiques
        """
        pca_results = {}
        
        # Vérifier qu'il y a au moins 2 variables numériques
        if len(self.numeric_cols) < 2:
            return None
        
        # Préparer les données: ne garder que les lignes sans valeurs manquantes
        df_numeric = self.df[self.numeric_cols].dropna()
        
        # Vérifier qu'il reste suffisamment de données
        if len(df_numeric) < 3:
            return None
        
        # Étape 1: Standardisation des données (OBLIGATOIRE pour PCA)
        # Chaque variable aura moyenne=0 et écart-type=1
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Déterminer le nombre de composantes à calculer
        # Maximum possible = min(nombre de variables, nombre d'observations)
        n_components = min(len(self.numeric_cols), len(df_numeric))
        
        # Étape 2: Exécuter la PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        # Créer un DataFrame avec les nouvelles coordonnées
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pca_cols)
        
        # Stocker les résultats principaux
        pca_results['explained_variance'] = pca.explained_variance_ratio_  # % variance par PC
        pca_results['cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)  # % cumulé
        pca_results['components'] = pca.components_  # Vecteurs propres
        pca_results['pca_df'] = pca_df  # Données transformées
        pca_results['n_components'] = n_components  # Nombre de composantes
        pca_results['eigenvalues'] = pca.explained_variance_  # Valeurs propres (λ)
        pca_results['feature_names'] = self.numeric_cols  # Noms des variables originales
        
        # Calculer les loadings (contributions des variables aux composantes)
        # Loadings = eigenvectors * sqrt(eigenvalues)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        pca_results['loadings'] = loadings
        
        # Étape 3: Créer les visualisations
        visualizations = {}
        
        # ===== VISUALISATION 1: Scree Plot amélioré =====
        fig, ax = plt.subplots(figsize=(10, 6))
        # Barres pour chaque composante
        ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
               color='steelblue', edgecolor='black', alpha=0.7, label='Explained Variance')
        # Ligne de tendance
        ax.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 
                'ro-', linewidth=2, markersize=8)
        # Ligne de référence (critère de Kaiser)
        ax.axhline(y=1/len(self.numeric_cols), color='orange', linestyle='--', 
                   label='Average (Kaiser criterion)', alpha=0.7)
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title('Scree Plot - Explained Variance by Component', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        # Ajouter les pourcentages au-dessus des barres
        for i, v in enumerate(pca.explained_variance_ratio_):
            ax.text(i + 1, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold', fontsize=9)
        visualizations['scree_plot'] = fig
        plt.close()
        
        # 2. Cumulative variance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, n_components + 1), pca_results['cumulative_variance'], 
                'go-', linewidth=2, markersize=8)
        ax.fill_between(range(1, n_components + 1), pca_results['cumulative_variance'], 
                        alpha=0.3, color='green')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(y=0.8, color='r', linestyle='--', label='80% threshold', linewidth=2)
        ax.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold', linewidth=2)
        ax.legend()
        for i, v in enumerate(pca_results['cumulative_variance']):
            ax.text(i + 1, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold', fontsize=9)
        visualizations['cumulative_variance'] = fig
        plt.close()
        
        # 3. Biplot (2D) - Principal Components with variable vectors
        if n_components >= 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot observations
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], 
                               c=range(len(pca_df)), cmap='viridis', 
                               s=50, alpha=0.5, edgecolors='black', linewidth=0.5)
            
            # Plot variable vectors (loadings)
            scale_factor = 3  # Scale factor for visibility
            for i, var in enumerate(self.numeric_cols):
                ax.arrow(0, 0, 
                        loadings[i, 0] * scale_factor, 
                        loadings[i, 1] * scale_factor,
                        head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', alpha=0.8, linewidth=2)
                ax.text(loadings[i, 0] * scale_factor * 1.15, 
                       loadings[i, 1] * scale_factor * 1.15,
                       var, fontsize=10, fontweight='bold', 
                       ha='center', color='darkred')
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.set_title('PCA Biplot - Observations and Variables', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Sample Index')
            visualizations['biplot'] = fig
            plt.close()
            
            # 4. Correlation Circle (for first 2 PCs)
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Draw circle
            circle = plt.Circle((0, 0), 1, color='navy', fill=False, linewidth=2)
            ax.add_patch(circle)
            
            # Plot correlations
            corr_with_pc = loadings[:, :2]
            for i, var in enumerate(self.numeric_cols):
                ax.arrow(0, 0, corr_with_pc[i, 0], corr_with_pc[i, 1],
                        head_width=0.05, head_length=0.05, 
                        fc='blue', ec='blue', alpha=0.8, linewidth=2)
                ax.text(corr_with_pc[i, 0] * 1.1, corr_with_pc[i, 1] * 1.1,
                       var, fontsize=10, fontweight='bold', ha='center')
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
            ax.set_title('Correlation Circle - Variable Contributions', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_aspect('equal')
            visualizations['correlation_circle'] = fig
            plt.close()
        
        # 5. 2D PCA plot (simple version)
        if n_components >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], 
                               c=range(len(pca_df)), cmap='plasma', 
                               s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.set_title('PCA - First Two Principal Components', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Sample Index')
            visualizations['pca_2d'] = fig
            plt.close()
        
        # 6. 3D Interactive PCA plot
        if n_components >= 3:
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=pca_df['PC1'],
                y=pca_df['PC2'],
                z=pca_df['PC3'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=range(len(pca_df)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sample Index"),
                    line=dict(color='black', width=0.5)
                ),
                text=[f'Sample {i}' for i in range(len(pca_df))],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
            )])
            
            fig_3d.update_layout(
                title='Interactive 3D PCA Visualization',
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
                    bgcolor='rgba(240, 240, 240, 0.9)'
                ),
                width=900,
                height=700
            )
            visualizations['pca_3d'] = fig_3d
        
        # 7. Enhanced Component loadings heatmap
        fig, ax = plt.subplots(figsize=(14, max(8, len(self.numeric_cols) * 0.5)))
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=pca_cols,
            index=self.numeric_cols
        )
        sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdBu_r', 
                   center=0, ax=ax, cbar_kws={'label': 'Loading Value'},
                   linewidths=0.5, linecolor='gray')
        ax.set_title('PCA Component Loadings (Eigenvectors)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Components', fontsize=12)
        ax.set_ylabel('Original Features', fontsize=12)
        visualizations['loadings'] = fig
        plt.close()
        
        # 8. Variable Contributions to each PC (bar plots)
        num_pcs_to_show = min(10, n_components)
        
        # Adjust layout based on number of components
        if num_pcs_to_show <= 3:
            ncols = num_pcs_to_show
            nrows = 1
            figsize = (6*ncols, 6)
        elif num_pcs_to_show <= 6:
            ncols = 3
            nrows = 2
            figsize = (18, 12)
        else:
            ncols = 4
            nrows = (num_pcs_to_show + 3) // 4
            figsize = (24, 6*nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx in range(num_pcs_to_show):
            contrib = (pca.components_[idx]**2) / (pca.components_[idx]**2).sum() * 100
            sorted_idx = np.argsort(contrib)[::-1]
            
            axes[idx].barh(np.array(self.numeric_cols)[sorted_idx], 
                          contrib[sorted_idx], color='teal', edgecolor='black')
            axes[idx].set_xlabel('Contribution (%)', fontsize=11)
            axes[idx].set_title(f'PC{idx+1} Variable Contributions\n({pca.explained_variance_ratio_[idx]:.1%} variance)', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3, axis='x')
            
            # Highlight top contributors
            for i, (var, val) in enumerate(zip(np.array(self.numeric_cols)[sorted_idx][:3], 
                                                contrib[sorted_idx][:3])):
                axes[idx].text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)
        
        # Hide unused subplots
        for idx in range(num_pcs_to_show, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        visualizations['contributions'] = fig
        plt.close()
        
        # 9. Eigenvalues plot (Kaiser criterion)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, n_components + 1), pca.explained_variance_, 
               color='coral', edgecolor='black', alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
                   label='Kaiser criterion (eigenvalue = 1)')
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title('Eigenvalues by Component (Kaiser Criterion)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        for i, v in enumerate(pca.explained_variance_):
            ax.text(i + 1, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)
        visualizations['eigenvalues'] = fig
        plt.close()
        
        pca_results['visualizations'] = visualizations
        
        # Generate enhanced PCA recommendations
        # Kaiser criterion
        n_kaiser = sum(pca.explained_variance_ > 1)
        if n_kaiser > 0:
            pca_results['kaiser_components'] = n_kaiser
            self.recommendations.append({
                'type': 'dimensionality',
                'reason': f'Kaiser criterion: {n_kaiser} components have eigenvalues > 1',
                'action': f'Consider retaining {n_kaiser} principal component(s) based on Kaiser criterion'
            })
        
        # Cumulative variance recommendations
        for threshold, pct in [(0.8, 80), (0.9, 90), (0.95, 95)]:
            n_comps = np.argmax(pca_results['cumulative_variance'] >= threshold) + 1
            if n_comps <= n_components and pca_results['cumulative_variance'][n_comps-1] >= threshold:
                self.recommendations.append({
                    'type': 'dimensionality',
                    'reason': f'{n_comps} components explain {pct}%+ variance',
                    'action': f'Use {n_comps} component(s) to retain {pct}% of information'
                })
                break
        
        # Check for highly correlated features
        high_corr_pairs = []
        if len(self.numeric_cols) > 1:
            corr_matrix = self.df[self.numeric_cols].corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            for col1, col2, corr_val in high_corr_pairs[:3]:
                self.recommendations.append({
                    'type': 'correlation',
                    'reason': f'High correlation ({corr_val:.2f}) between {col1} and {col2}',
                    'action': f'Consider removing one of the highly correlated features: {col1} or {col2}'
                })
        
        return pca_results
    
    def generate_recommendations(self):
        """Generate final recommendations"""
        # Add general recommendations
        
        # Check for constant columns
        for col in self.numeric_cols:
            if self.df[col].nunique() == 1:
                self.recommendations.append({
                    'type': 'constant_column',
                    'column': col,
                    'reason': 'Column has only one unique value',
                    'action': f"Remove constant column '{col}' as it provides no information"
                })
        
        # Check for low variance columns
        for col in self.numeric_cols:
            if self.df[col].std() < 0.01 * self.df[col].mean() and self.df[col].mean() != 0:
                self.recommendations.append({
                    'type': 'low_variance',
                    'column': col,
                    'reason': 'Very low variance',
                    'action': f"Consider removing '{col}' due to low variance (std/mean < 1%)"
                })
        
        # Check for skewness
        for col in self.numeric_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 2:
                self.recommendations.append({
                    'type': 'transformation',
                    'column': col,
                    'reason': f'High skewness: {skewness:.2f}',
                    'action': f"Apply log or Box-Cox transformation to '{col}' to reduce skewness"
                })
        
        # Normalization recommendation
        if len(self.numeric_cols) > 0:
            scales = []
            for col in self.numeric_cols:
                scales.append(self.df[col].std())
            if max(scales) / min(scales) > 100:
                self.recommendations.append({
                    'type': 'normalization',
                    'reason': 'Features have very different scales',
                    'action': 'Apply feature scaling (StandardScaler or MinMaxScaler) before modeling'
                })
        
        return self.recommendations

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = DataAnalyzer('devis.csv')
    
    # Load and clean data
    if analyzer.load_data():
        analyzer.clean_data()
        
        # Get overview
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"\nFirst 5 rows:")
        print(analyzer.df.head())
        print(f"\nDataset shape: {analyzer.df.shape}")
        print(f"\nColumn types:")
        print(analyzer.df.dtypes)
        
        # Missing values
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        missing = analyzer.analyze_missing_values()
        if len(missing) > 0:
            print(missing)
        else:
            print("No missing values found!")
        
        # Statistics
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        stats = analyzer.get_statistics()
        if stats is not None:
            print(stats)
        
        # Outliers
        print("\n" + "="*60)
        print("OUTLIER DETECTION")
        print("="*60)
        outliers = analyzer.detect_outliers()
        if outliers:
            for col, info in outliers.items():
                print(f"\n{col}:")
                print(f"  - Outliers: {info['count']} ({info['percentage']:.2f}%)")
                print(f"  - Normal range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        else:
            print("No significant outliers detected!")
        
        # Create visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        visualizations = analyzer.create_visualizations()
        print(f"✓ Created {len(visualizations)} visualizations")
        
        # PCA Analysis
        print("\n" + "="*60)
        print("PCA ANALYSIS")
        print("="*60)
        pca_results = analyzer.perform_pca()
        if pca_results:
            print(f"✓ PCA completed with {pca_results['n_components']} components")
            print(f"\nExplained variance by component:")
            for i, var in enumerate(pca_results['explained_variance']):
                print(f"  PC{i+1}: {var:.2%}")
            print(f"\nCumulative variance: {pca_results['cumulative_variance'][-1]:.2%}")
        
        # Generate recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        recommendations = analyzer.generate_recommendations()
        if recommendations:
            for i, rec in enumerate(recommendations[:10], 1):
                print(f"\n{i}. {rec['action']}")
                print(f"   Reason: {rec['reason']}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
