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
    """Class to perform comprehensive EDA on datasets"""
    
    def __init__(self, file_path):
        """Initialize the analyzer with a data file"""
        self.file_path = file_path
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.recommendations = []
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            # Try different encodings and separators
            try:
                self.df = pd.read_csv(self.file_path, sep=';', encoding='utf-8')
            except:
                self.df = pd.read_csv(self.file_path, encoding='latin-1')
            
            print(f"✓ Data loaded successfully!")
            print(f"  Dataset dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Convert numeric columns
        numeric_patterns = ['pu ht', 'quantite', 'prix total ht', 'montant total ht', 
                          'total ht', 'total ttc', 'timbre']
        
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in numeric_patterns):
                try:
                    # Remove quotes and convert to float
                    self.df[col] = self.df[col].astype(str).str.replace('"', '').str.replace(',', '.')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass
        
        # Convert date column
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            except:
                pass
        
        # Remove quotes from all string columns
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.replace('"', '').str.strip()
        
        # Identify column types
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove date from categorical if present
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
        """Analyze missing values"""
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum(),
            'Missing Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
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
        """Detect outliers using IQR method"""
        outliers_info = {}
        
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
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
        
        # 4. Correlation heatmap
        if len(self.numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = self.df[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
            visualizations['correlation'] = fig
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
    
    def perform_pca(self):
        """Perform PCA analysis with 2D and 3D visualizations"""
        pca_results = {}
        
        if len(self.numeric_cols) < 2:
            return None
        
        # Prepare data for PCA
        df_numeric = self.df[self.numeric_cols].dropna()
        
        if len(df_numeric) < 3:
            return None
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
        
        # Determine number of components
        n_components = min(3, len(self.numeric_cols), len(df_numeric))
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        # Create DataFrame with principal components
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pca_cols)
        
        # Store results
        pca_results['explained_variance'] = pca.explained_variance_ratio_
        pca_results['cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)
        pca_results['components'] = pca.components_
        pca_results['pca_df'] = pca_df
        pca_results['n_components'] = n_components
        
        # Create visualizations
        visualizations = {}
        
        # 1. Scree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
               color='steelblue', edgecolor='black')
        ax.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 
                'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title('Scree Plot - Explained Variance by Component', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        for i, v in enumerate(pca.explained_variance_ratio_):
            ax.text(i + 1, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')
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
        ax.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax.legend()
        visualizations['cumulative_variance'] = fig
        plt.close()
        
        # 3. 2D PCA plot
        if n_components >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], 
                               c=range(len(pca_df)), cmap='viridis', 
                               s=50, alpha=0.6, edgecolors='black')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.set_title('PCA - First Two Principal Components', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Sample Index')
            visualizations['pca_2d'] = fig
            plt.close()
        
        # 4. 3D Interactive PCA plot
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
        
        # 5. Component loadings heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=pca_cols,
            index=self.numeric_cols
        )
        sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax, cbar_kws={'label': 'Loading'})
        ax.set_title('PCA Component Loadings', fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Components', fontsize=12)
        ax.set_ylabel('Original Features', fontsize=12)
        visualizations['loadings'] = fig
        plt.close()
        
        pca_results['visualizations'] = visualizations
        
        # Generate PCA recommendations
        if pca_results['cumulative_variance'][0] > 0.8:
            self.recommendations.append({
                'type': 'dimensionality',
                'reason': 'First component explains >80% variance',
                'action': 'Consider using only the first principal component for analysis to reduce dimensionality'
            })
        
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
