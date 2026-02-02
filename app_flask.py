"""
Flask Web Application for Exploratory Data Analysis
Professional EDA Dashboard with visualizations and recommendations
All content in English
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'eda_secret_key_2026'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Global variable to store current analyzer
current_analyzer = None
current_filename = None

class DataAnalyzer:
    """Enhanced Data Analyzer for Flask"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = []
        self.categorical_cols = []
        self.recommendations = []
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare data"""
        # Convert all column names to strings to avoid AttributeError
        self.df.columns = [str(col) for col in self.df.columns]
        
        numeric_patterns = ['pu ht', 'quantite', 'prix total ht', 'montant total ht', 
                          'total ht', 'total ttc', 'timbre']
        
        for col in self.df.columns:
            if any(pattern in col.lower() for pattern in numeric_patterns):
                try:
                    self.df[col] = self.df[col].astype(str).str.replace('"', '').str.replace(',', '.')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    pass
        
        if 'date' in self.df.columns:
            try:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            except:
                pass
        
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str).str.replace('"', '').str.strip()
        
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
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
        """Analyze missing values"""
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum(),
            'Missing %': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
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
        """Perform PCA analysis - optimized version"""
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
        
        n_components = min(3, len(self.numeric_cols), len(df_numeric))
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        pca_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pca_cols)
        
        return {
            'pca': pca,
            'pca_df': pca_df,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'n_components': n_components,
            'feature_names': self.numeric_cols
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

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def load_data(file):
    """Load data from uploaded file with enhanced format support"""
    try:
        filename_lower = file.filename.lower()
        
        # Handle CSV files with multiple delimiters and encodings
        if filename_lower.endswith('.csv') or filename_lower.endswith('.txt'):
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            separators = [';', ',', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, sep=sep, encoding=encoding, on_bad_lines='skip')
                        if len(df.columns) > 1 and len(df) > 0:  # Valid dataframe
                            print(f"Successfully loaded with encoding={encoding}, separator={sep}")
                            return df
                    except Exception as e:
                        continue
            
            # Fallback: let pandas detect automatically
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding='utf-8')
                return df
            except:
                file.seek(0)
                df = pd.read_csv(file)
                return df
        
        # Handle Excel files (.xlsx, .xls, .xlsm, .xlsb)
        elif filename_lower.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            file.seek(0)
            df = pd.read_excel(file, engine='openpyxl' if filename_lower.endswith('.xlsx') else None)
            return df
        
        else:
            print(f"Unsupported file format: {file.filename}")
            return None
            
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Home page"""
    global current_analyzer, current_filename
    
    # Try to load default file
    if current_analyzer is None:
        try:
            df = pd.read_csv('devis.csv', sep=';', encoding='utf-8')
            current_analyzer = DataAnalyzer(df)
            current_filename = 'devis.csv'
        except:
            pass
    
    has_data = current_analyzer is not None
    return render_template('index.html', has_data=has_data, filename=current_filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    global current_analyzer, current_filename
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.xlsm', '.xlsb', '.txt'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid file type. Allowed: {', '.join(allowed_extensions)}'}), 400
        
        df = load_data(file)
        if df is None or len(df) == 0:
            return jsonify({'error': 'Failed to load file. Please ensure it is a valid CSV or Excel file with data.'}), 400
        
        current_analyzer = DataAnalyzer(df)
        current_filename = secure_filename(file.filename)
        
        return jsonify({'success': True, 'filename': current_filename})
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

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
    """API endpoint for distribution charts"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    distributions = []
    for col in current_analyzer.numeric_cols[:6]:
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
    
    return jsonify(distributions)

@app.route('/api/boxplots')
def api_boxplots():
    """API endpoint for boxplots"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    outliers_info = current_analyzer.detect_outliers()
    boxplots = []
    
    for col in current_analyzer.numeric_cols[:6]:
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
    
    return jsonify(boxplots)

@app.route('/api/correlation')
def api_correlation():
    """API endpoint for correlation matrix"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if len(current_analyzer.numeric_cols) > 1:
        corr_matrix = current_analyzer.df[current_analyzer.numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
        img = fig_to_base64(fig)
        
        return jsonify({'chart': img})
    
    return jsonify({'chart': None})

@app.route('/api/categorical')
def api_categorical():
    """API endpoint for categorical charts"""
    if current_analyzer is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    categorical_charts = []
    
    for col in current_analyzer.categorical_cols[:5]:
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
                                   c=range(len(pca_df)), cmap='viridis',
                                   s=50, alpha=0.6, edgecolors='black')
                ax.set_xlabel(f"PC1 ({pca_results['explained_variance'][0]:.1%})", fontsize=12)
                ax.set_ylabel(f"PC2 ({pca_results['explained_variance'][1]:.1%})", fontsize=12)
                ax.set_title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Sample Index')
                charts['pca_2d'] = fig_to_base64(fig)
                print("2D PCA plot created")
            except Exception as e:
                print(f"Error creating 2D PCA plot: {e}")
                raise
        
        print("PCA analysis completed successfully")
        return jsonify({
            'available': True,
            'n_components': pca_results['n_components'],
            'explained_variance': pca_results['explained_variance'],
            'cumulative_variance': pca_results['cumulative_variance'],
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
