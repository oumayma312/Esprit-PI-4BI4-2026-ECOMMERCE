import os
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ============================================================
# 1) Paramètres de connexion
# ============================================================
USER = "postgres"
PASSWORD = "123456789"
HOST = "localhost"
PORT = "5432"
DATABASE = "sougui_Final"

# dossier de sortie
OUTPUT_DIR = r"C:\Users\Hazem Bouchouicha\Downloads\Sougui_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2) Fonctions utilitaires
# ============================================================
def safe_export_csv(df: pd.DataFrame, filename: str) -> str:
    """
    Exporte un fichier CSV.
    Si le fichier est déjà ouvert (Excel par exemple), crée une version horodatée.
    """
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{base}_{timestamp}{ext}"
        backup_path = os.path.join(OUTPUT_DIR, backup_name)
        df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        return backup_path


def assign_recommendation(score: float) -> str:
    if score >= 0.60:
        return "recommande_fortement"
    elif score >= 0.40:
        return "recommande"
    return "faible_priorite"


def assign_business_class(score: float) -> str:
    if score >= 0.70:
        return "opportunite_top"
    elif score >= 0.55:
        return "opportunite_interessante"
    elif score >= 0.40:
        return "a_surveiller"
    return "faible_opportunite"


def score_to_label(score: float) -> str:
    """
    Transformation du score métier en label ML
    """
    if score >= 0.60:
        return "high"
    elif score >= 0.45:
        return "medium"
    return "low"


def ml_to_score(label: str) -> float:
    """
    Conversion du label ML prédit en score numérique
    """
    if label == "high":
        return 1.0
    elif label == "medium":
        return 0.6
    return 0.3


# ============================================================
# 3) Connexion PostgreSQL
# ============================================================
engine = create_engine(
    f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
)

print("Engine créé avec succès")

try:
    with engine.connect() as conn:
        print("Connexion à PostgreSQL réussie.")
except Exception as e:
    print("Erreur de connexion :", repr(e))
    raise


# ============================================================
# 4) Lecture des tables
# ============================================================
fact = pd.read_sql('SELECT * FROM datawarehouse."FactConcurrents"', engine)
product = pd.read_sql('SELECT * FROM datawarehouse."DimProduct"', engine)
category = pd.read_sql('SELECT * FROM datawarehouse.dim_category', engine)
concurrent = pd.read_sql('SELECT * FROM datawarehouse.dim_concurrents', engine)

print("\n===== TABLES CHARGÉES =====")
print("Fact shape      :", fact.shape)
print("Product shape   :", product.shape)
print("Category shape  :", category.shape)
print("Concurrent shape:", concurrent.shape)


# ============================================================
# 5) Harmonisation des noms de colonnes
# ============================================================
fact.columns = fact.columns.str.lower()
product.columns = product.columns.str.lower()
category.columns = category.columns.str.lower()
concurrent.columns = concurrent.columns.str.lower()


# ============================================================
# 6) Merge des tables
# ============================================================
df = fact.merge(
    concurrent,
    left_on="concurrentfk",
    right_on="concurrentpk",
    how="left"
)

df = df.merge(
    product,
    left_on="productfk",
    right_on="productpk",
    how="left"
)

df = df.merge(
    category,
    left_on="categoryfk",
    right_on="categorypk",
    how="left"
)

print("\n===== DATASET APRÈS MERGE =====")
print("Shape:", df.shape)
print(df.head())


# ============================================================
# 7) Garder seulement les lignes actuelles
# ============================================================
df = df[df["flag_x"] == "y"]
df = df[df["flag_y"] == "y"]


# ============================================================
# 8) Garder les colonnes utiles
# ============================================================
df_final = df[["product", "price_product", "concurrent", "category"]].copy()

print("\n===== AVANT NETTOYAGE =====")
print("Shape avant nettoyage:", df_final.shape)
print(df_final.head())


# ============================================================
# 9) Nettoyage de price_product
# ============================================================
df_final["price_product"] = df_final["price_product"].astype(str)
df_final["price_product"] = df_final["price_product"].str.replace("TND", "", regex=False)
df_final["price_product"] = df_final["price_product"].str.strip()
df_final["price_product"] = df_final["price_product"].str.replace(",", ".", regex=False)

df_final["price_product"] = pd.to_numeric(df_final["price_product"], errors="coerce")

df_final = df_final.dropna(subset=["price_product"])
df_final = df_final[df_final["price_product"] > 0]
df_final = df_final.drop_duplicates()

df_final["product"] = df_final["product"].astype(str).str.lower().str.strip()
df_final["concurrent"] = df_final["concurrent"].astype(str).str.lower().str.strip()
df_final["category"] = df_final["category"].astype(str).str.lower().str.strip()

print("\n===== DATASET FINALE PROPRE =====")
print("Shape finale:", df_final.shape)
print(df_final.dtypes)
print(df_final.head())


# ============================================================
# 10) Analyse par produit
# ============================================================
product_analysis = df_final.groupby(["product", "category"]).agg(
    avg_price=("price_product", "mean"),
    nb_occurrences=("product", "count")
).reset_index()

print("\n===== ANALYSE PAR PRODUIT =====")
print("Shape product_analysis:", product_analysis.shape)
print(product_analysis.head(10))


# ============================================================
# 11) Analyse par catégorie
# ============================================================
category_analysis = df_final.groupby("category").agg(
    category_count=("product", "count"),
    category_avg_price=("price_product", "mean")
).reset_index()

print("\n===== ANALYSE PAR CATEGORY =====")
print(category_analysis.head(10))

product_analysis = product_analysis.merge(
    category_analysis,
    on="category",
    how="left"
)


# ============================================================
# 12) Calcul des scores business
# ============================================================
# score category popularity
max_cat_count = product_analysis["category_count"].max()
product_analysis["score_category_popularity"] = (
    product_analysis["category_count"] / max_cat_count
)

# score market share category
total_products = product_analysis["category_count"].sum()
product_analysis["score_market_share"] = (
    product_analysis["category_count"] / total_products
)

# score price attractiveness
product_analysis["price_distance"] = abs(
    product_analysis["avg_price"] - product_analysis["category_avg_price"]
)
product_analysis["score_price_attractiveness"] = (
    1 / (1 + product_analysis["price_distance"])
)

# score occurrence
max_occ = product_analysis["nb_occurrences"].max()
product_analysis["score_occurrence"] = (
    product_analysis["nb_occurrences"] / max_occ
)

# final business score
product_analysis["final_score"] = (
    0.35 * product_analysis["score_category_popularity"] +
    0.25 * product_analysis["score_price_attractiveness"] +
    0.20 * product_analysis["score_market_share"] +
    0.20 * product_analysis["score_occurrence"]
)

print("\n===== SCORE BUSINESS =====")
print(
    product_analysis[
        [
            "product",
            "category",
            "avg_price",
            "final_score"
        ]
    ].head(10)
)


# ============================================================
# 13) Labels business
# ============================================================
product_analysis["recommendation_level"] = product_analysis["final_score"].apply(assign_recommendation)
product_analysis["business_class"] = product_analysis["final_score"].apply(assign_business_class)

print("\n===== DISTRIBUTION BUSINESS CLASS =====")
print(product_analysis["business_class"].value_counts())


# ============================================================
# 14) Préparation ML (Hybrid)
# ============================================================
# هنا ML يتعلم من scoring
product_analysis["ml_label"] = product_analysis["final_score"].apply(score_to_label)

print("\n===== DISTRIBUTION ML LABEL =====")
print(product_analysis["ml_label"].value_counts())


# ============================================================
# 15) Modèle ML : Random Forest Classifier
# ============================================================
X = product_analysis[["product", "category", "avg_price"]]
y = product_analysis["ml_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("text_product", TfidfVectorizer(), "product"),
        ("text_category", TfidfVectorizer(), "category"),
        ("num_price", "passthrough", ["avg_price"])
    ]
)

ml_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

ml_model.fit(X_train, y_train)
y_pred = ml_model.predict(X_test)

print("\n===== RESULTATS DU MODELE ML =====")
print(classification_report(y_test, y_pred))


# ============================================================
# 16) Prédiction ML sur tout le dataset
# ============================================================
product_analysis["ml_prediction"] = ml_model.predict(X)
product_analysis["ml_score"] = product_analysis["ml_prediction"].apply(ml_to_score)

print("\n===== EXEMPLES ML =====")
print(
    product_analysis[
        [
            "product",
            "category",
            "final_score",
            "ml_label",
            "ml_prediction",
            "ml_score"
        ]
    ].head(10)
)


# ============================================================
# 17) Hybrid score
# ============================================================
# scoring business + ML
product_analysis["hybrid_score"] = (
    0.70 * product_analysis["final_score"] +
    0.30 * product_analysis["ml_score"]
)

product_analysis["hybrid_recommendation_level"] = product_analysis["hybrid_score"].apply(assign_recommendation)
product_analysis["hybrid_business_class"] = product_analysis["hybrid_score"].apply(assign_business_class)

print("\n===== DISTRIBUTION HYBRID BUSINESS CLASS =====")
print(product_analysis["hybrid_business_class"].value_counts())


# ============================================================
# 18) TOP PRODUITS RECOMMANDÉS (HYBRID)
# ============================================================
recommended_products = product_analysis.sort_values(
    by="hybrid_score",
    ascending=False
).copy()

print("\n===== TOP PRODUITS RECOMMANDÉS POUR SOUGUI (HYBRID) =====")
print(
    recommended_products[
        [
            "product",
            "category",
            "avg_price",
            "nb_occurrences",
            "category_count",
            "category_avg_price",
            "final_score",
            "ml_prediction",
            "ml_score",
            "hybrid_score",
            "hybrid_recommendation_level"
        ]
    ].head(20)
)


# ============================================================
# 19) TOP CATÉGORIES RECOMMANDÉES (HYBRID)
# ============================================================
top_categories = recommended_products.groupby("category").agg(
    nb_products=("product", "count"),
    avg_hybrid_score=("hybrid_score", "mean"),
    avg_category_price=("category_avg_price", "mean")
).reset_index()

top_categories = top_categories.sort_values(
    by="avg_hybrid_score",
    ascending=False
)

print("\n===== TOP CATEGORIES RECOMMANDÉES (HYBRID) =====")
print(top_categories.head(10))


# ============================================================
# 20) PRODUITS AVEC PRIX LES PLUS ATTRACTIFS
# ============================================================
best_price_fit = recommended_products.sort_values(
    by="score_price_attractiveness",
    ascending=False
).drop_duplicates(subset=["product", "category"])

print("\n===== PRODUITS AVEC PRIX LES PLUS ATTRACTIFS =====")
print(
    best_price_fit[
        [
            "product",
            "category",
            "avg_price",
            "category_avg_price",
            "score_price_attractiveness",
            "hybrid_score"
        ]
    ].head(20)
)


# ============================================================
# 21) FICHIERS CLEAN
# ============================================================
clean_reco = recommended_products[
    [
        "product",
        "category",
        "avg_price",
        "hybrid_score",
        "hybrid_recommendation_level"
    ]
].copy()

clean_categories = top_categories[
    [
        "category",
        "nb_products",
        "avg_hybrid_score"
    ]
].copy()

clean_price = best_price_fit[
    [
        "product",
        "category",
        "avg_price",
        "score_price_attractiveness"
    ]
].copy()


# ============================================================
# 22) EXPORT DES RÉSULTATS
# ============================================================
exported_files = {
    "recommended_products_sougui_hybrid.csv": safe_export_csv(
        recommended_products, "recommended_products_sougui_hybrid.csv"
    ),
    "top_categories_sougui_hybrid.csv": safe_export_csv(
        top_categories, "top_categories_sougui_hybrid.csv"
    ),
    "best_price_fit_products_sougui_hybrid.csv": safe_export_csv(
        best_price_fit, "best_price_fit_products_sougui_hybrid.csv"
    ),
    "recommended_products_hybrid_clean.csv": safe_export_csv(
        clean_reco, "recommended_products_hybrid_clean.csv"
    ),
    "top_categories_hybrid_clean.csv": safe_export_csv(
        clean_categories, "top_categories_hybrid_clean.csv"
    ),
    "best_price_fit_hybrid_clean.csv": safe_export_csv(
        clean_price, "best_price_fit_hybrid_clean.csv"
    ),
}

print("\n===== FICHIERS EXPORTÉS =====")
for logical_name, real_path in exported_files.items():
    print(f"{logical_name} -> {real_path}")