import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from flask import Flask, jsonify, render_template

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================
DB_USER = "postgres"
DB_PASSWORD = "123456789"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "sougui_Final"

OUTPUT_DIR = r"C:\Users\Hazem Bouchouicha\Downloads\Sougui_outputs"
RANDOM_STATE = 42
API_PORT = 5002

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

# =============================================================================
# LISTE BLANCHE — CATEGORIES METIER VALIDES
# =============================================================================
VALID_BUSINESS_CATEGORIES = {
    "accessories",
    "arts de la table",
    "bijoux",
    "coffrets cadeaux",
    "cosmetiques et soins",
    "fouta towel & blankets",
    "habillement",
    "handcrafted ceramics",
    "homeware",
    "la touche déco",
    "linge de maison",
    "luminaires",
    "nouveautés",
    "olive wood",
    "rangements",
    "rugs & kilim",
    "social projects",
    "gifts",
    "gastronomy",
}

EXCLUDED_CATEGORY_LABELS = {
    "unknown",
    "nan",
    "",
    "none",
    "null",
    "-",
    "--",
    "shop",
}

CATEGORY_NORMALIZATION_MAP = {
    "art de la table": "arts de la table",
    "arts de la table": "arts de la table",
    "tableware": "arts de la table",

    "clothing": "habillement",
    "habillement": "habillement",
    "apparel": "habillement",

    "beauty products": "cosmetiques et soins",
    "beauty": "cosmetiques et soins",
    "cosmetiques et soins": "cosmetiques et soins",

    "fashion accessories": "accessories",
    "accessories": "accessories",

    "home & decor": "homeware",
    "home decor": "homeware",
    "homeware": "homeware",

    "gift": "gifts",
    "gifts": "gifts",

    "fouta towel & blankets": "fouta towel & blankets",
    "olive wood": "olive wood",
    "rugs & kilim": "rugs & kilim",
    "handcrafted ceramics": "handcrafted ceramics",
    "bijoux": "bijoux",
    "linge de maison": "linge de maison",
    "coffrets cadeaux": "coffrets cadeaux",
    "luminaires": "luminaires",
    "la touche déco": "la touche déco",
    "rangements": "rangements",
    "social projects": "social projects",
    "nouveautés": "nouveautés",
    "gastronomy": "gastronomy",
}

# =============================================================================
# HELPERS
# =============================================================================
def safe_export(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_{ts}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✔  Exporté → {path}")
    return path


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✔  Figure  → {path}")


def rec_level_cat(score: float) -> str:
    if score >= 0.60:
        return "Fortement recommandee"
    if score >= 0.40:
        return "Recommandee"
    return "A surveiller"


def rec_level_prod(score: float) -> str:
    if score >= 0.75:
        return "Priorite haute"
    if score >= 0.55:
        return "Priorite moyenne"
    return "Priorite basse"


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_category(cat: str) -> str:
    cat = clean_text(cat)

    if cat in EXCLUDED_CATEGORY_LABELS:
        return ""

    cat = cat.replace("›", ">").replace("»", ">")
    cat = re.sub(r"\s*>\s*", " > ", cat)
    cat = re.sub(r"\s+", " ", cat).strip()

    if ">" in cat:
        cat = cat.split(">")[0].strip()

    cat = CATEGORY_NORMALIZATION_MAP.get(cat, cat)

    if cat not in VALID_BUSINESS_CATEGORIES:
        return ""

    return cat


def clean_price_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("TND", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def coverage_label(ratio: float) -> str:
    if ratio >= 1.50:
        return "Fort"
    if ratio >= 0.80:
        return "Equivalent"
    return "Faible"


# =============================================================================
# SECTION A — CONNEXION & CHARGEMENT
# =============================================================================
print("\n" + "=" * 70)
print("SECTION A — DATA PREPARATION")
print("=" * 70)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

try:
    with engine.connect():
        print("✔  Connexion PostgreSQL réussie.")
except Exception as e:
    raise ConnectionError(f"Erreur connexion DB : {e}")

fact = pd.read_sql('SELECT * FROM datawarehouse."FactConcurrents"', engine)
product = pd.read_sql('SELECT * FROM datawarehouse."DimProduct"', engine)
category = pd.read_sql("SELECT * FROM datawarehouse.dim_category", engine)
concurrent = pd.read_sql("SELECT * FROM datawarehouse.dim_concurrents", engine)

for tbl in [fact, product, category, concurrent]:
    tbl.columns = tbl.columns.str.lower()

print(
    f"\nTables chargées — fact:{fact.shape}  product:{product.shape}"
    f"  category:{category.shape}  concurrent:{concurrent.shape}"
)

for col in ["productpk", "categoryfk"]:
    if col in product.columns:
        product[col] = pd.to_numeric(product[col], errors="coerce")

for col in ["categorypk"]:
    if col in category.columns:
        category[col] = pd.to_numeric(category[col], errors="coerce")

for col in ["productfk", "concurrentfk"]:
    if col in fact.columns:
        fact[col] = pd.to_numeric(fact[col], errors="coerce")

for col in ["concurrentpk"]:
    if col in concurrent.columns:
        concurrent[col] = pd.to_numeric(concurrent[col], errors="coerce")

# =============================================================================
# SECTION A1 — CONSTRUCTION DATASET CONCURRENTS
# =============================================================================
fact_active = fact[fact["flag"].astype(str).str.lower() == "y"].copy()
product_active = product[product["flag"].astype(str).str.lower() == "y"].copy()

df = (
    fact_active
    .merge(concurrent, left_on="concurrentfk", right_on="concurrentpk", how="left")
    .merge(product_active, left_on="productfk", right_on="productpk", how="left")
    .merge(category, left_on="categoryfk", right_on="categorypk", how="left")
)

print(f"Shape après merge (lignes actives) : {df.shape}")
df = df[["product", "price_product", "concurrent", "category"]].copy()

# =============================================================================
# SECTION A2 — NETTOYAGE CONCURRENTS
# =============================================================================
print("\n--- Nettoyage concurrents ---")

df["price_product"] = clean_price_series(df["price_product"])
df["product"] = df["product"].apply(clean_text)
df["concurrent"] = df["concurrent"].apply(clean_text)
df["category_raw"] = df["category"].apply(clean_text)
df["category"] = df["category_raw"].apply(normalize_category)

n_before = len(df)
df = df.dropna(subset=["price_product"])
df = df[df["price_product"] > 0]
df = df[df["product"] != ""]
df = df[df["category"] != ""]
df = df.drop_duplicates()

print(f"  Lignes avant nettoyage : {n_before}")
print(f"  Lignes après nettoyage : {len(df)}")
print(f"  Catégories brutes uniques : {df['category_raw'].nunique()}")
print(f"  Catégories métier retenues : {df['category'].nunique()}")
print(f"  Valeurs manquantes restantes :\n{df[['product','price_product','concurrent','category']].isnull().sum()}")

# =============================================================================
# SECTION A3 — GESTION DES OUTLIERS
# =============================================================================
Q1 = df["price_product"].quantile(0.25)
Q3 = df["price_product"].quantile(0.75)
IQR = Q3 - Q1
lower_b = max(0, Q1 - 3 * IQR)
upper_b = Q3 + 3 * IQR

n_out = ((df["price_product"] < lower_b) | (df["price_product"] > upper_b)).sum()
print(f"\n  Outliers prix détectés (3×IQR) : {n_out}")

df["price_product"] = df["price_product"].clip(lower=lower_b, upper=upper_b)

print(
    f"  Prix après winsorization — min:{df['price_product'].min():.2f}"
    f"  max:{df['price_product'].max():.2f}"
    f"  mean:{df['price_product'].mean():.2f}"
)

# =============================================================================
# SECTION A4 — FEATURE ENGINEERING
# =============================================================================
print("\n--- Feature Engineering ---")

product_stats = (
    df.groupby(["product", "category"])
    .agg(
        avg_price=("price_product", "mean"),
        nb_concurrents=("concurrent", "nunique"),
        nb_occurrences=("product", "count"),
    )
    .reset_index()
)

cat_stats = (
    df.groupby("category")
    .agg(
        cat_total_products=("product", "count"),
        cat_avg_price=("price_product", "mean"),
        cat_nb_concurrents=("concurrent", "nunique"),
    )
    .reset_index()
)

product_stats = product_stats.merge(cat_stats, on="category", how="left")

max_cat = max(product_stats["cat_total_products"].max(), 1)
max_occ = max(product_stats["nb_occurrences"].max(), 1)
max_conc = max(product_stats["nb_concurrents"].max(), 1)

product_stats["feat_cat_popularity"] = product_stats["cat_total_products"] / max_cat
product_stats["feat_price_distance"] = abs(
    product_stats["avg_price"] - product_stats["cat_avg_price"]
) / (product_stats["cat_avg_price"] + 1e-6)
product_stats["feat_occurrence_score"] = product_stats["nb_occurrences"] / max_occ
product_stats["feat_concurrent_coverage"] = product_stats["nb_concurrents"] / max_conc

print(f"  Dataset enrichi : {product_stats.shape}")
print(f"  Catégories uniques : {product_stats['category'].nunique()}")
print(product_stats.head(3))

# =============================================================================
# SECTION A5 — SUPPRESSION DES CLASSES RARES
# =============================================================================
class_counts = product_stats["category"].value_counts()
valid_classes = class_counts[class_counts >= 5].index
product_stats = product_stats[product_stats["category"].isin(valid_classes)].copy()

print(f"\n  Classes retenues (≥5 exemples) : {len(valid_classes)}")
print(f"  Dataset final après filtrage   : {product_stats.shape}")
print("\n  Distribution des classes :")
print(product_stats["category"].value_counts())

# =============================================================================
# SECTION B — MODELES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION B — MODÈLES UTILISÉS")
print("=" * 70)

numeric_features = [
    "avg_price",
    "feat_cat_popularity",
    "feat_price_distance",
    "feat_occurrence_score",
    "feat_concurrent_coverage",
]

X = product_stats[["product"] + numeric_features].copy()
y = product_stats["category"].copy()

print(f"\nX shape : {X.shape}  |  Classes : {y.nunique()}")
print("  Déséquilibre géré via class_weight='balanced' dans les modèles")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf_product", TfidfVectorizer(max_features=500, ngram_range=(1, 2)), "product"),
        ("num", "passthrough", numeric_features),
    ],
    remainder="drop",
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n--- Modèle 1 : Random Forest + GridSearchCV ---")
pipeline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

param_grid_rf = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 20, 30],
    "classifier__min_samples_split": [2, 5],
}

grid_rf = GridSearchCV(
    pipeline_rf,
    param_grid_rf,
    cv=cv_strategy,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print(f"\n  Meilleurs paramètres RF : {grid_rf.best_params_}")
print(f"  Meilleur F1-weighted (CV) : {grid_rf.best_score_:.4f}")

y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)
classes_list = best_rf.classes_
y_test_bin = label_binarize(y_test, classes=classes_list)
roc_auc_rf = roc_auc_score(y_test_bin, y_prob_rf, average="weighted", multi_class="ovr")
cv_scores_rf = cross_val_score(best_rf, X, y, cv=cv_strategy, scoring="f1_weighted", n_jobs=-1)

print("\n  Classification Report — Random Forest :")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print(f"  ROC-AUC weighted (OvR) : {roc_auc_rf:.4f}")
print(f"  Cross-Val F1 (5-fold)  : {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

print("\n--- Modèle 2 : Logistic Regression + GridSearchCV ---")
pipeline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )),
])

param_grid_lr = {
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__penalty": ["l2"],
}

grid_lr = GridSearchCV(
    pipeline_lr,
    param_grid_lr,
    cv=cv_strategy,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_

print(f"\n  Meilleurs paramètres LR : {grid_lr.best_params_}")
print(f"  Meilleur F1-weighted (CV) : {grid_lr.best_score_:.4f}")

y_pred_lr = best_lr.predict(X_test)
y_prob_lr = best_lr.predict_proba(X_test)
roc_auc_lr = roc_auc_score(y_test_bin, y_prob_lr, average="weighted", multi_class="ovr")
cv_scores_lr = cross_val_score(best_lr, X, y, cv=cv_strategy, scoring="f1_weighted", n_jobs=-1)

print("\n  Classification Report — Logistic Regression :")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print(f"  ROC-AUC weighted (OvR) : {roc_auc_lr:.4f}")
print(f"  Cross-Val F1 (5-fold)  : {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

print("\n--- Comparaison des Modèles ---")
comparison = pd.DataFrame({
    "Modèle": ["Random Forest", "Logistic Regression"],
    "Accuracy": [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_lr)],
    "Precision (W)": [
        precision_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        precision_score(y_test, y_pred_lr, average="weighted", zero_division=0),
    ],
    "Recall (W)": [
        recall_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_lr, average="weighted", zero_division=0),
    ],
    "F1 (Weighted)": [
        f1_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_lr, average="weighted", zero_division=0),
    ],
    "ROC-AUC (W)": [roc_auc_rf, roc_auc_lr],
    "CV F1 Mean": [cv_scores_rf.mean(), cv_scores_lr.mean()],
    "CV F1 Std": [cv_scores_rf.std(), cv_scores_lr.std()],
})
print(comparison.to_string(index=False))
safe_export(comparison, "model_comparison.csv")

best_model_name = "Random Forest" if roc_auc_rf >= roc_auc_lr else "Logistic Regression"
best_model = best_rf if roc_auc_rf >= roc_auc_lr else best_lr
best_y_pred = y_pred_rf if roc_auc_rf >= roc_auc_lr else y_pred_lr
best_y_prob = y_prob_rf if roc_auc_rf >= roc_auc_lr else y_prob_lr
print(f"\n  ✔  Meilleur modèle sélectionné : {best_model_name}")

# =============================================================================
# VISUALISATIONS CLASSIFICATION
# =============================================================================
print("\n--- Génération des visualisations classification ---")

fig, ax = plt.subplots(figsize=(10, 5))
metrics_list = ["Accuracy", "Precision (W)", "Recall (W)", "F1 (Weighted)", "ROC-AUC (W)"]
x = np.arange(len(metrics_list))
w = 0.35
b1 = ax.bar(x - w / 2, comparison.iloc[0][metrics_list], w, label="Random Forest")
b2 = ax.bar(x + w / 2, comparison.iloc[1][metrics_list], w, label="Logistic Regression")
ax.set_xticks(x)
ax.set_xticklabels(metrics_list, rotation=20, ha="right")
ax.set_ylim(0, 1.15)
ax.set_title("Comparaison des métriques — RF vs LR", fontsize=14, fontweight="bold")
ax.legend()
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)
plt.tight_layout()
save_fig(fig, "viz_model_comparison_metrics.png")

fig, ax = plt.subplots(figsize=(16, 14))
cm = confusion_matrix(y_test, best_y_pred, labels=classes_list)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes_list,
    yticklabels=classes_list,
    ax=ax,
    linewidths=0.3,
)
ax.set_xlabel("Prédit", fontsize=12)
ax.set_ylabel("Réel", fontsize=12)
ax.set_title(f"Matrice de Confusion — {best_model_name}", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
save_fig(fig, "viz_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(10, 7))
top5 = y.value_counts().head(5).index.tolist()
colors_roc = plt.cm.tab10(np.linspace(0, 1, min(5, len(top5))))
for cls, col in zip(top5, colors_roc):
    idx = list(classes_list).index(cls)
    fpr, tpr, _ = roc_curve(y_test_bin[:, idx], best_y_prob[:, idx])
    auc_val = roc_auc_score(y_test_bin[:, idx], best_y_prob[:, idx])
    ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.2f})", color=col, lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(f"Courbes ROC — Top 5 catégories ({best_model_name})", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
save_fig(fig, "viz_roc_curves.png")

rf_clf = best_rf.named_steps["classifier"]
prep = best_rf.named_steps["preprocessor"]
tfidf_feats = prep.named_transformers_["tfidf_product"].get_feature_names_out()
all_feats = list(tfidf_feats) + numeric_features
feat_imp_df = pd.DataFrame({
    "feature": all_feats,
    "importance": rf_clf.feature_importances_,
}).sort_values("importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=feat_imp_df, x="importance", y="feature", ax=ax)
ax.set_title("Top 20 Features Importantes — Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
save_fig(fig, "viz_feature_importance.png")

fig, ax = plt.subplots(figsize=(12, 5))
y.value_counts().plot(kind="bar", ax=ax, edgecolor="black")
ax.set_title("Distribution des Catégories (Classes)", fontsize=14, fontweight="bold")
ax.set_xlabel("Catégorie")
ax.set_ylabel("Nombre de produits")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
save_fig(fig, "viz_class_distribution.png")

# =============================================================================
# PREDICTION SUR TOUT LE DATASET
# =============================================================================
print("\n--- Prédiction sur tout le dataset ---")
product_stats["predicted_category"] = best_model.predict(X)
product_stats["confidence_score"] = best_model.predict_proba(X).max(axis=1)
product_stats["correct_prediction"] = product_stats["predicted_category"] == product_stats["category"]

print(f"  Précision globale sur dataset complet : {product_stats['correct_prediction'].mean():.4f}")
print(product_stats[["product", "category", "predicted_category", "confidence_score"]].head(10))

# =============================================================================
# SECTION E — VRAIES CATEGORIES SOUGUI
# =============================================================================
print("\n" + "=" * 70)
print("SECTION E — EXTRACTION DES VRAIES CATEGORIES SOUGUI")
print("=" * 70)

sougui_products = product[product["flag"].astype(str).str.lower() == "y"].copy()
sougui_products = sougui_products.merge(
    category,
    left_on="categoryfk",
    right_on="categorypk",
    how="left"
)

sougui_products["product"] = sougui_products["product"].apply(clean_text)
sougui_products["category_raw"] = sougui_products["category"].apply(clean_text)
sougui_products["category"] = sougui_products["category_raw"].apply(normalize_category)

sougui_products = sougui_products[
    (sougui_products["product"] != "") &
    (sougui_products["category"] != "")
].copy()

sougui_products = sougui_products.drop_duplicates(subset=["productpk", "product", "category"])

sougui_cat_counts = (
    sougui_products.groupby("category")["productpk"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"productpk": "nb_produits_sougui"})
)

sougui_real_cats = set(sougui_cat_counts["category"].unique())

print(f"Total produits Sougui actifs retenus : {sougui_products['productpk'].nunique()}")
print(f"Nombre réel de catégories Sougui détectées : {len(sougui_real_cats)}")
print("\nCatégories Sougui retenues :")
print(sorted(sougui_real_cats))
print("\nTop catégories Sougui :")
print(sougui_cat_counts.to_string(index=False))

# =============================================================================
# SECTION F — BENCHMARK & OPPORTUNITES
# =============================================================================
print("\n" + "=" * 70)
print("BENCHMARK CONCURRENTIEL — COUVERTURE ET OPPORTUNITES")
print("=" * 70)

df_brut = df.copy().rename(columns={"price_product": "price_clean"})

conc_cat_stats = (
    df_brut.groupby("category")
    .agg(
        nb_produits_conc=("product", "nunique"),
        avg_price_conc=("price_clean", "mean"),
    )
    .reset_index()
)

conc_cat_stats = conc_cat_stats[conc_cat_stats["nb_produits_conc"] >= 5].copy()
all_conc_cats = set(conc_cat_stats["category"].unique())

print(f"\n[VERIFICATION 1] Catégories concurrents (>=5 produits) : {len(all_conc_cats)}")
print(f"Liste : {sorted(all_conc_cats)}")

print(f"\n[VERIFICATION 2] Catégories Sougui détectées : {len(sougui_real_cats)}")
print(f"Liste : {sorted(sougui_real_cats)}")

benchmark = conc_cat_stats.merge(sougui_cat_counts, on="category", how="left")
benchmark["nb_produits_sougui"] = benchmark["nb_produits_sougui"].fillna(0)

benchmark["coverage_ratio"] = np.where(
    benchmark["nb_produits_conc"] > 0,
    benchmark["nb_produits_sougui"] / benchmark["nb_produits_conc"],
    0,
)

benchmark["positionnement_couverture"] = benchmark["coverage_ratio"].apply(coverage_label)

ml_cat_stats = (
    product_stats.groupby("category")
    .agg(
        avg_confidence=("confidence_score", "mean"),
        avg_cat_popularity=("feat_cat_popularity", "mean"),
        avg_occurrence_score=("feat_occurrence_score", "mean"),
        avg_conc_coverage=("feat_concurrent_coverage", "mean"),
    )
    .reset_index()
)

benchmark = benchmark.merge(ml_cat_stats, on="category", how="left")

max_nb = max(benchmark["nb_produits_conc"].max(), 1)
max_price = max(benchmark["avg_price_conc"].fillna(0).max(), 1)

benchmark["score_volume_concurrent"] = benchmark["nb_produits_conc"] / max_nb
benchmark["score_prix_concurrent"] = benchmark["avg_price_conc"].fillna(0) / max_price

# Ici on ne cherche plus l'absence.
# On cherche les catégories où il y a le plus d'intérêt concurrent / valeur.
benchmark["opportunity_score"] = (
    0.35 * benchmark["score_volume_concurrent"]
    + 0.25 * benchmark["score_prix_concurrent"]
    + 0.20 * benchmark["avg_cat_popularity"].fillna(0)
    + 0.10 * benchmark["avg_confidence"].fillna(0)
    + 0.10 * benchmark["avg_conc_coverage"].fillna(0)
)

benchmark["niveau_opportunite"] = benchmark["opportunity_score"].apply(rec_level_cat)

benchmark = benchmark.sort_values(
    ["opportunity_score", "nb_produits_conc", "avg_price_conc"],
    ascending=[False, False, False]
)

print("\nTOP CATÉGORIES D'OPPORTUNITÉ :")
print(benchmark[[
    "category",
    "nb_produits_sougui",
    "nb_produits_conc",
    "coverage_ratio",
    "positionnement_couverture",
    "avg_price_conc",
    "opportunity_score",
    "niveau_opportunite"
]].to_string(index=False))

# Catégories à analyser en priorité
priority_cats = set(benchmark.head(8)["category"])

print(f"\nCatégories stratégiques à analyser en priorité : {sorted(priority_cats)}")

# =============================================================================
# VISUALISATIONS BENCHMARK
# =============================================================================
if not benchmark.empty:
    top_cats_viz = benchmark.head(15).copy()

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.barh(
        top_cats_viz["category"][::-1],
        top_cats_viz["opportunity_score"][::-1],
        edgecolor="black",
        height=0.6,
    )
    ax.set_xlabel("Opportunity score", fontsize=12)
    ax.set_title("Top catégories d'opportunité", fontsize=13, fontweight="bold")
    for i, val in enumerate(top_cats_viz["opportunity_score"][::-1]):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "viz_category_recommendations.png")

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.barh(
        top_cats_viz["category"][::-1],
        top_cats_viz["coverage_ratio"][::-1],
        edgecolor="black",
        height=0.6,
    )
    ax.set_xlabel("Coverage ratio (Sougui / Concurrents)", fontsize=12)
    ax.set_title("Benchmark de couverture par catégorie", fontsize=13, fontweight="bold")
    for i, val in enumerate(top_cats_viz["coverage_ratio"][::-1]):
        ax.text(val + 0.02, i, f"{val:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "viz_category_coverage_ratio.png")
else:
    print("\nAucune donnée benchmark à visualiser.")

# =============================================================================
# SECTION G — PRODUITS PHARES / OPPORTUNITES PRODUITS
# =============================================================================
print("\n" + "=" * 70)
print("PRODUITS PHARES CONCURRENTS — OPPORTUNITES D'ASSORTIMENT")
print("=" * 70)

prod_recommendation = product_stats[
    product_stats["category"].isin(priority_cats)
].copy()

if not prod_recommendation.empty:
    cat_score_map = benchmark[["category", "opportunity_score"]].rename(
        columns={"opportunity_score": "cat_score"}
    )

    prod_recommendation = prod_recommendation.merge(cat_score_map, on="category", how="left")
    prod_recommendation["cat_score"] = prod_recommendation["cat_score"].fillna(0.3)

    prod_recommendation["feat_price_attractiveness"] = (
        1 / (1 + prod_recommendation["feat_price_distance"])
    )

    # Ici on recommande des produits phares concurrents,
    # pas nécessairement "absents" chez Sougui.
    prod_recommendation["product_score"] = (
        0.30 * prod_recommendation["cat_score"]
        + 0.25 * prod_recommendation["confidence_score"]
        + 0.20 * prod_recommendation["feat_cat_popularity"]
        + 0.10 * prod_recommendation["feat_concurrent_coverage"]
        + 0.10 * prod_recommendation["feat_occurrence_score"]
        + 0.05 * prod_recommendation["feat_price_attractiveness"]
    )

    prod_recommendation = prod_recommendation.sort_values("product_score", ascending=False)
    prod_recommendation["niveau_recommandation"] = prod_recommendation["product_score"].apply(rec_level_prod)
else:
    prod_recommendation["cat_score"] = pd.Series(dtype=float)
    prod_recommendation["product_score"] = pd.Series(dtype=float)
    prod_recommendation["niveau_recommandation"] = pd.Series(dtype=str)

print(f"\nTotal produits phares analysables : {len(prod_recommendation)}")
print("\nTOP 30 PRODUITS PHARES / OPPORTUNITÉS :")
if not prod_recommendation.empty:
    print(prod_recommendation[[
        "product",
        "category",
        "avg_price",
        "nb_concurrents",
        "confidence_score",
        "cat_score",
        "product_score",
        "niveau_recommandation"
    ]].head(30).to_string(index=False))
else:
    print("Aucun produit analysable.")

# =============================================================================
# VISUALISATIONS PRODUITS
# =============================================================================
if not prod_recommendation.empty:
    top25_prods = prod_recommendation.head(25).copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(
        top25_prods["product"][::-1],
        top25_prods["product_score"][::-1],
        edgecolor="black",
        height=0.7,
    )
    ax.set_xlabel("Score Produit", fontsize=12)
    ax.set_title("TOP 25 Produits phares concurrents", fontsize=13, fontweight="bold")
    for i, (val, cat) in enumerate(zip(top25_prods["product_score"][::-1], top25_prods["category"][::-1])):
        ax.text(val + 0.003, i, f"{val:.2f} [{cat}]", va="center", fontsize=7)
    plt.tight_layout()
    save_fig(fig, "viz_product_recommendations_top25.png")

    fig, ax = plt.subplots(figsize=(13, 7))
    categories_priority = top25_prods["category"].unique()
    palette = plt.cm.tab20(np.linspace(0, 1, len(categories_priority)))
    for cat_name, color in zip(categories_priority, palette):
        subset = top25_prods[top25_prods["category"] == cat_name]
        ax.scatter(subset["avg_price"], subset["product_score"], label=cat_name, alpha=0.7, s=60, color=color)
    ax.set_xlabel("Prix moyen du produit (TND)", fontsize=12)
    ax.set_ylabel("Score de produit", fontsize=12)
    ax.set_title("Prix vs score — Produits phares concurrents", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    save_fig(fig, "viz_product_price_vs_score.png")
else:
    print("\nAucune donnée produit à visualiser.")

# =============================================================================
# EXPORTS
# =============================================================================
print("\n" + "=" * 70)
print("EXPORT DES RÉSULTATS FINAUX")
print("=" * 70)

safe_export(product_stats, "products_with_predictions.csv")
safe_export(benchmark, "recommendations_categories_sougui.csv")
safe_export(sougui_cat_counts, "sougui_categories_actuelles.csv")

if not prod_recommendation.empty:
    top100_products = prod_recommendation[[
        "product",
        "category",
        "avg_price",
        "nb_concurrents",
        "confidence_score",
        "product_score",
        "feat_cat_popularity",
        "feat_concurrent_coverage",
        "niveau_recommandation",
        "cat_score"
    ]].head(100)
else:
    top100_products = pd.DataFrame(columns=[
        "product",
        "category",
        "avg_price",
        "nb_concurrents",
        "confidence_score",
        "product_score",
        "feat_cat_popularity",
        "feat_concurrent_coverage",
        "niveau_recommandation",
        "cat_score"
    ])

safe_export(top100_products, "recommendations_produits_sougui_top100.csv")
safe_export(prod_recommendation, "recommendations_produits_sougui_complet.csv")

print("\n" + "=" * 70)
print("PIPELINE ML FINAL TERMINE AVEC SUCCES")
print("=" * 70)
print(f"\n  Modèle final utilisé        : {best_model_name}")
print(f"  ROC-AUC (test set)          : {roc_auc_rf if best_model_name == 'Random Forest' else roc_auc_lr:.4f}")
print(f"  F1 Weighted (test set)      : {f1_score(y_test, best_y_pred, average='weighted', zero_division=0):.4f}")
print(f"  Catégories classifiées      : {product_stats['category'].nunique()}")
print(f"  Catégories Sougui détectées : {len(sougui_real_cats)}")
print(f"  Catégories benchmarkées     : {len(benchmark)}")
print(f"  Catégories prioritaires     : {len(priority_cats)}")
print(f"  Produits phares analysés    : {len(prod_recommendation)}")
print(f"  Visualisations générées     : 10")
print(f"  Fichiers exportés dans      : {OUTPUT_DIR}")

def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({np.nan: None})
    return df.where(pd.notnull(df), None)

# =============================================================================
# FLASK API + DASHBOARD
# =============================================================================
app = Flask(__name__, template_folder="templates")


def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace({np.nan: None})
    return df.where(pd.notnull(df), None)


def build_api_output():
    global product_stats, prod_recommendation

    df_api = product_stats.copy()

    df_api["predicted_category"] = df_api["predicted_category"]
    df_api["confidence"] = df_api["confidence_score"]

    if not prod_recommendation.empty:
        extra_cols = [
            "product",
            "category",
            "avg_price",
            "nb_concurrents",
            "product_score",
            "niveau_recommandation",
        ]

        extra_df = prod_recommendation[extra_cols].copy()

        df_api = df_api.merge(
            extra_df,
            on=["product", "category", "avg_price", "nb_concurrents"],
            how="left"
        )

        df_api["niveau_recommandation"] = df_api["niveau_recommandation"].fillna("A surveiller")
    else:
        df_api["niveau_recommandation"] = "A surveiller"
        df_api["product_score"] = None

    df_api = df_api[
        [
            "product",
            "category",
            "predicted_category",
            "confidence",
            "avg_price",
            "nb_concurrents",
            "product_score",
            "niveau_recommandation",
        ]
    ]

    df_api = sanitize_for_json(df_api)
    return df_api.to_dict(orient="records")


@app.route("/")
def home():
    return render_template("index_sougui_dashboard.html")


@app.route("/api/predict")
def api_predict():
    data = build_api_output()
    return jsonify({
        "count": len(data),
        "predictions": data
    })


@app.route("/api/top-products")
def api_top_products():
    global prod_recommendation

    if prod_recommendation.empty:
        return jsonify({"data": []})

    top = prod_recommendation.head(50).copy()

    columns_needed = [
        "product",
        "category",
        "avg_price",
        "nb_concurrents",
        "confidence_score",
        "product_score",
        "niveau_recommandation"
    ]

    for col in columns_needed:
        if col not in top.columns:
            top[col] = None

    top = sanitize_for_json(top[columns_needed])

    return jsonify({
        "data": top.to_dict(orient="records")
    })


@app.route("/api/categories")
def api_categories():
    global benchmark

    if benchmark.empty:
        return jsonify({"data": []})

    columns_needed = [
        "category",
        "nb_produits_sougui",
        "nb_produits_conc",
        "coverage_ratio",
        "positionnement_couverture",
        "avg_price_conc",
        "opportunity_score",
        "niveau_opportunite"
    ]

    data = benchmark.copy()

    for col in columns_needed:
        if col not in data.columns:
            data[col] = None

    data = sanitize_for_json(data[columns_needed])

    return jsonify({
        "data": data.to_dict(orient="records")
    })


if __name__ == "__main__":
    print("\n🚀 Lancement Flask API...")
    app.run(debug=False, host="0.0.0.0", port=5006)