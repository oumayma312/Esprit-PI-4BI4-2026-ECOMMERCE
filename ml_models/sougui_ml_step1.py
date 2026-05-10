import pandas as pd
import os

from dw_postgres import get_dw_schema, make_sqlalchemy_engine
from pipeline.pt_artifact import save_pt_artifact

engine = make_sqlalchemy_engine()
DW_SCHEMA = get_dw_schema()

query = f"""
SELECT 
    fv."productFK",
    dp.product,
    SUM(
        COALESCE(
            NULLIF(
                REGEXP_REPLACE(
                    REPLACE(REGEXP_REPLACE(CAST(fv.quantity AS TEXT), '[^0-9,.-]', '', 'g'), ',', '.'),
                    '\\.(?=.*\\.)',
                    '',
                    'g'
                ),
                ''
            ),
            '0'
        )::numeric
    ) AS total_quantity,
    AVG(
        COALESCE(
            NULLIF(
                REGEXP_REPLACE(
                    REPLACE(REGEXP_REPLACE(CAST(fv.price AS TEXT), '[^0-9,.-]', '', 'g'), ',', '.'),
                    '\\.(?=.*\\.)',
                    '',
                    'g'
                ),
                ''
            ),
            '0'
        )::numeric
    ) AS avg_price,
    SUM(
        COALESCE(
            NULLIF(
                REGEXP_REPLACE(
                    REPLACE(REGEXP_REPLACE(CAST(fv.total_ttc AS TEXT), '[^0-9,.-]', '', 'g'), ',', '.'),
                    '\\.(?=.*\\.)',
                    '',
                    'g'
                ),
                ''
            ),
            '0'
        )::numeric
    ) AS revenue
FROM {DW_SCHEMA}."FactVentee" fv
JOIN {DW_SCHEMA}."DimProduct" dp
    ON fv."productFK" = dp."productPK"
WHERE dp.product IS NOT NULL
  AND dp.product NOT IN ('Unknown', 'TOTAL')
GROUP BY 
    fv."productFK",
    dp.product
ORDER BY revenue DESC
"""

df = pd.read_sql(query, engine)

df["product_class"] = pd.qcut(
    df["revenue"],
    q=3,
    labels=["Low", "Medium", "High"]
)


X = df[["total_quantity", "avg_price"]]
y = df["product_class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# decision tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("decision tree Accuracy:", accuracy)

# random forest 
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)

print("Random Forest Accuracy:", rf_accuracy)
#logistic regression 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_y_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_y_pred)

print("Logistic Regression Accuracy:", lr_accuracy)
#comparing 3 models 
results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Logistic Regression"],
    "Accuracy": [accuracy, rf_accuracy, lr_accuracy]
})

print(results)
# confusion matrix for the best model
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_y_pred, labels=["Low", "Medium", "High"])

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, rf_y_pred))

comparison = df.loc[X_test.index, [
    "productFK", "product", "total_quantity", "avg_price", "revenue", "product_class"
]].copy()

comparison["predicted_class"] = rf_y_pred

comparison["is_correct"] = comparison["product_class"] == comparison["predicted_class"]

print(comparison[["product", "product_class", "predicted_class", "is_correct"]].head(20))

wrong_predictions = comparison[comparison["is_correct"] == False]

print("\nNumber of wrong predictions:", len(wrong_predictions))
print(wrong_predictions[["product", "product_class", "predicted_class"]].head(20))

# retrain the best model on all labeled data
final_rf_model = RandomForestClassifier(random_state=42)
final_rf_model.fit(X, y)

# predict all current products
df["predicted_class"] = final_rf_model.predict(X)

# simple business action
action_map = {
    "High": "Promote",
    "Medium": "Monitor",
    "Low": "Review"
}

df["suggested_action"] = df["predicted_class"].map(action_map)

# final business view
final_view = df[[
    "productFK",
    "product",
    "total_quantity",
    "avg_price",
    "predicted_class",
    "suggested_action"
]]

print("\nFINAL VIEW")
print(final_view.head(20))

print("\nPredicted class counts:")
print(df["predicted_class"].value_counts())

high_products = final_view[final_view["predicted_class"] == "High"]
medium_products = final_view[final_view["predicted_class"] == "Medium"]
low_products = final_view[final_view["predicted_class"] == "Low"]

print("\nHIGH PRODUCTS")
print(high_products.head(10))

print("\nMEDIUM PRODUCTS")
print(medium_products.head(10))

print("\nLOW PRODUCTS")
print(low_products.head(10))

# save prediction table into PostgreSQL
final_view.to_sql(
    "ml_product_predictions",
    engine,
    schema=DW_SCHEMA,
    if_exists="replace",
    index=False
)

print(f"\nTable saved to PostgreSQL: {DW_SCHEMA}.ml_product_predictions")

if os.getenv("EXPORT_PT", "0") in {"1", "true", "True"}:
    artifact_path = os.getenv(
        "MODEL_ARTIFACT_PATH",
        os.path.join(os.getenv("OUTPUT_DIR", "."), "sougui_step1_model.pt"),
    )
    metrics = {
        "decision_tree_accuracy": float(accuracy),
        "random_forest_accuracy": float(rf_accuracy),
        "logistic_regression_accuracy": float(lr_accuracy),
        "rows": int(df.shape[0]),
    }
    export_info = save_pt_artifact(
        final_rf_model,
        artifact_path=artifact_path,
        metadata={
            "model_name": "sougui_step1",
            "framework": "sklearn",
            "measures": metrics,
        },
    )
    print("MODEL_PT_EXPORTED:", export_info["artifact_path"])