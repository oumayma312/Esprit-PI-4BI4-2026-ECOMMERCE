import pandas as pd
from sqlalchemy import create_engine

host = "localhost"
port = "5435"
database = "dw_wlh_final"
user = "bi"
password = "bi123"

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")

query = """
SELECT 
    fv."productFK",
    dp.product,
    SUM(fv.quantity) AS total_quantity,
    AVG(fv.price) AS avg_price,
    SUM(fv.total_ttc) AS revenue
FROM datawarehouse."FactVentee" fv
JOIN datawarehouse."DimProduct" dp
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
    schema="datawarehouse",
    if_exists="replace",
    index=False
)

print("\nTable saved to PostgreSQL: datawarehouse.ml_product_predictions")