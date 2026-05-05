import os #lib bech creation folders
import joblib #lib saving trained models 
import pandas as pd #lib python manipulation mtaa dataframes

from sqlalchemy import create_engine #importi lfct connect_engine bch najm laccedi l postgres
from sklearn.pipeline import Pipeline #regroupement ML steps, maaneha mouch :
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X)
        #model = KNeighborsClassifier()
        #model.fit(X_scaled, y)
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier



os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


#hedhi conncetion maa bdd
engine = create_engine("postgresql://bi:bi123@localhost:5435/dw_wlh_final")



#hedhi tloadi data li bch ntraini beha m table directement
def load_base_data():
    query = """
    select
        "productFK",
        product,
        month,
        total_quantity,
        avg_price,
        revenue,
        "categoryFK",
        order_count
    from datawarehouse.ml_product_monthly_cat_orders
    """

    df = pd.read_sql(query, engine)

    #nadhef data li tablaa month :touskie month invalid , old nahehom
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"]).copy()
    df = df[df["month"].dt.year > 1900].copy()

    df["total_quantity"] = pd.to_numeric(df["total_quantity"], errors="coerce").fillna(0)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    df["order_count"] = pd.to_numeric(df["order_count"], errors="coerce").fillna(0)
    df["categoryFK"] = df["categoryFK"].fillna(-1).astype(str)

    df = df[df["productFK"].notna()].copy()
    df = df.sort_values(["productFK", "month"]).reset_index(drop=True)
    df["month_num"] = df["month"].dt.month

    return df



def prepare_training_data():
    df = load_base_data()

    df["next_month_revenue"] = df.groupby("productFK")["revenue"].shift(-1)
    model_df = df[df["next_month_revenue"].notna()].copy()

    threshold = model_df["next_month_revenue"].quantile(0.66)

    model_df["next_month_class"] = model_df["next_month_revenue"].apply(
        lambda x: "High" if x >= threshold else "Not High"
    )

    feature_cols = [
        "total_quantity",
        "avg_price",
        "revenue",
        "month_num",
        "categoryFK",
        "order_count"
    ]

    X = model_df[feature_cols].copy()
    X = pd.get_dummies(X, columns=["categoryFK"], drop_first=True)

    y = model_df["next_month_class"]

    return X, y



# def train_model():
#     X, y = prepare_training_data()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y
#     )

#     n_neighbors = 3

#     model = Pipeline([
#         ("scaler", StandardScaler()),
#         ("model", KNeighborsClassifier(n_neighbors=n_neighbors))
#     ])

#     mlflow.set_experiment("sougui_product_performance_prediction")

#     with mlflow.start_run(run_name="knn_product_performance"):
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
#         recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
#         f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

#         mlflow.log_param("model_name", "KNN")
#         mlflow.log_param("n_neighbors", n_neighbors)
#         mlflow.log_param("test_size", 0.2)
#         mlflow.log_param("rows_used", len(X))

#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("f1_score", f1)

#         joblib.dump(model, "models/product_model.pkl")
#         joblib.dump(X.columns.tolist(), "artifacts/training_columns.pkl")

#         mlflow.sklearn.log_model(model, "model")

#         print("Model trained successfully")
#         print(f"Rows used: {len(X)}")
#         print(f"Accuracy: {accuracy}")
#         print(f"F1-score: {f1}")
#         print("Model saved in: models/product_model.pkl")
#         print("Training columns saved in: artifacts/training_columns.pkl")




def train_model():
    X, y = prepare_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.set_tracking_uri("../mlruns")
    mlflow.set_experiment("sougui_product_performance_prediction")

    models = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "Random_Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])
    }

    best_model = None
    best_model_name = None
    best_f1 = 0

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("rows_used", len(X))

            if model_name == "KNN":
                mlflow.log_param("n_neighbors", 5)

            if model_name == "Random_Forest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("random_state", 42)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model")

            print("-----------------------------")
            print(f"Model: {model_name}")
            print(f"Accuracy: {accuracy}")
            print(f"F1-score: {f1}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name

    joblib.dump(best_model, "models/product_model.pkl")
    joblib.dump(X.columns.tolist(), "artifacts/training_columns.pkl")

    print("=============================")
    print(f"Best model: {best_model_name}")
    print(f"Best F1-score: {best_f1}")
    print("Best model saved in: models/product_model.pkl")
    print("Training columns saved in: artifacts/training_columns.pkl")
    
if __name__ == "__main__":
    train_model()