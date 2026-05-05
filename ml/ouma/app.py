from flask import Flask, jsonify, request, render_template
import pandas as pd
from sqlalchemy import create_engine
import joblib
import requests

app = Flask(__name__)

# DB connection
engine = create_engine("postgresql://bi:bi123@host.docker.internal:5435/dw_wlh_final")

# Load saved model
model = joblib.load("models/product_model.pkl")
training_columns = joblib.load("artifacts/training_columns.pkl")


def load_base_data():
    query = """
    SELECT
        "productFK",
        product,
        month,
        total_quantity,
        avg_price,
        revenue,
        "categoryFK",
        order_count
    FROM datawarehouse.ml_product_monthly_cat_orders
    """
    df = pd.read_sql(query, engine)

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


def build_latest_predictions():
    df = load_base_data()

    latest_df = (
        df.sort_values(["productFK", "month"])
        .groupby("productFK", as_index=False)
        .tail(1)
        .copy()
    )

    latest_X = latest_df[
        ["total_quantity", "avg_price", "revenue", "month_num", "categoryFK", "order_count"]
    ].copy()

    latest_X = pd.get_dummies(latest_X, columns=["categoryFK"], drop_first=True)
    latest_X = latest_X.reindex(columns=training_columns, fill_value=0)

    latest_df["predicted_next_month_class"] = model.predict(latest_X)
    latest_df["suggested_action"] = latest_df["predicted_next_month_class"].map({
        "High": "Promote",
        "Not High": "Monitor"
    })

    final_view = latest_df[[
        "productFK",
        "product",
        "month",
        "total_quantity",
        "avg_price",
        "revenue",
        "order_count",
        "predicted_next_month_class",
        "suggested_action"
    ]].copy()

    final_view["month"] = final_view["month"].astype(str)
    return final_view


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

    
import requests
from flask import request, jsonify

@app.route("/api/vision/match_live", methods=["POST"])
def vision_match_proxy():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file received"}), 400

        fastapi_response = requests.post(
            "http://127.0.0.1:8000/match_live",
            files={
                "image": (file.filename, file.stream, file.mimetype)
            }
        )

        return jsonify(fastapi_response.json()), fastapi_response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/latest")
def api_predict_latest():
    final_view = build_latest_predictions()
    return jsonify({
        "count": int(len(final_view)),
        "predicted_class_counts": final_view["predicted_next_month_class"].value_counts().to_dict(),
        "predictions": final_view.to_dict(orient="records")
    })


@app.route("/api/predict/save", methods=["GET", "POST"])
def api_predict_save():
    final_view = build_latest_predictions()

    final_view.to_sql(
        "ml_product_next_month_binary_predictions",
        engine,
        schema="datawarehouse",
        if_exists="replace",
        index=False
    )

    return jsonify({
        "message": "Predictions saved successfully",
        "table": "datawarehouse.ml_product_next_month_binary_predictions",
        "rows_saved": int(len(final_view))
    })


@app.route("/api/predictions")
def api_predictions():
    query = """
    SELECT *
    FROM datawarehouse.ml_product_next_month_binary_predictions
    ORDER BY "productFK"
    """
    saved_df = pd.read_sql(query, engine)
    if "month" in saved_df.columns:
        saved_df["month"] = saved_df["month"].astype(str)

    return jsonify({
        "count": int(len(saved_df)),
        "rows": saved_df.to_dict(orient="records")
    })


@app.route("/api/predict/one", methods=["POST"])
def api_predict_one():
    data = request.get_json(force=True)

    input_df = pd.DataFrame([{
        "total_quantity": float(data.get("total_quantity", 0)),
        "avg_price": float(data.get("avg_price", 0)),
        "revenue": float(data.get("revenue", 0)),
        "month_num": int(data.get("month_num", 1)),
        "categoryFK": str(data.get("categoryFK", -1)),
        "order_count": float(data.get("order_count", 0))
    }])

    input_df = pd.get_dummies(input_df, columns=["categoryFK"], drop_first=True)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    predicted_class = model.predict(input_df)[0]
    suggested_action = "Promote" if predicted_class == "High" else "Monitor"

    return jsonify({
        "predicted_next_month_class": predicted_class,
        "suggested_action": suggested_action
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)