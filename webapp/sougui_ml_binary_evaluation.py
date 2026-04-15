import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def run_sougui_ml_binary_evaluation():
    DB_HOST = "localhost"
    DB_PORT = "5435"
    DB_NAME = "dw_wlh_final"
    DB_USER = "bi"
    DB_PASSWORD = "bi123"
    RANDOM_STATE = 42

    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_model.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
    knn_model.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"],
        "Accuracy": [lr_acc, knn_acc, dt_acc, rf_acc],
        "Accuracy_%": [100 * lr_acc, 100 * knn_acc, 100 * dt_acc, 100 * rf_acc]
    }).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    best_model_name = results.loc[0, "Model"]
    best_model_acc = results.loc[0, "Accuracy"]

    if best_model_name == "Logistic Regression":
        best_pred = lr_model.predict(X_test)
    elif best_model_name == "K-Nearest Neighbors":
        best_pred = knn_model.predict(X_test)
    elif best_model_name == "Decision Tree":
        best_pred = dt_model.predict(X_test)
    else:
        best_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, best_pred)
    class_report_html = pd.DataFrame(classification_report(y_test, best_pred, output_dict=True)).transpose().to_html()
    results_html = results.to_html(index=False)

    cm_html = pd.DataFrame(
        cm,
        index=["Actual High", "Actual Not High"],
        columns=["Pred High", "Pred Not High"]
    ).to_html()

    majority_class = y_test.value_counts().index[0]
    baseline_acc = y_test.value_counts().iloc[0] / len(y_test)

    summary_html = f"""
    <table>
      <thead>
        <tr>
          <th>Best Model</th>
          <th>Best Accuracy</th>
          <th>Baseline Accuracy</th>
          <th>Improvement (pp)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{best_model_name}</td>
          <td>{100 * best_model_acc:.2f}%</td>
          <td>{100 * baseline_acc:.2f}%</td>
          <td>{100 * (best_model_acc - baseline_acc):.2f}</td>
        </tr>
      </tbody>
    </table>
    """

    target_html = f"""
    <table>
      <thead>
        <tr>
          <th>Threshold</th>
          <th>High</th>
          <th>Not High</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{threshold:.2f}</td>
          <td>next_month_revenue ≥ {threshold:.2f}</td>
          <td>next_month_revenue &lt; {threshold:.2f}</td>
        </tr>
      </tbody>
    </table>
    """

    feature_html = """
    <table>
      <thead>
        <tr>
          <th>Retained Features</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>total_quantity</td></tr>
        <tr><td>avg_price</td></tr>
        <tr><td>revenue</td></tr>
        <tr><td>month_num</td></tr>
        <tr><td>categoryFK</td></tr>
        <tr><td>order_count</td></tr>
      </tbody>
    </table>
    """

    return {
        "title": "Sougui ML Binary Evaluation",
        "subtitle": "Notebook visualizations",
        "updated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "models_used": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
        "hints": [
            "Binary classification: predict whether next-month revenue is High or Not High.",
            "Business action: High → Promote, Not High → Monitor."
        ],
        "third_tab_label": "Evaluation & Conclusion",
        "third_section_title": "Evaluation & Conclusion",
        "extra_tabs": [],
        "section_intro": {
            "data_prep": f"""
                <h3>Section A — Data Preparation</h3>
                <p>
                  Data is loaded from <strong>datawarehouse.ml_product_monthly_cat_orders</strong>,
                  cleaned, sorted by product and month, and transformed into a binary next-month target.
                </p>
                <p>
                  Cleaned dataset size: <strong>{df.shape[0]}</strong> rows.<br>
                  Rows with known future revenue: <strong>{len(model_df)}</strong>.
                </p>
            """,
            "model_understanding": """
                <h3>Section B — Model Understanding</h3>
                <p>Four models are compared:</p>
                <ul>
                  <li>Logistic Regression</li>
                  <li>K-Nearest Neighbors</li>
                  <li>Decision Tree</li>
                  <li>Random Forest</li>
                </ul>
            """,
            "models": f"""
                <h3>Section D / E / F — Final Evaluation</h3>
                <p>
                  Best model: <strong>{best_model_name}</strong><br>
                  Best accuracy: <strong>{100 * best_model_acc:.2f}%</strong><br>
                  Baseline accuracy: <strong>{100 * baseline_acc:.2f}%</strong><br>
                  Improvement: <strong>{100 * (best_model_acc - baseline_acc):.2f} pp</strong>
                </p>
            """,
        },
        "sections": {
            "data_prep": {
                "steps": [
                    {
                        "title": "Target definition",
                        "markdown": [],
                        "plotly": [],
                        "html": [target_html],
                        "images": [],
                        "text": []
                    },
                    {
                        "title": "Features",
                        "markdown": [],
                        "plotly": [],
                        "html": [feature_html],
                        "images": [],
                        "text": []
                    }
                ]
            },
            "model_understanding": {
                "steps": [
                    {
                        "title": "Model comparison",
                        "markdown": [],
                        "plotly": [],
                        "html": [results_html],
                        "images": [],
                        "text": []
                    }
                ]
            },
            "models": {
                "steps": [
                    {
                        "title": "Summary",
                        "markdown": [],
                        "plotly": [],
                        "html": [summary_html],
                        "images": [],
                        "text": []
                    },
                    {
                        "title": "Confusion Matrix",
                        "markdown": [],
                        "plotly": [],
                        "html": [cm_html],
                        "images": [],
                        "text": []
                    },
                    {
                        "title": "Classification Report",
                        "markdown": [],
                        "plotly": [],
                        "html": [class_report_html],
                        "images": [],
                        "text": []
                    }
                ]
            }
        }
    }


if __name__ == "__main__":
    view = run_sougui_ml_binary_evaluation()
    print(view["title"])