import os
import sys
from pathlib import Path
from unittest import result


from flask import Flask, render_template, send_from_directory, request, jsonify
from classification_concurrents_CORRIGE import run_concurrent_classification
from sougui_ml_binary_evaluation import run_sougui_ml_binary_evaluation

# Add parent directory to path for chatbot imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .notebook_assets import get_notebook_view
except Exception:  # pragma: no cover
    from notebook_assets import get_notebook_view

try:
    from chatbot import MarketingChatbot
    from chatbot.config import AppConfig
    CHATBOT_AVAILABLE = True
except Exception as e:
    print(f"Warning: Chatbot import failed: {e}")
    CHATBOT_AVAILABLE = False
    MarketingChatbot = None
    AppConfig = None
    
    CONCURRENT_VIEW_CACHE = None


def create_app() -> Flask:
    app = Flask(__name__)

    img_dir = Path(__file__).resolve().parent / "img"
    
    # Initialize chatbot with correct data directory
    chatbot_instance = None
    if CHATBOT_AVAILABLE:
        try:
            # Get the correct path to datawarehouse (parent/datawarehouse)
            project_root = Path(__file__).resolve().parent.parent
            data_dir = str(project_root / "datawarehouse")
            
            config = AppConfig(data_dir=data_dir)
            chatbot_instance = MarketingChatbot(config)
            print(f"✓ Chatbot initialized successfully with data_dir: {data_dir}")
        except Exception as e:
            print(f"Error: Chatbot initialization failed: {e}")
            import traceback
            traceback.print_exc()
    
    app.chatbot = chatbot_instance

    @app.get("/img/<path:filename>")
    def img(filename: str):
        return send_from_directory(img_dir, filename)

    @app.get("/")
    def home():
        return render_template(
            "home.html",
            pages=[
                {
                    "title": "Supplier Classification",
                    "subtitle": "Data prep + understanding + clustering visuals",
                    "href": "/supplier-classification",
                },
                {
                    "title": "Best Time To Sell",
                    "subtitle": "Data prep + understanding + forecasting visuals",
                    "href": "/best-time-to-sell",
                },
                {
                    "title": "Best Time To Promote",
                    "subtitle": "Data prep + understanding + promo visuals",
                    "href": "/best-time-to-promote",
                },
                {
                    "title": "Campaign Success Prediction",
                    "subtitle": "EDA + classification + regression + clustering",
                    "href": "/campaign-success-prediction",
                },
                {
                    "title": "Campaign Time Series Forecasting",
                    "subtitle": "Time series prep + ARIMA/SARIMA + XGBoost forecast",
                    "href": "/campaign-time-series-forecasting",
                },
                {
                    "title": "Concurrent Classification",
                    "subtitle": "classification + category recommendations + product recommendations",
                    "href": "/concurrent-classification"
                },
                {
                    "title": "Sougui ML Binary Evaluation",
                    "subtitle": "Binary revenue prediction + model comparison",
                    "href": "/sougui-ml-binary-evaluation",
                },
            ],
        )

    @app.get("/sougui-ml-binary-evaluation")
    def sougui_ml_binary_evaluation_page():
        view = run_sougui_ml_binary_evaluation()
        return render_template("notebook_view.html", **view)
    
    @app.get("/supplier-classification")
    def supplier_classification():
        view = get_notebook_view("supplier")
        return render_template("notebook_view.html", **view)

    @app.get("/best-time-to-sell")
    def best_time_to_sell():
        view = get_notebook_view("sell")
        return render_template("notebook_view.html", **view)

    @app.get("/best-time-to-promote")
    def best_time_to_promote():
        view = get_notebook_view("promote")
        return render_template("notebook_view.html", **view)

    @app.get("/campaign-success-prediction")
    def campaign_success_prediction():
        view = get_notebook_view("campaign_success")
        return render_template("notebook_view.html", **view)

    @app.get("/campaign-time-series-forecasting")
    def campaign_time_series_forecasting():
        view = get_notebook_view("campaign_forecasting")
        return render_template("notebook_view.html", **view)

    @app.get("/chatbot")
    def chatbot_page():
        return render_template("chatbot.html")
    
    ##################################################################
    @app.get("/concurrent-classification")
    def concurrent_classification():
        result = run_concurrent_classification()

        categories_html = """
        <table>
        <thead>
            <tr>
            <th>Category</th>
            <th>Score</th>
            <th>Level</th>
            </tr>
        </thead>
        <tbody>
        """

        for c in result["category_recommendations"]:
            categories_html += f"""
            <tr>
            <td>{c.get('category', '')}</td>
            <td>{c.get('recommendation_score', 0):.3f}</td>
            <td>{c.get('niveau_recommandation', '')}</td>
            </tr>
            """

        categories_html += """
        </tbody>
        </table>
        """

        products_html = """
        <table>
        <thead>
            <tr>
            <th>Product</th>
            <th>Category</th>
            <th>Score</th>
            </tr>
        </thead>
        <tbody>
        """

        for p in result["product_recommendations"]:
            products_html += f"""
            <tr>
            <td>{p.get('product', '')}</td>
            <td>{p.get('category', '')}</td>
            <td>{p.get('product_score', 0):.3f}</td>
            </tr>
            """

        products_html += """
        </tbody>
        </table>
        """

        view = {
            "title": "Concurrent Classification",
            "subtitle": "Notebook visualizations",
            "updated_at": "2026-04-15",
            "models_used": ["Random Forest", "Logistic Regression"],
            "hints": [
                "This page shows competitor classification, missing categories, and product recommendations for Sougui."
            ],
            "third_tab_label": "Result",
            "third_section_title": "Result",
            "extra_tabs": [],
            "section_intro": {
                "data_prep": "<h3>Section A — Data Preparation & Feature Engineering</h3><p>Competitor product data is cleaned, price values are standardized, outliers are treated, and product/category features are engineered before training.</p>",
                "model_understanding": "<h3>Section B — Model Understanding</h3><p>Two models are compared: Random Forest and Logistic Regression. Random Forest is selected as the best model based on the evaluation metrics.</p>",
                "models": f"""
                    <h3>Final Results</h3>
                    <ul>
                    <li><strong>Best model:</strong> {result['best_model_name']}</li>
                    <li><strong>ROC-AUC:</strong> {result['roc_auc']:.4f}</li>
                    <li><strong>F1 Weighted:</strong> {result['f1_weighted']:.4f}</li>
                    <li><strong>Missing categories:</strong> {result['missing_categories']}</li>
                    <li><strong>Recommended categories:</strong> {result['recommended_categories_count']}</li>
                    <li><strong>Recommended products:</strong> {result['recommended_products_count']}</li>
                    </ul>
                """,
            },
            "sections": {
                "data_prep": {
                    "steps": [
                        {
                            "title": "Overview",
                            "markdown": [],
                            "plotly": [],
                            "html": [],
                            "images": [
                                "/static/outputs/concurrents/viz_class_distribution.png"
                            ],
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
                            "html": [],
                            "images": [
                                "/static/outputs/concurrents/viz_model_comparison_metrics.png",
                                "/static/outputs/concurrents/viz_confusion_matrix.png",
                                "/static/outputs/concurrents/viz_roc_curves.png",
                                "/static/outputs/concurrents/viz_feature_importance.png"
                            ],
                            "text": []
                        }
                    ]
                },
                "models": {
                    "steps": [
                        {
                            "title": "Category recommendations",
                            "markdown": [],
                            "plotly": [],
                            "html": [categories_html],
                            "images": [
                                "/static/outputs/concurrents/viz_category_recommendations.png",
                                "/static/outputs/concurrents/viz_category_volume.png"
                            ],
                            "text": []
                        },
                        {
                            "title": "Product recommendations",
                            "markdown": [],
                            "plotly": [],
                            "html": [products_html],
                            "images": [
                                "/static/outputs/concurrents/viz_product_recommendations_top25.png",
                                "/static/outputs/concurrents/viz_product_price_vs_score.png",
                                "/static/outputs/concurrents/viz_product_heatmap_by_category.png"
                            ],
                            "text": []
                        }
                    ]
                }
            }
        }

        return render_template("notebook_view.html", **view)
    ##################################################################
    
   
    @app.post("/api/chat")
    def chat():
        if not app.chatbot:
            return jsonify({"reply": "Chatbot non disponible. Verifiez la configuration."}), 503
        
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        
        if not message:
            return jsonify({"reply": "Ecris une question pour commencer."}), 400
        
        try:
            reply = app.chatbot.respond(message)
            return jsonify({"reply": reply})
        except Exception as e:
            return jsonify({"reply": f"Erreur: {str(e)}"}), 500

    return app


if __name__ == "__main__":
    # Run with: python webapp/app.py
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
