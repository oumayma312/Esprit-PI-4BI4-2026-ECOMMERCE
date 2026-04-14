import os
import sys
from pathlib import Path

from flask import Flask, render_template, send_from_directory, request, jsonify

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
            ],
        )

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
