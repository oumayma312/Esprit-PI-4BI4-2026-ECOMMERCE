import os
from pathlib import Path

from flask import Flask, render_template, send_from_directory

try:
    from .notebook_assets import get_notebook_view
except Exception:  # pragma: no cover
    from notebook_assets import get_notebook_view


def create_app() -> Flask:
    app = Flask(__name__)

    img_dir = Path(__file__).resolve().parent / "img"

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

    return app


if __name__ == "__main__":
    # Run with: python webapp/app.py
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
