from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError, MlValidationError


class CampaignSuccessPredictionService:
    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository

    @cached_property
    def _bundle(self) -> dict[str, object]:
        frame = self.repository.load_campaign_frame()
        working = frame.copy()
        if working.empty:
            raise MlServiceError("Campaign snapshot is empty.")

        feature_columns = ["reach", "impressions", "frequency", "price"]
        working["engagement_score"] = (
            0.35 * working["views"].fillna(0)
            + 0.3 * working["result"].fillna(0) * 10
            + 0.2 * working["impressions"].fillna(0) / 100
            + 0.15 * working["reach"].fillna(0) / 100
        )
        threshold = float(working["engagement_score"].median())
        working["target"] = (working["engagement_score"] >= threshold).astype(int)

        classifier = RandomForestClassifier(
            n_estimators=220,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )
        views_model = RandomForestRegressor(n_estimators=220, max_depth=7, random_state=42)
        result_model = RandomForestRegressor(n_estimators=220, max_depth=7, random_state=42)

        accuracy = None
        if working["target"].nunique() > 1 and len(working) >= 24:
            X_train, X_test, y_train, y_test = train_test_split(
                working[feature_columns],
                working["target"],
                test_size=0.2,
                random_state=42,
                stratify=working["target"],
            )
            eval_model = RandomForestClassifier(
                n_estimators=180,
                max_depth=6,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced_subsample",
            )
            eval_model.fit(X_train, y_train)
            accuracy = float(accuracy_score(y_test, eval_model.predict(X_test)))

        classifier.fit(working[feature_columns], working["target"])
        views_model.fit(working[feature_columns], working["views"].fillna(0))
        result_model.fit(working[feature_columns], working["result"].fillna(0))

        training_probabilities = classifier.predict_proba(working[feature_columns])[:, 1]

        return {
            "frame": working,
            "classifier": classifier,
            "views_model": views_model,
            "result_model": result_model,
            "feature_columns": feature_columns,
            "accuracy": accuracy,
            "training_probabilities": training_probabilities,
        }

    def predict(self, reach: int, impressions: int, frequency: float, budget_price: float) -> dict[str, object]:
        if reach < 0 or impressions < 0 or frequency < 0 or budget_price < 0:
            raise MlValidationError("Campaign inputs must be positive or zero.")

        bundle = self._bundle
        classifier: RandomForestClassifier = bundle["classifier"]  # type: ignore[assignment]
        views_model: RandomForestRegressor = bundle["views_model"]  # type: ignore[assignment]
        result_model: RandomForestRegressor = bundle["result_model"]  # type: ignore[assignment]
        feature_columns: list[str] = bundle["feature_columns"]  # type: ignore[assignment]
        accuracy = bundle["accuracy"]
        training_probabilities: np.ndarray = bundle["training_probabilities"]  # type: ignore[assignment]

        payload = pd.DataFrame(
            [{"reach": reach, "impressions": impressions, "frequency": frequency, "price": budget_price}],
            columns=feature_columns,
        )
        success_probability = float(classifier.predict_proba(payload)[0][1])
        predicted_views = max(0.0, float(views_model.predict(payload)[0]))
        predicted_result = max(0.0, float(result_model.predict(payload)[0]))
        percentile = float((training_probabilities <= success_probability).mean())

        if success_probability >= 0.75:
            status = "strong"
            verdict = "High success potential"
        elif success_probability >= 0.5:
            status = "balanced"
            verdict = "Balanced performance outlook"
        else:
            status = "watch"
            verdict = "Risk of underperforming"

        insights = [
            f"Projected view volume is around {predicted_views:.0f}.",
            f"Projected result score is around {predicted_result:.1f}.",
            f"This campaign sits in the {percentile * 100:.0f}th percentile of the historical sample.",
        ]
        if accuracy is not None:
            insights.append(f"Offline classifier accuracy is {accuracy * 100:.1f}% on the available snapshot.")

        return {
            "workflow": "campaignSuccess",
            "headline": verdict,
            "summary": "The score combines historical campaign outcomes, price level, impressions and reach intensity.",
            "status": status,
            "confidence": round(success_probability, 4),
            "metrics": [
                {"label": "Success probability", "value": f"{success_probability * 100:.1f}%", "tone": "primary"},
                {"label": "Projected views", "value": f"{predicted_views:.0f}", "tone": "success"},
                {"label": "Historical percentile", "value": f"{percentile * 100:.0f}th", "tone": "neutral"},
            ],
            "insights": insights,
            "result": {
                "successProbability": round(success_probability, 4),
                "predictedViews": round(predicted_views, 0),
                "predictedResult": round(predicted_result, 2),
                "historicalPercentile": round(percentile, 4),
            },
        }

