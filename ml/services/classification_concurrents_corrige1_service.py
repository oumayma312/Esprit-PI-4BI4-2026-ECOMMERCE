from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError, MlValidationError
from .utils import clamp, normalise_series


class CompetitorClassificationService:
    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository

    @cached_property
    def _bundle(self) -> dict[str, object]:
        frame = self.repository.load_competitor_frame()
        working = frame.copy()
        category_counts = working["category"].value_counts()
        valid_categories = category_counts[category_counts >= 3].index
        working = working[working["category"].isin(valid_categories)].copy()

        if working.empty or working["category"].nunique() < 2:
            raise MlServiceError("Competitor classification data is not rich enough to build the model.")

        numeric_features = ["avg_price", "competitor_count", "observation_count"]
        features = working[["product"] + numeric_features].copy()
        labels = working["category"].copy()

        accuracy = None
        if len(working) >= 30 and labels.value_counts().min() >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
            evaluation_model = self._build_pipeline(numeric_features)
            evaluation_model.fit(X_train, y_train)
            accuracy = float(accuracy_score(y_test, evaluation_model.predict(X_test)))

        model = self._build_pipeline(numeric_features)
        model.fit(features, labels)

        category_stats = (
            working.groupby("category")
            .agg(
                products=("product", "size"),
                average_price=("avg_price", "mean"),
                average_competitors=("competitor_count", "mean"),
                average_observations=("observation_count", "mean"),
            )
            .reset_index()
        )
        category_stats["demand_score"] = normalise_series(category_stats["products"])
        category_stats["price_score"] = normalise_series(category_stats["average_price"].fillna(0))
        category_stats["pressure_score"] = normalise_series(category_stats["average_competitors"].fillna(0))
        category_stats["opportunity_score"] = (
            0.45 * category_stats["demand_score"]
            + 0.2 * category_stats["price_score"]
            + 0.35 * category_stats["pressure_score"]
        )

        return {
            "frame": working,
            "model": model,
            "numeric_features": numeric_features,
            "category_stats": category_stats.set_index("category"),
            "accuracy": accuracy,
        }

    def _build_pipeline(self, numeric_features: list[str]) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("product_name", TfidfVectorizer(max_features=800, ngram_range=(1, 2)), "product"),
                ("numeric", StandardScaler(), numeric_features),
            ],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1_200, class_weight="balanced")),
            ]
        )

    def predict(self, product_name: str, average_price: float, competitor_count: int) -> dict[str, object]:
        cleaned_name = product_name.strip()
        if len(cleaned_name) < 2:
            raise MlValidationError("Product name must contain at least two characters.")

        if average_price < 0:
            raise MlValidationError("Average price cannot be negative.")

        if competitor_count < 0:
            raise MlValidationError("Competitor count cannot be negative.")

        bundle = self._bundle
        model: Pipeline = bundle["model"]  # type: ignore[assignment]
        category_stats: pd.DataFrame = bundle["category_stats"]  # type: ignore[assignment]
        frame: pd.DataFrame = bundle["frame"]  # type: ignore[assignment]
        accuracy = bundle["accuracy"]

        input_frame = pd.DataFrame(
            [
                {
                    "product": cleaned_name,
                    "avg_price": average_price,
                    "competitor_count": competitor_count,
                    "observation_count": max(1, competitor_count),
                }
            ]
        )
        probabilities = model.predict_proba(input_frame)[0]
        classes = model.classes_
        best_index = int(np.argmax(probabilities))
        predicted_category = str(classes[best_index])
        confidence = float(probabilities[best_index])

        category_profile = category_stats.loc[predicted_category]
        opportunity_score = float(category_profile["opportunity_score"])
        if confidence >= 0.8 and opportunity_score >= 0.65:
            status = "strong"
            status_label = "High potential"
        elif confidence >= 0.6:
            status = "balanced"
            status_label = "Promising"
        else:
            status = "watch"
            status_label = "Needs review"

        related = (
            frame[frame["category"] == predicted_category]
            .assign(priceGap=lambda df: (df["avg_price"] - average_price).abs())
            .sort_values(["priceGap", "competitor_count"])
            .head(3)
        )

        insights = [
            f"Predicted segment leans toward {predicted_category}.",
            f"Observed market pressure is around {category_profile['average_competitors']:.1f} competitors per product.",
            f"Average catalog price for this category is {category_profile['average_price']:.2f} TND.",
        ]
        if accuracy is not None:
            insights.append(f"Offline validation accuracy is {accuracy * 100:.1f}% on the available snapshot.")

        return {
            "workflow": "competitorClassification",
            "headline": f"{predicted_category} looks like the best fit",
            "summary": f"{status_label} signal detected for this competitor product based on naming and market pricing.",
            "status": status,
            "confidence": round(confidence, 4),
            "metrics": [
                {"label": "Confidence", "value": f"{confidence * 100:.1f}%", "tone": "primary"},
                {"label": "Opportunity", "value": f"{opportunity_score * 100:.0f}/100", "tone": "success"},
                {"label": "Avg market price", "value": f"{category_profile['average_price']:.2f} TND", "tone": "neutral"},
            ],
            "insights": insights,
            "result": {
                "predictedCategory": predicted_category,
                "recommendationLevel": status_label,
                "opportunityScore": round(opportunity_score, 4),
                "averageCategoryPrice": round(float(category_profile["average_price"]), 2),
                "expectedCompetitorPressure": round(float(category_profile["average_competitors"]), 2),
                "relatedProducts": [
                    {
                        "product": row["product"],
                        "averagePrice": round(float(row["avg_price"]), 2),
                        "competitorCount": int(row["competitor_count"]),
                    }
                    for _, row in related.iterrows()
                ],
            },
        }

