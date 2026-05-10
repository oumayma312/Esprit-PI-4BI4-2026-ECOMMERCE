from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError, MlValidationError
from .utils import clamp


class SupplierClassificationService:
    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository

    @cached_property
    def _bundle(self) -> dict[str, object]:
        frame = self.repository.load_supplier_frame()
        if len(frame) < 8:
            raise MlServiceError("Supplier data is not rich enough to create segments.")

        numeric_features = [
            "total_quantity",
            "total_spend",
            "purchase_count",
            "average_unit_price",
            "active_products",
            "spend_per_purchase",
        ]
        categorical_features = ["governorate", "city"]

        cluster_count = 4 if len(frame) >= 32 else 3 if len(frame) >= 15 else 2
        preprocessor = ColumnTransformer(
            transformers=[
                ("geo", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                ("numeric", StandardScaler(), numeric_features),
            ],
            remainder="drop",
        )
        transformed = preprocessor.fit_transform(frame[categorical_features + numeric_features])

        model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        labels = model.fit_predict(transformed)
        frame = frame.copy()
        frame["cluster"] = labels

        distances = model.transform(transformed)
        assigned_distance = distances[np.arange(len(frame)), labels]
        max_distance = float(assigned_distance.max()) if len(assigned_distance) else 1.0

        cluster_profiles = (
            frame.groupby("cluster")
            .agg(
                suppliers=("supplierfk", "size"),
                avg_spend=("total_spend", "mean"),
                avg_quantity=("total_quantity", "mean"),
                avg_purchases=("purchase_count", "mean"),
                avg_price=("average_unit_price", "mean"),
                avg_products=("active_products", "mean"),
            )
            .reset_index()
        )
        cluster_profiles["profileName"] = cluster_profiles.apply(self._profile_name, axis=1)
        cluster_profiles["nextAction"] = cluster_profiles.apply(self._next_action, axis=1)

        return {
            "frame": frame,
            "preprocessor": preprocessor,
            "model": model,
            "profiles": cluster_profiles.set_index("cluster"),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "max_distance": max(max_distance, 1.0),
        }

    def _profile_name(self, row: pd.Series) -> str:
        if row["avg_spend"] >= row[["avg_spend", "avg_quantity", "avg_purchases"]].max():
            return "Strategic spend partner"
        if row["avg_quantity"] >= row[["avg_spend", "avg_quantity", "avg_purchases"]].max():
            return "Volume supplier"
        if row["avg_purchases"] >= row[["avg_spend", "avg_quantity", "avg_purchases"]].max():
            return "Frequent operational supplier"
        return "Emerging specialist"

    def _next_action(self, row: pd.Series) -> str:
        if row["avg_spend"] > 10_000:
            return "Prioritize contract review and long-term pricing."
        if row["avg_quantity"] > 100:
            return "Secure capacity and optimize replenishment cadence."
        if row["avg_purchases"] > 8:
            return "Automate recurring orders and payment follow-up."
        return "Develop the relationship with targeted onboarding."

    def predict(
        self,
        governorate: str,
        city: str,
        total_quantity: float,
        total_spend: float,
        purchase_count: int,
        average_unit_price: float,
        active_products: int,
    ) -> dict[str, object]:
        if total_quantity < 0 or total_spend < 0 or purchase_count < 0 or average_unit_price < 0 or active_products < 0:
            raise MlValidationError("Supplier values must be positive or zero.")

        if len(governorate.strip()) < 2 or len(city.strip()) < 2:
            raise MlValidationError("Governorate and city are required.")

        bundle = self._bundle
        frame: pd.DataFrame = bundle["frame"]  # type: ignore[assignment]
        preprocessor: ColumnTransformer = bundle["preprocessor"]  # type: ignore[assignment]
        model: KMeans = bundle["model"]  # type: ignore[assignment]
        profiles: pd.DataFrame = bundle["profiles"]  # type: ignore[assignment]
        max_distance: float = bundle["max_distance"]  # type: ignore[assignment]
        numeric_features: list[str] = bundle["numeric_features"]  # type: ignore[assignment]
        categorical_features: list[str] = bundle["categorical_features"]  # type: ignore[assignment]

        payload = pd.DataFrame(
            [
                {
                    "governorate": governorate.strip().title(),
                    "city": city.strip().title(),
                    "total_quantity": total_quantity,
                    "total_spend": total_spend,
                    "purchase_count": purchase_count,
                    "average_unit_price": average_unit_price,
                    "active_products": active_products,
                    "spend_per_purchase": total_spend / purchase_count if purchase_count else total_spend,
                }
            ]
        )

        transformed = preprocessor.transform(payload[categorical_features + numeric_features])
        cluster = int(model.predict(transformed)[0])
        distance = float(model.transform(transformed)[0][cluster])
        confidence = clamp(1 - distance / max_distance, 0.1, 0.99)

        profile = profiles.loc[cluster]
        peers = frame[frame["cluster"] == cluster].copy()
        peers["peerScore"] = (
            (peers["total_spend"] - total_spend).abs()
            + (peers["purchase_count"] - purchase_count).abs() * 100
            + (peers["average_unit_price"] - average_unit_price).abs() * 20
        )
        peers = peers.sort_values("peerScore").head(3)

        status = "strong" if confidence >= 0.72 else "balanced" if confidence >= 0.5 else "watch"

        return {
            "workflow": "supplierClassification",
            "headline": f"Segment match: {profile['profileName']}",
            "summary": "This supplier most closely matches an existing operational cluster from the purchase warehouse.",
            "status": status,
            "confidence": round(confidence, 4),
            "metrics": [
                {"label": "Similarity", "value": f"{confidence * 100:.1f}%", "tone": "primary"},
                {"label": "Avg spend in cluster", "value": f"{float(profile['avg_spend']):.0f} TND", "tone": "success"},
                {"label": "Avg active products", "value": f"{float(profile['avg_products']):.1f}", "tone": "neutral"},
            ],
            "insights": [
                profile["nextAction"],
                f"Typical cluster cadence is {float(profile['avg_purchases']):.1f} purchases.",
                f"Typical cluster quantity sits around {float(profile['avg_quantity']):.1f} units.",
            ],
            "result": {
                "clusterId": cluster,
                "profileName": profile["profileName"],
                "nextAction": profile["nextAction"],
                "clusterAverages": {
                    "totalSpend": round(float(profile["avg_spend"]), 2),
                    "totalQuantity": round(float(profile["avg_quantity"]), 2),
                    "purchaseCount": round(float(profile["avg_purchases"]), 2),
                    "averageUnitPrice": round(float(profile["avg_price"]), 2),
                },
                "peerSuppliers": [
                    {
                        "supplier": row["supplier"],
                        "governorate": row["governorate"],
                        "city": row["city"],
                        "totalSpend": round(float(row["total_spend"]), 2),
                    }
                    for _, row in peers.iterrows()
                ],
            },
        }

