from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError


def _iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5, non_negative: bool = True) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or np.isclose(iqr, 0):
        return pd.Series(False, index=series.index)

    lower = q1 - multiplier * iqr
    if non_negative:
        lower = max(0, lower)
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def _make_rfm_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    clean = series.fillna(series.median()).astype(float)
    quantiles = clean.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    scores = []
    for value in clean:
        score = 1
        for threshold in quantiles:
            if value > threshold:
                score += 1
        if not higher_is_better:
            score = 6 - score
        scores.append(int(min(5, max(1, score))))
    return pd.Series(scores, index=series.index, dtype=int)


def _score_single(value: float, reference: pd.Series, higher_is_better: bool) -> int:
    quantiles = reference.fillna(reference.median()).astype(float).quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    score = 1
    for threshold in quantiles:
        if value > threshold:
            score += 1
    if not higher_is_better:
        score = 6 - score
    return int(min(5, max(1, score)))


def _risk_band(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


class CustomerChurnAnalyticsService:
    churn_threshold_days = 90
    high_risk_threshold = 0.7

    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository

    @cached_property
    def _bundle(self) -> dict[str, object]:
        frame = self.repository.load_customer_behavior_frame()
        if len(frame) < 30:
            raise MlServiceError("Customer sales history is not rich enough to build churn analytics.")

        working = frame.copy()
        reference_date = pd.to_datetime(working["sale_date"], errors="coerce").max()
        if pd.isna(reference_date):
            raise MlServiceError("Customer sales history does not contain valid dates.")

        for column in ["revenue", "quantity", "price"]:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)

        outlier_mask = pd.Series(False, index=working.index)
        for column in ["revenue", "quantity", "price"]:
            outlier_mask |= _iqr_outlier_mask(working[column], multiplier=1.5, non_negative=True)
        working = working.loc[~outlier_mask].copy()

        customer_features = (
            working.groupby(["customer_fk", "customer_name", "customer_code"], dropna=False)
            .agg(
                recency=("sale_date", lambda values: int((reference_date - values.max()).days)),
                first_purchase=("sale_date", "min"),
                last_purchase=("sale_date", "max"),
                frequency=("transaction_id", "count"),
                monetary_total=("revenue", "sum"),
                monetary_mean=("revenue", "mean"),
                monetary_std=("revenue", "std"),
                quantity_total=("quantity", "sum"),
                quantity_mean=("quantity", "mean"),
                product_diversity=("product_fk", "nunique"),
                channel=("channel_name", lambda values: values.mode().iloc[0] if not values.mode().empty else values.iloc[0]),
                avg_price=("price", "mean"),
            )
            .reset_index()
        )
        customer_features["monetary_std"] = customer_features["monetary_std"].fillna(0)

        customer_outlier_mask = pd.Series(False, index=customer_features.index)
        for column in ["frequency", "monetary_total"]:
            customer_outlier_mask |= _iqr_outlier_mask(customer_features[column], multiplier=3.0, non_negative=True)
        customer_features = customer_features.loc[~customer_outlier_mask].copy()

        if len(customer_features) < 20:
            raise MlServiceError("Customer analytical mart is too small after cleansing.")

        customer_features["customer_lifetime_days"] = (
            customer_features["last_purchase"] - customer_features["first_purchase"]
        ).dt.days.clip(lower=0)
        customer_features["purchase_velocity"] = customer_features["frequency"] / (
            customer_features["customer_lifetime_days"] / 30 + 1
        )
        customer_features["avg_days_between_purchases"] = customer_features["customer_lifetime_days"] / (
            customer_features["frequency"] + 1
        )
        customer_features["avg_days_between_purchases"] = customer_features["avg_days_between_purchases"].fillna(0)

        customer_features["churn"] = (customer_features["recency"] > self.churn_threshold_days).astype(int)
        if customer_features["churn"].nunique() < 2:
            raise MlServiceError("Churn target has only one class. Adjust the analytical window.")

        customer_features["r_score"] = _make_rfm_score(customer_features["recency"], higher_is_better=False)
        customer_features["f_score"] = _make_rfm_score(customer_features["frequency"], higher_is_better=True)
        customer_features["m_score"] = _make_rfm_score(customer_features["monetary_total"], higher_is_better=True)
        customer_features["rfm_score"] = (
            customer_features["r_score"] + customer_features["f_score"] + customer_features["m_score"]
        )

        feature_columns = [
            "recency",
            "frequency",
            "monetary_total",
            "monetary_mean",
            "monetary_std",
            "quantity_total",
            "quantity_mean",
            "product_diversity",
            "avg_price",
            "customer_lifetime_days",
            "purchase_velocity",
            "avg_days_between_purchases",
            "r_score",
            "f_score",
            "m_score",
            "rfm_score",
        ]

        df_model = pd.get_dummies(customer_features.copy(), columns=["channel"], prefix="channel", drop_first=True)
        encoded_channel_columns = [column for column in df_model.columns if column.startswith("channel_")]
        feature_columns_final = feature_columns + encoded_channel_columns

        X = df_model[feature_columns_final].fillna(0)
        y = df_model["churn"].astype(int)
        min_class_count = int(y.value_counts().min())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if min_class_count >= 2 else None,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=220,
                max_depth=8,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        }

        model_results: dict[str, dict[str, object]] = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]
            predictions = model.predict(X_test_scaled)
            model_results[name] = {
                "model": model,
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
            }

        best_model_name = max(model_results, key=lambda name: float(model_results[name]["roc_auc"]))
        best_model = model_results[best_model_name]["model"]

        X_all_scaled = scaler.transform(X)
        df_model["churn_probability"] = best_model.predict_proba(X_all_scaled)[:, 1]
        customer_features = customer_features.merge(
            df_model[["customer_fk", "churn_probability"]],
            on="customer_fk",
            how="left",
        )
        customer_features["churn_probability"] = customer_features["churn_probability"].fillna(0.0)
        customer_features["risk_band"] = customer_features["churn_probability"].map(_risk_band)

        cluster_features = [
            "recency",
            "frequency",
            "monetary_total",
            "product_diversity",
            "purchase_velocity",
            "rfm_score",
        ]
        cluster_scaler = StandardScaler()
        X_cluster = cluster_scaler.fit_transform(customer_features[cluster_features].fillna(0))
        cluster_count = min(4, max(2, len(customer_features) // 60 + 2))
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        customer_features["segment"] = kmeans.fit_predict(X_cluster)

        segment_profile = (
            customer_features.groupby("segment", dropna=False)
            .agg(
                n_customers=("customer_fk", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary_total", "mean"),
                avg_product_diversity=("product_diversity", "mean"),
                churn_rate=("churn", "mean"),
                avg_churn_prob=("churn_probability", "mean"),
                avg_rfm_score=("rfm_score", "mean"),
            )
            .reset_index()
            .sort_values("segment")
        )

        total_customers = int(segment_profile["n_customers"].sum())
        tiny_segment_threshold = max(2, int(total_customers * 0.01))
        recency_q40 = float(segment_profile["avg_recency"].quantile(0.40))
        recency_q50 = float(segment_profile["avg_recency"].quantile(0.50))
        recency_q80 = float(segment_profile["avg_recency"].quantile(0.80))
        churn_q30 = float(segment_profile["churn_rate"].quantile(0.30))
        churn_q70 = float(segment_profile["churn_rate"].quantile(0.70))
        frequency_q30 = float(segment_profile["avg_frequency"].quantile(0.30))
        frequency_q60 = float(segment_profile["avg_frequency"].quantile(0.60))
        frequency_q90 = float(segment_profile["avg_frequency"].quantile(0.90))
        monetary_q80 = float(segment_profile["avg_monetary"].quantile(0.80))
        monetary_q90 = float(segment_profile["avg_monetary"].quantile(0.90))

        def label_segment(row: pd.Series) -> str:
            if row["n_customers"] <= tiny_segment_threshold:
                if row["avg_frequency"] >= frequency_q90 and row["churn_rate"] >= churn_q70:
                    return "Whale Accounts At-Risk"
                if row["avg_monetary"] >= monetary_q90:
                    return "High-Value Outliers"
            if row["avg_recency"] >= recency_q80 and row["churn_rate"] >= churn_q70:
                return "Lost Customers"
            if row["churn_rate"] >= churn_q70 and row["avg_recency"] >= recency_q50:
                return "At-Risk Declining"
            if row["avg_monetary"] >= monetary_q80 and row["avg_frequency"] >= frequency_q60:
                return "High-Value Buyers"
            if row["churn_rate"] <= churn_q30 and row["avg_recency"] <= recency_q40:
                return "Active Loyal"
            if row["avg_frequency"] <= frequency_q30:
                return "Low Engagement"
            return "Regular Buyers"

        segment_profile["segment_name"] = segment_profile.apply(label_segment, axis=1)
        segment_name_map = dict(zip(segment_profile["segment"], segment_profile["segment_name"]))
        customer_features["segment_name"] = customer_features["segment"].map(segment_name_map)

        channel_summary = (
            customer_features.groupby("channel", dropna=False)
            .agg(
                total_customers=("customer_fk", "count"),
                churned=("churn", "sum"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_churn_probability=("churn_probability", "mean"),
            )
            .reset_index()
        )
        channel_summary["churn_rate"] = channel_summary["churned"] / channel_summary["total_customers"].replace(0, np.nan)
        channel_summary["churn_rate"] = channel_summary["churn_rate"].fillna(0.0)
        channel_summary = channel_summary.sort_values("churn_rate", ascending=False)

        feature_weights = pd.Series(dtype=float)
        if hasattr(best_model, "feature_importances_"):
            feature_weights = pd.Series(best_model.feature_importances_, index=feature_columns_final)
        elif hasattr(best_model, "coef_"):
            feature_weights = pd.Series(np.abs(best_model.coef_[0]), index=feature_columns_final)
        feature_weights = feature_weights.sort_values(ascending=False)

        risk_distribution = (
            customer_features.groupby("risk_band", dropna=False)
            .agg(customers=("customer_fk", "count"))
            .reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
            .reset_index()
        )
        risk_distribution["share"] = risk_distribution["customers"] / max(len(customer_features), 1)

        high_risk = customer_features[customer_features["churn_probability"] >= self.high_risk_threshold].copy()
        high_risk = high_risk.sort_values(["churn_probability", "monetary_total"], ascending=[False, False])

        def recommended_action(row: pd.Series) -> str:
            if row["churn_probability"] >= 0.85:
                return "Launch immediate win-back outreach."
            if "High-Value" in str(row["segment_name"]):
                return "Offer premium retention bundle."
            if row["recency"] >= self.churn_threshold_days:
                return "Schedule reactivation campaign this week."
            return "Maintain personalized nurturing."

        customer_features["recommended_action"] = customer_features.apply(recommended_action, axis=1)

        strongest_channel = (
            channel_summary.iloc[0]["channel"] if not channel_summary.empty else "All channels"
        )
        most_exposed_segment = (
            segment_profile.sort_values("churn_rate", ascending=False).iloc[0]["segment_name"]
            if not segment_profile.empty
            else "Unassigned"
        )
        top_feature_name = str(feature_weights.index[0]) if not feature_weights.empty else "recency"

        return {
            "reference_date": reference_date,
            "customer_features": customer_features.sort_values("churn_probability", ascending=False).reset_index(drop=True),
            "segment_profile": segment_profile.reset_index(drop=True),
            "channel_summary": channel_summary.reset_index(drop=True),
            "risk_distribution": risk_distribution.reset_index(drop=True),
            "feature_weights": feature_weights,
            "performance": {
                "best_model": best_model_name,
                "roc_auc": float(model_results[best_model_name]["roc_auc"]),
                "accuracy": float(model_results[best_model_name]["accuracy"]),
                "precision": float(model_results[best_model_name]["precision"]),
                "recall": float(model_results[best_model_name]["recall"]),
                "f1": float(model_results[best_model_name]["f1"]),
                "evaluated_customers": int(len(customer_features)),
                "feature_count": int(len(feature_columns_final)),
            },
            "insights": [
                f"The highest churn pressure currently sits on {strongest_channel}.",
                f"The most exposed segment is {most_exposed_segment}.",
                f"{top_feature_name.replace('_', ' ')} is the strongest signal in the selected churn model.",
            ],
            "model": best_model,
            "scaler": scaler,
            "feature_columns": feature_columns_final,
            "encoded_channel_columns": encoded_channel_columns,
            "channel_reference": customer_features["channel"].copy(),
            "rfm_reference": customer_features[["recency", "frequency", "monetary_total"]].copy(),
            "cluster_scaler": cluster_scaler,
            "cluster_features": cluster_features,
            "segment_name_map": segment_name_map,
            "kmeans": kmeans,
        }

    def summary(self) -> dict[str, object]:
        bundle = self._bundle
        customer_features: pd.DataFrame = bundle["customer_features"]  # type: ignore[assignment]
        segment_profile: pd.DataFrame = bundle["segment_profile"]  # type: ignore[assignment]
        channel_summary: pd.DataFrame = bundle["channel_summary"]  # type: ignore[assignment]
        risk_distribution: pd.DataFrame = bundle["risk_distribution"]  # type: ignore[assignment]
        feature_weights: pd.Series = bundle["feature_weights"]  # type: ignore[assignment]
        performance: dict[str, object] = bundle["performance"]  # type: ignore[assignment]
        reference_date: pd.Timestamp = bundle["reference_date"]  # type: ignore[assignment]
        insights: list[str] = bundle["insights"]  # type: ignore[assignment]

        high_risk_customers = customer_features[customer_features["churn_probability"] >= self.high_risk_threshold]
        average_probability = float(customer_features["churn_probability"].mean()) if not customer_features.empty else 0.0

        return {
            "workflow": "customerChurn",
            "headline": "Customer churn intelligence is ready",
            "summary": "The churn model scores every customer, groups them into actionable segments, and highlights the strongest retention priorities.",
            "referenceDate": reference_date.date().isoformat(),
            "churnThresholdDays": self.churn_threshold_days,
            "metrics": [
                {"label": "Customers scored", "value": int(len(customer_features)), "hint": "active customer profiles"},
                {"label": "High-risk accounts", "value": int(len(high_risk_customers)), "hint": "probability >= 70%"},
                {"label": "Average churn risk", "value": round(average_probability * 100, 1), "hint": "portfolio-wide probability %"},
                {"label": "Segments", "value": int(segment_profile["segment"].nunique()), "hint": "behavior clusters"},
            ],
            "modelPerformance": performance,
            "insights": insights,
            "riskDistribution": risk_distribution.to_dict(orient="records"),
            "channelSummary": [
                {
                    "channel": row["channel"],
                    "totalCustomers": int(row["total_customers"]),
                    "churnRate": round(float(row["churn_rate"]), 4),
                    "avgRecency": round(float(row["avg_recency"]), 1),
                    "avgFrequency": round(float(row["avg_frequency"]), 1),
                    "avgChurnProbability": round(float(row["avg_churn_probability"]), 4),
                }
                for _, row in channel_summary.iterrows()
            ],
            "segmentSummary": [
                {
                    "segment": int(row["segment"]),
                    "segmentName": row["segment_name"],
                    "customers": int(row["n_customers"]),
                    "churnRate": round(float(row["churn_rate"]), 4),
                    "avgChurnProbability": round(float(row["avg_churn_prob"]), 4),
                    "avgRecency": round(float(row["avg_recency"]), 1),
                    "avgFrequency": round(float(row["avg_frequency"]), 1),
                    "avgMonetary": round(float(row["avg_monetary"]), 2),
                    "avgRfmScore": round(float(row["avg_rfm_score"]), 2),
                }
                for _, row in segment_profile.iterrows()
            ],
            "featureImportance": [
                {
                    "feature": str(feature).replace("_", " "),
                    "weight": round(float(weight), 4),
                }
                for feature, weight in feature_weights.head(8).items()
            ],
            "customers": [
                {
                    "customerId": int(row["customer_fk"]),
                    "customerName": row["customer_name"],
                    "customerCode": row["customer_code"],
                    "channel": row["channel"],
                    "segment": row["segment_name"],
                    "riskBand": row["risk_band"],
                    "recency": int(row["recency"]),
                    "frequency": int(row["frequency"]),
                    "monetaryTotal": round(float(row["monetary_total"]), 2),
                    "rfmScore": int(row["rfm_score"]),
                    "churnProbability": round(float(row["churn_probability"]), 4),
                    "recommendedAction": row["recommended_action"],
                }
                for _, row in customer_features.iterrows()
            ],
        }

    def predict(self, payload: dict[str, object]) -> dict[str, object]:
        bundle = self._bundle
        model = bundle["model"]
        scaler: StandardScaler = bundle["scaler"]  # type: ignore[assignment]
        feature_columns: list[str] = bundle["feature_columns"]  # type: ignore[assignment]
        encoded_channel_columns: list[str] = bundle["encoded_channel_columns"]  # type: ignore[assignment]
        rfm_reference: pd.DataFrame = bundle["rfm_reference"]  # type: ignore[assignment]
        cluster_scaler: StandardScaler = bundle["cluster_scaler"]  # type: ignore[assignment]
        cluster_features: list[str] = bundle["cluster_features"]  # type: ignore[assignment]
        kmeans: KMeans = bundle["kmeans"]  # type: ignore[assignment]
        segment_name_map: dict[int, str] = bundle["segment_name_map"]  # type: ignore[assignment]

        channel = str(payload.get("channel", "All channels")).strip() or "All channels"
        recency = max(0.0, float(payload.get("recency", 0)))
        frequency = max(1.0, float(payload.get("frequency", 1)))
        monetary_total = max(0.0, float(payload.get("monetaryTotal", 0)))
        monetary_mean = max(0.0, float(payload.get("monetaryMean", monetary_total / max(frequency, 1))))
        quantity_total = max(0.0, float(payload.get("quantityTotal", frequency)))
        product_diversity = max(1.0, float(payload.get("productDiversity", 1)))
        avg_price = max(0.0, float(payload.get("avgPrice", monetary_mean)))
        customer_lifetime_days = max(0.0, float(payload.get("customerLifetimeDays", recency + 30)))

        quantity_mean = quantity_total / max(frequency, 1)
        purchase_velocity = frequency / (customer_lifetime_days / 30 + 1)
        avg_days_between_purchases = customer_lifetime_days / (frequency + 1)
        monetary_std = float(payload.get("monetaryStd", 0))

        r_score = _score_single(recency, rfm_reference["recency"], higher_is_better=False)
        f_score = _score_single(frequency, rfm_reference["frequency"], higher_is_better=True)
        m_score = _score_single(monetary_total, rfm_reference["monetary_total"], higher_is_better=True)
        rfm_score = r_score + f_score + m_score

        feature_row = {
            "recency": recency,
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_mean": monetary_mean,
            "monetary_std": monetary_std,
            "quantity_total": quantity_total,
            "quantity_mean": quantity_mean,
            "product_diversity": product_diversity,
            "avg_price": avg_price,
            "customer_lifetime_days": customer_lifetime_days,
            "purchase_velocity": purchase_velocity,
            "avg_days_between_purchases": avg_days_between_purchases,
            "r_score": r_score,
            "f_score": f_score,
            "m_score": m_score,
            "rfm_score": rfm_score,
        }

        for encoded_channel in encoded_channel_columns:
            channel_name = encoded_channel.removeprefix("channel_")
            feature_row[encoded_channel] = 1 if channel == channel_name else 0

        input_frame = pd.DataFrame([{column: feature_row.get(column, 0) for column in feature_columns}])
        probability = float(model.predict_proba(scaler.transform(input_frame))[0][1])

        cluster_payload = pd.DataFrame(
            [
                {
                    "recency": recency,
                    "frequency": frequency,
                    "monetary_total": monetary_total,
                    "product_diversity": product_diversity,
                    "purchase_velocity": purchase_velocity,
                    "rfm_score": rfm_score,
                }
            ]
        )
        segment_id = int(kmeans.predict(cluster_scaler.transform(cluster_payload[cluster_features]))[0])
        segment_name = segment_name_map.get(segment_id, "Regular Buyers")

        if probability >= 0.85:
            status = "watch"
            headline = "Critical churn risk detected"
            action = "Escalate to immediate retention outreach."
        elif probability >= self.high_risk_threshold:
            status = "watch"
            headline = "High churn risk detected"
            action = "Prepare a reactivation offer in the next cycle."
        elif probability >= 0.4:
            status = "balanced"
            headline = "Moderate churn exposure"
            action = "Keep the customer warm with personalized nudges."
        else:
            status = "strong"
            headline = "Retention outlook looks healthy"
            action = "Preserve the current engagement pattern."

        return {
            "workflow": "customerChurn",
            "headline": headline,
            "summary": "This scenario was scored against the same churn model and segment map used by the dashboard analytics.",
            "status": status,
            "confidence": round(max(0.1, min(0.99, probability)), 4),
            "metrics": [
                {"label": "Churn probability", "value": f"{probability * 100:.1f}%", "tone": "primary"},
                {"label": "Segment", "value": segment_name, "tone": "success"},
                {"label": "RFM score", "value": f"{rfm_score}/15", "tone": "neutral"},
            ],
            "insights": [
                action,
                f"Recency is {int(recency)} days and frequency is {frequency:.0f} orders.",
                f"Primary channel assumption is {channel}.",
            ],
            "result": {
                "predictedChurnProbability": round(probability, 4),
                "riskBand": _risk_band(probability),
                "segmentId": segment_id,
                "segmentName": segment_name,
                "recommendedAction": action,
            },
        }
