"""Churn prediction pipeline — shared inference module.

Adapted from the original ml_pipeline.py to integrate with the existing
api_common.py patterns in the ML API project.

Usage:
    from churn_pipeline import ChurnPipeline

    pipeline = ChurnPipeline()  # loads artifacts once
    result = pipeline.predict_customer({
        "recency": 120, "frequency": 2, "monetary_total": 250,
        "monetary_trend": 1.0, "product_diversity": 3,
        "channel_diversity": 1, "lifetime_days": 365,
        "purchase_velocity": 1.5, "avg_price": 80, "channel": "vente direct"
    })
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_LABELS: dict[int, str] = {
    0: "Loyaux a forte valeur",
    1: "Clients perdus",
    2: "Intermediaires a risque",
    3: "Clients recents ou nouveaux",
}

SEGMENT_PROFILES: dict[int, dict[str, str]] = {
    0: {
        "meaning": "Clients premium avec depenses elevees et achats frequents.",
        "typical_profile": "Panier moyen eleve, engagement fort, vitesse d'achat soutenue.",
        "action": "Preserver la fidelite via avantages VIP, offres exclusives et suivi proactif.",
    },
    1: {
        "meaning": "Clients probablement inactifs depuis une longue periode.",
        "typical_profile": "Engagement recent faible, dynamique en baisse et risque de churn eleve.",
        "action": "Activer des campagnes de reconquete avec offre limitee et personnalisation.",
    },
    2: {
        "meaning": "Clients de valeur moyenne montrant des signes de desengagement.",
        "typical_profile": "Valeur correcte mais frequence qui baisse et risque qui monte.",
        "action": "Lancer des parcours de nurturing et des bundles cibles rapidement.",
    },
    3: {
        "meaning": "Clients recents avec historique d'achat encore limite.",
        "typical_profile": "Anciennete faible, volume d'achat bas, mais activite recente.",
        "action": "Accelerer l'onboarding avec education produit et cross-sell precoce.",
    },
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = os.getenv("CHURN_ARTIFACTS_DIR", str(APP_DIR / "churn_artifacts"))


def _resolve_artifacts_dir(artifacts_dir: str) -> Path:
    """Resolve artifacts directory, trying multiple candidates."""
    raw = Path(artifacts_dir)
    if raw.is_absolute():
        if raw.exists():
            return raw
    candidates = [
        Path.cwd() / artifacts_dir,
        APP_DIR / artifacts_dir,
        raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class ChurnPipeline:
    """Unified ML pipeline for churn prediction and customer segmentation.

    Loads models, scalers, and config once. Provides single-row and
    batch inference, feature derivation from raw transactions, and PCA plot
    generation.
    """

    def __init__(self, artifacts_dir: str | None = None):
        self._base = _resolve_artifacts_dir(artifacts_dir or DEFAULT_ARTIFACTS_DIR)
        required = {
            "churn_model": self._base / "churn_model.joblib",
            "churn_scaler": self._base / "churn_scaler.joblib",
            "segment_model": self._base / "segment_model.joblib",
            "segment_scaler": self._base / "segment_scaler.joblib",
            "config": self._base / "inference_config.json",
        }
        missing = [str(p) for p in required.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing ML artifacts:\n" + "\n".join(missing)
            )

        with open(required["config"], "r", encoding="utf-8") as f:
            self._config: dict[str, Any] = json.load(f)

        self.churn_model = joblib.load(required["churn_model"])
        self.churn_scaler = joblib.load(required["churn_scaler"])
        self.segment_model = joblib.load(required["segment_model"])
        self.segment_scaler = joblib.load(required["segment_scaler"])

        self.reference_date = pd.to_datetime(
            self._config.get("reference_date"), errors="coerce"
        )
        if pd.isna(self.reference_date):
            self.reference_date = pd.Timestamp.today().normalize()

        self.decision_threshold = float(
            self._config.get("decision_threshold", 0.5)
        )

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def known_channels(self) -> list[str]:
        channels = []
        for col in self._config.get("churn_channel_dummy_columns", []):
            if col.startswith("ch_"):
                channels.append(col[3:])
        return sorted(set(channels))

    # ------------------------------------------------------------------
    # Single-customer prediction
    # ------------------------------------------------------------------

    def predict_customer(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict churn probability and segment for a single customer.

        Parameters
        ----------
        features : dict
            Must contain: recency, frequency, monetary_total, monetary_trend,
            product_diversity, channel_diversity, lifetime_days,
            purchase_velocity, avg_price, channel (optional).

        Returns
        -------
        dict with keys: churn_probability, churn_prediction, segment,
        segment_label, risk_level, recommendation.
        """
        row_df = pd.DataFrame([features])
        result_df, _ = self.run_inference(row_df, features.get("channel", ""))
        row = result_df.iloc[0]
        return {
            "churn_probability": float(row["churn_probability"]),
            "churn_prediction": str(row["churn_prediction"]),
            "segment": int(row["segment"]),
            "segment_label": str(row["segment_label"]),
            "risk_level": self._risk_level(float(row["churn_probability"])),
            "recommendation": self._business_recommendation(
                float(row["churn_probability"]), int(row["segment"])
            ),
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def batch_predict(
        self, df: pd.DataFrame, channel_value: str = ""
    ) -> tuple[pd.DataFrame, list[str]]:
        """Run inference on a DataFrame of customer features.

        Returns (results_df, parse_issues).
        """
        return self.run_inference(df, channel_value)

    # ------------------------------------------------------------------
    # Internal inference
    # ------------------------------------------------------------------

    def run_inference(
        self, df: pd.DataFrame, channel_value: str
    ) -> tuple[pd.DataFrame, list[str]]:
        churn_features, churn_issues = self._build_churn_features(
            df, channel_value
        )
        segment_features, segment_issues = self._build_segment_features(df)

        churn_scaled = self.churn_scaler.transform(churn_features)
        segment_scaled = self.segment_scaler.transform(segment_features)

        churn_probability = self.churn_model.predict_proba(churn_scaled)[:, 1]
        churn_label = (churn_probability >= self.decision_threshold).astype(int)
        segment = self.segment_model.predict(segment_scaled)

        out = df.copy()
        out["churn_probability"] = np.round(churn_probability, 6)
        out["churn_prediction"] = churn_label
        out["segment"] = segment
        out["segment_label"] = out["segment"].map(SEGMENT_LABELS).fillna(
            "Segment non étiqueté"
        )
        out["risk_level"] = out["churn_probability"].apply(self._risk_level)
        out["recommendation"] = out.apply(
            lambda r: self._business_recommendation(
                r["churn_probability"], r["segment"]
            ),
            axis=1,
        )
        parse_issues = sorted(set(churn_issues + segment_issues))
        return out, parse_issues

    def _build_churn_features(
        self, df: pd.DataFrame, channel_value: str
    ) -> tuple[pd.DataFrame, list[str]]:
        prepared = self._ensure_log_features(df)
        prepared = self._encode_channel_dummies(prepared, channel_value)
        required = self._config["churn_feature_columns"]
        missing = [c for c in required if c not in prepared.columns]
        if missing:
            raise ValueError("Missing churn columns:\n" + "\n".join(missing))
        return self._coerce_numeric_frame(prepared[required], required)

    def _build_segment_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        prepared = self._ensure_log_features(df)
        required = self._config["segmentation_feature_columns"]
        missing = [c for c in required if c not in prepared.columns]
        if missing:
            raise ValueError("Missing segmentation columns:\n" + "\n".join(missing))
        return self._coerce_numeric_frame(prepared[required], required)

    def _coerce_numeric_frame(
        self, df: pd.DataFrame, columns: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        out = df.copy()
        issues: list[str] = []
        for col in columns:
            before = out[col]
            out[col] = pd.to_numeric(out[col], errors="coerce")
            failures = (out[col].isna() & before.notna()).sum()
            if failures > 0:
                issues.append(f"{col}: {int(failures)} non-numeric value(s)")
        return out.fillna(0), issues

    def _ensure_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for base_col in self._config.get("log_features", []):
            log_col = f"{base_col}_log"
            if log_col not in out.columns:
                if base_col in out.columns:
                    vals = pd.to_numeric(out[base_col], errors="coerce").fillna(0)
                else:
                    vals = pd.Series(0.0, index=out.index)
                out[log_col] = np.log1p(vals.clip(lower=0))
        return out

    def _encode_channel_dummies(
        self, df: pd.DataFrame, channel_value: str
    ) -> pd.DataFrame:
        out = df.copy()
        dummy_cols = self._config.get("churn_channel_dummy_columns", [])
        for col in dummy_cols:
            out[col] = 0
        if channel_value:
            selected = f"ch_{channel_value}"
            if selected in dummy_cols:
                out[selected] = 1
        return out

    # ------------------------------------------------------------------
    # Feature derivation from raw transactions
    # ------------------------------------------------------------------

    def derive_features_from_transactions(
        self, tx_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Derive RFM features from raw transaction data.

        Parameters
        ----------
        tx_df : pd.DataFrame
            Must contain: dateFK, total_ttc, quantity, price, channelFK,
            productFK. Optional: customerFK, FactVentePK.

        Returns
        -------
        (customer_features_df, issues_list)
        """
        issues: list[str] = []
        df = tx_df.copy()

        required_cols = {"dateFK", "total_ttc", "quantity", "price", "channelFK", "productFK"}
        alt_cols = {"transaction_date", "total_ttc", "quantity", "price", "channelFK", "productFK"}
        if not required_cols.issubset(set(df.columns)) and not alt_cols.issubset(set(df.columns)):
            raise ValueError(
                "Missing transaction columns: need dateFK/transaction_date, "
                "total_ttc, quantity, price, channelFK, productFK"
            )
        if "transaction_date" in df.columns and "dateFK" not in df.columns:
            df = df.rename(columns={"transaction_date": "dateFK"})

        if "customerFK" not in df.columns:
            df["customerFK"] = "client_importe_1"
            issues.append("customerFK column missing: single customer assumed.")

        strict_dates = pd.to_datetime(df["dateFK"].astype(str), format="%Y%m%d", errors="coerce")
        loose_dates = pd.to_datetime(df["dateFK"], errors="coerce")
        df["dateFK"] = strict_dates.fillna(loose_dates)

        invalid = int(df["dateFK"].isna().sum())
        if invalid > 0:
            issues.append(f"{invalid} row(s) ignored due to invalid dateFK.")
            df = df.dropna(subset=["dateFK"]).copy()

        if df.empty:
            raise ValueError("No usable rows after date cleaning.")

        for col in ["total_ttc", "quantity", "price"]:
            before = df[col]
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            failures = int(
                (before.notna() & pd.to_numeric(before, errors="coerce").isna()).sum()
            )
            if failures > 0:
                issues.append(f"{col}: {failures} value(s) converted to 0.")

        if "FactVentePK" not in df.columns:
            df["FactVentePK"] = 1
            issues.append("FactVentePK missing: frequency calculated on row count.")

        df = df.sort_values(["customerFK", "dateFK"])

        customer_features = df.groupby("customerFK").agg(
            recency=("dateFK", lambda x: (self.reference_date - x.max()).days),
            first_purchase=("dateFK", "min"),
            last_purchase=("dateFK", "max"),
            frequency=("FactVentePK", "count"),
            monetary_total=("total_ttc", "sum"),
            monetary_mean=("total_ttc", "mean"),
            monetary_std=("total_ttc", "std"),
            quantity_total=("quantity", "sum"),
            quantity_mean=("quantity", "mean"),
            product_diversity=("productFK", "nunique"),
            channel_diversity=("channelFK", "nunique"),
            primary_channel=("channelFK", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
            avg_price=("price", "mean"),
            max_price=("price", "max"),
        ).reset_index()

        customer_features["monetary_std"] = customer_features["monetary_std"].fillna(0)
        customer_features["lifetime_days"] = (
            customer_features["last_purchase"] - customer_features["first_purchase"]
        ).dt.days
        customer_features["purchase_velocity"] = customer_features["frequency"] / (
            customer_features["lifetime_days"] / 30 + 1
        )

        last3_spend = (
            df.groupby("customerFK", group_keys=False)
            .apply(lambda g: g.tail(3)["total_ttc"].mean())
            .reset_index(name="last3_monetary_mean")
        )
        customer_features = customer_features.merge(last3_spend, on="customerFK", how="left")
        customer_features["monetary_trend"] = customer_features["last3_monetary_mean"] / (
            customer_features["monetary_mean"] + 1e-9
        )
        customer_features["recency_norm"] = customer_features["recency"] / (
            customer_features["lifetime_days"] + 1
        )

        return customer_features, issues

    # ------------------------------------------------------------------
    # PCA plot generation
    # ------------------------------------------------------------------

    def generate_pca_plot(
        self,
        source_csv_path: str | None = None,
    ) -> bytes | None:
        """Generate PCA visualization and return as PNG bytes.

        Returns None if source file is missing or insufficient data.
        """
        if source_csv_path is None:
            candidates = [
                APP_DIR / "customer_churn_predictions.csv",
                Path("customer_churn_predictions.csv"),
                APP_DIR / "data" / "customer_churn_predictions.csv",
            ]
            for candidate in candidates:
                if candidate.exists():
                    source_csv_path = str(candidate)
                    break
            if source_csv_path is None:
                return None

        source_path = Path(source_csv_path)
        if not source_path.exists():
            return None

        df = pd.read_csv(source_path)
        needed = {"recency", "frequency", "monetary_total", "segment"}
        if not needed.issubset(set(df.columns)):
            return None

        df = df.dropna(subset=["recency", "frequency", "monetary_total", "segment"]).copy()
        if len(df) < 5:
            return None

        df["monetary_total_log"] = np.log1p(
            pd.to_numeric(df["monetary_total"], errors="coerce").fillna(0).clip(lower=0)
        )

        feats = df[["recency", "frequency", "monetary_total_log"]].copy()
        feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0)
        scaled = StandardScaler().fit_transform(feats)
        coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

        plot_df = pd.DataFrame({
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "segment": df["segment"].astype(int).values,
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        for seg_id in sorted(plot_df["segment"].unique()):
            part = plot_df[plot_df["segment"] == seg_id]
            label = f"{seg_id} - {SEGMENT_LABELS.get(seg_id, 'Unknown')}"
            ax.scatter(part["pc1"], part["pc2"], s=18, alpha=0.65, label=label)

        ax.set_title("PCA Projection of Customer Segments")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, dpi=160)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _risk_level(self, prob: float) -> str:
        if prob >= 0.7:
            return "Eleve"
        if prob >= 0.4:
            return "Modere"
        return "Faible"

    def _business_recommendation(self, prob: float, segment_id: int) -> str:
        if prob >= 0.7:
            return "Action de retention immediate: offre ciblee et prise de contact proactive."
        if prob >= 0.4:
            return "A suivre de pres: lancer une campagne d'engagement et surveiller la prochaine fenetre d'achat."
        if segment_id == 0:
            return "Conserver la dynamique de fidelite avec avantages VIP et upsell personnalise."
        return "Profil sain: maintenir l'engagement regulier et monitorer les tendances."


# ---------------------------------------------------------------------------
# Lazy-loaded singleton (for FastAPI)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_pipeline(artifacts_dir: str | None = None) -> ChurnPipeline:
    """Return a cached ChurnPipeline instance."""
    return ChurnPipeline(artifacts_dir)
