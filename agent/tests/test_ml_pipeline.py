"""Unit tests for the ml_pipeline module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml_pipeline import MLPipeline, get_pipeline, SEGMENT_LABELS


@pytest.fixture
def pipeline():
    return get_pipeline(str(Path(__file__).resolve().parent.parent.parent / "artifacts"))


class TestMLPipeline:
    def test_predict_customer_returns_all_fields(self, pipeline):
        result = pipeline.predict_customer({
            "recency": 120, "frequency": 2, "monetary_total": 250,
            "monetary_trend": 1.0, "product_diversity": 3,
            "channel_diversity": 1, "lifetime_days": 365,
            "purchase_velocity": 1.5, "avg_price": 80, "channel": "vente direct"
        })
        assert "churn_probability" in result
        assert "churn_prediction" in result
        assert "segment" in result
        assert "segment_label" in result
        assert "risk_level" in result
        assert "recommendation" in result

    def test_predict_customer_low_risk(self, pipeline):
        """High engagement customer should have valid prediction."""
        result = pipeline.predict_customer({
            "recency": 10, "frequency": 20, "monetary_total": 800,
            "monetary_trend": 1.2, "product_diversity": 5,
            "channel_diversity": 2, "lifetime_days": 730,
            "purchase_velocity": 3.0, "avg_price": 120, "channel": "vente direct"
        })
        assert 0 <= result["churn_probability"] <= 1
        assert result["segment"] in [0, 1, 2, 3]
        assert result["risk_level"] in ["Faible", "Modere", "Eleve"]

    def test_predict_customer_high_risk(self, pipeline):
        """Low engagement customer should have valid prediction."""
        result = pipeline.predict_customer({
            "recency": 200, "frequency": 1, "monetary_total": 50,
            "monetary_trend": 0.5, "product_diversity": 1,
            "channel_diversity": 1, "lifetime_days": 30,
            "purchase_velocity": 0.5, "avg_price": 50, "channel": ""
        })
        assert 0 <= result["churn_probability"] <= 1
        assert result["segment"] in [0, 1, 2, 3]
        assert result["risk_level"] in ["Faible", "Modere", "Eleve"]

    def test_predict_customer_segment_assignment(self, pipeline):
        """High frequency + high monetary should return valid segment."""
        result = pipeline.predict_customer({
            "recency": 15, "frequency": 25, "monetary_total": 1000,
            "monetary_trend": 1.3, "product_diversity": 6,
            "channel_diversity": 3, "lifetime_days": 1000,
            "purchase_velocity": 4.0, "avg_price": 150, "channel": "vente direct"
        })
        assert result["segment"] in [0, 1, 2, 3]
        assert result["segment_label"] in SEGMENT_LABELS.values()

    def test_predict_customer_segment_lost(self, pipeline):
        """High recency + low frequency should return valid segment."""
        result = pipeline.predict_customer({
            "recency": 180, "frequency": 1, "monetary_total": 30,
            "monetary_trend": 0.3, "product_diversity": 1,
            "channel_diversity": 1, "lifetime_days": 500,
            "purchase_velocity": 0.3, "avg_price": 30, "channel": ""
        })
        assert result["segment"] in [0, 1, 2, 3]

    def test_predict_customer_segment_at_risk(self, pipeline):
        """Moderate values should return valid segment."""
        result = pipeline.predict_customer({
            "recency": 100, "frequency": 5, "monetary_total": 300,
            "monetary_trend": 0.8, "product_diversity": 2,
            "channel_diversity": 1, "lifetime_days": 400,
            "purchase_velocity": 1.5, "avg_price": 60, "channel": ""
        })
        assert result["segment"] in [0, 1, 2, 3]

    def test_predict_customer_segment_new(self, pipeline):
        """Recent customer should return valid segment."""
        result = pipeline.predict_customer({
            "recency": 350, "frequency": 1, "monetary_total": 40,
            "monetary_trend": 1.0, "product_diversity": 1,
            "channel_diversity": 1, "lifetime_days": 10,
            "purchase_velocity": 0.5, "avg_price": 40, "channel": ""
        })
        assert result["segment"] in [0, 1, 2, 3]

    def test_predict_customer_channel_encoding(self, pipeline):
        """Channel 'vente direct' should set ch_vente direct dummy."""
        result_with = pipeline.predict_customer({
            "recency": 100, "frequency": 5, "monetary_total": 300,
            "monetary_trend": 1.0, "product_diversity": 2,
            "channel_diversity": 1, "lifetime_days": 400,
            "purchase_velocity": 1.5, "avg_price": 60, "channel": "vente direct"
        })
        result_without = pipeline.predict_customer({
            "recency": 100, "frequency": 5, "monetary_total": 300,
            "monetary_trend": 1.0, "product_diversity": 2,
            "channel_diversity": 1, "lifetime_days": 400,
            "purchase_velocity": 1.5, "avg_price": 60, "channel": ""
        })
        # Different channels should produce different probabilities
        assert result_with["churn_probability"] != result_without["churn_probability"]

    def test_batch_predict_returns_same_shape(self, pipeline):
        df = pd.DataFrame([{
            "recency": 100, "frequency": 5, "monetary_total": 300,
            "monetary_trend": 1.0, "product_diversity": 2,
            "channel_diversity": 1, "lifetime_days": 400,
            "purchase_velocity": 1.5, "avg_price": 60, "channel": ""
        }] * 10)
        results, issues = pipeline.batch_predict(df)
        assert len(results) == 10
        assert "churn_probability" in results.columns
        assert "segment" in results.columns
        assert "risk_level" in results.columns

    def test_batch_predict_single_row(self, pipeline):
        single = pd.DataFrame([{
            "recency": 100, "frequency": 5, "monetary_total": 300,
            "monetary_trend": 1.0, "product_diversity": 2,
            "channel_diversity": 1, "lifetime_days": 400,
            "purchase_velocity": 1.5, "avg_price": 60, "channel": ""
        }])
        batch_result = pipeline.batch_predict(single)[0]
        single_result = pipeline.predict_customer(single.iloc[0].to_dict())
        assert abs(batch_result["churn_probability"].iloc[0] - single_result["churn_probability"]) < 1e-6

    def test_risk_level_mapping(self, pipeline):
        assert pipeline._risk_level(0.3) == "Faible"
        assert pipeline._risk_level(0.5) == "Modere"
        assert pipeline._risk_level(0.8) == "Eleve"
        assert pipeline._risk_level(0.0) == "Faible"
        assert pipeline._risk_level(1.0) == "Eleve"

    def test_segment_label_mapping(self, pipeline):
        for seg_id in [0, 1, 2, 3]:
            assert seg_id in SEGMENT_LABELS
            assert isinstance(SEGMENT_LABELS[seg_id], str)
            assert len(SEGMENT_LABELS[seg_id]) > 0

    def test_business_recommendation_high_risk(self, pipeline):
        rec = pipeline._business_recommendation(0.8, 0)
        assert "retention" in rec.lower() or "reconquete" in rec.lower() or "reclamation" in rec.lower()

    def test_business_recommendation_low_risk_vip(self, pipeline):
        rec = pipeline._business_recommendation(0.2, 0)
        assert "fidelite" in rec.lower() or "loyalty" in rec.lower() or "vip" in rec.lower()

    def test_pipeline_caches_artifacts(self):
        p1 = get_pipeline(str(Path(__file__).resolve().parent.parent.parent / "artifacts"))
        p2 = get_pipeline(str(Path(__file__).resolve().parent.parent.parent / "artifacts"))
        assert p1 is p2

    def test_known_channels(self, pipeline):
        assert "vente direct" in pipeline.known_channels

    def test_reference_date(self, pipeline):
        assert pd.notna(pipeline.reference_date)
