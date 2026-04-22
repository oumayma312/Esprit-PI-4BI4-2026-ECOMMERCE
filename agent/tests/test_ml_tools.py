"""Integration tests for ML LangChain tools."""
import pytest
from backend.tools import (
    predict_customer_churn, list_customers,
    list_high_risk_customers, get_customer_segment_stats
)


class TestPredictCustomerChurnTool:
    def test_tool_with_customer_id(self):
        result = predict_customer_churn.invoke({"customer_id": 1})
        assert isinstance(result, str)
        assert "Attijari" in result or "Customer" in result
        assert "Churn Probability" in result
        assert "Segment" in result

    def test_tool_with_unknown_customer(self):
        result = predict_customer_churn.invoke({"customer_id": 99999})
        assert "not found" in result.lower()

    def test_tool_missing_all_inputs(self):
        result = predict_customer_churn.invoke({})
        assert "customer_id" in result.lower() or "recency" in result.lower()

    def test_tool_with_manual_features(self):
        result = predict_customer_churn.invoke({
            "recency": 120, "frequency": 2, "monetary_total": 250,
            "monetary_trend": 1.0, "product_diversity": 3,
            "channel_diversity": 1, "lifetime_days": 365,
            "purchase_velocity": 1.5, "avg_price": 80, "channel": "vente direct"
        })
        assert "Churn Probability" in result
        assert "Segment" in result

    def test_tool_output_contains_probability(self):
        result = predict_customer_churn.invoke({"customer_id": 1})
        assert "%" in result or "0." in result or "1." in result

    def test_tool_output_contains_segment(self):
        result = predict_customer_churn.invoke({"customer_id": 1})
        assert "Segment" in result

    def test_tool_output_contains_recommendation(self):
        result = predict_customer_churn.invoke({"customer_id": 1})
        assert "Recommendation" in result


class TestListCustomersTool:
    def test_tool_no_filters(self):
        result = list_customers.invoke({})
        assert "Found" in result

    def test_tool_with_segment_filter(self):
        result = list_customers.invoke({"segment": 0})
        assert "Found" in result

    def test_tool_with_risk_filter(self):
        result = list_customers.invoke({"risk_level": "Eleve"})
        assert "Found" in result or "No customers" in result

    def test_tool_with_search(self):
        result = list_customers.invoke({"search": "Attijari"})
        assert "Found" in result


class TestListHighRiskTool:
    def test_tool_default_threshold(self):
        result = list_high_risk_customers.invoke({})
        assert "High-risk" in result or "probability" in result.lower()

    def test_tool_custom_threshold(self):
        result = list_high_risk_customers.invoke({"min_probability": 0.5})
        assert "probability" in result.lower() or "High-risk" in result

    def test_tool_empty_result(self):
        result = list_high_risk_customers.invoke({"min_probability": 0.999})
        assert "No customers" in result or "probability" in result.lower()


class TestGetSegmentStatsTool:
    def test_tool_returns_all_segments(self):
        result = get_customer_segment_stats.invoke({})
        assert "Segment" in result
        assert "0" in result
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_tool_contains_counts(self):
        result = get_customer_segment_stats.invoke({})
        assert "Customers" in result or "customers" in result

    def test_tool_contains_avg_probability(self):
        result = get_customer_segment_stats.invoke({})
        assert "%" in result or "avg" in result.lower() or "Avg" in result
