"""Integration tests for customer database functions."""
import pytest
from httpx import ASGITransport, AsyncClient
from backend.main import app


@pytest.fixture
def test_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestCustomerAPI:
    async def test_list_customers_no_filters(self, test_client):
        resp = await test_client.get("/api/customers?limit=100")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) > 0

    async def test_list_customers_segment_filter(self, test_client):
        resp = await test_client.get("/api/customers?segment=0&limit=100")
        assert resp.status_code == 200
        customers = resp.json()
        assert len(customers) > 0
        for c in customers:
            assert c.get("segment") == 0

    async def test_list_customers_risk_filter(self, test_client):
        resp = await test_client.get("/api/customers?risk_level=Eleve&limit=100")
        assert resp.status_code == 200
        customers = resp.json()
        for c in customers:
            assert c.get("risk_level") == "Eleve"

    async def test_list_customers_search(self, test_client):
        resp = await test_client.get("/api/customers?search=Attijari&limit=100")
        assert resp.status_code == 200
        customers = resp.json()
        assert len(customers) > 0
        for c in customers:
            assert "Attijari" in c["name"]

    async def test_list_customers_combined_filters(self, test_client):
        resp = await test_client.get("/api/customers?segment=0&risk_level=Eleve&limit=100")
        assert resp.status_code == 200
        customers = resp.json()
        for c in customers:
            assert c.get("segment") == 0
            assert c.get("risk_level") == "Eleve"

    async def test_list_customers_pagination(self, test_client):
        resp = await test_client.get("/api/customers?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    async def test_get_customer_detail(self, test_client):
        # Find a customer with prediction
        customers = await test_client.get("/api/customers?limit=100")
        cust = customers.json()[0]
        resp = await test_client.get(f"/api/customers/{cust['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "prediction" in data
        assert "transactions" in data

    async def test_get_customer_not_found(self, test_client):
        resp = await test_client.get("/api/customers/99999")
        assert resp.status_code == 404

    async def test_high_risk_customers_default(self, test_client):
        resp = await test_client.get("/api/customers/high-risk")
        assert resp.status_code == 200
        customers = resp.json()
        for c in customers:
            assert c.get("churn_probability", 0) >= 0.7

    async def test_high_risk_customers_custom_threshold(self, test_client):
        resp = await test_client.get("/api/customers/high-risk?min_probability=0.5")
        assert resp.status_code == 200
        customers = resp.json()
        for c in customers:
            assert c.get("churn_probability", 0) >= 0.5

    async def test_segment_stats(self, test_client):
        resp = await test_client.get("/api/customers/segments")
        assert resp.status_code == 200
        stats = resp.json()
        assert len(stats) == 4  # 4 segments
        for s in stats:
            assert "segment" in s
            assert "segment_label" in s
            assert "count" in s
            assert "avg_probability" in s


class TestChurnPredictionAPI:
    async def test_predict_single_customer(self, test_client):
        resp = await test_client.post("/api/predict/churn", json={
            "recency": 120, "frequency": 2, "monetary_total": 250,
            "monetary_trend": 1.0, "product_diversity": 3,
            "channel_diversity": 1, "lifetime_days": 365,
            "purchase_velocity": 1.5, "avg_price": 80, "channel": "vente direct"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "churn_probability" in data
        assert "segment" in data
        assert "risk_level" in data
        assert 0 <= data["churn_probability"] <= 1

    async def test_predict_single_customer_required_fields(self, test_client):
        resp = await test_client.post("/api/predict/churn", json={
            "frequency": 2, "monetary_total": 250
        })
        assert resp.status_code == 422

    async def test_predict_batch(self, test_client):
        resp = await test_client.post("/api/predict/churn/batch", json={
            "data": [
                {"recency": 100, "frequency": 5, "monetary_total": 300,
                 "monetary_trend": 1.0, "product_diversity": 2,
                 "channel_diversity": 1, "lifetime_days": 400,
                 "purchase_velocity": 1.5, "avg_price": 60, "channel": ""},
                {"recency": 50, "frequency": 10, "monetary_total": 500,
                 "monetary_trend": 1.2, "product_diversity": 3,
                 "channel_diversity": 2, "lifetime_days": 600,
                 "purchase_velocity": 2.0, "avg_price": 80, "channel": "vente direct"}
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2

    async def test_predict_batch_empty(self, test_client):
        resp = await test_client.post("/api/predict/churn/batch", json={"data": []})
        assert resp.status_code == 400

    async def test_pca_plot(self, test_client):
        resp = await test_client.get("/api/predict/pca-plot")
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/png"
        assert len(resp.content) > 0
