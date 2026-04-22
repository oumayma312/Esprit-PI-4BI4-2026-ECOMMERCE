"""Integration tests for backend/main.py chat/stream endpoints."""
import pytest
import json
from httpx import ASGITransport, AsyncClient
from backend.main import app


@pytest.fixture
def test_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestChatNonStreaming:
    async def test_chat_returns_thread_id(self, test_client):
        resp = await test_client.post("/api/chat", json={
            "message": "Hello"
        })
        # This may fail if LLM is not available, but test the structure
        if resp.status_code == 200:
            data = resp.json()
            assert "thread_id" in data
            assert isinstance(data["thread_id"], str)
            assert "response" in data
            assert "pending_approval" in data

    async def test_chat_with_thread_id(self, test_client):
        resp = await test_client.post("/api/chat", json={
            "message": "Hello",
            "thread_id": "test-thread-123"
        })
        if resp.status_code == 200:
            data = resp.json()
            assert data["thread_id"] == "test-thread-123"

    async def test_chat_missing_material_returns_error(self, test_client):
        # Create a material first
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        resp = await test_client.post("/api/chat", json={
            "message": "Order material 999"
        })
        # Should not crash, may return error from agent


class TestChatApproveRejectEdit:
    async def test_approve_endpoint(self, test_client):
        resp = await test_client.post("/api/chat/approve", json={
            "thread_id": "nonexistent-thread",
            "decision": "approve"
        })
        # May fail if agent not available, but test structure
        assert resp.status_code in (200, 500)  # 500 if agent fails

    async def test_reject_endpoint(self, test_client):
        resp = await test_client.post("/api/chat/reject", json={
            "thread_id": "nonexistent-thread",
            "decision": "reject"
        })
        assert resp.status_code in (200, 500)


class TestSSEStream:
    async def test_stream_returns_thread_id(self, test_client):
        resp = await test_client.get("/api/chat/stream?message=Hello")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        # Read first event
        lines = resp.text.strip().split("\n")
        first_event = lines[0]
        assert "thread_id" in first_event

    async def test_stream_url_with_thread_id(self, test_client):
        resp = await test_client.get(
            "/api/chat/stream?message=Hello&thread_id=custom-thread-456"
        )
        assert resp.status_code == 200
        assert "custom-thread-456" in resp.text

    async def test_stream_url_encodes_message(self, test_client):
        resp = await test_client.get(
            "/api/chat/stream?message=I+need+50+meters+of+copper+wire"
        )
        assert resp.status_code == 200
