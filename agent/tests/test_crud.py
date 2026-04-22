"""Integration tests for backend/main.py CRUD endpoints."""
import pytest
from httpx import ASGITransport, AsyncClient
from backend.main import app


@pytest.fixture
def test_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestManufacturers:
    async def test_list_manufacturers(self, test_client):
        resp = await test_client.get("/api/manufacturers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_create_manufacturer(self, test_client):
        resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com",
            "phone": "+123", "location": "Here",
            "materials_supplied": "Copper wire"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Mfr"
        assert data["email"] == "test@test.com"

    async def test_get_manufacturer(self, test_client):
        create_resp = await test_client.post("/api/manufacturers", json={
            "name": "Get Me", "email": "get@test.com"
        })
        mfr_id = create_resp.json()["id"]
        resp = await test_client.get(f"/api/manufacturers/{mfr_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Get Me"

    async def test_get_manufacturer_not_found(self, test_client):
        resp = await test_client.get("/api/manufacturers/999")
        assert resp.status_code == 404

    async def test_update_manufacturer(self, test_client):
        create_resp = await test_client.post("/api/manufacturers", json={
            "name": "Old Name", "email": "old@test.com"
        })
        mfr_id = create_resp.json()["id"]
        resp = await test_client.put(f"/api/manufacturers/{mfr_id}", json={
            "name": "New Name"
        })
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"

    async def test_delete_manufacturer(self, test_client):
        create_resp = await test_client.post("/api/manufacturers", json={
            "name": "Delete Me", "email": "del@test.com"
        })
        mfr_id = create_resp.json()["id"]
        resp = await test_client.delete(f"/api/manufacturers/{mfr_id}")
        assert resp.status_code == 204
        resp = await test_client.get(f"/api/manufacturers/{mfr_id}")
        assert resp.status_code == 404


class TestMaterials:
    async def test_create_material_with_stock(self, test_client):
        # Create a manufacturer first
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]

        resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Copper Wire"
        assert data["stock_quantity"] == 150

    async def test_create_material_default_stock(self, test_client):
        resp = await test_client.post("/api/materials", json={
            "name": "Test Material", "category": "Test",
            "unit_price": 1.0, "unit": "piece"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["stock_quantity"] == 0

    async def test_list_materials(self, test_client):
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
        resp = await test_client.get("/api/materials")
        assert resp.status_code == 200
        materials = resp.json()
        assert len(materials) >= 1
        assert materials[0]["stock_quantity"] == 150

    async def test_get_material(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        create_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = create_resp.json()["id"]
        resp = await test_client.get(f"/api/materials/{mat_id}")
        assert resp.status_code == 200
        assert resp.json()["stock_quantity"] == 150

    async def test_get_material_not_found(self, test_client):
        resp = await test_client.get("/api/materials/999")
        assert resp.status_code == 404

    async def test_update_material_stock(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        create_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = create_resp.json()["id"]
        resp = await test_client.put(f"/api/materials/{mat_id}", json={
            "stock_quantity": 200
        })
        assert resp.status_code == 200
        assert resp.json()["stock_quantity"] == 200

    async def test_delete_material(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        create_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = create_resp.json()["id"]
        resp = await test_client.delete(f"/api/materials/{mat_id}")
        assert resp.status_code == 204
        resp = await test_client.get(f"/api/materials/{mat_id}")
        assert resp.status_code == 404

    async def test_materials_context_endpoint(self, test_client):
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
        resp = await test_client.get("/api/materials/context")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "stock_quantity" in data[0]

    async def test_materials_context_with_category(self, test_client):
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
        await test_client.post("/api/materials", json={
            "name": "Leather Hide", "category": "Leather",
            "unit_price": 35.0, "unit": "sq meter",
            "min_order_qty": 3, "stock_quantity": 20,
            "manufacturer_id": mfr_id
        })
        resp = await test_client.get("/api/materials/context?category=Metal")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["category"] == "Metal"


class TestOrders:
    async def test_create_order(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        mat_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = mat_resp.json()["id"]
        resp = await test_client.post("/api/orders", json={
            "material_id": mat_id, "quantity": 50, "notes": "Urgent"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["material_id"] == mat_id
        assert data["quantity"] == 50
        assert data["status"] == "pending"

    async def test_create_order_below_minimum(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        mat_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = mat_resp.json()["id"]
        resp = await test_client.post("/api/orders", json={
            "material_id": mat_id, "quantity": 5
        })
        assert resp.status_code == 400

    async def test_create_order_missing_material(self, test_client):
        resp = await test_client.post("/api/orders", json={
            "material_id": 999, "quantity": 10
        })
        assert resp.status_code == 404

    async def test_list_orders(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        mat_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = mat_resp.json()["id"]
        await test_client.post("/api/orders", json={
            "material_id": mat_id, "quantity": 50
        })
        resp = await test_client.get("/api/orders")
        assert resp.status_code == 200
        orders = resp.json()
        assert len(orders) >= 1
        assert orders[0]["material_name"] == "Copper Wire"

    async def test_update_order_status(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        mat_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = mat_resp.json()["id"]
        order_resp = await test_client.post("/api/orders", json={
            "material_id": mat_id, "quantity": 50
        })
        order_id = order_resp.json()["id"]
        resp = await test_client.patch(f"/api/orders/{order_id}/status", json={
            "status": "approved"
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    async def test_delete_order(self, test_client):
        mfr_resp = await test_client.post("/api/manufacturers", json={
            "name": "Test Mfr", "email": "test@test.com"
        })
        mfr_id = mfr_resp.json()["id"]
        mat_resp = await test_client.post("/api/materials", json={
            "name": "Copper Wire", "category": "Metal",
            "unit_price": 4.50, "unit": "meter",
            "min_order_qty": 10, "stock_quantity": 150,
            "manufacturer_id": mfr_id
        })
        mat_id = mat_resp.json()["id"]
        order_resp = await test_client.post("/api/orders", json={
            "material_id": mat_id, "quantity": 50
        })
        order_id = order_resp.json()["id"]
        resp = await test_client.delete(f"/api/orders/{order_id}")
        assert resp.status_code == 204


class TestHealth:
    async def test_health(self, test_client):
        resp = await test_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
