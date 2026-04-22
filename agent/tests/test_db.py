"""Unit tests for backend/db.py — stock_quantity column and CRUD."""
import pytest
from backend import db


class TestInitDB:
    def test_creates_all_tables(self):
        db.init_db()
        with db.get_db() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            names = [t["name"] for t in tables]
        assert "manufacturers" in names
        assert "materials" in names
        assert "orders" in names

    def test_materials_table_has_stock_quantity(self):
        db.init_db()
        with db.get_db() as conn:
            cols = conn.execute("PRAGMA table_info(materials)").fetchall()
            col_names = [c["name"] for c in cols]
        assert "stock_quantity" in col_names


class TestCreateMaterial:
    def test_creates_with_default_stock(self):
        m = db.create_material(
            name="Test Material", category="Test", unit_price=1.0, unit="piece"
        )
        assert m["id"] is not None
        assert m["name"] == "Test Material"
        assert m["stock_quantity"] == 0

    def test_creates_with_explicit_stock(self):
        m = db.create_material(
            name="Test Material", category="Test", unit_price=1.0, unit="piece",
            stock_quantity=42
        )
        assert m["stock_quantity"] == 42

    def test_creates_with_manufacturer(self, mfr_id):
        m = db.create_material(
            name="Copper Wire", category="Metal", unit_price=4.50, unit="meter",
            min_order_qty=10, stock_quantity=150, manufacturer_id=mfr_id
        )
        assert m["manufacturer_id"] == mfr_id

    def test_min_order_qty_defaults_to_1(self):
        m = db.create_material(
            name="Test", category="Test", unit_price=1.0, unit="piece"
        )
        assert m["min_order_qty"] == 1


class TestListMaterials:
    def test_returns_all_materials(self, material_id):
        # Create a second material
        db.create_material(
            name="Leather Hide", category="Leather", unit_price=35.0, unit="sq meter",
            stock_quantity=20
        )
        materials = db.list_materials()
        assert len(materials) == 2

    def test_includes_stock_quantity(self, material_id):
        materials = db.list_materials()
        assert materials[0]["stock_quantity"] == 150

    def test_includes_manufacturer_name(self, material_id):
        materials = db.list_materials()
        assert materials[0]["manufacturer_name"] == "Test Copper Works"


class TestGetMaterial:
    def test_returns_material(self, material_id):
        m = db.get_material(material_id)
        assert m["id"] == material_id
        assert m["stock_quantity"] == 150

    def test_returns_none_for_missing(self):
        assert db.get_material(999) is None


class TestUpdateMaterial:
    def test_updates_stock_quantity(self, material_id):
        result = db.update_material(material_id, stock_quantity=200)
        assert result["stock_quantity"] == 200

    def test_updates_name(self, material_id):
        result = db.update_material(material_id, name="Copper Wire 3mm")
        assert result["name"] == "Copper Wire 3mm"

    def test_updates_multiple_fields(self, material_id):
        result = db.update_material(material_id, stock_quantity=300, unit_price=5.00)
        assert result["stock_quantity"] == 300
        assert result["unit_price"] == 5.00


class TestDeleteMaterial:
    def test_deletes_material(self, material_id):
        assert db.delete_material(material_id) is True
        assert db.get_material(material_id) is None

    def test_returns_false_for_missing(self):
        assert db.delete_material(999) is False


class TestCreateManufacturer:
    def test_creates_manufacturer(self):
        m = db.create_manufacturer(
            name="Test Mfr", email="test@test.com",
            phone="+123", location="Here", materials_supplied="Stuff"
        )
        assert m["id"] is not None
        assert m["name"] == "Test Mfr"


class TestCreateOrder:
    def test_creates_order(self, material_id):
        o = db.create_order(material_id, quantity=50, notes="Urgent")
        assert o["id"] is not None
        assert o["material_id"] == material_id
        assert o["quantity"] == 50
        assert o["status"] == "pending"
        assert o["notes"] == "Urgent"

    def test_order_defaults_to_pending(self, material_id):
        o = db.create_order(material_id, quantity=10)
        assert o["status"] == "pending"


class TestListOrders:
    def test_returns_orders_with_material_info(self, material_id):
        db.create_order(material_id, quantity=50)
        orders = db.list_orders()
        assert len(orders) == 1
        assert orders[0]["material_name"] == "Copper Wire 2mm"
        assert orders[0]["manufacturer_name"] == "Test Copper Works"


class TestSeedDB:
    def test_seed_creates_manufacturers_and_materials(self):
        from backend.tools import seed_db
        # Clear first
        with db.get_db() as conn:
            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM materials")
            conn.execute("DELETE FROM manufacturers")
        seed_db()
        mfrs = db.list_manufacturers()
        mats = db.list_materials()
        assert len(mfrs) == 5
        assert len(mats) == 27
        # Verify stock_quantity is set
        for mat in mats:
            assert mat["stock_quantity"] is not None
            assert mat["stock_quantity"] >= 0
