"""Unit tests for backend/tools.py — tool functions."""
import pytest
from unittest.mock import patch, MagicMock
from backend.tools import (
    search_materials,
    add_material,
    create_order,
    list_orders,
)


def _invoke_tool(tool, **kwargs):
    """Helper to invoke LangChain tools with kwargs."""
    input_dict = {"input": ""}
    input_dict.update(kwargs)
    return tool.invoke(input_dict)


class TestSearchMaterials:
    def test_returns_all_when_no_query(self):
        mock_materials = [
            {"id": 1, "name": "Copper Wire", "category": "Metal",
             "unit_price": 4.50, "unit": "meter", "min_order_qty": 10,
             "stock_quantity": 150, "manufacturer_name": "Test Mfr"}
        ]
        with patch("backend.tools.db.list_materials", return_value=mock_materials):
            result = _invoke_tool(search_materials)
        assert "Copper Wire" in result
        assert "150" in result  # stock_quantity in output
        assert "4.5" in result  # price format

    def test_filters_by_name(self):
        mock_materials = [
            {"id": 1, "name": "Copper Wire", "category": "Metal",
             "unit_price": 4.50, "unit": "meter", "min_order_qty": 10,
             "stock_quantity": 150, "manufacturer_name": None},
            {"id": 2, "name": "Leather Hide", "category": "Leather",
             "unit_price": 35.0, "unit": "sq meter", "min_order_qty": 3,
             "stock_quantity": 20, "manufacturer_name": None},
        ]
        with patch("backend.tools.db.list_materials", return_value=mock_materials):
            result = _invoke_tool(search_materials, query="copper")
        assert "Copper Wire" in result
        assert "Leather Hide" not in result

    def test_filters_by_category(self):
        mock_materials = [
            {"id": 1, "name": "Copper Wire", "category": "Metal",
             "unit_price": 4.50, "unit": "meter", "min_order_qty": 10,
             "stock_quantity": 150, "manufacturer_name": None},
            {"id": 2, "name": "Leather Hide", "category": "Leather",
             "unit_price": 35.0, "unit": "sq meter", "min_order_qty": 3,
             "stock_quantity": 20, "manufacturer_name": None},
        ]
        with patch("backend.tools.db.list_materials", return_value=mock_materials):
            result = _invoke_tool(search_materials, query="leather")
        assert "Leather Hide" in result
        assert "Copper Wire" not in result

    def test_returns_no_materials_message(self):
        with patch("backend.tools.db.list_materials", return_value=[]):
            result = _invoke_tool(search_materials, query="nonexistent")
        assert "No materials found" in result


class TestAddMaterial:
    def test_adds_material_with_stock(self):
        mock_result = {"id": 1, "name": "Test Material", "stock_quantity": 50, "unit_price": 10.0, "unit": "piece"}
        with patch("backend.tools.db.create_material", return_value=mock_result):
            result = _invoke_tool(add_material, name="Test Material", category="Test",
                                  unit_price=10.0, unit="piece", min_order_qty=5,
                                  stock_quantity=50)
        assert "Test Material" in result
        assert "stock: 50" in result

    def test_adds_material_with_default_stock(self):
        mock_result = {"id": 1, "name": "Test Material", "stock_quantity": 0, "unit_price": 10.0, "unit": "piece"}
        with patch("backend.tools.db.create_material", return_value=mock_result):
            result = _invoke_tool(add_material, name="Test Material", category="Test",
                                  unit_price=10.0, unit="piece")
        assert "stock: 0" in result


class TestCreateOrder:
    def test_creates_order_single_material(self):
        mock_mat = {"id": 1, "name": "Copper Wire", "min_order_qty": 10, "unit": "meter"}
        mock_order = {"id": 1, "material_id": 1, "quantity": 50}
        with patch("backend.tools.db.get_material", return_value=mock_mat):
            with patch("backend.tools.db.create_order", return_value=mock_order):
                with patch("backend.tools.db.create_order_items"):
                    result = _invoke_tool(create_order, materials=[{"material_id": 1, "quantity": 50}])
        assert "Order created" in result
        assert "Copper Wire" in result
        assert "x50" in result

    def test_creates_order_multiple_materials(self):
        mock_mat1 = {"id": 1, "name": "Copper Wire", "min_order_qty": 10, "unit": "meter"}
        mock_mat2 = {"id": 2, "name": "Leather", "min_order_qty": 3, "unit": "sq meter"}
        mock_order = {"id": 1, "material_id": 1, "quantity": 50}
        with patch("backend.tools.db.get_material", side_effect=[mock_mat1, mock_mat2]):
            with patch("backend.tools.db.create_order", return_value=mock_order):
                with patch("backend.tools.db.create_order_items"):
                    result = _invoke_tool(create_order, materials=[{"material_id": 1, "quantity": 50}, {"material_id": 2, "quantity": 20}])
        assert "Order created" in result
        assert "Copper Wire" in result
        assert "Leather" in result
        assert "x50" in result
        assert "x20" in result

    def test_rejects_below_minimum(self):
        mock_mat = {"id": 1, "name": "Copper Wire", "min_order_qty": 10}
        with patch("backend.tools.db.get_material", return_value=mock_mat):
            result = _invoke_tool(create_order, materials=[{"material_id": 1, "quantity": 5}])
        assert "below minimum" in result

    def test_rejects_missing_material(self):
        with patch("backend.tools.db.get_material", return_value=None):
            result = _invoke_tool(create_order, materials=[{"material_id": 999, "quantity": 10}])
        assert "not found" in result

    def test_rejects_empty_materials(self):
        result = _invoke_tool(create_order, materials=[])
        assert "No materials" in result


class TestListOrders:
    def test_returns_orders(self):
        mock_orders = [
            {
                "id": 1, "material_name": "Copper Wire", "quantity": 50,
                "status": "pending", "manufacturer_name": "Test Mfr",
                "manufacturer_email": "test@test.com", "notes": "Urgent"
            }
        ]
        with patch("backend.tools.db.list_orders", return_value=mock_orders):
            result = _invoke_tool(list_orders)
        assert "Order #1" in result
        assert "Copper Wire" in result
        assert "50" in result
        assert "pending" in result
        assert "Urgent" in result

    def test_returns_empty(self):
        with patch("backend.tools.db.list_orders", return_value=[]):
            result = _invoke_tool(list_orders)
        assert "No orders found" in result
