import sys
import pytest
import sqlite3
import os
import tempfile
from pathlib import Path

# Ensure backend module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import db


@pytest.fixture(autouse=True)
def _setup_test_db():
    """Use a temp DB for every test run."""
    db.DB_PATH = os.path.join(tempfile.gettempdir(), "test_materials.db")
    db.init_db()
    # Clear any seed data
    with db.get_db() as conn:
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM materials")
        conn.execute("DELETE FROM manufacturers")
        conn.execute("DELETE FROM customer_predictions")
        conn.execute("DELETE FROM customer_transactions")
        conn.execute("DELETE FROM customers")
    # Seed test customers with predictions
    _seed_test_customers()
    yield
    # Cleanup
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)


def _seed_test_customers():
    """Add test customers with predictions for integration tests."""
    c1 = db.create_customer(name="Attijari Bank Test", code="TEST001", channel_id=1)
    db.save_prediction(c1["id"], 0.75, 1, 2, "Intermediaires a risque", "Eleve", "Action de retention immediate.")
    c2 = db.create_customer(name="VIP Client Test", code="TEST002", channel_id=1)
    db.save_prediction(c2["id"], 0.15, 0, 0, "Loyaux a forte valeur", "Faible", "Conserver la fidelite.")
    c3 = db.create_customer(name="Lost Client Test", code="TEST003", channel_id=2)
    db.save_prediction(c3["id"], 0.85, 1, 1, "Clients perdus", "Eleve", "Action de retention immediate.")
    c4 = db.create_customer(name="New Client Test", code="TEST004", channel_id=4)
    db.save_prediction(c4["id"], 0.90, 1, 3, "Clients recents ou nouveaux", "Eleve", "Accelerer l'onboarding.")
    c5 = db.create_customer(name="At-Risk Client Test", code="TEST005", channel_id=1)
    db.save_prediction(c5["id"], 0.55, 0, 2, "Intermediaires a risque", "Modere", "A suivre de pres.")


@pytest.fixture
def mfr_id():
    """Create a manufacturer and return its ID."""
    m = db.create_manufacturer(
        name="Test Copper Works",
        email="test@example.com",
        phone="+1234567890",
        location="Test City",
        materials_supplied="Copper wire, copper sheets"
    )
    return m["id"]


@pytest.fixture
def material_id(mfr_id):
    """Create a material with stock and return its ID."""
    m = db.create_material(
        name="Copper Wire 2mm",
        category="Metal",
        unit_price=4.50,
        unit="meter",
        min_order_qty=10,
        stock_quantity=150,
        manufacturer_id=mfr_id
    )
    return m["id"]
