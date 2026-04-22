"""Tests for ML pipeline database operations — specifically testing for SQLite locking issues.

These tests verify that saving 443 predictions (simulating a real ML pipeline run)
does NOT trigger "database is locked" errors.
"""
import sys
import os
import tempfile
from pathlib import Path

import pytest

# Ensure backend module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import db


@pytest.fixture(autouse=True)
def _setup_ml_test_db():
    """Use a temp DB for ML pipeline tests."""
    db.DB_PATH = os.path.join(tempfile.gettempdir(), "test_ml_pipeline.db")
    db.init_db()
    # Seed 500 customers to cover all test scenarios
    for i in range(1, 501):
        c = db.create_customer(name=f"Customer {i}", code=f"CUST_{i:04d}", channel_id=1)
    yield
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)


class TestMLPipelineDBNoLock:
    """Test that ML pipeline DB operations don't cause 'database is locked' errors."""

    def test_save_443_predictions_no_lock(self):
        """Simulate ML pipeline saving 443 predictions — should NOT raise 'database is locked'."""
        errors = []
        for i in range(1, 444):
            try:
                db.save_prediction(
                    customer_id=i,
                    churn_probability=0.5 + (i % 100) / 200,
                    churn_prediction=i % 2,
                    segment=i % 4,
                    segment_label=f"Segment {i % 4}",
                    risk_level="Eleve" if i % 3 == 0 else "Faible",
                    recommendation="Test recommendation",
                )
            except Exception as e:
                errors.append(f"Customer {i}: {e}")

        assert not errors, f"Failed to save predictions:\n" + "\n".join(errors[:10])

    def test_save_443_transactions_no_lock(self):
        """Simulate ML pipeline saving 2342 transactions — should NOT raise 'database is locked'."""
        errors = []
        for i in range(1, 2343):
            customer_id = (i % 50) + 1
            try:
                db.save_transaction(
                    customer_id=customer_id,
                    transaction_date=f"2025010{i % 10}",
                    total_ttc=100.0 + i,
                    quantity=1 + (i % 5),
                    price=50.0 + (i % 100),
                    channelFK=1,
                    productFK=i % 20,
                )
            except Exception as e:
                errors.append(f"Transaction {i}: {e}")

        assert not errors, f"Failed to save transactions:\n" + "\n".join(errors[:10])

    def test_save_prediction_and_transaction_same_customer(self):
        """Test saving prediction + transactions for the same customer in sequence."""
        customer_id = 1
        # Save prediction
        db.save_prediction(
            customer_id=customer_id,
            churn_probability=0.75,
            churn_prediction=1,
            segment=2,
            segment_label="Intermediaires a risque",
            risk_level="Eleve",
            recommendation="Test",
        )
        # Save 10 transactions
        for i in range(10):
            db.save_transaction(
                customer_id=customer_id,
                transaction_date=f"2025010{i}",
                total_ttc=100.0 + i,
                quantity=1,
                price=50.0 + i,
                channelFK=1,
                productFK=1,
            )

        # Verify data was saved
        pred = db.get_customer_prediction(customer_id)
        assert pred is not None
        assert pred["churn_probability"] == 0.75

        txns = db.get_customer_transactions(customer_id)
        assert len(txns) == 10

    def test_batch_save_predictions_with_executemany(self):
        """Test using executemany for batch prediction saves (more efficient)."""
        predictions = []
        for i in range(1, 444):
            predictions.append((
                i,
                0.5 + (i % 100) / 200,
                i % 2,
                i % 4,
                f"Segment {i % 4}",
                "Eleve" if i % 3 == 0 else "Faible",
                "Test recommendation",
            ))

        with db.get_db() as conn:
            conn.executemany("""
                INSERT INTO customer_predictions
                    (customer_id, churn_probability, churn_prediction, segment, segment_label, risk_level, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    churn_probability = excluded.churn_probability,
                    churn_prediction = excluded.churn_prediction,
                    segment = excluded.segment,
                    segment_label = excluded.segment_label,
                    risk_level = excluded.risk_level,
                    recommendation = excluded.recommendation,
                    predicted_at = datetime('now')
            """, predictions)
            count = conn.execute("SELECT COUNT(*) as cnt FROM customer_predictions").fetchone()["cnt"]

        assert count == 443
