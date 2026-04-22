"""Import raw data warehouse CSVs into the materials SQLite database.

Reads DimCustomer.csv and FactVentee.csv, imports customers and transactions,
then runs the ML pipeline to generate predictions.

Usage:
    cd /home/mohamed/projects/koudos && python -m agent.backend.import_data
    cd /home/mohamed/projects/koudos && .venv/bin/python agent/backend/import_data.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("koudos.import")

# Ensure project root and agent directory are importable
_project_root = Path(__file__).resolve().parent.parent.parent  # /home/mohamed/projects/koudos
_agent_dir = Path(__file__).resolve().parent.parent            # /home/mohamed/projects/koudos/agent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_agent_dir))

from ml_pipeline import get_pipeline
from backend import db


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


def import_customers(csv_path: str | Path = None) -> int:
    """Import DimCustomer.csv into customers table.

    Returns the number of customers imported/updated.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "DimCustomer.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"Customer CSV not found: {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} rows in {csv_path.name}")

    count = 0
    errors = 0
    for _, row in df.iterrows():
        code = str(row.get("customerCode", "")).strip()
        name = str(row.get("customer", "")).strip()
        if not name:
            continue

        channel_id = row.get("channelFK")
        try:
            channel_id = int(channel_id) if pd.notna(channel_id) else None
        except (ValueError, TypeError):
            channel_id = None

        address_id = row.get("adressFK")
        try:
            address_id = int(address_id) if pd.notna(address_id) else None
        except (ValueError, TypeError):
            address_id = None

        start_date = str(row.get("startDate", "")) if pd.notna(row.get("startDate")) else ""
        end_date = str(row.get("endDate", "")) if pd.notna(row.get("endDate")) else ""

        try:
            db.create_customer(
                name=name,
                code=code if code else None,
                channel_id=channel_id,
                address_id=address_id,
                start_date=start_date,
                end_date=end_date,
                active=1 if row.get("flag") == "y" else 0,
            )
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Could not import customer '{name}': {e}")

    logger.info(f"Imported/updated {count} customers ({errors} errors)")
    return count


def import_transactions(csv_path: str | Path = None) -> int:
    """Import FactVentee.csv into customer_transactions table.

    Returns the number of transactions imported.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "FactVentee.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"Transaction CSV not found: {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} rows in {csv_path.name}")

    # Parse dates
    strict_dates = pd.to_datetime(df["dateFK"].astype(str), format="%Y%m%d", errors="coerce")
    loose_dates = pd.to_datetime(df["dateFK"], errors="coerce")
    df["dateFK"] = strict_dates.fillna(loose_dates)
    invalid_dates = int(df["dateFK"].isna().sum())
    df = df.dropna(subset=["dateFK"]).copy()
    logger.info(f"Parsed dates: {invalid_dates} invalid rows removed, {len(df)} valid rows remain")

    # Coerce numeric
    for col in ["total_ttc", "quantity", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Map productFK/channelFK to names if dimension tables exist
    product_lookup = {}
    product_path = DATA_DIR / "DimProduct.csv"
    if product_path.exists():
        prod_df = pd.read_csv(product_path)
        for _, r in prod_df.iterrows():
            product_lookup[int(r["productPK"])] = str(r.get("product", ""))
        logger.info(f"Loaded {len(product_lookup)} product lookups")

    count = 0
    skipped = 0
    for _, row in df.iterrows():
        customer_fk = row.get("customerFK")
        if pd.isna(customer_fk):
            continue

        # Look up customer by customerFK (which is the PK in DimCustomer)
        customer = db.get_customer(int(customer_fk))
        if not customer:
            skipped += 1
            continue

        txn_date = str(row["dateFK"])
        if isinstance(txn_date, pd.Timestamp):
            txn_date = txn_date.strftime("%Y-%m-%d")

        try:
            db.save_transaction(
                customer_id=customer["id"],
                transaction_date=txn_date,
                total_ttc=float(row["total_ttc"]),
                quantity=int(row["quantity"]),
                price=float(row["price"]),
                channelFK=int(row["channelFK"]) if pd.notna(row.get("channelFK")) else None,
                productFK=int(row["productFK"]) if pd.notna(row.get("productFK")) else None,
            )
            count += 1
        except Exception as e:
            pass  # Skip problematic rows silently

    logger.info(f"Imported {count} transactions ({skipped} skipped - customer not found)")
    return count


def run_predictions() -> int:
    """Run ML pipeline on all customers and save predictions.

    Returns the number of predictions saved.
    """
    logger.info("Loading ML pipeline...")
    pipeline = get_pipeline(str(ARTIFACTS_DIR))
    logger.info("ML pipeline loaded, running predictions...")

    # Get all customers with transactions
    all_customers = db.list_customers(limit=10000)
    if not all_customers:
        logger.error("No customers found. Run import_customers() first.")
        return 0

    logger.info(f"Running predictions for {len(all_customers)} customers...")

    # Build feature dataframe from transactions
    features_list = []
    failed = 0
    for i, cust in enumerate(all_customers):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(all_customers)} customers processed")
        txns = db.get_customer_transactions(cust["id"])
        if not txns:
            continue

        tx_df = pd.DataFrame(txns)
        try:
            feat_df, issues = pipeline.derive_features_from_transactions(tx_df)
            if feat_df.empty:
                continue

            feat_row = feat_df.iloc[0].to_dict()
            feat_row["channel"] = ""  # Will be encoded by pipeline

            result = pipeline.predict_customer(feat_row)
            features_list.append({
                "customer_id": cust["id"],
                **result,
            })
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"  Prediction failed for customer {cust['id']}: {e}")
            continue

    # Save predictions
    saved = 0
    for fp in features_list:
        try:
            db.save_prediction(
                customer_id=fp["customer_id"],
                churn_probability=fp["churn_probability"],
                churn_prediction=int(fp["churn_prediction"]),
                segment=fp["segment"],
                segment_label=fp["segment_label"],
                risk_level=fp["risk_level"],
                recommendation=fp["recommendation"],
            )
            saved += 1
        except Exception as e:
            logger.warning(f"  Save failed for customer {fp['customer_id']}: {e}")

    # Summary
    churned = sum(1 for fp in features_list if fp["churn_prediction"] == "1")
    active = len(features_list) - churned
    logger.info(f"Predictions complete: {saved} saved, {churned} churned, {active} active, {failed} failed")
    return saved


def import_all() -> None:
    """Run full import pipeline: customers → transactions → predictions."""
    print("=" * 60)
    print("Data Import Pipeline")
    print("=" * 60)

    db.init_db()
    import_customers()
    import_transactions()
    run_predictions()

    # Summary
    customers = db.list_customers(limit=10000)
    stats = db.get_segment_stats()
    print(f"\n{'=' * 60}")
    print(f"Summary: {len(customers)} customers imported")
    if stats:
        for s in stats:
            print(f"  Segment {s['segment']} ({s['segment_label']}): "
                  f"{s['count']} customers, avg risk {s['avg_probability']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    import_all()
