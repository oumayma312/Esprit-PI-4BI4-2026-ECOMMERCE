import sqlite3
import os
from contextlib import contextmanager
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "materials.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    conn = _get_conn()
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS manufacturers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                location TEXT,
                materials_supplied TEXT
            );

            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price REAL NOT NULL,
                unit TEXT NOT NULL,
                min_order_qty INTEGER NOT NULL DEFAULT 1,
                stock_quantity INTEGER NOT NULL DEFAULT 0,
                manufacturer_id INTEGER,
                FOREIGN KEY (manufacturer_id) REFERENCES manufacturers(id)
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (material_id) REFERENCES materials(id)
            );

            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                material_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                unit TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id),
                FOREIGN KEY (material_id) REFERENCES materials(id)
            );

            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT UNIQUE,
                channel_id INTEGER,
                address_id INTEGER,
                start_date TEXT,
                end_date TEXT,
                active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS customer_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL UNIQUE,
                churn_probability REAL,
                churn_prediction INTEGER,
                segment INTEGER,
                segment_label TEXT,
                risk_level TEXT,
                recommendation TEXT,
                predicted_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );

            CREATE TABLE IF NOT EXISTS customer_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                transaction_date TEXT,
                total_ttc REAL,
                quantity INTEGER,
                price REAL,
                channelFK INTEGER,
                productFK INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
        """)


# --- Manufacturers ---

def create_manufacturer(name: str, email: str = "", phone: str = "",
                        location: str = "", materials_supplied: str = "") -> dict:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO manufacturers (name, email, phone, location, materials_supplied) VALUES (?,?,?,?,?)",
            (name, email, phone, location, materials_supplied),
        )
        return _manufacturer_row_to_dict(conn, cur.lastrowid)


def list_manufacturers() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM manufacturers ORDER BY name").fetchall()
        return [_manufacturer_row_to_dict(conn, r["id"]) for r in rows]


def get_manufacturer(mfr_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM manufacturers WHERE id = ?", (mfr_id,)).fetchone()
        return _manufacturer_row_to_dict(conn, row["id"]) if row else None


def update_manufacturer(mfr_id: int, **fields) -> Optional[dict]:
    allowed = {"name", "email", "phone", "location", "materials_supplied"}
    fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not fields:
        return None
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [mfr_id]
    with get_db() as conn:
        conn.execute(f"UPDATE manufacturers SET {set_clause} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM manufacturers WHERE id = ?", (mfr_id,)).fetchone()
        return dict(row) if row else None


def delete_manufacturer(mfr_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM manufacturers WHERE id = ?", (mfr_id,))
        return cur.rowcount > 0


# --- Materials ---

def create_material(name: str, category: str, unit_price: float, unit: str,
                    min_order_qty: int = 1, stock_quantity: int = 0, manufacturer_id: Optional[int] = None) -> dict:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO materials (name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id) VALUES (?,?,?,?,?,?,?)",
            (name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id),
        )
        return _material_row_to_dict(conn, cur.lastrowid)


def list_materials() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT m.*, mf.name as manufacturer_name
            FROM materials m
            LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id
            ORDER BY m.name
        """).fetchall()
        return [_material_row_to_dict(conn, r["id"], r) for r in rows]


def get_material(mat_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("""
            SELECT m.*, mf.name as manufacturer_name
            FROM materials m LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id
            WHERE m.id = ?
        """, (mat_id,)).fetchone()
        return _material_row_to_dict(conn, row["id"], row) if row else None


def update_material(mat_id: int, **fields) -> Optional[dict]:
    allowed = {"name", "category", "unit_price", "unit", "min_order_qty", "stock_quantity", "manufacturer_id"}
    fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not fields:
        return None
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [mat_id]
    with get_db() as conn:
        conn.execute(f"UPDATE materials SET {set_clause} WHERE id = ?", values)
        row = conn.execute("SELECT m.*, mf.name as manufacturer_name FROM materials m LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id WHERE m.id = ?", (mat_id,)).fetchone()
        return dict(row) if row else None


def delete_material(mat_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM materials WHERE id = ?", (mat_id,))
        return cur.rowcount > 0


# --- Orders ---

def create_order(material_id: int, quantity: int, notes: str = "") -> dict:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO orders (material_id, quantity, status, notes) VALUES (?,?,?,?)",
            (material_id, quantity, "pending", notes),
        )
        return _order_row_to_dict(conn, cur.lastrowid)


def create_order_items(order_id: int, items: list[dict]) -> list[dict]:
    """Insert order items for a given order. items = [{'material_id': 1, 'quantity': 50, 'unit': 'meter'}, ...]"""
    with get_db() as conn:
        for item in items:
            conn.execute(
                "INSERT INTO order_items (order_id, material_id, quantity, unit) VALUES (?,?,?,?)",
                (order_id, item["material_id"], item["quantity"], item["unit"]),
            )
        return get_order_items(order_id)


def get_order_items(order_id: int) -> list[dict]:
    """Get all items for an order with material details."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT oi.id, oi.order_id, oi.material_id, oi.quantity, oi.unit,
                   m.name as material_name, m.category, m.unit_price, m.unit
            FROM order_items oi
            LEFT JOIN materials m ON oi.material_id = m.id
            WHERE oi.order_id = ?
            ORDER BY oi.id
        """, (order_id,)).fetchall()
        return [dict(r) for r in rows]


def delete_order_items(order_id: int) -> bool:
    """Delete all items for an order."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM order_items WHERE order_id = ?", (order_id,))
        return cur.rowcount > 0


def list_orders() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT o.*, m.name as material_name, m.category, mf.name as manufacturer_name, mf.email as manufacturer_email
            FROM orders o
            LEFT JOIN materials m ON o.material_id = m.id
            LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id
            ORDER BY o.created_at DESC
        """).fetchall()
        return [_order_row_to_dict(conn, r["id"], r) for r in rows]


def get_order(order_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("""
            SELECT o.*, m.name as material_name, m.category, mf.name as manufacturer_name, mf.email as manufacturer_email
            FROM orders o
            LEFT JOIN materials m ON o.material_id = m.id
            LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id
            WHERE o.id = ?
        """, (order_id,)).fetchone()
        return _order_row_to_dict(conn, row["id"], row) if row else None


def update_order_status(order_id: int, status: str) -> Optional[dict]:
    with get_db() as conn:
        conn.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id))
        row = conn.execute("""
            SELECT o.*, m.name as material_name, m.category, mf.name as manufacturer_name, mf.email as manufacturer_email
            FROM orders o LEFT JOIN materials m ON o.material_id = m.id LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id
            WHERE o.id = ?
        """, (order_id,)).fetchone()
        return dict(row) if row else None


def delete_order(order_id: int) -> bool:
    with get_db() as conn:
        delete_order_items(order_id)
        cur = conn.execute("DELETE FROM orders WHERE id = ?", (order_id,))
        return cur.rowcount > 0


# --- Customers ---

def create_customer(name: str, code: str = "", channel_id: Optional[int] = None,
                    address_id: Optional[int] = None, start_date: str = "",
                    end_date: str = "", active: int = 1) -> dict:
    with get_db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO customers (name, code, channel_id, address_id, start_date, end_date, active) VALUES (?,?,?,?,?,?,?)",
                (name, code, channel_id, address_id, start_date, end_date, active),
            )
            return _customer_row_to_dict(conn, cur.lastrowid)
        except sqlite3.IntegrityError:
            # Customer with same code exists — update instead
            row = conn.execute("SELECT * FROM customers WHERE code = ?", (code,)).fetchone()
            if row:
                return dict(row)
            raise


def list_customers(
    segment: Optional[int] = None,
    risk_level: Optional[str] = None,
    search: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    with get_db() as conn:
        query = """
            SELECT c.id, c.name, c.code, c.channel_id, c.active,
                   cp.churn_probability, cp.churn_prediction, cp.segment,
                   cp.segment_label, cp.risk_level, cp.recommendation,
                   cp.predicted_at
            FROM customers c
            LEFT JOIN customer_predictions cp ON c.id = cp.customer_id
        """
        params: list = []
        conditions = []
        if segment is not None:
            conditions.append("cp.segment = ?")
            params.append(segment)
        if risk_level is not None:
            conditions.append("cp.risk_level = ?")
            params.append(risk_level)
        if search:
            conditions.append("c.name LIKE ?")
            params.append(f"%{search}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.name LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [_customer_with_prediction(r) for r in rows]


def get_customer(customer_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        pred = conn.execute(
            "SELECT * FROM customer_predictions WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        result["prediction"] = dict(pred) if pred else None
        txns = conn.execute(
            "SELECT * FROM customer_transactions WHERE customer_id = ? ORDER BY transaction_date DESC",
            (customer_id,),
        ).fetchall()
        result["transactions"] = [dict(t) for t in txns]
        return result


def update_customer(customer_id: int, **fields) -> Optional[dict]:
    allowed = {"name", "code", "channel_id", "address_id", "start_date", "end_date", "active"}
    fields = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not fields:
        return None
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [customer_id]
    with get_db() as conn:
        conn.execute(f"UPDATE customers SET {set_clause} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None


def delete_customer(customer_id: int) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM customer_predictions WHERE customer_id = ?", (customer_id,))
        conn.execute("DELETE FROM customer_transactions WHERE customer_id = ?", (customer_id,))
        cur = conn.execute("DELETE FROM customers WHERE id = ?", (customer_id,))
        return cur.rowcount > 0


def save_prediction(customer_id: int, churn_probability: float, churn_prediction: int,
                    segment: int, segment_label: str, risk_level: str,
                    recommendation: str) -> None:
    """Upsert prediction for a customer."""
    with get_db() as conn:
        conn.execute("""
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
        """, (customer_id, churn_probability, churn_prediction, segment, segment_label, risk_level, recommendation))


def get_customer_prediction(customer_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM customer_predictions WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        return dict(row) if row else None


def save_transaction(customer_id: int, transaction_date: str, total_ttc: float,
                     quantity: int, price: float, channelFK: Optional[int] = None,
                     productFK: Optional[int] = None) -> dict:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO customer_transactions (customer_id, transaction_date, total_ttc, quantity, price, channelFK, productFK) VALUES (?,?,?,?,?,?,?)",
            (customer_id, transaction_date, total_ttc, quantity, price, channelFK, productFK),
        )
        return dict(conn.execute(
            "SELECT * FROM customer_transactions WHERE id = ?", (cur.lastrowid,)
        ).fetchone())


def get_customer_transactions(customer_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM customer_transactions WHERE customer_id = ? ORDER BY transaction_date DESC",
            (customer_id,),
        ).fetchall()
        result = [dict(r) for r in rows]
        # Normalize column names for pipeline compatibility
        for r in result:
            if "channel_id" in r and "channelFK" not in r:
                r["channelFK"] = r.pop("channel_id")
            if "product_id" in r and "productFK" not in r:
                r["productFK"] = r.pop("product_id")
        return result


def delete_customer_transactions(customer_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM customer_transactions WHERE customer_id = ?", (customer_id,))
        return cur.rowcount > 0


def get_high_risk_customers(min_probability: float = 0.7) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT c.id, c.name, c.code, cp.churn_probability, cp.segment,
                   cp.segment_label, cp.risk_level, cp.recommendation
            FROM customers c
            JOIN customer_predictions cp ON c.id = cp.customer_id
            WHERE cp.churn_probability >= ?
            ORDER BY cp.churn_probability DESC
        """, (min_probability,)).fetchall()
        return [_customer_with_prediction(r) for r in rows]


def get_segment_stats() -> list[dict]:
    """Return stats per segment: count, avg probability, total."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT cp.segment, cp.segment_label,
                   COUNT(*) as count,
                   ROUND(AVG(cp.churn_probability), 4) as avg_probability,
                   ROUND(MIN(cp.churn_probability), 4) as min_probability,
                   ROUND(MAX(cp.churn_probability), 4) as max_probability
            FROM customer_predictions cp
            GROUP BY cp.segment
            ORDER BY cp.segment
        """).fetchall()
        return [dict(r) for r in rows]


def _customer_row_to_dict(conn, customer_id: int) -> dict:
    row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    if not row:
        return {}
    return dict(row)


def _customer_with_prediction(row) -> dict:
    """Convert a row from the joined customers+predictions query to dict."""
    if not row:
        return {}
    result = dict(row)
    # Normalize churn_prediction to string for API consistency
    if "churn_prediction" in result and result["churn_prediction"] is not None:
        result["churn_prediction"] = "Churné" if int(result["churn_prediction"]) == 1 else "Actif"
    return result


# --- Helpers ---

def _manufacturer_row_to_dict(conn, mfr_id: int) -> dict:
    row = conn.execute("SELECT * FROM manufacturers WHERE id = ?", (mfr_id,)).fetchone()
    if not row:
        return {}
    return dict(row)


def _material_row_to_dict(conn, mat_id: int, row=None) -> dict:
    if row is None:
        row = conn.execute("SELECT m.*, mf.name as manufacturer_name FROM materials m LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id WHERE m.id = ?", (mat_id,)).fetchone()
    if not row:
        return {}
    return dict(row)


def seed_tunisian_materials():
    """Seed the database with Tunisian artisan materials and manufacturers."""
    with get_db() as conn:
        # Delete all existing materials and manufacturers
        conn.execute("DELETE FROM order_items")
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM materials")
        conn.execute("DELETE FROM manufacturers")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('manufacturers', 'materials')")

        # Insert manufacturers
        manufacturers = [
            ("Tunisian Glass Works", "contact@glassworks.tn", "", "Sidi Bou Said", "Glass sheets, glass bottles, glass panels"),
            ("Atelier de Ceramique Nabeul", "contact@ceramique-nabeul.tn", "", "Nabeul", "Ceramic clay, glazes, underglazes"),
            ("Tunisian Copper Atelier", "contact@copper-atelier.tn", "", "Medina", "Copper sheets, tubing, wire"),
            ("Kasserine Olive Wood Coop", "contact@olive-wood.tn", "", "Kasserine", "Olive wood blocks, blanks"),
            ("Tunisian Leather Tannery", "contact@leather-tannery.tn", "", "El Jem", "Vegetable and chrome tanned leather"),
        ]
        mfr_ids = []
        for name, email, phone, location, materials_supplied in manufacturers:
            cur = conn.execute(
                "INSERT INTO manufacturers (name, email, phone, location, materials_supplied) VALUES (?,?,?,?,?)",
                (name, email, phone, location, materials_supplied),
            )
            mfr_ids.append(cur.lastrowid)

        # Insert artisan materials
        materials = [
            # Glass (3) - for bonbonnieres, perfume bottles
            ("Crackle Glass Vessel 80ml", "Glass", 18.00, "piece", 10, 100, mfr_ids[0]),
            ("Glass Blown Bottles Clear", "Glass", 8.00, "piece", 12, 60, mfr_ids[0]),
            ("Glass Perfume Bottle 10ml", "Glass", 4.00, "piece", 20, 100, mfr_ids[0]),
            # Metal (8) - for bonbonnieres lids, lunch boxes, thermos
            ("Copper Sheet 0.5mm martele", "Metal", 12.00, "sheet", 10, 80, mfr_ids[2]),
            ("Copper Hinge Wire 1mm", "Metal", 0.50, "10cm", 50, 500, mfr_ids[2]),
            ("Stainless Steel Container 500ml", "Metal", 12.00, "piece", 10, 60, mfr_ids[2]),
            ("Silicone Seal Ring", "Metal", 0.30, "piece", 100, 1000, mfr_ids[2]),
            ("Double-wall Steel Tumbler 500ml", "Metal", 10.00, "piece", 10, 60, mfr_ids[2]),
            ("Silicone Lid with Straw", "Metal", 1.50, "piece", 20, 100, mfr_ids[2]),
            ("Bamboo Straw", "Metal", 0.50, "piece", 50, 500, mfr_ids[2]),
            ("Copper Brazier (mathred)", "Metal", 15.00, "piece", 10, 40, mfr_ids[2]),
            # Ceramic (4) - for mug sets, plates
            ("Ceramic Mug Blank (pre-formed)", "Ceramic", 3.00, "piece", 20, 200, mfr_ids[1]),
            ("Ceramic Plate Blank", "Ceramic", 2.00, "piece", 20, 200, mfr_ids[1]),
            ("Ceramic Glaze (white, food-safe)", "Ceramic", 5.00, "bottle", 10, 80, mfr_ids[1]),
            ("Food-safe Paint (red/blue/yellow)", "Ceramic", 1.50, "bottle", 20, 100, mfr_ids[1]),
            # Textile (3) - for tote bags, lunch box pouches
            ("Cotton Canvas Pouch (30x25cm)", "Textile", 1.50, "piece", 20, 200, mfr_ids[4]),
            ("Zipper Closure 15cm", "Textile", 0.50, "piece", 50, 500, mfr_ids[4]),
            ("Screen Print Ink (1-color)", "Textile", 1.00, "piece", 20, 200, mfr_ids[4]),
            # Wood (4) - for games, Mouton figurine
            ("Plywood Board 3mm (30x20cm)", "Wood", 4.00, "piece", 20, 100, mfr_ids[3]),
            ("Wooden Game Tokens (set 12)", "Wood", 2.00, "set", 20, 100, mfr_ids[3]),
            ("Acrylic Paint (wood, assorted)", "Wood", 2.25, "set", 20, 100, mfr_ids[3]),
            ("Clear Wood Varnish (matte)", "Wood", 1.00, "bottle", 20, 100, mfr_ids[3]),
            ("Hardwood Carving Block", "Wood", 8.00, "piece", 10, 40, mfr_ids[3]),
            # Fragrance (1) - for Coffret Mathred
            ("Perfume Oil/Musk (ml)", "Fragrance", 0.80, "ml", 50, 500, mfr_ids[4]),
            # Accessory (1) - for Coffret Mathred
            ("Prayer Beads (wooden, 33)", "Accessory", 5.00, "set", 20, 100, mfr_ids[4]),
            # Packaging (2) - for all products
            ("Kraft Paper Sheet (15x10cm)", "Packaging", 0.15, "piece", 100, 2000, mfr_ids[0]),
            ("Gift Box Small (branded)", "Packaging", 2.50, "piece", 20, 200, mfr_ids[0]),
        ]
        for name, category, unit_price, unit, min_order_qty, stock_qty, mfr_id in materials:
            conn.execute(
                "INSERT INTO materials (name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id) VALUES (?,?,?,?,?,?,?)",
                (name, category, unit_price, unit, min_order_qty, stock_qty, mfr_id),
            )


def seed_data():
    """Seed the database with Tunisian artisan materials and manufacturers."""
    seed_tunisian_materials()


def _order_row_to_dict(conn, order_id: int, row=None) -> dict:
    if row is None:
        row = conn.execute("""
            SELECT o.*, m.name as material_name, m.category, mf.name as manufacturer_name, mf.email as manufacturer_email
            FROM orders o LEFT JOIN materials m ON o.material_id = m.id LEFT JOIN manufacturers mf ON m.manufacturer_id = mf.id
            WHERE o.id = ?
        """, (order_id,)).fetchone()
    if not row:
        return {}
    return dict(row)
