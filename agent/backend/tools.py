import os
from typing import Optional

from langchain.tools import tool

from . import db


@tool
def search_materials(query: str = "") -> str:
    """Search materials in the database. Pass a query string to filter by name or category. Returns matching materials with prices and manufacturer info."""
    materials = db.list_materials()
    if query:
        q = query.lower()
        materials = [m for m in materials if q in m["name"].lower() or q in m["category"].lower()]
    if not materials:
        return "No materials found."
    lines = [f"Material ID: {m['id']} | {m['name']} | Category: {m['category']} | Price: ${m['unit_price']}/{m['unit']} | Stock: {m.get('stock_quantity', 0)} | Min Order: {m['min_order_qty']} {m['unit']}"
             + (f" | Manufacturer: {m.get('manufacturer_name', 'N/A')}" if m.get('manufacturer_name') else "")
             for m in materials]
    return "\n".join(lines)


@tool
def add_material(name: str, category: str, unit_price: float, unit: str,
                 min_order_qty: int = 1, stock_quantity: int = 0, manufacturer_id: Optional[int] = None) -> str:
    """Add a new material to the inventory. Requires: name, category, unit_price, unit. Optional: min_order_qty (default 1), stock_quantity (default 0), manufacturer_id."""
    try:
        m = db.create_material(name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id)
        return f"Created material: ID={m['id']}, {m['name']} (${m['unit_price']}/{m['unit']}, stock: {m['stock_quantity']})"
    except Exception as e:
        return f"Error creating material: {e}"


@tool
def update_material(mat_id: int, **fields) -> str:
    """Update an existing material. Pass fields as key=value pairs: name, category, unit_price, unit, min_order_qty, manufacturer_id."""
    try:
        m = db.update_material(mat_id, **fields)
        if not m:
            return f"Material {mat_id} not found or no fields to update."
        return f"Updated material {mat_id}: {m['name']} (${m['unit_price']}/{m['unit']})"
    except Exception as e:
        return f"Error updating material: {e}"


@tool
def delete_material(mat_id: int) -> str:
    """Delete a material from the inventory. WARNING: This is irreversible."""
    try:
        if db.delete_material(mat_id):
            return f"Deleted material {mat_id}."
        return f"Material {mat_id} not found."
    except Exception as e:
        return f"Error deleting material: {e}"


@tool
def list_orders() -> str:
    """List all orders with their status, materials, and quantities."""
    orders = db.list_orders()
    if not orders:
        return "No orders found."
    lines = [f"Order #{o['id']} | {o.get('material_name', 'Unknown')} | Qty: {o['quantity']} | Status: {o['status']}"
             + (f" | Mfr: {o.get('manufacturer_name', 'N/A')}" if o.get('manufacturer_name') else "")
             + (f" | Email: {o.get('manufacturer_email', 'N/A')}" if o.get('manufacturer_email') else "")
             + (f" | {o['notes']}" if o.get('notes') else "")
             for o in orders]
    return "\n".join(lines)


@tool
def get_order(order_id: int) -> str:
    """Get details of a specific order by ID."""
    o = db.get_order(order_id)
    if not o:
        return f"Order {order_id} not found."
    items = db.get_order_items(order_id)
    item_strs = [f"{i['material_name']} x{i['quantity']} {i['unit']}" for i in items]
    items_detail = ", ".join(item_strs) if item_strs else f"{o.get('material_name', 'Unknown')} x{o['quantity']}"
    return (f"Order #{o['id']} | Items: {items_detail}"
            f" | Status: {o['status']} | Created: {o['created_at']}"
            + (f" | Notes: {o['notes']}" if o.get('notes') else ""))


@tool
def create_order(materials: list[dict], notes: str = "") -> str:
    """Create a new order for one or more materials. materials is a list of dicts: [{'material_id': 1, 'quantity': 50}, {'material_id': 2, 'quantity': 20}]. Optional: notes for the order."""
    try:
        if not materials:
            return "No materials provided for the order."

        validated_items = []
        item_descriptions = []
        first_mat = None

        for item in materials:
            mat_id = item.get("material_id")
            quantity = item.get("quantity", 1)
            if not mat_id:
                return f"Missing material_id in one of the order items."
            mat = db.get_material(mat_id)
            if not mat:
                return f"Material {mat_id} not found."
            if quantity < mat["min_order_qty"]:
                return f"Quantity {quantity} is below minimum order qty ({mat['min_order_qty']}) for {mat['name']}."
            validated_items.append({
                "material_id": mat_id,
                "quantity": quantity,
                "unit": mat["unit"],
            })
            item_descriptions.append(f"{mat['name']} x{quantity}")
            if first_mat is None:
                first_mat = mat

        if not first_mat:
            return "No materials to create order for."

        o = db.create_order(first_mat["id"], validated_items[0]["quantity"], notes)
        db.create_order_items(o["id"], validated_items)

        items_str = ", ".join(item_descriptions)
        return f"Order created: #{o['id']} | {items_str} | Status: pending"
    except Exception as e:
        return f"Error creating order: {e}"


@tool
def update_order_status(order_id: int, status: str) -> str:
    """Update an order status. Valid statuses: pending, approved, completed, cancelled."""
    valid = {"pending", "approved", "completed", "cancelled"}
    if status not in valid:
        return f"Invalid status '{status}'. Valid: {valid}"
    o = db.update_order_status(order_id, status)
    if not o:
        return f"Order {order_id} not found."
    return f"Order {order_id} updated to '{status}'."


@tool
def delete_order(order_id: int) -> str:
    """Delete an order from the system."""
    if db.delete_order(order_id):
        return f"Deleted order {order_id}."
    return f"Order {order_id} not found."


@tool
def list_manufacturers() -> str:
    """List all manufacturers in the database with contact info."""
    mfrs = db.list_manufacturers()
    if not mfrs:
        return "No manufacturers found."
    lines = [f"Mfr ID: {m['id']} | {m['name']} | Email: {m.get('email', 'N/A')} | Phone: {m.get('phone', 'N/A')} | Loc: {m.get('location', 'N/A')} | Supplies: {m.get('materials_supplied', 'N/A')}"
             for m in mfrs]
    return "\n".join(lines)


@tool
def add_manufacturer(name: str, email: str = "", phone: str = "",
                     location: str = "", materials_supplied: str = "") -> str:
    """Add a new manufacturer/supplier to the database."""
    try:
        m = db.create_manufacturer(name, email, phone, location, materials_supplied)
        return f"Created manufacturer: ID={m['id']}, {m['name']}"
    except Exception as e:
        return f"Error creating manufacturer: {e}"


@tool
def search_manufacturers(query: str) -> str:
    """Search for manufacturers and pricing info on the web using SearXNG. Pass a query like 'copper wire supplier France' or 'leather wholesale Italy'."""
    from langchain_community.utilities import SearxSearchWrapper
    searx_host = "http://localhost:8888"
    try:
        wrapper = SearxSearchWrapper(searx_host=searx_host)
        results = wrapper.run(query)
        return results[:2000]
    except Exception as e:
        return f"SearXNG search failed (is SearXNG running on {searx_host}?): {e}"


@tool
def send_notification(order_id: int, contact: str = "") -> str:
    """Send order notification email for the given order. Fetches all items from the database and sends to the default notification email."""
    from .email import get_email_service
    email_service = get_email_service()
    to_email = contact or os.getenv("DEFAULT_NOTIFICATION_EMAIL", "")
    if not to_email:
        return "No default notification email configured. Set DEFAULT_NOTIFICATION_EMAIL in .env"

    items = db.get_order_items(order_id)
    if not items:
        return f"No items found for order {order_id}."

    item_strs = [f"{i['material_name']} x{i['quantity']} {i['unit']}" for i in items]
    material_name = ", ".join(item_strs)
    notes = ""
    for i in items:
        if i.get('material_name'):
            notes += f"{i['material_name']} x{i['quantity']} {i['unit']}; "

    result = email_service.send_order_notification(
        to=to_email,
        material_name=material_name,
        quantity=len(items),
        unit="items",
        order_id=order_id,
        notes=notes.strip(),
    )
    if result:
        return f"Notification sent to {to_email}: Order #{order_id} ({material_name})"
    else:
        return f"Email service unavailable. Logged to console: Order #{order_id} ({material_name}) -> {to_email}"


# --- ML Tools ---

@tool
def predict_customer_churn(
    customer_id: Optional[int] = None,
    recency: Optional[float] = None,
    frequency: Optional[float] = None,
    monetary_total: Optional[float] = None,
    monetary_trend: Optional[float] = None,
    product_diversity: Optional[float] = None,
    channel_diversity: Optional[float] = None,
    lifetime_days: Optional[float] = None,
    purchase_velocity: Optional[float] = None,
    avg_price: Optional[float] = None,
    channel: str = "",
) -> str:
    """
    Predict churn probability and customer segment using ML models.

    Use EITHER customer_id (lookup from database) OR provide all feature values
    manually (recency, frequency, monetary_total, monetary_trend, product_diversity,
    channel_diversity, lifetime_days, purchase_velocity, avg_price, channel).

    Returns: churn probability (0-1), status (Churne/Actif), segment ID + label,
    risk level (Faible/Modere/Eleve), and business recommendation.
    """
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from ml_pipeline import get_pipeline

    pipeline = get_pipeline()

    if customer_id is not None:
        cust = db.get_customer(customer_id)
        if not cust:
            return f"Customer {customer_id} not found."
        pred = cust.get("prediction")
        if not pred:
            return f"No prediction available for customer {customer_id} ({cust['name']}). Run data import first."
        return (
            f"Customer: {cust['name']} (ID: {customer_id})\n"
            f"Churn Probability: {pred['churn_probability']:.1%}\n"
            f"Status: {pred['churn_prediction']}\n"
            f"Segment: {pred['segment']} - {pred['segment_label']}\n"
            f"Risk Level: {pred['risk_level']}\n"
            f"Recommendation: {pred['recommendation']}"
        )

    if recency is None or frequency is None:
        return "Provide customer_id OR at least recency and frequency. For full accuracy, provide all features: recency, frequency, monetary_total, monetary_trend, product_diversity, channel_diversity, lifetime_days, purchase_velocity, avg_price, channel."

    features = {
        "recency": recency,
        "frequency": frequency,
        "monetary_total": monetary_total or 0,
        "monetary_trend": monetary_trend or 1.0,
        "product_diversity": product_diversity or 1,
        "channel_diversity": channel_diversity or 1,
        "lifetime_days": lifetime_days or 365,
        "purchase_velocity": purchase_velocity or 1.0,
        "avg_price": avg_price or 50,
        "channel": channel,
    }
    result = pipeline.predict_customer(features)
    return (
        f"Churn Probability: {result['churn_probability']:.1%}\n"
        f"Status: {result['churn_prediction']}\n"
        f"Segment: {result['segment']} - {result['segment_label']}\n"
        f"Risk Level: {result['risk_level']}\n"
        f"Recommendation: {result['recommendation']}"
    )


@tool
def list_customers(
    segment: Optional[int] = None,
    risk_level: Optional[str] = None,
    search: str = "",
) -> str:
    """
    List customers with their churn predictions and segments.

    Filter by segment (0=VIP, 1=Lost, 2=At-risk, 3=New) or risk level
    ('Faible', 'Modere', 'Eleve'). Search by customer name.
    Returns summary of matching customers with their risk profiles.
    """
    customers = db.list_customers(segment=segment, risk_level=risk_level, search=search, limit=100)
    if not customers:
        filters = []
        if segment is not None:
            filters.append(f"segment={segment}")
        if risk_level:
            filters.append(f"risk={risk_level}")
        if search:
            filters.append(f"search='{search}'")
        return f"No customers found. Filters: {', '.join(filters) if filters else 'none'}"

    lines = []
    lines.append(f"Found {len(customers)} customer(s):")
    lines.append("-" * 80)
    for c in customers:
        pred = c.get("segment_label", "N/A")
        prob = f"{c.get('churn_probability', 0):.0%}" if c.get("churn_probability") else "N/A"
        risk = c.get("risk_level", "N/A")
        status = c.get("churn_prediction", "N/A")
        lines.append(f"  ID:{c['id']:>3} | {c['name']:<40} | Seg:{pred:<25} | Prob:{prob:>6} | Risk:{risk:<7} | {status}")
    return "\n".join(lines)


@tool
def list_high_risk_customers(
    min_probability: float = 0.7,
) -> str:
    """
    List customers with high churn risk (probability >= threshold).

    Useful before creating orders — high-risk customers may need retention attention.
    Default threshold is 0.7 (70%). Lower to see more customers (e.g., 0.5 for 50%).
    """
    customers = db.get_high_risk_customers(min_probability=min_probability)
    if not customers:
        return f"No customers with churn probability >= {min_probability:.0%}."

    lines = []
    lines.append(f"High-risk customers (probability >= {min_probability:.0%}): {len(customers)} found")
    lines.append("-" * 80)
    for c in customers:
        prob = f"{c.get('churn_probability', 0):.1%}"
        lines.append(f"  ID:{c['id']:>3} | {c['name']:<40} | {prob:>7} | {c.get('segment_label', 'N/A')} | {c.get('recommendation', '')[:50]}")
    return "\n".join(lines)


@tool
def get_customer_segment_stats() -> str:
    """
    Get statistics about customer segments: count per segment,
    average churn probability per segment, and segment profiles.
    """
    stats = db.get_segment_stats()
    if not stats:
        return "No segment data available. Run data import first."

    lines = []
    lines.append("Customer Segment Statistics:")
    lines.append("=" * 60)
    for s in stats:
        lines.append(f"\nSegment {s['segment']} - {s['segment_label']}:")
        lines.append(f"  Customers: {s['count']}")
        lines.append(f"  Avg Churn Risk: {s['avg_probability']:.1%}")
        lines.append(f"  Risk Range: {s['min_probability']:.1%} - {s['max_probability']:.1%}")

        # Add segment profile info
        profiles = {
            0: "Premium clients with high spending and frequent purchases.",
            1: "Likely inactive clients for a long time.",
            2: "Medium-value clients showing signs of disengagement.",
            3: "Recent clients with limited purchase history.",
        }
        lines.append(f"  Profile: {profiles.get(s['segment'], 'Unknown')}")

    return "\n".join(lines)


# Seed data
def seed_db():
    """Add sample data if DB is empty."""
    if db.list_manufacturers():
        return
    m1 = db.create_manufacturer(
        name="Tunisian Glass Works",
        email="contact@glassworks.tn",
        phone="",
        location="Sidi Bou Said",
        materials_supplied="Glass sheets, glass bottles, glass panels"
    )
    m2 = db.create_manufacturer(
        name="Atelier de Ceramique Nabeul",
        email="contact@ceramique-nabeul.tn",
        phone="",
        location="Nabeul",
        materials_supplied="Ceramic clay, glazes, underglazes"
    )
    m3 = db.create_manufacturer(
        name="Tunisian Copper Atelier",
        email="contact@copper-atelier.tn",
        phone="",
        location="Medina",
        materials_supplied="Copper sheets, tubing, wire"
    )
    m4 = db.create_manufacturer(
        name="Kasserine Olive Wood Coop",
        email="contact@olive-wood.tn",
        phone="",
        location="Kasserine",
        materials_supplied="Olive wood blocks, blanks"
    )
    m5 = db.create_manufacturer(
        name="Tunisian Leather Tannery",
        email="contact@leather-tannery.tn",
        phone="",
        location="El Jem",
        materials_supplied="Vegetable and chrome tanned leather"
    )
    db.create_material("Crackle Glass Vessel 80ml", "Glass", 18.00, "piece", 10, 100, m1["id"])
    db.create_material("Glass Blown Bottles Clear", "Glass", 8.00, "piece", 12, 60, m1["id"])
    db.create_material("Glass Perfume Bottle 10ml", "Glass", 4.00, "piece", 20, 100, m1["id"])
    db.create_material("Copper Sheet 0.5mm martele", "Metal", 12.00, "sheet", 10, 80, m3["id"])
    db.create_material("Copper Hinge Wire 1mm", "Metal", 0.50, "10cm", 50, 500, m3["id"])
    db.create_material("Stainless Steel Container 500ml", "Metal", 12.00, "piece", 10, 60, m3["id"])
    db.create_material("Silicone Seal Ring", "Metal", 0.30, "piece", 100, 1000, m3["id"])
    db.create_material("Double-wall Steel Tumbler 500ml", "Metal", 10.00, "piece", 10, 60, m3["id"])
    db.create_material("Silicone Lid with Straw", "Metal", 1.50, "piece", 20, 100, m3["id"])
    db.create_material("Bamboo Straw", "Metal", 0.50, "piece", 50, 500, m3["id"])
    db.create_material("Copper Brazier (mathred)", "Metal", 15.00, "piece", 10, 40, m3["id"])
    db.create_material("Ceramic Mug Blank (pre-formed)", "Ceramic", 3.00, "piece", 20, 200, m2["id"])
    db.create_material("Ceramic Plate Blank", "Ceramic", 2.00, "piece", 20, 200, m2["id"])
    db.create_material("Ceramic Glaze (white, food-safe)", "Ceramic", 5.00, "bottle", 10, 80, m2["id"])
    db.create_material("Food-safe Paint (red/blue/yellow)", "Ceramic", 1.50, "bottle", 20, 100, m2["id"])
    db.create_material("Cotton Canvas Pouch (30x25cm)", "Textile", 1.50, "piece", 20, 200, m5["id"])
    db.create_material("Zipper Closure 15cm", "Textile", 0.50, "piece", 50, 500, m5["id"])
    db.create_material("Screen Print Ink (1-color)", "Textile", 1.00, "piece", 20, 200, m5["id"])
    db.create_material("Plywood Board 3mm (30x20cm)", "Wood", 4.00, "piece", 20, 100, m4["id"])
    db.create_material("Wooden Game Tokens (set 12)", "Wood", 2.00, "set", 20, 100, m4["id"])
    db.create_material("Acrylic Paint (wood, assorted)", "Wood", 2.25, "set", 20, 100, m4["id"])
    db.create_material("Clear Wood Varnish (matte)", "Wood", 1.00, "bottle", 20, 100, m4["id"])
    db.create_material("Hardwood Carving Block", "Wood", 8.00, "piece", 10, 40, m4["id"])
    db.create_material("Perfume Oil/Musk (ml)", "Fragrance", 0.80, "ml", 50, 500, m5["id"])
    db.create_material("Prayer Beads (wooden, 33)", "Accessory", 5.00, "set", 20, 100, m5["id"])
    db.create_material("Kraft Paper Sheet (15x10cm)", "Packaging", 0.15, "piece", 100, 2000, m1["id"])
    db.create_material("Gift Box Small (branded)", "Packaging", 2.50, "piece", 20, 200, m1["id"])
    print("Seeded database with 5 manufacturers and 25 materials.")
