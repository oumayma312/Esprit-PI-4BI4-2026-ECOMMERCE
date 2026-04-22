"""
Seed script that replaces generic materials with Tunisian artisan materials
and adds top 20 products from sales data.

Usage: python seed_products.py
"""
import csv
import os
import sqlite3
import sys

# Resolve DB path the same way db.py does
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "materials.db")

# Top 20 productFKs by total quantity sold (pre-computed from FactVentee.csv)
TOP_20_PRODUCTS = [
    (6536, "Bonbonniere en verre craquele + Couvercle en cuivre", 66.50),
    (6543, "Stylo LilasDream", 1.19),
    (6487, "Coffret Mathred + Lance Parfum + Chapelet", 62.00),
    (6503, "Bonbonniere en verre craquele+ couvercle en cuivre", 66.50),
    (6542, "Bon d'achat de 100 DT", 100.00),
    (6524, "Lunch Box", 29.00),
    (6527, "Bourse Lunch Box", 5.00),
    (6528, "Personnalisation Lunch Box", 2.00),
    (6572, "Personnalisation Bourse Lunch Box", 2.00),
    (6530, "Papier Craft 15*10 personnalise", 1.00),
    (6494, "Thermos a paille", 28.15),
    (6495, "Personnalisation", 2.00),
    (6496, "Tote Bag personnalise", 8.00),
    (6651, 'Set "Coeur a Coeur" - Mug et Plateau en Ceramique', 24.93),
    (7079, "Mouton T2", 45.38),
    (6587, "Jeu Nes Bekri Kalou", 30.00),
    (6577, 'Set "Duo Envoitant" - Mug et Plateau en Ceramique', 27.37),
    (6580, "Mug Boheme en ceramique artisanale - Avec sous-tasse", 25.00),
    (6552, "4 Postes", 32.00),
    (6584, "Mug Boheme en ceramique artisanale - Avec sous-tasse", 25.00),
]

# Tunisian manufacturers
MANUFACTURERS = [
    ("Tunisian Glass Works", "contact@glassworks.tn", "", "Sidi Bou Said", "Glass sheets, glass bottles, glass panels"),
    ("Atelier de Ceramique Nabeul", "contact@ceramique-nabeul.tn", "", "Nabeul", "Ceramic clay, glazes, underglazes"),
    ("Tunisian Copper Atelier", "contact@copper-atelier.tn", "", "Medina", "Copper sheets, tubing, wire"),
    ("Kasserine Olive Wood Coop", "contact@olive-wood.tn", "", "Kasserine", "Olive wood blocks, blanks"),
    ("Tunisian Leather Tannery", "contact@leather-tannery.tn", "", "El Jem", "Vegetable and chrome tanned leather"),
]

# Materials grouped by category, each tuple: (name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id)
# manufacturer_id is set dynamically after manufacturers are inserted
MATERIALS = [
    # Glass (3) - for bonbonnieres, perfume bottles
    ("Crackle Glass Vessel 80ml", "Glass", 18.00, "piece", 10, 100, 1),
    ("Glass Blown Bottles Clear", "Glass", 8.00, "piece", 12, 60, 1),
    ("Glass Perfume Bottle 10ml", "Glass", 4.00, "piece", 20, 100, 1),

    # Metal (8) - for bonbonnieres lids, lunch boxes, thermos
    ("Copper Sheet 0.5mm martele", "Metal", 12.00, "sheet", 10, 80, 3),
    ("Copper Hinge Wire 1mm", "Metal", 0.50, "10cm", 50, 500, 3),
    ("Stainless Steel Container 500ml", "Metal", 12.00, "piece", 10, 60, 3),
    ("Silicone Seal Ring", "Metal", 0.30, "piece", 100, 1000, 3),
    ("Double-wall Steel Tumbler 500ml", "Metal", 10.00, "piece", 10, 60, 3),
    ("Silicone Lid with Straw", "Metal", 1.50, "piece", 20, 100, 3),
    ("Bamboo Straw", "Metal", 0.50, "piece", 50, 500, 3),
    ("Copper Brazier (mathred)", "Metal", 15.00, "piece", 10, 40, 3),

    # Ceramic (4) - for mug sets, plates
    ("Ceramic Mug Blank (pre-formed)", "Ceramic", 3.00, "piece", 20, 200, 2),
    ("Ceramic Plate Blank", "Ceramic", 2.00, "piece", 20, 200, 2),
    ("Ceramic Glaze (white, food-safe)", "Ceramic", 5.00, "bottle", 10, 80, 2),
    ("Food-safe Paint (red/blue/yellow)", "Ceramic", 1.50, "bottle", 20, 100, 2),

    # Textile (3) - for tote bags, lunch box pouches
    ("Cotton Canvas Pouch (30x25cm)", "Textile", 1.50, "piece", 20, 200, 5),
    ("Zipper Closure 15cm", "Textile", 0.50, "piece", 50, 500, 5),
    ("Screen Print Ink (1-color)", "Textile", 1.00, "piece", 20, 200, 5),

    # Wood (4) - for games, Mouton figurine
    ("Plywood Board 3mm (30x20cm)", "Wood", 4.00, "piece", 20, 100, 4),
    ("Wooden Game Tokens (set 12)", "Wood", 2.00, "set", 20, 100, 4),
    ("Acrylic Paint (wood, assorted)", "Wood", 2.25, "set", 20, 100, 4),
    ("Clear Wood Varnish (matte)", "Wood", 1.00, "bottle", 20, 100, 4),
    ("Hardwood Carving Block", "Wood", 8.00, "piece", 10, 40, 4),

    # Fragrance (1) - for Coffret Mathred
    ("Perfume Oil/Musk (ml)", "Fragrance", 0.80, "ml", 50, 500, 5),

    # Accessory (1) - for Coffret Mathred
    ("Prayer Beads (wooden, 33)", "Accessory", 5.00, "set", 20, 100, 5),

    # Packaging (2) - for all products
    ("Kraft Paper Sheet (15x10cm)", "Packaging", 0.15, "piece", 100, 2000, 1),
    ("Gift Box Small (branded)", "Packaging", 2.50, "piece", 20, 200, 1),
]


def seed():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    # Delete all existing materials and manufacturers
    conn.execute("DELETE FROM order_items")
    conn.execute("DELETE FROM orders")
    conn.execute("DELETE FROM materials")
    conn.execute("DELETE FROM manufacturers")

    # Reset auto-increment counters
    conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('manufacturers', 'materials')")

    # Insert manufacturers
    mfr_ids = []
    for name, email, phone, location, materials_supplied in MANUFACTURERS:
        cur = conn.execute(
            "INSERT INTO manufacturers (name, email, phone, location, materials_supplied) VALUES (?,?,?,?,?)",
            (name, email, phone, location, materials_supplied),
        )
        mfr_ids.append(cur.lastrowid)

    # Insert materials
    for mat in MATERIALS:
        name, category, unit_price, unit, min_order_qty, stock_qty, mfr_idx = mat
        conn.execute(
            "INSERT INTO materials (name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id) VALUES (?,?,?,?,?,?,?)",
            (name, category, unit_price, unit, min_order_qty, stock_qty, mfr_ids[mfr_idx - 1]),
        )

    # Insert top 20 products from FactVentee.csv
    for product_fk, name, avg_price in TOP_20_PRODUCTS:
        conn.execute(
            "INSERT INTO materials (name, category, unit_price, unit, min_order_qty, stock_quantity, manufacturer_id) VALUES (?,?,?,?,?,?,?)",
            (name, "Product", avg_price, "piece", 1, 50, None),
        )

    conn.commit()
    conn.close()

    total_mfrs = len(MANUFACTURERS)
    total_mats = len(MATERIALS) + len(TOP_20_PRODUCTS)
    print(f"Seeded {total_mfrs} manufacturers and {total_mats} materials ({len(MATERIALS)} artisan materials + {len(TOP_20_PRODUCTS)} top products).")


if __name__ == "__main__":
    seed()
