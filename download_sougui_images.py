import os
import re
import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://sougui.tn"

pages = [
    "https://sougui.tn/shop/",
    "https://sougui.tn/shop/page/2/",
    "https://sougui.tn/shop/page/3/",
    "https://sougui.tn/shop/page/4/",
    "https://sougui.tn/shop/page/5/",
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

save_folder = "sougui_photos"
os.makedirs(save_folder, exist_ok=True)

session = requests.Session()
session.headers.update(headers)

# Main categories
MAIN_ALIASES = {
    "déco": "Décoration",
    "deco": "Décoration",
    "décoration": "Décoration",
    "decoration": "Décoration",
    "arts de la table": "Arts de la table",
    "couffins & foutas": "Couffins & Foutas",
    "gifts": "Gifts",
    "recycl'art": "Recycl’art",
    "recyclart": "Recycl’art",
    "jeux": "Jeux",
}

# Subcategory -> Main category
SUB_TO_MAIN = {
    "sculptures": "Décoration",
    "luminaires": "Décoration",
    "bougeoirs": "Décoration",
    "vases": "Décoration",

    "service de table": "Arts de la table",
    "verres": "Arts de la table",
    "céramiques": "Arts de la table",
    "ceramiques": "Arts de la table",
    "cuivres": "Arts de la table",
    "bois d'olivier": "Arts de la table",
    "bois d’olivier": "Arts de la table",

    "corpogift": "Gifts",
    "esprit romance": "Gifts",
}

IGNORE = {"accueil", "home", "shop", "boutiques"}


def clean_name(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text.strip().replace(" ", "_")


def normalize(text):
    text = text.strip().lower()
    text = text.replace("’", "'")
    return text


def get_soup(url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def unique_keep_order(items):
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_real_category(soup):
    # 1) Try breadcrumb first
    breadcrumb = soup.find(class_=re.compile("breadcrumb", re.I))
    cats = []

    if breadcrumb:
        for a in breadcrumb.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text(" ", strip=True)
            if "/product-category/" in href and text:
                if normalize(text) not in IGNORE:
                    cats.append(text)

    cats = unique_keep_order(cats)

    if len(cats) >= 2:
        main = cats[0]
        sub = cats[1]

        main_norm = normalize(main)
        if main_norm in MAIN_ALIASES:
            main = MAIN_ALIASES[main_norm]

        return main, sub

    if len(cats) == 1:
        x = cats[0]
        x_norm = normalize(x)

        if x_norm in SUB_TO_MAIN:
            return SUB_TO_MAIN[x_norm], x

        if x_norm in MAIN_ALIASES:
            return MAIN_ALIASES[x_norm], ""

    # 2) Fallback: scan all product-category links on page
    all_cats = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        if "/product-category/" in href and text:
            if normalize(text) not in IGNORE:
                all_cats.append(text)

    all_cats = unique_keep_order(all_cats)

    # first try subcategory match
    for x in all_cats:
        x_norm = normalize(x)
        if x_norm in SUB_TO_MAIN:
            return SUB_TO_MAIN[x_norm], x

    # then try main category match
    for x in all_cats:
        x_norm = normalize(x)
        if x_norm in MAIN_ALIASES:
            return MAIN_ALIASES[x_norm], ""

    return "", ""


# Step 1: collect product links
product_links = set()

print("Collecting product links...")
for page in pages:
    print("Reading:", page)
    soup = get_soup(page)

    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"]).split("?")[0]
        if "/product/" in href and "/product-category/" not in href:
            product_links.add(href)

    time.sleep(2)

print("Found", len(product_links), "product links")


# Step 2: open product pages and save data
rows = []
count = 1

for link in sorted(product_links):
    try:
        print("Opening:", link)
        soup = get_soup(link)

        # product name
        h1 = soup.find("h1")
        product_name = h1.get_text(strip=True) if h1 else f"product_{count}"

        # real category
        main_category, subcategory = extract_real_category(soup)

        # image
        img_url = None
        og = soup.find("meta", attrs={"property": "og:image"})
        if og and og.get("content"):
            img_url = og["content"]

        if not img_url:
            img = soup.find("img")
            if img:
                img_url = img.get("src") or img.get("data-src")

        image_name = ""
        if img_url:
            img_url = urljoin(link, img_url)
            r = session.get(img_url, timeout=30)
            r.raise_for_status()

            ext = os.path.splitext(urlparse(img_url).path)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"

            image_name = f"{count:03d}_{clean_name(product_name)[:80]}{ext}"
            file_path = os.path.join(save_folder, image_name)

            with open(file_path, "wb") as f:
                f.write(r.content)

        rows.append({
            "image_name": image_name,
            "product_name": product_name,
            "main_category": main_category,
            "subcategory": subcategory,
            "product_url": link
        })

        print("Saved:", product_name, "|", main_category, "|", subcategory)
        count += 1
        time.sleep(2)

    except Exception as e:
        print("Error:", link, "->", e)

# Step 3: save CSV
with open("sougui_products.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_name", "product_name", "main_category", "subcategory", "product_url"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Done. CSV saved as sougui_products.csv")