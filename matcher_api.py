import os
import math
import tempfile
from pathlib import Path
from collections import Counter
from urllib.parse import urljoin

import pandas as pd
import requests
import torch
from PIL import Image
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, CLIPVisionModelWithProjection

CSV_FILE = "sougui_products.csv"
IMAGES_FOLDER = "sougui_photos"
TOP_K = 1000
TOP_N_FOR_CATEGORY = 7
TOP_CATEGORIES_TO_USE = 3

CATEGORY_URLS = {
    "Décoration": "https://sougui.tn/product-category/shop/deco-maison/",
    "Arts de la table": "https://sougui.tn/product-category/shop/arts-de-la-table/",
    "Couffins & Foutas": "https://sougui.tn/product-category/shop/couffins-foutas/",
    "Gifts": "https://sougui.tn/product-category/shop/gifts/",
    "Jeux": "https://sougui.tn/product-category/jeux/",
    "Recycl’art": "https://sougui.tn/product-category/recyclart/",
}

http = requests.Session()
http.headers.update({
    "User-Agent": "Mozilla/5.0"
})

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

df = pd.read_csv(CSV_FILE, sep=";", encoding="utf-8-sig")
df.columns = df.columns.str.strip()


def get_image_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"])
        features = outputs.image_embeds

    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    norms = torch.norm(features, dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-12)
    features = features / norms

    return features.squeeze(0)


catalog_embeddings = []
catalog_rows = []

for _, row in df.iterrows():
    image_name = row["image_name"]
    image_path = os.path.join(IMAGES_FOLDER, image_name)

    if not os.path.exists(image_path):
        continue

    try:
        emb = get_image_embedding(image_path)
        catalog_embeddings.append(emb)
        catalog_rows.append(row.to_dict())
    except Exception as e:
        print("[LOCAL ERROR]", image_name, "->", e)

if not catalog_embeddings:
    raise RuntimeError("No catalog images found.")

catalog_embeddings = torch.stack(catalog_embeddings)


def predict_main_categories(query_embedding):
    similarities = torch.matmul(catalog_embeddings, query_embedding)
    similarities = torch.nan_to_num(similarities, nan=-1.0, posinf=1.0, neginf=-1.0)

    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    top_indices = sorted_indices[:TOP_N_FOR_CATEGORY]

    cats = []
    for idx in top_indices:
        c = str(catalog_rows[idx]["main_category"]).strip()
        if c and c.lower() != "nan":
            cats.append(c)

    if not cats:
        return []

    return [x[0] for x in Counter(cats).most_common(TOP_CATEGORIES_TO_USE)]


def fetch_live_products_for_category(category_name):
    url = CATEGORY_URLS.get(category_name)
    if not url:
        return []

    r = http.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    seen_links = set()
    product_links = []

    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"]).split("?")[0]
        if "/product/" in href and "/product-category/" not in href and href not in seen_links:
            product_links.append(href)
            seen_links.add(href)

    results = []
    for product_url in product_links:
        try:
            rp = http.get(product_url, timeout=30)
            rp.raise_for_status()
            sp = BeautifulSoup(rp.text, "html.parser")

            og = sp.find("meta", attrs={"property": "og:image"})
            if not og or not og.get("content"):
                continue

            image_url = og["content"]
            if image_url.startswith("data:") or ".svg" in image_url.lower():
                continue

            h1 = sp.find("h1")
            product_name = h1.get_text(" ", strip=True) if h1 else product_url.rstrip("/").split("/")[-1]

            results.append({
                "product_name": product_name,
                "product_url": product_url,
                "image_url": image_url,
                "main_category": category_name,
                "subcategory": ""
            })
        except Exception as e:
            print("[LIVE PRODUCT ERROR]", product_url, "->", e)
            continue

    print(f"[LIVE] category={category_name} url={url} products_found={len(results)}")
    return results


def get_image_embedding_from_url(image_url):
    r = http.get(image_url, timeout=30)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    try:
        return get_image_embedding(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/match_live")
async def match_live(image: UploadFile = File(...)):
    suffix = Path(image.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        query_embedding = get_image_embedding(tmp_path)

        predicted_categories = predict_main_categories(query_embedding)
        print("[LIVE] predicted_categories =", predicted_categories)

        if not predicted_categories:
            return {"predicted_category": "", "matches": []}

        live_products = []
        seen_urls = set()

        for cat in predicted_categories:
            products = fetch_live_products_for_category(cat)
            for p in products:
                if p["product_url"] not in seen_urls:
                    live_products.append(p)
                    seen_urls.add(p["product_url"])

        print("[LIVE] fetched_live_products =", len(live_products))

        if not live_products:
            return {
                "predicted_category": ", ".join(predicted_categories),
                "matches": []
            }

        scored = []

        for p in live_products:
            try:
                print("[LIVE IMG]", p["image_url"])
                emb = get_image_embedding_from_url(p["image_url"])
                score = float(torch.matmul(emb, query_embedding).item())

                if not math.isfinite(score):
                    continue

                scored.append({
                    "product_name": p["product_name"],
                    "main_category": p["main_category"],
                    "subcategory": p["subcategory"],
                    "product_url": p["product_url"],
                    "image_url": p["image_url"],
                    "score": score
                })
            except Exception as e:
                print("[LIVE ERROR]", p["image_url"], "->", e)
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "predicted_category": ", ".join(predicted_categories),
            "matches": scored[:TOP_K]
        }

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass