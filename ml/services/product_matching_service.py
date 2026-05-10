from __future__ import annotations

import base64
import binascii
import io
from functools import cached_property
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError, MlValidationError


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(vector)
    return vector / norm


def _safe_text(value: object) -> str:
    return str(value or '').strip()


class ProductMatchingService:
    MAX_UPLOAD_BYTES = 8 * 1024 * 1024

    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository

    @cached_property
    def _bundle(self) -> dict[str, object]:
        catalog = self.repository.load_sougui_catalog_frame()
        images_dir = self.repository.sougui_images_dir()
        working = catalog.copy()
        working['image_path'] = working['image_name'].map(lambda name: images_dir / name)
        working = working[working['image_path'].map(Path.is_file)].copy()

        if working.empty:
            raise MlServiceError('The Sougui catalog could not find any usable product images.')

        working['main_category'] = working['main_category'].replace('', 'Uncategorized')
        working['subcategory'] = working['subcategory'].replace('', 'General')
        working['search_blob'] = (
            working['product_name'].fillna('')
            + ' '
            + working['main_category'].fillna('')
            + ' '
            + working['subcategory'].fillna('')
        )

        text_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        text_matrix = text_vectorizer.fit_transform(working['search_blob'].tolist())

        visual_vectors = np.vstack([self._image_embedding(path) for path in working['image_path']])
        category_counts = (
            working.groupby('main_category', dropna=False)
            .agg(products=('product_id', 'count'))
            .reset_index()
            .sort_values(['products', 'main_category'], ascending=[False, True])
        )

        return {
            'catalog': working.reset_index(drop=True),
            'text_vectorizer': text_vectorizer,
            'text_matrix': text_matrix,
            'visual_matrix': visual_vectors,
            'category_counts': category_counts.reset_index(drop=True),
            'images_dir': images_dir,
            'embedding_backend': 'hybrid-color-text',
        }

    def _image_embedding(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as original:
            return self._image_embedding_from_image(original)

    def _image_embedding_from_image(self, source_image: Image.Image) -> np.ndarray:
        original = source_image.convert('RGB')
        try:
            resized = original.resize((96, 96))
            rgb = np.asarray(resized, dtype=np.float32) / 255.0
            hsv = np.asarray(resized.convert('HSV'), dtype=np.float32) / 255.0
            gray = np.asarray(resized.convert('L'), dtype=np.float32) / 255.0

            histograms: list[np.ndarray] = []
            for channel_index in range(3):
                histograms.append(np.histogram(rgb[:, :, channel_index], bins=8, range=(0, 1), density=True)[0])
                histograms.append(np.histogram(hsv[:, :, channel_index], bins=8, range=(0, 1), density=True)[0])

            rgb_means = rgb.mean(axis=(0, 1))
            rgb_stds = rgb.std(axis=(0, 1))
            grad_x = np.abs(np.diff(gray, axis=1)).mean()
            grad_y = np.abs(np.diff(gray, axis=0)).mean()
            feature_vector = np.concatenate(
                [
                    *histograms,
                    rgb_means,
                    rgb_stds,
                    np.array([grad_x, grad_y, original.width / max(original.height, 1)], dtype=np.float32),
                ]
            ).astype(np.float32)
            return _normalize_vector(feature_vector)
        finally:
            original.close()

    def _decode_uploaded_image(self, image_data_url: str) -> Image.Image:
        payload = image_data_url.strip()
        if not payload:
            raise MlValidationError('Provide an image to analyze.')

        if ',' in payload:
            payload = payload.split(',', 1)[1]

        try:
            raw_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MlValidationError('The uploaded image could not be decoded.') from exc

        if len(raw_bytes) > self.MAX_UPLOAD_BYTES:
            raise MlValidationError('The uploaded image is too large. Keep it below 8 MB.')

        try:
            with Image.open(io.BytesIO(raw_bytes)) as uploaded:
                return uploaded.convert('RGB')
        except (UnidentifiedImageError, OSError) as exc:
            raise MlValidationError('The uploaded file is not a valid image.') from exc

    def _uploaded_query_payload(self, uploaded_image_name: str | None) -> dict[str, object]:
        file_name = _safe_text(uploaded_image_name) or 'uploaded-image'
        title = Path(file_name).stem.replace('_', ' ').replace('-', ' ').strip() or 'Uploaded image'
        return {
            'productId': 0,
            'imageName': file_name,
            'productName': title,
            'mainCategory': 'User upload',
            'subcategory': 'Local image',
            'productUrl': '',
            'imageUrl': '',
        }

    def _uploaded_text_scores(
        self,
        *,
        text_vectorizer: TfidfVectorizer,
        text_matrix,
        uploaded_image_name: str | None,
        catalog_size: int,
    ) -> np.ndarray:
        text_hint = Path(_safe_text(uploaded_image_name)).stem.replace('_', ' ').replace('-', ' ').strip()
        if not text_hint:
            return np.zeros(catalog_size, dtype=float)

        query_vector = text_vectorizer.transform([text_hint])
        return cosine_similarity(query_vector, text_matrix)[0]

    def _serialize_product(
        self,
        row: pd.Series,
        *,
        score: float | None = None,
        visual_score: float | None = None,
        text_score: float | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            'productId': int(row['product_id']),
            'imageName': row['image_name'],
            'productName': row['product_name'],
            'mainCategory': row['main_category'],
            'subcategory': row['subcategory'],
            'productUrl': row['product_url'],
            'imageUrl': f"/api/assets/sougui/{quote(row['image_name'])}",
        }
        if score is not None:
            payload['similarityScore'] = round(float(score), 4)
        if visual_score is not None:
            payload['visualScore'] = round(float(visual_score), 4)
        if text_score is not None:
            payload['textScore'] = round(float(text_score), 4)
        return payload

    def overview(self) -> dict[str, object]:
        bundle = self._bundle
        catalog: pd.DataFrame = bundle['catalog']  # type: ignore[assignment]
        category_counts: pd.DataFrame = bundle['category_counts']  # type: ignore[assignment]
        embedding_backend: str = bundle['embedding_backend']  # type: ignore[assignment]

        return {
            'workflow': 'productMatcher',
            'headline': 'Sougui matching engine is ready',
            'summary': 'The catalog is loaded from CSV, enriched with local photos, and indexed for intelligent visual and textual comparison.',
            'embeddingBackend': embedding_backend,
            'metrics': [
                {'label': 'Catalog products', 'value': int(len(catalog)), 'hint': 'searchable Sougui items'},
                {'label': 'Main categories', 'value': int(catalog['main_category'].nunique()), 'hint': 'distinct sections'},
                {'label': 'Subcategories', 'value': int(catalog['subcategory'].nunique()), 'hint': 'granular taxonomy'},
                {'label': 'Images indexed', 'value': int(catalog['image_name'].nunique()), 'hint': 'local visual assets'},
            ],
            'categories': [
                {
                    'name': row['main_category'],
                    'products': int(row['products']),
                    'share': round(float(row['products']) / max(len(catalog), 1), 4),
                }
                for _, row in category_counts.iterrows()
            ],
            'featuredProducts': [self._serialize_product(row) for _, row in catalog.head(8).iterrows()],
        }

    def search(self, query: str = '', category: str = '', limit: int = 24) -> dict[str, object]:
        bundle = self._bundle
        catalog: pd.DataFrame = bundle['catalog']  # type: ignore[assignment]
        text_vectorizer: TfidfVectorizer = bundle['text_vectorizer']  # type: ignore[assignment]
        text_matrix = bundle['text_matrix']

        filtered = catalog.copy()
        if category.strip():
            filtered = filtered[filtered['main_category'].str.lower() == category.strip().lower()].copy()

        scores = np.zeros(len(catalog), dtype=float)
        trimmed_query = query.strip()
        if trimmed_query:
            query_vector = text_vectorizer.transform([trimmed_query])
            scores = cosine_similarity(query_vector, text_matrix)[0]

        filtered['search_score'] = filtered.index.map(lambda index: float(scores[index]))
        if trimmed_query:
            filtered = filtered.sort_values(['search_score', 'product_name'], ascending=[False, True])
        else:
            filtered = filtered.sort_values(['main_category', 'subcategory', 'product_name'])

        limited = filtered.head(max(1, min(limit, 60)))
        return {
            'query': trimmed_query,
            'category': category,
            'total': int(len(filtered)),
            'products': [self._serialize_product(row, score=float(row['search_score'])) for _, row in limited.iterrows()],
        }

    def compare(
        self,
        *,
        image_name: str | None = None,
        product_name: str | None = None,
        uploaded_image_data_url: str | None = None,
        uploaded_image_name: str | None = None,
        limit: int = 6,
    ) -> dict[str, object]:
        bundle = self._bundle
        catalog: pd.DataFrame = bundle['catalog']  # type: ignore[assignment]
        text_vectorizer: TfidfVectorizer = bundle['text_vectorizer']  # type: ignore[assignment]
        text_matrix = bundle['text_matrix']
        visual_matrix: np.ndarray = bundle['visual_matrix']  # type: ignore[assignment]
        embedding_backend: str = bundle['embedding_backend']  # type: ignore[assignment]

        query_index: int | None = None
        query_payload: dict[str, object]
        query_visual: np.ndarray
        text_scores: np.ndarray
        same_main_category = np.zeros(len(catalog), dtype=float)
        same_subcategory = np.zeros(len(catalog), dtype=float)
        summary = 'The hybrid matcher combines local product photos, naming similarity, and category context to rank visually related products.'

        if uploaded_image_data_url and uploaded_image_data_url.strip():
            uploaded_image = self._decode_uploaded_image(uploaded_image_data_url)
            try:
                query_visual = self._image_embedding_from_image(uploaded_image)
            finally:
                uploaded_image.close()

            text_scores = self._uploaded_text_scores(
                text_vectorizer=text_vectorizer,
                text_matrix=text_matrix,
                uploaded_image_name=uploaded_image_name,
                catalog_size=len(catalog),
            )
            query_payload = self._uploaded_query_payload(uploaded_image_name)
            summary = 'The hybrid matcher compared the uploaded image against the local catalog using visual similarity and optional filename cues.'
        else:
            if image_name and image_name.strip():
                matching = catalog[catalog['image_name'].str.lower() == image_name.strip().lower()]
                if not matching.empty:
                    query_index = int(matching.index[0])
            elif product_name and product_name.strip():
                query_vector = text_vectorizer.transform([product_name.strip()])
                query_scores = cosine_similarity(query_vector, text_matrix)[0]
                if len(query_scores):
                    query_index = int(np.argmax(query_scores))

            if query_index is None:
                raise MlValidationError('Provide a valid Sougui product name, image name, or uploaded image to compare.')

            query_row = catalog.loc[query_index]
            query_visual = visual_matrix[query_index]
            query_text = text_matrix[query_index]
            text_scores = cosine_similarity(query_text, text_matrix)[0]
            same_main_category = (catalog['main_category'] == query_row['main_category']).astype(float) * 0.08
            same_subcategory = (catalog['subcategory'] == query_row['subcategory']).astype(float) * 0.04
            query_payload = self._serialize_product(query_row)

        visual_scores = cosine_similarity(query_visual.reshape(1, -1), visual_matrix)[0]
        combined_scores = 0.68 * visual_scores + 0.2 * text_scores + same_main_category + same_subcategory
        if query_index is not None:
            combined_scores[query_index] = -1

        ranked_indices = np.argsort(combined_scores)[::-1]
        top_indices = [int(index) for index in ranked_indices[: max(1, min(limit, 12))]]

        top_categories = (
            catalog.iloc[top_indices]
            .groupby('main_category', dropna=False)
            .agg(matches=('product_id', 'count'))
            .reset_index()
            .sort_values(['matches', 'main_category'], ascending=[False, True])
        )

        average_similarity = float(np.mean([combined_scores[index] for index in top_indices])) if top_indices else 0.0
        headline_subject = str(query_payload['productName'])
        query_metric_value = str(query_payload['mainCategory']) if query_index is not None else 'Uploaded image'

        return {
            'workflow': 'productMatcher',
            'headline': f'Similar products found for {headline_subject}',
            'summary': summary,
            'embeddingBackend': embedding_backend,
            'queryProduct': query_payload,
            'metrics': [
                {'label': 'Query category', 'value': query_metric_value, 'tone': 'primary'},
                {'label': 'Top match score', 'value': f"{max(0.0, combined_scores[top_indices[0]]) * 100:.1f}%", 'tone': 'success'},
                {'label': 'Average similarity', 'value': f'{average_similarity * 100:.1f}%', 'tone': 'neutral'},
            ],
            'predictedCategories': [
                {
                    'name': row['main_category'],
                    'matches': int(row['matches']),
                }
                for _, row in top_categories.head(3).iterrows()
            ],
            'matches': [
                self._serialize_product(
                    catalog.loc[index],
                    score=float(max(0.0, combined_scores[index])),
                    visual_score=float(max(0.0, visual_scores[index])),
                    text_score=float(max(0.0, text_scores[index])),
                )
                for index in top_indices
            ],
        }
