import base64
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


_RE_A = re.compile(r"\bData Preparation\b|\bFeature Engineering\b|\bA\s*[—-]\s*Data Preparation", re.IGNORECASE)
_RE_B = re.compile(r"\bModel Understanding\b|\bB\s*[—-]\s*Model Understanding", re.IGNORECASE)
_RE_B_DIAG = re.compile(
    r"\bDaily sales time series\b|\bStationarity\b|\bdecomposition\b|\bADF\b|\bKPSS\b",
    re.IGNORECASE,
)
_RE_C_TS = re.compile(r"^###\s*F\.\d+\b", re.IGNORECASE | re.MULTILINE)
_RE_C = re.compile(
    r"\bClustering\b|\bForecast\b|\bModel\b|\bTraining\b|\bC\s*[—-]\s*Clustering\b|\bC\s*[—-]\s*Forecast", re.IGNORECASE
)


def _render_markdown(md_text: str) -> str:
    """Render notebook Markdown cells to HTML for web display."""

    try:
        import markdown  # type: ignore

        return markdown.markdown(
            md_text,
            extensions=["fenced_code", "tables", "sane_lists"],
            output_format="html5",
        )
    except Exception:
        # Fallback: keep it readable even if the markdown lib isn't installed.
        return "<pre class=\"pre\">" + html.escape(md_text) + "</pre>"


@dataclass(frozen=True)
class NotebookSpec:
    key: str
    title: str
    subtitle: str
    notebook_path: Path
    models_used: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_notebook_path(filename: str) -> Path:
    """Resolve notebook path, preferring user Downloads copies when available."""

    candidates: list[Path] = [
        Path("D:/Downloads") / filename,
        Path.home() / "Downloads" / filename,
        _repo_root() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return _repo_root() / filename


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _assets_dir(key: str) -> Path:
    return _static_dir() / "notebook_assets" / key


def _load_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cell_text(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "\n".join(src)
    return str(src)


def _iter_image_outputs(cell: dict[str, Any]):
    for out in cell.get("outputs", []) or []:
        data = out.get("data") or {}
        if not isinstance(data, dict):
            continue
        for mime, payload in data.items():
            if not isinstance(mime, str):
                continue
            if not mime.startswith("image/"):
                continue
            if payload is None:
                continue
            if isinstance(payload, list):
                payload = "".join(str(x) for x in payload)
            yield mime, payload


def _iter_html_outputs(cell: dict[str, Any]):
    for out in cell.get("outputs", []) or []:
        data = out.get("data") or {}
        html = data.get("text/html")
        if not html:
            continue
        if isinstance(html, list):
            html = "".join(html)
        yield str(html)


def _iter_plotly_outputs(cell: dict[str, Any]):
    for out in cell.get("outputs", []) or []:
        data = out.get("data") or {}
        if not isinstance(data, dict):
            continue
        for key in ("application/vnd.plotly.v1+json", "application/vnd.plotly.v2+json"):
            if key in data:
                yield data[key]


def _iter_text_outputs(cell: dict[str, Any]):
    for out in cell.get("outputs", []) or []:
        otype = out.get("output_type")
        if otype == "stream":
            txt = out.get("text", "")
            if isinstance(txt, list):
                txt = "".join(txt)
            if str(txt).strip():
                yield str(txt)
            continue

        if otype == "error":
            tb = out.get("traceback") or []
            if isinstance(tb, list):
                tb = "\n".join(tb)
            if str(tb).strip():
                yield str(tb)
            continue

        data = out.get("data") or {}
        plain = data.get("text/plain")
        if plain:
            if isinstance(plain, list):
                plain = "".join(plain)
            if str(plain).strip():
                yield str(plain)


def _decode_image_bytes(mime: str, payload: Any) -> tuple[bytes, str] | None:
    """Return (bytes, extension) for an image/* output."""

    if mime == "image/svg+xml":
        svg = str(payload)
        if not svg.strip():
            return None
        return svg.encode("utf-8"), "svg"

    # Most raster images in ipynb are base64-encoded strings.
    b64 = str(payload)
    if not b64.strip():
        return None

    try:
        raw = base64.b64decode(b64.encode("utf-8"))
    except Exception:
        return None

    ext = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(mime, "img")
    return raw, ext


def _first_heading_title(md_text: str) -> str | None:
    for line in md_text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        if not re.match(r"^#{2,6}\s+", s):
            continue
        title = re.sub(r"^#{2,6}\s+", "", s).strip()
        if title:
            return title
    return None


def _new_step(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "markdown": [],
        "html": [],
        "text": [],
        "images": [],
        "plotly": [],
    }


def _extract_assets(spec: NotebookSpec) -> dict[str, Any]:
    nb = _load_notebook(spec.notebook_path)
    cells = nb.get("cells") or []

    out_dir = _assets_dir(spec.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest: if notebook hasn't changed, reuse
    manifest_path = out_dir / "manifest.json"
    nb_mtime = spec.notebook_path.stat().st_mtime
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_mtime") == nb_mtime and manifest.get("version") == 6:
                return manifest.get("sections", {})
        except Exception:
            pass

    section: str | None = None
    current_step_title: str | None = None

    sections: dict[str, Any] = {
        "data_prep": {"steps": []},
        "model_understanding": {"steps": []},
        "models": {"steps": []},
    }

    max_per_section = {"images": 40, "markdown": 120, "html": 40, "text": 80, "plotly": 25}
    counters = {k: {kk: 0 for kk in max_per_section} for k in sections}

    def ensure_step(sec: str, title: str | None = None) -> dict[str, Any]:
        steps = sections[sec]["steps"]
        if not steps:
            steps.append(_new_step("Overview"))
        if title:
            if steps and steps[-1]["title"] != title:
                steps.append(_new_step(title))
        return steps[-1]

    for cell in cells:
        ctype = cell.get("cell_type")

        if ctype == "markdown":
            raw = _cell_text(cell).strip()
            if not raw:
                continue

            # Section detection
            if _RE_A.search(raw):
                section = "data_prep"
                current_step_title = None
            elif _RE_C_TS.search(raw):
                section = "models"
                current_step_title = None
            elif _RE_B.search(raw) or _RE_B_DIAG.search(raw):
                section = "model_understanding"
                current_step_title = None
            elif _RE_C.search(raw):
                section = "models"
                current_step_title = None

            if section not in sections:
                continue

            # Step detection (use first heading inside the markdown block)
            heading = _first_heading_title(raw)
            if heading:
                current_step_title = heading

            step = ensure_step(section, current_step_title)
            if counters[section]["markdown"] < max_per_section["markdown"]:
                counters[section]["markdown"] += 1
                step["markdown"].append(_render_markdown(raw))
            continue

        if ctype != "code":
            continue

        if section not in sections:
            continue

        step = ensure_step(section, current_step_title)

        # Plotly (interactive)
        for plot in _iter_plotly_outputs(cell):
            if counters[section]["plotly"] >= max_per_section["plotly"]:
                break
            counters[section]["plotly"] += 1
            step["plotly"].append(plot)

        # HTML outputs (tables, rich text)
        for h in _iter_html_outputs(cell):
            if counters[section]["html"] >= max_per_section["html"]:
                break
            counters[section]["html"] += 1
            step["html"].append(h)

        # Text outputs (prints, metrics, errors)
        for t in _iter_text_outputs(cell):
            if counters[section]["text"] >= max_per_section["text"]:
                break
            counters[section]["text"] += 1
            step["text"].append(t)

        # Image outputs (plots)
        for mime, payload in _iter_image_outputs(cell):
            if counters[section]["images"] >= max_per_section["images"]:
                break
            decoded = _decode_image_bytes(mime, payload)
            if not decoded:
                continue
            blob, ext = decoded
            counters[section]["images"] += 1
            filename = f"{section}_{counters[section]['images']:02d}.{ext}"
            file_path = out_dir / filename
            try:
                file_path.write_bytes(blob)
                step["images"].append(f"/static/notebook_assets/{spec.key}/{filename}")
            except Exception:
                continue

    manifest = {
        "version": 6,
        "source_mtime": nb_mtime,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sections": sections,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return sections


def _extract_assets_targeted(
    spec: NotebookSpec,
    section_groups: dict[str, list[dict[str, Any]]],
    *,
    include_markdown: set[str],
    include_images: set[str],
    include_html: set[str],
    include_text: set[str],
    include_plotly: set[str],
    version: int,
) -> dict[str, Any]:
    nb = _load_notebook(spec.notebook_path)
    cells = nb.get("cells") or []

    out_dir = _assets_dir(spec.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    nb_mtime = spec.notebook_path.stat().st_mtime

    plan_sig = {
        "section_groups": section_groups,
        "include_markdown": sorted(include_markdown),
        "include_images": sorted(include_images),
        "include_html": sorted(include_html),
        "include_text": sorted(include_text),
        "include_plotly": sorted(include_plotly),
    }

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source_mtime") == nb_mtime
                and manifest.get("version") == version
                and manifest.get("plan") == plan_sig
            ):
                return manifest.get("sections", {})
        except Exception:
            pass

    sections: dict[str, Any] = {sec_key: {"steps": []} for sec_key in section_groups.keys()}
    max_per_section = {"images": 80, "markdown": 200, "html": 80, "text": 160, "plotly": 40}
    counters = {k: {kk: 0 for kk in max_per_section} for k in sections}

    for sec_key, groups in section_groups.items():
        steps: list[dict[str, Any]] = sections[sec_key]["steps"]
        for g in groups:
            title = str(g.get("title") or "Overview")
            indices = g.get("cells") or []
            step = _new_step(title)
            steps.append(step)

            for i in indices:
                idx = i - 1
                if idx < 0 or idx >= len(cells):
                    continue
                cell = cells[idx]
                ctype = cell.get("cell_type")

                if ctype == "markdown" and sec_key in include_markdown:
                    raw = _cell_text(cell).strip()
                    if not raw:
                        continue
                    if counters[sec_key]["markdown"] < max_per_section["markdown"]:
                        counters[sec_key]["markdown"] += 1
                        step["markdown"].append(_render_markdown(raw))
                    continue

                if ctype != "code":
                    continue

                if sec_key in include_plotly:
                    for plot in _iter_plotly_outputs(cell):
                        if counters[sec_key]["plotly"] >= max_per_section["plotly"]:
                            break
                        counters[sec_key]["plotly"] += 1
                        step["plotly"].append(plot)

                if sec_key in include_html:
                    for h in _iter_html_outputs(cell):
                        if counters[sec_key]["html"] >= max_per_section["html"]:
                            break
                        counters[sec_key]["html"] += 1
                        step["html"].append(h)

                if sec_key in include_text:
                    for t in _iter_text_outputs(cell):
                        if counters[sec_key]["text"] >= max_per_section["text"]:
                            break
                        counters[sec_key]["text"] += 1
                        step["text"].append(t)

                if sec_key in include_images:
                    for mime, payload in _iter_image_outputs(cell):
                        if counters[sec_key]["images"] >= max_per_section["images"]:
                            break
                        decoded = _decode_image_bytes(mime, payload)
                        if not decoded:
                            continue
                        blob, ext = decoded
                        counters[sec_key]["images"] += 1
                        filename = f"{sec_key}_{counters[sec_key]['images']:02d}.{ext}"
                        file_path = out_dir / filename
                        try:
                            file_path.write_bytes(blob)
                            step["images"].append(f"/static/notebook_assets/{spec.key}/{filename}")
                        except Exception:
                            continue

        # If no explicit groups were provided, keep UI stable
        if not steps:
            steps.append(_new_step("Overview"))

    manifest = {
        "version": version,
        "source_mtime": nb_mtime,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "plan": plan_sig,
        "sections": sections,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return sections


def get_notebook_view(key: str) -> dict[str, Any]:
    specs = {
        "supplier": NotebookSpec(
            key="supplier",
            title="Supplier Classification",
            subtitle="Notebook visualizations",
            notebook_path=_resolve_notebook_path("1_supplier_classification_postgres_v2.ipynb"),
            models_used=["KMeans", "DBSCAN"],
        ),
        "sell": NotebookSpec(
            key="sell",
            title="Best Time To Sell",
            subtitle="Notebook visualizations",
            notebook_path=_resolve_notebook_path("2_best_time_to_sell_postgres_v2.ipynb"),
            models_used=["Prophet", "SARIMAX", "XGBoost"],
        ),
        "promote": NotebookSpec(
            key="promote",
            title="Best Time To Promote",
            subtitle="Notebook visualizations",
            notebook_path=_resolve_notebook_path("3_best_time_to_promote_postgres_v2.ipynb"),
            models_used=["Prophet", "SARIMAX", "XGBoost"],
        ),
    }

    if key not in specs:
        raise RuntimeError(f"Unknown notebook key: {key}")

    spec = specs[key]
    if not spec.notebook_path.exists():
        raise RuntimeError(f"Notebook not found: {spec.notebook_path}")

    third_tab_label = "Models"
    third_section_title = "Models (results & comparison)"
    if key == "supplier":
        third_tab_label = "Clustering"
        third_section_title = "Clustering"
    elif key in ("sell", "promote"):
        third_tab_label = "Time Series / Forecasting"
        third_section_title = "Time Series / Forecasting"

    curated_intro_md: dict[str, dict[str, str]] = {
        "supplier": {
            "data_prep": (
                "### A — Data Preparation & Feature Engineering\n\n"
                "We build a **supplier-level feature table** to make segmentation possible. Typical steps:\n\n"
                "- Clean and standardize key numeric business signals (cost, quantities, frequency).\n"
                "- Encode categorical fields (supplier region / type) using one-hot encoding.\n"
                "- Scale numeric features (critical for distance-based clustering).\n\n"
                "Goal: produce a single matrix $X$ where each row is a supplier and each column is a business-driven feature."
            ),
            "model_understanding": (
                "### B — Model Understanding (unsupervised clustering)\n\n"
                "#### Problem context\n"
                "- There is **no target variable** (no labels). We want to **segment** suppliers using business criteria: "
                "**address/distance**, **cost**, **volume**, and **purchase frequency**.\n"
                "- We work with mixed features: numerical (standardized) + categorical (one-hot).\n\n"
                "---\n"
                "#### Model 1 — K-Means\n\n"
                "**1) Intuition (how it works)**\n"
                "- K-Means partitions suppliers into $k$ groups by minimizing the sum of (Euclidean) distances to **centroids**.\n"
                "- Each supplier is assigned to the nearest centroid; centroids are iteratively updated.\n\n"
                "**2) Key parameters**\n"
                "- `n_clusters (k)`: number of clusters to form.\n"
                "- `n_init`: number of initializations (stability).\n"
                "- `random_state`: reproducibility.\n\n"
                "**3) Assumptions / validity conditions**\n"
                "- Clusters are relatively **compact** and separable in the feature space (roughly “spherical” under Euclidean distance).\n"
                "- Sensitive to feature scale ⇒ numerical **standardization** is required.\n"
                "- One-hot encoding increases dimensionality; interpretation depends on the chosen features.\n\n"
                "**4) Limitations**\n"
                "- Can struggle with non-spherical clusters, very different densities, or heavy noise.\n"
                "- Requires choosing $k$ (Elbow/Silhouette/DBI help, but do not mathematically prove a “true” $k$).\n\n"
                "---\n"
                "#### Model 2 — DBSCAN\n\n"
                "**1) Intuition (how it works)**\n"
                "- DBSCAN groups points by **density**: a point belongs to a cluster if it lies in a dense region (enough neighbors in a radius).\n"
                "- Isolated points are labeled as **noise** (`-1`).\n\n"
                "**2) Key parameters**\n"
                "- `eps`: neighborhood radius.\n"
                "- `min_samples`: minimum number of neighbors to be considered “dense”.\n"
                "- (implicit) **distance**: Euclidean distance on the preprocessed matrix.\n\n"
                "**3) Assumptions / validity conditions**\n"
                "- Clusters correspond to sufficiently dense regions separated by low-density areas.\n"
                "- Requires an `eps` consistent with the scaled space (hence standardization and the k-distance plot).\n"
                "- In high dimension (one-hot), distances can be less informative ⇒ DBSCAN may return 0/1 cluster + noise.\n\n"
                "**4) Limitations**\n"
                "- Very sensitive to `eps` and `min_samples` (small changes can drastically alter results).\n"
                "- Can fail to find multiple clusters when densities are similar or dimensionality is high.\n"
                "- Many noise points can be useful (atypical suppliers), but can also indicate poor parameter choice.\n\n"
                "---\n"
                "#### Why these models (justification)\n"
                "- **Why K-Means**: we want a **stable, business-friendly** segmentation (controlled number of segments), easy to profile and communicate.\n"
                "- **Why DBSCAN**: we also want to detect **atypical suppliers** (rare purchases, very different profiles, data issues) via the **noise label `-1`**.\n"
                "- **Why 2 models**: comparing a centroid-based model (KMeans) and a density/outlier model (DBSCAN) provides a more robust view: segmentation + anomaly detection.\n\n"
                "---\n"
                "#### Unsupervised evaluation metrics\n"
                "- **Silhouette** (higher is better): intra-cluster cohesion vs inter-cluster separation.\n"
                "- **Davies–Bouldin** (lower is better): compactness and separation.\n"
                "- For **DBSCAN**, also track: `n_clusters` (excluding noise) and `noise_points` (size of the noise set).\n"
            ),
            "models": (
                "### C — Clustering (≥ 2 models + comparison)\n\n"
                "### C.1 Clustering dataset\n\n"
                "We cluster suppliers using:\n\n"
                "Categorical: governorate, city (grouped)\n\n"
                "Numerical: distance, cost, spend, volume, and purchase frequency\n\n"
                "The matrix Xc_mat is already prepared in A.4 (encoding + scaling).\n\n"
                "---\n\n"
                "### C.2 Model 2 — DBSCAN (density + outlier detection)\n\n"
                "We test a few values for `eps` and `min_samples` and keep a configuration that produces **at least 2 clusters** when possible.\n\n"
                "Evaluation: silhouette + Davies–Bouldin (when clusters are valid) + number of noise points (`-1`).\n\n"
                "Also include a cluster-size figure for the chosen DBSCAN configuration (clusters vs noise).\n\n"
                "---\n\n"
            ),
        },
        "sell": {
            "data_prep": (
                "### A — Data Preparation & Feature Engineering\n\n"
                "We create a **daily revenue time series** (and optional calendar features) to prepare forecasting:\n\n"
                "- Aggregate raw transactions to daily granularity\n"
                "- Handle missing days and outliers\n"
                "- Build time-based features (lags, rolling means, day-of-week/month)\n"
            ),

        }
    }

    section_intro: dict[str, str] = {}
    curated_for_key = curated_intro_md.get(key, {})
    for sec_key, md_text in curated_for_key.items():
        if md_text and md_text.strip():
            section_intro[sec_key] = _render_markdown(md_text)

    extra_tabs: list[dict[str, str]] = []

    if key == "supplier":
        # Targeted supplier tabs by specific notebook cells (1-based global indices)
        # Data Prep: show visuals from the requested region (feature engineering + A.3 plots + encoding/scaling)
        # Model Understanding: show markdown + pedagogical visualizations
        # Clustering: show clustering comparison visuals
        # Results: show the requested results cell (27)
        section_groups = {
            "data_prep": [
                {"title": "Raw data quality", "cells": [9, 10]},
                {"title": "Data cleaning", "cells": [11, 12]},
                {"title": "After cleaning — visual checks", "cells": [13, 14]},
                {"title": "Feature engineering", "cells": [15, 16]},
                {"title": "Supplier features — visualizations", "cells": [17, 18]},
                {"title": "Encoding + scaling", "cells": [19, 20]},
            ],
            "model_understanding": [
                {"title": "KMeans / DBSCAN — pedagogical visuals", "cells": [23, 24]},
            ],
            "models": [
                {"title": "Evaluation & comparison — KMeans", "cells": [29]},
                {"title": "Evaluation & comparison — DBSCAN", "cells": [30, 31]},
                {"title": "Evaluation & comparison — Model comparison", "cells": [32, 33]},
                {"title": "Cluster visual comparison", "cells": [34, 35]},
            ],
            "results": [
                {"title": "Results", "cells": [37]},
            ],
        }
        sections = _extract_assets_targeted(
            spec,
            section_groups,
            include_markdown={"data_prep", "model_understanding", "models"},
            include_images={"data_prep", "model_understanding", "models"},
            include_html={"data_prep", "results"},
            include_text={"results"},
            include_plotly={"data_prep", "model_understanding", "models"},
            version=10,
        )
        extra_tabs = [{"key": "results", "label": "Results", "title": "Results"}]
    elif key == "promote":
        # Keep web sections in the exact same order as notebook cells.
        section_groups = {
            "data_prep": [
                {"title": "1) Data Preparation & Feature Engineering", "cells": [7]},
                {"title": "DW extraction & series build", "cells": [8, 9, 10]},
                {"title": "Data Cleaning (missing values, outliers)", "cells": [11, 12]},
                {"title": "Feature Engineering & Selection", "cells": [13, 14, 15]},
            ],
            "model_understanding": [
                {"title": "2) Model Understanding (mandatory for all models used)", "cells": [16]},
            ],
            "models": [
                {"title": "3) Time Series / Forecasting (intro)", "cells": [17]},
                {"title": "Stationarity & decomposition", "cells": [18]},
                {"title": "Model comparison (holdout)", "cells": [19]},
                {"title": "Forecast + promotion calendar (2027)", "cells": [20]},
                {"title": "DB target sanity check", "cells": [21]},
                {"title": "Export final results", "cells": [22]},
            ],
        }
        sections = _extract_assets_targeted(
            spec,
            section_groups,
            include_markdown={"data_prep", "model_understanding", "models"},
            include_images={"data_prep", "models"},
            include_html={"data_prep", "models"},
            include_text={"data_prep", "models"},
            include_plotly={"data_prep", "models"},
            version=11,
        )
    else:
        sections = _extract_assets(spec)

    if key == "sell":
        recommendation_steps: list[dict[str, Any]] = []
        model_steps = (sections.get("models") or {}).get("steps") or []
        kept_model_steps: list[dict[str, Any]] = []

        for st in model_steps:
            title = str(st.get("title") or "").lower()
            if "recommendation" in title and "best time to sell" in title:
                recommendation_steps.append(st)
            else:
                kept_model_steps.append(st)

        if recommendation_steps:
            for rec_step in recommendation_steps:
                html_blocks = rec_step.get("html") or []
                if len(html_blocks) > 1:
                    rec_step["html"] = [html_blocks[0]]
            sections["models"]["steps"] = kept_model_steps
            sections["recommendation"] = {"steps": recommendation_steps}
            extra_tabs.append(
                {
                    "key": "recommendation",
                    "label": "Result",
                    "title": "Recommendation — Best Time to Sell",
                }
            )

    if key == "promote":
        result_steps: list[dict[str, Any]] = []
        model_steps = (sections.get("models") or {}).get("steps") or []
        kept_model_steps: list[dict[str, Any]] = []

        result_titles = {
            "forecast + promotion calendar (2027)",
            "db target sanity check",
            "export final results",
        }

        for st in model_steps:
            title = str(st.get("title") or "").strip().lower()
            if title in result_titles:
                result_steps.append(st)
            else:
                kept_model_steps.append(st)

        if result_steps:
            sections["models"]["steps"] = kept_model_steps
            sections["results"] = {"steps": result_steps}
            extra_tabs.append(
                {
                    "key": "results",
                    "label": "Result",
                    "title": "Result",
                }
            )

    # If the notebook was never executed (no outputs), there may be no assets.
    hints = []

    def _has_any(sec_key: str) -> bool:
        sec = sections.get(sec_key, {})
        steps = sec.get("steps") or []
        for st in steps:
            if (st.get("images") or st.get("plotly") or st.get("html") or st.get("text") or st.get("markdown")):
                return True
        return False

    if not any(_has_any(k) for k in ("data_prep", "model_understanding", "models")):
        hints.append(
            "No notebook outputs found for display. Run the notebook cells to generate plots/tables, then refresh this page."
        )

    return {
        "key": key,
        "title": spec.title,
        "subtitle": spec.subtitle,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "models_used": spec.models_used,
        "sections": sections,
        "third_tab_label": third_tab_label,
        "third_section_title": third_section_title,
        "extra_tabs": extra_tabs,
        "section_intro": section_intro,
        "hints": hints,
    }
