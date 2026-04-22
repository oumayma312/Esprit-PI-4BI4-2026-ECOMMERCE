from pathlib import Path
import html
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st


st.set_page_config(page_title="Prediction Churn et Segmentation", layout="wide")


SEGMENT_LABELS = {
    0: "Loyaux a forte valeur",
    1: "Clients perdus",
    2: "Intermediaires a risque",
    3: "Clients recents ou nouveaux",
}

SEGMENT_PROFILES = {
    0: {
        "meaning": "Clients premium avec depenses elevees et achats frequents.",
        "typical_profile": "Panier moyen eleve, engagement fort, vitesse d'achat soutenue.",
        "action": "Preserver la fidelite via avantages VIP, offres exclusives et suivi proactif.",
    },
    1: {
        "meaning": "Clients probablement inactifs depuis une longue periode.",
        "typical_profile": "Engagement recent faible, dynamique en baisse et risque de churn eleve.",
        "action": "Activer des campagnes de reconquete avec offre limitee et personnalisation.",
    },
    2: {
        "meaning": "Clients de valeur moyenne montrant des signes de desengagement.",
        "typical_profile": "Valeur correcte mais frequence qui baisse et risque qui monte.",
        "action": "Lancer des parcours de nurturing et des bundles cibles rapidement.",
    },
    3: {
        "meaning": "Clients recents avec historique d'achat encore limite.",
        "typical_profile": "Anciennete faible, volume d'achat bas, mais activite recente.",
        "action": "Accelerer l'onboarding avec education produit et cross-sell precoce.",
    },
}


PREPARED_REQUIRED_COLUMNS = {
    "recency",
    "frequency",
    "monetary_total",
    "monetary_trend",
    "product_diversity",
    "channel_diversity",
    "lifetime_days",
    "purchase_velocity",
    "avg_price",
}

TRANSACTION_REQUIRED_COLUMNS = {
    "dateFK",
    "total_ttc",
    "quantity",
    "price",
    "channelFK",
    "productFK",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Work+Sans:wght@400;500;600;700&display=swap');

            :root {
                --bg-1: #f2f6f9;
                --bg-2: #eaf0f4;
                --surface: #ffffff;
                --ink-900: #102f44;
                --ink-800: #1f4158;
                --ink-700: #284b63;
                --ink-500: #385a70;
                --line: #cfdde7;
                --brand: #0f6d8f;
                --brand-dark: #0b4f68;
                --good: #1f8f5f;
                --warn: #b97815;
                --bad: #b64249;
            }

            html, body, [class*="css"] {
                font-family: 'Work Sans', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(1000px 420px at 8% -8%, #dceffe 0%, transparent 58%),
                    radial-gradient(900px 380px at 100% 2%, #eafcee 0%, transparent 52%),
                    linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
            }

            section.main > div {
                padding-top: 1rem;
            }

            /* Force readable text in main area even when Streamlit theme is dark. */
            [data-testid="stAppViewContainer"] section.main {
                --text-color: #102f44;
                color: #102f44 !important;
            }

            [data-testid="stAppViewContainer"] section.main p,
            [data-testid="stAppViewContainer"] section.main span,
            [data-testid="stAppViewContainer"] section.main li,
            [data-testid="stAppViewContainer"] section.main label,
            [data-testid="stAppViewContainer"] section.main [data-testid="stWidgetLabel"],
            [data-testid="stAppViewContainer"] section.main [data-testid="stWidgetLabel"] *,
            [data-testid="stAppViewContainer"] section.main [data-testid="stMarkdownContainer"],
            [data-testid="stAppViewContainer"] section.main [data-testid="stMarkdownContainer"] *,
            [data-testid="stAppViewContainer"] section.main [data-testid="stCaptionContainer"],
            [data-testid="stAppViewContainer"] section.main [data-testid="stCaptionContainer"] * {
                color: #102f44 !important;
                opacity: 1 !important;
                background: transparent !important;
            }

            [data-testid="stAppViewContainer"] section.main [data-testid="stExpander"] [role="region"],
            [data-testid="stAppViewContainer"] section.main [data-testid="stExpander"] [role="region"] * {
                color: #102f44 !important;
                opacity: 1 !important;
                background: transparent !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1b4a63 0%, #1b5d7d 100%);
                border-right: 1px solid #2d6b8c;
            }

            [data-testid="stSidebar"] * {
                color: #ffffff;
            }

            [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] input,
            [data-testid="stSidebar"] [data-baseweb="select"] > div {
                background: rgba(255, 255, 255, 0.26) !important;
                border: 1px solid rgba(255, 255, 255, 0.5) !important;
                color: #ffffff !important;
            }

            .hero {
                border-radius: 20px;
                padding: 1.35rem 1.5rem;
                border: 1px solid #c3d9e7;
                background: linear-gradient(130deg, #ffffff 0%, #f1f8fd 45%, #ecf7f2 100%);
                box-shadow: 0 8px 28px rgba(13, 55, 78, 0.09);
                margin-bottom: 0.85rem;
            }

            .hero h1 {
                margin: 0;
                font-family: 'Sora', sans-serif;
                font-size: clamp(1.5rem, 2vw, 2.05rem);
                font-weight: 800;
                letter-spacing: -0.02em;
                color: var(--ink-900);
            }

            .hero-sub {
                margin: 0.55rem 0 0;
                color: var(--ink-800);
                font-size: 1.01rem;
                max-width: 74ch;
                line-height: 1.55;
            }

            .hero-chips {
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.9rem;
            }

            .hero-chip {
                border: 1px solid #cfdfeb;
                background: #fff;
                color: var(--ink-800);
                border-radius: 999px;
                padding: 0.25rem 0.62rem;
                font-size: 0.78rem;
                font-weight: 700;
            }

            .section-title {
                font-family: 'Sora', sans-serif;
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--ink-900);
                margin: 0.35rem 0 0.55rem;
                letter-spacing: -0.01em;
            }

            .status-card {
                border-radius: 12px;
                border: 1px solid #b5d8c4;
                background: #eefaf2;
                color: #173f2a;
                padding: 0.62rem 0.8rem;
                font-size: 0.94rem;
                margin-bottom: 0.65rem;
            }

            .status-card code {
                background: rgba(29, 95, 67, 0.12);
                border-radius: 6px;
                padding: 0.1rem 0.35rem;
                color: #123d2a;
            }

            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-left: 5px solid #6297b3;
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
                box-shadow: 0 4px 16px rgba(8, 42, 63, 0.07);
            }

            div[data-testid="stMetricLabel"] {
                color: var(--ink-500);
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                font-size: 0.74rem;
            }

            div[data-testid="stMetricValue"] {
                color: var(--ink-900);
                font-family: 'Sora', sans-serif;
                font-weight: 700;
            }

            .stButton > button,
            .stFormSubmitButton > button,
            .stDownloadButton > button {
                border-radius: 10px !important;
                border: 1px solid #0f6382 !important;
                background: linear-gradient(180deg, #147ca1 0%, #106a8b 100%) !important;
                color: #ffffff !important;
                font-weight: 700 !important;
                box-shadow: 0 6px 18px rgba(11, 81, 107, 0.24);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover,
            .stDownloadButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 8px 22px rgba(11, 81, 107, 0.3);
            }

            .stButton > button:focus-visible,
            .stFormSubmitButton > button:focus-visible,
            .stDownloadButton > button:focus-visible,
            div[data-testid="stTextInputRootElement"] input:focus-visible,
            div[data-testid="stNumberInputContainer"] input:focus-visible,
            div[data-baseweb="select"] > div:focus-within {
                outline: 3px solid #09374d !important;
                outline-offset: 1px !important;
                box-shadow: none !important;
            }

            div[data-baseweb="select"] > div,
            div[data-testid="stNumberInputContainer"] input,
            div[data-testid="stTextInputRootElement"] input {
                border-radius: 10px !important;
                border-color: #abc9db !important;
                background: #ffffff !important;
                color: #17394c !important;
            }

            [data-testid="stExpander"] {
                border: 1px solid var(--line);
                border-radius: 12px;
                background: #fbfdff;
            }

            [data-testid="stExpander"] details > summary {
                background: #122f44;
                color: #ffffff !important;
                border-radius: 10px;
                padding: 0.2rem 0.35rem;
                font-weight: 700;
            }

            [data-testid="stExpander"] details > summary * {
                color: #ffffff !important;
                background: transparent !important;
            }

            section.main [data-testid="stMarkdownContainer"] p,
            section.main [data-testid="stMarkdownContainer"] li,
            section.main [data-testid="stMarkdownContainer"] span,
            section.main [data-testid="stMarkdownContainer"] div,
            section.main [data-testid="stMarkdownContainer"] strong {
                color: var(--ink-900) !important;
                background: transparent !important;
            }

            section.main [data-testid="stWidgetLabel"] *,
            section.main label {
                color: var(--ink-900) !important;
                background: transparent !important;
                font-weight: 600;
            }

            section.main [data-testid="stSelectbox"] label,
            section.main [data-testid="stNumberInput"] label,
            section.main [data-testid="stTextInput"] label,
            section.main [data-testid="stFileUploader"] label,
            section.main [data-testid="stSelectbox"] label *,
            section.main [data-testid="stNumberInput"] label *,
            section.main [data-testid="stTextInput"] label *,
            section.main [data-testid="stFileUploader"] label * {
                color: var(--ink-900) !important;
                opacity: 1 !important;
                background: transparent !important;
                font-weight: 600;
            }

            section.main [data-testid="stExpanderDetails"],
            section.main [data-testid="stExpanderDetails"] *,
            section.main [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
            section.main [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
                color: var(--ink-900) !important;
                opacity: 1 !important;
                background: transparent !important;
            }

            section.main small,
            section.main [data-testid="stCaptionContainer"],
            section.main [data-testid="stCaptionContainer"] * {
                color: var(--ink-700) !important;
                opacity: 1 !important;
            }

            section.main [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
                color: var(--ink-900) !important;
                background: transparent !important;
            }

            section.main mark {
                background: #ffe6a6 !important;
                color: #1a3345 !important;
            }

            .viz-card {
                border: 1px solid var(--line);
                border-radius: 14px;
                background: #ffffff;
                padding: 0.7rem 0.85rem;
                margin-bottom: 0.7rem;
            }

            .legend-item {
                border: 1px solid #d8e5ef;
                border-radius: 10px;
                background: #ffffff;
                padding: 0.45rem 0.6rem;
                margin-bottom: 0.4rem;
                color: var(--ink-800);
                font-size: 0.9rem;
            }

            .legend-id {
                font-family: 'Sora', sans-serif;
                font-weight: 700;
                color: var(--brand-dark);
            }

            .risk-panel {
                border: 1px solid var(--line);
                border-radius: 14px;
                background: #ffffff;
                padding: 0.8rem 0.9rem;
                margin-top: 0.6rem;
            }

            .risk-head {
                display: flex;
                justify-content: space-between;
                gap: 0.6rem;
                font-size: 0.95rem;
                color: var(--ink-800);
                margin-bottom: 0.5rem;
                font-weight: 600;
            }

            .risk-track {
                width: 100%;
                height: 11px;
                border-radius: 999px;
                background: #e7eef3;
                overflow: hidden;
            }

            .risk-fill {
                height: 100%;
                border-radius: 999px;
                transition: width 0.6s ease;
            }

            .risk-bad {
                background:
                    repeating-linear-gradient(-45deg, rgba(255,255,255,0.2), rgba(255,255,255,0.2) 6px, rgba(0,0,0,0) 6px, rgba(0,0,0,0) 12px),
                    var(--bad);
            }

            .risk-warn {
                background: var(--warn);
            }

            .risk-good {
                background: var(--good);
            }

            .segment-box {
                border: 1px solid #d2e2ec;
                background: linear-gradient(180deg, #ffffff 0%, #f7fcff 100%);
                border-radius: 14px;
                padding: 0.95rem 1rem;
                margin-top: 0.55rem;
                color: #1f4358;
                line-height: 1.56;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65);
            }

            .segment-title {
                font-family: 'Sora', sans-serif;
                font-weight: 700;
                color: #10384f;
                margin-bottom: 0.35rem;
            }

            .insight-box {
                border-radius: 12px;
                border: 1px dashed #bdd6e4;
                padding: 0.75rem 0.85rem;
                color: var(--ink-800);
                background: #f9fdff;
                font-size: 0.9rem;
                line-height: 1.52;
            }

            .subtle {
                color: var(--ink-700);
                font-size: 0.91rem;
                line-height: 1.5;
            }

            @media (max-width: 900px) {
                .hero {
                    padding: 1.15rem 1.1rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts(artifacts_dir: str) -> dict:
    base = Path(artifacts_dir)
    required = {
        "churn_model": base / "churn_model.joblib",
        "churn_scaler": base / "churn_scaler.joblib",
        "segment_model": base / "segment_model.joblib",
        "segment_scaler": base / "segment_scaler.joblib",
        "config": base / "inference_config.json",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Fichiers d'artefacts manquants:\n" + "\n".join(missing))

    with open(required["config"], "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "churn_model": joblib.load(required["churn_model"]),
        "churn_scaler": joblib.load(required["churn_scaler"]),
        "segment_model": joblib.load(required["segment_model"]),
        "segment_scaler": joblib.load(required["segment_scaler"]),
        "config": config,
    }


@st.cache_data
def load_channel_lookup(channel_path: str = "data/dim_channel.csv") -> dict[str, str]:
    path = Path(channel_path)
    if not path.exists():
        return {}

    try:
        channel_df = pd.read_csv(path)
    except Exception:
        return {}

    if "channelPK" not in channel_df.columns or "channel" not in channel_df.columns:
        return {}

    lookup: dict[str, str] = {}
    for _, row in channel_df[["channelPK", "channel"]].dropna().iterrows():
        key = str(row["channelPK"]).strip()
        value = str(row["channel"]).strip()
        if not key or not value:
            continue
        lookup[key] = value
        try:
            lookup[str(int(float(key)))] = value
        except Exception:
            pass

    return lookup


def _coerce_numeric_frame(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    parse_issues = []
    for col in columns:
        before = out[col]
        out[col] = pd.to_numeric(out[col], errors="coerce")
        parse_failures = (out[col].isna() & before.notna()).sum()
        if parse_failures > 0:
            parse_issues.append(f"{col}: {int(parse_failures)} valeur(s) non numerique(s)")

    return out.fillna(0), parse_issues


def _ensure_log_features(df: pd.DataFrame, log_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for base_col in log_features:
        log_col = f"{base_col}_log"
        if log_col not in out.columns:
            if base_col in out.columns:
                base_values = pd.to_numeric(out[base_col], errors="coerce").fillna(0)
            else:
                base_values = pd.Series(0.0, index=out.index)
            out[log_col] = np.log1p(base_values.clip(lower=0))
    return out


def _known_channels_from_config(config: dict) -> list[str]:
    channels = []
    for col in config.get("churn_channel_dummy_columns", []):
        if col.startswith("ch_"):
            channels.append(col[3:])
    return sorted(set(channels))


def _encode_channel_dummies(df: pd.DataFrame, channel_value: str, config: dict) -> pd.DataFrame:
    out = df.copy()
    dummy_cols = config["churn_channel_dummy_columns"]

    for col in dummy_cols:
        out[col] = 0

    if channel_value:
        selected_dummy_col = f"ch_{channel_value}"
        if selected_dummy_col in dummy_cols:
            out[selected_dummy_col] = 1

    return out


def _build_churn_features(df: pd.DataFrame, channel_value: str, config: dict) -> tuple[pd.DataFrame, list[str]]:
    prepared = _ensure_log_features(df, config["log_features"])
    prepared = _encode_channel_dummies(prepared, channel_value, config)

    required = config["churn_feature_columns"]
    missing = [col for col in required if col not in prepared.columns]
    if missing:
        raise ValueError("Colonnes churn manquantes:\n" + "\n".join(missing))

    return _coerce_numeric_frame(prepared[required], required)


def _build_segment_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    prepared = _ensure_log_features(df, config["log_features"])
    required = config["segmentation_feature_columns"]

    missing = [col for col in required if col not in prepared.columns]
    if missing:
        raise ValueError("Colonnes segmentation manquantes:\n" + "\n".join(missing))

    return _coerce_numeric_frame(prepared[required], required)


def run_inference(df: pd.DataFrame, channel_value: str, artifacts: dict, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    churn_features, churn_parse_issues = _build_churn_features(df, channel_value, artifacts["config"])
    segment_features, segment_parse_issues = _build_segment_features(df, artifacts["config"])

    churn_scaled = artifacts["churn_scaler"].transform(churn_features)
    segment_scaled = artifacts["segment_scaler"].transform(segment_features)

    churn_probability = artifacts["churn_model"].predict_proba(churn_scaled)[:, 1]
    churn_label = (churn_probability >= threshold).astype(int)
    segment = artifacts["segment_model"].predict(segment_scaled)

    out = df.copy()
    out["churn_probability"] = np.round(churn_probability, 6)
    out["churn_prediction"] = churn_label
    out["segment"] = segment
    out["segment_label"] = out["segment"].map(SEGMENT_LABELS).fillna("Segment non etiquete")

    parse_issues = sorted(set(churn_parse_issues + segment_parse_issues))
    return out, parse_issues


def _business_recommendation(prob: float, segment_id: int) -> str:
    if prob >= 0.7:
        return "Action de retention immediate: offre ciblee et prise de contact proactive."
    if prob >= 0.4:
        return "A suivre de pres: lancer une campagne d'engagement et surveiller la prochaine fenetre d'achat."
    if segment_id == 0:
        return "Conserver la dynamique de fidelite avec avantages VIP et upsell personnalise."
    return "Profil sain: maintenir l'engagement regulier et monitorer les tendances."


def _parse_reference_date(config: dict) -> pd.Timestamp:
    raw_date = config.get("reference_date")
    parsed = pd.to_datetime(raw_date, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(parsed)


def _resolve_channel_name(channel_value: object, channel_lookup: dict[str, str]) -> str:
    if pd.isna(channel_value):
        return ""

    value = str(channel_value).strip()
    if not value:
        return ""

    if value in channel_lookup:
        return channel_lookup[value]

    try:
        as_int = str(int(float(value)))
        if as_int in channel_lookup:
            return channel_lookup[as_int]
    except Exception:
        pass

    return value


def _extract_channel_from_row(row: pd.Series, channel_lookup: dict[str, str]) -> str:
    for col in ["channel", "primary_channel", "canal", "channel_name", "channelFK"]:
        if col in row.index:
            resolved = _resolve_channel_name(row[col], channel_lookup)
            if resolved:
                return resolved
    return ""


def _detect_uploaded_schema(df: pd.DataFrame) -> str:
    columns = set(df.columns)
    has_prepared = PREPARED_REQUIRED_COLUMNS.issubset(columns)
    has_transactions = TRANSACTION_REQUIRED_COLUMNS.issubset(columns)

    if has_transactions:
        return "transactions"
    if has_prepared:
        return "prepared"
    return "unknown"


def _missing_columns(df: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required.difference(set(df.columns)))


def _derive_features_from_transactions(
    tx_df: pd.DataFrame,
    reference_date: pd.Timestamp,
    channel_lookup: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    df = tx_df.copy()

    missing_raw = _missing_columns(df, TRANSACTION_REQUIRED_COLUMNS)
    if missing_raw:
        raise ValueError("Colonnes transactions manquantes: " + ", ".join(missing_raw))

    if "customerFK" not in df.columns:
        df["customerFK"] = "client_importe_1"
        issues.append("Colonne customerFK absente: un client unique a ete suppose.")

    strict_dates = pd.to_datetime(df["dateFK"].astype(str), format="%Y%m%d", errors="coerce")
    loose_dates = pd.to_datetime(df["dateFK"], errors="coerce")
    df["dateFK"] = strict_dates.fillna(loose_dates)

    invalid_dates = int(df["dateFK"].isna().sum())
    if invalid_dates > 0:
        issues.append(f"{invalid_dates} ligne(s) ignoree(s) a cause de dateFK invalide.")
        df = df.dropna(subset=["dateFK"]).copy()

    if df.empty:
        raise ValueError("Aucune ligne exploitable apres nettoyage des dates.")

    for col in ["total_ttc", "quantity", "price"]:
        before = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        parse_failures = int((before.notna() & pd.to_numeric(before, errors="coerce").isna()).sum())
        if parse_failures > 0:
            issues.append(f"{col}: {parse_failures} valeur(s) convertie(s) a 0.")

    if "FactVentePK" not in df.columns:
        df["FactVentePK"] = 1
        issues.append("FactVentePK absent: la frequence est calculee sur le nombre de lignes.")

    df = df.sort_values(["customerFK", "dateFK"])

    customer_features = df.groupby("customerFK").agg(
        recency=("dateFK", lambda x: (reference_date - x.max()).days),
        first_purchase=("dateFK", "min"),
        last_purchase=("dateFK", "max"),
        frequency=("FactVentePK", "count"),
        monetary_total=("total_ttc", "sum"),
        monetary_mean=("total_ttc", "mean"),
        monetary_std=("total_ttc", "std"),
        quantity_total=("quantity", "sum"),
        quantity_mean=("quantity", "mean"),
        product_diversity=("productFK", "nunique"),
        channel_diversity=("channelFK", "nunique"),
        primary_channel=("channelFK", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
    ).reset_index()

    customer_features["monetary_std"] = customer_features["monetary_std"].fillna(0)
    customer_features["lifetime_days"] = (
        customer_features["last_purchase"] - customer_features["first_purchase"]
    ).dt.days
    customer_features["purchase_velocity"] = customer_features["frequency"] / (
        customer_features["lifetime_days"] / 30 + 1
    )

    last3_spend = (
        df.groupby("customerFK", group_keys=False)
        .apply(lambda g: g.tail(3)["total_ttc"].mean())
        .reset_index(name="last3_monetary_mean")
    )

    customer_features = customer_features.merge(last3_spend, on="customerFK", how="left")
    customer_features["monetary_trend"] = customer_features["last3_monetary_mean"] / (
        customer_features["monetary_mean"] + 1e-9
    )
    customer_features["recency_norm"] = customer_features["recency"] / (
        customer_features["lifetime_days"] + 1
    )

    customer_features["channel"] = customer_features["primary_channel"].map(
        lambda x: _resolve_channel_name(x, channel_lookup)
    )

    return customer_features, issues


def _run_batch_predictions(
    df: pd.DataFrame,
    artifacts: dict,
    threshold: float,
    known_channels: list[str],
    channel_lookup: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    results: list[pd.DataFrame] = []
    failed_rows: list[str] = []

    parse_issue_rows = 0
    unknown_channel_rows = 0

    source_df = df.reset_index().rename(columns={"index": "source_row"})

    for _, row in source_df.iterrows():
        source_row = int(row["source_row"]) + 1
        payload = row.drop(labels=["source_row"]).to_dict()
        row_df = pd.DataFrame([payload])

        raw_channel = _extract_channel_from_row(row, channel_lookup)
        if raw_channel and raw_channel not in known_channels:
            unknown_channel_rows += 1
            selected_channel = ""
        else:
            selected_channel = raw_channel

        try:
            pred, parse_issues = run_inference(row_df, selected_channel, artifacts, threshold)
            pred.insert(0, "ligne_source_csv", source_row)
            if parse_issues:
                parse_issue_rows += 1
                pred["nettoyage_entree"] = "; ".join(parse_issues)
            else:
                pred["nettoyage_entree"] = ""
            results.append(pred)
        except Exception as exc:
            failed_rows.append(f"Ligne {source_row}: {exc}")

    if not results:
        raise ValueError("Aucune ligne n'a pu etre predite a partir du fichier CSV.")

    out = pd.concat(results, ignore_index=True)
    summary = {
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "rows_failed": int(len(failed_rows)),
        "parse_issue_rows": int(parse_issue_rows),
        "unknown_channel_rows": int(unknown_channel_rows),
    }

    return out, summary, failed_rows


def _prepared_template_csv() -> bytes:
    template_df = pd.DataFrame(
        {
            "customerFK": [1001],
            "recency": [120],
            "frequency": [2],
            "monetary_total": [250.0],
            "monetary_trend": [1.0],
            "product_diversity": [3],
            "channel_diversity": [1],
            "lifetime_days": [365],
            "purchase_velocity": [1.5],
            "avg_price": [80.0],
            "channel": ["vente direct"],
        }
    )
    return template_df.to_csv(index=False).encode("utf-8")


def _transactions_template_csv() -> bytes:
    template_df = pd.DataFrame(
        {
            "customerFK": [1001, 1001, 1002],
            "FactVentePK": [1, 2, 3],
            "dateFK": [20251201, 20251228, 20251121],
            "total_ttc": [120.0, 210.0, 65.0],
            "quantity": [1, 2, 1],
            "price": [120.0, 105.0, 65.0],
            "channelFK": [1, 1, 1],
            "productFK": [10, 11, 7],
        }
    )
    return template_df.to_csv(index=False).encode("utf-8")


@st.cache_data
def ensure_pca_plot(
    source_csv_path: str = "customer_churn_predictions.csv",
    plots_dir: str = "plots",
    out_name: str = "segmentation_pca.png",
) -> str | None:
    plots_path = Path(plots_dir)
    plots_path.mkdir(parents=True, exist_ok=True)
    out_path = plots_path / out_name

    if out_path.exists():
        return str(out_path)

    source_path = Path(source_csv_path)
    if not source_path.exists():
        return None

    df = pd.read_csv(source_path)
    needed = {"recency", "frequency", "monetary_total", "segment"}
    if not needed.issubset(set(df.columns)):
        return None

    df = df.dropna(subset=["recency", "frequency", "monetary_total", "segment"]).copy()
    if len(df) < 5:
        return None

    df["monetary_total_log"] = np.log1p(pd.to_numeric(df["monetary_total"], errors="coerce").fillna(0).clip(lower=0))

    feats = df[["recency", "frequency", "monetary_total_log"]].copy()
    feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled = StandardScaler().fit_transform(feats)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

    plot_df = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "segment": df["segment"].astype(int).values,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    for seg_id in sorted(plot_df["segment"].unique()):
        part = plot_df[plot_df["segment"] == seg_id]
        label = f"{seg_id} - {SEGMENT_LABELS.get(seg_id, 'Inconnu')}"
        ax.scatter(part["pc1"], part["pc2"], s=18, alpha=0.65, label=label)

    ax.set_title("Projection PCA des segments clients")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return str(out_path)


_inject_styles()
st.markdown(
    """
    <div class="hero">
        <h1>Conseiller Churn et Segmentation Client</h1>
        <p class="hero-sub">Espace de scoring client pour estimer le risque de churn et le segment comportemental, avec recommandations operationnelles.</p>
        <div class="hero-chips">
            <span class="hero-chip">Inference temps reel</span>
            <span class="hero-chip">Scoring double modele</span>
            <span class="hero-chip">Insights actionnables</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Parametres")
    artifacts_dir = st.text_input("Repertoire des artefacts", value="artifacts")
    if st.button("Actualiser le graphique PCA"):
        st.cache_data.clear()
    if st.button("Recharger les artefacts"):
        st.cache_resource.clear()

try:
    artifacts = load_artifacts(artifacts_dir)
except Exception as exc:
    st.error(str(exc))
    st.info(
        "Executez d'abord la cellule d'export du notebook pour generer artifacts/. "
        "Fichiers attendus: churn_model.joblib, churn_scaler.joblib, "
        "segment_model.joblib, segment_scaler.joblib, inference_config.json"
    )
    st.stop()

st.markdown(
    f"""
    <div class="status-card">
        Artefacts du modele charges avec succes depuis <code>{html.escape(artifacts_dir)}</code>.
    </div>
    """,
    unsafe_allow_html=True,
)

config = artifacts["config"]
reference_date = _parse_reference_date(config)
channel_lookup = load_channel_lookup("data/dim_channel.csv")
known_channels = _known_channels_from_config(config)
channel_options = ["Autre / baseline"] + known_channels
fixed_threshold = float(config.get("decision_threshold", 0.5))
pca_plot_path = ensure_pca_plot("customer_churn_predictions.csv", "plots")

with st.expander("Comment fonctionne cette prediction", expanded=False):
    st.write("Cet ecran execute deux modeles sur un profil client:")
    st.write("1. Modele churn: estime la probabilite de churn et la classe churn/actif.")
    st.write("2. Modele segmentation: assigne un segment comportemental.")
    st.write(f"Le seuil de decision est fixe depuis le notebook: {fixed_threshold:.2f}.")

left_col, right_col = st.columns([1.35, 0.9], gap="large")

with left_col:
    st.markdown('<div class="section-title">Profil Client</div>', unsafe_allow_html=True)
    with st.form("single_customer_form", clear_on_submit=False):
        channel_value = st.selectbox("Canal principal", options=channel_options, index=0)

        st.markdown('<div class="section-title">Variables d activite</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            recency = st.number_input("Recence (jours)", min_value=0.0, value=120.0)
            frequency = st.number_input("Frequence (transactions)", min_value=0.0, value=2.0)
            lifetime_days = st.number_input("Anciennete (jours)", min_value=0.0, value=365.0)
        with a2:
            purchase_velocity = st.number_input("Vitesse d achat", min_value=0.0, value=1.5)
            product_diversity = st.number_input("Diversite produits", min_value=0.0, value=3.0)
            channel_diversity = st.number_input("Diversite canaux", min_value=0.0, value=1.0)
        with a3:
            monetary_trend = st.number_input("Tendance depense", min_value=0.0, value=1.0)
            monetary_total = st.number_input("Depense totale", min_value=0.0, value=250.0)
            avg_price = st.number_input("Prix moyen", min_value=0.0, value=80.0)
        submitted = st.form_submit_button("Predire churn et segment", width="stretch")

with right_col:
    st.markdown('<div class="section-title">Carte PCA des segments</div>', unsafe_allow_html=True)
    st.markdown('<div class="viz-card">', unsafe_allow_html=True)
    if pca_plot_path and Path(pca_plot_path).exists():
        st.image(pca_plot_path, caption="Image chargee depuis ./plots/segmentation_pca.png", width="stretch")
    else:
        st.info("Aucune image PCA disponible. Ajoutez customer_churn_predictions.csv puis actualisez PCA.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Legende des segments</div>', unsafe_allow_html=True)
    for seg_id in sorted(SEGMENT_LABELS):
        st.markdown(
            f"<div class='legend-item'><span class='legend-id'>Segment {seg_id}</span> - {html.escape(SEGMENT_LABELS[seg_id])}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="insight-box">
            Conseil: comparez le segment predit et la probabilite de churn. Les segments a forte valeur avec risque en hausse meritent une intervention prioritaire.
        </div>
        """,
        unsafe_allow_html=True,
    )

if submitted:
    one_row = {
        "recency": recency,
        "frequency": frequency,
        "monetary_total": monetary_total,
        "monetary_trend": monetary_trend,
        "product_diversity": product_diversity,
        "channel_diversity": channel_diversity,
        "lifetime_days": lifetime_days,
        "purchase_velocity": purchase_velocity,
        "avg_price": avg_price,
    }

    one_df = pd.DataFrame([one_row])

    selected_channel = "" if channel_value == "Autre / baseline" else channel_value

    try:
        with st.spinner("Execution des modeles..."):
            one_pred, parse_issues = run_inference(one_df, selected_channel, artifacts, fixed_threshold)

        score = float(one_pred.loc[0, "churn_probability"])
        churn_class = "Churne" if int(one_pred.loc[0, "churn_prediction"]) == 1 else "Actif"
        segment_id = int(one_pred.loc[0, "segment"])
        segment_label = str(one_pred.loc[0, "segment_label"])

        st.markdown('<div class="section-title">Resultat de prediction</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilite churn", f"{score:.1%}")
        m2.metric("Statut predit", churn_class)
        m3.metric("Segment", f"{segment_id} - {segment_label}")

        st.caption(f"Classification basee sur le seuil fixe du notebook: {fixed_threshold:.2f}")

        risk_tier = "Faible"
        risk_class = "risk-good"
        if score >= 0.7:
            risk_tier = "Eleve"
            risk_class = "risk-bad"
        elif score >= 0.4:
            risk_tier = "Modere"
            risk_class = "risk-warn"

        st.markdown(
            f"""
            <div class="risk-panel">
                <div class="risk-head">
                    <span>Niveau de risque: <strong>{risk_tier}</strong></span>
                    <span>Probabilite: <strong>{score:.1%}</strong></span>
                </div>
                <div class="risk-track">
                    <div class="risk-fill {risk_class}" style="width: {score * 100:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        recommendation = _business_recommendation(score, segment_id)
        st.info(f"Action recommandee: {recommendation}")

        profile = SEGMENT_PROFILES.get(segment_id)
        if profile:
            st.markdown('<div class="section-title">Interpretation du segment</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="segment-box">
                    <div class="segment-title">{html.escape(segment_label)}</div>
                    <strong>Signification:</strong> {html.escape(profile['meaning'])}<br>
                    <strong>Profil type:</strong> {html.escape(profile['typical_profile'])}<br>
                    <strong>Strategie suggeree:</strong> {html.escape(profile['action'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if parse_issues:
            st.warning("Nettoyage automatique applique: " + "; ".join(parse_issues))

        if selected_channel and selected_channel not in known_channels:
            st.warning("Canal hors apprentissage: encodage baseline applique.")

    except Exception as exc:
        st.error(
            "La prediction a echoue. Verifiez les entrees et la compatibilite des artefacts. "
            f"Detail: {exc}"
        )

st.markdown("---")
st.markdown('<div class="section-title">Import CSV et calcul automatique des features derives</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='subtle'>Vous pouvez importer soit des donnees deja agregees par client, soit des transactions brutes. Le calcul des features derives reproduit les formules du notebook.</div>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Telecharger modele CSV (donnees agregees)",
        data=_prepared_template_csv(),
        file_name="modele_agrege.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "Telecharger modele CSV (transactions brutes)",
        data=_transactions_template_csv(),
        file_name="modele_transactions.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Importer un fichier CSV client", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Impossible de lire le CSV: {exc}")
        uploaded_df = None

    if uploaded_df is not None:
        if uploaded_df.empty:
            st.warning("Le fichier CSV est vide.")
        else:
            st.caption(f"Fichier charge: {len(uploaded_df):,} ligne(s), {len(uploaded_df.columns)} colonne(s).")

            mode_choice = st.selectbox(
                "Type de donnees CSV",
                options=[
                    "Auto-detection",
                    "Donnees clients deja agregees",
                    "Transactions brutes (calcul des features)",
                ],
            )

            detected_mode = _detect_uploaded_schema(uploaded_df)
            if mode_choice == "Donnees clients deja agregees":
                selected_mode = "prepared"
            elif mode_choice == "Transactions brutes (calcul des features)":
                selected_mode = "transactions"
            else:
                selected_mode = detected_mode

            inference_input_df: pd.DataFrame | None = None
            derivation_issues: list[str] = []

            if selected_mode == "prepared":
                missing_prepared = _missing_columns(uploaded_df, PREPARED_REQUIRED_COLUMNS)
                if missing_prepared:
                    st.error("Colonnes manquantes pour mode agrege: " + ", ".join(missing_prepared))
                else:
                    inference_input_df = uploaded_df.copy()
                    st.success("Mode agrege detecte. Le fichier est pret pour le scoring.")

            elif selected_mode == "transactions":
                missing_tx = _missing_columns(uploaded_df, TRANSACTION_REQUIRED_COLUMNS)
                if missing_tx:
                    st.error("Colonnes manquantes pour mode transaction brut: " + ", ".join(missing_tx))
                else:
                    try:
                        with st.spinner("Calcul des features derives a partir des transactions..."):
                            inference_input_df, derivation_issues = _derive_features_from_transactions(
                                uploaded_df,
                                reference_date,
                                channel_lookup,
                            )
                        st.success(
                            f"Features derives calcules pour {len(inference_input_df):,} client(s)."
                        )
                        st.dataframe(inference_input_df.head(20), use_container_width=True)
                    except Exception as exc:
                        st.error(f"Echec du calcul des features derives: {exc}")

            else:
                missing_prepared = _missing_columns(uploaded_df, PREPARED_REQUIRED_COLUMNS)
                missing_tx = _missing_columns(uploaded_df, TRANSACTION_REQUIRED_COLUMNS)
                st.error("Schema CSV non reconnu.")
                st.info(
                    "Colonnes attendues (agrege): "
                    + ", ".join(sorted(PREPARED_REQUIRED_COLUMNS))
                    + "\n\nColonnes attendues (transactions): "
                    + ", ".join(sorted(TRANSACTION_REQUIRED_COLUMNS))
                )
                if missing_prepared:
                    st.caption("Colonnes manquantes mode agrege: " + ", ".join(missing_prepared))
                if missing_tx:
                    st.caption("Colonnes manquantes mode transactions: " + ", ".join(missing_tx))

            if inference_input_df is not None:
                if derivation_issues:
                    st.warning("Infos de nettoyage derive: " + " | ".join(derivation_issues))

                run_csv = st.button("Lancer le scoring du CSV")
                if run_csv:
                    try:
                        with st.spinner("Prediction en cours sur le fichier CSV..."):
                            pred_df, batch_summary, failed_rows = _run_batch_predictions(
                                inference_input_df,
                                artifacts,
                                fixed_threshold,
                                known_channels,
                                channel_lookup,
                            )

                        st.success(
                            f"Scoring termine: {batch_summary['rows_out']} ligne(s) predite(s) sur {batch_summary['rows_in']}."
                        )

                        b1, b2, b3 = st.columns(3)
                        b1.metric("Risque moyen", f"{pred_df['churn_probability'].mean():.1%}")
                        b2.metric(
                            "Clients a risque eleve (>=70%)",
                            int((pred_df["churn_probability"] >= 0.7).sum()),
                        )
                        b3.metric("Segments identifies", int(pred_df["segment"].nunique()))

                        if batch_summary["parse_issue_rows"] > 0:
                            st.warning(
                                f"{batch_summary['parse_issue_rows']} ligne(s) ont recu un nettoyage automatique des valeurs."
                            )

                        if batch_summary["unknown_channel_rows"] > 0:
                            st.warning(
                                f"{batch_summary['unknown_channel_rows']} ligne(s) avec canal hors apprentissage: encodage baseline applique."
                            )

                        if batch_summary["rows_failed"] > 0:
                            st.error(f"{batch_summary['rows_failed']} ligne(s) en echec pendant la prediction.")
                            st.write("Exemples d erreurs:")
                            for msg in failed_rows[:10]:
                                st.write(f"- {msg}")

                        st.dataframe(pred_df, use_container_width=True)

                        out_csv = pred_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Telecharger les resultats CSV",
                            data=out_csv,
                            file_name="predictions_clients.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Erreur pendant le scoring CSV: {exc}")
