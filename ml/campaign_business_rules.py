from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CAMPAIGN_INPUT_COLUMNS = ("reach", "impressions", "frequency", "result", "views", "price")


def _clean_numeric_series(series: pd.Series | list[float] | np.ndarray | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned.astype("float64")


def _quantile(series: pd.Series, q: float, default: float) -> float:
    cleaned = _clean_numeric_series(series)
    if cleaned.empty:
        return float(default)
    return float(cleaned.quantile(q))


def _metric_profile(
    series: pd.Series | None,
    *,
    default_median: float,
    default_p75: float,
    default_p90: float,
    default_max: float,
) -> dict[str, float]:
    cleaned = _clean_numeric_series(series)
    if cleaned.empty:
        return {
            "median": float(default_median),
            "p75": float(default_p75),
            "p90": float(default_p90),
            "max": float(default_max),
        }
    return {
        "median": float(cleaned.median()),
        "p75": float(cleaned.quantile(0.75)),
        "p90": float(cleaned.quantile(0.90)),
        "max": float(cleaned.max()),
    }


def default_business_context() -> dict[str, Any]:
    return {
        "rules_version": "campaign_business_v2",
        "metrics": {
            "reach": {"median": 1073.0, "p75": 1695.0, "p90": 2182.0, "max": 4694.0},
            "impressions": {"median": 1109.0, "p75": 1858.0, "p90": 2262.0, "max": 5649.0},
            "frequency": {"median": 1.03, "p75": 1.05, "p90": 1.12, "max": 1.27},
            "result": {"median": 1.0, "p75": 2.0, "p90": 3.0, "max": 10.0},
            "views": {"median": 43.48, "p75": 53.85, "p90": 68.18, "max": 100.0},
            "price": {"median": 1.02, "p75": 1.86, "p90": 18.0, "max": 300.0},
        },
        "efficiency": {
            "cost_per_result": {"p75": 0.95, "p90": 1.63},
            "cost_per_mille": {"p75": 0.61, "p90": 0.92},
        },
        "frequency_rules": {"ideal_min": 0.90, "ideal_max": 1.15, "warning_max": 1.25},
        "success_score_threshold": 3.8,
    }


def _context_metric(context: dict[str, Any] | None, metric_name: str, stat_name: str, default: float) -> float:
    metrics = context.get("metrics") if isinstance(context, dict) else None
    metric = metrics.get(metric_name) if isinstance(metrics, dict) else None
    value = metric.get(stat_name) if isinstance(metric, dict) else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _context_efficiency(context: dict[str, Any] | None, metric_name: str, stat_name: str, default: float) -> float:
    efficiency = context.get("efficiency") if isinstance(context, dict) else None
    metric = efficiency.get(metric_name) if isinstance(efficiency, dict) else None
    value = metric.get(stat_name) if isinstance(metric, dict) else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _context_frequency_rule(context: dict[str, Any] | None, rule_name: str, default: float) -> float:
    rules = context.get("frequency_rules") if isinstance(context, dict) else None
    value = rules.get(rule_name) if isinstance(rules, dict) else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def build_business_context(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return default_business_context()

    frame = df.copy()
    for column_name in CAMPAIGN_INPUT_COLUMNS:
        if column_name not in frame.columns:
            frame[column_name] = 0.0
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0)

    delivered_mask = (frame["reach"] > 0) & (frame["impressions"] > 0)
    positive_views = frame.loc[frame["views"] > 0, "views"]
    positive_results = frame.loc[frame["result"] > 0, "result"]
    positive_cost_per_result = pd.Series(
        np.where(frame["result"] > 0, frame["price"] / np.maximum(frame["result"], 1e-6), np.nan)
    )
    positive_cost_per_mille = pd.Series(
        np.where(frame["impressions"] > 0, frame["price"] / np.maximum(frame["impressions"] / 1000.0, 1e-6), np.nan)
    )
    delivered_frequency = frame.loc[delivered_mask, "frequency"]

    ideal_min = max(0.85, _quantile(delivered_frequency, 0.25, 0.90) - 0.10)
    ideal_max = min(1.25, _quantile(delivered_frequency, 0.75, 1.15) + 0.10)
    warning_max = max(1.25, _quantile(delivered_frequency, 0.95, 1.25) + 0.05)

    return {
        "rules_version": "campaign_business_v2",
        "metrics": {
            "reach": _metric_profile(frame["reach"], default_median=1073.0, default_p75=1695.0, default_p90=2182.0, default_max=4694.0),
            "impressions": _metric_profile(frame["impressions"], default_median=1109.0, default_p75=1858.0, default_p90=2262.0, default_max=5649.0),
            "frequency": _metric_profile(frame["frequency"], default_median=1.03, default_p75=1.05, default_p90=1.12, default_max=1.27),
            "result": _metric_profile(positive_results, default_median=1.0, default_p75=2.0, default_p90=3.0, default_max=10.0),
            "views": _metric_profile(positive_views, default_median=43.48, default_p75=53.85, default_p90=68.18, default_max=100.0),
            "price": _metric_profile(frame["price"], default_median=1.02, default_p75=1.86, default_p90=18.0, default_max=300.0),
        },
        "efficiency": {
            "cost_per_result": {
                "p75": _quantile(positive_cost_per_result, 0.75, 0.95),
                "p90": _quantile(positive_cost_per_result, 0.90, 1.63),
            },
            "cost_per_mille": {
                "p75": _quantile(positive_cost_per_mille, 0.75, 0.61),
                "p90": _quantile(positive_cost_per_mille, 0.90, 0.92),
            },
        },
        "frequency_rules": {
            "ideal_min": float(ideal_min),
            "ideal_max": float(max(ideal_min + 0.05, ideal_max)),
            "warning_max": float(max(ideal_max + 0.05, warning_max)),
        },
        "success_score_threshold": 3.8,
    }


def validate_campaign_inputs(
    raw_inputs: dict[str, Any],
    business_context: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[str], list[str]]:
    context = business_context or default_business_context()
    errors: list[str] = []
    warnings: list[str] = []
    clean_inputs: dict[str, float] = {}

    labels = {
        "reach": "reach",
        "impressions": "impressions",
        "frequency": "frequency",
        "result": "result",
        "views": "views",
        "price": "price",
    }

    for field_name in CAMPAIGN_INPUT_COLUMNS:
        raw_value = raw_inputs.get(field_name, 0.0)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"Le champ {labels[field_name]} doit etre numerique.")
            continue
        if not math.isfinite(numeric_value):
            errors.append(f"Le champ {labels[field_name]} doit etre une valeur finie.")
            continue
        clean_inputs[field_name] = numeric_value

    if errors:
        return clean_inputs, errors, warnings

    for field_name in CAMPAIGN_INPUT_COLUMNS:
        if clean_inputs[field_name] < 0:
            errors.append(f"Le champ {labels[field_name]} doit etre positif ou nul.")

    reach = clean_inputs["reach"]
    impressions = clean_inputs["impressions"]
    frequency = clean_inputs["frequency"]
    result = clean_inputs["result"]
    views = clean_inputs["views"]
    price = clean_inputs["price"]

    if impressions < reach:
        errors.append("Les impressions doivent etre superieures ou egales au reach.")

    if reach == 0 and impressions > 0:
        errors.append("Le reach ne peut pas etre nul si des impressions sont declarees.")

    if impressions == 0 and result > 0:
        errors.append("Le champ result ne peut pas etre positif si impressions vaut 0.")

    if impressions == 0 and views > 0:
        errors.append("Le champ views ne peut pas etre positif si impressions vaut 0.")

    if reach > 0 and result > reach:
        errors.append("Le champ result ne peut pas depasser le reach.")

    if impressions > 0 and result > impressions:
        errors.append("Le champ result ne peut pas depasser les impressions.")

    if errors:
        return clean_inputs, errors, warnings

    expected_frequency = impressions / reach if reach > 0 else 0.0
    if reach == 0 and impressions == 0 and frequency > 0:
        clean_inputs["frequency"] = 0.0
        warnings.append("La frequence a ete ramenee a 0 car aucune diffusion n'est declaree.")
    elif reach > 0 and impressions > 0:
        tolerance = max(0.15, expected_frequency * 0.25)
        if frequency <= 0:
            clean_inputs["frequency"] = float(expected_frequency)
            warnings.append("La frequence a ete recalculee a partir du reach et des impressions.")
        elif abs(frequency - expected_frequency) > tolerance:
            clean_inputs["frequency"] = float(expected_frequency)
            warnings.append("La frequence a ete ajustee pour rester coherente avec reach et impressions.")

    observed_views_cap = _context_metric(context, "views", "max", 100.0)
    views_cap = max(100.0, observed_views_cap)
    if clean_inputs["views"] > views_cap:
        clean_inputs["views"] = float(views_cap)
        warnings.append(
            f"Le champ views depasse la plage observee; il a ete plafonne a {views_cap:.0f} pour le scoring."
        )

    if price > _context_metric(context, "price", "p90", 18.0) * 5:
        warnings.append("Le budget saisi est tres eleve par rapport a l'historique et sera fortement penalise.")

    return clean_inputs, errors, warnings


def score_campaign_frame(
    df: pd.DataFrame,
    business_context: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, pd.Series], dict[str, pd.Series], dict[str, Any]]:
    defaults = default_business_context()
    context = business_context or defaults
    frame = df.copy()
    for column_name in CAMPAIGN_INPUT_COLUMNS:
        if column_name not in frame.columns:
            frame[column_name] = 0.0
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0)

    reach_p50 = _context_metric(context, "reach", "median", defaults["metrics"]["reach"]["median"])
    reach_p75 = _context_metric(context, "reach", "p75", defaults["metrics"]["reach"]["p75"])
    impressions_p50 = _context_metric(context, "impressions", "median", defaults["metrics"]["impressions"]["median"])
    impressions_p75 = _context_metric(context, "impressions", "p75", defaults["metrics"]["impressions"]["p75"])
    result_p50 = _context_metric(context, "result", "median", defaults["metrics"]["result"]["median"])
    result_p75 = _context_metric(context, "result", "p75", defaults["metrics"]["result"]["p75"])
    views_p50 = _context_metric(context, "views", "median", defaults["metrics"]["views"]["median"])
    views_p75 = _context_metric(context, "views", "p75", defaults["metrics"]["views"]["p75"])
    cpr_p75 = _context_efficiency(context, "cost_per_result", "p75", defaults["efficiency"]["cost_per_result"]["p75"])
    cpr_p90 = _context_efficiency(context, "cost_per_result", "p90", defaults["efficiency"]["cost_per_result"]["p90"])
    cpm_p75 = _context_efficiency(context, "cost_per_mille", "p75", defaults["efficiency"]["cost_per_mille"]["p75"])
    cpm_p90 = _context_efficiency(context, "cost_per_mille", "p90", defaults["efficiency"]["cost_per_mille"]["p90"])
    ideal_min = _context_frequency_rule(context, "ideal_min", defaults["frequency_rules"]["ideal_min"])
    ideal_max = _context_frequency_rule(context, "ideal_max", defaults["frequency_rules"]["ideal_max"])
    warning_max = _context_frequency_rule(context, "warning_max", defaults["frequency_rules"]["warning_max"])

    expected_frequency = pd.Series(
        np.where(frame["reach"] > 0, frame["impressions"] / np.maximum(frame["reach"], 1e-6), 0.0),
        index=frame.index,
        dtype="float64",
    )
    cost_per_result = pd.Series(
        np.where(frame["result"] > 0, frame["price"] / np.maximum(frame["result"], 1e-6), np.nan),
        index=frame.index,
        dtype="float64",
    )
    cost_per_mille = pd.Series(
        np.where(frame["impressions"] > 0, frame["price"] / np.maximum(frame["impressions"] / 1000.0, 1e-6), np.nan),
        index=frame.index,
        dtype="float64",
    )

    flags: dict[str, pd.Series] = {
        "campaign_delivered": (frame["reach"] > 0) & (frame["impressions"] > 0),
        "reach_above_median": frame["reach"] >= reach_p50,
        "reach_top_quartile": frame["reach"] >= reach_p75,
        "impressions_above_median": frame["impressions"] >= impressions_p50,
        "impressions_top_quartile": frame["impressions"] >= impressions_p75,
        "positive_response": frame["result"] > 0,
        "response_above_median": frame["result"] >= result_p50,
        "response_top_quartile": frame["result"] >= result_p75,
        "views_above_median": frame["views"] >= views_p50,
        "views_top_quartile": frame["views"] >= views_p75,
        "healthy_frequency": (frame["frequency"] >= ideal_min) & (frame["frequency"] <= ideal_max),
        "frequency_saturation_risk": (frame["frequency"] > warning_max) & (frame["impressions"] > 0),
        "frequency_too_low": (frame["frequency"] < max(0.5, ideal_min - 0.25)) & (frame["impressions"] > 0),
        "budget_without_delivery": (frame["price"] > 0) & ~((frame["reach"] > 0) & (frame["impressions"] > 0)),
        "delivery_without_conversion": (frame["result"] == 0) & (frame["impressions"] >= impressions_p50),
        "no_view_signal": (frame["views"] == 0) & (frame["impressions"] > 0),
        "efficient_cost_per_result": cost_per_result <= cpr_p75,
        "expensive_cost_per_result": cost_per_result >= cpr_p90,
        "efficient_cost_per_mille": cost_per_mille <= cpm_p75,
        "high_cost_per_mille": cost_per_mille >= cpm_p90,
        "limited_sample": (frame["reach"] < max(50.0, reach_p50 * 0.10)) & (frame["impressions"] < max(50.0, impressions_p50 * 0.10)),
    }

    score = pd.Series(0.0, index=frame.index, dtype="float64")
    score += flags["campaign_delivered"].astype(float) * 1.0
    score += flags["reach_above_median"].astype(float) * 0.6
    score += flags["impressions_above_median"].astype(float) * 0.6
    score += flags["reach_top_quartile"].astype(float) * 0.4
    score += flags["impressions_top_quartile"].astype(float) * 0.4
    score += flags["positive_response"].astype(float) * 0.7
    score += flags["response_above_median"].astype(float) * 0.5
    score += flags["response_top_quartile"].astype(float) * 0.5
    score += flags["views_above_median"].astype(float) * 0.8
    score += flags["views_top_quartile"].astype(float) * 0.6
    score += flags["healthy_frequency"].astype(float) * 0.6
    score -= flags["frequency_saturation_risk"].astype(float) * 1.0
    score -= flags["frequency_too_low"].astype(float) * 0.4
    score -= flags["budget_without_delivery"].astype(float) * 2.0
    score -= flags["delivery_without_conversion"].astype(float) * 1.2
    score -= flags["no_view_signal"].astype(float) * 0.8
    score += flags["efficient_cost_per_result"].fillna(False).astype(float) * 0.8
    score -= flags["expensive_cost_per_result"].fillna(False).astype(float) * 0.9
    score += flags["efficient_cost_per_mille"].fillna(False).astype(float) * 0.5
    score -= flags["high_cost_per_mille"].fillna(False).astype(float) * 0.7
    score -= flags["limited_sample"].astype(float) * 0.3

    derived = {
        "expected_frequency": expected_frequency,
        "cost_per_result": cost_per_result,
        "cost_per_mille": cost_per_mille,
    }
    return score, flags, derived, context


def build_business_target(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any], float, str]:
    business_context = build_business_context(df)
    score, _, _, _ = score_campaign_frame(df, business_context)
    threshold = max(float(score.quantile(0.62)), 3.0)
    target = (score >= threshold).astype(int)

    if target.nunique(dropna=True) < 2:
        threshold = max(float(score.median()), 2.5)
        target = (score >= threshold).astype(int)

    if target.nunique(dropna=True) < 2:
        threshold = float(score.quantile(0.55))
        target = (score >= threshold).astype(int)

    business_context["success_score_threshold"] = float(threshold)
    return target, business_context, float(threshold), "business_score_threshold_v2"


def evaluate_campaign_prediction(
    raw_inputs: dict[str, Any],
    business_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = business_context or default_business_context()
    clean_inputs, errors, warnings = validate_campaign_inputs(raw_inputs, context)
    if errors:
        raise ValueError(" | ".join(errors))

    frame = pd.DataFrame([clean_inputs], columns=list(CAMPAIGN_INPUT_COLUMNS))
    score, flags, derived, scoring_context = score_campaign_frame(frame, context)
    score_value = float(score.iloc[0])
    success_threshold = float(
        scoring_context.get("success_score_threshold", default_business_context()["success_score_threshold"])
    )
    margin = score_value - success_threshold
    probability = 1.0 / (1.0 + math.exp(-(margin / 0.9)))

    if bool(flags["limited_sample"].iloc[0]):
        probability = 0.5 + (probability - 0.5) * 0.7
        warnings.append("Le volume est faible; la confiance est volontairement moderee.")

    if bool(flags["budget_without_delivery"].iloc[0]):
        probability = min(probability, 0.08)
    elif bool(flags["delivery_without_conversion"].iloc[0]):
        probability = min(probability, 0.32)

    probability = max(0.02, min(0.98, float(probability)))
    predicted_success = int(score_value >= success_threshold and probability >= 0.55)

    highlights: list[str] = []
    if bool(flags["campaign_delivered"].iloc[0]):
        highlights.append("La campagne dispose d'une diffusion reelle sur le marche.")
    if bool(flags["reach_top_quartile"].iloc[0]) or bool(flags["impressions_top_quartile"].iloc[0]):
        highlights.append("Le volume de diffusion se situe dans la partie haute de l'historique.")
    elif bool(flags["reach_above_median"].iloc[0]) or bool(flags["impressions_above_median"].iloc[0]):
        highlights.append("La diffusion depasse le niveau median observe sur les campagnes precedentes.")
    if bool(flags["response_top_quartile"].iloc[0]):
        highlights.append("Le niveau de resultats atteint un palier fort pour ce type de campagne.")
    elif bool(flags["positive_response"].iloc[0]):
        highlights.append("La campagne genere deja un signal de resultat exploitable.")
    if bool(flags["views_top_quartile"].iloc[0]):
        highlights.append("Le taux de vues se place parmi les meilleurs niveaux historiques.")
    elif bool(flags["views_above_median"].iloc[0]):
        highlights.append("Le signal de vues reste au-dessus du niveau median.")
    if bool(flags["healthy_frequency"].iloc[0]):
        highlights.append("La frequence reste saine, sans pression excessive sur l'audience.")
    if bool(flags["efficient_cost_per_result"].iloc[0]):
        highlights.append("Le cout par resultat reste competitif par rapport a l'historique.")
    if bool(flags["efficient_cost_per_mille"].iloc[0]):
        highlights.append("Le cout de diffusion reste efficace pour le volume obtenu.")

    if bool(flags["frequency_saturation_risk"].iloc[0]):
        warnings.append("La frequence indique un risque de saturation de l'audience.")
    if bool(flags["frequency_too_low"].iloc[0]):
        warnings.append("La frequence reste faible et peut limiter la memorisation de la campagne.")
    if bool(flags["budget_without_delivery"].iloc[0]):
        warnings.append("Un budget est engage sans diffusion effective, ce qui rend le succes tres improbable.")
    if bool(flags["delivery_without_conversion"].iloc[0]):
        warnings.append("La campagne diffuse correctement mais ne convertit pas encore.")
    if bool(flags["no_view_signal"].iloc[0]):
        warnings.append("Aucun signal de vues n'est remonte malgre la diffusion.")
    if bool(flags["expensive_cost_per_result"].iloc[0]):
        warnings.append("Le cout par resultat est eleve et degrade la rentabilite attendue.")
    if bool(flags["high_cost_per_mille"].iloc[0]):
        warnings.append("Le cout pour mille impressions est au-dessus de la norme historique.")

    derived_payload = {
        "expected_frequency": float(derived["expected_frequency"].iloc[0]),
        "cost_per_result": None
        if pd.isna(derived["cost_per_result"].iloc[0])
        else float(derived["cost_per_result"].iloc[0]),
        "cost_per_mille": None
        if pd.isna(derived["cost_per_mille"].iloc[0])
        else float(derived["cost_per_mille"].iloc[0]),
        "success_threshold": success_threshold,
    }

    return {
        "predicted_success": predicted_success,
        "probability": probability,
        "decision": "Success" if predicted_success == 1 else "Not Success",
        "strategy": "business_scorecard_v2",
        "score": round(score_value, 3),
        "inputs": {field_name: float(clean_inputs[field_name]) for field_name in CAMPAIGN_INPUT_COLUMNS},
        "highlights": highlights,
        "warnings": warnings,
        "derived_metrics": derived_payload,
    }
