from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from datetime import date as calendar_date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from flask import Flask, render_template, request, send_file


def _normalize_base_urls(raw_values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in raw_values:
        base = str(value).strip().rstrip('/')
        if not base or base in normalized:
            continue
        normalized.append(base)
    return normalized


def _parse_configured_base_urls() -> list[str]:
    configured = os.getenv('FASTAPI_BASE_URLS', '')
    single = os.getenv('FASTAPI_BASE_URL', 'http://127.0.0.1:8000')
    return _normalize_base_urls([single] + configured.split(','))


def _parse_base_urls() -> list[str]:
    return _normalize_base_urls(CONFIGURED_BASE_URLS + ['http://127.0.0.1:8000', 'http://localhost:8000'])


CONFIGURED_BASE_URLS = _parse_configured_base_urls()
BASE_URLS = _parse_base_urls()
API_TIMEOUT_SECONDS = float(os.getenv('FASTAPI_TIMEOUT_SECONDS', '15'))
FASTAPI_AUTO_START = os.getenv('FASTAPI_AUTO_START', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
FASTAPI_START_TIMEOUT_SECONDS = float(os.getenv('FASTAPI_START_TIMEOUT_SECONDS', '20'))
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', '3000'))
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = Path(__file__).resolve().parent.parent / 'logo.png'
TEMPLATE_LOGO_PATH = Path(__file__).resolve().parent / 'templates' / 'logo.png'
CAMPAIGN_SUMMARY_PATH = Path(__file__).resolve().parent / 'mlops_artifacts' / 'campaign_summary.json'
_LAST_WORKING_BASE = BASE_URLS[0]
_FASTAPI_PROCESS: subprocess.Popen[Any] | None = None

app = Flask(__name__, template_folder='templates')


def _default_supplier_inputs() -> dict[str, str]:
    return {
        'quantity': '12',
        'unit_price': '18.5',
        'total_ht': '222',
        'total_ttc': '265',
        'governorate': 'Tunis',
        'city': 'Tunis',
    }


def _default_promote_inputs() -> dict[str, str]:
    return {
        'date': calendar_date.today().isoformat(),
        'last_day_revenue': '8400',
        'last_week_revenue': '7900',
        'two_weeks_revenue': '7600',
        'rolling_7d_revenue': '8050',
        'planned_discount_pct': '18',
        'ramadan_days': '0',
        'eid_window_days': '0',
    }


def _default_campaign_inputs() -> dict[str, str]:
    return {
        'reach': '0',
        'impressions': '0',
        'frequency': '0',
        'result': '0',
        'views': '0',
        'price': '0',
    }


def _parse_source_page(form_data: Any, default: str = 'home') -> str:
    source_page = str(form_data.get('source_page', default)).strip().lower()
    return source_page if source_page in {'home', 'promote'} else default


def _build_base_candidates(base_candidates: list[str] | None = None) -> list[str]:
    candidates: list[str] = []
    for base in ([base_candidates or []] + [[_LAST_WORKING_BASE], BASE_URLS]):
        for value in base:
            url = str(value).strip().rstrip('/')
            if not url or url in candidates:
                continue
            candidates.append(url)
    return candidates


def _is_local_fastapi_base(base_url: str) -> bool:
    parsed = urlparse(str(base_url).strip())
    host = (parsed.hostname or '').lower()
    return host in {'127.0.0.1', 'localhost'}


def _autostart_base_candidates(base_candidates: list[str] | None = None) -> list[str]:
    if base_candidates:
        source = base_candidates
    else:
        source = CONFIGURED_BASE_URLS
    return [base for base in _normalize_base_urls(source) if _is_local_fastapi_base(base)]


def _stop_spawned_fastapi() -> None:
    global _FASTAPI_PROCESS
    proc = _FASTAPI_PROCESS
    if proc is None or proc.poll() is not None:
        _FASTAPI_PROCESS = None
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _FASTAPI_PROCESS = None


atexit.register(_stop_spawned_fastapi)


def _ensure_local_fastapi_running(base_candidates: list[str] | None = None) -> str | None:
    global _FASTAPI_PROCESS

    if not FASTAPI_AUTO_START:
        return 'demarrage automatique desactive'

    local_bases = _autostart_base_candidates(base_candidates)
    if not local_bases:
        return None

    target_base = local_bases[0]
    parsed = urlparse(target_base)
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or 8000

    if _FASTAPI_PROCESS is not None and _FASTAPI_PROCESS.poll() is None:
        process = _FASTAPI_PROCESS
    else:
        creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        command = [sys.executable, '-m', 'uvicorn', 'main:app', '--host', host, '--port', str(port)]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as exc:
            _FASTAPI_PROCESS = None
            return f'demarrage automatique impossible: {exc}'
        _FASTAPI_PROCESS = process

    deadline = time.time() + FASTAPI_START_TIMEOUT_SECONDS
    health_url = f'{target_base}/health'
    while time.time() < deadline:
        if process.poll() is not None:
            _FASTAPI_PROCESS = None
            return f'processus FastAPI termine avec le code {process.returncode}'
        try:
            request_object = Request(health_url, headers={'Accept': 'application/json'}, method='GET')
            with urlopen(request_object, timeout=min(API_TIMEOUT_SECONDS, 2.0)) as response:
                raw = response.read().decode('utf-8')
                payload = json.loads(raw) if raw else {}
                if isinstance(payload, dict) and payload.get('status') == 'ok':
                    return target_base
        except Exception:
            time.sleep(0.5)

    return f'timeout de demarrage sur {health_url}'


def _http_error_message(exc: HTTPError) -> str:
    try:
        raw = exc.read().decode('utf-8')
        if not raw:
            return f'{exc.code} {exc.reason}'
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get('detail'):
            return f'{exc.code} {parsed.get("detail")}'
        return f'{exc.code} {raw}'
    except Exception:
        return f'{exc.code} {exc.reason}'


def _api_request(
    path: str,
    method: str = 'GET',
    payload: dict[str, Any] | None = None,
    base_candidates: list[str] | None = None,
    allow_autostart: bool = True,
) -> tuple[dict[str, Any], str]:
    global _LAST_WORKING_BASE

    body = None
    headers = {'Accept': 'application/json'}
    if payload is not None:
        headers['Content-Type'] = 'application/json'
        body = json.dumps(payload).encode('utf-8')

    attempted: list[str] = []
    for base_url in _build_base_candidates(base_candidates):
        request_url = f'{base_url}{path}'
        request_object = Request(request_url, data=body, headers=headers, method=method)
        try:
            with urlopen(request_object, timeout=API_TIMEOUT_SECONDS) as response:
                raw = response.read().decode('utf-8')
                data = json.loads(raw) if raw else {}
                _LAST_WORKING_BASE = base_url
                return data, base_url
        except HTTPError as exc:
            raise RuntimeError(f'FastAPI a repondu en erreur sur {request_url}: {_http_error_message(exc)}') from exc
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            attempted.append(f'{request_url} ({exc})')

    if allow_autostart:
        local_autostart_bases = _autostart_base_candidates(base_candidates)
        autostart_result = _ensure_local_fastapi_running(base_candidates)
        if autostart_result and autostart_result in local_autostart_bases:
            return _api_request(
                path,
                method=method,
                payload=payload,
                base_candidates=base_candidates,
                allow_autostart=False,
            )
        if autostart_result:
            attempted.append(f'auto-start ({autostart_result})')

    joined_attempts = ' | '.join(attempted) if attempted else 'aucune tentative'
    raise ConnectionError(f'Impossible de contacter l\'API FastAPI. Tentatives: {joined_attempts}')


def _dashboard_api_context() -> dict[str, Any]:
    health = None
    models = None
    api_base = _LAST_WORKING_BASE
    api_error = None

    try:
        health, api_base = _api_request('/health')
        models, _ = _api_request('/models', base_candidates=[api_base])
    except Exception as exc:
        api_error = str(exc)

    return {
        'api_base': api_base,
        'health': health,
        'models': models,
        'api_error': api_error,
    }


def _campaign_summary_from_sources(api_base: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    summary = None
    campaign_error = None

    if api_base:
        try:
            payload, _ = _api_request('/predict/campaign', base_candidates=[api_base])
            if isinstance(payload, dict):
                summary = payload.get('summary')
        except Exception as exc:
            campaign_error = f'Impossible de lire le resume de campagne via l\'API: {exc}'

    if summary is None and CAMPAIGN_SUMMARY_PATH.exists():
        try:
            summary = json.loads(CAMPAIGN_SUMMARY_PATH.read_text(encoding='utf-8'))
        except Exception:
            summary = None

    if summary is None and campaign_error is None:
        campaign_error = f'Resume MLOps introuvable: {CAMPAIGN_SUMMARY_PATH}'

    return summary, campaign_error


def _base_page_context(local_errors: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    api_ctx = _dashboard_api_context()
    errors = [msg for msg in (local_errors or []) if msg]
    if api_ctx.get('api_error'):
        errors.append(str(api_ctx.get('api_error')))
    return api_ctx, errors


def _promote_supporting_data(api_base: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    result = None
    train_result = None
    errors: list[str] = []
    if not api_base:
        return result, train_result, errors

    try:
        result, _ = _api_request('/predict/promote', base_candidates=[api_base])
    except Exception as exc:
        errors.append(f'Impossible de charger le resume Promote: {exc}')

    try:
        payload, _ = _api_request('/train/promote/results', base_candidates=[api_base])
        if isinstance(payload, dict) and payload.get('available') and isinstance(payload.get('results'), dict):
            train_result = payload.get('results')
    except Exception as exc:
        errors.append(f'Impossible de charger le suivi du modele Promote: {exc}')

    return result, train_result, errors


def _parse_promote_form(form_data: Any) -> tuple[dict[str, str], dict[str, Any] | None, dict[str, str]]:
    values = _default_promote_inputs()
    field_errors: dict[str, str] = {}
    payload: dict[str, Any] = {}
    for key in values:
        values[key] = str(form_data.get(key, values[key])).strip()

    raw_date = values['date']
    if not raw_date:
        field_errors['date'] = 'La date est obligatoire.'
    else:
        try:
            payload['date'] = calendar_date.fromisoformat(raw_date).isoformat()
        except ValueError:
            field_errors['date'] = 'La date doit respecter le format YYYY-MM-DD.'

    def _parse_float(name: str, label: str, minimum: float = 0.0, maximum: float | None = None) -> None:
        raw_value = values[name]
        if not raw_value:
            field_errors[name] = f'{label} est obligatoire.'
            return
        try:
            numeric_value = float(raw_value)
        except ValueError:
            field_errors[name] = f'{label} doit etre numerique.'
            return
        if numeric_value < minimum:
            field_errors[name] = f'{label} doit etre >= {minimum:g}.'
            return
        if maximum is not None and numeric_value > maximum:
            field_errors[name] = f'{label} doit etre <= {maximum:g}.'
            return
        payload[name] = numeric_value

    def _parse_int(name: str, label: str, minimum: int = 0, maximum: int | None = None) -> None:
        raw_value = values[name]
        if not raw_value:
            field_errors[name] = f'{label} est obligatoire.'
            return
        try:
            numeric_value = int(raw_value)
        except ValueError:
            field_errors[name] = f'{label} doit etre un entier.'
            return
        if numeric_value < minimum:
            field_errors[name] = f'{label} doit etre >= {minimum}.'
            return
        if maximum is not None and numeric_value > maximum:
            field_errors[name] = f'{label} doit etre <= {maximum}.'
            return
        payload[name] = numeric_value

    _parse_float('last_day_revenue', 'Le CA du dernier jour')
    _parse_float('last_week_revenue', 'Le CA de la semaine precedente')
    _parse_float('two_weeks_revenue', 'Le CA de la deuxieme semaine precedente')
    _parse_float('rolling_7d_revenue', 'La moyenne mobile 7 jours')
    _parse_float('planned_discount_pct', 'La remise planifiee', minimum=0.0, maximum=90.0)
    _parse_int('ramadan_days', 'Les jours Ramadan restants', minimum=0, maximum=30)
    _parse_int('eid_window_days', 'Les jours de fenetre Eid', minimum=0, maximum=10)

    if field_errors:
        return values, None, field_errors
    return values, payload, field_errors


def _parse_campaign_form(form_data: Any) -> tuple[dict[str, str], dict[str, Any] | None, dict[str, str]]:
    values = _default_campaign_inputs()
    field_errors: dict[str, str] = {}
    payload: dict[str, Any] = {}

    for key in values:
        values[key] = str(form_data.get(key, values[key])).strip()

    def _parse_float(name: str, label: str) -> None:
        raw_value = values[name]
        if not raw_value:
            field_errors[name] = f'{label} est obligatoire.'
            return
        try:
            numeric_value = float(raw_value)
        except ValueError:
            field_errors[name] = f'{label} doit etre numerique.'
            return
        if numeric_value < 0:
            field_errors[name] = f'{label} doit etre positif ou nul.'
            return
        payload[name] = numeric_value

    _parse_float('reach', 'Reach')
    _parse_float('impressions', 'Impressions')
    _parse_float('frequency', 'Frequency')
    _parse_float('result', 'Result')
    _parse_float('views', 'Views')
    _parse_float('price', 'Price')

    if field_errors:
        return values, None, field_errors
    return values, payload, field_errors


def _render_home(*, result: dict[str, Any] | None = None, local_errors: list[str] | None = None) -> str:
    api_ctx, errors = _base_page_context(local_errors)
    campaign_summary, campaign_error = _campaign_summary_from_sources(api_ctx.get('api_base'))
    if campaign_error:
        errors.append(campaign_error)

    return render_template(
        'index.html',
        api_base=api_ctx.get('api_base'),
        health=api_ctx.get('health'),
        models=api_ctx.get('models'),
        result=result,
        errors=errors,
        active_page='home',
        campaign_summary=campaign_summary,
    )


def _render_promote_page(
    *,
    result: dict[str, Any] | None = None,
    form_result: dict[str, Any] | None = None,
    train_result: dict[str, Any] | None = None,
    input_values: dict[str, str] | None = None,
    field_errors: dict[str, str] | None = None,
    local_errors: list[str] | None = None,
) -> str:
    api_ctx, errors = _base_page_context(local_errors)
    support_result, support_train_result, support_errors = _promote_supporting_data(api_ctx.get('api_base'))
    if result is None:
        result = support_result
    if train_result is None:
        train_result = support_train_result
    errors.extend(support_errors)
    return render_template(
        'promote.html',
        api_base=api_ctx.get('api_base'),
        health=api_ctx.get('health'),
        models=api_ctx.get('models'),
        result=result,
        form_result=form_result,
        train_result=train_result,
        input_values=input_values or _default_promote_inputs(),
        field_errors=field_errors or {},
        errors=errors,
        active_page='promote',
    )


def _render_campaign_page(
    *,
    summary: dict[str, Any] | None,
    input_values: dict[str, str],
    train_result: dict[str, Any] | None = None,
    predict_result: dict[str, Any] | None = None,
    field_errors: dict[str, str] | None = None,
    local_errors: list[str] | None = None,
) -> str:
    errors = [msg for msg in (local_errors or []) if msg]
    return render_template(
        'campaign.html',
        summary=summary,
        error=errors[0] if errors else None,
        train_result=train_result,
        predict_result=predict_result,
        field_errors=field_errors or {},
        active_page='campaign',
        input_values=input_values,
    )


@app.get('/logo.png')
def logo() -> Any:
    if LOGO_PATH.exists():
        return send_file(LOGO_PATH)
    if TEMPLATE_LOGO_PATH.exists():
        return send_file(TEMPLATE_LOGO_PATH)
    return {'detail': 'Logo not found'}, 404


@app.get('/')
def index() -> str:
    return _render_home()


@app.get('/promote')
def promote() -> str:
    return _render_promote_page()


@app.post('/predict/<task_name>')
def predict(task_name: str) -> str:
    source_page = _parse_source_page(request.form, default='home')
    error = None
    result: dict[str, Any] | None = None
    input_values: dict[str, Any] = _default_supplier_inputs()

    try:
        if task_name == 'supplier':
            input_values = {
                'quantity': request.form.get('quantity', '12'),
                'unit_price': request.form.get('unit_price', '18.5'),
                'total_ht': request.form.get('total_ht', '222'),
                'total_ttc': request.form.get('total_ttc', '265'),
                'governorate': request.form.get('governorate', 'Tunis'),
                'city': request.form.get('city', 'Tunis'),
            }
            payload = {
                'quantity': float(input_values['quantity']),
                'unit_price': float(input_values['unit_price']),
                'total_ht': float(input_values['total_ht']),
                'total_ttc': float(input_values['total_ttc']),
                'governorate': str(input_values['governorate']).strip() or 'Tunis',
                'city': str(input_values['city']).strip() or 'Tunis',
            }
            result, _ = _api_request('/predict/supplier', method='POST', payload=payload)
        elif task_name == 'sell':
            result, _ = _api_request('/predict/sell')
        elif task_name == 'promote':
            result, _ = _api_request('/predict/promote')
        else:
            error = f'Task inconnue: {task_name}'
    except Exception as exc:
        error = f'La prediction a echoue: {exc}'

    if source_page == 'promote':
        return _render_promote_page(result=result, local_errors=[error] if error else [])
    return _render_home(result=result, local_errors=[error] if error else [])


@app.post('/promote/run')
def promote_run() -> str:
    error = None
    result = None
    try:
        result, _ = _api_request('/predict/promote')
    except Exception as exc:
        error = f'Impossible de recuperer la prediction pour Promote: {exc}'

    return _render_promote_page(result=result, local_errors=[error] if error else [])


@app.post('/promote/train')
def promote_train() -> str:
    error = None
    result = None
    train_result = None
    try:
        train_result, api_base = _api_request('/train/promote/train', method='POST')
        result, _ = _api_request('/predict/promote', base_candidates=[api_base])
    except Exception as exc:
        error = f'Impossible de reentrainer le modele Promote: {exc}'

    return _render_promote_page(
        result=result,
        train_result=train_result,
        local_errors=[error] if error else [],
    )


@app.post('/promote/predict')
def promote_predict() -> str:
    input_values, payload, field_errors = _parse_promote_form(request.form)
    if payload is None:
        return _render_promote_page(
            input_values=input_values,
            field_errors=field_errors,
            local_errors=['Veuillez corriger les champs du formulaire Promote.'],
        )

    error = None
    form_result = None
    try:
        form_result, _ = _api_request('/predict/promote/form', method='POST', payload=payload)
    except Exception as exc:
        error = f'Impossible de calculer la prediction Promote: {exc}'

    return _render_promote_page(
        form_result=form_result,
        input_values=input_values,
        field_errors=field_errors,
        local_errors=[error] if error else [],
    )


@app.get('/campaign')
def campaign() -> str:
    summary = None
    input_values = _default_campaign_inputs()

    api_ctx = _dashboard_api_context()
    summary, campaign_error = _campaign_summary_from_sources(api_ctx.get('api_base'))
    local_errors = []
    if api_ctx.get('api_error'):
        local_errors.append(str(api_ctx.get('api_error')))
    if campaign_error:
        local_errors.append(campaign_error)

    return _render_campaign_page(
        summary=summary,
        input_values=input_values,
        local_errors=local_errors,
    )


@app.post('/campaign/run')
def campaign_run() -> str:
    error = None
    summary = None
    train_result = None
    input_values = _default_campaign_inputs()

    try:
        train_result, api_base = _api_request('/train/campaign/train', method='POST')
        predict_payload, _ = _api_request('/predict/campaign', base_candidates=[api_base])
        if isinstance(predict_payload, dict):
            summary = predict_payload.get('summary')
    except Exception as exc:
        error = f'Execution campagne echouee: {exc}'

    if summary is None and CAMPAIGN_SUMMARY_PATH.exists():
        try:
            summary = json.loads(CAMPAIGN_SUMMARY_PATH.read_text(encoding='utf-8'))
        except Exception:
            summary = None

    if summary is None and error is None:
        error = f'Resume MLOps introuvable apres execution: {CAMPAIGN_SUMMARY_PATH}'

    return _render_campaign_page(
        summary=summary,
        input_values=input_values,
        train_result=train_result,
        local_errors=[error] if error else [],
    )


@app.post('/campaign/predict')
def campaign_predict() -> str:
    summary = None
    predict_result = None
    input_values, payload, field_errors = _parse_campaign_form(request.form)
    if payload is None:
        api_ctx = _dashboard_api_context()
        summary, campaign_error = _campaign_summary_from_sources(api_ctx.get('api_base'))
        local_errors = ['Veuillez corriger les champs du formulaire Campaign.']
        if api_ctx.get('api_error'):
            local_errors.append(str(api_ctx.get('api_error')))
        if campaign_error:
            local_errors.append(campaign_error)
        return _render_campaign_page(
            summary=summary,
            input_values=input_values,
            field_errors=field_errors,
            local_errors=local_errors,
        )

    error = None
    try:
        predict_result, api_base = _api_request('/predict/campaign/form', method='POST', payload=payload)
        predict_payload, _ = _api_request('/predict/campaign', base_candidates=[api_base])
        if isinstance(predict_payload, dict):
            summary = predict_payload.get('summary')
    except Exception as exc:
        error = f'Prediction campagne echouee: {exc}'

    if summary is None and CAMPAIGN_SUMMARY_PATH.exists():
        try:
            summary = json.loads(CAMPAIGN_SUMMARY_PATH.read_text(encoding='utf-8'))
        except Exception:
            summary = None

    return _render_campaign_page(
        summary=summary,
        input_values=input_values,
        predict_result=predict_result,
        field_errors=field_errors,
        local_errors=[error] if error else [],
    )


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
