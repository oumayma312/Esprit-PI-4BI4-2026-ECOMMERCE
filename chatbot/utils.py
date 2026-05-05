import re
import unicodedata
from typing import Optional


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def normalize_search_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def tokenize_query(value: str) -> list[str]:
    return [token for token in normalize_search_text(value).split() if len(token) > 1]


def compact_whitespace(value: str) -> str:
    return " ".join(str(value or "").split())


def parse_mixed_number(value) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[^0-9,.-]", "", text)

    if not text or text in {"-", ".", ","}:
        return None

    if text.count(",") > 0 and text.count(".") > 0:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "")
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") > 0:
        if text.count(",") == 1:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None
