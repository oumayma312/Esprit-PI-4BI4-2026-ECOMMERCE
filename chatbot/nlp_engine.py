from __future__ import annotations

import re
from dataclasses import dataclass, field

import spacy
from spacy.matcher import PhraseMatcher


@dataclass
class ParsedQuery:
    intent: str
    entities: dict = field(default_factory=dict)


class NLPEngine:
    def __init__(self) -> None:
        self.nlp = spacy.blank("fr")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._register_patterns()

    def _register_patterns(self) -> None:
        patterns = {
            "greeting": ["bonjour", "salut", "hello"],
            "kpi_summary": ["resume", "synthese", "kpi", "performance globale"],
            "top_products": ["top produits", "meilleurs produits", "best sellers"],
            "dataset_overview": [
                "fichiers csv",
                "fichier csv",
                "tables csv",
                "table csv",
                "donnees chargees",
                "analyse des fichiers",
                "quels fichiers",
                "quels csv",
            ],
            "cheap_low_sales": [
                "petit budget",
                "faible budget",
                "pas cher",
                "peu vendus",
                "a promouvoir",
            ],
            "strategy": ["strategie", "recommandation marketing", "campagne", "promouvoir"],
            "search_product": ["chercher", "trouver produit", "produit", "categorie"],
            "exit": ["quit", "exit", "au revoir", "bye"],
        }

        for intent, words in patterns.items():
            self.matcher.add(intent, [self.nlp.make_doc(w) for w in words])

    def parse(self, text: str) -> ParsedQuery:
        doc = self.nlp(text)
        matches = self.matcher(doc)

        intent_scores: dict[str, int] = {}
        for match_id, start, end in matches:
            intent = self.nlp.vocab.strings[match_id]
            score = end - start
            intent_scores[intent] = intent_scores.get(intent, 0) + score

        if intent_scores:
            intent = max(intent_scores, key=intent_scores.get)
        else:
            intent = "strategy"

        entities = self._extract_entities(text)
        return ParsedQuery(intent=intent, entities=entities)

    @staticmethod
    def _extract_entities(text: str) -> dict:
        out: dict = {}
        t = text.lower()

        if any(x in t for x in ["petit budget", "budget faible", "low budget"]):
            out["budget_small"] = True

        top_match = re.search(r"(?:top|meilleurs?)\s*(\d+)", t)
        if top_match:
            out["top_n"] = int(top_match.group(1))

        price_match = re.search(r"(\d+[\.,]?\d*)\s*(dt|tnd|\$|eur|euro|euros)", t)
        if price_match:
            out["price_cap"] = float(price_match.group(1).replace(",", "."))

        product_match = re.search(
            r"(?:produit|reference|categorie|cat[ée]gorie)\s*[:\-]?\s*([\w\s\-àâçéèêëîïôûùüÿñæœ]+)",
            t,
        )
        if product_match:
            value = product_match.group(1).strip()
            if value and len(value) > 1:
                out["query"] = value

        csv_match = re.search(r"(?:csv|table|fichier|dataset)\s*[:\-]\s*([\w\s\-_.]+)", t)
        if csv_match:
            value = csv_match.group(1).strip()
            if value and len(value) > 1:
                out["dataset_query"] = value

        return out
