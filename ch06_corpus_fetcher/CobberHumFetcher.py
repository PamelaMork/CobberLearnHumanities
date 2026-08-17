#!/usr/bin/env python3
"""
CobberHumFetcher.py

Companion code for *Foundations of Machine Learning in the Humanities*,
Chapter 6, "From Source to Corpus."

CobberHumFetcher is the first stage of the humanities data bench. It helps
students search public humanities collections, inspect candidate records,
follow promising records back to their sources, and export a documented candidate pool
for later corpus construction in CobberHumCurator.

Design boundary:
- Fetcher finds and inspects candidate records.
- Curator constructs and documents the corpus.
- Preprocessor later cleans and prepares corpus contents for analysis.

Public sources:
- Library of Congress JSON API
- Gutendex / Project Gutenberg metadata API
- Internet Archive Advanced Search API
- World Bank World Development Indicators API

Classroom design choices:
- Launch directly into a single query workspace.
- Keep humanities searches open-ended so students can return for capstone work.
- Use student-facing result tabs: Query summary and Returned records.
  WDI results use Query summary, Returned values, and Plot.
- Normalize archival result tables while preserving source-specific raw
  metadata for inspection and export.
- Export a CSV candidate pool, a JSON metadata file, and a TXT query summary.

Run:
    python CobberHumFetcher.py

Dependencies:
    pip install PyQt6 requests matplotlib
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

try:
    from cobber_hum_branding import apply_app_stylesheet
except ModuleNotFoundError:  # standalone run from this application folder
    import sys as _cobber_sys
    from pathlib import Path as _CobberPath

    _cobber_root = str(_CobberPath(__file__).resolve().parents[1])
    if _cobber_root not in _cobber_sys.path:
        _cobber_sys.path.insert(0, _cobber_root)
    from cobber_hum_branding import apply_app_stylesheet


APP_TITLE = "CobberHumFetcher"
APP_VERSION = "2.0 classroom rebuild"
USER_AGENT = "CobberHumFetcher/2.0 educational app"
REQUEST_TIMEOUT = 30
WDI_COUNTRY_MAP = None
WDI_INDICATOR_INDEX = None


SOURCE_DESCRIPTIONS = {
    "Library of Congress": (
        "Search photographs, recordings, documents, maps, and other cultural materials "
        "described by the Library of Congress."
    ),
    "Gutendex / Project Gutenberg": (
        "Search metadata for public-domain books available through Project Gutenberg."
    ),
    "Internet Archive": (
        "Search books, scans, catalogs, periodicals, and many other digitized materials."
    ),
    "World Bank WDI": (
        "Retrieve a quantitative time series that can provide historical or social context."
    ),
}

QUERY_PLACEHOLDERS = {
    "Library of Congress": (
        "Enter one search per line.\n\n"
        "Examples:\n"
        "migrant workers\n"
        "migrant workers camp\n"
        "women suffrage\n"
        "civil rights oral history"
    ),
    "Gutendex / Project Gutenberg": (
        "Enter one search per line.\n\n"
        "Examples:\n"
        "ghost stories\n"
        "education\n"
        "slavery\n"
        "religion"
    ),
    "Internet Archive": (
        "Enter one search per line.\n\n"
        "Examples:\n"
        "Japanese textiles\n"
        "school yearbook\n"
        "local history Minnesota\n"
        "civil war diary"
    ),
    "World Bank WDI": (
        "Enter an indicator and country, one per line.\n\n"
        "Examples:\n"
        "population Papua New Guinea\n"
        "life expectancy South Africa\n"
        "GDP Brazil\n\n"
        "Advanced form also works:\n"
        "PNG, SP.POP.TOTL"
    ),
}

SAMPLE_QUERIES = {
    "LOC - Migrant workers": {
        "source": "Library of Congress",
        "queries": "migrant workers\nmigrant workers camp",
        "limit": 10,
    },
    "Gutenberg - Ghost stories": {
        "source": "Gutendex / Project Gutenberg",
        "queries": "ghost stories",
        "limit": 10,
    },
    "Internet Archive - Japanese textiles": {
        "source": "Internet Archive",
        "queries": "Japanese textiles",
        "limit": 10,
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_join(value: Any, max_items: int = 5) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "name" in value:
            return str(value["name"])
        return json.dumps(value, ensure_ascii=False)[:300]
    if isinstance(value, list):
        items = []
        for x in value[:max_items]:
            joined = safe_join(x)
            if joined:
                items.append(joined)
        return "; ".join(items)
    return str(value)


def truncate(text: str, n: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def first_format_url(formats: Dict[str, str], preferred_words: List[str]) -> str:
    if not isinstance(formats, dict):
        return ""
    for word in preferred_words:
        for mime, url in formats.items():
            if word.lower() in mime.lower() and isinstance(url, str):
                return url
    return ""




# A few very common shortcuts remain for convenience, but CobberHumFetcher
# now retrieves the full World Development Indicators list dynamically.
WDI_COMMON_INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp": "NY.GDP.MKTP.CD",
    "life expectancy": "SP.DYN.LE00.IN",
}


def _normalize_wdi_text(value: str) -> str:
    text = str(value or "").casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def fetch_wdi_country_map() -> Dict[str, str]:
    """
    Retrieve World Bank country names and API country codes.

    In addition to the World Bank's formal display names, build safe
    student-friendly aliases when a shortened everyday name maps to only
    one country. For example:

        Yemen, Rep.        -> Yemen
        Egypt, Arab Rep.   -> Egypt
        Iran, Islamic Rep. -> Iran
        Venezuela, RB      -> Venezuela
        Bahamas, The       -> Bahamas

    Ambiguous shortened names are not added automatically. For example,
    "Congo" is not assigned silently because more than one World Bank
    country name begins with Congo.

    The mapping is cached for the remainder of the app session.
    """
    global WDI_COUNTRY_MAP

    if WDI_COUNTRY_MAP is not None:
        return WDI_COUNTRY_MAP

    url = "https://api.worldbank.org/v2/country"
    params = {
        "format": "json",
        "per_page": 400,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=(5, 30),
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            "CobberHumFetcher could not retrieve the World Bank country list. "
            "Please check your internet connection and try again."
        ) from exc

    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(
            "The World Bank returned an unexpected response while loading countries."
        )

    country_map: Dict[str, str] = {}

    # Track possible shortened everyday names first. We add them only
    # when they resolve uniquely.
    shortened_candidates: Dict[str, set] = {}

    for country in payload[1] or []:
        name = str(country.get("name") or "").strip()
        code = str(country.get("id") or "").strip()

        if not name or not code:
            continue

        code = code.upper()
        formal_key = name.casefold()
        country_map[formal_key] = code

        # Many World Bank display names use a comma followed by a formal
        # constitutional or catalog-style qualifier.
        #
        # Examples:
        #   Yemen, Rep.
        #   Egypt, Arab Rep.
        #   Iran, Islamic Rep.
        #   Venezuela, RB
        #   Bahamas, The
        if "," in name:
            short_name = name.split(",", 1)[0].strip()

            if short_name:
                short_key = short_name.casefold()
                shortened_candidates.setdefault(short_key, set()).add(code)

        # Also support names ending in ", The" by moving the article out
        # of the way for normal student usage.
        lowered_name = name.casefold()
        if lowered_name.endswith(", the"):
            short_name = name[:-5].strip()
            if short_name:
                short_key = short_name.casefold()
                shortened_candidates.setdefault(short_key, set()).add(code)

    # Add only shortened names that point to exactly one country.
    # This prevents "Congo" from silently resolving to the wrong country.
    for short_key, codes in shortened_candidates.items():
        if len(codes) == 1 and short_key not in country_map:
            country_map[short_key] = next(iter(codes))

    # Common student-friendly aliases that cannot be derived reliably
    # from World Bank display names.
    country_map.update({
        "usa": "USA",
        "us": "USA",
        "u.s.": "USA",
        "u.s.a.": "USA",
        "united states of america": "USA",
        "uk": "GBR",
        "u.k.": "GBR",
        "great britain": "GBR",
        "britain": "GBR",
        "england": "GBR",
        "scotland": "GBR",
        "wales": "GBR",
        "northern ireland": "GBR",
        "democratic republic of the congo": "COD",
        "democratic republic of congo": "COD",
        "dr congo": "COD",
        "drc": "COD",
        "congo kinshasa": "COD",

        "republic of the congo": "COG",
        "republic of congo": "COG",
        "congo brazzaville": "COG",
        "south korea": "KOR",
        "north korea": "PRK",
    })

    WDI_COUNTRY_MAP = country_map
    return WDI_COUNTRY_MAP

def fetch_wdi_indicator_index() -> List[Dict[str, str]]:
    """
    Retrieve the complete World Development Indicators series list.

    Source 2 is the World Development Indicators database. The list is
    cached for the remainder of the app session.
    """
    global WDI_INDICATOR_INDEX

    if WDI_INDICATOR_INDEX is not None:
        return WDI_INDICATOR_INDEX

    url = "https://api.worldbank.org/v2/indicator"
    params = {
        "format": "json",
        "source": "2",
        "per_page": 20000,
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=(5, 45),
        )
        response.raise_for_status()
        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            "CobberHumFetcher could not retrieve the World Bank indicator list. "
            "Please wait a moment and try again."
        ) from exc

    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(
            "The World Bank returned an unexpected response while loading indicators."
        )

    indicators: List[Dict[str, str]] = []

    for indicator in payload[1] or []:
        code = str(indicator.get("id") or "").strip()
        name = str(indicator.get("name") or "").strip()

        if not code or not name:
            continue

        indicators.append({
            "code": code,
            "name": name,
            "normalized_name": _normalize_wdi_text(name),
        })

    if not indicators:
        raise RuntimeError(
            "World Bank WDI did not return an indicator list. "
            "Please wait a moment and try again."
        )

    WDI_INDICATOR_INDEX = indicators
    return WDI_INDICATOR_INDEX


def _indicator_token_similarity(query_token: str, candidate_token: str) -> float:
    if query_token == candidate_token:
        return 1.0

    # Handle common word-family changes such as salary -> salaried.
    if len(query_token) >= 5 and len(candidate_token) >= 5:
        common_prefix = 0
        for a, b in zip(query_token, candidate_token):
            if a != b:
                break
            common_prefix += 1
        if common_prefix >= 5:
            return 0.9

    return difflib.SequenceMatcher(None, query_token, candidate_token).ratio()


def resolve_wdi_indicator(indicator_phrase: str) -> Tuple[str, str]:
    """
    Resolve a student's indicator phrase against the complete WDI indicator list.

    Returns (indicator_code, indicator_name).
    """
    phrase = " ".join(str(indicator_phrase or "").strip().split())
    if not phrase:
        raise ValueError(
            "Enter an indicator before the country name, such as "
            "'population Papua New Guinea'."
        )

    normalized = _normalize_wdi_text(phrase)

    # Familiar shortcuts.
    shortcut_code = WDI_COMMON_INDICATORS.get(normalized)
    if shortcut_code:
        for item in fetch_wdi_indicator_index():
            if item["code"].upper() == shortcut_code.upper():
                return item["code"], item["name"]
        return shortcut_code, phrase

    # Advanced users may enter a WDI code directly.
    if re.fullmatch(r"[A-Za-z0-9_.-]+", phrase) and "." in phrase:
        code = phrase.upper()
        for item in fetch_wdi_indicator_index():
            if item["code"].upper() == code:
                return item["code"], item["name"]
        return code, code

    indicators = fetch_wdi_indicator_index()

    # Exact indicator-name match.
    for item in indicators:
        if item["normalized_name"] == normalized:
            return item["code"], item["name"]

    # Strong phrase containment match.
    containment = [
        item for item in indicators
        if normalized and normalized in item["normalized_name"]
    ]

    if containment:
        # Prefer a general "total" series over sex-specific versions when
        # the student's query did not itself specify male/female.
        def containment_score(item):
            name = item["normalized_name"]
            score = 0
            if name.startswith(normalized):
                score += 100
            if " total " in f" {name} " and not any(
                word in normalized for word in ("male", "female", "women", "men")
            ):
                score += 20
            if "male" in name or "female" in name:
                score -= 10
            score -= len(name) / 1000
            return score

        containment.sort(key=containment_score, reverse=True)

        if len(containment) == 1:
            best = containment[0]
            return best["code"], best["name"]

        top_score = containment_score(containment[0])
        second_score = containment_score(containment[1])

        # A clearly better generic match can be chosen automatically.
        if top_score - second_score >= 10:
            best = containment[0]
            return best["code"], best["name"]

    # Token-level fuzzy matching lets natural forms such as "salary"
    # find indicators containing "salaried".
    query_tokens = normalized.split()
    scored = []

    for item in indicators:
        candidate_tokens = item["normalized_name"].split()
        if not candidate_tokens:
            continue

        token_scores = []
        for query_token in query_tokens:
            best_token_score = max(
                _indicator_token_similarity(query_token, candidate_token)
                for candidate_token in candidate_tokens
            )
            token_scores.append(best_token_score)

        if token_scores and min(token_scores) >= 0.68:
            score = sum(token_scores) / len(token_scores)

            name = item["normalized_name"]
            if " total " in f" {name} " and not any(
                word in normalized for word in ("male", "female", "women", "men")
            ):
                score += 0.08
            if (" male " in f" {name} " or " female " in f" {name} ") and not any(
                word in normalized for word in ("male", "female", "women", "men")
            ):
                score -= 0.05

            scored.append((score, item))

    scored.sort(key=lambda pair: (pair[0], -len(pair[1]["name"])), reverse=True)

    if scored:
        best_score, best = scored[0]

        # Choose automatically only when the match is reasonably strong
        # and meaningfully better than the next candidate.
        if best_score >= 0.82:
            if len(scored) == 1 or best_score - scored[1][0] >= 0.04:
                return best["code"], best["name"]

        suggestions = []
        for _, item in scored[:4]:
            if item["name"] not in suggestions:
                suggestions.append(item["name"])

        if suggestions:
            suggestion_text = "; ".join(suggestions)
            raise ValueError(
                f"Several World Bank WDI indicators could match '{phrase}'. "
                f"Try a more specific indicator phrase. Possible matches include: "
                f"{suggestion_text}"
            )

    raise ValueError(
        f"World Bank WDI could not find an indicator matching '{phrase}'. "
        "Try a more specific indicator name."
    )


def parse_wdi_line(line: str) -> Tuple[str, str]:
    cleaned = " ".join(line.strip().split())

    if not cleaned:
        raise ValueError(
            "Enter an indicator and a country, such as "
            "'population Papua New Guinea'."
        )

    # Preserve the advanced code-based form:
    # PNG, SP.POP.TOTL
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
        if (
            len(parts) == 2
            and re.fullmatch(r"[A-Za-z]{2,3}", parts[0] or "")
            and "." in (parts[1] or "")
        ):
            return parts[0].upper(), parts[1].upper()

    if ";" in cleaned:
        parts = [p.strip() for p in cleaned.split(";", 1)]
        if (
            len(parts) == 2
            and re.fullmatch(r"[A-Za-z]{2,3}", parts[0] or "")
            and "." in (parts[1] or "")
        ):
            return parts[0].upper(), parts[1].upper()

    country_map = fetch_wdi_country_map()
    lowered = cleaned.casefold()

    # Find the longest country name or alias at the END of the search.
    # This supports multi-word names such as Papua New Guinea and Great Britain.
    country_match = None

    for country_name in sorted(country_map, key=len, reverse=True):
        if lowered == country_name or lowered.endswith(" " + country_name):
            country_match = country_name
            break

    if not country_match:
        raise ValueError(
            "World Bank WDI could not recognize the country name. "
            "Check the spelling or try the country's full name. "
            "For example: 'population Papua New Guinea'."
        )

    country_code = country_map[country_match]
    indicator_phrase = cleaned[: len(cleaned) - len(country_match)].strip()

    # Natural-language forms such as "population of Papua New Guinea".
    if indicator_phrase.casefold().endswith(" of"):
        indicator_phrase = indicator_phrase[:-3].strip()

    indicator_code, _indicator_name = resolve_wdi_indicator(indicator_phrase)
    return country_code, indicator_code

def html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def clean_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "candidate_pool"


# ---------------------------------------------------------------------------
# Result packages
# ---------------------------------------------------------------------------

@dataclass
class ResultPackage:
    source: str
    query: str
    fetched_at: str
    preview_rows: List[Dict[str, Any]]
    raw_metadata: Any
    total_matching: Any = "unknown"
    summary_details: Dict[str, Any] = field(default_factory=dict)
    export_name: str = "candidate_pool"
    result_type: str = "archive"

    @property
    def title(self) -> str:
        short_query = truncate(self.query, 28)
        if self.source == "Gutendex / Project Gutenberg":
            source_label = "Gutenberg"
        elif self.source == "Library of Congress":
            source_label = "LOC"
        elif self.source == "Internet Archive":
            source_label = "Internet Archive"
        else:
            source_label = "WDI"
        return f"{source_label}: {short_query}"


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

class SourceAdapters:
    @staticmethod
    def fetch_loc(query: str, max_results: int) -> ResultPackage:
        url = "https://www.loc.gov/search/"
        params = {"fo": "json", "q": query, "c": str(max_results)}
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        rows = []
        for item in payload.get("results", [])[:max_results]:
            rows.append(
                {
                    "title": safe_join(item.get("title")),
                    "date": safe_join(item.get("date")),
                    "creator": safe_join(item.get("contributor") or item.get("creator")),
                    "type": safe_join(item.get("original_format") or item.get("type") or item.get("format")),
                    "language": safe_join(item.get("language")),
                    "subjects": safe_join(item.get("subject")),
                    "url": safe_join(item.get("url")),
                    "description": truncate(safe_join(item.get("description")), 700),
                    "source_id": safe_join(item.get("id")),
                    "raw": item,
                }
            )

        pagination = payload.get("pagination") or {}
        total = pagination.get("total", "unknown")

        return ResultPackage(
            source="Library of Congress",
            query=query,
            fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            preview_rows=rows,
            raw_metadata=payload,
            total_matching=total,
            summary_details={
                "preview_count": len(rows),
                "preview_requested": max_results,
                "total_matching": total,
                "record_types": "Photographs, recordings, documents, maps, and other cataloged materials may appear.",
            },
            export_name=f"loc_{clean_filename(query)}",
        )

    @staticmethod
    def fetch_gutendex(query: str, max_results: int) -> ResultPackage:
        url = "https://gutendex.com/books"
        params = {"search": query}
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        rows = []
        for item in payload.get("results", [])[:max_results]:
            authors = "; ".join(a.get("name", "") for a in item.get("authors", []) if a.get("name"))
            formats = item.get("formats", {})
            txt_url = first_format_url(formats, ["text/plain"])
            html_url = first_format_url(formats, ["text/html"])
            rows.append(
                {
                    "title": safe_join(item.get("title")),
                    "date": "",
                    "creator": authors,
                    "type": "Book",
                    "language": safe_join(item.get("languages")),
                    "subjects": safe_join(item.get("subjects")),
                    "url": html_url or txt_url or f"https://www.gutenberg.org/ebooks/{item.get('id')}",
                    "description": (
                        f"Languages: {safe_join(item.get('languages'))}; "
                        f"Downloads: {item.get('download_count', '')}"
                    ),
                    "source_id": str(item.get("id", "")),
                    "text_url": txt_url,
                    "raw": item,
                }
            )

        total = payload.get("count", "unknown")
        return ResultPackage(
            source="Gutendex / Project Gutenberg",
            query=query,
            fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            preview_rows=rows,
            raw_metadata=payload,
            total_matching=total,
            summary_details={
                "preview_count": len(rows),
                "preview_requested": max_results,
                "total_matching": total,
                "record_types": "Public-domain books and book-length works described by Gutendex.",
            },
            export_name=f"gutenberg_{clean_filename(query)}",
        )

    @staticmethod
    def fetch_internet_archive(query: str, max_results: int) -> ResultPackage:
        url = "https://archive.org/advancedsearch.php"
        fields = ["identifier", "title", "creator", "date", "mediatype", "language", "description", "subject"]
        headers = {"User-Agent": USER_AGENT}

        quoted = query.replace('"', "").strip()
        attempts = [f'title:("{quoted}") OR subject:("{quoted}")', query]
        last_error = None

        for query_used in attempts:
            params = [("q", query_used), ("rows", str(max_results)), ("page", "1"), ("output", "json")]
            for field_name in fields:
                params.append(("fl[]", field_name))

            try:
                response = requests.get(url, params=params, headers=headers, timeout=(5, 12))
                response.raise_for_status()
                payload = response.json()
                response_block = payload.get("response", {})
                docs = response_block.get("docs", [])

                rows = []
                for item in docs[:max_results]:
                    identifier = safe_join(item.get("identifier"))
                    rows.append(
                        {
                            "title": safe_join(item.get("title")),
                            "date": safe_join(item.get("date")),
                            "creator": safe_join(item.get("creator")),
                            "type": safe_join(item.get("mediatype")),
                            "language": safe_join(item.get("language")),
                            "subjects": safe_join(item.get("subject")),
                            "url": f"https://archive.org/details/{identifier}" if identifier else "",
                            "description": truncate(safe_join(item.get("description")), 700),
                            "source_id": identifier,
                            "mediatype": safe_join(item.get("mediatype")),
                            "query_used": query_used,
                                            "raw": item,
                        }
                    )

                total = response_block.get("numFound", "unknown")
                return ResultPackage(
                    source="Internet Archive",
                    query=query,
                    fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    preview_rows=rows,
                    raw_metadata=payload,
                    total_matching=total,
                    summary_details={
                        "preview_count": len(rows),
                        "total_matching": total,
                        "query_used": query_used,
                        "record_types": "Books, scans, catalogs, periodicals, and other digitized materials may appear.",
                    },
                    export_name=f"internet_archive_{clean_filename(query)}",
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Internet Archive is slow or unavailable for '{query}'. "
            f"Try a narrower query later, or use Library of Congress/Gutendex for now. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def fetch_wdi(query: str, max_results: int) -> ResultPackage:
        del max_results  # WDI returns a full time series rather than a preview limit.
        country, indicator = parse_wdi_line(query)
        url = f"https://api.worldbank.org/v2/country/{quote_plus(country)}/indicator/{quote_plus(indicator)}"
        params = {"format": "json", "per_page": "20000"}
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(
            "World Bank WDI did not return data for this search. "
            "Check the country name and indicator, then try again."
        )

        rows = []
        indicator_name = ""
        country_name = country
        for item in payload[1] or []:
            value = item.get("value")
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except Exception:
                continue
            try:
                year = int(item.get("date"))
            except Exception:
                continue
            indicator_name = item.get("indicator", {}).get("value", indicator_name)
            country_name = item.get("country", {}).get("value", country_name)
            rows.append(
                {
                    "year": year,
                    "value": numeric_value,
                    "country": country_name,
                    "country_code": country,
                    "indicator": indicator,
                    "indicator_name": indicator_name,
                }
            )

        rows.sort(key=lambda row: row["year"])
        if not rows:
            raise RuntimeError(
            "World Bank WDI found the country and indicator, "
            "but returned no numeric data for this search."
        )

        return ResultPackage(
            source="World Bank WDI",
            query=f"{country}, {indicator}",
            fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            preview_rows=rows,
            raw_metadata=payload,
            total_matching=len(rows),
            summary_details={
                "preview_count": len(rows),
                "country": country,
                "country_name": country_name,
                "indicator": indicator,
                "indicator_name": indicator_name or indicator,
            },
            export_name=f"wdi_{clean_filename(country)}_{clean_filename(indicator)}",
            result_type="wdi",
        )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


class FetchWorker(QRunnable):
    def __init__(self, source: str, queries: List[str], max_results: int):
        super().__init__()
        self.source = source
        self.queries = queries
        self.max_results = max_results
        self.signals = WorkerSignals()

    def run(self):
        try:
            for query in self.queries:
                query = query.strip()
                if not query:
                    continue

                try:
                    self.signals.progress.emit(f"Fetching from {self.source}: {query}")
                    if self.source == "Library of Congress":
                        result = SourceAdapters.fetch_loc(query, self.max_results)
                    elif self.source == "Gutendex / Project Gutenberg":
                        result = SourceAdapters.fetch_gutendex(query, self.max_results)
                    elif self.source == "Internet Archive":
                        result = SourceAdapters.fetch_internet_archive(query, self.max_results)
                    elif self.source == "World Bank WDI":
                        result = SourceAdapters.fetch_wdi(query, self.max_results)
                    else:
                        raise RuntimeError(f"Unknown source: {self.source}")
                    self.signals.result.emit(result)
                except Exception as exc:
                    self.signals.error.emit(f"Could not process '{query}': {exc}")
        finally:
            self.signals.finished.emit()


# ---------------------------------------------------------------------------
# Student-facing panels
# ---------------------------------------------------------------------------


class RecordDetailsDialog:
    """Build and show a student-readable metadata popup for one record."""

    @staticmethod
    def show(parent: QWidget, result: ResultPackage, record: Dict[str, Any]):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox

        dialog = QDialog(parent)
        dialog.setWindowTitle("Record details")
        dialog.resize(760, 700)

        outer = QVBoxLayout(dialog)

        title = QLabel(record.get("title") or "Untitled record")
        title_font = QFont("Lato", 15)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setWordWrap(True)
        title.setStyleSheet("color: #6C1D45;")
        outer.addWidget(title)

        intro = QLabel(
            "CobberHumFetcher has organized selected fields from the source metadata "
            "to make this record easier to read."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555555; margin-bottom: 6px;")
        outer.addWidget(intro)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        raw = record.get("raw", {}) or {}
        item = raw.get("item", {}) if isinstance(raw, dict) else {}

        def first_nonempty(*values):
            for value in values:
                if value not in (None, "", [], {}):
                    return value
            return ""

        def display_value(value):
            if value is None:
                return ""
            if isinstance(value, list):
                return "<br>".join(html_escape(safe_join(v, max_items=20)) for v in value if safe_join(v, max_items=20))
            if isinstance(value, dict):
                return html_escape(json.dumps(value, ensure_ascii=False))
            return html_escape(value)

        # Source-specific richer fields, with normalized fallbacks.
        creator_detail = first_nonempty(
            item.get("contributors") if isinstance(item, dict) else "",
            raw.get("contributor") if isinstance(raw, dict) else "",
            record.get("creator"),
        )
        type_detail = first_nonempty(
            item.get("format") if isinstance(item, dict) else "",
            raw.get("original_format") if isinstance(raw, dict) else "",
            record.get("type"),
        )
        language_detail = first_nonempty(
            raw.get("language") if isinstance(raw, dict) else "",
            item.get("language") if isinstance(item, dict) else "",
            record.get("language"),
        )
        subjects_detail = first_nonempty(
            item.get("subjects") if isinstance(item, dict) else "",
            raw.get("subject") if isinstance(raw, dict) else "",
            record.get("subjects"),
        )
        description_detail = first_nonempty(
            item.get("summary") if isinstance(item, dict) else "",
            raw.get("description") if isinstance(raw, dict) else "",
            record.get("description"),
        )
        published_detail = item.get("created_published") if isinstance(item, dict) else ""
        medium_detail = item.get("medium") if isinstance(item, dict) else ""
        notes_detail = item.get("notes") if isinstance(item, dict) else ""
        rights_detail = first_nonempty(
            item.get("rights_advisory") if isinstance(item, dict) else "",
            item.get("rights") if isinstance(item, dict) else "",
            raw.get("rights") if isinstance(raw, dict) else "",
        )

        rows = [
            ("Date", record.get("date")),
            ("Creator / Contributor", creator_detail),
            ("Type", type_detail),
            ("Language", language_detail),
            ("Subjects", subjects_detail),
            ("Description", description_detail),
            ("Published", published_detail),
            ("Medium", medium_detail),
            ("Notes", notes_detail),
            ("Rights", rights_detail),
            ("Source ID", record.get("source_id")),
        ]

        html_parts = ["<div style='font-family:Lato,Arial,sans-serif; font-size:10.5pt;'>"]
        for label, value in rows:
            if value in (None, "", [], {}):
                continue
            html_parts.append(
                f"<p style='margin:0 0 11px 0;'><b>{html_escape(label)}</b><br>"
                f"{display_value(value)}</p>"
            )
        html_parts.append("</div>")
        browser.setHtml("".join(html_parts))
        outer.addWidget(browser, 1)

        button_row = QHBoxLayout()

        open_btn = QPushButton("Open source")
        open_btn.setEnabled(bool(record.get("url")))
        if record.get("url"):
            open_btn.clicked.connect(
                lambda checked=False, target=str(record.get("url")):
                QDesktopServices.openUrl(QUrl(target))
            )

        raw_btn = QPushButton("Show raw metadata")
        raw_btn.setCheckable(True)

        button_row.addWidget(open_btn)
        button_row.addWidget(raw_btn)
        button_row.addStretch(1)
        outer.addLayout(button_row)

        raw_browser = QTextBrowser()
        raw_browser.setPlainText(json.dumps(raw, indent=2, ensure_ascii=False))
        raw_browser.setVisible(False)
        outer.addWidget(raw_browser, 1)

        def toggle_raw(checked: bool):
            raw_browser.setVisible(checked)
            raw_btn.setText("Hide raw metadata" if checked else "Show raw metadata")

        raw_btn.toggled.connect(toggle_raw)

        close_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_buttons.rejected.connect(dialog.reject)
        outer.addWidget(close_buttons)

        dialog.exec()


class ArchiveResultPanel(QWidget):
    """Wide returned-record comparison table with student-readable record details."""

    TABLE_COLUMNS = ["title", "date", "creator", "type", "language", "subjects"]

    def __init__(self, result: ResultPackage):
        super().__init__()
        self.result = result

        layout = QVBoxLayout(self)

        heading = QLabel("Returned records")
        heading_font = QFont("Lato", 14)
        heading_font.setWeight(QFont.Weight.Bold)
        heading.setFont(heading_font)
        heading.setStyleSheet("color: #6C1D45;")
        layout.addWidget(heading)

        table_note = QLabel(
            "Compare the candidate records across the table. Select a row to view "
            "a readable record summary or return to the original source."
        )
        table_note.setWordWrap(True)
        table_note.setStyleSheet("color: #555555;")
        layout.addWidget(table_note)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(False)
        self._populate_table()
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        self.details_btn = QPushButton("View record details")
        self.details_btn.setEnabled(False)
        self.open_source_btn = QPushButton("Open source")
        self.open_source_btn.setEnabled(False)

        button_row.addWidget(self.details_btn)
        button_row.addWidget(self.open_source_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(lambda _item: self._show_details())
        self.details_btn.clicked.connect(self._show_details)
        self.open_source_btn.clicked.connect(self._open_source)

        if self.result.preview_rows:
            self.table.selectRow(0)

    @staticmethod
    def _subjects_for_table(value: Any, limit: int = 180) -> str:
        text = safe_join(value, max_items=6)
        return truncate(text, limit)

    def _populate_table(self):
        rows = self.result.preview_rows

        self.table.setColumnCount(len(self.TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(
            ["Title", "Date", "Creator / Contributor", "Type", "Language", "Subjects"]
        )
        self.table.setRowCount(len(rows))

        for r, record in enumerate(rows):
            display_values = {
                "title": record.get("title", ""),
                "date": record.get("date", ""),
                "creator": record.get("creator", ""),
                "type": record.get("type", ""),
                "language": record.get("language", ""),
                "subjects": self._subjects_for_table(record.get("subjects", "")),
            }

            for c, key in enumerate(self.TABLE_COLUMNS):
                item = QTableWidgetItem(str(display_values.get(key, "") or ""))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                )
                self.table.setItem(r, c, item)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)           # Title
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Date
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)           # Creator
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Type
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents) # Language
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)           # Subjects

        self.table.verticalHeader().setDefaultSectionSize(58)
        self.table.resizeRowsToContents()
        self.table.setSortingEnabled(True)

    def _selected_record(self) -> Optional[Dict[str, Any]]:
        row = self.table.currentRow()
        if 0 <= row < len(self.result.preview_rows):
            return self.result.preview_rows[row]
        return None

    def _update_buttons(self):
        record = self._selected_record()
        self.details_btn.setEnabled(record is not None)
        self.open_source_btn.setEnabled(bool(record and record.get("url")))

    def _show_details(self):
        record = self._selected_record()
        if record is None:
            return
        RecordDetailsDialog.show(self, self.result, record)

    def _open_source(self):
        record = self._selected_record()
        if not record:
            return
        url = str(record.get("url") or "").strip()
        if url:
            QDesktopServices.openUrl(QUrl(url))


class SimpleTablePanel(QWidget):
    def __init__(self, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None):
        super().__init__()
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.populate(rows, columns)

    def populate(self, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None):
        self.table.clear()
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        if columns is None:
            columns = list(rows[0].keys())
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, col in enumerate(columns):
                raw_value = row.get(col)

                if raw_value is None:
                    value = ""

                elif col == "value" and isinstance(raw_value, (int, float)):
                    if float(raw_value).is_integer():
                        value = f"{raw_value:,.0f}"
                    else:
                        value = f"{raw_value:,.2f}".rstrip("0").rstrip(".")

                else:
                    value = str(raw_value)

                self.table.setItem(r, c, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)


class WDIPlotPanel(QWidget):
    def __init__(self, result: ResultPackage):
        super().__init__()
        layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure(figsize=(7, 4)))
        layout.addWidget(self.canvas)
        self._draw(result)

    def _draw(self, result: ResultPackage):
        rows = result.preview_rows
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if rows:
            years = [row["year"] for row in rows]
            values = [row["value"] for row in rows]
            ax.plot(years, values, marker="o", linewidth=1.5, markersize=3)
            indicator_name = result.summary_details.get("indicator_name", "Value")

            title = (
                f"{result.summary_details.get('country_name', '')}: "
                f"{indicator_name}"
            )

            ax.set_title(title)
            ax.set_xlabel("Year")
            ax.set_ylabel(indicator_name)

            def human_number(value, _position):
                if abs(value) >= 1000:
                    return f"{value:,.0f}"
                return f"{value:,.2f}".rstrip("0").rstrip(".")

            ax.yaxis.set_major_formatter(FuncFormatter(human_number))
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data returned", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        self.canvas.draw()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class CobberHumFetcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.results: List[ResultPackage] = []

        self.cobber_maroon = QColor(108, 29, 69)
        self.base_font = QFont("Lato", 10)
        self.setFont(self.base_font)

        self.setWindowTitle(APP_TITLE)
        self._set_laptop_friendly_geometry()
        self._build_ui()
        self.statusBar().showMessage("Ready. Build a query and fetch candidate records.")

    def _set_laptop_friendly_geometry(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1180, 720)
            return
        geom = screen.availableGeometry()
        width = min(1380, max(1120, int(geom.width() * 0.95)))
        height = min(780, max(650, int(geom.height() * 0.90)))
        self.resize(width, height)
        x = geom.x() + max(0, (geom.width() - width) // 2)
        y = geom.y() + max(0, (geom.height() - height) // 2)
        self.move(x, y)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_main_page())
        self.setStatusBar(QStatusBar())

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left workspace.
        left = QWidget()
        left.setMinimumWidth(285)
        left.setMaximumWidth(330)
        left_layout = QVBoxLayout(left)

        group_title_style = f"""
        QGroupBox::title {{
            color: {self.cobber_maroon.name()};
            padding: 0 4px;
        }}
        """

        title_font = QFont(self.base_font)
        title_font.setPointSize(13)
        title_font.setWeight(QFont.Weight.Bold)

        form_box = QGroupBox("Build your query")
        form_box.setFont(title_font)
        form_box.setStyleSheet(group_title_style)
        form_box_layout = QVBoxLayout(form_box)

        form_content = QWidget()
        form_content.setFont(self.base_font)
        form_layout = QVBoxLayout(form_content)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_box_layout.addWidget(form_content)

        def bold_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold;")
            return label

        self.source_combo = QComboBox()
        self.source_combo.addItems(
            [
                "Library of Congress",
                "Gutendex / Project Gutenberg",
                "Internet Archive",
                "World Bank WDI",
            ]
        )
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        form_layout.addWidget(bold_label("Source:"))
        form_layout.addWidget(self.source_combo)

        self.source_description = QLabel()
        self.source_description.setWordWrap(True)
        self.source_description.setStyleSheet("color: #555555;")
        form_layout.addWidget(self.source_description)

        self.input_label = bold_label("Search terms or phrases, one per line:")
        self.query_input = QTextEdit()
        self.query_input.setMinimumHeight(155)
        form_layout.addWidget(self.input_label)
        form_layout.addWidget(self.query_input)

        self.limit_input = QLineEdit()
        self.limit_input.setValidator(QIntValidator(1, 50, self))
        self.limit_input.setText("10")
        self.limit_input.setMaximumWidth(90)
        form_layout.addWidget(bold_label("Preview size:"))
        form_layout.addWidget(self.limit_input)

        self.button_height = 60
        self.active_button_style = f"""
        QPushButton {{
            background-color: {self.cobber_maroon.name()};
            color: white;
            font-weight: bold;
            font-size: 11pt;
            border: none;
            border-radius: 6px;
            padding: 10px 14px;
        }}
        QPushButton:hover {{ background-color: #5A1839; }}
        QPushButton:disabled {{ background-color: #9A7287; color: white; }}
        """
        self.inactive_button_style = """
        QPushButton {
            background-color: #666666;
            color: white;
            font-weight: normal;
            font-size: 11pt;
            border: none;
            border-radius: 6px;
            padding: 10px 14px;
        }
        QPushButton:hover { background-color: #555555; }
        QPushButton:disabled { background-color: #777777; color: white; }
        """

        button_column = QVBoxLayout()
        button_column.setSpacing(6)

        self.fetch_btn = QPushButton("Fetch candidate records")
        self.fetch_btn.setFixedHeight(self.button_height)
        self.fetch_btn.setStyleSheet(self.active_button_style)
        self.fetch_btn.clicked.connect(self.start_fetch)

        self.export_btn = QPushButton("Export current result")
        self.export_btn.setFixedHeight(self.button_height)
        self.export_btn.setStyleSheet(self.inactive_button_style)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_current_result)

        self.clear_btn = QPushButton("Clear query and results")
        self.clear_btn.setFixedHeight(self.button_height)
        self.clear_btn.setStyleSheet(self.inactive_button_style)
        self.clear_btn.clicked.connect(self.clear_query)

        button_column.addWidget(self.fetch_btn)
        button_column.addWidget(self.export_btn)
        button_column.addWidget(self.clear_btn)

        left_layout.addWidget(form_box)
        left_layout.addLayout(button_column)
        left_layout.addStretch(1)

        # Right result workspace.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.close_result_tab)
        right_layout.addWidget(self.results_tabs)
        self._add_start_tab()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([310, 1040])

        self.on_source_changed(self.source_combo.currentText())
        return page

    def _add_start_tab(self):
        empty = QTextBrowser()
        empty.setHtml(
            "<h2>Welcome to CobberHumFetcher</h2>"
            "<p>Search a public humanities collection, inspect the returned candidate records, "
            "follow promising records back to their sources, and save a candidate pool for later corpus building.</p>"
            "<p><b>Library of Congress</b> provides records for many kinds of cultural materials. "
            "<b>Project Gutenberg</b> provides records for public-domain books. "
            "<b>Internet Archive</b> provides access to many kinds of digitized materials.</p>"
            "<p>A returned record is not automatically part of your corpus. Inspect the record and the source "
            "before deciding what it can contribute to your question.</p>"
        )
        self.results_tabs.addTab(empty, "Start here")

    def on_source_changed(self, source: str):
        self.source_description.setText(SOURCE_DESCRIPTIONS.get(source, ""))
        self.query_input.setPlaceholderText(QUERY_PLACEHOLDERS.get(source, ""))
        if source == "World Bank WDI":
            self.input_label.setText("Country code and indicator, one per line:")
            self.limit_input.setEnabled(False)
        else:
            self.input_label.setText("Search terms or phrases, one per line:")
            self.limit_input.setEnabled(True)
            if not self.limit_input.text().strip():
                self.limit_input.setText("10")
        self._mark_query_changed()

    def _mark_query_changed(self, *args):
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet(self.inactive_button_style)

    def _collect_queries(self) -> Tuple[str, List[str], int]:
        source = self.source_combo.currentText()
        queries = [line.strip() for line in self.query_input.toPlainText().splitlines() if line.strip()]
        limit_text = self.limit_input.text().strip()
        limit = int(limit_text) if limit_text else 10
        return source, queries, limit

    def start_fetch(self):
        source, queries, limit = self._collect_queries()
        if not queries:
            QMessageBox.warning(self, APP_TITLE, "Please enter at least one search term or identifier.")
            return

        self.fetch_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        worker = FetchWorker(source, queries, limit)
        worker.signals.progress.connect(self.log_progress)
        worker.signals.result.connect(self.add_result_tab)
        worker.signals.error.connect(self.log_error)
        worker.signals.finished.connect(self.on_fetch_finished)
        self.threadpool.start(worker)

    def log_progress(self, message: str):
        self.statusBar().showMessage(message)

    def log_error(self, message: str):
        self.statusBar().showMessage("Fetch error.", 5000)
        QMessageBox.warning(self, APP_TITLE, message)

    def on_fetch_finished(self):
        self.fetch_btn.setEnabled(True)
        self.statusBar().showMessage("Fetch complete.", 5000)

    def add_result_tab(self, result: ResultPackage):
        if self.results_tabs.count() == 1 and self.results_tabs.tabText(0) == "Start here":
            self.results_tabs.removeTab(0)

        self.results.append(result)
        container = QWidget()
        layout = QVBoxLayout(container)

        subtabs = QTabWidget()
        subtabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid #B5B5B5;
                top: -1px;
                background-color: white;
            }}
            QTabBar::tab {{
                background-color: #666666;
                color: white;
                font-weight: normal;
                padding: 8px 14px;
                border: 1px solid #555555;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.cobber_maroon.name()};
                color: white;
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{ background-color: #555555; }}
            """
        )

        subtabs.addTab(self._query_summary_panel(result), "Query summary")

        if result.result_type == "wdi":
            subtabs.addTab(
                SimpleTablePanel(result.preview_rows, ["year", "value", "country", "indicator"]),
                "Returned values",
            )
            subtabs.addTab(WDIPlotPanel(result), "Plot")
        else:
            archive_panel = ArchiveResultPanel(result)
            subtabs.addTab(archive_panel, "Returned records")

        layout.addWidget(subtabs)
        self.results_tabs.addTab(container, result.title)
        self.results_tabs.setCurrentWidget(container)

        self.export_btn.setEnabled(True)
        self.export_btn.setStyleSheet(self.active_button_style)
        self.clear_btn.setStyleSheet(self.active_button_style)
        if result.result_type == "wdi":
            self.statusBar().showMessage(
                f"Fetch complete. {len(result.preview_rows)} values returned.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                f"Fetch complete. {len(result.preview_rows)} candidate records shown.",
                5000,
            )

    def _query_summary_panel(self, result: ResultPackage) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        browser = QTextBrowser()
        browser.setHtml(self._query_summary_html(result))
        layout.addWidget(browser)
        return panel

    def _query_summary_html(self, result: ResultPackage) -> str:
        details = result.summary_details
        if result.result_type == "wdi":
            return f"""
            <h2>World Bank WDI query summary</h2>
            <h3>Query used</h3>
            <ul>
              <li><b>Country:</b> {html_escape(details.get('country_name'))} ({html_escape(details.get('country'))})</li>
              <li><b>Indicator:</b> {html_escape(details.get('indicator_name'))}</li>
              <li><b>Indicator code:</b> {html_escape(details.get('indicator'))}</li>
            </ul>
            <h3>Values returned</h3>
            <ul>
              <li><b>Years with numeric values:</b> {html_escape(details.get('preview_count'))}</li>
            </ul>
            <p><b>Fetched at:</b> {html_escape(result.fetched_at)}</p>
            <p><i>A quantitative series can provide context for a humanities question, but it does not replace the sources through which people described or experienced that history.</i></p>
            """

        extra = ""
        if result.source == "Internet Archive":
            extra = f"<li><b>Query sent to Internet Archive:</b> {html_escape(details.get('query_used'))}</li>"

        return f"""
        <h2>{html_escape(result.source)} query summary</h2>

        <h3>Search used</h3>
        <ul>
          <li><b>Search term or phrase:</b> {html_escape(result.query)}</li>
          <li><b>Preview size requested:</b> {html_escape(details.get('preview_requested'))}</li>
          {extra}
        </ul>

        <h3>Records returned</h3>
        <ul>
          <li><b>Candidate records shown:</b> {html_escape(details.get('preview_count'))}</li>
          <li><b>Total matching records reported by source:</b> {html_escape(details.get('total_matching'))}</li>
        </ul>

        <h3>What this source may return</h3>
        <p>{html_escape(details.get('record_types'))}</p>

        <p><b>Fetched at:</b> {html_escape(result.fetched_at)}</p>
        <p><i>The preview is a limited set of returned records, not a corpus. Follow promising records back to their sources before deciding what they can contribute to your question.</i></p>
        """

    def close_result_tab(self, index: int):
        self.results_tabs.removeTab(index)
        if self.results_tabs.count() == 0:
            self._add_start_tab()

    def current_result(self) -> Optional[ResultPackage]:
        current_index = self.results_tabs.currentIndex()
        if current_index < 0:
            return None
        tab_title = self.results_tabs.tabText(current_index)
        for result in reversed(self.results):
            if result.title == tab_title:
                return result
        return None

    def export_current_result(self):
        result = self.current_result()
        if result is None:
            QMessageBox.information(self, APP_TITLE, "There is no result tab selected to export.")
            return
        self.export_result(result)

    def export_result(self, result: ResultPackage):
        target_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not target_dir:
            return
        export_dir = Path(target_dir)
        base = result.export_name or "candidate_pool"
        csv_path = export_dir / f"{base}.csv"
        json_path = export_dir / f"{base}_metadata.json"
        txt_path = export_dir / f"{base}_query_summary.txt"

        try:
            self._write_csv(csv_path, result.preview_rows)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.raw_metadata, f, indent=2, ensure_ascii=False)
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(self._plain_text_query_summary(result))

            self.statusBar().showMessage(f"Exported files to {export_dir}", 5000)
            QMessageBox.information(
                self,
                APP_TITLE,
                f"Exported:\n{csv_path.name}\n{json_path.name}\n{txt_path.name}",
            )
        except Exception as exc:
            self.log_error(f"Export failed: {exc}")

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]):
        if not rows:
            with path.open("w", encoding="utf-8") as f:
                f.write("No preview rows were returned.\n")
            return

        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key != "raw" and key not in keys:
                    keys.append(key)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in keys})

    def _plain_text_query_summary(self, result: ResultPackage) -> str:
        if result.result_type == "wdi":
            lines = [
                "World Bank WDI query summary",
                "",
                f"Query: {result.query}",
                f"Fetched at: {result.fetched_at}",
                f"Values returned: {len(result.preview_rows)}",
                "",
                "Interpretive reminder",
                "A quantitative series can provide context for a humanities question, but it does not replace close work with historical or cultural sources.",
            ]
        else:
            lines = [
                f"{result.source} query summary",
                "",
                f"Search term or phrase: {result.query}",
                f"Fetched at: {result.fetched_at}",
                f"Candidate records shown: {len(result.preview_rows)}",
                f"Total matching records reported by source: {result.total_matching}",
                "",
                "Interpretive reminder",
                "This export is a candidate pool, not a finished corpus.",
                "Inspect individual sources and document the criteria you later use to construct the corpus.",
            ]

        lines.extend(
            [
                "",
                "Exported files",
                "CSV file: returned records or values shown in the app, shown in the app",
                "JSON file: full source response returned during this search",
                "TXT file: query summary and export notes",
                "",
                f"Created by {APP_TITLE} {APP_VERSION}.",
            ]
        )
        return "\n".join(lines) + "\n"

    def clear_query(self):
        self.query_input.clear()
        self.results_tabs.clear()
        self._add_start_tab()
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet(self.inactive_button_style)
        self.clear_btn.setStyleSheet(self.inactive_button_style)
        self.statusBar().showMessage("Ready. Query cleared.")

    def load_sample_query(self):
        # Lightweight menu-based sample loader. Samples teach the chapter but never restrict later searches.
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Load sample query")
        dialog.resize(560, 320)
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItems(list(SAMPLE_QUERIES.keys()))
        details = QTextBrowser()

        def show_details(name: str):
            payload = SAMPLE_QUERIES[name]
            query_html = "<br>".join(html_escape(q) for q in payload["queries"].splitlines())
            details.setHtml(
                f"<h3>{html_escape(name)}</h3>"
                f"<p><b>Source:</b> {html_escape(payload['source'])}</p>"
                f"<p><b>Search:</b><br>{query_html}</p>"
                f"<p><b>Preview size:</b> {payload['limit']}</p>"
            )

        show_details(combo.currentText())
        combo.currentTextChanged.connect(show_details)
        layout.addWidget(QLabel("Choose a sample query:"))
        layout.addWidget(combo)
        layout.addWidget(details)

        buttons = QDialogButtonBox()
        buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            payload = SAMPLE_QUERIES[combo.currentText()]
            self.source_combo.setCurrentText(payload["source"])
            self.query_input.setPlainText(payload["queries"])
            self.limit_input.setText(str(payload["limit"]))
            self.statusBar().showMessage(
                f"Loaded sample query for {payload['source']}.",
                5000,
            )


def main() -> int:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberHumFetcherApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
