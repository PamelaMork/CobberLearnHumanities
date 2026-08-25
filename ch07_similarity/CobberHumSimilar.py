#!/usr/bin/env python3
"""
CobberHumSimilar.py

Humanities Text Similarity Explorer for Foundations of Machine Learning
in the Humanities.

The app has three teaching spaces:

1. Mina's Lab
   A controlled four-text analysis using warning-language features:
   must, not, shall, should, and no. Count-based features are reported
   per 1,000 words. Students can change the prepared feature set, inspect
   feature space when it can be drawn, examine distances, and read the
   resulting similarity matrix.

2. Explore Similarity
   Curated Gutenberg packs explored one collection at a time. Students
   can compare prepared length, voice, topic-word, vocabulary-overlap,
   word-similarity, and broad mixed representations.

3. All Texts
   A scale-up view across the full curated collection using three broad
   representations: broad mixed profile, vocabulary overlap, and TF-IDF
   word similarity.

Important design choice:
    CobberHumSimilar provides prepared representations. It deliberately
    does not include a custom feature builder. Students create and defend
    their own similarity representation later in the Style Detective
    vibe-coding project.

Expected resources
------------------
Place this script beside:

    gutenberg_similarity_packs_curated/
        all_records.jsonl
        packs/...

or beside:

    gutenberg_similarity_packs_curated.zip

The four classroom excerpts used by Mina may also be placed beside the app:

    fairy_folk_tales_03_andersen_s_fairy_tales.txt
    fairy_folk_tales_04_the_arabian_nights_their_best_known_tales.txt
    fairy_folk_tales_09_stories_the_iroquois_tell_their_children.txt
    fairy_folk_tales_10_jewish_fairy_tales_and_legends.txt

If those files are absent, the app falls back to the corresponding copies
inside the curated Gutenberg collection.

Dependencies:
    pip install PyQt6 matplotlib numpy
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------

try:
    from cobber_hum_branding import apply_app_stylesheet
except ModuleNotFoundError:
    def apply_app_stylesheet(app):
        app.setStyleSheet("""
            QWidget {
                font-family: "Lato", "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                color: #3D3D3D;
            }
            QMainWindow { background: #F7F7F7; }
            QTabWidget::pane {
                border: 1px solid #D3D3D3;
                background: white;
            }
            QTabBar::tab {
                padding: 10px 16px;
                background: #ECECEC;
                border: 1px solid #D3D3D3;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: white;
                color: #6C1D45;
                font-weight: 700;
            }
            QGroupBox {
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                background: white;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #6C1D45;
            }
            QPushButton {
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                background: #6C1D45;
                color: white;
                font-weight: 700;
            }
            QPushButton:hover { background: #531634; }
            QComboBox, QTextEdit, QTableWidget {
                background: white;
            }
        """)


MAROON = "#6C1D45"
INFO_BLUE = "#3E6990"
QUESTION_PURPLE = "#3F3158"
PROJECT_GREEN = "#184F35"
CHARCOAL = "#3D3D3D"
SOFT_GRAY = "#D3D3D3"
GOLD = "#A9823A"
PALE_BLUE = "#EDF3F8"
PALE_GREEN = "#EDF5F0"
PALE_GOLD = "#F6F1E7"
PALE_PURPLE = "#F1EEF5"


MINA_COLORS = {
    "Emperor": "#6C1D45",
    "Arabian": "#3E6990",
    "Iroquois": "#756D59",
    "Palace": "#184F35",
}

# ---------------------------------------------------------------------
# Paths and resource setup
# ---------------------------------------------------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "gutenberg_similarity_packs_curated"
PACKED_DATA = PROJECT_ROOT / "gutenberg_similarity_packs_curated.zip"


def ensure_data_dir() -> Path:
    """Return the curated Gutenberg data directory, extracting the zip if needed."""
    global DATA_DIR

    direct = DATA_DIR / "all_records.jsonl"
    if direct.exists():
        return DATA_DIR

    if PACKED_DATA.exists():
        # Prefer suite resource helper when available.
        try:
            from cobber_hum_resources import ensure_extracted_zip
            extracted = ensure_extracted_zip(
                PACKED_DATA,
                "gutenberg_similarity",
                "all_records.jsonl",
            )
            DATA_DIR = Path(extracted)
            return DATA_DIR
        except Exception:
            # Standalone fallback. Extract beside the app.
            with zipfile.ZipFile(PACKED_DATA, "r") as zf:
                zf.extractall(PROJECT_ROOT)

            direct = DATA_DIR / "all_records.jsonl"
            if direct.exists():
                return DATA_DIR

    return DATA_DIR


DATA_DIR = ensure_data_dir()
ALL_RECORDS = DATA_DIR / "all_records.jsonl"


# ---------------------------------------------------------------------
# Feature vocabularies
# ---------------------------------------------------------------------

FIRST_PERSON = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}

SECOND_PERSON = {
    "you", "your", "yours", "yourself", "yourselves",
    "thee", "thou", "thy", "thine"
}

THIRD_PERSON = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves"
}

RELIGIOUS_TERMS = {
    "god", "lord", "heaven", "hell", "soul", "sin", "faith", "prayer", "pray",
    "church", "holy", "angel", "devil", "christ", "spirit", "divine", "grace",
    "blessed", "sermon", "bible"
}

SUPERNATURAL_TERMS = {
    "ghost", "spirit", "haunted", "haunt", "apparition", "phantom", "spectre",
    "specter", "supernatural", "mystery", "terror", "fear", "strange", "shadow",
    "dead", "death", "grave", "night", "dark", "dream"
}

DETECTIVE_TERMS = {
    "detective", "clue", "murder", "crime", "criminal", "police", "inspector",
    "evidence", "case", "mystery", "suspect", "investigation", "arrest",
    "guilty", "innocent", "weapon", "footprint", "letter"
}

FAIRY_TERMS = {
    "king", "queen", "prince", "princess", "castle", "witch", "fairy", "giant",
    "dragon", "magic", "spell", "forest", "gold", "silver", "daughter", "son",
    "old", "young", "bird", "wolf"
}

ARGUMENT_TERMS = {
    "truth", "reason", "therefore", "because", "society", "government", "liberty",
    "justice", "moral", "law", "rights", "duty", "believe", "opinion", "public",
    "human", "life", "nature"
}

EMOTION_TERMS = {
    "happy", "sad", "joy", "sorrow", "fear", "afraid", "terror", "love", "hate",
    "angry", "anger", "hope", "despair", "delight", "grief", "weep", "smile",
    "laugh", "cry"
}

DIALOGUE_MARKERS = {
    "said", "asked", "answered", "replied", "cried", "whispered", "shouted",
    "spoke", "told", "called"
}

STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "it", "is", "was", "he", "for",
    "with", "as", "his", "on", "be", "at", "by", "i", "this", "had", "not", "are",
    "but", "from", "or", "have", "an", "they", "which", "one", "you", "were", "her",
    "all", "she", "there", "would", "their", "we", "him", "been", "has", "when",
    "who", "will", "more", "no", "if", "out", "so", "said", "what", "up", "its",
    "about", "into", "than", "them", "can", "could", "my", "me"
}

MINA_WORDS = ["must", "not", "shall", "should", "no"]


MATRIX_TITLE_ALIASES = [
    ("The Book of the Thousand Nights and a Night", "Thousand Nights"),
    ("Russian Fairy Tales", "Russian Fairy Tales"),
    ("Andersen's Fairy Tales", "Andersen"),
    ("The Arabian Nights", "Arabian Nights"),
    ("The Olive Fairy Book", "Olive Fairy Book"),
    ("The Indian Fairy Book", "Indian Fairy Book"),
    ("The Children of Odin", "Children of Odin"),
    ("Tales of Folk and Fairies", "Folk and Fairies"),
    ("Stories the Iroquois Tell Their Children", "Iroquois Stories"),
    ("Jewish Fairy Tales and Legends", "Jewish Fairy Tales"),
    ("A Modest Proposal", "A Modest Proposal"),
    ("Walden", "Walden"),
    ("The Souls of Black Folk", "Souls of Black Folk"),
    ("De Profundis", "De Profundis"),
    ("The Will to Believe", "Will to Believe"),
    ("Anarchism and Other Essays", "Anarchism"),
    ("Essays by Ralph Waldo Emerson", "Emerson Essays"),
    ("Pascal's Pensées", "Pascal"),
    ("Essays of Michel de Montaigne", "Montaigne"),
    ("Plutarch's Morals", "Plutarch"),
]


def matrix_title(record: "TextRecord") -> str:
    clean = re.sub(r"\s+", " ", record.title).strip()
    for prefix, label in MATRIX_TITLE_ALIASES:
        if clean.startswith(prefix):
            return label
    return short_title(clean, 24)


def content_word_counter(record: "TextRecord") -> Counter:
    return Counter(
        token
        for token in record.tokens
        if token not in STOPWORDS and len(token) > 2
    )


def shared_vocabulary_examples(
    a: "TextRecord",
    b: "TextRecord",
    n: int = 12,
) -> List[str]:
    ca = content_word_counter(a)
    cb = content_word_counter(b)
    shared = set(ca) & set(cb)
    ranked = sorted(
        shared,
        key=lambda word: (-(ca[word] + cb[word]), word),
    )
    return ranked[:n]


def unique_word_examples(
    a: "TextRecord",
    b: "TextRecord",
    n: int = 6,
) -> Tuple[List[str], List[str]]:
    ca = content_word_counter(a)
    cb = content_word_counter(b)
    only_a = sorted(
        set(ca) - set(cb),
        key=lambda word: (-ca[word], word),
    )[:n]
    only_b = sorted(
        set(cb) - set(ca),
        key=lambda word: (-cb[word], word),
    )[:n]
    return only_a, only_b


def tfidf_pair_terms(
    a: "TextRecord",
    b: "TextRecord",
    records: List["TextRecord"],
    n: int = 12,
) -> List[str]:
    counters = [content_word_counter(record) for record in records]
    n_docs = max(len(counters), 1)

    df = Counter()
    for counter in counters:
        for word in counter:
            df[word] += 1

    ca = content_word_counter(a)
    cb = content_word_counter(b)
    total_a = sum(ca.values()) or 1
    total_b = sum(cb.values()) or 1

    contributions = []
    for word in set(ca) & set(cb):
        if df[word] < 2:
            continue
        idf = math.log((1 + n_docs) / (1 + df[word])) + 1.0
        wa = (ca[word] / total_a) * idf
        wb = (cb[word] / total_b) * idf
        contributions.append((wa * wb, word))

    contributions.sort(key=lambda item: (-item[0], item[1]))
    return [word for _, word in contributions[:n]]


TOPIC_GROUPS = {
    "Religious": RELIGIOUS_TERMS,
    "Supernatural": SUPERNATURAL_TERMS,
    "Detective": DETECTIVE_TERMS,
    "Fairy-tale": FAIRY_TERMS,
    "Argument": ARGUMENT_TERMS,
    "Emotion": EMOTION_TERMS,
}

PACK_TOPIC_GROUP = {
    "fairy_folk_tales": "Fairy-tale",
    "grimm_fairy_tales": "Fairy-tale",
    "ghost_supernatural": "Supernatural",
    "detective_mystery": "Detective",
    "essays_speeches": "Argument",
}

def topic_group_for_pack(pack_key: str) -> Tuple[str, set[str]] | None:
    group_name = PACK_TOPIC_GROUP.get(pack_key)
    if not group_name:
        return None
    return group_name, TOPIC_GROUPS[group_name]

def format_number(value: float, feature_name: str = "") -> str:
    if feature_name in {"Word count", "Sentence count"}:
        return f"{value:,.0f}"
    if feature_name == "Unique word ratio":
        return f"{value:.3f}"
    return f"{value:.2f}"

def highlight_terms(text: str, terms: set[str]) -> str:
    excerpt = html.escape(short_excerpt(text))
    if not terms:
        return excerpt
    pattern = r"\b(" + "|".join(sorted((re.escape(term) for term in terms), key=len, reverse=True)) + r")\b"
    return re.sub(pattern, r'<span style="font-weight:700; color:#6C1D45;">\1</span>', excerpt, flags=re.IGNORECASE)

def criterion_highlight_terms(criterion: str) -> set[str]:
    if criterion == "Topic-word profile":
        terms: set[str] = set()
        for group_terms in TOPIC_GROUPS.values():
            terms |= group_terms
        return terms
    if criterion == "Voice profile":
        return FIRST_PERSON | SECOND_PERSON | THIRD_PERSON | DIALOGUE_MARKERS
    return set()


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def sentence_count(text: str) -> int:
    count = len(re.findall(r"[.!?]+", text))
    return max(count, 1)


def count_terms(tokens: List[str], terms: set[str]) -> int:
    return sum(1 for token in tokens if token in terms)


def per_1000(count: float, word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    return float(count) / float(word_count) * 1000.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def euclidean_distances(X: np.ndarray, standardize: bool = False) -> np.ndarray:
    """
    Calculate all pairwise Euclidean distances.

    standardize=False is appropriate when the selected features are already
    expressed in the same meaningful unit, as in Mina's word frequencies
    per 1,000 words.

    standardize=True is useful when a prepared profile combines unlike
    numerical scales.
    """
    if X.size == 0:
        return np.zeros((0, 0))

    Z = X.astype(float).copy()

    if standardize and Z.shape[1] > 0:
        means = Z.mean(axis=0)
        stds = Z.std(axis=0)
        keep = stds > 0
        if np.any(keep):
            Z = (Z[:, keep] - means[keep]) / stds[keep]
        else:
            Z = np.zeros((Z.shape[0], 1), dtype=float)

    n = Z.shape[0]
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            distances[i, j] = float(np.linalg.norm(Z[i] - Z[j]))
    return distances


def distance_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Rescale a distance matrix to similarity values from 0 to 1.

    The closest pair receives a larger value; the largest observed distance
    becomes 0. Diagonal self-comparisons remain 1.
    """
    if distances.size == 0:
        return np.zeros((0, 0))

    max_d = float(distances.max())
    if max_d <= 0:
        return np.ones_like(distances, dtype=float)

    sim = 1.0 - (distances / max_d)
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, 0.0, 1.0)


def euclidean_similarity_matrix(
    X: np.ndarray,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    distances = euclidean_distances(X, standardize=standardize)
    return distance_to_similarity(distances), distances


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    similarity = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            similarity[i, j] = cosine(X[i], X[j])
    return np.clip(similarity, 0.0, 1.0)


def jaccard_similarity_matrix(word_sets: List[set[str]]) -> np.ndarray:
    n = len(word_sets)
    similarity = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            union = word_sets[i] | word_sets[j]
            if not union:
                similarity[i, j] = 0.0
            else:
                similarity[i, j] = len(word_sets[i] & word_sets[j]) / len(union)
    return similarity


def tfidf_similarity_matrix(
    token_lists: List[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    document_counters = []
    document_frequency = Counter()

    for toks in token_lists:
        filtered = [t for t in toks if t not in STOPWORDS and len(t) > 2]
        counter = Counter(filtered)
        document_counters.append(counter)
        document_frequency.update(counter.keys())

    vocab_items = document_frequency.most_common(500)
    vocab = [word for word, df in vocab_items if df >= 2]

    if not vocab:
        return np.eye(len(token_lists)), []

    word_to_i = {word: i for i, word in enumerate(vocab)}
    n_docs = len(token_lists)
    X = np.zeros((n_docs, len(vocab)), dtype=float)

    df = Counter()
    for counter in document_counters:
        for word in counter:
            if word in word_to_i:
                df[word] += 1

    for d, counter in enumerate(document_counters):
        total = sum(counter.values()) or 1
        for word, count in counter.items():
            if word not in word_to_i:
                continue
            tf = count / total
            idf = math.log((1 + n_docs) / (1 + df[word])) + 1.0
            X[d, word_to_i[word]] = tf * idf

    return cosine_similarity_matrix(X), vocab


def short_excerpt(text: str, max_chars: int = 3000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def mina_feature_excerpt(
    text: str,
    selected_features: List[str],
    max_chars: int = 3000,
) -> str:
    """
    Return a readable excerpt that includes at least one occurrence of
    every selected feature word that actually occurs in the text.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    # Find every occurrence of each selected feature.
    occurrences = {}
    for feature in selected_features:
        matches = [
            match.start()
            for match in re.finditer(
                rf"\b{re.escape(feature)}\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
        if matches:
            occurrences[feature] = matches

    # If none of the selected words occurs, keep the ordinary preview.
    if not occurrences:
        return short_excerpt(text, max_chars)

    # Find the tightest span containing at least one occurrence
    # of every selected feature that is actually present.
    events = []
    for feature, positions in occurrences.items():
        for position in positions:
            events.append((position, feature))

    events.sort()

    needed = len(occurrences)
    counts = Counter()
    have = 0
    left = 0
    best_start = 0
    best_end = len(text)

    for right, (right_pos, right_feature) in enumerate(events):
        counts[right_feature] += 1
        if counts[right_feature] == 1:
            have += 1

        while have == needed:
            left_pos, left_feature = events[left]

            if right_pos - left_pos < best_end - best_start:
                best_start = left_pos
                best_end = right_pos

            counts[left_feature] -= 1
            if counts[left_feature] == 0:
                have -= 1

            left += 1

    # Give the matching words surrounding context.
    span_length = best_end - best_start

    if span_length < max_chars:
        extra = max_chars - span_length
        start = max(0, best_start - extra // 2)
        end = min(len(text), start + max_chars)

        # If we hit the end of the text, shift backward.
        start = max(0, end - max_chars)
    else:
        # Rare case: selected words are farther apart than 3,000 characters.
        # Expand enough to include them all.
        start = max(0, best_start - 200)
        end = min(len(text), best_end + 200)

    excerpt = text[start:end]

    # Avoid cutting through words.
    if start > 0:
        first_space = excerpt.find(" ")
        if first_space != -1:
            excerpt = excerpt[first_space + 1:]
        excerpt = "... " + excerpt

    if end < len(text):
        last_space = excerpt.rfind(" ")
        if last_space != -1:
            excerpt = excerpt[:last_space]
        excerpt += " ..."

    return excerpt


def short_title(title: str, n: int = 36) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) <= n:
        return title
    return title[: n - 3].rstrip() + "..."


# ---------------------------------------------------------------------
# Text records
# ---------------------------------------------------------------------

class TextRecord:
    def __init__(self, record: Dict[str, Any], data_dir: Path):
        self.record = record
        self.data_dir = data_dir
        self.doc_id = str(record.get("doc_id", ""))
        self.pack_key = str(record.get("pack_key", ""))
        self.pack_label = str(record.get("pack_label", ""))
        self.title = str(record.get("title", "Untitled"))
        self.author = str(record.get("author", "") or "")
        self.source_title = str(record.get("source_title", "") or "")
        self.gutenberg_id = str(record.get("gutenberg_id", "") or "")
        self.gutenberg_url = str(record.get("gutenberg_url", "") or "")
        self.text_file = str(record.get("text_file", "") or "")
        self.curation_note = str(record.get("curation_note", "") or "")
        self.text = self._load_text()
        self.tokens = tokenize(self.text)
        self.features = self._compute_features()

    def _load_text(self) -> str:
        path = self.data_dir / self.text_file
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def display_name(self) -> str:
        author = f" — {self.author}" if self.author and self.author.lower() != "nan" else ""
        return f"{self.title}{author}"

    def _compute_features(self) -> Dict[str, float]:
        text = self.text
        tokens = self.tokens
        wc = len(tokens)
        sc = sentence_count(text)
        unique = len(set(tokens))
        quotes = (
            text.count('"')
            + text.count("“")
            + text.count("”")
            + text.count("‘")
            + text.count("’")
        )

        return {
            # Scale/surface features retain their natural units.
            "Word count": float(wc),
            "Sentence count": float(sc),
            "Avg sentence length": float(wc / sc if sc else wc),
            "Unique word ratio": float(unique / wc if wc else 0.0),

            # Count-based linguistic features are frequencies per 1,000 words.
            "First-person pronouns / 1,000": per_1000(
                count_terms(tokens, FIRST_PERSON), wc
            ),
            "Second-person pronouns / 1,000": per_1000(
                count_terms(tokens, SECOND_PERSON), wc
            ),
            "Third-person pronouns / 1,000": per_1000(
                count_terms(tokens, THIRD_PERSON), wc
            ),
            "Religious terms / 1,000": per_1000(
                count_terms(tokens, RELIGIOUS_TERMS), wc
            ),
            "Supernatural terms / 1,000": per_1000(
                count_terms(tokens, SUPERNATURAL_TERMS), wc
            ),
            "Detective terms / 1,000": per_1000(
                count_terms(tokens, DETECTIVE_TERMS), wc
            ),
            "Fairy-tale terms / 1,000": per_1000(
                count_terms(tokens, FAIRY_TERMS), wc
            ),
            "Argument terms / 1,000": per_1000(
                count_terms(tokens, ARGUMENT_TERMS), wc
            ),
            "Emotion terms / 1,000": per_1000(
                count_terms(tokens, EMOTION_TERMS), wc
            ),
            "Dialogue markers / 1,000": per_1000(
                count_terms(tokens, DIALOGUE_MARKERS), wc
            ),
            "Quotation marks / 1,000": per_1000(quotes, wc),
        }


@dataclass
class MinaRecord:
    key: str
    title: str
    display_title: str
    source_title: str
    author: str
    text: str
    source_path: str

    def __post_init__(self):
        self.tokens = tokenize(self.text)
        wc = len(self.tokens)
        self.features = {
            word: per_1000(self.tokens.count(word), wc)
            for word in MINA_WORDS
        }
        self.word_count = wc


def load_records() -> List[TextRecord]:
    if not ALL_RECORDS.exists():
        return []

    records: List[TextRecord] = []
    with ALL_RECORDS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(TextRecord(json.loads(line), DATA_DIR))
    return records


# ---------------------------------------------------------------------
# Mina's exact classroom texts
# ---------------------------------------------------------------------

MINA_TEXT_SPECS = [
    {
        "key": "emperor",
        "title": "The Emperor's New Clothes",
        "display_title": "Emperor",
        "source_title": "Andersen's Fairy Tales",
        "author": "Hans Christian Andersen",
        "filename": "fairy_folk_tales_03_andersen_s_fairy_tales.txt",
        "pack_rel": (
            "packs/fairy_folk_tales/texts/"
            "fairy_folk_tales_03_andersen_s_fairy_tales.txt"
        ),
        "prepared": "andersen_preprocessed.txt",
    },
    {
        "key": "arabian",
        "title": "The Talking Bird, the Singing Tree, and the Golden Water",
        "display_title": "Arabian",
        "source_title": "The Arabian Nights: Their Best-known Tales",
        "author": "",
        "filename": (
            "fairy_folk_tales_04_the_arabian_nights_"
            "their_best_known_tales.txt"
        ),
        "pack_rel": (
            "packs/fairy_folk_tales/texts/"
            "fairy_folk_tales_04_the_arabian_nights_"
            "their_best_known_tales.txt"
        ),
        "prepared": "arabian_nights_preprocessed.txt",
    },
    {
        "key": "iroquois",
        "title": "Stories the Iroquois Tell Their Children",
        "display_title": "Iroquois",
        "source_title": "Stories the Iroquois Tell Their Children",
        "author": "Mabel Powers",
        "filename": (
            "fairy_folk_tales_09_stories_the_iroquois_"
            "tell_their_children.txt"
        ),
        "pack_rel": (
            "packs/fairy_folk_tales/texts/"
            "fairy_folk_tales_09_stories_the_iroquois_"
            "tell_their_children.txt"
        ),
        "prepared": "iroquois_preprocessed.txt",
    },
    {
        "key": "palace",
        "title": "The Palace of the Eagles",
        "display_title": "Palace",
        "source_title": "Jewish Fairy Tales and Legends",
        "author": "Gertrude Landa",
        "filename": (
            "fairy_folk_tales_10_jewish_fairy_tales_and_legends.txt"
        ),
        "pack_rel": (
            "packs/fairy_folk_tales/texts/"
            "fairy_folk_tales_10_jewish_fairy_tales_and_legends.txt"
        ),
        "prepared": "jewish_fairy_tales_preprocessed.txt",
    },
]


def find_mina_text(spec: Dict[str, str]) -> Tuple[str, str]:
    """
    Load Mina's text with a transparent priority order:

    1. prepared output beside this app
    2. classroom source file beside this app
    3. corresponding curated pack file
    """
    candidates = [
        PROJECT_ROOT / spec["prepared"],
        PROJECT_ROOT / spec["filename"],
        DATA_DIR / spec["pack_rel"],
    ]

    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace"), str(path)

    return "", ""


def load_mina_records() -> List[MinaRecord]:
    records = []
    for spec in MINA_TEXT_SPECS:
        text, source_path = find_mina_text(spec)
        records.append(
            MinaRecord(
                key=spec["key"],
                title=spec["title"],
                display_title=spec["display_title"],
                source_title=spec["source_title"],
                author=spec["author"],
                text=text,
                source_path=source_path,
            )
        )
    return records


# ---------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------

def info_box(text: str, color: str = INFO_BLUE, pale: str = PALE_BLUE) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setStyleSheet(
        f"background:{pale}; border:1px solid {color}; "
        "border-radius:7px; padding:10px;"
    )
    return label


def set_table_values(
    table: QTableWidget,
    headers: List[str],
    rows: List[List[str]],
):
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))

    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            if c > 0:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            table.setItem(r, c, item)

    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setVisible(False)


# ---------------------------------------------------------------------
# Matplotlib canvases
# ---------------------------------------------------------------------

class SimilarityMatrixCanvas(FigureCanvas):
    def __init__(self, click_callback=None):
        self.figure = Figure(figsize=(6.5, 5.5), dpi=100)
        super().__init__(self.figure)
        self.click_callback = click_callback
        self.ax = self.figure.add_subplot(111)
        self.similarity = None
        self.names: List[str] = []
        self.show_values = False
        self.mpl_connect("button_press_event", self.on_click)

    def plot_matrix(
        self,
        similarity: np.ndarray,
        names: List[str],
        title: str,
        show_values: bool = False,
    ):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.similarity = similarity
        self.names = names
        self.show_values = show_values

        if similarity is None or similarity.size == 0:
            self.ax.text(
                0.5, 0.5, "No matrix to show",
                ha="center", va="center"
            )
            self.ax.set_axis_off()
            self.draw()
            return

        image = self.ax.imshow(
            similarity,
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="equal",
        )

        self.ax.set_title(title, fontsize=14, pad=12)
        self.ax.set_xticks(range(len(names)))
        self.ax.set_yticks(range(len(names)))
        self.ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
        self.ax.set_yticklabels(names, fontsize=8)

        self.ax.set_xticks(
            np.arange(-0.5, len(names), 1),
            minor=True,
        )
        self.ax.set_yticks(
            np.arange(-0.5, len(names), 1),
            minor=True,
        )
        self.ax.grid(
            which="minor",
            linewidth=0.7,
            alpha=0.45,
        )
        self.ax.tick_params(which="minor", bottom=False, left=False)

        if show_values and len(names) <= 8:
            for i in range(len(names)):
                for j in range(len(names)):
                    value = float(similarity[i, j])
                    text_color = "black" if value >= 0.55 else "white"
                    self.ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                    )

        cbar = self.figure.colorbar(
            image,
            ax=self.ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label("Similarity")

        self.figure.tight_layout()
        self.draw()

    def on_click(self, event):
        if (
            self.similarity is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        j = int(round(event.xdata))
        i = int(round(event.ydata))

        if (
            0 <= i < len(self.names)
            and 0 <= j < len(self.names)
            and self.click_callback is not None
        ):
            self.click_callback(i, j)


class FeatureSpaceCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6.5, 5.5), dpi=100)
        super().__init__(self.figure)

    def plot_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        labels: List[str],
    ):
        self.figure.clear()

        if len(feature_names) == 2:
            ax = self.figure.add_subplot(111)

            for i, label in enumerate(labels):
                ax.scatter(
                    X[i, 0],
                    X[i, 1],
                    s=90,
                    color=MINA_COLORS.get(label, CHARCOAL),
                    label=label,
                    zorder=3,
                )

            ax.set_xlabel(
                rf"Frequency of $\mathbf{{{feature_names[0]}}}$"
            )
            ax.set_ylabel(
                rf"Frequency of $\mathbf{{{feature_names[1]}}}$"
            )


            ax.set_title(
                "Four Stories in a Two-Feature Space\nper 1,000 words",
                fontsize=14,
                pad=12,
            )

            ax.grid(alpha=0.28)





        elif len(feature_names) == 3:

            ax = self.figure.add_subplot(111, projection="3d")

            for i, label in enumerate(labels):
                ax.scatter(
                    X[i, 0],
                    X[i, 1],
                    X[i, 2],
                    s=90,
                    color=MINA_COLORS.get(label, CHARCOAL),
                    zorder=3,
                )




            ax.set_xlabel(f"{feature_names[0]} / 1,000")
            ax.set_ylabel(f"{feature_names[1]} / 1,000")
            ax.set_zlabel(f"{feature_names[2]} / 1,000")
            ax.set_title(
                "Four Stories in a Three-Feature Space",
                fontsize=14,
                pad=12,
            )
            ax.view_init(elev=22, azim=55)


        else:
            ax = self.figure.add_subplot(111)
            ax.set_axis_off()
            n = len(feature_names)
            if n < 2:
                message = (
                    "Choose at least two features to build a feature space."
                )
            else:
                message = (
                    f"{n} selected features create a {n}-dimensional "
                    "feature space.\n\n"
                    "The model can still calculate distances between the "
                    "stories even though this space cannot be drawn directly."
                )
            ax.text(
                0.5,
                0.52,
                message,
                ha="center",
                va="center",
                fontsize=13,
                wrap=True,
            )

        self.figure.tight_layout()
        self.draw()


# ---------------------------------------------------------------------
# Mina's Lab
# ---------------------------------------------------------------------

class MinaPairDialog(QDialog):
    """Focused pair inspection for Mina's four-text lab."""

    def __init__(
        self,
        records: List[MinaRecord],
        similarity: np.ndarray,
        distances: np.ndarray,
        selected_features: List[str],
        i: int,
        j: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Inspect a Pair")
        self.resize(1050, 760)

        a = records[i]
        b = records[j]
        score = float(similarity[i, j])
        distance = float(distances[i, j])

        outer = QVBoxLayout(self)

        header = QLabel(
            f"<h2>{a.title} / {b.title}</h2>"
            f"<b>Distance:</b> {distance:.2f} &nbsp;&nbsp; "
            f"<b>Similarity:</b> {score:.2f}<br>"
            f"<b>Features used:</b> {', '.join(selected_features)}"
        )
        header.setWordWrap(True)
        outer.addWidget(header)

        feature_table = QTableWidget()
        rows = []
        for feature in selected_features:
            av = a.features[feature]
            bv = b.features[feature]
            rows.append([
                feature,
                f"{av:.2f}",
                f"{bv:.2f}",
            ])
        set_table_values(
            feature_table,
            ["Feature", a.display_title, b.display_title],
            rows,
        )
        feature_table.setMaximumHeight(
            min(260, 48 + 34 * max(1, len(rows)))
        )

        for row in range(feature_table.rowCount()):
            for column in range(feature_table.columnCount()):
                item = feature_table.item(row, column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        outer.addWidget(feature_table)

        text_split = QSplitter(Qt.Orientation.Horizontal)
        doc_a = QTextEdit()
        doc_a.setReadOnly(True)
        doc_b = QTextEdit()
        doc_b.setReadOnly(True)
        doc_a.setHtml(
            MinaLab.format_mina_doc(a, selected_features)
        )
        doc_b.setHtml(
            MinaLab.format_mina_doc(b, selected_features)
        )
        text_split.addWidget(doc_a)
        text_split.addWidget(doc_b)
        outer.addWidget(text_split, 1)

        prompt = info_box(
            "<b>Return to the texts.</b><br>"
            "What in these excerpts helps explain the score? What important "
            "similarities or differences are invisible to the selected features?",
            QUESTION_PURPLE,
            PALE_PURPLE,
        )
        outer.addWidget(prompt)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        outer.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


class ExplorePairDialog(QDialog):
    """Focused pair inspection for the larger exploration workspaces."""

    def __init__(
        self,
        a: TextRecord,
        b: TextRecord,
        similarity: float,
        distance: float | None,
        criterion: str,
        current_feature_names: List[str],
        standardized: bool,
        context_records: List[TextRecord] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Inspect a Pair")
        self.resize(1100, 800)

        context_records = context_records or [a, b]

        outer = QVBoxLayout(self)

        details = [
            f"<h2>{html.escape(a.title)} / {html.escape(b.title)}</h2>",
            (
                "<span style='font-size:16px;'><b>Similarity:</b></span> "
                f"<span style='font-size:22px; font-weight:700; color:#6C1D45;'>"
                f"{similarity:.2f}</span>"
            ),
        ]

        if distance is not None:
            details.append(
                "<span style='font-size:16px;'><b>Distance:</b></span> "
                f"<span style='font-size:22px; font-weight:700; color:#6C1D45;'>"
                f"{distance:.2f}</span>"
            )

        details.append(
            f"<b>Criterion:</b> {html.escape(criterion)}"
        )
        if standardized:
            details.append(
                "<b>Distance calculation:</b> feature columns were standardized "
                "before distance was measured."
            )

        header = QLabel("<br>".join(details))
        header.setWordWrap(True)
        outer.addWidget(header)

        names = list(current_feature_names)
        show_feature_table = bool(names) and not (
            names[0].startswith("TF-IDF")
            or names[0] == "word-set overlap"
        )

        if show_feature_table:
            feature_table = QTableWidget()
            rows = []
            for name in names:
                av = a.features.get(name, 0.0)
                bv = b.features.get(name, 0.0)
                rows.append([
                    name,
                    format_number(av, name),
                    format_number(bv, name),
                ])

            set_table_values(
                feature_table,
                ["Feature", "Text A", "Text B"],
                rows,
            )

            for row in range(feature_table.rowCount()):
                for column in range(feature_table.columnCount()):
                    item = feature_table.item(row, column)
                    if item is not None:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            feature_table.setMaximumHeight(
                min(330, 48 + 31 * max(1, len(rows)))
            )
            outer.addWidget(feature_table)

        highlight_a: set[str] = set()
        highlight_b: set[str] = set()
        evidence_html = ""

        if criterion == "Length profile":
            only_a, only_b = unique_word_examples(a, b)
            a_words = ", ".join(only_a) if only_a else "none"
            b_words = ", ".join(only_b) if only_b else "none"
            evidence_html = (
                "<b>A few words unique to each story:</b><br>"
                f"<b>{html.escape(matrix_title(a))}:</b> {html.escape(a_words)}<br>"
                f"<b>{html.escape(matrix_title(b))}:</b> {html.escape(b_words)}<br>"
                "<span style='font-size:12px;'>"
            )

        elif criterion == "Voice profile":
            voice_terms = FIRST_PERSON | SECOND_PERSON | THIRD_PERSON | DIALOGUE_MARKERS
            highlight_a = voice_terms
            highlight_b = voice_terms
            evidence_html = (
                "<b>Words visible in the voice profile</b><br>"
                "Pronouns and dialogue markers used by the profile are bolded "
                "in Cobber maroon below."
            )


        elif criterion == "Topic-word profile":

            all_topic_terms = set()

            for terms in TOPIC_GROUPS.values():
                all_topic_terms |= terms

            highlight_a = all_topic_terms

            highlight_b = all_topic_terms

            evidence_html = (

                "<b>Topic-word profile</b><br>"

                "This comparison uses six prepared word groups: religious, "

                "supernatural, detective, fairy-tale, argument, and emotion terms.<br>"

                "<span style='font-size:12px;'>Words from any of these six groups "

                "are bolded below.</span>"

            )

        elif criterion == "Vocabulary overlap (Jaccard)":
            shared = shared_vocabulary_examples(a, b, n=12)
            highlight_a = set(shared)
            highlight_b = set(shared)
            shared_text = ", ".join(shared) if shared else "No shared non-common words found."
            evidence_html = (
                "<b>Examples of shared vocabulary</b><br>"
                f"{html.escape(shared_text)}<br>"
                "<span style='font-size:12px;'>Jaccard compares the full sets of "
                "non-common words. These examples are bolded below so some of "
                "the overlap is visible in the texts.</span>"
            )

        elif criterion == "TF-IDF word similarity":
            shared = tfidf_pair_terms(a, b, context_records, n=12)
            highlight_a = set(shared)
            highlight_b = set(shared)
            shared_text = ", ".join(shared) if shared else "No strong shared weighted words found."
            evidence_html = (
                "<b>Strong shared weighted words</b><br>"
                f"{html.escape(shared_text)}<br>"
                "<span style='font-size:12px;'>These words make some of the strongest "
                "pairwise contributions to the weighted vocabulary comparison and "
                "are bolded below.</span>"
            )

        if evidence_html:
            outer.addWidget(info_box(evidence_html, GOLD, PALE_GOLD))

        text_split = QSplitter(Qt.Orientation.Horizontal)
        doc_a = QTextEdit()
        doc_a.setReadOnly(True)
        doc_b = QTextEdit()
        doc_b.setReadOnly(True)

        doc_a.setHtml(
            ExploreSimilarity.format_doc_html(a, highlight_a)
        )
        doc_b.setHtml(
            ExploreSimilarity.format_doc_html(b, highlight_b)
        )

        text_split.addWidget(doc_a)
        text_split.addWidget(doc_b)
        outer.addWidget(text_split, 1)

        prompt = info_box(
            "<b>Return to the texts.</b><br>"
            "What in these excerpts helps explain the similarity? "
            "What does this representation make visible, and what does it miss?",
            QUESTION_PURPLE,
            PALE_PURPLE,
        )
        outer.addWidget(prompt)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        outer.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


# ---------------------------------------------------------------------
# Mina's Lab
# ---------------------------------------------------------------------

class MinaLab(QWidget):
    """
    A compact, staged workspace.

    The left sidebar holds only the choices and the small feature table.
    The large right panel moves through Feature Space -> Distances -> Matrix.
    Pair inspection opens in a dialog instead of consuming permanent screen space.
    """

    def __init__(self, parent_app: "CobberHumSimilarApp"):
        super().__init__()
        self.parent_app = parent_app
        self.records = load_mina_records()
        self.selected_features = ["must", "not"]
        self.X = np.zeros((0, 0))
        self.distances = np.zeros((0, 0))
        self.similarity = np.zeros((0, 0))
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        # ---------------------------
        # Compact left sidebar
        # ---------------------------
        sidebar = QWidget()
        sidebar.setMinimumWidth(500)
        sidebar.setMaximumWidth(540)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)

        heading = QLabel("<h2>Mina's Similarity Lab</h2>")
        side.addWidget(heading)

        intro = QLabel(
            "Recreate Mina's four-story comparison. Change only the prepared "
            "warning-language features and watch the representation change."
        )
        intro.setWordWrap(True)
        side.addWidget(intro)

        self.source_box = info_box(
            self.source_status_text(),
            QUESTION_PURPLE,
            PALE_PURPLE,
        )
        side.addWidget(self.source_box)

        controls = QGroupBox("Features")
        controls_layout = QVBoxLayout(controls)

        self.feature_checks: Dict[str, QCheckBox] = {}
        for word in MINA_WORDS:
            checkbox = QCheckBox(word)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.refresh)
            self.feature_checks[word] = checkbox
            controls_layout.addWidget(checkbox)

        reset = QPushButton("Clear Features")
        reset.clicked.connect(self.reset_features)
        controls_layout.addWidget(reset)
        side.addWidget(controls)

        table_group = QGroupBox("Feature Values per 1,000 Words")
        table_layout = QVBoxLayout(table_group)

        self.feature_table = QTableWidget()
        self.feature_table.setMinimumHeight(190)
        self.feature_table.setMaximumHeight(250)
        table_layout.addWidget(self.feature_table)

        legend = QLabel(
            '<span style="font-size:18px; color:#6C1D45;">●</span> '
            '<span style="font-size:14px;">Emperor</span>'
            '&nbsp;&nbsp;&nbsp;&nbsp;'
            '<span style="font-size:18px; color:#3E6990;">●</span> '
            '<span style="font-size:14px;">Arabian</span>'
            '<br>'
            '<span style="font-size:18px; color:#756D59;">●</span> '
            '<span style="font-size:14px;">Iroquois</span>'
            '&nbsp;&nbsp;&nbsp;&nbsp;'
            '<span style="font-size:18px; color:#184F35;">●</span> '
            '<span style="font-size:14px;">Palace</span>'
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setStyleSheet("padding: 7px 2px;")
        table_layout.addWidget(legend)

        side.addWidget(table_group)


        side.addStretch(1)
        outer.addWidget(sidebar, 0)

        # ---------------------------
        # Large staged display
        # ---------------------------
        self.display_tabs = QTabWidget()

        # Feature space
        feature_space_page = QWidget()
        feature_space_layout = QVBoxLayout(feature_space_page)
        feature_space_layout.setContentsMargins(6, 6, 6, 6)
        feature_space_layout.addWidget(
            QLabel("<b>Feature Space</b>")
        )
        self.space_canvas = FeatureSpaceCanvas()
        feature_space_layout.addWidget(self.space_canvas, 1)
        self.display_tabs.addTab(feature_space_page, "1. Feature Space")

        # Distances
        distance_page = QWidget()
        distance_layout = QVBoxLayout(distance_page)
        distance_layout.setContentsMargins(10, 10, 10, 10)
        distance_layout.addWidget(
            QLabel("<h3>Distances and Normalized Similarity</h3>")
        )
        self.distance_table = QTableWidget()
        distance_layout.addWidget(self.distance_table, 1)
        self.normalization_note = info_box(
            "Distances are calculated directly from Mina's selected feature "
            "frequencies because they all use the same unit: frequency per "
            "1,000 words. The largest observed distance is then rescaled to "
            "a similarity of 0.",
            GOLD,
            PALE_GOLD,
        )
        distance_layout.addWidget(self.normalization_note)
        self.display_tabs.addTab(distance_page, "2. Distances")

        # Matrix
        matrix_page = QWidget()
        matrix_layout = QVBoxLayout(matrix_page)
        matrix_layout.setContentsMargins(6, 6, 6, 6)
        matrix_layout.addWidget(
            QLabel(
                "<b>Similarity Matrix</b> &nbsp; "
                "<span style='font-weight:normal;'>Click a square to inspect the pair.</span>"
            )
        )
        self.matrix_canvas = SimilarityMatrixCanvas(self.select_pair)
        matrix_layout.addWidget(self.matrix_canvas, 1)
        self.display_tabs.addTab(matrix_page, "3. Similarity Matrix")

        outer.addWidget(self.display_tabs, 1)

    def source_status_text(self) -> str:
        missing = [r for r in self.records if not r.text]

        if missing:
            names = ", ".join(r.display_title for r in missing)
            return (
                "<b>Some of Mina's preprocessed files are missing.</b><br>"
                f"Missing: {names}."
            )

        return "<b>Mina's four preprocessed files are loaded.</b>"

        prepared = sum(
            1 for r in self.records
            if Path(r.source_path).name.endswith("_preprocessed.txt")
        )

        if prepared == 4:
            source = "the four exported preprocessing files"
        elif prepared:
            source = (
                f"{prepared} exported preprocessing file(s) plus available "
                "classroom/curated source files"
            )
        else:
            source = "the available classroom/curated source files"

        return (
            f"<b>Four texts loaded from {source}.</b>"
        )

    def reset_features(self):
        for checkbox in self.feature_checks.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self.refresh()

    def current_features(self) -> List[str]:
        return [
            word
            for word in MINA_WORDS
            if self.feature_checks[word].isChecked()
        ]

    def refresh(self):
        self.selected_features = self.current_features()
        labels = [r.display_title for r in self.records]

        rows = []
        for record in self.records:
            row = [record.display_title]
            for feature in self.selected_features:
                row.append(f"{record.features[feature]:.2f}")
            rows.append(row)

        set_table_values(
            self.feature_table,
            ["Excerpt"] + self.selected_features,
            rows,
        )

        for row in range(self.feature_table.rowCount()):
            for column in range(self.feature_table.columnCount()):
                item = self.feature_table.item(row, column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        if not self.selected_features:
            self.X = np.zeros((len(self.records), 0))
            self.distances = np.zeros((len(self.records), len(self.records)))
            self.similarity = np.eye(len(self.records))
            self.space_canvas.plot_features(self.X, [], labels)

            set_table_values(
                self.distance_table,
                ["Excerpt 1", "Excerpt 2", "Distance", "Similarity"],
                [],
            )

            self.matrix_canvas.plot_matrix(
                self.similarity,
                labels,
                "Mina's Similarity Matrix",
                show_values=True,
            )
            return

        self.X = np.array([
            [record.features[feature] for feature in self.selected_features]
            for record in self.records
        ], dtype=float)

        self.similarity, self.distances = euclidean_similarity_matrix(
            self.X,
            standardize=False,
        )

        self.space_canvas.plot_features(
            self.X,
            self.selected_features,
            labels,
        )



        pair_rows = []
        for i in range(len(self.records)):
            for j in range(i + 1, len(self.records)):
                pair_rows.append([
                    self.records[i].display_title,
                    self.records[j].display_title,
                    f"{self.distances[i, j]:.2f}",
                    f"{self.similarity[i, j]:.2f}",
                ])

        set_table_values(
            self.distance_table,
            ["Excerpt 1", "Excerpt 2", "Distance", "Similarity"],
            pair_rows,
        )

        for row in range(self.distance_table.rowCount()):
            for column in range(self.distance_table.columnCount()):
                item = self.distance_table.item(row, column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.matrix_canvas.plot_matrix(
            self.similarity,
            labels,
            "Mina's Similarity Matrix",
            show_values=True,
        )

    def select_pair(self, i: int, j: int):
        if (
            i >= len(self.records)
            or j >= len(self.records)
            or not self.selected_features
        ):
            return

        dialog = MinaPairDialog(
            self.records,
            self.similarity,
            self.distances,
            self.selected_features,
            i,
            j,
            parent=self,
        )
        dialog.exec()

    @staticmethod
    def format_mina_doc(
            record: MinaRecord,
            selected_features: List[str],
    ) -> str:
        excerpt = html.escape(
            mina_feature_excerpt(record.text, selected_features)
        )

        for feature in selected_features:
            excerpt = re.sub(
                rf"\b({re.escape(feature)})\b",
                r'<span style="font-weight:700; color:#6C1D45;">\1</span>',
                excerpt,
                flags=re.IGNORECASE,
            )

        author_line = (
            f"<b>Author:</b> {html.escape(record.author)}<br>"
            if record.author
            else ""
        )

        return (
            f"<b>Excerpt:</b> {html.escape(record.title)}<br>"
            f"<b>Source:</b> {html.escape(record.source_title)}<br>"
            f"{author_line}"
            f"<b>Words loaded:</b> {record.word_count:,}"
            f"<br><br>"
            f"{excerpt}"
        )


# ---------------------------------------------------------------------
# Larger exploration workspace
# ---------------------------------------------------------------------

class ExploreSimilarity(QWidget):
    """Pack-based exploration workspace."""

    def __init__(self, parent_app: "CobberHumSimilarApp"):
        super().__init__()
        self.parent_app = parent_app
        self.records = parent_app.records
        self.current_records: List[TextRecord] = []
        self.current_similarity: np.ndarray | None = None
        self.current_distances: np.ndarray | None = None
        self.current_feature_names: List[str] = []
        self.current_explanation = ""
        self.current_standardized = False
        self._build_ui()
        self.populate_packs()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        sidebar = QWidget()
        sidebar.setMinimumWidth(360)
        sidebar.setMaximumWidth(430)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)

        side.addWidget(QLabel("<h2>Explore Similarity</h2>"))
        intro = QLabel(
            "Scale up from Mina's four excerpts. Choose one curated text pack "
            "and one prepared similarity representation."
        )
        intro.setWordWrap(True)
        side.addWidget(intro)

        chooser = QGroupBox("Define the Comparison")
        chooser_layout = QVBoxLayout(chooser)
        chooser_layout.addWidget(QLabel("Text pack:"))
        self.pack_combo = QComboBox()
        chooser_layout.addWidget(self.pack_combo)
        chooser_layout.addWidget(QLabel("Similarity criterion:"))
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems([
            "Length profile",
            "Voice profile",
            "Topic-word profile",
            "Vocabulary overlap (Jaccard)",
            "TF-IDF word similarity",
            "Broad mixed profile",
        ])
        chooser_layout.addWidget(self.criterion_combo)
        self.generate_btn = QPushButton("Generate Similarity")
        self.generate_btn.clicked.connect(self.generate_matrix)
        chooser_layout.addWidget(self.generate_btn)
        side.addWidget(chooser)

        titles_group = QGroupBox("Texts in This Comparison")
        titles_layout = QVBoxLayout(titles_group)
        self.title_list = QTextEdit()
        self.title_list.setReadOnly(True)
        self.title_list.setMinimumHeight(250)
        titles_layout.addWidget(self.title_list)
        side.addWidget(titles_group, 1)
        outer.addWidget(sidebar, 0)

        self.display_tabs = QTabWidget()

        distance_page = QWidget()
        distance_layout = QVBoxLayout(distance_page)
        distance_layout.setContentsMargins(10, 10, 10, 10)
        distance_layout.addWidget(QLabel("<h3>Distances</h3>"))
        self.distance_table = QTableWidget()
        distance_layout.addWidget(self.distance_table, 1)
        self.distance_message = QLabel("")
        self.distance_message.setWordWrap(True)
        self.distance_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.distance_message.setStyleSheet("font-size:14px; padding:20px;")
        distance_layout.addWidget(self.distance_message)
        self.display_tabs.addTab(distance_page, "1. Distances")

        matrix_page = QWidget()
        matrix_outer = QHBoxLayout(matrix_page)
        matrix_outer.setContentsMargins(6, 6, 6, 6)
        matrix_outer.setSpacing(12)

        matrix_panel = QWidget()
        matrix_layout = QVBoxLayout(matrix_panel)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.addWidget(QLabel(
            "<b>Similarity Matrix</b> &nbsp; "
            "<span style='font-weight:normal;'>Click a square to inspect the pair.</span>"
        ))
        self.matrix_canvas = SimilarityMatrixCanvas(self.select_pair)
        matrix_layout.addWidget(self.matrix_canvas, 1)
        matrix_outer.addWidget(matrix_panel, 1)

        note_panel = QWidget()
        note_panel.setMinimumWidth(260)
        note_panel.setMaximumWidth(330)
        note_layout = QVBoxLayout(note_panel)
        note_layout.setContentsMargins(0, 0, 0, 0)
        note_layout.setSpacing(10)
        self.method_note = info_box("", INFO_BLUE, PALE_BLUE)
        note_layout.addWidget(self.method_note)
        self.topic_note = info_box("", GOLD, PALE_GOLD)
        self.topic_note.hide()
        note_layout.addWidget(self.topic_note)
        note_layout.addStretch(1)
        matrix_outer.addWidget(note_panel, 0)
        self.display_tabs.addTab(matrix_page, "2. Similarity Matrix")
        outer.addWidget(self.display_tabs, 1)

    def populate_packs(self):
        if not self.records:
            self.generate_btn.setEnabled(False)
            self.title_list.setPlainText("Curated Gutenberg collection not found.")
            return
        pack_labels = {}
        for record in self.records:
            pack_labels[record.pack_key] = record.pack_label
        preferred_order = [
            "fairy_folk_tales",
            "grimm_fairy_tales",
            "ghost_supernatural",
            "detective_mystery",
            "essays_speeches",
        ]
        for key in preferred_order:
            if key in pack_labels:
                count = sum(1 for record in self.records if record.pack_key == key)
                self.pack_combo.addItem(f"{pack_labels[key]} ({count})", key)
        for key, label in sorted(pack_labels.items()):
            if key not in preferred_order:
                count = sum(1 for record in self.records if record.pack_key == key)
                self.pack_combo.addItem(f"{label} ({count})", key)

        # Show the available titles, but wait for the student to generate results.
        pack_key = self.pack_combo.currentData()
        self.current_records = [
            record for record in self.records
            if record.pack_key == pack_key
        ]
        self.update_title_list()

        self.distance_table.hide()
        self.distance_message.setText(
            "Choose a text pack and similarity criterion, then click Generate Similarity."
        )
        self.distance_message.show()

        self.method_note.setText(
            "<b>Generate a comparison</b><br>"
            "The explanation for the selected representation will appear here."
        )

        self.matrix_canvas.plot_matrix(
            np.zeros((0, 0)),
            [],
            "",
            show_values=False,
        )

        self.current_records = []
        self.current_similarity = None
        self.current_distances = None

    def update_title_list(self):
        lines = []
        for record in self.current_records:
            if record.author and record.author.lower() != "nan":
                lines.append(f"{record.title}\n{record.author}")
            else:
                lines.append(record.title)
        self.title_list.setPlainText("\n\n".join(lines))

    def update_topic_note(self, pack_key: str, criterion: str):
        if criterion != "Topic-word profile":
            self.topic_note.hide()
            return

        self.topic_note.setText(
            "<b>Topic-word profile</b><br>"
            "Uses six prepared word groups: religious, supernatural, "
            "detective, fairy-tale, argument, and emotion terms."
        )
        self.topic_note.show()

    def generate_matrix(self):
        if not self.records:
            return
        pack_key = self.pack_combo.currentData()
        criterion = self.criterion_combo.currentText()
        self.current_records = [record for record in self.records if record.pack_key == pack_key]
        if not self.current_records:
            QMessageBox.warning(self, "No records", "No records were found for this pack.")
            return
        similarity, distances, feature_names, explanation, standardized = self.compute_similarity(
            self.current_records, criterion
        )
        self.current_similarity = similarity
        self.current_distances = distances
        self.current_feature_names = feature_names
        self.current_explanation = explanation
        self.current_standardized = standardized
        names = [matrix_title(record) for record in self.current_records]
        self.matrix_canvas.plot_matrix(similarity, names, criterion, show_values=False)
        self.method_note.setText(explanation)
        self.update_topic_note(pack_key, criterion)
        self.update_title_list()
        self.update_distance_view()
        self.parent_app.statusBar().showMessage(
            f"Generated {criterion} matrix for {len(self.current_records)} texts.", 5000
        )

    def update_distance_view(self):
        criterion = self.criterion_combo.currentText()
        if self.current_distances is None:
            self.distance_table.hide()
            if criterion == "Vocabulary overlap (Jaccard)":
                message = (
                    "<b>Vocabulary overlap calculates similarity directly.</b><br><br>"
                    "There is no separate distance table for this comparison."
                )
            else:
                message = (
                    "<b>TF-IDF word similarity calculates similarity directly.</b><br><br>"
                    "There is no separate distance table for this comparison."
                )
            self.distance_message.setText(message)
            self.distance_message.show()
            return
        self.distance_message.hide()
        self.distance_table.show()
        rows = []
        for i in range(len(self.current_records)):
            for j in range(i + 1, len(self.current_records)):
                rows.append([
                    matrix_title(self.current_records[i]),
                    matrix_title(self.current_records[j]),
                    f"{self.current_distances[i, j]:.2f}",
                    f"{self.current_similarity[i, j]:.2f}",
                ])
        set_table_values(
            self.distance_table,
            ["Text A", "Text B", "Distance", "Similarity"],
            rows,
        )
        for row in range(self.distance_table.rowCount()):
            for column in range(self.distance_table.columnCount()):
                item = self.distance_table.item(row, column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def compute_similarity(self, records: List[TextRecord], criterion: str) -> Tuple[np.ndarray, np.ndarray | None, List[str], str, bool]:
        if criterion == "Length profile":
            names = ["Word count", "Sentence count", "Avg sentence length", "Unique word ratio"]
            X = np.array([[record.features[name] for name in names] for record in records], dtype=float)
            sim, distances = euclidean_similarity_matrix(X, standardize=True)
            return sim, distances, names, (
                "<b>Length profile</b><br>Word count, sentence count, average sentence length, and "
                "vocabulary variety are standardized before distance is calculated because they use "
                "unlike numerical scales."
            ), True
        if criterion == "Voice profile":
            names = [
                "First-person pronouns / 1,000", "Second-person pronouns / 1,000",
                "Third-person pronouns / 1,000", "Dialogue markers / 1,000",
                "Quotation marks / 1,000",
            ]
            X = np.array([[record.features[name] for name in names] for record in records], dtype=float)
            sim, distances = euclidean_similarity_matrix(X, standardize=False)
            return sim, distances, names, (
                "<b>Voice profile</b><br>Pronouns, dialogue markers, and quotation marks are measured "
                "per 1,000 words before distance is calculated."
            ), False
        if criterion == "Topic-word profile":
            names = [
                "Religious terms / 1,000", "Supernatural terms / 1,000",
                "Detective terms / 1,000", "Fairy-tale terms / 1,000",
                "Argument terms / 1,000", "Emotion terms / 1,000",
            ]
            X = np.array([[record.features[name] for name in names] for record in records], dtype=float)
            sim, distances = euclidean_similarity_matrix(X, standardize=False)
            return sim, distances, names, (
                "<b>Topic-word profile</b><br>Prepared topic-word groups are measured per 1,000 words. "
                "The result reflects these particular word lists and their limits."
            ), False
        if criterion == "Vocabulary overlap (Jaccard)":
            # Build a vocabulary set from the most frequently used
            # non-stopwords in each text.
            top_n_words = 100

            word_sets = []

            for record in records:
                content_words = [
                    token
                    for token in record.tokens
                    if token not in STOPWORDS and len(token) > 2
                ]

                word_counts = Counter(content_words)

                most_frequent_words = {
                    word
                    for word, count in word_counts.most_common(top_n_words)
                }

                word_sets.append(most_frequent_words)

            similarity = jaccard_similarity_matrix(word_sets)

            return similarity, None, ["word-set overlap"], (
                "<b>Vocabulary overlap (Jaccard)</b><br>"
                "Common stopwords are removed first. For each text, Jaccard uses "
                "the 100 most frequently occurring remaining words. It compares "
                "the vocabulary shared by the two sets with their combined vocabulary."
            ), False
        if criterion == "TF-IDF word similarity":
            similarity, vocab = tfidf_similarity_matrix([record.tokens for record in records])
            return similarity, None, [f"TF-IDF vocabulary size: {len(vocab)}"], (
                "<b>TF-IDF word similarity</b><br>Each text becomes a weighted vocabulary profile. "
                "More distinctive shared words receive more weight, and the weighted profiles are compared."
            ), False
        names = [
            "Word count", "Sentence count", "Avg sentence length", "Unique word ratio",
            "First-person pronouns / 1,000", "Second-person pronouns / 1,000",
            "Third-person pronouns / 1,000", "Religious terms / 1,000",
            "Supernatural terms / 1,000", "Detective terms / 1,000",
            "Fairy-tale terms / 1,000", "Argument terms / 1,000",
            "Emotion terms / 1,000", "Dialogue markers / 1,000", "Quotation marks / 1,000",
        ]
        X = np.array([[record.features[name] for name in names] for record in records], dtype=float)
        sim, distances = euclidean_similarity_matrix(X, standardize=True)
        return sim, distances, names, (
            "<b>Broad mixed profile</b><br>This representation combines unlike features, so the app "
            "standardizes them before calculating distance. More features do not automatically create "
            "a better representation."
        ), True

    def select_pair(self, i: int, j: int):
        if not self.current_records or self.current_similarity is None or i >= len(self.current_records) or j >= len(self.current_records):
            return
        distance = None if self.current_distances is None else float(self.current_distances[i, j])
        dialog = ExplorePairDialog(
            self.current_records[i],
            self.current_records[j],
            float(self.current_similarity[i, j]),
            distance,
            self.criterion_combo.currentText(),
            self.current_feature_names,
            self.current_standardized,
            context_records=self.current_records,
            parent=self,
        )
        dialog.exec()

    @staticmethod
    def format_doc_html(record: TextRecord, highlight: set[str] | None = None) -> str:
        author_line = (
            f"<b>Author:</b> {html.escape(record.author)}<br>"
            if record.author and record.author.lower() != "nan" else ""
        )
        excerpt = highlight_terms(record.text, highlight or set())
        return (
            f"<b>Title:</b> {html.escape(record.title)}<br>"
            f"{author_line}"
            f"<b>Source:</b> {html.escape(record.source_title)}<br>"
            f"<b>Pack:</b> {html.escape(record.pack_label)}<br>"
            f"<b>Words in excerpt:</b> {len(record.tokens):,}<br><br>{excerpt}"
        )


class AllTextsSimilarity(QWidget):
    """Full-collection similarity view with deliberately limited choices."""

    def __init__(self, parent_app: "CobberHumSimilarApp"):
        super().__init__()
        self.parent_app = parent_app
        self.records = parent_app.records
        self.current_similarity: np.ndarray | None = None
        self.current_distances: np.ndarray | None = None
        self.current_feature_names: List[str] = []
        self.current_standardized = False
        self._build_ui()
        self.generate_matrix()

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        sidebar = QWidget()
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(360)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)
        side.addWidget(QLabel("<h2>All Texts</h2>"))
        intro = QLabel(
            "Compare the full curated collection at once. At this scale, the matrix becomes an overview. "
            "Click a square to return to the two excerpts behind that comparison."
        )
        intro.setWordWrap(True)
        side.addWidget(intro)
        chooser = QGroupBox("Similarity Criterion")
        chooser_layout = QVBoxLayout(chooser)
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems([
            "TF-IDF word similarity",
            "Broad mixed profile",
        ])
        chooser_layout.addWidget(self.criterion_combo)
        self.generate_btn = QPushButton("Generate Similarity")
        self.generate_btn.clicked.connect(self.generate_matrix)
        chooser_layout.addWidget(self.generate_btn)
        side.addWidget(chooser)
        self.method_note = info_box("", INFO_BLUE, PALE_BLUE)
        side.addWidget(self.method_note)
        side.addStretch(1)
        outer.addWidget(sidebar, 0)
        matrix_panel = QWidget()
        matrix_layout = QVBoxLayout(matrix_panel)
        matrix_layout.setContentsMargins(6, 6, 6, 6)
        matrix_layout.addWidget(QLabel(
            "<b>Similarity Matrix</b> &nbsp; "
            "<span style='font-weight:normal;'>Click any square to inspect the pair.</span>"
        ))
        self.matrix_canvas = SimilarityMatrixCanvas(self.select_pair)
        matrix_layout.addWidget(self.matrix_canvas, 1)
        outer.addWidget(matrix_panel, 1)

    def generate_matrix(self):
        if not self.records:
            self.generate_btn.setEnabled(False)
            self.method_note.setText("<b>Curated Gutenberg collection not found.</b>")
            return
        criterion = self.criterion_combo.currentText()
        similarity, distances, feature_names, explanation, standardized = ExploreSimilarity.compute_similarity(
            self, self.records, criterion
        )
        self.current_similarity = similarity
        self.current_distances = distances
        self.current_feature_names = feature_names
        self.current_standardized = standardized
        self.method_note.setText(explanation)
        blank_names = ["" for _ in self.records]
        self.matrix_canvas.plot_matrix(similarity, blank_names, criterion, show_values=False)
        self.parent_app.statusBar().showMessage(
            f"Generated {criterion} matrix for {len(self.records)} texts.", 5000
        )

    def select_pair(self, i: int, j: int):
        if self.current_similarity is None or i >= len(self.records) or j >= len(self.records):
            return
        distance = None if self.current_distances is None else float(self.current_distances[i, j])
        dialog = ExplorePairDialog(
            self.records[i],
            self.records[j],
            float(self.current_similarity[i, j]),
            distance,
            self.criterion_combo.currentText(),
            self.current_feature_names,
            self.current_standardized,
            context_records=self.records,
            parent=self,
        )
        dialog.exec()


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

class CobberHumSimilarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberHumSimilar")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)
        self.setFont(QFont("Lato", 10))
        self.cobber_maroon = QColor(108, 29, 69)

        self.records = load_records()
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(5)
        self.setCentralWidget(central)

        header = QLabel("<h1 style='margin:0;'>CobberHumSimilar</h1>")



        outer.addWidget(header)


        self.tabs = QTabWidget()
        self.mina_tab = MinaLab(self)
        self.explore_tab = ExploreSimilarity(self)
        self.all_texts_tab = AllTextsSimilarity(self)

        self.tabs.addTab(self.mina_tab, "Mina's Lab")
        self.tabs.addTab(self.explore_tab, "Explore Similarity")
        self.tabs.addTab(self.all_texts_tab, "All Texts")
        outer.addWidget(self.tabs, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")


def main() -> int:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberHumSimilarApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
