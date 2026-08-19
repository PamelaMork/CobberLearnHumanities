#!/usr/bin/env python3
"""
CobberHumPreprocessor.py

A humanities-centered preprocessing workbench for *Foundations of Machine
Learning in the Humanities*.

Design principle:
    Every preprocessing action creates a candidate version.
    The student compares it with the current accepted version, then either
    accepts it into a shared collection-wide pipeline or discards it.

The same accepted pipeline is ultimately applied to all four texts so the
exported files are prepared consistently for the similarity chapter.

This app begins with already-digitized text. It does not perform OCR/HTR.
"""

from __future__ import annotations

import json
import re
import string
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

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
            QPushButton:disabled { color: white; background: #B8A5AF; }
            QPlainTextEdit, QLineEdit, QComboBox {
                border: 1px solid #C8C8C8;
                border-radius: 5px;
                background: white;
                padding: 5px;
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

BOOK = "Foundations of Machine Learning in the Humanities"

SPACY_NLP = None


def get_spacy_nlp():
    """Load spaCy's small English pipeline only when lemmatization is used."""
    global SPACY_NLP
    if SPACY_NLP is not None:
        return SPACY_NLP

    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "Lemmatization requires spaCy.\n\n"
            "Install it in this project environment with:\n"
            "python -m pip install spacy\n"
            "python -m spacy download en_core_web_sm"
        ) from exc

    try:
        SPACY_NLP = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy is installed, but the English language model is missing.\n\n"
            "Install it with:\n"
            "python -m spacy download en_core_web_sm"
        ) from exc

    return SPACY_NLP


APP_DIR = Path(__file__).resolve().parent

COLLECTION_FILES = {
    "Andersen's Fairy Tales":
        "fairy_folk_tales_03_andersen_s_fairy_tales.txt",
    "The Arabian Nights: Their Best-known Tales":
        "fairy_folk_tales_04_the_arabian_nights_their_best_known_tales.txt",
    "Stories the Iroquois Tell Their Children":
        "fairy_folk_tales_09_stories_the_iroquois_tell_their_children.txt",
    "Jewish Fairy Tales and Legends":
        "fairy_folk_tales_10_jewish_fairy_tales_and_legends.txt",
}


def load_classroom_collection() -> dict[str, str]:
    """Load Mina's four exact classroom excerpts from beside this app."""
    collection = {}
    missing = []

    for title, filename in COLLECTION_FILES.items():
        path = APP_DIR / filename
        if path.exists():
            collection[title] = path.read_text(encoding="utf-8", errors="replace")
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "CobberHumPreprocessor could not find all four classroom excerpts "
            "beside the app file.\n\nMissing:\n  - "
            + "\n  - ".join(missing)
        )

    return collection


DEFAULT_STOPWORDS = [
    "a", "an", "the", "and", "or", "but", "nor", "of", "to", "in", "on", "for",
    "with", "by", "as", "at", "is", "are", "was", "were", "be", "been", "that",
    "this", "it", "its", "we", "you", "they", "he", "she", "i", "no", "not",
    "all", "which", "then", "so", "from", "man",
]

QUOTE_MAP = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2014": "--", "\u2013": "-",
}

def tokens(text: str) -> list[str]:
    return re.findall(r"[\w'&]+", text, flags=re.UNICODE)

def op_unicode_nfc(text: str, params: dict) -> str:
    return unicodedata.normalize("NFC", text)

def op_lowercase(text: str, params: dict) -> str:
    return text.lower()

def op_normalize_quotes(text: str, params: dict) -> str:
    for bad, good in QUOTE_MAP.items():
        text = text.replace(bad, good)
    return text

def op_strip_punct(text: str, params: dict) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

def op_remove_stopwords(text: str, params: dict) -> str:
    stop = {w.lower() for w in params.get("stopwords", DEFAULT_STOPWORDS)}
    out = []
    for part in re.split(r"(\s+)", text):
        if part.strip() and part.lower() in stop:
            continue
        out.append(part)
    return "".join(out)

def op_custom_replace(text: str, params: dict) -> str:
    old = params.get("from", "")
    new = params.get("to", "")
    if not old:
        return text
    return text.replace(old, new)

def op_fix_long_s(text: str, params: dict) -> str:
    return text.replace("\u017f", "s")

def op_dehyphenate(text: str, params: dict) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

def op_collapse_spaces_preserve_paragraphs(text: str, params: dict) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return "\n".join(lines).strip()

def op_collapse_all_whitespace(text: str, params: dict) -> str:
    return re.sub(r"\s+", " ", text).strip()

def op_lemmatize(text: str, params: dict) -> str:
    """
    Lemmatize with spaCy's en_core_web_sm pipeline while preserving the
    original whitespace, including paragraph boundaries.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)

    pieces = []
    for token in doc:
        if token.is_space:
            pieces.append(token.text)
            continue

        if token.is_alpha and token.lemma_:
            replacement = token.lemma_
        else:
            replacement = token.text

        pieces.append(replacement)
        pieces.append(token.whitespace_)

    return "".join(pieces)


@dataclass
class OperationSpec:
    key: str
    label: str
    category: str
    description: str
    consequence: str
    function: Callable[[str, dict], str]
    information_reducing: bool = False
    explore_more: bool = False

OPERATIONS = {
    "unicode_nfc": OperationSpec(
        "unicode_nfc", "Unicode normalization (NFC)", "Regularize",
        "Makes canonically equivalent Unicode character sequences consistent.",
        "Usually changes digital encoding rather than visible wording.",
        op_unicode_nfc,
    ),
    "normalize_quotes": OperationSpec(
        "normalize_quotes", "Normalize quotes and dashes", "Regularize",
        "Converts curly quotation marks and typographic dashes to simpler forms.",
        "Typography distinctions are collapsed.",
        op_normalize_quotes, True,
    ),
    "custom_normalization": OperationSpec(
        "custom_normalization", "Selected normalization", "Correct or regularize",
        "Replaces one explicitly chosen form with another.",
        "The chosen difference is no longer available to later analysis.",
        op_custom_replace, True,
    ),
    "verified_correction": OperationSpec(
        "verified_correction", "Verified correction", "Correct or regularize",
        "Repairs a form that has been checked against the source.",
        "The received digital form changes; the rationale should record the evidence.",
        op_custom_replace,
    ),
    "lowercase": OperationSpec(
        "lowercase", "Lowercase", "Collapse or remove",
        "Converts all letters to lowercase.",
        "Capitalization is no longer available to later analysis.",
        op_lowercase, True,
    ),
    "strip_punct": OperationSpec(
        "strip_punct", "Remove punctuation", "Collapse or remove",
        "Removes punctuation while keeping words and whitespace.",
        "Quotation marks and sentence punctuation disappear.",
        op_strip_punct, True,
    ),
    "remove_stopwords": OperationSpec(
        "remove_stopwords", "Remove stopwords", "Collapse or remove",
        "Removes words from a selected stopword list.",
        "Removed words cannot contribute to later analysis.",
        op_remove_stopwords, True,
    ),
    "fix_long_s": OperationSpec(
        "fix_long_s", "Convert long-s (ſ → s)", "Source-specific",
        "Converts the historical long-s character to modern s.",
        "The historical long-s character (ſ), used in some older printing, becomes the modern letter s.",
        op_fix_long_s, True, True,
    ),
    "dehyphenate": OperationSpec(
        "dehyphenate", "Dehyphenate line breaks", "Source-specific",
        "Rejoins words split by hyphens at line endings.",
        "True hyphenated compounds can be joined incorrectly.",
        op_dehyphenate, True, True,
    ),
    "collapse_spaces": OperationSpec(
        "collapse_spaces", "Normalize extra spaces", "Source-specific",
        "Collapses repeated spaces while preserving line and paragraph boundaries.",
        "Spacing distinctions are reduced; paragraph structure remains.",
        op_collapse_spaces_preserve_paragraphs, True, True,
    ),
    "collapse_all_ws": OperationSpec(
        "collapse_all_ws", "Collapse all whitespace", "Source-specific",
        "Turns all whitespace, including line and paragraph breaks, into single spaces.",
        "Paragraph structure disappears and paragraph chunking may no longer work.",
        op_collapse_all_whitespace, True, True,
    ),
    "lemmatize": OperationSpec(
        "lemmatize", "Lemmatization", "Prepare for comparison",
        "Groups related word forms under a shared base form.",
        "Related forms such as walks, walking, and walked may become walk. Historical or unfamiliar language still needs review.",
        op_lemmatize, True, True,
    ),
}

WHY_TRY = {
    "lowercase": (
        "<b>Why might a humanist try this?</b><br>"
        "If capitalization itself is not part of the question, lowercasing can keep "
        "<i>Emperor</i> and <i>emperor</i> from being treated as different forms."
    ),
    "remove_stopwords": (
        "<b>Why might a humanist try this?</b><br>"
        "Very common words can dominate some comparisons. Removing them may bring "
        "less frequent vocabulary into view, but words such as <i>not</i> may still matter."
    ),
    "normalize_quotes": (
        "<b>Why might a humanist try this?</b><br>"
        "Different editions or transcriptions may use curly quotes, straight quotes, "
        "em dashes, or hyphens differently. Normalizing them can keep typography from "
        "driving a comparison when typography is not the object of study."
    ),
    "strip_punct": (
        "<b>Why might a humanist try this?</b><br>"
        "Some vocabulary comparisons focus on words rather than punctuation. Removing "
        "punctuation can simplify those comparisons, but it also removes evidence about "
        "dialogue and sentence structure."
    ),
    "custom_normalization": (
        "<b>Why might a humanist try this?</b><br>"
        "Two forms such as <i>to-day</i> and <i>today</i> may represent a difference "
        "the research question does not need to preserve."
    ),
    "verified_correction": (
        "<b>Why might a humanist try this?</b><br>"
        "Digitization can introduce text that was not present in the source. A verified "
        "correction can keep an OCR, transcription, caption, or navigation artifact from "
        "becoming evidence in the analysis."
    ),
    "unicode_nfc": (
        "<b>Why might a humanist try this?</b><br>"
        "Two characters can look identical on screen while being represented differently "
        "in a file. Unicode normalization can prevent that invisible encoding difference "
        "from becoming an accidental distinction."
    ),
    "lemmatize": (
        "<b>Why might a humanist try this?</b><br>"
        "Lemmatization can keep grammatical variation such as <i>walks</i>, "
        "<i>walking</i>, and <i>walked</i> from separating vocabulary that may serve "
        "the same role in a similarity comparison."
    ),
    "fix_long_s": (
        "<b>Why might a humanist try this?</b><br>"
        "Converting long-s can make historical spellings easier to compare with modern "
        "forms when the typography itself is not part of the research question."
    ),
    "dehyphenate": (
        "<b>Why might a humanist try this?</b><br>"
        "Line endings in scans or transcriptions can split one word into two pieces. "
        "Dehyphenation can restore the word, but true hyphenated forms need review."
    ),
    "collapse_spaces": (
        "<b>Why might a humanist try this?</b><br>"
        "Extra spaces may come from transcription or layout rather than the language "
        "being studied. This option preserves paragraph boundaries."
    ),
    "collapse_all_ws": (
        "<b>Why might a humanist try this?</b><br>"
        "Aggressive whitespace collapse can simplify text for some tasks, but it removes "
        "line and paragraph structure."
    ),
}

# Known teaching cases in Mina's four-text collection. These do not replace
# source checking; they let the workbench respond to examples students are
# likely to notice in the supplied excerpts.
VERIFIED_TEACHING_CASES = {
    "muflog": (
        False,
        "<b>Check again before accepting.</b><br><i>Muflog</i> is the name of a "
        "wise man in the story. Unfamiliarity alone is not evidence of an error."
    ),
    "eyrie": (
        False,
        "<b>Check again before accepting.</b><br><i>Eyrie</i> means an eagle's "
        "nest. It belongs to the story."
    ),
    "* * *": (
        False,
        "<b>Check again before accepting.</b><br>The stars mark words that break "
        "off in the inscription. They carry meaning in the story."
    ),
    "og, riding gaily": (
        True,
        "<b>This is a plausible correction.</b><br>This phrase belongs to stray "
        "illustration/page material rather than the surrounding story."
    ),
    "tolist": (
        True,
        "<b>This is a plausible correction.</b><br><i>ToList</i> is navigation "
        "material rather than story text."
    ),
}


@dataclass
class PipelineStep:
    key: str
    label: str
    params: dict
    rationale: str
    information_reducing: bool

    def serialize(self):
        return asdict(self)

def apply_step(text: str, step: PipelineStep) -> str:
    return OPERATIONS[step.key].function(text, step.params)

def apply_pipeline(text: str, pipeline: list[PipelineStep]) -> str:
    out = text
    for step in pipeline:
        out = apply_step(out, step)
    return out

def difference_summary(before: str, after: str) -> dict:
    return {
        "chars_before": len(before),
        "chars_after": len(after),
        "chars_delta": len(after) - len(before),
        "tokens_before": len(tokens(before)),
        "tokens_after": len(tokens(after)),
        "tokens_delta": len(tokens(after)) - len(tokens(before)),
    }

def chunk_text(text: str, mode: str = "paragraph", size: int = 100) -> list[str]:
    if mode == "whole":
        return [text]
    if mode == "sentence":
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]
    if mode == "paragraph":
        parts = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in parts if p.strip()] or ([text.strip()] if text.strip() else [])
    if mode == "fixed":
        toks = text.split()
        return [" ".join(toks[i:i + size]) for i in range(0, len(toks), size)] or [""]
    return [text]

def render_simple_markup(text: str, entities: list[dict]) -> str:
    tag_for = {"person": "persName", "place": "placeName", "date": "date"}
    out = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        tag = tag_for.get(ent["type"], "seg")
        ref = ent.get("ref", "").strip()
        ref_attr = f' ref="{ref}"' if ref else ""
        s, e = ent["start"], ent["end"]
        out = out[:s] + f"<{tag}{ref_attr}>" + out[s:e] + f"</{tag}>" + out[e:]
    return out

def launch_gui():
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
            QHBoxLayout, QPlainTextEdit, QPushButton, QLabel, QComboBox,
            QFileDialog, QGroupBox, QMessageBox, QLineEdit, QScrollArea, QFrame,
        )
    except ImportError:
        print("PyQt6 is not installed. Install it with: pip install PyQt6")
        return

    def heading(text, color=MAROON, size=18):
        lab = QLabel(text)
        lab.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: 700;")
        return lab

    def info_box(text, color=INFO_BLUE, pale=PALE_BLUE):
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet(
            f"background: {pale}; border: 1px solid {color}; "
            "border-radius: 7px; padding: 10px;"
        )
        return lab

    def colored_button(text, color):
        b = QPushButton(text)
        b.setStyleSheet(
            f"QPushButton {{background:{color}; color:white; font-weight:700; "
            "border:none; border-radius:6px; padding:9px 14px;}}"
            "QPushButton:disabled {background:#AAB8B0; color:white;}"
        )
        return b

    class Main(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("CobberHumPreprocessor")
            self.resize(1180, 790)

            self.collection = load_classroom_collection()
            self.working_title = None
            self.pipeline: list[PipelineStep] = []
            self.candidate_step: Optional[PipelineStep] = None
            self.candidate_text: Optional[str] = None
            self.previous_pipeline_order: Optional[list[PipelineStep]] = None
            self.chunk_mode = "paragraph"
            self.chunk_size = 100
            self.markup_entities: list[dict] = []
            self.accepted_markup = {}
            self.applied_collection = None

            self.tabs = QTabWidget()
            self.tabs.addTab(self._tab_start(), "Start Here")
            self.tabs.addTab(self._tab_build(), "Build a Pipeline")
            self.tabs.addTab(self._tab_pipeline(), "Pipeline")
            self.tabs.addTab(self._tab_structure(), "Structure the Text")
            self.tabs.addTab(self._tab_log(), "Log & Source Return")
            self.setCentralWidget(self.tabs)

            self._refresh_all()

        def received_text(self):
            if self.working_title is None:
                return ""
            return self.collection[self.working_title]

        def current_text(self):
            if self.working_title is None:
                return ""
            return apply_pipeline(self.received_text(), self.pipeline)

        def _refresh_all(self):
            self._refresh_start()
            self._refresh_build()
            self._refresh_pipeline()
            self._refresh_structure()
            self._refresh_log()

        def _working_changed(self, title):
            if title in self.collection:
                self.working_title = title
                self.candidate_step = None
                self.candidate_text = None
                self.markup_entities = []
                self.applied_collection = None
                self._refresh_all()

        def _tab_start(self):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(heading("Choose a text to prepare"))

            lay.addWidget(info_box(
                "<b>Collection:</b> Mina's Fairy/Folk Excerpts &nbsp;&nbsp; "
                "<b>4 texts</b><br><b>Source collection ready.</b> Choose one excerpt "
                "to use while you build a shared preprocessing pipeline."
            ))

            controls = QHBoxLayout()
            controls.addWidget(QLabel("Working text:"))
            self.start_combo = QComboBox()
            self.start_combo.addItem("Select a working text...")
            self.start_combo.addItems(self.collection.keys())
            self.start_combo.currentTextChanged.connect(self._working_changed)
            controls.addWidget(self.start_combo, 1)
            lay.addLayout(controls)

            self.start_preview = QPlainTextEdit()
            self.start_preview.setReadOnly(True)
            self.start_preview.setPlaceholderText(
                "Choose one of Mina's four excerpts to inspect."
            )
            lay.addWidget(self.start_preview, 1)

            self.start_status = info_box("", QUESTION_PURPLE, PALE_PURPLE)
            lay.addWidget(self.start_status)

            self.begin_btn = colored_button("Begin Preprocessing", MAROON)
            self.begin_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
            self.begin_btn.setEnabled(False)
            lay.addWidget(self.begin_btn, alignment=Qt.AlignmentFlag.AlignRight)
            return w

        def _refresh_start(self):
            if not hasattr(self, "start_combo"):
                return

            if self.working_title is None:
                self.start_preview.clear()
                self.start_status.setText(
                    "<b>Four-text collection ready.</b><br>"
                    "Select one excerpt above. You will inspect decisions on that text "
                    "before applying the accepted pipeline to all four."
                )
                self.begin_btn.setEnabled(False)
                return

            self.start_combo.blockSignals(True)
            self.start_combo.setCurrentText(self.working_title)
            self.start_combo.blockSignals(False)
            self.start_preview.setPlainText(self.received_text())
            words = len(re.findall(r"\b[\w’'-]+\b", self.received_text(), flags=re.UNICODE))
            self.start_status.setText(
                f"<b>Selected:</b> {self.working_title}<br>"
                f"<b>Received text:</b> about {words:,} words &nbsp;&nbsp; "
                f"<b>Current shared pipeline:</b> {len(self.pipeline)} accepted step(s)"
            )
            self.begin_btn.setEnabled(True)

        def _tab_build(self):
            w = QWidget()
            outer = QVBoxLayout(w)
            outer.addWidget(heading("Build a Pipeline"))
            self.build_scope = info_box("")
            outer.addWidget(self.build_scope)

            body = QHBoxLayout()

            chooser = QGroupBox("Choose a preprocessing action")
            chooser_lay = QVBoxLayout(chooser)

            self.action_combo = QComboBox()
            self.action_combo.addItem("Select an action to test...")
            ordered_keys = [
                "lowercase",
                "remove_stopwords",
                "lemmatize",
                "normalize_quotes",
                "strip_punct",
                "custom_normalization",
                "verified_correction",
                "unicode_nfc",
                "dehyphenate",
                "collapse_spaces",
                "fix_long_s",
                "collapse_all_ws",
            ]
            self._main_label_to_key = {
                OPERATIONS[k].label: k for k in ordered_keys
            }
            self.action_combo.addItems([OPERATIONS[k].label for k in ordered_keys])
            chooser_lay.addWidget(self.action_combo)

            self.action_desc = QLabel(
                "Select one preprocessing action. CobberHumPreprocessor will create "
                "a temporary version so you can inspect the change before accepting it."
            )
            self.action_desc.setWordWrap(True)
            chooser_lay.addWidget(self.action_desc)

            self.custom_box = QWidget()
            custom_lay = QVBoxLayout(self.custom_box)
            custom_lay.setContentsMargins(0, 0, 0, 0)
            self.custom_instruction = QLabel()
            self.custom_instruction.setWordWrap(True)
            self.custom_from = QLineEdit()
            self.custom_to = QLineEdit()
            self.custom_rationale = QLineEdit()
            custom_lay.addWidget(self.custom_instruction)
            custom_lay.addWidget(self.custom_from)
            custom_lay.addWidget(self.custom_to)
            custom_lay.addWidget(self.custom_rationale)
            chooser_lay.addWidget(self.custom_box)
            self.custom_box.hide()

            self.stopword_box = QWidget()
            stop_lay = QVBoxLayout(self.stopword_box)
            stop_lay.setContentsMargins(0, 0, 0, 0)
            self.stopword_label = QLabel(
                "Stopword list. Read it before deciding what to remove:"
            )
            self.stopword_label.setWordWrap(True)
            self.stopword_edit = QPlainTextEdit(", ".join(DEFAULT_STOPWORDS))
            self.stopword_edit.setMaximumHeight(120)
            stop_lay.addWidget(self.stopword_label)
            stop_lay.addWidget(self.stopword_edit)
            chooser_lay.addWidget(self.stopword_box)
            self.stopword_box.hide()

            self.preview_btn = colored_button("Preview This Change", MAROON)
            self.preview_btn.clicked.connect(self._generate_candidate)
            self.preview_btn.setEnabled(False)
            chooser_lay.addWidget(self.preview_btn)

            self.why_box = info_box(
                "Select an action to see why a humanist might use it.",
                INFO_BLUE, PALE_BLUE
            )
            chooser_lay.addWidget(self.why_box)
            chooser_lay.addStretch()
            body.addWidget(chooser, 1)

            compare = QGroupBox("Current accepted version  →  Candidate version")
            compare_lay = QVBoxLayout(compare)
            cols = QHBoxLayout()

            left = QVBoxLayout()
            left.addWidget(QLabel("<b>Current accepted version</b>"))
            self.current_view = QPlainTextEdit()
            self.current_view.setReadOnly(True)
            left.addWidget(self.current_view)

            right = QVBoxLayout()
            right.addWidget(QLabel("<b>Candidate version</b>"))
            self.candidate_view = QPlainTextEdit()
            self.candidate_view.setReadOnly(True)
            right.addWidget(self.candidate_view)

            cols.addLayout(left)
            cols.addLayout(right)
            compare_lay.addLayout(cols)

            self.change_note = info_box(
                "No change has been previewed yet.",
                GOLD, PALE_GOLD
            )
            compare_lay.addWidget(self.change_note)

            buttons = QHBoxLayout()
            self.accept_btn = colored_button("Accept and Pass Forward", MAROON)
            self.accept_btn.clicked.connect(self._accept_candidate)
            self.discard_btn = colored_button("Discard", MAROON)
            self.discard_btn.clicked.connect(self._discard_candidate)
            buttons.addWidget(self.accept_btn)
            buttons.addWidget(self.discard_btn)
            buttons.addStretch()
            compare_lay.addLayout(buttons)

            body.addWidget(compare, 2)
            outer.addLayout(body, 1)

            self.accepted_summary = info_box("", MAROON, PALE_GREEN)
            outer.addWidget(self.accepted_summary)

            self.action_combo.currentTextChanged.connect(self._update_action_desc)
            return w

        def _update_action_desc(self):
            label = self.action_combo.currentText()
            key = self._main_label_to_key.get(label)

            if not key:
                self.action_desc.setText(
                    "Select one preprocessing action. CobberHumPreprocessor will create "
                    "a temporary version so you can inspect the change before accepting it."
                )
                self.custom_box.hide()
                self.stopword_box.hide()
                self.preview_btn.setEnabled(False)
                self.why_box.setText(
                    "Select an action to see why a humanist might use it."
                )
                return

            op = OPERATIONS[key]
            self.action_desc.setText(f"<b>{op.description}</b><br>{op.consequence}")
            self.why_box.setText(WHY_TRY.get(key, ""))
            self.preview_btn.setEnabled(self.working_title is not None)

            self.custom_box.setVisible(key in ("custom_normalization", "verified_correction"))
            self.stopword_box.setVisible(key == "remove_stopwords")

            if key == "custom_normalization":
                self.custom_instruction.setText(
                    "<b>Selected normalization</b><br>"
                    "Choose two forms you want this analysis to treat as equivalent."
                )
                self.custom_from.setPlaceholderText("Form in current text")
                self.custom_to.setPlaceholderText("Prepared form")
                self.custom_rationale.setPlaceholderText(
                    "Why should this difference not matter for this analysis?"
                )
            elif key == "verified_correction":
                self.custom_instruction.setText(
                    "<b>Verified correction</b><br>"
                    "Change a form only when you have evidence that the received digital text is wrong."
                )
                self.custom_from.setPlaceholderText("Form in current text")
                self.custom_to.setPlaceholderText("Corrected form (leave blank to remove)")
                self.custom_rationale.setPlaceholderText(
                    "What evidence supports this correction?"
                )

        def _stopwords(self):
            raw = self.stopword_edit.toPlainText().replace("\n", ",")
            return [w.strip() for w in raw.split(",") if w.strip()]

        def _candidate_feedback(self, before, after, step):
            """Explain what changed using terms already introduced in the chapter."""
            key = step.key

            if before == after:
                if key == "unicode_nfc":
                    return (
                        "<b>No visible changes in this excerpt.</b><br>"
                        "This working text is already using a consistent Unicode representation."
                    )
                if key in ("custom_normalization", "verified_correction"):
                    form = step.params.get("from", "")
                    return (
                        "<b>No changes in this excerpt.</b><br>"
                        f"The form <i>{form}</i> was not found in the current accepted version."
                    )
                if key == "fix_long_s":
                    return (
                        "<b>No historical long-s characters were found in this working text.</b>"
                    )
                if key == "dehyphenate":
                    return (
                        "<b>No line-break hyphenations were found in this working text.</b>"
                    )
                if key == "lemmatize":
                    return (
                        "<b>Lemmatization did not change any word forms in this working text.</b>"
                    )
                if key == "collapse_spaces":
                    return (
                        "<b>No extra spaces needed normalization in this working text.</b>"
                    )
                if key == "collapse_all_ws":
                    return (
                        "<b>No whitespace changes were needed in this working text.</b>"
                    )
                return "<b>No changes in this excerpt.</b>"

            if key == "lowercase":
                changed = sum(
                    1 for a, b in zip(before, after)
                    if a != b and a.isalpha() and a.lower() == b
                )
                return (
                    f"<b>{changed} uppercase letter(s) changed to lowercase.</b><br>"
                    "Capitalization will no longer be available in those places."
                )

            if key == "strip_punct":
                punct_before = sum(
                    1 for ch in before
                    if ch in string.punctuation or ch in "“”‘’—–…"
                )
                punct_after = sum(
                    1 for ch in after
                    if ch in string.punctuation or ch in "“”‘’—–…"
                )
                removed = max(0, punct_before - punct_after)
                return (
                    f"<b>{removed} punctuation mark(s) removed.</b><br>"
                    "Punctuation is no longer available to later analysis."
                )

            if key == "remove_stopwords":
                before_words = re.findall(r"[\w']+", before, flags=re.UNICODE)
                after_words = re.findall(r"[\w']+", after, flags=re.UNICODE)
                removed = max(0, len(before_words) - len(after_words))
                return (
                    f"<b>{removed} word(s) removed.</b><br>"
                    "Check whether any removed words carry meaning needed by the research question."
                )

            if key == "lemmatize":
                before_words = re.findall(r"\b[\w’'-]+\b", before, flags=re.UNICODE)
                after_words = re.findall(r"\b[\w’'-]+\b", after, flags=re.UNICODE)
                changed = sum(
                    1 for a, b in zip(before_words, after_words)
                    if a != b
                ) + abs(len(before_words) - len(after_words))
                return (
                    f"<b>{changed} word form(s) changed by lemmatization.</b><br>"
                    "Inspect the candidate for historical, unfamiliar, or proper-name forms "
                    "before passing the change forward."
                )

            if key in ("custom_normalization", "verified_correction"):
                old = step.params.get("from", "")
                new = step.params.get("to", "")
                occurrences = before.count(old)
                verb = "normalized" if key == "custom_normalization" else "corrected"
                base = (
                    f"<b>{occurrences} occurrence(s) {verb}.</b><br>"
                    f"<i>{old}</i> → <i>{new}</i>"
                )

                if key == "verified_correction":
                    probe = old.lower().strip()
                    for pattern, (_, message) in VERIFIED_TEACHING_CASES.items():
                        if pattern in probe:
                            return base + "<br><br>" + message
                    return (
                        base
                        + "<br><br><b>Verification still matters.</b><br>"
                        "The workbench can show what your change does, but you should "
                        "check the source before treating unfamiliar text as an error."
                    )
                return base

            if key == "normalize_quotes":
                changes = sum(1 for a, b in zip(before, after) if a != b)
                return (
                    f"<b>{changes} typographic character(s) regularized.</b><br>"
                    "Curly/straight quote or dash distinctions may no longer be available."
                )

            if key == "unicode_nfc":
                return (
                    "<b>The digital character representation changed.</b><br>"
                    "The visible wording may look the same even when the underlying "
                    "Unicode representation becomes consistent."
                )

            if key == "dehyphenate":
                pattern = re.compile(r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])")
                changed = len(pattern.findall(before))
                if changed:
                    return (
                        f"<b>{changed} line-break hyphenation(s) joined.</b><br>"
                        "Check that true hyphenated compounds were not joined accidentally."
                    )
                return (
                    "<b>Line-break hyphenation changed in this working text.</b><br>"
                    "Compare the candidate before passing the change forward."
                )

            if key == "collapse_spaces":
                repeated_runs = re.findall(r"[ \t]{2,}", before)
                changed_runs = len(repeated_runs)
                chars_removed = max(0, len(before) - len(after))
                return (
                    f"<b>{changed_runs} extra-spacing run(s) normalized "
                    f"({chars_removed} spacing character(s) removed).</b><br>"
                    "Paragraph boundaries are preserved."
                )

            if key == "fix_long_s":
                changed = before.count("ſ")
                return (
                    f"<b>{changed} historical long-s character(s) changed to s.</b><br>"
                    "The historical typographic distinction is no longer available."
                )

            if key == "collapse_all_ws":
                before_runs = len(re.findall(r"\s+", before))
                after_runs = len(re.findall(r"\s+", after))
                return (
                    "<b>Whitespace was collapsed across the text.</b><br>"
                    f"{before_runs} whitespace run(s) became {after_runs}. "
                    "Paragraph structure is no longer available."
                )

            delta = len(after) - len(before)
            if delta == 0:
                return "<b>The candidate differs from the current accepted version.</b>"
            direction = "added" if delta > 0 else "removed"
            return (
                f"<b>{abs(delta)} character(s) {direction}.</b><br>"
                "Compare the two versions before deciding whether to pass this change forward."
            )

        def _make_step_from_selection(self, key):
            op = OPERATIONS[key]
            params = {}
            rationale = ""
            if key in ("custom_normalization", "verified_correction"):
                old = self.custom_from.text()
                new = self.custom_to.text()
                rationale = self.custom_rationale.text().strip()
                if not old:
                    QMessageBox.information(self, "Enter a form", "Enter the form you want to change.")
                    return None
                if not rationale:
                    QMessageBox.information(
                        self, "Add a rationale",
                        "Record why this correction or normalization belongs in the pipeline."
                    )
                    return None
                params = {"from": old, "to": new}
            if key == "remove_stopwords":
                params = {"stopwords": self._stopwords()}
                rationale = "Use the selected stopword list."
            if not rationale:
                rationale = op.description
            return PipelineStep(key, op.label, params, rationale, op.information_reducing)

        def _generate_candidate(self):
            key = self._main_label_to_key.get(self.action_combo.currentText())
            if not key:
                return
            if self.working_title is None:
                QMessageBox.information(
                    self, "Choose a working text",
                    "Return to Start Here and choose one of Mina's four excerpts first."
                )
                return

            step = self._make_step_from_selection(key)
            if step is None:
                return

            before = self.current_text()
            try:
                after = apply_step(before, step)
            except RuntimeError as exc:
                QMessageBox.warning(self, "Preprocessing tool unavailable", str(exc))
                return

            self.candidate_step = step
            self.candidate_text = after
            self.current_view.setPlainText(before)
            self.candidate_view.setPlainText(after)
            self.change_note.setText(self._candidate_feedback(before, after, step))
            self.accept_btn.setEnabled(True)
            self.discard_btn.setEnabled(True)

        def _accept_candidate(self):
            if self.candidate_step is None:
                return
            accepted_label = self.candidate_step.label
            self.pipeline.append(self.candidate_step)
            self.candidate_step = None
            self.candidate_text = None
            self.applied_collection = None
            self.action_combo.setCurrentIndex(0)
            self._refresh_all()
            self.change_note.setText(
                f"<b>{accepted_label} added to the shared pipeline.</b>"
            )

        def _discard_candidate(self):
            self.candidate_step = None
            self.candidate_text = None
            self._refresh_build()

        def _refresh_build(self):
            if not hasattr(self, "build_scope"):
                return

            if self.working_title is None:
                self.build_scope.setText(
                    "<b>No working text selected.</b><br>"
                    "Return to Start Here and choose one of Mina's four excerpts."
                )
                self.current_view.clear()
                self.candidate_view.clear()
                self.preview_btn.setEnabled(False)
                self.accept_btn.setEnabled(False)
                self.discard_btn.setEnabled(False)
                self.accepted_summary.setText(
                    "<b>Accepted shared pipeline:</b> No accepted steps yet."
                )
                return

            self.build_scope.setText(
                f"<b>Testing on:</b> {self.working_title}<br>"
                "<b>Pipeline scope:</b> accepted steps will be applied to all 4 texts."
            )
            self.current_view.setPlainText(self.current_text())

            if self.candidate_text is None:
                self.candidate_view.clear()
                self.accept_btn.setEnabled(False)
                self.discard_btn.setEnabled(False)

            n_steps = len(self.pipeline)
            if n_steps == 0:
                current_label = "Current accepted version — received text"
            elif n_steps == 1:
                current_label = "Current accepted version — after 1 pipeline step"
            else:
                current_label = f"Current accepted version — after {n_steps} pipeline steps"

            names = (
                "  →  ".join(f"{i}. {step.label}" for i, step in enumerate(self.pipeline, 1))
                if self.pipeline else "No accepted steps yet."
            )
            self.accepted_summary.setText(
                f"<b>{current_label}</b><br>"
                f"<b>Accepted shared pipeline:</b> {names}"
            )
            self._update_action_desc()

        def _tab_pipeline(self):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(heading("Inspect and Reorder the Pipeline"))
            lay.addWidget(info_box(
                "The pipeline is ordered. Use the large buttons to move a step earlier or later, "
                "then re-run the pipeline from the received text."
            ))
            self.pipeline_scroll = QScrollArea()
            self.pipeline_scroll.setWidgetResizable(True)
            self.pipeline_container = QWidget()
            self.pipeline_cards = QVBoxLayout(self.pipeline_container)
            self.pipeline_scroll.setWidget(self.pipeline_container)
            lay.addWidget(self.pipeline_scroll, 1)

            row = QHBoxLayout()
            self.compare_order_btn = colored_button("Re-run and Compare Order", MAROON)
            self.compare_order_btn.clicked.connect(self._compare_pipeline_order)
            row.addWidget(self.compare_order_btn)
            row.addStretch()
            lay.addLayout(row)

            self.order_result = info_box("", QUESTION_PURPLE, PALE_PURPLE)
            lay.addWidget(self.order_result)
            return w

        def _clear_layout(self, layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.deleteLater()
                elif child_layout is not None:
                    self._clear_layout(child_layout)

        def _refresh_pipeline(self):
            if not hasattr(self, "pipeline_cards"):
                return
            self._clear_layout(self.pipeline_cards)
            if not self.pipeline:
                self.pipeline_cards.addWidget(info_box("No accepted steps yet. Build the pipeline first."))
                self.compare_order_btn.setEnabled(False)
                self.order_result.setText("")
                return
            self.compare_order_btn.setEnabled(True)
            for i, step in enumerate(self.pipeline):
                card = QFrame()
                card.setStyleSheet(
                    f"QFrame {{background:white; border:1px solid {SOFT_GRAY}; border-radius:8px; padding:6px;}}"
                )
                row = QHBoxLayout(card)
                text = QLabel(f"<b>{i+1}. {step.label}</b><br>{step.rationale}")
                text.setWordWrap(True)
                row.addWidget(text, 1)

                earlier = colored_button("Move Earlier", MAROON)
                later = colored_button("Move Later", MAROON)
                remove = colored_button("Remove", MAROON)
                earlier.setEnabled(i > 0)
                later.setEnabled(i < len(self.pipeline) - 1)
                earlier.clicked.connect(lambda _, idx=i: self._move_step(idx, -1))
                later.clicked.connect(lambda _, idx=i: self._move_step(idx, 1))
                remove.clicked.connect(lambda _, idx=i: self._remove_step(idx))
                row.addWidget(earlier)
                row.addWidget(later)
                row.addWidget(remove)
                self.pipeline_cards.addWidget(card)
            self.pipeline_cards.addStretch()
            self.order_result.setText(
                f"<b>Current pipeline:</b> {len(self.pipeline)} accepted step(s)."
            )

        def _move_step(self, idx, delta):
            new_idx = idx + delta
            if not 0 <= new_idx < len(self.pipeline):
                return
            self.previous_pipeline_order = [PipelineStep(**s.serialize()) for s in self.pipeline]
            self.pipeline[idx], self.pipeline[new_idx] = self.pipeline[new_idx], self.pipeline[idx]
            self.applied_collection = None
            self._refresh_all()

        def _remove_step(self, idx):
            self.previous_pipeline_order = [PipelineStep(**s.serialize()) for s in self.pipeline]
            self.pipeline.pop(idx)
            self.applied_collection = None
            self._refresh_all()

        def _compare_pipeline_order(self):
            if self.previous_pipeline_order is None:
                self.order_result.setText(
                    "Move or remove a step first. Then compare the previous pipeline with the current one."
                )
                return

            received = self.received_text()
            try:
                old = apply_pipeline(received, self.previous_pipeline_order)
                new = apply_pipeline(received, self.pipeline)
            except RuntimeError as exc:
                QMessageBox.warning(self, "Preprocessing tool unavailable", str(exc))
                return

            if old == new:
                self.order_result.setText(
                    "<b>Reordering these steps did not change the prepared text.</b><br>"
                    "Some preprocessing steps produce the same result in either order."
                )
            else:
                changed_chars = sum(
                    1 for a, b in zip(old, new) if a != b
                ) + abs(len(old) - len(new))
                self.order_result.setText(
                    "<b>The reordered pipeline changed the prepared text.</b><br>"
                    f"About {changed_chars} character position(s) differ. "
                    "Return to Build a Pipeline if you want to inspect or revise the result."
                )

        def _tab_structure(self):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(heading("Structure the Text"))

            chunk_group = QGroupBox("Choose the unit of analysis")
            cg = QVBoxLayout(chunk_group)
            cg.addWidget(info_box(
                "Chunking changes what counts as one object in the analysis. "
                "The same choice will be applied to all four texts."
            ))

            row = QHBoxLayout()
            self.chunk_combo = QComboBox()
            self.chunk_combo.addItems(["whole", "sentence", "paragraph", "fixed"])
            self.chunk_combo.setCurrentText("paragraph")
            self.chunk_combo.currentTextChanged.connect(self._chunk_mode_changed)

            self.fixed_label = QLabel("Fixed words:")
            self.fixed_size_edit = QLineEdit("100")
            self.fixed_size_edit.setMaximumWidth(80)
            self.fixed_size_edit.editingFinished.connect(self._chunk_mode_changed)

            row.addWidget(QLabel("Unit:"))
            row.addWidget(self.chunk_combo)
            row.addWidget(self.fixed_label)
            row.addWidget(self.fixed_size_edit)
            row.addStretch()
            cg.addLayout(row)

            self.fixed_label.hide()
            self.fixed_size_edit.hide()

            self.chunk_summary = info_box("", MAROON, PALE_GREEN)
            cg.addWidget(self.chunk_summary)

            self.chunk_preview = QPlainTextEdit()
            self.chunk_preview.setReadOnly(True)
            cg.addWidget(self.chunk_preview)
            lay.addWidget(chunk_group, 2)

            markup_group = QGroupBox("Add markup to one object")
            mg = QVBoxLayout(markup_group)
            mg.addWidget(info_box(
                "Markup records distinctions you interpret in a particular passage. "
                "Choose one object from the working text, tag a few features, then save "
                "that markup with this source."
            ))

            object_row = QHBoxLayout()
            object_row.addWidget(QLabel("Object:"))
            self.markup_object_combo = QComboBox()
            self.markup_object_combo.currentIndexChanged.connect(self._load_markup_object)
            object_row.addWidget(self.markup_object_combo, 1)
            mg.addLayout(object_row)

            self.markup_view = QPlainTextEdit()
            mg.addWidget(self.markup_view)

            row2 = QHBoxLayout()
            self.markup_kind = QComboBox()
            self.markup_kind.addItems(["person", "place", "date"])
            self.markup_ref = QLineEdit()
            self.markup_ref.setPlaceholderText("optional shared reference")
            tag_btn = colored_button("Tag Selected Text", MAROON)
            tag_btn.clicked.connect(self._tag_selection)
            row2.addWidget(QLabel("Tag:"))
            row2.addWidget(self.markup_kind)
            row2.addWidget(self.markup_ref, 1)
            row2.addWidget(tag_btn)
            mg.addLayout(row2)

            self.markup_out = QPlainTextEdit()
            self.markup_out.setReadOnly(True)
            mg.addWidget(self.markup_out)

            markup_buttons = QHBoxLayout()
            self.accept_markup_btn = colored_button("Accept Markup", MAROON)
            self.accept_markup_btn.clicked.connect(self._accept_markup)
            self.discard_markup_btn = colored_button("Discard Markup", MAROON)
            self.discard_markup_btn.clicked.connect(self._discard_markup)
            markup_buttons.addWidget(self.accept_markup_btn)
            markup_buttons.addWidget(self.discard_markup_btn)
            markup_buttons.addStretch()
            mg.addLayout(markup_buttons)

            self.markup_status = info_box("", INFO_BLUE, PALE_BLUE)
            mg.addWidget(self.markup_status)
            lay.addWidget(markup_group, 1)
            return w

        def _chunk_mode_changed(self):
            if not hasattr(self, "chunk_combo"):
                return
            self.chunk_mode = self.chunk_combo.currentText()
            self.fixed_label.setVisible(self.chunk_mode == "fixed")
            self.fixed_size_edit.setVisible(self.chunk_mode == "fixed")
            try:
                self.chunk_size = max(1, int(self.fixed_size_edit.text()))
            except ValueError:
                self.chunk_size = 100
                self.fixed_size_edit.setText("100")
            self.applied_collection = None
            self._refresh_structure()

        def _refresh_structure(self):
            if not hasattr(self, "chunk_summary"):
                return

            if self.working_title is None:
                self.chunk_summary.setText("Choose a working text on Start Here first.")
                self.chunk_preview.clear()
                self.markup_object_combo.clear()
                self.markup_view.clear()
                self.markup_out.clear()
                return

            working_chunks = chunk_text(self.current_text(), self.chunk_mode, self.chunk_size)
            lines = []
            total = 0
            for title, received in self.collection.items():
                try:
                    prepared = apply_pipeline(received, self.pipeline)
                except RuntimeError as exc:
                    self.chunk_summary.setText(str(exc))
                    return
                n = len(chunk_text(prepared, self.chunk_mode, self.chunk_size))
                total += n
                lines.append(f"{title}: {n} object(s)")

            self.chunk_summary.setText(
                "<b>Collection result:</b><br>" + "<br>".join(lines)
                + f"<br><br><b>Total objects:</b> {total}"
            )

            preview = []
            for i, chunk in enumerate(working_chunks[:8], 1):
                preview.append(f"OBJECT {i}\n{chunk}")
            if len(working_chunks) > 8:
                preview.append(f"... {len(working_chunks) - 8} more object(s)")
            self.chunk_preview.setPlainText("\n\n--------------------\n\n".join(preview))

            current_index = self.markup_object_combo.currentIndex()
            self.markup_object_combo.blockSignals(True)
            self.markup_object_combo.clear()
            self.markup_object_combo.addItems(
                [f"Object {i}" for i in range(1, len(working_chunks) + 1)]
            )
            if working_chunks:
                self.markup_object_combo.setCurrentIndex(
                    min(max(current_index, 0), len(working_chunks) - 1)
                )
            self.markup_object_combo.blockSignals(False)
            self._load_markup_object()

        def _load_markup_object(self):
            if self.working_title is None or not hasattr(self, "markup_object_combo"):
                return
            chunks = chunk_text(self.current_text(), self.chunk_mode, self.chunk_size)
            idx = self.markup_object_combo.currentIndex()
            if not chunks or idx < 0 or idx >= len(chunks):
                self.markup_view.clear()
                self.markup_out.clear()
                return
            self.markup_entities = []
            self.markup_view.setPlainText(chunks[idx])
            saved = self.accepted_markup.get((self.working_title, idx))
            if saved:
                self.markup_out.setPlainText(saved)
                self.markup_status.setText("<b>Saved markup exists for this object.</b>")
            else:
                self.markup_out.clear()
                self.markup_status.setText("Select a span in the object and add a tag.")

        def _accept_markup(self):
            if self.working_title is None:
                return
            idx = self.markup_object_combo.currentIndex()
            rendered = self.markup_out.toPlainText().strip()
            if not rendered:
                QMessageBox.information(
                    self, "No markup to save",
                    "Tag one or more spans before accepting the markup."
                )
                return
            self.accepted_markup[(self.working_title, idx)] = rendered
            self.markup_status.setText(
                f"<b>Markup saved for Object {idx + 1} in {self.working_title}.</b>"
            )

        def _discard_markup(self):
            if self.working_title is None:
                return
            idx = self.markup_object_combo.currentIndex()
            self.accepted_markup.pop((self.working_title, idx), None)
            self.markup_entities = []
            self.markup_out.clear()
            self.markup_status.setText("Markup discarded for this object.")

        def _tag_selection(self):
            cur = self.markup_view.textCursor()
            if not cur.hasSelection():
                QMessageBox.information(self, "Select text", "Highlight a word or phrase first.")
                return
            text = self.markup_view.toPlainText()
            s, e = cur.selectionStart(), cur.selectionEnd()
            self.markup_entities.append({
                "start": s, "end": e,
                "type": self.markup_kind.currentText(),
                "text": text[s:e],
                "ref": self.markup_ref.text().strip(),
            })
            self.markup_out.setPlainText(render_simple_markup(text, self.markup_entities))

        def _tab_log(self):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(heading("Log & Source Return"))

            self.log_summary = QLabel()
            self.log_summary.setWordWrap(True)
            self.log_summary.setTextFormat(Qt.TextFormat.RichText)
            self.log_summary.setStyleSheet(
                "QLabel { background: white; border: 1px solid #D3D3D3; "
                "border-radius: 6px; padding: 12px; }"
            )
            lay.addWidget(self.log_summary)

            cols = QHBoxLayout()
            left = QVBoxLayout()
            left.addWidget(QLabel("<b>Received text</b>"))
            self.source_received = QPlainTextEdit()
            self.source_received.setReadOnly(True)
            left.addWidget(self.source_received)

            right = QVBoxLayout()
            right.addWidget(QLabel("<b>Prepared working text</b>"))
            self.source_final = QPlainTextEdit()
            self.source_final.setReadOnly(True)
            right.addWidget(self.source_final)

            cols.addLayout(left)
            cols.addLayout(right)
            lay.addLayout(cols, 1)

            self.collection_summary = info_box("", MAROON, PALE_GREEN)
            lay.addWidget(self.collection_summary)

            row = QHBoxLayout()
            revise = colored_button("Revise Pipeline", MAROON)
            revise.clicked.connect(lambda: self.tabs.setCurrentIndex(2))

            self.apply_all_btn = colored_button("Apply Pipeline to All Four Texts", MAROON)
            self.apply_all_btn.clicked.connect(self._apply_to_collection)

            self.export_btn = colored_button("Export Prepared Files", MAROON)
            self.export_btn.clicked.connect(self._export_collection)
            self.export_btn.setEnabled(False)

            row.addWidget(revise)
            row.addWidget(self.apply_all_btn)
            row.addWidget(self.export_btn)
            row.addStretch()
            lay.addLayout(row)
            return w

        def _build_human_log(self):
            parts = [
                "<h3 style='margin:0 0 8px 0;'>Your Preprocessing Record</h3>",
                f"<b>Working text inspected:</b> {self.working_title or 'none selected'}<br>",
                "<b>Collection:</b> 4 texts<br><br>",
                "<b>Shared pipeline:</b><br>",
            ]

            if not self.pipeline:
                parts.append("&nbsp;&nbsp;No text transformations accepted.<br>")
            else:
                for i, step in enumerate(self.pipeline, 1):
                    label = step.label
                    if step.key in ("custom_normalization", "verified_correction"):
                        old = step.params.get("from", "")
                        new = step.params.get("to", "")
                        label += f" ({old} → {new})"
                    parts.append(f"&nbsp;&nbsp;{i}. {label}<br>")

            unit = self.chunk_mode
            if self.chunk_mode == "fixed":
                unit += f" ({self.chunk_size} words)"

            parts.extend([
                "<br>",
                f"<b>Unit of analysis:</b> {unit}",
            ])

            if self.accepted_markup:
                parts.extend([
                    "<br>",
                    f"<b>Saved markup:</b> {len(self.accepted_markup)} annotated object(s)",
                ])

            return "".join(parts)

        def _refresh_log(self):
            if not hasattr(self, "log_summary"):
                return

            self.log_summary.setText(self._build_human_log())

            if self.working_title is None:
                self.source_received.clear()
                self.source_final.clear()
                self.collection_summary.setText(
                    "Choose a working text and build a pipeline before applying it."
                )
                self.apply_all_btn.setEnabled(False)
                self.export_btn.setEnabled(False)
                return

            self.source_received.setPlainText(self.received_text())

            try:
                prepared_working = self.current_text()
            except RuntimeError as exc:
                self.source_final.setPlainText(str(exc))
                return

            self.source_final.setPlainText(prepared_working)

            lines = []
            total_objects = 0
            for title, received in self.collection.items():
                try:
                    final = apply_pipeline(received, self.pipeline)
                except RuntimeError as exc:
                    self.collection_summary.setText(str(exc))
                    return

                received_words = len(re.findall(r"\b[\w’'-]+\b", received, flags=re.UNICODE))
                prepared_words = len(re.findall(r"\b[\w’'-]+\b", final, flags=re.UNICODE))
                n_objects = len(chunk_text(final, self.chunk_mode, self.chunk_size))
                total_objects += n_objects
                lines.append(
                    f"<b>{title}</b>: about {received_words:,} → {prepared_words:,} words; "
                    f"{n_objects} object(s)"
                )

            state = (
                "<br><br><b>Pipeline applied to all four texts. Files are ready to export.</b>"
                if self.applied_collection is not None
                else "<br><br><b>Pipeline has not yet been applied to the four-text collection.</b>"
            )
            self.collection_summary.setText(
                "<b>Collection preview</b><br>" + "<br>".join(lines)
                + f"<br><br><b>Total analysis objects:</b> {total_objects}"
                + state
            )
            self.apply_all_btn.setEnabled(True)
            self.export_btn.setEnabled(self.applied_collection is not None)

        def _apply_to_collection(self):
            if self.working_title is None:
                return
            try:
                self.applied_collection = {
                    title: apply_pipeline(received, self.pipeline)
                    for title, received in self.collection.items()
                }
            except RuntimeError as exc:
                QMessageBox.warning(self, "Preprocessing tool unavailable", str(exc))
                return

            self._refresh_log()
            QMessageBox.information(
                self,
                "Pipeline applied",
                "The shared pipeline has been applied to all four texts. "
                "Review the collection summary, then export when you are ready."
            )

        def _export_collection(self):
            if self.applied_collection is None:
                QMessageBox.information(
                    self,
                    "Apply the pipeline first",
                    "Use “Apply Pipeline to All Four Texts” before exporting."
                )
                return

            folder = QFileDialog.getExistingDirectory(
                self,
                "Choose a folder for the four prepared text files"
            )
            if not folder:
                return

            outdir = Path(folder)

            filename_map = {
                "Andersen's Fairy Tales": "andersen_preprocessed.txt",
                "The Arabian Nights: Their Best-known Tales": "arabian_nights_preprocessed.txt",
                "Stories the Iroquois Tell Their Children": "iroquois_preprocessed.txt",
                "Jewish Fairy Tales and Legends": "jewish_fairy_tales_preprocessed.txt",
            }

            collection_log = {
                "tool": "CobberHumPreprocessor",
                "book": BOOK,
                "working_text_used_for_inspection": self.working_title,
                "pipeline": [step.serialize() for step in self.pipeline],
                "chunking": {
                    "unit": self.chunk_mode,
                    "size": self.chunk_size if self.chunk_mode == "fixed" else None,
                },
                "texts": {},
                "saved_markup_objects": len(self.accepted_markup),
            }

            exported_names = []

            for title, final in self.applied_collection.items():
                received = self.collection[title]
                filename = filename_map[title]
                (outdir / filename).write_text(final, encoding="utf-8")
                exported_names.append(filename)

                received_words = len(re.findall(r"\b[\w’'-]+\b", received, flags=re.UNICODE))
                prepared_words = len(re.findall(r"\b[\w’'-]+\b", final, flags=re.UNICODE))
                collection_log["texts"][title] = {
                    "file": filename,
                    "words_received": received_words,
                    "words_prepared": prepared_words,
                    "objects": len(chunk_text(final, self.chunk_mode, self.chunk_size)),
                }

            if self.accepted_markup:
                markup_dir = outdir / "markup"
                markup_dir.mkdir(exist_ok=True)
                for (title, idx), marked in self.accepted_markup.items():
                    slug = re.sub(r"\W+", "_", title.lower()).strip("_")
                    (markup_dir / f"{slug}_object_{idx + 1}.xml").write_text(
                        marked, encoding="utf-8"
                    )

            (outdir / "preprocessing_log.json").write_text(
                json.dumps(collection_log, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            # Plain-text record for portability.
            text_lines = [
                "Your Preprocessing Record",
                "",
                f"Working text inspected: {self.working_title}",
                "Collection: 4 texts",
                "",
                "Shared pipeline:",
            ]
            if self.pipeline:
                for i, step in enumerate(self.pipeline, 1):
                    label = step.label
                    if step.key in ("custom_normalization", "verified_correction"):
                        old = step.params.get("from", "")
                        new = step.params.get("to", "")
                        label += f" ({old} -> {new})"
                    text_lines.append(f"{i}. {label}")
            else:
                text_lines.append("No text transformations accepted.")

            unit = self.chunk_mode
            if self.chunk_mode == "fixed":
                unit += f" ({self.chunk_size} words)"
            text_lines.extend(["", f"Unit of analysis: {unit}"])

            if self.accepted_markup:
                text_lines.extend([
                    "",
                    f"Saved markup: {len(self.accepted_markup)} annotated object(s)"
                ])

            (outdir / "preprocessing_record.txt").write_text(
                "\n".join(text_lines),
                encoding="utf-8"
            )

            QMessageBox.information(
                self,
                "Export complete",
                "Four separate prepared .txt files were exported, along with the "
                "preprocessing record and technical log.\n\n"
                f"Folder:\n{outdir}"
            )

    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    win = Main()
    win.show()
    sys.exit(app.exec())

def main():
    try:
        launch_gui()
    except FileNotFoundError as exc:
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Classroom excerpts missing", str(exc))
        except Exception:
            print(str(exc))

if __name__ == "__main__":
    main()
