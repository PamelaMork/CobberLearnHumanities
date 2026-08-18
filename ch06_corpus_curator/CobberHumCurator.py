#!/usr/bin/env python3
"""
CobberHumCurator.py

Companion code for *Foundations of Machine Learning in the Humanities*.

CobberHumCurator is the second stage of the humanities data bench. It helps
students move from a fetched candidate pool to a documented corpus by reviewing
one record at a time, making an explicit decision, recording a brief reason,
revisiting uncertain records, and then stepping back to inspect patterns in the
corpus they are building.

Design boundary:
- Fetcher finds and structures candidate records.
- Curator supports and documents human judgment about corpus membership.
- Preprocessor later cleans and prepares corpus contents for analysis.

Curator does NOT:
- fetch new records,
- automatically decide what belongs,
- generate a corpus statement,
- interpret patterns in the student's decisions.

The app records what the student did so the textbook's Pause and Think work can
guide the interpretation.

Run:
    python CobberHumCurator.py

Dependencies:
    pip install PyQt6

The app uses the shared Cobber Humanities stylesheet when available.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from cobber_hum_branding import apply_app_stylesheet
except ModuleNotFoundError:
    try:
        import sys as _cobber_sys
        from pathlib import Path as _CobberPath

        _cobber_root = str(_CobberPath(__file__).resolve().parents[1])
        if _cobber_root not in _cobber_sys.path:
            _cobber_sys.path.insert(0, _cobber_root)
        from cobber_hum_branding import apply_app_stylesheet
    except Exception:
        def apply_app_stylesheet(app):
            # Fallback so the app still runs as a standalone file.
            app.setStyleSheet("""
                QWidget {
                    font-family: Lato, Arial, sans-serif;
                    font-size: 10pt;
                    color: #3D3D3D;
                }
                QPushButton {
                    min-height: 32px;
                }
                QTabBar::tab {
                    padding: 8px 14px;
                }
            """)


APP_TITLE = "CobberHumCurator"
APP_VERSION = "1.0 classroom rebuild"

COBBER_MAROON = "#6C1D45"
CHARCOAL = "#3D3D3D"
SOFT_GRAY = "#D3D3D3"
MEDIUM_GRAY = "#666666"
PALE_MAROON = "#F8F0F4"

DECISIONS = ["", "Include", "Exclude", "Check Further"]

REASONS = [
    "",
    "first-person account",
    "useful performance",
    "relevant to everyday camp life",
    "contextual evidence",
    "outside date range",
    "wrong location",
    "unrelated topic",
    "duplicate / collection page",
    "needs source review",
    "unclear source type",
    "missing context",
    "other / add note",
]

CURATION_FIELDS = [
    "curation_decision",
    "curation_reason",
    "curation_note",
    "source_opened",
    "reviewed_at",
]

DISPLAY_FIELDS = [
    ("title", "Title"),
    ("date", "Date"),
    ("creator", "Creator / Contributor"),
    ("type", "Type"),
    ("language", "Language"),
    ("subjects", "Subjects"),
    ("description", "Description"),
    ("source_id", "Source ID"),
    ("url", "Source URL"),
]

TABLE_FIELDS = [
    ("title", "Title"),
    ("date", "Date"),
    ("creator", "Creator / Contributor"),
    ("type", "Type"),
    ("language", "Language"),
    ("curation_reason", "Reason"),
]

SUMMARY_FIELD_CANDIDATES = [
    ("type", "Type"),
    ("language", "Language"),
    ("date", "Date"),
    ("creator", "Creator / Contributor"),
    ("source", "Source"),
    ("mediatype", "Media type"),
]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(safe_text(v) for v in value if safe_text(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def html_escape(value: Any) -> str:
    text = safe_text(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def clean_filename(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in (value or "").strip()
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "curated_corpus"


def normalize_loaded_record(row: Dict[str, Any], index: int) -> Dict[str, str]:
    rec = {str(k): safe_text(v) for k, v in row.items()}

    # Accommodate common alternate field names without rewriting the input file.
    aliases = {
        "creator": ["creator", "author", "contributor", "contributors"],
        "type": ["type", "mediatype", "format"],
        "subjects": ["subjects", "subject"],
        "source_id": ["source_id", "id", "identifier"],
        "url": ["url", "source_url", "link"],
        "description": ["description", "summary", "notes"],
    }

    for canonical, possibilities in aliases.items():
        if rec.get(canonical):
            continue
        for field in possibilities:
            if rec.get(field):
                rec[canonical] = rec[field]
                break
        rec.setdefault(canonical, "")

    rec.setdefault("title", f"Record {index + 1}")
    rec.setdefault("date", "")
    rec.setdefault("language", "")
    rec.setdefault("source", "")
    rec.setdefault("curation_decision", "")
    rec.setdefault("curation_reason", "")
    rec.setdefault("curation_note", "")
    rec.setdefault("source_opened", "No")
    rec.setdefault("reviewed_at", "")
    return rec


def read_csv_records(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("The CSV does not contain a header row.")
        for i, row in enumerate(reader):
            rows.append(normalize_loaded_record(row, i))
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


@dataclass
class CuratorProject:
    input_csv: str = ""
    research_question: str = ""
    current_index: int = 0


class RecordDetailsDialog(QDialog):
    """Read a record closely and, optionally, revise its curation decision."""

    def __init__(
        self,
        parent: QWidget,
        record: Dict[str, str],
        on_save,
    ):
        super().__init__(parent)
        self.record = record
        self.on_save = on_save

        self.setWindowTitle("Record details")
        self.resize(790, 720)

        outer = QVBoxLayout(self)

        title = QLabel(record.get("title") or "Untitled record")
        title_font = QFont("Lato", 15)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setWordWrap(True)
        title.setStyleSheet(f"color: {COBBER_MAROON};")
        outer.addWidget(title)

        browser = QTextBrowser()
        html = ["<div style='font-family:Lato,Arial,sans-serif;font-size:10.5pt;'>"]
        for field, label in DISPLAY_FIELDS:
            value = record.get(field, "")
            if not value:
                continue
            html.append(
                f"<p style='margin:0 0 10px 0;'><b>{html_escape(label)}</b><br>"
                f"{html_escape(value)}</p>"
            )
        html.append("</div>")
        browser.setHtml("".join(html))
        outer.addWidget(browser, 1)

        source_row = QHBoxLayout()
        self.open_source_btn = QPushButton("Open source")
        self.open_source_btn.setEnabled(True)
        self.open_source_btn.clicked.connect(self._open_source)
        source_row.addWidget(self.open_source_btn)
        source_row.addStretch(1)
        outer.addLayout(source_row)

        revise_box = QGroupBox("Revise curation decision")
        form = QFormLayout(revise_box)

        self.decision_combo = QComboBox()
        self.decision_combo.addItems(DECISIONS)
        current_decision = record.get("curation_decision", "")
        idx = self.decision_combo.findText(current_decision)
        self.decision_combo.setCurrentIndex(max(0, idx))

        self.reason_combo = QComboBox()
        self.reason_combo.addItems(REASONS)
        current_reason = record.get("curation_reason", "")
        idx = self.reason_combo.findText(current_reason)
        self.reason_combo.setCurrentIndex(max(0, idx))

        self.note_edit = QLineEdit(record.get("curation_note", ""))
        self.note_edit.setPlaceholderText("Short note for 'other / add note'")

        form.addRow("Decision:", self.decision_combo)
        form.addRow("Reason:", self.reason_combo)
        form.addRow("Other note:", self.note_edit)
        outer.addWidget(revise_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _open_source(self):
        url = self.record.get("url", "").strip()
        if not url:
            QMessageBox.information(
                self,
                "Source link unavailable",
                "This record does not include a source URL."
            )
            return
        self.record["source_opened"] = "Yes"
        self.on_save(self.record)
        QDesktopServices.openUrl(QUrl(url))

    def _save(self):
        decision = self.decision_combo.currentText().strip()
        reason = self.reason_combo.currentText().strip()
        note = self.note_edit.text().strip()

        if decision and not reason:
            QMessageBox.information(
                self,
                "Reason needed",
                "Choose a reason before saving this decision."
            )
            return

        if reason == "other / add note" and not note:
            QMessageBox.information(
                self,
                "Add a short note",
                "Enter a short note for the 'other / add note' reason."
            )
            return

        self.record["curation_decision"] = decision
        self.record["curation_reason"] = reason
        self.record["curation_note"] = note
        if decision:
            self.record["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        self.on_save(self.record)


class DecisionTablePanel(QWidget):
    """A filtered scorekeeping table for one decision category."""

    def __init__(self, app, decision: str):
        super().__init__()
        self.app = app
        self.decision = decision
        self.records: List[Dict[str, str]] = []

        layout = QVBoxLayout(self)

        heading = QLabel(decision)
        heading_font = QFont("Lato", 14)
        heading_font.setWeight(QFont.Weight.Bold)
        heading.setFont(heading_font)
        heading.setStyleSheet(f"color: {COBBER_MAROON};")
        layout.addWidget(heading)

        note = QLabel(
            "This tab keeps score. Double-click any row to inspect the record, "
            "open the source, or revise the decision."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555555;")
        layout.addWidget(note)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemDoubleClicked.connect(self._show_selected)
        layout.addWidget(self.table, 1)

    def refresh(self):
        self.records = [
            rec for rec in self.app.records
            if rec.get("curation_decision") == self.decision
        ]

        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(TABLE_FIELDS))
        self.table.setHorizontalHeaderLabels([label for _, label in TABLE_FIELDS])
        self.table.setRowCount(len(self.records))

        for r, rec in enumerate(self.records):
            for c, (field, _label) in enumerate(TABLE_FIELDS):
                value = rec.get(field, "")
                if field == "curation_reason" and rec.get("curation_note"):
                    if value == "other / add note":
                        value = rec.get("curation_note")
                    else:
                        value = f"{value} — {rec.get('curation_note')}"
                item = QTableWidgetItem(value)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                )
                self.table.setItem(r, c, item)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.resizeRowsToContents()
        self.table.setSortingEnabled(True)

    def _show_selected(self, _item=None):
        row = self.table.currentRow()
        if not (0 <= row < len(self.records)):
            return
        self.app.show_record_dialog(self.records[row])


class OverviewPanel(QWidget):
    """Whole-corpus descriptive summaries. It reports patterns; it does not interpret them."""

    def __init__(self, app):
        super().__init__()
        self.app = app

        layout = QVBoxLayout(self)

        heading = QLabel("Corpus Overview")
        heading_font = QFont("Lato", 14)
        heading_font.setWeight(QFont.Weight.Bold)
        heading.setFont(heading_font)
        heading.setStyleSheet(f"color: {COBBER_MAROON};")
        layout.addWidget(heading)

        note = QLabel(
            "Step back from individual records and look at the pattern your decisions are creating. "
            "The app reports the pattern. You decide what it means."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555555;")
        layout.addWidget(note)

        self.counts_label = QLabel()
        self.counts_label.setWordWrap(True)
        self.counts_label.setStyleSheet(
            f"background:{PALE_MAROON}; border:1px solid #D8BBC9; "
            "padding:10px; font-weight:bold;"
        )
        layout.addWidget(self.counts_label)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Summarize by:"))
        self.summary_combo = QComboBox()
        self.summary_combo.currentTextChanged.connect(self._refresh_summary_table)
        control_row.addWidget(self.summary_combo)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self.summary_table = QTableWidget()
        self.summary_table.setAlternatingRowColors(True)
        layout.addWidget(self.summary_table, 1)

    def refresh(self):
        total = len(self.app.records)
        included = sum(r.get("curation_decision") == "Include" for r in self.app.records)
        excluded = sum(r.get("curation_decision") == "Exclude" for r in self.app.records)
        check = sum(r.get("curation_decision") == "Check Further" for r in self.app.records)
        not_reviewed = total - included - excluded - check

        self.counts_label.setText(
            f"Candidate pool: {total}    "
            f"Included: {included}    "
            f"Excluded: {excluded}    "
            f"Check Further: {check}    "
            f"Not reviewed: {not_reviewed}"
        )

        existing_field = self.summary_combo.currentData()
        available = []
        for field, label in SUMMARY_FIELD_CANDIDATES:
            if any(safe_text(rec.get(field)) for rec in self.app.records):
                available.append((field, label))

        self.summary_combo.blockSignals(True)
        self.summary_combo.clear()
        for field, label in available:
            self.summary_combo.addItem(label, field)

        if existing_field:
            idx = self.summary_combo.findData(existing_field)
            if idx >= 0:
                self.summary_combo.setCurrentIndex(idx)
        self.summary_combo.blockSignals(False)

        self._refresh_summary_table()

    def _summary_value(self, record: Dict[str, str], field: str) -> str:
        value = safe_text(record.get(field))
        if not value:
            return "(missing)"

        # Keep dates readable without forcing a decade interpretation.
        if field == "date":
            # If a clear 4-digit year appears, summarize by that year.
            import re
            match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", value)
            if match:
                return match.group(1)
        return value

    def _refresh_summary_table(self):
        field = self.summary_combo.currentData()
        if not field:
            self.summary_table.clear()
            self.summary_table.setRowCount(0)
            self.summary_table.setColumnCount(0)
            return

        counts: Dict[str, Dict[str, int]] = {}

        for record in self.app.records:
            category = self._summary_value(record, field)
            decision = record.get("curation_decision") or "Not Reviewed"

            # Multi-valued fields can be visually overwhelming. For the first
            # classroom version, treat the stored field value as one category.
            counts.setdefault(
                category,
                {"Include": 0, "Exclude": 0, "Check Further": 0, "Not Reviewed": 0},
            )
            counts[category][decision] += 1

        rows = sorted(
            counts.items(),
            key=lambda pair: (
                -(pair[1]["Include"] + pair[1]["Exclude"] +
                  pair[1]["Check Further"] + pair[1]["Not Reviewed"]),
                pair[0].casefold(),
            ),
        )

        self.summary_table.setSortingEnabled(False)
        self.summary_table.clear()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels(
            ["Category", "Included", "Excluded", "Check Further", "Not Reviewed"]
        )
        self.summary_table.setRowCount(len(rows))

        for r, (category, values) in enumerate(rows):
            self.summary_table.setItem(r, 0, QTableWidgetItem(category))
            self.summary_table.setItem(r, 1, QTableWidgetItem(str(values["Include"])))
            self.summary_table.setItem(r, 2, QTableWidgetItem(str(values["Exclude"])))
            self.summary_table.setItem(r, 3, QTableWidgetItem(str(values["Check Further"])))
            self.summary_table.setItem(r, 4, QTableWidgetItem(str(values["Not Reviewed"])))

        header = self.summary_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 5):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.summary_table.setSortingEnabled(True)


class CobberHumCuratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.records: List[Dict[str, str]] = []
        self.project = CuratorProject()
        self.current_index = 0
        self.current_decision = ""

        self.cobber_maroon = QColor(108, 29, 69)
        self.base_font = QFont("Lato", 10)
        self.setFont(self.base_font)

        self.setWindowTitle(APP_TITLE)
        self._set_laptop_friendly_geometry()
        self._build_ui()
        self._set_loaded_state(False)
        self.statusBar().showMessage("Ready. Load a candidate CSV from CobberHumFetcher.")

    # ------------------------------------------------------------------
    # Window and common styles
    # ------------------------------------------------------------------

    def _set_laptop_friendly_geometry(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1180, 760)
            return

        geom = screen.availableGeometry()
        width = min(1380, max(1120, int(geom.width() * 0.95)))
        height = min(820, max(680, int(geom.height() * 0.92)))
        self.resize(width, height)

        x = geom.x() + max(0, (geom.width() - width) // 2)
        y = geom.y() + max(0, (geom.height() - height) // 2)
        self.move(x, y)

    @property
    def primary_button_style(self):
        return f"""
            QPushButton {{
                background-color: {COBBER_MAROON};
                color: white;
                font-weight: bold;
                font-size: 10.5pt;
                border: none;
                border-radius: 6px;
                padding: 9px 13px;
            }}
            QPushButton:hover {{ background-color: #5A1839; }}
            QPushButton:disabled {{
                background-color: #9A7287;
                color: white;
            }}
        """

    @property
    def secondary_button_style(self):
        return """
            QPushButton {
                background-color: #666666;
                color: white;
                font-size: 10.5pt;
                border: none;
                border-radius: 6px;
                padding: 9px 13px;
            }
            QPushButton:hover { background-color: #555555; }
            QPushButton:disabled {
                background-color: #888888;
                color: white;
            }
        """

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        project_bar = self._build_project_bar()
        outer.addWidget(project_bar)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        self.review_tab = self._build_review_tab()
        self.included_tab = DecisionTablePanel(self, "Include")
        self.excluded_tab = DecisionTablePanel(self, "Exclude")
        self.check_tab = DecisionTablePanel(self, "Check Further")
        self.overview_tab = OverviewPanel(self)

        self.tabs.addTab(self.review_tab, "Review")
        self.tabs.addTab(self.included_tab, "Included")
        self.tabs.addTab(self.excluded_tab, "Excluded")
        self.tabs.addTab(self.check_tab, "Check Further")
        self.tabs.addTab(self.overview_tab, "Corpus Overview")

        self.tabs.currentChanged.connect(lambda _i: self.refresh_all_views())

    def _build_project_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        self.loaded_label = QLabel("No candidate pool loaded")
        self.loaded_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(self.loaded_label, 1)

        load_btn = QPushButton("Load Candidate Records")
        load_btn.setStyleSheet(self.secondary_button_style)
        load_btn.clicked.connect(self.load_candidate_csv)
        layout.addWidget(load_btn)

        export_btn = QPushButton("Export curated corpus")
        export_btn.setStyleSheet(self.primary_button_style)
        export_btn.clicked.connect(self.export_corpus)
        self.export_btn = export_btn
        layout.addWidget(export_btn)

        return bar

    def _build_review_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        question_box = QGroupBox("Research question")
        question_layout = QVBoxLayout(question_box)
        self.question_edit = QTextEdit()
        self.question_edit.setPlainText(
            "How did migrant workers describe everyday life in FSA camps "
            "in California in 1940–1941?"
        )
        self.question_edit.setPlaceholderText(
            "Enter the research question that will guide your corpus decisions."
        )
        self.question_edit.setMaximumHeight(90)
        self.question_edit.textChanged.connect(self._research_question_changed)
        question_layout.addWidget(self.question_edit)
        layout.addWidget(question_box)

        progress_row = QHBoxLayout()
        self.previous_btn = QPushButton("Previous")
        self.previous_btn.setStyleSheet(self.secondary_button_style)
        self.previous_btn.clicked.connect(self.previous_record)
        progress_row.addWidget(self.previous_btn)

        self.record_counter = QLabel("No record loaded")
        counter_font = QFont("Lato", 11)
        counter_font.setWeight(QFont.Weight.Bold)
        self.record_counter.setFont(counter_font)
        progress_row.addWidget(self.record_counter)
        progress_row.addStretch(1)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color:#555555;")
        progress_row.addWidget(self.progress_label)
        layout.addLayout(progress_row)

        self.record_card = QTextBrowser()
        self.record_card.setOpenExternalLinks(False)
        self.record_card.setMinimumHeight(260)
        layout.addWidget(self.record_card, 1)

        source_row = QHBoxLayout()
        self.open_source_btn = QPushButton("Open source")
        self.open_source_btn.setStyleSheet(self.secondary_button_style)
        self.open_source_btn.clicked.connect(self.open_current_source)
        source_row.addWidget(self.open_source_btn)
        source_row.addStretch(1)
        layout.addLayout(source_row)

        decision_box = QGroupBox("Your curation decision")
        decision_layout = QVBoxLayout(decision_box)

        button_row = QHBoxLayout()

        self.include_btn = QPushButton("Include")
        self.exclude_btn = QPushButton("Exclude")
        self.check_btn = QPushButton("Check Further")
        self.next_btn = QPushButton("Next Record")

        for btn in (self.include_btn, self.exclude_btn, self.check_btn, self.next_btn):
            btn.setMinimumHeight(48)

        self.include_btn.clicked.connect(lambda: self.select_decision("Include"))
        self.exclude_btn.clicked.connect(lambda: self.select_decision("Exclude"))
        self.check_btn.clicked.connect(lambda: self.select_decision("Check Further"))
        self.next_btn.clicked.connect(self.next_record)

        self.next_btn.setStyleSheet(self.primary_button_style)

        button_row.addWidget(self.include_btn)
        button_row.addWidget(self.exclude_btn)
        button_row.addWidget(self.check_btn)
        button_row.addWidget(self.next_btn)
        decision_layout.addLayout(button_row)

        reason_row = QHBoxLayout()
        reason_label = QLabel("Reason:")
        reason_label.setStyleSheet("font-weight:bold;")
        reason_row.addWidget(reason_label)

        self.reason_combo = QComboBox()
        self.reason_combo.addItems(REASONS)
        self.reason_combo.currentTextChanged.connect(self._reason_changed)
        reason_row.addWidget(self.reason_combo, 1)

        self.other_note = QLineEdit()
        self.other_note.setPlaceholderText("Add a short note")
        self.other_note.setVisible(False)
        self.other_note.textChanged.connect(self._update_next_enabled)
        reason_row.addWidget(self.other_note, 1)

        decision_layout.addLayout(reason_row)

        helper = QLabel(
            "Choose a decision and a reason before moving on. "
            "A short reason is enough."
        )
        helper.setStyleSheet("color:#555555;")
        helper.setWordWrap(True)
        decision_layout.addWidget(helper)

        layout.addWidget(decision_box)
        return page

    # ------------------------------------------------------------------
    # Loading and autosave
    # ------------------------------------------------------------------

    def load_candidate_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open candidate pool",
            "",
            "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            records = read_csv_records(path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not load candidate pool",
                str(exc)
            )
            return

        if not records:
            QMessageBox.information(
                self,
                "No records found",
                "The CSV contains no candidate records."
            )
            return

        self.records = records
        self.project.input_csv = path
        self.current_index = 0

        self._restore_autosave_if_available()
        self.loaded_label.setText(
            f"Loaded pool: {Path(path).name}    {len(self.records)} records"
        )

        self._set_loaded_state(True)
        self.refresh_all_views()
        self.show_record(self.current_index)

        self.statusBar().showMessage(
            f"Loaded {len(self.records)} candidate records."
        )

    def _autosave_path(self) -> Optional[Path]:
        if not self.project.input_csv:
            return None
        src = Path(self.project.input_csv)
        return src.with_name(src.stem + "_curator_autosave.json")

    def _autosave(self):
        path = self._autosave_path()
        if path is None or not self.records:
            return

        data = {
            "app": APP_TITLE,
            "version": APP_VERSION,
            "input_csv": self.project.input_csv,
            "research_question": self.question_edit.toPlainText().strip(),
            "current_index": self.current_index,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "decisions": [
                {
                    "source_id": rec.get("source_id", ""),
                    "title": rec.get("title", ""),
                    "curation_decision": rec.get("curation_decision", ""),
                    "curation_reason": rec.get("curation_reason", ""),
                    "curation_note": rec.get("curation_note", ""),
                    "source_opened": rec.get("source_opened", "No"),
                    "reviewed_at": rec.get("reviewed_at", ""),
                }
                for rec in self.records
            ],
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception:
            # Autosave should never interrupt student work.
            pass

    def _restore_autosave_if_available(self):
        path = self._autosave_path()
        if path is None or not path.exists():
            return

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return

        response = QMessageBox.question(
            self,
            "Resume previous curation?",
            "CobberHumCurator found saved work for this candidate pool. "
            "Would you like to resume where you left off?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        saved = data.get("decisions", [])
        by_key = {}
        for item in saved:
            key = (item.get("source_id", ""), item.get("title", ""))
            by_key[key] = item

        for rec in self.records:
            key = (rec.get("source_id", ""), rec.get("title", ""))
            item = by_key.get(key)
            if not item:
                continue
            for field in CURATION_FIELDS:
                if field in item:
                    rec[field] = safe_text(item[field])

        self.current_index = min(
            max(0, int(data.get("current_index", 0))),
            len(self.records) - 1,
        )

        self.question_edit.blockSignals(True)
        self.question_edit.setPlainText(data.get("research_question", ""))
        self.question_edit.blockSignals(False)

    # ------------------------------------------------------------------
    # Review workflow
    # ------------------------------------------------------------------

    def _set_loaded_state(self, loaded: bool):
        self.export_btn.setEnabled(
            loaded and any(
                r.get("curation_decision") == "Include"
                for r in self.records
            )
        )
        self.question_edit.setEnabled(loaded)
        self.previous_btn.setEnabled(False)
        self.open_source_btn.setEnabled(loaded)
        self.include_btn.setEnabled(loaded)
        self.exclude_btn.setEnabled(loaded)
        self.check_btn.setEnabled(loaded)
        self.reason_combo.setEnabled(loaded)
        self.next_btn.setEnabled(False)

        if not loaded:
            self.record_card.setHtml(
                "<h2>Welcome to CobberHumCurator</h2>"
                "<p>Load a candidate CSV exported from <i>CobberHumFetcher</i>.</p>"
                "<p>You will review one record at a time, decide whether to "
                "<b>Include</b>, <b>Exclude</b>, or <b>Check Further</b>, and leave "
                "a brief reason for the decision.</p>"
                "<p>The other tabs keep score and show patterns in the corpus you are building.</p>"
            )

    def _research_question_changed(self):
        self.project.research_question = self.question_edit.toPlainText().strip()
        self._autosave()

    def show_record(self, index: int):
        if not self.records:
            return

        self.current_index = max(0, min(index, len(self.records) - 1))
        rec = self.records[self.current_index]

        self.record_counter.setText(
            f"Record {self.current_index + 1} of {len(self.records)}"
        )

        html = [
            "<div style='font-family:Lato,Arial,sans-serif;'>",
            f"<h2 style='color:{COBBER_MAROON}; margin-top:0;'>"
            f"{html_escape(rec.get('title') or 'Untitled record')}</h2>",
        ]

        for field, label in DISPLAY_FIELDS:
            if field == "title":
                continue
            value = rec.get(field, "")
            if not value:
                continue
            html.append(
                f"<p style='margin:0 0 10px 0;'><b>{html_escape(label)}</b><br>"
                f"{html_escape(value)}</p>"
            )

        html.append("</div>")
        self.record_card.setHtml("".join(html))

        self.open_source_btn.setEnabled(True)

        self.current_decision = rec.get("curation_decision", "")
        self.reason_combo.blockSignals(True)
        self.reason_combo.setCurrentText(rec.get("curation_reason", ""))
        self.reason_combo.blockSignals(False)

        note = rec.get("curation_note", "")
        self.other_note.setText(note)
        self.other_note.setVisible(
            self.reason_combo.currentText() == "other / add note"
        )

        self.previous_btn.setEnabled(self.current_index > 0)

        self._update_decision_button_styles()
        self._update_next_enabled()
        self._refresh_progress_text()

    def select_decision(self, decision: str):
        if not self.records:
            return
        self.current_decision = decision
        self._update_decision_button_styles()
        self._update_next_enabled()

    def _update_decision_button_styles(self):
        mapping = {
            "Include": self.include_btn,
            "Exclude": self.exclude_btn,
            "Check Further": self.check_btn,
        }

        for decision, btn in mapping.items():
            if decision == self.current_decision:
                btn.setStyleSheet(self.primary_button_style)
            else:
                btn.setStyleSheet(self.secondary_button_style)

    def _reason_changed(self, reason: str):
        self.other_note.setVisible(reason == "other / add note")
        if reason != "other / add note":
            self.other_note.clear()
        self._update_next_enabled()

    def _update_next_enabled(self):
        reason = self.reason_combo.currentText().strip()
        note_ok = (
            reason != "other / add note"
            or bool(self.other_note.text().strip())
        )
        ready = bool(self.current_decision and reason and note_ok)
        self.next_btn.setEnabled(ready and bool(self.records))

    def _save_current_decision(self):
        if not self.records:
            return False

        reason = self.reason_combo.currentText().strip()
        note = self.other_note.text().strip()

        if not self.current_decision or not reason:
            return False
        if reason == "other / add note" and not note:
            return False

        rec = self.records[self.current_index]
        rec["curation_decision"] = self.current_decision
        rec["curation_reason"] = reason
        rec["curation_note"] = note
        rec["reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        self._autosave()
        self.refresh_all_views()
        return True

    def next_record(self):
        if not self._save_current_decision():
            return

        if self.current_index < len(self.records) - 1:
            self.show_record(self.current_index + 1)
        else:
            QMessageBox.information(
                self,
                "End of candidate pool",
                "You have reached the last candidate record. "
                "Use Check Further or Corpus Overview to review your work."
            )
            self.tabs.setCurrentWidget(self.overview_tab)

    def previous_record(self):
        if self.current_index > 0:
            self.show_record(self.current_index - 1)

    def open_current_source(self):
        if not self.records:
            return
        rec = self.records[self.current_index]
        url = rec.get("url", "").strip()
        if not url:
            QMessageBox.information(
                self,
                "Source link unavailable",
                "This record does not include a source URL."
            )
            return
        rec["source_opened"] = "Yes"
        self._autosave()
        QDesktopServices.openUrl(QUrl(url))

    def _refresh_progress_text(self):
        reviewed = sum(bool(r.get("curation_decision")) for r in self.records)
        remaining = len(self.records) - reviewed
        self.progress_label.setText(
            f"{reviewed} reviewed • {remaining} remaining"
        )

    # ------------------------------------------------------------------
    # Tables, dialog, overview
    # ------------------------------------------------------------------

    def refresh_all_views(self):
        if not self.records:
            self.export_btn.setEnabled(False)
            return

        self.included_tab.refresh()
        self.excluded_tab.refresh()
        self.check_tab.refresh()
        self.overview_tab.refresh()
        self._refresh_progress_text()

        self.export_btn.setEnabled(
            any(
                r.get("curation_decision") == "Include"
                for r in self.records
            )
        )

    def show_record_dialog(self, record: Dict[str, str]):
        def save_callback(updated_record):
            self._autosave()
            self.refresh_all_views()

            # If the dialog changed the record currently on the Review tab,
            # refresh the controls there too.
            if self.records and self.records[self.current_index] is updated_record:
                self.show_record(self.current_index)

        dialog = RecordDetailsDialog(self, record, save_callback)
        dialog.exec()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_corpus(self):
        if not self.records:
            return

        unresolved = sum(
            r.get("curation_decision") == "Check Further"
            for r in self.records
        )
        not_reviewed = sum(
            not r.get("curation_decision")
            for r in self.records
        )

        if unresolved or not_reviewed:
            message = (
                f"{unresolved} record(s) are still marked Check Further and "
                f"{not_reviewed} record(s) have not been reviewed.\n\n"
                "Only records marked Include will be written to the corpus CSV. "
                "The complete curation log will preserve every record and its current status.\n\n"
                "Export anyway?"
            )
            response = QMessageBox.question(
                self,
                "Unresolved records remain",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        default_name = "curated_corpus.csv"
        if self.project.input_csv:
            stem = Path(self.project.input_csv).stem
            default_name = f"{stem}_corpus.csv"

        corpus_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export curated corpus",
            default_name,
            "CSV files (*.csv)"
        )
        if not corpus_path:
            return

        corpus_path = Path(corpus_path)
        included = [
            rec for rec in self.records
            if rec.get("curation_decision") == "Include"
        ]

        # Preserve original dataset fields, excluding bulky raw JSON if present.
        original_fields = []
        for rec in self.records:
            for field in rec.keys():
                if field == "raw":
                    continue
                if field not in CURATION_FIELDS and field not in original_fields:
                    original_fields.append(field)

        corpus_fields = list(original_fields)
        log_fields = list(original_fields)
        for field in CURATION_FIELDS:
            if field not in log_fields:
                log_fields.append(field)

        log_path = corpus_path.with_name(
            corpus_path.stem.replace("_corpus", "") + "_curation_log.csv"
        )
        project_path = corpus_path.with_name(
            corpus_path.stem.replace("_corpus", "") + "_curator_project.json"
        )

        try:
            write_csv(str(corpus_path), included, corpus_fields)
            write_csv(str(log_path), self.records, log_fields)

            project_data = {
                "app": APP_TITLE,
                "version": APP_VERSION,
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "input_csv": self.project.input_csv,
                "research_question": self.question_edit.toPlainText().strip(),
                "candidate_pool_count": len(self.records),
                "included_count": len(included),
                "excluded_count": sum(
                    r.get("curation_decision") == "Exclude"
                    for r in self.records
                ),
                "check_further_count": unresolved,
                "not_reviewed_count": not_reviewed,
                "corpus_csv": str(corpus_path.name),
                "curation_log_csv": str(log_path.name),
            }
            with open(project_path, "w", encoding="utf-8") as fh:
                json.dump(project_data, fh, indent=2, ensure_ascii=False)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not export corpus",
                str(exc)
            )
            return

        QMessageBox.information(
            self,
            "Corpus exported",
            "CobberHumCurator saved:\n\n"
            f"{corpus_path.name}\n"
            f"{log_path.name}\n"
            f"{project_path.name}\n\n"
            "The corpus CSV contains only included records. "
            "The curation log preserves the complete decision history."
        )
        self.statusBar().showMessage(
            f"Exported {len(included)} included records."
        )


def main():
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberHumCuratorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
