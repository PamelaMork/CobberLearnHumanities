#!/usr/bin/env python3
"""
CobberHumImpute.py

CobberHumImpute for the ML for Humanities imputation chapter.

Current teaching structure:
    Explore the Survey
    Create an Age Test Gap
    Establish an Age Baseline
    Test three Age models, then reveal prepared wage and income evidence
    Decide what to do for all three fields
    Return to the Record

The application loads three fixed Ravi Mehta / Simpson Street survey files:

    data/ravi_community_survey_complete.csv
    data/ravi_community_survey_missing.csv
    data/ravi_community_survey_missing_truth.csv

The complete survey and missing-truth files remain internal. Students work
with the fixed survey containing the real teaching gaps.

Dependencies:
    pip install pandas numpy PyQt6 scikit-learn matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QButtonGroup,
    QRadioButton,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------

APP_TITLE = "CobberHumImpute"

COBBER_MAROON = "#6C1D45"
INFO_BLUE = "#3E6990"

CORE_FIELDS = [
    "age",
    "hourly_wage",
    "household_income",
]

DISPLAY_COLUMNS = [
    "respondent_id",
    "age",
    "years_in_neighborhood",
    "housing_status",
    "work_schedule",
    "occupation_group",
    "hourly_wage",
    "household_income",
    "redevelopment_concern",
]

TEST_TABLE_COLUMNS = {
    "age": [
        "respondent_id",
        "age",
        "years_in_neighborhood",
        "housing_status",
        "work_schedule",
        "occupation_group",
    ],
    "hourly_wage": [
        "respondent_id",
        "hourly_wage",
        "occupation_group",
        "work_schedule",
        "age",
        "years_in_neighborhood",
    ],
    "household_income": [
        "respondent_id",
        "household_income",
        "hourly_wage",
        "housing_status",
        "years_in_neighborhood",
        "occupation_group",
    ],
}

FIELD_LABELS = {
    "respondent_id": "Respondent ID",
    "age": "Age",
    "years_in_neighborhood": "Years in neighborhood",
    "housing_status": "Housing status",
    "work_schedule": "Work schedule",
    "occupation_group": "Occupation group",
    "hourly_wage": "Hourly wage",
    "household_income": "Household income",
    "redevelopment_concern": "Redevelopment concern",
}

FIELD_DESCRIPTIONS = {
    "age": (
        "Respondent age in years. A small number of survey records "
        "do not contain an age."
    ),
    "hourly_wage": (
        "Reported hourly wage. The surrounding record still contains "
        "information such as occupation group and work schedule when "
        "wage is missing."
    ),
    "household_income": (
        "Approximate annual household income reported by the respondent. "
        "Respondents could leave this question unanswered."
    ),
}

FIELD_QUESTIONS = {
    "age": (
        "When age is missing, what information about the respondent "
        "is still available?"
    ),
    "hourly_wage": (
        "When hourly wage is missing, what might occupation and work "
        "schedule tell Ravi about the record around the gap?"
    ),
    "household_income": (
        "When household income is missing, what can Ravi know from the "
        "rest of the record, and what remains unknowable?"
    ),
}

MECHANISM_CONTEXT = {
    "age": (
        "<b>About the real age gaps:</b> In this teaching survey, the missing "
        "ages were introduced independently of the survey values. This is a "
        "known MCAR case."
    ),
    "hourly_wage": (
        "<b>About the real hourly-wage gaps:</b> Missingness was created using "
        "information that remains observed, including work schedule and "
        "occupation group. This is a known MAR case."
    ),
    "household_income": (
        "<b>About the real household-income gaps:</b> Lower underlying "
        "household incomes were made more likely to be missing. This is a "
        "known MNAR case."
    ),
}

TEST_SEEDS = {
    "age": 1701,
    "hourly_wage": 1702,
    "household_income": 1703,
}

TEST_FRACTION = 0.20

MODEL_FEATURES = [
    "age",
    "years_in_neighborhood",
    "housing_status",
    "work_schedule",
    "occupation_group",
    "hourly_wage",
    "household_income",
    "redevelopment_concern",
]

CATEGORICAL_FEATURES = [
    "housing_status",
    "work_schedule",
    "occupation_group",
]

MODEL_LABELS = {
    "linear_regression": "Linear regression",
    "knn": "K-nearest neighbors",
    "random_forest": "Random forest",
}

MODEL_DESCRIPTIONS = {
    "linear_regression": (
        "Fits an overall relationship using information in the observed records."
    ),
    "knn": (
        "Estimates from observed records that are most similar to the record "
        "with the Test Gap."
    ),
    "random_forest": (
        "Combines many decision trees to estimate the hidden value from the "
        "rest of the record."
    ),
}

MECHANISM_LABELS = {
    "age": "MCAR",
    "hourly_wage": "MAR",
    "household_income": "MNAR",
}

DECISION_LABELS = {
    "leave_visible": "Leave the gaps visible",
    "estimate": "Estimate the missing values",
    "exclude": "Exclude incomplete records from an analysis that requires this field",
}


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def candidate_data_dirs() -> List[Path]:
    roots = [
        app_dir() / "data",
        app_dir(),
        Path.cwd() / "data",
        Path.cwd(),
    ]

    unique = []

    for root in roots:
        if root not in unique:
            unique.append(root)

    return unique


def find_data_file(filename: str) -> Optional[Path]:
    for root in candidate_data_dirs():
        candidate = root / filename

        if candidate.exists():
            return candidate

    return None


def load_ravi_survey_files() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    filenames = {
        "complete": "ravi_community_survey_complete.csv",
        "missing": "ravi_community_survey_missing.csv",
        "truth": "ravi_community_survey_missing_truth.csv",
    }

    paths = {
        key: find_data_file(filename)
        for key, filename in filenames.items()
    }

    not_found = [
        filenames[key]
        for key, path in paths.items()
        if path is None
    ]

    if not_found:
        searched = "\n".join(
            str(path)
            for path in candidate_data_dirs()
        )

        raise FileNotFoundError(
            "Could not find the Ravi survey data files:\n\n"
            + "\n".join(not_found)
            + "\n\nSearched:\n"
            + searched
        )

    complete_df = pd.read_csv(
        paths["complete"]
    )

    missing_df = pd.read_csv(
        paths["missing"]
    )

    truth_df = pd.read_csv(
        paths["truth"]
    )

    missing_columns = (
        set(DISPLAY_COLUMNS)
        - set(missing_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Student-facing survey is missing expected columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return (
        complete_df,
        missing_df,
        truth_df,
    )


def missing_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for field in CORE_FIELDS:
        missing_n = int(
            df[field]
            .isna()
            .sum()
        )

        if len(df) > 0:
            missing_pct = (
                100
                * df[field]
                .isna()
                .mean()
            )
        else:
            missing_pct = 0.0

        rows.append(
            {
                "Field":
                    FIELD_LABELS[field],

                "Missing n":
                    missing_n,

                "Missing %":
                    missing_pct,
            }
        )

    return pd.DataFrame(
        rows
    )


def format_value(
    value,
    column: str,
) -> str:
    if pd.isna(value):
        return "missing"

    if column == "hourly_wage":
        return (
            f"${float(value):,.2f}"
        )

    if column == "household_income":
        return (
            f"${float(value):,.0f}"
        )

    if column in (
        "age",
        "years_in_neighborhood",
        "redevelopment_concern",
    ):
        return str(
            int(
                round(
                    float(value)
                )
            )
        )

    return (
        str(value)
        .replace(
            "_",
            " ",
        )
    )


# ---------------------------------------------------------------------
# Survey table model
# ---------------------------------------------------------------------


class SurveyTableModel(
    QAbstractTableModel
):

    def __init__(
        self,
        df: pd.DataFrame,
        focus_field: Optional[str] = None,
    ):
        super().__init__()

        self.df = (
            df
            .reset_index(
                drop=True
            )
        )

        self.focus_field = (
            focus_field
        )


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(
            self.df
        )


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return (
            self.df
            .shape[1]
        )


    def data(
        self,
        index,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        value = self.df.iat[
            index.row(),
            index.column(),
        ]

        column = str(
            self.df.columns[
                index.column()
            ]
        )

        if (
            role
            == Qt.ItemDataRole.DisplayRole
        ):
            return format_value(
                value,
                column,
            )

        if (
            role
            == Qt.ItemDataRole.TextAlignmentRole
        ):
            if column in (
                "age",
                "years_in_neighborhood",
                "hourly_wage",
                "household_income",
                "redevelopment_concern",
            ):
                return int(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )

            return int(
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
            )

        if (
            role
            == Qt.ItemDataRole.ForegroundRole
            and pd.isna(value)
        ):
            return QBrush(
                QColor(
                    COBBER_MAROON
                )
            )

        if (
            role
            == Qt.ItemDataRole.FontRole
            and pd.isna(value)
        ):
            font = QFont()
            font.setBold(
                True
            )
            return font

        if (
            role
            == Qt.ItemDataRole.BackgroundRole
            and column
            == self.focus_field
        ):
            return QBrush(
                QColor(
                    "#F5F0E8"
                )
            )

        if (
            role
            == Qt.ItemDataRole.ToolTipRole
        ):
            if pd.isna(value):
                return (
                    f"{FIELD_LABELS.get(column, column)} "
                    "is missing in this record."
                )

            return format_value(
                value,
                column,
            )

        return None


    def headerData(
        self,
        section,
        orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if (
            role
            != Qt.ItemDataRole.DisplayRole
        ):
            return None

        if (
            orientation
            == Qt.Orientation.Horizontal
        ):
            column = str(
                self.df.columns[
                    section
                ]
            )

            return FIELD_LABELS.get(
                column,
                column,
            )

        return str(
            section + 1
        )


# ---------------------------------------------------------------------
# Missing-summary table model
# ---------------------------------------------------------------------


class MissingSummaryModel(
    QAbstractTableModel
):

    def __init__(
        self,
        df: pd.DataFrame,
        focus_field_label: str,
    ):
        super().__init__()

        self.df = (
            df
            .reset_index(
                drop=True
            )
        )

        self.focus_field_label = (
            focus_field_label
        )


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(
            self.df
        )


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return (
            self.df
            .shape[1]
        )


    def data(
        self,
        index,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        value = self.df.iat[
            index.row(),
            index.column(),
        ]

        column = str(
            self.df.columns[
                index.column()
            ]
        )

        field_label = str(
            self.df.iloc[
                index.row()
            ]["Field"]
        )

        is_focus = (
            field_label
            == self.focus_field_label
        )

        if (
            role
            == Qt.ItemDataRole.DisplayRole
        ):
            if column == "Missing %":
                return (
                    f"{float(value):.1f}%"
                )

            if column == "Missing n":
                return str(
                    int(value)
                )

            return str(
                value
            )

        if (
            role
            == Qt.ItemDataRole.TextAlignmentRole
            and column in (
                "Missing n",
                "Missing %",
            )
        ):
            return int(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

        if (
            role
            == Qt.ItemDataRole.FontRole
            and is_focus
        ):
            font = QFont()
            font.setBold(
                True
            )
            return font

        if (
            role
            == Qt.ItemDataRole.ForegroundRole
            and is_focus
        ):
            return QBrush(
                QColor(
                    COBBER_MAROON
                )
            )

        if (
            role
            == Qt.ItemDataRole.BackgroundRole
            and is_focus
        ):
            return QBrush(
                QColor(
                    "#F5F0E8"
                )
            )

        return None


    def headerData(
        self,
        section,
        orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if (
            role
            != Qt.ItemDataRole.DisplayRole
        ):
            return None

        if (
            orientation
            == Qt.Orientation.Horizontal
        ):
            return str(
                self.df.columns[
                    section
                ]
            )

        return str(
            section + 1
        )


# ---------------------------------------------------------------------
# Test-gap table model
# ---------------------------------------------------------------------


class TestGapTableModel(
    QAbstractTableModel
):
    """
    Distinguish missing ages from artificial test gaps.

    Real gap:
        missing

    Artificial test gap:
        hidden for test
    """

    def __init__(
        self,
        working_df: pd.DataFrame,
        source_missing_df: pd.DataFrame,
        columns: List[str],
        test_field: str,
        test_indices: set,
    ):
        super().__init__()

        self.columns = (
            columns
        )

        self.test_field = (
            test_field
        )

        self.test_indices = (
            test_indices
        )

        self.source_missing_df = (
            source_missing_df
        )

        self.df = (
            working_df[
                columns
            ]
            .copy()
        )

        self.df[
            "_source_index"
        ] = working_df.index

        self.df = (
            self.df
            .reset_index(
                drop=True
            )
        )


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(
            self.df
        )


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return len(
            self.columns
        )


    def source_index(
        self,
        row: int,
    ) -> int:
        return int(
            self.df.iloc[
                row
            ][
                "_source_index"
            ]
        )


    def is_test_gap(
        self,
        row: int,
        column: str,
    ) -> bool:
        return (
            column
            == self.test_field
            and self.source_index(
                row
            )
            in self.test_indices
        )


    def is_real_gap(
        self,
        row: int,
        column: str,
    ) -> bool:
        source_index = (
            self.source_index(
                row
            )
        )

        return (
            column
            == self.test_field
            and source_index
            not in self.test_indices
            and pd.isna(
                self.source_missing_df.at[
                    source_index,
                    column,
                ]
            )
        )


    def data(
        self,
        index,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        row = (
            index.row()
        )

        column = (
            self.columns[
                index.column()
            ]
        )

        value = (
            self.df.iloc[
                row
            ][column]
        )

        test_gap = (
            self.is_test_gap(
                row,
                column,
            )
        )

        real_gap = (
            self.is_real_gap(
                row,
                column,
            )
        )

        if (
            role
            == Qt.ItemDataRole.DisplayRole
        ):
            if test_gap:
                return (
                    "hidden for test"
                )

            if real_gap:
                return (
                    "missing"
                )

            return format_value(
                value,
                column,
            )

        if (
            role
            == Qt.ItemDataRole.TextAlignmentRole
        ):
            if column in (
                "age",
                "years_in_neighborhood",
                "hourly_wage",
                "household_income",
                "redevelopment_concern",
            ):
                return int(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )

            return int(
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
            )

        if (
            role
            == Qt.ItemDataRole.ForegroundRole
        ):
            if real_gap:
                return QBrush(
                    QColor(
                        COBBER_MAROON
                    )
                )

            if test_gap:
                return QBrush(
                    QColor(
                        INFO_BLUE
                    )
                )

        if (
            role
            == Qt.ItemDataRole.FontRole
            and (
                real_gap
                or test_gap
            )
        ):
            font = QFont()
            font.setBold(
                True
            )
            return font

        if (
            role
            == Qt.ItemDataRole.BackgroundRole
            and column
            == self.test_field
        ):
            if test_gap:
                return QBrush(
                    QColor(
                        "#EAF1F6"
                    )
                )

            if real_gap:
                return QBrush(
                    QColor(
                        "#F7EEF3"
                    )
                )

            return QBrush(
                QColor(
                    "#F5F0E8"
                )
            )

        if (
            role
            == Qt.ItemDataRole.ToolTipRole
        ):
            if test_gap:
                return (
                    "This value is known but temporarily hidden "
                    "so Ravi can test an estimation method."
                )

            if real_gap:
                return (
                    "This value was already missing in the survey. "
                    "There is no hidden answer key for this gap."
                )

            return format_value(
                value,
                column,
            )

        return None


    def headerData(
        self,
        section,
        orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if (
            role
            != Qt.ItemDataRole.DisplayRole
        ):
            return None

        if (
            orientation
            == Qt.Orientation.Horizontal
        ):
            column = (
                self.columns[
                    section
                ]
            )

            return FIELD_LABELS.get(
                column,
                column,
            )

        return str(
            section + 1
        )



# ---------------------------------------------------------------------
# Baseline table model
# ---------------------------------------------------------------------


class BaselineTableModel(
    QAbstractTableModel
):

    def __init__(
        self,
        comparison: Optional[pd.DataFrame] = None,
        field: Optional[str] = None,
    ):
        super().__init__()

        if comparison is None:
            comparison = pd.DataFrame()

        self.df = (
            comparison
            .reset_index(
                drop=True
            )
        )

        self.field = (
            field
        )

        self.columns = [
            "respondent_id",
            "hidden_truth",
            "mean_estimate",
            "absolute_error",
        ]

        self.headers = {
            "respondent_id": "Respondent",
            "hidden_truth": "Hidden truth",
            "mean_estimate": "Mean estimate",
            "absolute_error": "Absolute error",
        }


    def set_data(
        self,
        comparison: pd.DataFrame,
        field: Optional[str],
    ) -> None:
        self.beginResetModel()

        self.df = (
            comparison
            .reset_index(
                drop=True
            )
        )

        self.field = (
            field
        )

        self.endResetModel()


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(
            self.df
        )


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return len(
            self.columns
        )


    def data(
        self,
        index,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        row = (
            self.df
            .iloc[
                index.row()
            ]
        )

        column = (
            self.columns[
                index.column()
            ]
        )

        value = (
            row[column]
        )

        if (
            role
            == Qt.ItemDataRole.DisplayRole
        ):
            if (
                column
                == "respondent_id"
            ):
                return str(
                    value
                )

            if pd.isna(
                value
            ):
                return ""

            if (
                self.field
                == "age"
            ):
                if (
                    column
                    == "hidden_truth"
                ):
                    return (
                        f"{float(value):.0f}"
                    )

                return (
                    f"{float(value):.2f}"
                )

            if (
                self.field
                == "hourly_wage"
            ):
                return (
                    f"${float(value):,.2f}"
                )

            if (
                self.field
                == "household_income"
            ):
                return (
                    f"${float(value):,.2f}"
                )

            return (
                f"{float(value):,.2f}"
            )

        if (
            role
            == Qt.ItemDataRole.TextAlignmentRole
            and column
            != "respondent_id"
        ):
            return int(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

        if (
            role
            == Qt.ItemDataRole.BackgroundRole
        ):
            if (
                column
                == "hidden_truth"
            ):
                return QBrush(
                    QColor(
                        "#EAF1F6"
                    )
                )

            if (
                column
                == "mean_estimate"
            ):
                return QBrush(
                    QColor(
                        "#F5F0E8"
                    )
                )

        if (
            role
            == Qt.ItemDataRole.ForegroundRole
            and column
            == "hidden_truth"
        ):
            return QBrush(
                QColor(
                    INFO_BLUE
                )
            )

        if (
            role
            == Qt.ItemDataRole.FontRole
            and column
            == "hidden_truth"
        ):
            font = QFont()

            font.setBold(
                True
            )

            return font

        return None


    def headerData(
        self,
        section,
        orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if (
            role
            != Qt.ItemDataRole.DisplayRole
        ):
            return None

        if (
            orientation
            == Qt.Orientation.Horizontal
        ):
            column = (
                self.columns[
                    section
                ]
            )

            return (
                self.headers[
                    column
                ]
            )

        return str(
            section + 1
        )


# ---------------------------------------------------------------------
# Tab 1: Explore the Survey
# ---------------------------------------------------------------------


class ExploreSurveyPage(QWidget):

    def __init__(self, main: "CobberHumImputeApp"):
        super().__init__(main)
        self.main = main

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "<b>Ravi's Simpson Street Community Survey</b><br>"
            "Ravi's team collected a short community survey to supplement "
            "the recorded interviews from Simpson Street. The survey includes "
            "basic demographic information, work patterns, housing history, "
            "and concern about neighborhood redevelopment. Some records are "
            "incomplete. Before Ravi decides what to do with the gaps, he "
            "needs to understand the data around them."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 10px;"
        )
        layout.addWidget(intro)

        top = QHBoxLayout()
        top.setSpacing(12)

        field_box = QGroupBox("Choose the missing field to investigate")
        field_layout = QVBoxLayout(field_box)

        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Field:</b>"))
        self.field_combo = QComboBox()
        self.field_combo.addItems([FIELD_LABELS[field] for field in CORE_FIELDS])
        self.field_combo.setFixedWidth(180)
        row.addWidget(self.field_combo)
        row.addStretch()
        field_layout.addLayout(row)

        self.about_label = QLabel()
        self.about_label.setStyleSheet("font-weight: bold; color: #6C1D45;")
        self.about_field = QLabel()
        self.about_field.setWordWrap(True)
        field_layout.addWidget(self.about_label)
        field_layout.addWidget(self.about_field)
        field_layout.addStretch()
        top.addWidget(field_box, stretch=1)

        summary_box = QGroupBox("Missing-data summary")
        summary_layout = QVBoxLayout(summary_box)
        summary_help = QLabel(
            "The selected field is highlighted. These are the missing ages "
            "already present in Ravi's survey."
        )
        summary_help.setWordWrap(True)
        summary_help.setStyleSheet("color: #555555;")
        summary_layout.addWidget(summary_help)

        self.missing_table = QTableView()
        self.missing_table.setMinimumHeight(145)
        self.missing_table.setMaximumHeight(170)
        self.missing_table.verticalHeader().setVisible(False)
        self.missing_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.missing_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.missing_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.missing_table)
        top.addWidget(summary_box, stretch=1)

        layout.addLayout(top)

        self.preview_label = QLabel("Survey records")
        self.preview_label.setStyleSheet("font-weight: bold; color: #6C1D45;")
        layout.addWidget(self.preview_label)

        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setFixedHeight(42)
        layout.addWidget(self.table, stretch=1)

        self.prompt = QLabel()
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet(
            "background-color: #FAFAFA; "
            "border-left: 4px solid #6C1D45; "
            "padding: 8px;"
        )
        layout.addWidget(self.prompt)

        self.field_combo.currentTextChanged.connect(self.refresh)
        self.refresh()

    def selected_field(self) -> str:
        label = self.field_combo.currentText()
        for field in CORE_FIELDS:
            if FIELD_LABELS[field] == label:
                return field
        return CORE_FIELDS[0]

    def refresh(self) -> None:
        field = self.selected_field()
        field_label = FIELD_LABELS[field]

        self.about_label.setText(field_label)
        self.about_field.setText(FIELD_DESCRIPTIONS[field])

        report = missing_report(self.main.missing_df)
        self.missing_model = MissingSummaryModel(
            report,
            focus_field_label=field_label,
        )
        self.missing_table.setModel(self.missing_model)
        self.missing_table.resizeColumnsToContents()
        self.missing_table.setColumnWidth(0, 150)
        self.missing_table.setColumnWidth(1, 90)
        self.missing_table.setColumnWidth(2, 100)
        self.missing_table.horizontalHeader().setStretchLastSection(False)
        self.missing_table.resizeRowsToContents()

        preview = self.main.missing_df[DISPLAY_COLUMNS].copy()
        self.table_model = SurveyTableModel(preview, focus_field=field)
        self.table.setModel(self.table_model)

        survey_widths = {
            0: 110,
            1: 70,
            2: 150,
            3: 145,
            4: 120,
            5: 135,
            6: 105,
            7: 135,
            8: 170,
        }
        for column_index, column_width in survey_widths.items():
            self.table.setColumnWidth(column_index, column_width)
        self.table.horizontalHeader().setStretchLastSection(False)

        self.preview_label.setText(
            f"Survey records: {len(self.main.missing_df)} respondents"
        )
        self.prompt.setText(
            f"<b>Look closely:</b> {FIELD_QUESTIONS[field]}"
        )

# ---------------------------------------------------------------------
# Tab 2: Create a Test Gap
# ---------------------------------------------------------------------


class CreateTestGapPage(QWidget):

    FIELD = "age"

    def __init__(self, main: "CobberHumImputeApp"):
        super().__init__(main)
        self.main = main

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "<b>Create a Test Gap</b><br>"
            "Ravi cannot test an estimate against a value that was never "
            "recorded because he does not know the true value. He will begin "
            "with age. CobberHumImpute temporarily hides some recorded ages "
            "while keeping their true values as an answer key."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 10px;"
        )
        layout.addWidget(intro)

        controls_box = QGroupBox("Create an Age Test Gap")
        controls = QHBoxLayout(controls_box)
        controls.addWidget(QLabel("<b>Field to test:</b> Age"))
        controls.addSpacing(20)
        controls.addWidget(QLabel("<b>Test size:</b> 20% of currently recorded ages"))
        controls.addStretch()

        self.create_btn = QPushButton("Create Test Gap")
        self.create_btn.setFixedWidth(150)
        self.create_btn.setStyleSheet(
            "background-color: #6C1D45; color: white; font-weight: bold; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 6px 10px;"
        )
        self.clear_btn = QPushButton("Clear Test Gap")
        self.clear_btn.setFixedWidth(140)
        controls.addWidget(self.create_btn)
        controls.addWidget(self.clear_btn)
        layout.addWidget(controls_box)

        self.mechanism_context = QLabel(MECHANISM_CONTEXT[self.FIELD])
        self.mechanism_context.setWordWrap(True)
        self.mechanism_context.setStyleSheet(
            "background-color: #F7F7F7; "
            "border-left: 4px solid #3E6990; padding: 8px;"
        )
        layout.addWidget(self.mechanism_context)

        distinction = QLabel(
            "<b>Do not confuse the two kinds of gaps.</b> "
            "<span style='color:#6C1D45;'><b>Missing ages</b></span> "
            "were already missing from the survey. "
            "<span style='color:#3E6990;'><b>Test Gaps</b></span> "
            "contain ages Ravi actually knows, but the app temporarily hides "
            "them for the experiment."
        )
        distinction.setWordWrap(True)
        distinction.setStyleSheet(
            "background-color: #FAFAFA; border: 1px solid #D6D6D6; "
            "border-radius: 5px; padding: 8px;"
        )
        layout.addWidget(distinction)

        status_box = QGroupBox("Experiment status")
        status_layout = QHBoxLayout(status_box)
        self.observed_status = QLabel()
        self.real_status = QLabel()
        self.test_status = QLabel()
        status_style = (
            "background-color: #FFFFFF; border: 1px solid #CFCFCF; "
            "border-radius: 4px; padding: 10px;"
        )
        for widget in (self.observed_status, self.real_status, self.test_status):
            widget.setStyleSheet(status_style)
            status_layout.addWidget(widget)
        status_layout.addStretch()
        layout.addWidget(status_box)

        self.change_summary = QLabel()
        self.change_summary.setWordWrap(True)
        self.change_summary.setStyleSheet(
            "background-color: #F7F7F7; "
            "border-left: 4px solid #3E6990; padding: 8px;"
        )
        layout.addWidget(self.change_summary)

        self.table_label = QLabel("Survey records")
        self.table_label.setStyleSheet("font-weight: bold; color: #6C1D45;")
        layout.addWidget(self.table_label)

        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setFixedHeight(42)
        layout.addWidget(self.table, stretch=1)

        prompt = QLabel(
            "<b>Think about it:</b> Why does a Test Gap give Ravi "
            "something the missing ages cannot?"
        )
        prompt.setWordWrap(True)
        prompt.setStyleSheet(
            "background-color: #FAFAFA; "
            "border-left: 4px solid #6C1D45; padding: 8px;"
        )
        layout.addWidget(prompt)

        self.create_btn.clicked.connect(self.create_gap)
        self.clear_btn.clicked.connect(self.clear_gap)
        self.refresh()

    def selected_field(self) -> str:
        return self.FIELD

    def set_selected_field(self, field: str) -> None:
        # Retained for compatibility with older navigation code.
        return

    def create_gap(self) -> None:
        field = self.FIELD

        if self.main.test_gap_field is not None:
            reply = QMessageBox.question(
                self,
                "Replace current Test Gap?",
                "An Age Test Gap already exists. Create it again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.main.clear_test_gap()

        self.main.baseline_results.pop(field, None)
        self.main.model_results.pop(field, None)
        self.main.decision_results.pop(field, None)
        self.main.update_return_tab_state()

        source = self.main.missing_df
        observed_indices = source.index[source[field].notna()].to_numpy()
        n_hide = max(1, int(round(TEST_FRACTION * len(observed_indices))))
        rng = np.random.default_rng(TEST_SEEDS[field])
        test_indices = np.sort(
            rng.choice(observed_indices, size=n_hide, replace=False)
        )

        truth = source.loc[test_indices, ["respondent_id", field]].copy()
        truth = truth.rename(columns={field: "true_value"})
        truth["field"] = field
        truth["source_index"] = test_indices
        truth = truth[
            ["source_index", "respondent_id", "field", "true_value"]
        ].reset_index(drop=True)

        self.main.working_df = self.main.missing_df.copy(deep=True)
        self.main.working_df.loc[test_indices, field] = np.nan
        self.main.test_gap_field = field
        self.main.test_gap_indices = {int(index) for index in test_indices}
        self.main.test_truth = truth

        self.refresh()
        QMessageBox.information(
            self,
            "Test Gap created",
            f"{n_hide} known age values are now hidden for testing. "
            "Their true values are stored internally.",
        )

    def clear_gap(self) -> None:
        if self.main.test_gap_field is None:
            QMessageBox.information(
                self, "No Test Gap", "There is no Test Gap to clear."
            )
            return
        self.main.clear_test_gap()
        self.refresh()

    def refresh(self) -> None:
        field = self.FIELD
        source = self.main.missing_df
        total_n = len(source)
        real_missing_n = int(source[field].isna().sum())
        original_observed_n = total_n - real_missing_n
        active_for_field = self.main.test_gap_field == field

        if active_for_field:
            test_n = len(self.main.test_gap_indices)
            still_visible_n = original_observed_n - test_n
        else:
            test_n = 0
            still_visible_n = original_observed_n

        self.observed_status.setText(
            "<b>Recorded ages still visible</b><br>" f"{still_visible_n}"
        )
        self.real_status.setText(
            "<span style='color:#6C1D45;'><b>Missing ages</b></span><br>"
            f"{real_missing_n}"
        )
        self.test_status.setText(
            "<span style='color:#3E6990;'><b>Age Test Gaps</b></span><br>"
            f"{test_n}"
        )

        if active_for_field:
            self.change_summary.setText(
                f"<b>What changed?</b> {test_n} known age values are now hidden "
                "for testing. Their true values are still stored internally. "
                "The missing ages remain different because their true values are unknown."
            )
            working_df = self.main.working_df
            test_indices = self.main.test_gap_indices
            self.table_label.setText(
                f"Survey records: Age with {real_missing_n} missing ages and "
                f"{test_n} Test Gaps"
            )
        else:
            self.change_summary.setText(
                "<b>What will change?</b> The app will temporarily hide some "
                "known ages while keeping their true values internally for later comparison."
            )
            working_df = self.main.missing_df
            test_indices = set()
            self.table_label.setText("Survey records: Age before a Test Gap is created")

        columns = TEST_TABLE_COLUMNS[field]
        self.table_model = TestGapTableModel(
            working_df=working_df,
            source_missing_df=self.main.missing_df,
            columns=columns,
            test_field=field,
            test_indices=test_indices,
        )
        self.table.setModel(self.table_model)

        width_map = {
            "respondent_id": 110,
            "age": 150,
            "years_in_neighborhood": 150,
            "housing_status": 145,
            "work_schedule": 120,
            "occupation_group": 135,
        }
        for column_index, column_name in enumerate(columns):
            self.table.setColumnWidth(column_index, width_map.get(column_name, 120))
        self.table.horizontalHeader().setStretchLastSection(False)
        self.clear_btn.setEnabled(self.main.test_gap_field is not None)

# ---------------------------------------------------------------------
# Tab 3: Establish a Baseline
# ---------------------------------------------------------------------


class EstablishBaselinePage(
    QWidget
):

    def __init__(
        self,
        main: "CobberHumImputeApp",
    ):
        super().__init__(
            main
        )

        self.main = (
            main
        )

        layout = QVBoxLayout(
            self
        )

        layout.setSpacing(
            8
        )

        # -------------------------------------------------------------
        # Introduction
        # -------------------------------------------------------------

        intro = QLabel(
            "<b>Establish a Baseline</b><br>"
            "Before Ravi tries a machine-learning model, he can begin with "
            "the simplest possible estimate. He can use the mean of the "
            "values that remain visible to estimate every Test Gap. "
            "Because the Test Gap values are actually known, he can then "
            "measure how far those estimates are from the hidden truth."
        )

        intro.setWordWrap(
            True
        )

        intro.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 10px;"
        )

        layout.addWidget(
            intro
        )

        # -------------------------------------------------------------
        # Top row: Current Test Gap + Baseline result
        # -------------------------------------------------------------

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        status_box = QGroupBox(
            "Current Test Gap"
        )

        status_layout = QGridLayout(
            status_box
        )

        status_layout.setColumnStretch(
            1,
            1,
        )

        status_layout.addWidget(
            QLabel(
                "<b>Field under investigation</b>"
            ),
            0,
            0,
        )

        self.field_label = QLabel(
            "—"
        )

        status_layout.addWidget(
            self.field_label,
            0,
            1,
        )

        status_layout.addWidget(
            QLabel(
                "<b>Test Gaps</b>"
            ),
            1,
            0,
        )

        self.test_n_label = QLabel(
            "—"
        )

        status_layout.addWidget(
            self.test_n_label,
            1,
            1,
        )

        status_layout.addWidget(
            QLabel(
                "<b>Recorded values still available</b>"
            ),
            2,
            0,
        )

        self.observed_n_label = QLabel(
            "—"
        )

        status_layout.addWidget(
            self.observed_n_label,
            2,
            1,
        )

        self.test_btn = QPushButton(
            "Test the Mean"
        )

        self.test_btn.setFixedWidth(
            150
        )

        self.test_btn.setStyleSheet(
            "background-color: #6C1D45; "
            "color: white; "
            "font-weight: bold; "
            "border: 1px solid #6C1D45; "
            "border-radius: 4px; "
            "padding: 6px 10px;"
        )

        self.test_btn.clicked.connect(
            self.test_mean
        )

        status_layout.addWidget(
            self.test_btn,
            3,
            0,
            1,
            2,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        top_row.addWidget(
            status_box,
            stretch=1,
        )

        result_box = QGroupBox(
            "Baseline result"
        )

        result_layout = QHBoxLayout(
            result_box
        )

        self.mean_result = QLabel(
            "<b>Mean estimate</b><br>—"
        )

        self.mean_result.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.mean_result.setStyleSheet(
            "background-color: #FFFFFF; "
            "border: 1px solid #CFCFCF; "
            "border-radius: 4px; "
            "padding: 12px;"
        )

        self.mae_result = QLabel(
            "<b>Mean Absolute Error (MAE)</b><br>—"
        )

        self.mae_result.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.mae_result.setStyleSheet(
            "background-color: #FFFFFF; "
            "border: 1px solid #CFCFCF; "
            "border-radius: 4px; "
            "padding: 12px;"
        )

        result_layout.addWidget(
            self.mean_result
        )

        result_layout.addWidget(
            self.mae_result
        )

        top_row.addWidget(
            result_box,
            stretch=1,
        )

        layout.addLayout(
            top_row
        )

        # -------------------------------------------------------------
        # Evidence area: table + scatterplot
        # -------------------------------------------------------------

        evidence_layout = QHBoxLayout()
        evidence_layout.setSpacing(12)

        table_box = QGroupBox(
            "Individual Test Gap estimates"
        )

        table_layout = QVBoxLayout(
            table_box
        )

        self.table = QTableView()

        self.table_model = (
            BaselineTableModel()
        )

        self.table.setModel(
            self.table_model
        )

        self.table.setAlternatingRowColors(
            True
        )

        self.table.verticalHeader().setVisible(
            False
        )

        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.table.setWordWrap(
            False
        )

        self.table.horizontalHeader().setFixedHeight(
            42
        )

        table_layout.addWidget(
            self.table,
            stretch=1,
        )

        evidence_layout.addWidget(
            table_box,
            stretch=1,
        )

        plot_box = QGroupBox(
            "Mean Estimate vs. Hidden Truth"
        )

        plot_layout = QVBoxLayout(
            plot_box
        )

        self.figure = Figure(
            figsize=(
                5.5,
                4.2,
            ),
            dpi=100,
        )

        self.canvas = FigureCanvas(
            self.figure
        )

        plot_layout.addWidget(
            self.canvas,
            stretch=1,
        )

        evidence_layout.addWidget(
            plot_box,
            stretch=1,
        )

        layout.addLayout(
            evidence_layout,
            stretch=1,
        )

        # -------------------------------------------------------------
        # Thinking prompt
        # -------------------------------------------------------------

        prompt = QLabel(
            "<b>Look closely:</b> Every Test Gap received exactly the "
            "same estimate. What does the graph reveal about what the "
            "mean ignores?"
        )

        prompt.setWordWrap(
            True
        )

        prompt.setStyleSheet(
            "background-color: #FAFAFA; "
            "border-left: 4px solid #6C1D45; "
            "padding: 8px;"
        )

        layout.addWidget(
            prompt
        )

        self.clear_plot()
        self.refresh()


    def format_result(
        self,
        value: float,
        field: str,
    ) -> str:
        if (
            field
            == "age"
        ):
            return (
                f"{value:.2f} years"
            )

        if (
            field
            == "hourly_wage"
        ):
            return (
                f"${value:,.2f}/hour"
            )

        if (
            field
            == "household_income"
        ):
            return (
                f"${value:,.2f}"
            )

        return (
            f"{value:,.2f}"
        )


    def clear_plot(
        self,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Test the mean to see the estimate pattern.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#555555",
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()


    def update_plot(
        self,
        comparison: pd.DataFrame,
        field: str,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        truth = pd.to_numeric(
            comparison["hidden_truth"],
            errors="coerce",
        ).to_numpy(dtype=float)

        estimate = pd.to_numeric(
            comparison["mean_estimate"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(truth) & np.isfinite(estimate)
        truth = truth[valid]
        estimate = estimate[valid]

        ax.scatter(
            truth,
            estimate,
            s=30,
            alpha=0.75,
            color=INFO_BLUE,
            edgecolors="white",
            linewidths=0.4,
        )

        if len(truth) > 0:
            lower = float(min(truth.min(), estimate.min()))
            upper = float(max(truth.max(), estimate.max()))

            if upper <= lower:
                upper = lower + 1.0

            padding = max((upper - lower) * 0.06, 1.0)
            lower -= padding
            upper += padding

            ax.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--",
                linewidth=1.5,
                color=COBBER_MAROON,
                label="Perfect estimate",
            )

            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)

        if field == "age":
            x_label = "Hidden true age (years)"
            y_label = "Mean estimate (years)"
        elif field == "hourly_wage":
            x_label = "Hidden true wage ($/hour)"
            y_label = "Mean estimate ($/hour)"
        elif field == "household_income":
            x_label = "Hidden true income ($)"
            y_label = "Mean estimate ($)"
        else:
            x_label = "Hidden truth"
            y_label = "Mean estimate"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.20)
        ax.legend(loc="lower right")

        self.figure.tight_layout()
        self.canvas.draw_idle()


    def refresh(
        self,
    ) -> None:
        field = (
            self.main
            .test_gap_field
        )

        if (
            field
            is None
            or self.main.test_truth.empty
        ):
            self.field_label.setText(
                "Create a Test Gap on the previous tab."
            )

            self.test_n_label.setText(
                "—"
            )

            self.observed_n_label.setText(
                "—"
            )

            self.mean_result.setText(
                "<b>Mean estimate</b><br>—"
            )

            self.mae_result.setText(
                "<b>Mean Absolute Error (MAE)</b><br>—"
            )

            self.table_model.set_data(
                pd.DataFrame(),
                None,
            )

            self.clear_plot()

            self.test_btn.setEnabled(
                False
            )

            return

        truth = (
            self.main
            .test_truth
            .copy()
        )

        observed = (
            self.main
            .working_df[
                field
            ]
            .dropna()
        )

        self.field_label.setText(
            FIELD_LABELS[
                field
            ]
        )

        self.test_n_label.setText(
            str(
                len(
                    truth
                )
            )
        )

        self.observed_n_label.setText(
            str(
                len(
                    observed
                )
            )
        )

        self.test_btn.setEnabled(
            True
        )

        result = (
            self.main
            .baseline_results
            .get(
                field
            )
        )

        if (
            result is not None
            and result["test_gap_indices"]
            == self.main.test_gap_indices
        ):
            self.show_result(
                result
            )

        else:
            self.mean_result.setText(
                "<b>Mean estimate</b><br>—"
            )

            self.mae_result.setText(
                "<b>Mean Absolute Error (MAE)</b><br>—"
            )

            self.table_model.set_data(
                pd.DataFrame(),
                field,
            )

            self.clear_plot()


    def test_mean(
        self,
    ) -> None:
        field = (
            self.main
            .test_gap_field
        )

        if (
            field
            is None
            or self.main.test_truth.empty
        ):
            QMessageBox.information(
                self,
                "Create a Test Gap First",
                "Create a Test Gap before testing the mean.",
            )
            return

        observed = (
            self.main
            .working_df[
                field
            ]
            .dropna()
        )

        mean_value = float(
            observed.mean()
        )

        truth = (
            self.main
            .test_truth
            .copy()
        )

        comparison = pd.DataFrame(
            {
                "respondent_id":
                    truth[
                        "respondent_id"
                    ]
                    .astype(
                        str
                    ),

                "hidden_truth":
                    pd.to_numeric(
                        truth[
                            "true_value"
                        ],
                        errors="coerce",
                    ),

                "mean_estimate":
                    mean_value,
            }
        )

        comparison[
            "absolute_error"
        ] = (
            comparison[
                "hidden_truth"
            ]
            - comparison[
                "mean_estimate"
            ]
        ).abs()

        comparison = (
            comparison
            .dropna(
                subset=[
                    "hidden_truth",
                    "absolute_error",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        mae = float(
            comparison[
                "absolute_error"
            ]
            .mean()
        )

        self.main.baseline_results[
            field
        ] = {
            "field":
                field,

            "mean_estimate":
                mean_value,

            "mae":
                mae,

            "comparison":
                comparison.copy(),

            "test_gap_indices":
                self.main.test_gap_indices.copy(),

            "n_test":
                len(
                    comparison
                ),
        }

        self.show_result(
            self.main.baseline_results[
                field
            ]
        )


    def show_result(
        self,
        result,
    ) -> None:
        field = (
            result[
                "field"
            ]
        )

        mean_value = (
            result[
                "mean_estimate"
            ]
        )

        mae = (
            result[
                "mae"
            ]
        )

        self.mean_result.setText(
            "<b>Mean estimate</b><br>"
            + self.format_result(
                mean_value,
                field,
            )
        )

        self.mae_result.setText(
            "<b>Mean Absolute Error (MAE)</b><br>"
            + self.format_result(
                mae,
                field,
            )
        )

        comparison = (
            result[
                "comparison"
            ]
        )

        self.table_model.set_data(
            comparison,
            field,
        )

        baseline_widths = {
            0: 120,
            1: 140,
            2: 140,
            3: 140,
        }

        for column_index, column_width in baseline_widths.items():
            self.table.setColumnWidth(
                column_index,
                column_width,
            )

        self.table.horizontalHeader().setStretchLastSection(
            True
        )

        self.update_plot(
            comparison,
            field,
        )

# ---------------------------------------------------------------------
# Tab 4 result table models
# ---------------------------------------------------------------------


class ModelComparisonTableModel(QAbstractTableModel):

    FIELD_TINTS = {
        "Age": "#F7EEF3",
        "Hourly wage": "#EAF1F6",
        "Household income": "#F5F0E8",
    }

    def __init__(self, rows: Optional[pd.DataFrame] = None):
        super().__init__()
        self.df = (
            rows.reset_index(drop=True)
            if rows is not None
            else pd.DataFrame(columns=["field", "model", "mae", "comparison"])
        )
        self.columns = ["field", "model", "mae", "comparison"]
        self.headers = {
            "field": "Field",
            "model": "Model",
            "mae": "MAE",
            "comparison": "Compared with baseline",
        }

    def set_data(self, rows: pd.DataFrame) -> None:
        self.beginResetModel()
        self.df = rows.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = self.df.iloc[index.row()]
        column = self.columns[index.column()]
        value = row[column]
        field_label = str(row.get("field", ""))
        field_key = str(row.get("field_key", ""))

        if role == Qt.ItemDataRole.DisplayRole:
            if column in ("field", "model", "comparison"):
                return str(value)

            if pd.isna(value):
                return "—"

            if field_key == "age":
                return f"{float(value):.2f} years"
            if field_key == "hourly_wage":
                return f"${float(value):,.2f}/hour"
            if field_key == "household_income":
                return f"${float(value):,.0f}"
            return f"{float(value):,.2f}"

        if role == Qt.ItemDataRole.TextAlignmentRole and column == "mae":
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.BackgroundRole:
            color = self.FIELD_TINTS.get(field_label)
            if color:
                return QBrush(QColor(color))

        if role == Qt.ItemDataRole.FontRole and column == "field":
            font = QFont()
            font.setBold(True)
            return font

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.headers[self.columns[section]]
        return str(section + 1)

class ModelPredictionTableModel(
    QAbstractTableModel
):

    def __init__(
        self,
        rows: Optional[pd.DataFrame] = None,
        field: Optional[str] = None,
    ):
        super().__init__()

        self.df = (
            rows.reset_index(drop=True)
            if rows is not None
            else pd.DataFrame(
                columns=[
                    "respondent_id",
                    "hidden_truth",
                    "estimate",
                    "absolute_error",
                ]
            )
        )

        self.field = field
        self.columns = [
            "respondent_id",
            "hidden_truth",
            "estimate",
            "absolute_error",
        ]

        self.headers = {
            "respondent_id": "Respondent",
            "hidden_truth": "Hidden truth",
            "estimate": "Model estimate",
            "absolute_error": "Absolute error",
        }


    def set_data(
        self,
        rows: pd.DataFrame,
        field: Optional[str],
    ) -> None:
        self.beginResetModel()
        self.df = rows.reset_index(drop=True)
        self.field = field
        self.endResetModel()


    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(self.df)


    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return len(self.columns)


    def data(
        self,
        index,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        row = self.df.iloc[index.row()]
        column = self.columns[index.column()]
        value = row[column]

        if role == Qt.ItemDataRole.DisplayRole:
            if column == "respondent_id":
                return str(value)

            if pd.isna(value):
                return ""

            if self.field == "age":
                if column == "hidden_truth":
                    return f"{float(value):.0f}"
                return f"{float(value):.2f}"

            if self.field == "hourly_wage":
                return f"${float(value):,.2f}"

            if self.field == "household_income":
                return f"${float(value):,.0f}"

            return f"{float(value):,.2f}"

        if (
            role == Qt.ItemDataRole.TextAlignmentRole
            and column != "respondent_id"
        ):
            return int(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

        if (
            role == Qt.ItemDataRole.BackgroundRole
            and column == "hidden_truth"
        ):
            return QBrush(QColor("#EAF1F6"))

        if (
            role == Qt.ItemDataRole.ForegroundRole
            and column == "hidden_truth"
        ):
            return QBrush(QColor(INFO_BLUE))

        if (
            role == Qt.ItemDataRole.FontRole
            and column == "hidden_truth"
        ):
            font = QFont()
            font.setBold(True)
            return font

        return None


    def headerData(
        self,
        section,
        orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return self.headers[self.columns[section]]

        return str(section + 1)


class EstimateGraphDialog(
    QDialog
):
    """
    Matplotlib popup for hidden truth versus model estimate.
    """

    def __init__(
        self,
        comparison: pd.DataFrame,
        field: str,
        method_label: str,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self.setWindowTitle(
            f"{method_label} — Hidden Truth and Model Estimate"
        )

        self.resize(
            820,
            650,
        )

        layout = QVBoxLayout(
            self
        )

        explanation = QLabel(
            "<b>Hidden truth and model estimate</b><br>"
            "Each point represents one Test Gap. Points closer to the "
            "diagonal line have estimates closer to the hidden true value."
        )

        explanation.setWordWrap(
            True
        )

        explanation.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 10px;"
        )

        layout.addWidget(
            explanation
        )

        figure = Figure(
            figsize=(
                8,
                5.5,
            ),
            dpi=100,
        )

        self.canvas = FigureCanvas(
            figure
        )

        layout.addWidget(
            self.canvas,
            stretch=1,
        )

        ax = figure.add_subplot(
            111
        )

        truth = pd.to_numeric(
            comparison[
                "hidden_truth"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        estimate = pd.to_numeric(
            comparison[
                "estimate"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(
                truth
            )
            & np.isfinite(
                estimate
            )
        )

        truth = truth[
            valid
        ]

        estimate = estimate[
            valid
        ]

        ax.scatter(
            truth,
            estimate,
            s=32,
            alpha=0.75,
            color=INFO_BLUE,
            edgecolors="white",
            linewidths=0.4,
        )

        if (
            len(
                truth
            )
            > 0
        ):
            lower = float(
                min(
                    truth.min(),
                    estimate.min(),
                )
            )

            upper = float(
                max(
                    truth.max(),
                    estimate.max(),
                )
            )

            if (
                upper
                <= lower
            ):
                upper = (
                    lower
                    + 1.0
                )

            padding = max(
                (
                    upper
                    - lower
                )
                * 0.06,
                1.0,
            )

            lower -= padding
            upper += padding

            ax.plot(
                [
                    lower,
                    upper,
                ],
                [
                    lower,
                    upper,
                ],
                linestyle="--",
                linewidth=1.5,
                color=COBBER_MAROON,
                label="Perfect estimate",
            )

            ax.set_xlim(
                lower,
                upper,
            )

            ax.set_ylim(
                lower,
                upper,
            )

        if (
            field
            == "age"
        ):
            x_label = (
                "Hidden truth (years)"
            )

            y_label = (
                "Model estimate (years)"
            )

        elif (
            field
            == "hourly_wage"
        ):
            x_label = (
                "Hidden truth ($/hour)"
            )

            y_label = (
                "Model estimate ($/hour)"
            )

        elif (
            field
            == "household_income"
        ):
            x_label = (
                "Hidden truth ($)"
            )

            y_label = (
                "Model estimate ($)"
            )

        else:
            x_label = (
                "Hidden truth"
            )

            y_label = (
                "Model estimate"
            )

        ax.set_title(
            method_label,
            fontweight="bold",
        )

        ax.set_xlabel(
            x_label
        )

        ax.set_ylabel(
            y_label
        )

        ax.grid(
            True,
            alpha=0.20,
        )

        ax.legend(
            loc="lower right"
        )

        figure.tight_layout()

        button_row = QHBoxLayout()

        button_row.addStretch()

        close_btn = QPushButton(
            "Close"
        )

        close_btn.setFixedWidth(
            110
        )

        close_btn.clicked.connect(
            self.accept
        )

        button_row.addWidget(
            close_btn
        )

        layout.addLayout(
            button_row
        )


class IndividualEstimatesDialog(
    QDialog
):
    """
    Popup containing the full record-level Test Gap comparison table.
    """

    def __init__(
        self,
        comparison: pd.DataFrame,
        field: str,
        method_label: str,
        parent=None,
    ):
        super().__init__(
            parent
        )

        self.setWindowTitle(
            f"{method_label} — Individual Test Gap Estimates"
        )

        self.resize(
            760,
            620,
        )

        layout = QVBoxLayout(
            self
        )

        explanation = QLabel(
            f"<b>{method_label}</b><br>"
            "Compare each hidden true value with the model estimate and "
            "the resulting absolute error."
        )

        explanation.setWordWrap(
            True
        )

        explanation.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 10px;"
        )

        layout.addWidget(
            explanation
        )

        table = QTableView()

        model = ModelPredictionTableModel(
            comparison,
            field,
        )

        table.setModel(
            model
        )

        # Keep a reference so the model lives as long as the dialog.
        self.table_model = model

        table.verticalHeader().setVisible(
            False
        )

        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        table.setAlternatingRowColors(
            True
        )

        widths = {
            0: 130,
            1: 150,
            2: 150,
            3: 150,
        }

        for (
            column_index,
            column_width,
        ) in widths.items():
            table.setColumnWidth(
                column_index,
                column_width,
            )

        table.horizontalHeader().setStretchLastSection(
            False
        )

        layout.addWidget(
            table,
            stretch=1,
        )

        button_row = QHBoxLayout()

        button_row.addStretch()

        close_btn = QPushButton(
            "Close"
        )

        close_btn.setFixedWidth(
            110
        )

        close_btn.clicked.connect(
            self.accept
        )

        button_row.addWidget(
            close_btn
        )

        layout.addLayout(
            button_row
        )


# ---------------------------------------------------------------------
# Tab 4: Can a Model Beat the Baseline?
# ---------------------------------------------------------------------


class CompareModelsPage(QWidget):

    MODEL_KEYS = ("linear_regression", "knn", "random_forest")

    def __init__(self, main: "CobberHumImputeApp"):
        super().__init__(main)
        self.main = main

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "<b>Can a Model Beat the Baseline?</b><br>"
            "Ravi first tests all three models on the Age Test Gap. When that "
            "experiment is complete, CobberHumImpute reveals the same tests "
            "already prepared for hourly wage and household income. The goal "
            "is to compare evidence, not repeat the same button sequence three times."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background-color: #FAFAFA; border: 1px solid #D6D6D6; "
            "border-radius: 5px; padding: 10px;"
        )
        layout.addWidget(intro)

        controls_box = QGroupBox("Test the Age models")
        controls = QGridLayout(controls_box)
        controls.setColumnStretch(1, 1)
        controls.addWidget(QLabel("<b>Field under investigation</b>"), 0, 0)
        self.field_label = QLabel("Age")
        controls.addWidget(self.field_label, 0, 1)
        controls.addWidget(QLabel("<b>Mean baseline MAE</b>"), 1, 0)
        self.baseline_label = QLabel("—")
        self.baseline_label.setStyleSheet("color: #6C1D45; font-weight: bold;")
        controls.addWidget(self.baseline_label, 1, 1)
        controls.addWidget(QLabel("<b>Model</b>"), 2, 0)

        method_row = QHBoxLayout()
        self.method_combo = QComboBox()
        for key in self.MODEL_KEYS:
            self.method_combo.addItem(MODEL_LABELS[key], key)
        self.method_combo.setFixedWidth(190)
        self.test_btn = QPushButton("Test This Model")
        self.test_btn.setFixedWidth(150)
        self.test_btn.setStyleSheet(
            "background-color: #6C1D45; color: white; font-weight: bold; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 6px 10px;"
        )
        method_row.addWidget(self.method_combo)
        method_row.addSpacing(12)
        method_row.addWidget(self.test_btn)
        method_row.addStretch()
        controls.addLayout(method_row, 2, 1)

        self.method_description = QLabel()
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet("color: #555555;")
        controls.addWidget(self.method_description, 3, 1)
        layout.addWidget(controls_box)

        comparison_box = QGroupBox("Model comparison")
        comparison_layout = QVBoxLayout(comparison_box)
        comparison_help = QLabel(
            "Age results appear as you test each model. After all three Age "
            "models are tested, the prepared hourly-wage and household-income "
            "results appear automatically."
        )
        comparison_help.setWordWrap(True)
        comparison_help.setStyleSheet("color: #555555;")
        comparison_layout.addWidget(comparison_help)

        self.comparison_table = QTableView()
        self.comparison_model = ModelComparisonTableModel()
        self.comparison_table.setModel(self.comparison_model)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.comparison_table.setAlternatingRowColors(False)
        self.comparison_table.setMinimumHeight(280)
        comparison_layout.addWidget(self.comparison_table)
        layout.addWidget(comparison_box, stretch=1)

        inspect_box = QGroupBox("Inspect the evidence")
        inspect_layout = QHBoxLayout(inspect_box)
        inspect_layout.addWidget(QLabel("<b>Field:</b>"))
        self.view_field_combo = QComboBox()
        self.view_field_combo.addItem("Age", "age")
        self.view_field_combo.setFixedWidth(165)
        inspect_layout.addWidget(self.view_field_combo)

        inspect_layout.addWidget(QLabel("<b>Model:</b>"))
        self.view_model_combo = QComboBox()
        self.view_model_combo.setFixedWidth(190)
        inspect_layout.addWidget(self.view_model_combo)
        inspect_layout.addStretch()

        self.graph_btn = QPushButton("View Estimate Graph")
        self.individual_btn = QPushButton("View Individual Estimates")
        for button in (self.graph_btn, self.individual_btn):
            button.setStyleSheet(
                "background-color: #6C1D45; color: white; font-weight: bold; "
                "border: 1px solid #6C1D45; border-radius: 4px; padding: 7px 12px;"
            )
        inspect_layout.addWidget(self.graph_btn)
        inspect_layout.addWidget(self.individual_btn)
        layout.addWidget(inspect_box)

        self.prompt = QLabel(
            "<b>Think about it:</b> A model is more complicated than the mean. "
            "Did it actually earn that extra complexity?"
        )
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet(
            "background-color: #FAFAFA; border-left: 4px solid #6C1D45; padding: 8px;"
        )
        layout.addWidget(self.prompt)

        self.method_combo.currentIndexChanged.connect(self.method_changed)
        self.test_btn.clicked.connect(self.test_method)
        self.view_field_combo.currentIndexChanged.connect(self.refresh_view_models)
        self.graph_btn.clicked.connect(self.open_graph)
        self.individual_btn.clicked.connect(self.open_individual_estimates)

        self.method_changed()
        self.refresh()

    def format_metric(self, value: float, field: str) -> str:
        if field == "age":
            return f"{value:.2f} years"
        if field == "hourly_wage":
            return f"${value:,.2f}/hour"
        if field == "household_income":
            return f"${value:,.0f}"
        return f"{value:,.2f}"

    def selected_method_key(self) -> str:
        value = self.method_combo.currentData()
        return "linear_regression" if value is None else str(value)

    def method_changed(self) -> None:
        self.method_description.setText(
            MODEL_DESCRIPTIONS[self.selected_method_key()]
        )

    def valid_age_baseline(self):
        result = self.main.baseline_results.get("age")
        if result is None:
            return None
        if result["test_gap_indices"] != self.main.test_gap_indices:
            return None
        return result

    def refresh(self) -> None:
        baseline = self.valid_age_baseline()
        active_age = (
            self.main.test_gap_field == "age"
            and not self.main.test_truth.empty
            and baseline is not None
        )

        if baseline is None:
            self.baseline_label.setText("Establish the Age baseline on the previous tab.")
            self.test_btn.setEnabled(False)
        else:
            self.baseline_label.setText(self.format_metric(baseline["mae"], "age"))
            self.test_btn.setEnabled(active_age)

        self.update_comparison_table()
        self.update_view_fields()

    def test_method(self) -> None:
        field = "age"
        if self.main.test_gap_field != field or self.main.test_truth.empty:
            QMessageBox.information(
                self, "Create an Age Test Gap First",
                "Create the Age Test Gap before testing a model."
            )
            return

        baseline = self.valid_age_baseline()
        if baseline is None:
            QMessageBox.information(
                self, "Establish a Baseline First",
                "Test the mean on the previous tab before testing a model."
            )
            return

        method_key = self.selected_method_key()
        model, features = self.main.make_model(field, method_key)
        working = self.main.working_df.copy()
        train_mask = working[field].notna()
        test_indices = sorted(self.main.test_gap_indices)
        X_train = working.loc[train_mask, features]
        y_train = pd.to_numeric(working.loc[train_mask, field], errors="coerce")
        X_test = working.loc[test_indices, features]

        try:
            model.fit(X_train, y_train)
            estimates = model.predict(X_test)
        except Exception as exc:
            QMessageBox.critical(self, "Model error", str(exc))
            return

        truth = (
            self.main.test_truth.set_index("source_index")
            .loc[test_indices]
            .reset_index()
        )
        comparison = pd.DataFrame({
            "source_index": test_indices,
            "respondent_id": truth["respondent_id"].astype(str),
            "hidden_truth": pd.to_numeric(truth["true_value"], errors="coerce"),
            "estimate": estimates,
        })
        comparison["absolute_error"] = (
            comparison["hidden_truth"] - comparison["estimate"]
        ).abs()
        comparison = comparison.dropna(
            subset=["hidden_truth", "estimate", "absolute_error"]
        ).reset_index(drop=True)

        mae = float(comparison["absolute_error"].mean())
        baseline_mae = float(baseline["mae"])
        difference_pct = 100.0 * (mae - baseline_mae) / baseline_mae
        if mae < baseline_mae:
            comparison_text = f"{abs(difference_pct):.1f}% lower error"
        elif mae > baseline_mae:
            comparison_text = f"{abs(difference_pct):.1f}% higher error"
        else:
            comparison_text = "Same error as baseline"

        self.main.model_results.setdefault(field, {})[method_key] = {
            "field": field,
            "method_key": method_key,
            "method_label": MODEL_LABELS[method_key],
            "mae": mae,
            "baseline_mae": baseline_mae,
            "comparison_text": comparison_text,
            "comparison": comparison.copy(),
            "test_gap_indices": self.main.test_gap_indices.copy(),
            "n_test": len(comparison),
        }

        self.update_comparison_table()
        self.update_view_fields()
        self.main.decide_page.refresh()

    def update_comparison_table(self) -> None:
        rows = []
        age_results = self.main.model_results.get("age", {})
        baseline = self.valid_age_baseline()

        for method_key in self.MODEL_KEYS:
            result = age_results.get(method_key)
            valid = (
                result is not None
                and baseline is not None
                and result["test_gap_indices"] == baseline["test_gap_indices"]
            )
            rows.append({
                "field": "Age",
                "field_key": "age",
                "model": MODEL_LABELS[method_key],
                "mae": result["mae"] if valid else np.nan,
                "comparison": result["comparison_text"] if valid else "Not tested yet",
            })

        if self.main.age_models_complete():
            for field in ("hourly_wage", "household_income"):
                for method_key in self.MODEL_KEYS:
                    result = self.main.model_results[field][method_key]
                    rows.append({
                        "field": FIELD_LABELS[field],
                        "field_key": field,
                        "model": MODEL_LABELS[method_key],
                        "mae": result["mae"],
                        "comparison": result["comparison_text"],
                    })

        frame = pd.DataFrame(rows)
        self.comparison_model.set_data(frame)
        widths = {0: 155, 1: 190, 2: 135, 3: 200}
        for index, width in widths.items():
            self.comparison_table.setColumnWidth(index, width)
        self.comparison_table.horizontalHeader().setStretchLastSection(True)

    def update_view_fields(self) -> None:
        current = self.view_field_combo.currentData()
        self.view_field_combo.blockSignals(True)
        self.view_field_combo.clear()
        self.view_field_combo.addItem("Age", "age")
        if self.main.age_models_complete():
            self.view_field_combo.addItem("Hourly wage", "hourly_wage")
            self.view_field_combo.addItem("Household income", "household_income")
        index = self.view_field_combo.findData(current)
        if index >= 0:
            self.view_field_combo.setCurrentIndex(index)
        self.view_field_combo.blockSignals(False)
        self.refresh_view_models()

    def refresh_view_models(self) -> None:
        field = self.view_field_combo.currentData()
        self.view_model_combo.clear()
        if field is None:
            self.graph_btn.setEnabled(False)
            self.individual_btn.setEnabled(False)
            return

        results = self.main.model_results.get(str(field), {})
        for method_key in self.MODEL_KEYS:
            if method_key in results:
                self.view_model_combo.addItem(MODEL_LABELS[method_key], method_key)

        enabled = self.view_model_combo.count() > 0
        self.graph_btn.setEnabled(enabled)
        self.individual_btn.setEnabled(enabled)

    def current_view_result(self):
        field = self.view_field_combo.currentData()
        method_key = self.view_model_combo.currentData()
        if field is None or method_key is None:
            return None
        return self.main.model_results.get(str(field), {}).get(str(method_key))

    def open_graph(self) -> None:
        result = self.current_view_result()
        if result is None:
            QMessageBox.information(self, "No Model Result", "Choose a tested model first.")
            return
        EstimateGraphDialog(
            comparison=result["comparison"],
            field=result["field"],
            method_label=result["method_label"],
            parent=self,
        ).exec()

    def open_individual_estimates(self) -> None:
        result = self.current_view_result()
        if result is None:
            QMessageBox.information(self, "No Model Result", "Choose a tested model first.")
            return
        IndividualEstimatesDialog(
            comparison=result["comparison"],
            field=result["field"],
            method_label=result["method_label"],
            parent=self,
        ).exec()

# ---------------------------------------------------------------------
# Tab 5: Decide What to Do
# ---------------------------------------------------------------------


class DecideWhatToDoPage(QWidget):

    MODEL_KEYS = ("linear_regression", "knn", "random_forest")

    def __init__(self, main: "CobberHumImputeApp"):
        super().__init__(main)
        self.main = main
        self.method_buttons = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            "<b>Decide What to Do</b><br>"
            "Ravi now has evidence for all three fields. The Age evidence came "
            "from the experiment you completed. The wage and household-income "
            "evidence was prepared the same way. Use the evidence to make a "
            "separate decision for each kind of gap."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background-color: #FAFAFA; border: 1px solid #D6D6D6; "
            "border-radius: 5px; padding: 10px;"
        )
        layout.addWidget(intro)

        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("<b>Field to decide:</b>"))
        self.field_combo = QComboBox()
        for field in CORE_FIELDS:
            self.field_combo.addItem(FIELD_LABELS[field], field)
        self.field_combo.setFixedWidth(190)
        field_row.addWidget(self.field_combo)
        field_row.addStretch()
        layout.addLayout(field_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        evidence_box = QGroupBox("Evidence for this field")
        evidence = QGridLayout(evidence_box)
        evidence.setColumnStretch(1, 1)
        labels = [
            ("Field", "field_label"),
            ("Missing ages", "real_gaps_label"),
            ("Known missingness mechanism", "mechanism_label"),
            ("Mean baseline MAE", "baseline_label"),
            ("Tested models", "models_tested_label"),
        ]
        for row, (title, attr) in enumerate(labels):
            evidence.addWidget(QLabel(f"<b>{title}</b>"), row, 0)
            widget = QLabel("—")
            setattr(self, attr, widget)
            evidence.addWidget(widget, row, 1)
        self.mechanism_label.setStyleSheet("font-weight: bold; color: #6C1D45;")
        self.evidence_note = QLabel("")
        self.evidence_note.setWordWrap(True)
        self.evidence_note.setStyleSheet("color: #555555;")
        evidence.addWidget(self.evidence_note, 5, 0, 1, 2)
        grid.addWidget(evidence_box, 0, 0)

        decision_box = QGroupBox("What should Ravi do?")
        decision_layout = QVBoxLayout(decision_box)
        self.decision_group = QButtonGroup(self)
        self.leave_radio = QRadioButton("Leave the gaps visible")
        self.estimate_radio = QRadioButton("Estimate the missing values")
        self.exclude_radio = QRadioButton(
            "Exclude incomplete records from an analysis that requires this field"
        )
        for button in (self.leave_radio, self.estimate_radio, self.exclude_radio):
            self.decision_group.addButton(button)
            decision_layout.addWidget(button)
            if button is self.leave_radio:
                help_text = "Keep the missing values visible and do not add estimates."
            elif button is self.estimate_radio:
                help_text = "Use one of the tested models to estimate the missing ages."
            else:
                help_text = (
                    "Keep the survey record intact, but omit the respondent "
                    "from an analysis that requires this field."
                )
            helper = QLabel(help_text)
            helper.setWordWrap(True)
            helper.setStyleSheet("color: #666666; margin-left: 22px;")
            decision_layout.addWidget(helper)
        decision_layout.addStretch()
        grid.addWidget(decision_box, 0, 1)

        self.method_box = QGroupBox("If Ravi estimates, which model should he use?")
        self.method_layout = QVBoxLayout(self.method_box)
        self.method_group = QButtonGroup(self)
        self.no_methods_label = QLabel("No model results are available for this field yet.")
        self.no_methods_label.setWordWrap(True)
        self.no_methods_label.setStyleSheet("color: #555555;")
        self.method_layout.addWidget(self.no_methods_label)
        self.method_box.setVisible(False)
        grid.addWidget(self.method_box, 1, 0)

        consequence_box = QGroupBox("What would this decision do?")
        consequence_layout = QVBoxLayout(consequence_box)
        self.consequence = QLabel("Choose an option to preview its consequence.")
        self.consequence.setWordWrap(True)
        self.consequence.setStyleSheet(
            "background-color: #FFFFFF; border: 1px solid #CFCFCF; "
            "border-radius: 4px; padding: 12px;"
        )
        consequence_layout.addWidget(self.consequence)
        self.save_btn = QPushButton("Save Decision")
        self.save_btn.setFixedWidth(140)
        self.save_btn.setStyleSheet(
            "background-color: #6C1D45; color: white; font-weight: bold; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 7px 12px;"
        )
        consequence_layout.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.saved_label = QLabel("")
        self.saved_label.setWordWrap(True)
        self.saved_label.setStyleSheet("color: #6C1D45; font-weight: bold;")
        consequence_layout.addWidget(self.saved_label)
        grid.addWidget(consequence_box, 1, 1)

        layout.addLayout(grid, stretch=1)

        boundary = QLabel(
            "<b>Remember:</b> A missingness mechanism describes how a gap "
            "may have entered the data. Test Gap performance shows how well "
            "an estimate worked when the answer was known. Neither one "
            "automatically determines what Ravi should do with a missing age."
        )
        boundary.setWordWrap(True)
        boundary.setStyleSheet(
            "background-color: #FAFAFA; border-left: 4px solid #6C1D45; padding: 8px;"
        )
        layout.addWidget(boundary)

        self.field_combo.currentIndexChanged.connect(self.refresh)
        self.leave_radio.toggled.connect(self.decision_changed)
        self.estimate_radio.toggled.connect(self.decision_changed)
        self.exclude_radio.toggled.connect(self.decision_changed)
        self.save_btn.clicked.connect(self.save_decision)
        self.refresh()

    def selected_field(self) -> str:
        value = self.field_combo.currentData()
        return "age" if value is None else str(value)

    def format_metric(self, value: float, field: str) -> str:
        if field == "age":
            return f"{value:.2f} years"
        if field == "hourly_wage":
            return f"${value:,.2f}/hour"
        if field == "household_income":
            return f"${value:,.0f}"
        return f"{value:,.2f}"

    def current_decision_key(self) -> Optional[str]:
        if self.leave_radio.isChecked():
            return "leave_visible"
        if self.estimate_radio.isChecked():
            return "estimate"
        if self.exclude_radio.isChecked():
            return "exclude"
        return None

    def clear_method_buttons(self) -> None:
        for button in list(self.method_buttons.values()):
            self.method_group.removeButton(button)
            self.method_layout.removeWidget(button)
            button.deleteLater()
        self.method_buttons = {}

    def rebuild_method_choices(self, field: str) -> None:
        self.clear_method_buttons()
        field_results = self.main.model_results.get(field, {})
        available = []
        for method_key in self.MODEL_KEYS:
            result = field_results.get(method_key)
            if result is not None:
                available.append((method_key, result))

        self.no_methods_label.setVisible(len(available) == 0)
        for method_key, result in available:
            button = QRadioButton(
                f"{result['method_label']} — "
                f"MAE {self.format_metric(result['mae'], field)} "
                f"({result['comparison_text']})"
            )
            self.method_group.addButton(button)
            self.method_layout.addWidget(button)
            self.method_buttons[method_key] = button
            button.toggled.connect(self.method_changed)

    def selected_model_key(self) -> Optional[str]:
        for method_key, button in self.method_buttons.items():
            if button.isChecked():
                return method_key
        return None

    def evidence_note_for_field(self, field: str) -> str:
        if field == "age":
            return "The real age gaps are a known MCAR case in this teaching survey."
        if field == "hourly_wage":
            return (
                "The real wage gaps are a known MAR case tied to recorded "
                "work schedule and occupation information."
            )
        if field == "household_income":
            return (
                "The real household-income gaps are a known MNAR case. "
                "People with lower underlying incomes were more likely "
                "to leave the value missing."
            )
        return ""

    def refresh(self) -> None:
        field = self.selected_field()
        self.saved_label.setText("")

        ready = self.main.age_models_complete()
        for button in (self.leave_radio, self.estimate_radio, self.exclude_radio):
            button.setEnabled(ready)
        self.save_btn.setEnabled(ready)

        self.field_label.setText(FIELD_LABELS[field])
        real_gaps = int(self.main.missing_df[field].isna().sum())
        self.real_gaps_label.setText(str(real_gaps))
        self.mechanism_label.setText(MECHANISM_LABELS[field])

        baseline = self.main.baseline_results.get(field)
        if baseline is not None:
            self.baseline_label.setText(self.format_metric(baseline["mae"], field))
        else:
            self.baseline_label.setText("Not established")

        field_results = self.main.model_results.get(field, {})
        self.models_tested_label.setText(f"{len(field_results)} of 3")
        self.evidence_note.setText(self.evidence_note_for_field(field))
        self.rebuild_method_choices(field)
        self.estimate_radio.setEnabled(ready and len(field_results) > 0)

        self.decision_group.setExclusive(False)
        self.leave_radio.setChecked(False)
        self.estimate_radio.setChecked(False)
        self.exclude_radio.setChecked(False)
        self.decision_group.setExclusive(True)

        saved = self.main.decision_results.get(field)
        if saved is not None:
            decision_key = saved["decision"]
            if decision_key == "leave_visible":
                self.leave_radio.setChecked(True)
            elif decision_key == "estimate":
                self.estimate_radio.setChecked(True)
                button = self.method_buttons.get(saved.get("method_key"))
                if button is not None:
                    button.setChecked(True)
            elif decision_key == "exclude":
                self.exclude_radio.setChecked(True)
            self.saved_label.setText("Saved decision loaded.")

        if not ready:
            self.consequence.setText(
                "Complete all three Age model tests before making the three decisions."
            )
            self.method_box.setVisible(False)
        else:
            self.decision_changed()

    def decision_changed(self) -> None:
        if not self.main.age_models_complete():
            return
        self.method_box.setVisible(self.current_decision_key() == "estimate")
        self.update_consequence()

    def method_changed(self) -> None:
        self.update_consequence()

    def update_consequence(self) -> None:
        field = self.selected_field()
        real_gaps = int(self.main.missing_df[field].isna().sum())
        field_label = FIELD_LABELS[field].lower()
        decision_key = self.current_decision_key()

        if decision_key is None:
            self.consequence.setText("Choose an option to preview its consequence.")
            return

        if decision_key == "leave_visible":
            self.consequence.setText(
                f"<b>{real_gaps} real {field_label} gaps would remain visible.</b><br><br>"
                "Ravi would add no estimated values. The rest of each survey "
                "record would remain available for analyses that do not require this field."
            )
            return

        if decision_key == "exclude":
            self.consequence.setText(
                f"<b>{real_gaps} respondents would be excluded from an analysis "
                f"that requires {field_label}.</b><br><br>"
                "Their survey records would remain intact. They simply would not "
                "contribute to that particular analysis."
            )
            return

        method_key = self.selected_model_key()
        if method_key is None:
            self.consequence.setText(
                "<b>Choose one of the tested models.</b><br><br>"
                "The consequence preview will update after you select the model Ravi would use."
            )
            return

        result = self.main.model_results[field][method_key]
        text = (
            f"<b>{real_gaps} real {field_label} gaps would receive estimated values.</b><br><br>"
            f"<b>Model:</b> {result['method_label']}<br>"
            f"<b>Test Gap MAE:</b> {self.format_metric(result['mae'], field)}<br>"
            f"<b>Compared with the mean baseline:</b> {result['comparison_text']}<br><br>"
            "The new values would be marked as estimates rather than observations."
        )

        if field == "household_income":
            text += (
                "<br><br><span style='color:#6C1D45;'><b>Important uncertainty:</b> "
                "The real household-income gaps are MNAR. The people with real "
                "gaps may differ systematically from the observed people used "
                "to test the models.</span>"
            )
        self.consequence.setText(text)

    def save_decision(self) -> None:
        field = self.selected_field()
        if not self.main.age_models_complete():
            return

        decision_key = self.current_decision_key()
        if decision_key is None:
            QMessageBox.information(
                self, "Choose a Decision", "Choose what Ravi should do before saving the decision."
            )
            return

        method_key = None
        if decision_key == "estimate":
            method_key = self.selected_model_key()
            if method_key is None:
                QMessageBox.information(
                    self, "Choose a Model", "Choose one of the tested models before saving the decision."
                )
                return

        baseline = self.main.baseline_results.get(field)
        test_indices = set() if baseline is None else baseline["test_gap_indices"].copy()
        saved = {
            "field": field,
            "decision": decision_key,
            "decision_label": DECISION_LABELS[decision_key],
            "method_key": method_key,
            "test_gap_indices": test_indices,
        }
        if method_key is not None:
            result = self.main.model_results[field][method_key]
            saved["method_label"] = result["method_label"]
            saved["model_mae"] = result["mae"]

        self.main.decision_results[field] = saved
        self.main.update_return_tab_state()

        # If this was the third saved decision, make the final tab
        # immediately available and refresh its cumulative record.
        if self.main.all_decisions_saved():
            self.main.tabs.setTabEnabled(
                self.main.return_tab_index,
                True,
            )
            self.main.return_record_page.setEnabled(True)
            self.main.return_record_page.refresh()

        remaining = [
            FIELD_LABELS[item]
            for item in CORE_FIELDS
            if item not in self.main.decision_results
        ]
        if remaining:
            self.saved_label.setText(
                "Decision saved for " + FIELD_LABELS[field] + ". Still to decide: "
                + ", ".join(remaining) + "."
            )
        else:
            self.saved_label.setText(
                "All three decisions are saved. Return to the Record is now available."
            )

# ---------------------------------------------------------------------
# Tab 6: Return to the Record
# ---------------------------------------------------------------------


class ReturnRecordTableModel(QAbstractTableModel):
    """Display the cumulative record after all three saved decisions."""

    def __init__(
        self,
        frame: pd.DataFrame,
        estimated_cells=None,
        excluded_cells=None,
        parent=None,
    ):
        super().__init__(parent)
        self.frame = frame.copy()
        self.estimated_cells = set(estimated_cells or [])
        self.excluded_cells = set(excluded_cells or [])

    def rowCount(self, parent=QModelIndex()):
        return len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return len(self.frame.columns)

    def headerData(
        self,
        section,
        orientation,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            column = self.frame.columns[section]
            return FIELD_LABELS.get(
                column,
                column.replace('_', ' ').title(),
            )

        return str(section + 1)

    def data(
        self,
        index,
        role=Qt.ItemDataRole.DisplayRole,
    ):
        if not index.isValid():
            return None

        source_index = self.frame.index[index.row()]
        column = self.frame.columns[index.column()]
        value = self.frame.iloc[index.row(), index.column()]

        cell_key = (source_index, column)
        is_estimated = cell_key in self.estimated_cells
        is_excluded = cell_key in self.excluded_cells
        is_core_field = column in CORE_FIELDS
        is_missing = is_core_field and pd.isna(value)

        if role == Qt.ItemDataRole.DisplayRole:
            if is_core_field:
                if is_missing:
                    return 'missing (exclude)' if is_excluded else 'missing'

                if column == 'age':
                    shown = (
                        f'{float(value):.1f}'
                        if is_estimated
                        else f'{float(value):.0f}'
                    )
                elif column == 'hourly_wage':
                    shown = f'${float(value):,.2f}'
                elif column == 'household_income':
                    shown = f'${float(value):,.0f}'
                else:
                    shown = str(value)

                return f'≈ {shown}' if is_estimated else shown

            if pd.isna(value):
                return 'missing'

            return format_value(value, column)

        if role == Qt.ItemDataRole.BackgroundRole:
            if is_estimated:
                return QBrush(QColor('#E8F3EA'))

            if is_excluded:
                return QBrush(QColor('#F2F2F2'))

            if is_missing:
                return QBrush(QColor('#F6EAF0'))

        if role == Qt.ItemDataRole.ForegroundRole:
            if is_estimated:
                return QBrush(QColor('#2F6B3C'))

            if is_excluded:
                return QBrush(QColor('#666666'))

            if is_missing:
                return QBrush(QColor(COBBER_MAROON))

        if role == Qt.ItemDataRole.FontRole:
            if is_estimated or is_excluded or is_missing:
                font = QFont()
                font.setBold(True)
                return font

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if column in {
                'age',
                'years_in_neighborhood',
                'hourly_wage',
                'household_income',
            }:
                return int(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )

        if role == Qt.ItemDataRole.ToolTipRole:
            if is_estimated:
                return (
                    'This value was estimated after Ravi tested the method. '
                    'It was not reported by the respondent.'
                )

            if is_excluded:
                return (
                    'This value remains missing. The respondent would be '
                    'excluded only from an analysis that requires this field.'
                )

            if is_missing:
                return (
                    'This gap remains visible. No estimated value was added.'
                )

        return None


class ReturnToRecordPage(QWidget):

    def __init__(self, main: 'CobberHumImputeApp'):
        super().__init__(main)
        self.main = main

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        intro = QLabel(
            '<b>Return to the Record</b><br>'
            'Ravi has now made a decision about each kind of missing '
            'information in the survey. Return to the record and look at '
            'what those decisions changed. Some gaps may remain visible, '
            'some values may have been estimated, and some respondents may '
            'be omitted only from analyses that require a particular field.'
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(
            'background-color: #FAFAFA; '
            'border: 1px solid #D6D6D6; '
            'border-radius: 5px; '
            'padding: 10px;'
        )
        layout.addWidget(intro)

        # -------------------------------------------------------------
        # Cumulative decision summary
        # -------------------------------------------------------------

        summary_box = QGroupBox('Saved decisions')
        summary = QGridLayout(summary_box)
        summary.setColumnStretch(0, 1)
        summary.setColumnStretch(1, 2)
        summary.setColumnStretch(2, 2)

        summary.addWidget(QLabel('<b>Field</b>'), 0, 0)
        summary.addWidget(QLabel('<b>Ravi\'s decision</b>'), 0, 1)
        summary.addWidget(QLabel('<b>Method, if estimated</b>'), 0, 2)

        self.summary_decision_labels = {}
        self.summary_method_labels = {}

        for row, field in enumerate(CORE_FIELDS, start=1):
            field_label = QLabel(FIELD_LABELS[field])
            field_label.setStyleSheet('font-weight: bold;')
            summary.addWidget(field_label, row, 0)

            decision_label = QLabel('—')
            decision_label.setWordWrap(True)
            summary.addWidget(decision_label, row, 1)
            self.summary_decision_labels[field] = decision_label

            method_label = QLabel('—')
            method_label.setWordWrap(True)
            summary.addWidget(method_label, row, 2)
            self.summary_method_labels[field] = method_label

        layout.addWidget(summary_box)

        # -------------------------------------------------------------
        # Cumulative survey record
        # -------------------------------------------------------------

        record_box = QGroupBox('What does the record look like now?')
        record_layout = QVBoxLayout(record_box)

        self.legend = QLabel(
            "<span style='background-color:#E8F3EA; color:#2F6B3C; "
            "padding:2px 6px;'><b>≈ estimated value</b></span> "
            "marks a value added by a model. "
            "<span style='background-color:#F6EAF0; color:#6C1D45; "
            "padding:2px 6px;'><b>missing</b></span> "
            "marks a gap Ravi chose to leave visible. "
            "<span style='background-color:#F2F2F2; color:#666666; "
            "padding:2px 6px;'><b>missing (exclude)</b></span> "
            "marks a respondent omitted only from analyses that require "
            "that field."
        )
        self.legend.setWordWrap(True)
        record_layout.addWidget(self.legend)

        self.table = QTableView()
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setFixedHeight(42)
        record_layout.addWidget(self.table, stretch=1)

        layout.addWidget(record_box, stretch=1)

        principle = QLabel(
            '<b>An estimate is not a recovered fact.</b> '
            'Estimated values remain visibly marked, and missing values do '
            'not disappear simply because Ravi has decided how to work '
            'around them.'
        )
        principle.setWordWrap(True)
        principle.setStyleSheet(
            'background-color: #FAFAFA; '
            'border-left: 4px solid #6C1D45; '
            'padding: 8px;'
        )
        layout.addWidget(principle)

        self.refresh()

    def show_not_ready(self) -> None:
        for field in CORE_FIELDS:
            saved = self.main.decision_results.get(field)

            if saved is None:
                self.summary_decision_labels[field].setText('Not decided yet')
                self.summary_method_labels[field].setText('—')
            else:
                self.summary_decision_labels[field].setText(
                    saved.get('decision_label', '—')
                )
                self.summary_method_labels[field].setText(
                    saved.get('method_label', 'Not applicable')
                    if saved.get('decision') == 'estimate'
                    else 'Not applicable'
                )

        empty = pd.DataFrame(
            columns=[
                'respondent_id',
                'age',
                'hourly_wage',
                'household_income',
                'years_in_neighborhood',
                'housing_status',
                'work_schedule',
            ]
        )

        self.table_model = ReturnRecordTableModel(empty)
        self.table.setModel(self.table_model)

    def model_for_real_gaps(
        self,
        field: str,
        method_key: str,
    ):
        model, features = self.main.make_model(
            field,
            method_key,
        )

        # Apply each saved model to the original student-facing survey.
        # This prevents an estimate made for one field from becoming an
        # input to the model used for another field.
        source = self.main.missing_df
        train_mask = source[field].notna()
        real_gap_indices = list(
            source.index[source[field].isna()]
        )

        if not real_gap_indices:
            return [], []

        X_train = source.loc[train_mask, features]
        y_train = pd.to_numeric(
            source.loc[train_mask, field],
            errors='coerce',
        )
        X_missing = source.loc[real_gap_indices, features]

        model.fit(X_train, y_train)
        estimates = model.predict(X_missing)

        return real_gap_indices, estimates

    def refresh(self) -> None:
        if not self.main.all_decisions_saved():
            self.show_not_ready()
            return

        display = self.main.missing_df.copy()
        estimated_cells = set()
        excluded_cells = set()

        for field in CORE_FIELDS:
            saved = self.main.decision_results[field]
            decision_key = saved['decision']

            self.summary_decision_labels[field].setText(
                saved['decision_label']
            )

            if decision_key == 'estimate':
                self.summary_method_labels[field].setText(
                    saved.get('method_label', '—')
                )

                method_key = saved.get('method_key')

                if method_key is None:
                    self.summary_method_labels[field].setText(
                        'No method saved'
                    )
                    continue

                try:
                    real_gap_indices, estimates = self.model_for_real_gaps(
                        field,
                        method_key,
                    )
                except Exception as exc:
                    QMessageBox.critical(
                        self,
                        'Model error',
                        f'{FIELD_LABELS[field]}: {exc}',
                    )
                    return

                for source_index, estimate in zip(
                    real_gap_indices,
                    estimates,
                ):
                    display.loc[source_index, field] = float(estimate)
                    estimated_cells.add((source_index, field))

            else:
                self.summary_method_labels[field].setText('Not applicable')

                if decision_key == 'exclude':
                    for source_index in display.index[
                        display[field].isna()
                    ]:
                        excluded_cells.add((source_index, field))

        columns = [
            'respondent_id',
            'age',
            'hourly_wage',
            'household_income',
            'years_in_neighborhood',
            'housing_status',
            'work_schedule',
        ]

        frame = display[columns].copy()

        self.table_model = ReturnRecordTableModel(
            frame=frame,
            estimated_cells=estimated_cells,
            excluded_cells=excluded_cells,
        )
        self.table.setModel(self.table_model)

        width_map = {
            'respondent_id': 115,
            'age': 115,
            'hourly_wage': 135,
            'household_income': 155,
            'years_in_neighborhood': 155,
            'housing_status': 145,
            'work_schedule': 125,
        }

        for column_index, column_name in enumerate(columns):
            self.table.setColumnWidth(
                column_index,
                width_map.get(column_name, 130),
            )

        self.table.horizontalHeader().setStretchLastSection(True)


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------


class CobberHumImputeApp(
    QMainWindow
):

    def __init__(
        self,
    ):
        super().__init__()

        self.setWindowTitle(
            "CobberHumImpute — "
            "Ravi's Simpson Street Survey"
        )

        self.resize(
            1220,
            780,
        )

        self.setFont(
            QFont(
                "Lato",
                10,
            )
        )

        try:
            (
                self.complete_df,
                self.missing_df,
                self.missing_truth,
            ) = load_ravi_survey_files()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Data load error",
                str(exc),
            )

            raise

        self.complete_df = (
            self.complete_df
            .copy(
                deep=True
            )
        )

        self.missing_df = (
            self.missing_df
            .copy(
                deep=True
            )
        )

        self.missing_truth = (
            self.missing_truth
            .copy(
                deep=True
            )
        )

        self.working_df = (
            self.missing_df
            .copy(
                deep=True
            )
        )

        # -------------------------------------------------------------
        # Test Gap state
        # -------------------------------------------------------------

        self.test_gap_field: Optional[str] = (
            None
        )

        self.test_gap_indices: set = (
            set()
        )

        self.test_truth = pd.DataFrame(
            columns=[
                "source_index",
                "respondent_id",
                "field",
                "true_value",
            ]
        )

        # -------------------------------------------------------------
        # Baseline state
        # -------------------------------------------------------------

        self.baseline_results = {}

        # -------------------------------------------------------------
        # Model comparison state
        # -------------------------------------------------------------

        self.model_results = {}

        # -------------------------------------------------------------
        # Decision state
        # -------------------------------------------------------------

        self.decision_results = {}

        # Hourly wage and household income use the same fixed Test Gap,
        # baseline, and model-comparison logic as Age, but their evidence is
        # prepared internally so students do not repeat the full workflow.
        self._precompute_supporting_evidence()

        # -------------------------------------------------------------
        # Tabs
        # -------------------------------------------------------------

        self.tabs = QTabWidget()

        self.setCentralWidget(
            self.tabs
        )

        self.explore_page = (
            ExploreSurveyPage(
                self
            )
        )

        self.test_gap_page = (
            CreateTestGapPage(
                self
            )
        )

        self.baseline_page = (
            EstablishBaselinePage(
                self
            )
        )

        self.compare_models_page = (
            CompareModelsPage(
                self
            )
        )

        self.decide_page = (
            DecideWhatToDoPage(
                self
            )
        )

        self.return_record_page = (
            ReturnToRecordPage(
                self
            )
        )

        self.tabs.addTab(
            self.explore_page,
            "Explore the Survey",
        )

        self.tabs.addTab(
            self.test_gap_page,
            "Create a Test Gap",
        )

        self.tabs.addTab(
            self.baseline_page,
            "Establish a Baseline",
        )

        self.tabs.addTab(
            self.compare_models_page,
            "Can a Model Beat the Baseline?",
        )

        self.tabs.addTab(
            self.decide_page,
            "Decide What to Do",
        )

        self.return_tab_index = self.tabs.addTab(
            self.return_record_page,
            "Return to the Record",
        )

        self.tabs.setTabToolTip(
            self.return_tab_index,
            "Available after Ravi has saved decisions for Age, "
            "Hourly wage, and Household income.",
        )

        self.tabs.currentChanged.connect(
            self.tab_changed
        )

        self.update_return_tab_state()


    def all_decisions_saved(
        self,
    ) -> bool:
        return all(
            field in self.decision_results
            for field in CORE_FIELDS
        )


    def age_models_complete(self) -> bool:
        baseline = self.baseline_results.get("age")
        if baseline is None:
            return False

        results = self.model_results.get("age", {})
        for method_key in ("linear_regression", "knn", "random_forest"):
            result = results.get(method_key)
            if result is None:
                return False
            if result["test_gap_indices"] != baseline["test_gap_indices"]:
                return False
        return True

    def _precompute_supporting_evidence(self) -> None:
        for field in ("hourly_wage", "household_income"):
            source = self.missing_df
            observed_indices = source.index[source[field].notna()].to_numpy()
            n_hide = max(1, int(round(TEST_FRACTION * len(observed_indices))))
            rng = np.random.default_rng(TEST_SEEDS[field])
            test_indices = np.sort(
                rng.choice(observed_indices, size=n_hide, replace=False)
            )
            test_index_set = {int(index) for index in test_indices}

            truth = source.loc[test_indices, ["respondent_id", field]].copy()
            truth = truth.rename(columns={field: "true_value"})
            truth["source_index"] = test_indices

            working = source.copy(deep=True)
            working.loc[test_indices, field] = np.nan

            observed = working[field].dropna()
            mean_value = float(observed.mean())
            baseline_comparison = pd.DataFrame({
                "respondent_id": truth["respondent_id"].astype(str).to_numpy(),
                "hidden_truth": pd.to_numeric(truth["true_value"], errors="coerce").to_numpy(),
                "mean_estimate": mean_value,
            })
            baseline_comparison["absolute_error"] = (
                baseline_comparison["hidden_truth"]
                - baseline_comparison["mean_estimate"]
            ).abs()
            baseline_comparison = baseline_comparison.dropna(
                subset=["hidden_truth", "absolute_error"]
            ).reset_index(drop=True)
            baseline_mae = float(baseline_comparison["absolute_error"].mean())

            self.baseline_results[field] = {
                "field": field,
                "mean_estimate": mean_value,
                "mae": baseline_mae,
                "comparison": baseline_comparison.copy(),
                "test_gap_indices": test_index_set.copy(),
                "n_test": len(baseline_comparison),
            }

            self.model_results[field] = {}
            train_mask = working[field].notna()

            for method_key in ("linear_regression", "knn", "random_forest"):
                model, features = self.make_model(field, method_key)
                X_train = working.loc[train_mask, features]
                y_train = pd.to_numeric(
                    working.loc[train_mask, field], errors="coerce"
                )
                X_test = working.loc[test_indices, features]
                model.fit(X_train, y_train)
                estimates = model.predict(X_test)

                comparison = pd.DataFrame({
                    "source_index": test_indices,
                    "respondent_id": truth["respondent_id"].astype(str).to_numpy(),
                    "hidden_truth": pd.to_numeric(
                        truth["true_value"], errors="coerce"
                    ).to_numpy(),
                    "estimate": estimates,
                })
                comparison["absolute_error"] = (
                    comparison["hidden_truth"] - comparison["estimate"]
                ).abs()
                comparison = comparison.dropna(
                    subset=["hidden_truth", "estimate", "absolute_error"]
                ).reset_index(drop=True)
                mae = float(comparison["absolute_error"].mean())
                difference_pct = 100.0 * (mae - baseline_mae) / baseline_mae

                if mae < baseline_mae:
                    comparison_text = f"{abs(difference_pct):.1f}% lower error"
                elif mae > baseline_mae:
                    comparison_text = f"{abs(difference_pct):.1f}% higher error"
                else:
                    comparison_text = "Same error as baseline"

                self.model_results[field][method_key] = {
                    "field": field,
                    "method_key": method_key,
                    "method_label": MODEL_LABELS[method_key],
                    "mae": mae,
                    "baseline_mae": baseline_mae,
                    "comparison_text": comparison_text,
                    "comparison": comparison.copy(),
                    "test_gap_indices": test_index_set.copy(),
                    "n_test": len(comparison),
                }

    def update_return_tab_state(
        self,
    ) -> None:
        if not hasattr(self, "return_tab_index"):
            return

        ready = self.all_decisions_saved()

        self.tabs.setTabEnabled(
            self.return_tab_index,
            ready,
        )

        # Keep the page state synchronized with the tab itself.
        # This is deliberately explicit because the final tab begins disabled.
        self.return_record_page.setEnabled(ready)

        if ready:
            self.return_record_page.refresh()
            self.tabs.setTabToolTip(
                self.return_tab_index,
                "All three decisions are saved. Review the cumulative record.",
            )
        else:
            remaining = [
                FIELD_LABELS[field]
                for field in CORE_FIELDS
                if field not in self.decision_results
            ]

            self.tabs.setTabToolTip(
                self.return_tab_index,
                "Save decisions for: " + ", ".join(remaining),
            )


    def clear_test_gap(
        self,
    ) -> None:
        self.working_df = (
            self.missing_df
            .copy(
                deep=True
            )
        )

        self.test_gap_field = (
            None
        )

        self.test_gap_indices = (
            set()
        )

        self.test_truth = pd.DataFrame(
            columns=[
                "source_index",
                "respondent_id",
                "field",
                "true_value",
            ]
        )

        # Clearing the student-created Age experiment also clears any Age
        # evidence and decision derived from it. Prepared wage and income
        # evidence remains intact.
        self.baseline_results.pop("age", None)
        self.model_results.pop("age", None)
        self.decision_results.pop("age", None)

        if hasattr(self, "return_tab_index"):
            self.update_return_tab_state()


    def make_one_hot_encoder(
        self,
    ):
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )

        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )


    def make_preprocessor(
        self,
        field: str,
        scale_numeric: bool,
    ):
        features = [
            column
            for column
            in MODEL_FEATURES
            if column
            != field
        ]

        categorical = [
            column
            for column
            in CATEGORICAL_FEATURES
            if column
            in features
        ]

        numeric = [
            column
            for column
            in features
            if column
            not in categorical
        ]

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]

        if (
            scale_numeric
        ):
            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        numeric_pipeline = Pipeline(
            numeric_steps
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    self.make_one_hot_encoder(),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    numeric_pipeline,
                    numeric,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical,
                ),
            ]
        )

        return (
            preprocessor,
            features,
        )


    def make_model(
        self,
        field: str,
        method_key: str,
    ):
        if (
            method_key
            == "linear_regression"
        ):
            (
                preprocessor,
                features,
            ) = self.make_preprocessor(
                field,
                scale_numeric=True,
            )

            estimator = (
                LinearRegression()
            )

        elif (
            method_key
            == "knn"
        ):
            (
                preprocessor,
                features,
            ) = self.make_preprocessor(
                field,
                scale_numeric=True,
            )

            estimator = (
                KNeighborsRegressor(
                    n_neighbors=5
                )
            )

        elif (
            method_key
            == "random_forest"
        ):
            (
                preprocessor,
                features,
            ) = self.make_preprocessor(
                field,
                scale_numeric=False,
            )

            estimator = (
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=1704,
                    n_jobs=1,
                )
            )

        else:
            raise ValueError(
                f"Unknown method: "
                f"{method_key}"
            )

        model = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        return (
            model,
            features,
        )



    def tab_changed(
        self,
        index: int,
    ) -> None:
        page = (
            self.tabs
            .widget(
                index
            )
        )

        if (
            page
            is self.test_gap_page
        ):
            self.test_gap_page.refresh()

        elif (
            page
            is self.explore_page
        ):
            self.explore_page.refresh()

        elif (
            page
            is self.baseline_page
        ):
            self.baseline_page.refresh()

        elif (
            page
            is self.compare_models_page
        ):
            self.compare_models_page.refresh()

        elif (
            page
            is self.decide_page
        ):
            self.decide_page.refresh()

        elif (
            page
            is self.return_record_page
        ):
            self.return_record_page.refresh()


# ---------------------------------------------------------------------
# Application style
# ---------------------------------------------------------------------


def apply_app_stylesheet(
    app: QApplication,
) -> None:
    app.setStyleSheet(
        """
        QWidget {
            color: #222222;
            background-color: #ffffff;
        }

        QMainWindow,
        QDialog {
            background-color: #ffffff;
        }

        QGroupBox {
            color: #222222;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #fafafa;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #6c1d45;
            background-color: #fafafa;
        }

        QLabel {
            color: #222222;
            background-color: transparent;
        }

        QComboBox {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 3px;
            padding: 3px 6px;
            min-height: 24px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }

        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #9a9a9a;
        }

        QTableView,
        QTextEdit {
            background-color: #ffffff;
            color: #111111;
            alternate-background-color: #f4f4f4;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #cfcfcf;
        }

        QHeaderView::section {
            background-color: #6c1d45;
            color: #ffffff;
            font-weight: bold;
            padding: 6px;
            border: 1px solid #ffffff;
        }

        QRadioButton {
            spacing: 7px;
        }

        QRadioButton::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #7A7A7A;
            border-radius: 7px;
            background-color: white;
        }

        QRadioButton::indicator:checked {
            background-color: #6C1D45;
            border: 1px solid #6C1D45;
        }

        QRadioButton::indicator:unchecked:hover {
            border: 1px solid #6C1D45;
        }

        QPushButton {
            background-color: #f7f7f7;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            padding: 6px 10px;
        }

        QPushButton:hover {
            background-color: #eeeeee;
        }

        QPushButton:pressed {
            background-color: #dddddd;
        }

        QPushButton:disabled {
            background-color: #eeeeee;
            color: #888888;
        }

        QCheckBox {
            padding: 3px;
        }

        QTabBar::tab {
            background-color: #8a8a8a;
            color: #ffffff;
            font-weight: bold;
            padding: 8px 16px;
            border: 1px solid #ffffff;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background-color: #6c1d45;
            color: #ffffff;
        }

        QTabBar::tab:hover {
            background-color: #6c1d45;
            color: #ffffff;
        }
        """
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    app = QApplication(
        sys.argv
    )

    apply_app_stylesheet(
        app
    )

    try:
        window = (
            CobberHumImputeApp()
        )

    except Exception:
        sys.exit(
            1
        )

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
