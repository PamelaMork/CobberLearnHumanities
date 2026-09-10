#!/usr/bin/env python3
"""
CobberHumImpute.py

CobberHumImpute for the ML for Humanities imputation chapter.

Current teaching structure:
    Explore the Survey
    Create a Test Gap
    Establish a Baseline

The application loads three fixed Ravi Mehta / Simpson Street survey files:

    data/ravi_community_survey_complete.csv
    data/ravi_community_survey_missing.csv
    data/ravi_community_survey_missing_truth.csv

The complete survey and missing-truth files remain internal. Students work
with the fixed survey containing the real teaching gaps.

Dependencies:
    pip install pandas numpy PyQt6 scikit-learn
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

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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

GROUP_FIELDS = {
    "Housing status": "housing_status",
    "Work schedule": "work_schedule",
    "Occupation": "occupation_group",
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
    Distinguish real gaps from artificial test gaps.

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


class ExploreSurveyPage(
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
        # Narrative introduction
        # -------------------------------------------------------------

        intro = QLabel(
            "<b>Ravi's Simpson Street Community Survey</b><br>"
            "Ravi's team collected a short community survey to supplement "
            "the recorded interviews from Simpson Street. The survey includes "
            "basic demographic information, work patterns, housing history, "
            "and concern about neighborhood redevelopment. Some records are "
            "incomplete. Before Ravi decides what to do with the gaps, he "
            "needs to understand the data around them."
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
        # 2 x 2 investigation area
        # -------------------------------------------------------------

        investigation_grid = QGridLayout()

        investigation_grid.setHorizontalSpacing(
            12
        )

        investigation_grid.setVerticalSpacing(
            8
        )

        investigation_grid.setColumnStretch(
            0,
            1,
        )

        investigation_grid.setColumnStretch(
            1,
            1,
        )

        # -------------------------------------------------------------
        # Upper left:
        # Choose missing field
        # -------------------------------------------------------------

        field_box = QGroupBox(
            "Choose the missing field to investigate"
        )

        field_layout = QVBoxLayout(
            field_box
        )

        field_row = QHBoxLayout()

        field_label = QLabel(
            "Field:"
        )

        field_label.setStyleSheet(
            "font-weight: bold;"
        )

        self.field_combo = QComboBox()

        self.field_combo.addItems(
            [
                FIELD_LABELS[field]
                for field
                in CORE_FIELDS
            ]
        )

        self.field_combo.setFixedWidth(
            180
        )

        field_row.addWidget(
            field_label
        )

        field_row.addWidget(
            self.field_combo
        )

        field_row.addStretch()

        field_layout.addLayout(
            field_row
        )

        field_help = QLabel(
            "This is the missing field you are investigating."
        )

        field_help.setWordWrap(
            True
        )

        field_help.setStyleSheet(
            "color: #555555;"
        )

        field_layout.addWidget(
            field_help
        )

        field_layout.addStretch()

        investigation_grid.addWidget(
            field_box,
            0,
            0,
        )

        # -------------------------------------------------------------
        # Upper right:
        # Field description
        # -------------------------------------------------------------

        field_info_box = QGroupBox(
            "Field under investigation"
        )

        field_info_layout = QVBoxLayout(
            field_info_box
        )

        self.about_label = QLabel()

        self.about_label.setStyleSheet(
            "font-weight: bold; "
            "color: #6C1D45;"
        )

        self.about_field = QLabel()

        self.about_field.setWordWrap(
            True
        )

        field_info_layout.addWidget(
            self.about_label
        )

        field_info_layout.addWidget(
            self.about_field
        )

        field_info_layout.addStretch()

        investigation_grid.addWidget(
            field_info_box,
            0,
            1,
        )

        # -------------------------------------------------------------
        # Lower left:
        # Narrow the records
        # -------------------------------------------------------------

        filter_box = QGroupBox(
            "Narrow the records"
        )

        filter_layout = QVBoxLayout(
            filter_box
        )

        filter_help = QLabel(
            "Compare the selected field within different groups."
        )

        filter_help.setWordWrap(
            True
        )

        filter_help.setStyleSheet(
            "color: #555555;"
        )

        filter_layout.addWidget(
            filter_help
        )


        # -------------------------------------------------------------
        # Housing status
        # -------------------------------------------------------------

        housing_row = QHBoxLayout()

        housing_label = QLabel(
            "Housing status:"
        )

        housing_label.setStyleSheet(
            "font-weight: bold;"
        )

        housing_label.setFixedWidth(
            155
        )

        self.housing_combo = QComboBox()

        self.housing_combo.addItem(
            "All housing"
        )

        self.housing_combo.addItems(
            [
                "homeowner",
                "renter",
                "living_with_family",
                "temporary_or_other",
            ]
        )

        self.housing_combo.setFixedWidth(
            160
        )

        housing_row.addWidget(
            housing_label
        )

        housing_row.addWidget(
            self.housing_combo
        )

        housing_row.addStretch()

        filter_layout.addLayout(
            housing_row
        )

        # -------------------------------------------------------------
        # Work schedule
        # -------------------------------------------------------------

        schedule_row = QHBoxLayout()

        schedule_label = QLabel(
            "Work schedule:"
        )

        schedule_label.setStyleSheet(
            "font-weight: bold;"
        )

        schedule_label.setFixedWidth(
            155
        )

        self.schedule_combo = QComboBox()

        self.schedule_combo.addItem(
            "All schedules"
        )

        self.schedule_combo.addItems(
            [
                "day",
                "evening",
                "overnight",
                "rotating",
            ]
        )

        self.schedule_combo.setFixedWidth(
            160
        )

        schedule_row.addWidget(
            schedule_label
        )

        schedule_row.addWidget(
            self.schedule_combo
        )

        schedule_row.addStretch()

        filter_layout.addLayout(
            schedule_row
        )

        # -------------------------------------------------------------
        # Occupation
        # -------------------------------------------------------------

        occupation_row = QHBoxLayout()

        occupation_label = QLabel(
            "Occupation:"
        )

        occupation_label.setStyleSheet(
            "font-weight: bold;"
        )

        occupation_label.setFixedWidth(
            155
        )

        self.occupation_combo = QComboBox()

        self.occupation_combo.addItem(
            "All occupations"
        )

        self.occupation_combo.addItems(
            [
                "service",
                "health_care",
                "education",
                "trades",
                "office",
                "other",
            ]
        )

        self.occupation_combo.setFixedWidth(
            160
        )

        occupation_row.addWidget(
            occupation_label
        )

        occupation_row.addWidget(
            self.occupation_combo
        )

        occupation_row.addStretch()

        filter_layout.addLayout(
            occupation_row
        )

        # -------------------------------------------------------------
        # Missing only + reset
        # -------------------------------------------------------------

        final_filter_row = QHBoxLayout()

        self.missing_only = QCheckBox(
            "Only records where selected field is missing"
        )

        self.reset_btn = QPushButton(
            "Reset Filters"
        )

        self.reset_btn.clicked.connect(
            self.reset_filters
        )

        final_filter_row.addWidget(
            self.missing_only
        )

        final_filter_row.addStretch()

        final_filter_row.addWidget(
            self.reset_btn
        )

        filter_layout.addLayout(
            final_filter_row
        )






        investigation_grid.addWidget(
            filter_box,
            1,
            0,
        )

        # -------------------------------------------------------------
        # Lower right:
        # Missing-data summary
        # -------------------------------------------------------------

        summary_box = QGroupBox(
            "Missing-data summary"
        )

        summary_layout = QVBoxLayout(
            summary_box
        )

        self.summary_label = QLabel(
            "All respondents"
        )

        self.summary_label.setStyleSheet(
            "font-weight: bold; "
            "color: #6C1D45;"
        )

        summary_layout.addWidget(
            self.summary_label
        )

        self.missing_table = QTableView()

        self.missing_table.setMinimumHeight(
            145
        )

        self.missing_table.setMaximumHeight(
            170
        )

        self.missing_table.verticalHeader().setVisible(
            False
        )

        self.missing_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.missing_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )

        self.missing_table.setAlternatingRowColors(
            True
        )

        summary_layout.addWidget(
            self.missing_table
        )

        investigation_grid.addWidget(
            summary_box,
            1,
            1,
        )

        layout.addLayout(
            investigation_grid
        )

        # -------------------------------------------------------------
        # Survey records
        # -------------------------------------------------------------

        self.preview_label = QLabel(
            "Survey records"
        )

        self.preview_label.setStyleSheet(
            "font-weight: bold; "
            "color: #6C1D45;"
        )

        layout.addWidget(
            self.preview_label
        )

        self.table = QTableView()

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

        layout.addWidget(
            self.table,
            stretch=1,
        )

        # -------------------------------------------------------------
        # Thinking prompt
        # -------------------------------------------------------------

        self.prompt = QLabel()

        self.prompt.setWordWrap(
            True
        )

        self.prompt.setStyleSheet(
            "background-color: #FAFAFA; "
            "border-left: 4px solid #6C1D45; "
            "padding: 8px;"
        )

        layout.addWidget(
            self.prompt
        )

        # -------------------------------------------------------------
        # Signals
        # -------------------------------------------------------------

        self.field_combo.currentTextChanged.connect(
            self.refresh
        )

        self.schedule_combo.currentTextChanged.connect(
            self.refresh
        )

        self.occupation_combo.currentTextChanged.connect(
            self.refresh
        )

        self.housing_combo.currentTextChanged.connect(
            self.refresh
        )

        self.missing_only.stateChanged.connect(
            self.refresh
        )

        self.refresh()


    def selected_field(
        self,
    ) -> str:
        label = (
            self.field_combo
            .currentText()
        )

        for field in CORE_FIELDS:
            if (
                FIELD_LABELS[field]
                == label
            ):
                return field

        return CORE_FIELDS[0]


    def subgroup_df(
        self,
    ) -> pd.DataFrame:
        df = (
            self.main
            .missing_df
            .copy()
        )

        schedule = (
            self.schedule_combo
            .currentText()
        )

        if (
            schedule
            != "All schedules"
        ):
            df = df[
                df["work_schedule"]
                == schedule
            ]

        occupation = (
            self.occupation_combo
            .currentText()
        )

        if (
            occupation
            != "All occupations"
        ):
            df = df[
                df["occupation_group"]
                == occupation
            ]

        housing = (
            self.housing_combo
            .currentText()
        )

        if (
            housing
            != "All housing"
        ):
            df = df[
                df["housing_status"]
                == housing
            ]

        return df


    def displayed_df(
        self,
    ) -> pd.DataFrame:
        df = (
            self.subgroup_df()
            .copy()
        )

        field = (
            self.selected_field()
        )

        if (
            self.missing_only
            .isChecked()
        ):
            df = df[
                df[field]
                .isna()
            ]

        return df


    def subgroup_description(
        self,
    ) -> str:
        parts = []

        schedule = (
            self.schedule_combo
            .currentText()
        )

        occupation = (
            self.occupation_combo
            .currentText()
        )

        housing = (
            self.housing_combo
            .currentText()
        )

        if (
            schedule
            != "All schedules"
        ):
            parts.append(
                f"{schedule} workers"
            )

        if (
            occupation
            != "All occupations"
        ):
            parts.append(
                occupation
                .replace(
                    "_",
                    " ",
                )
                + " occupations"
            )

        if (
            housing
            != "All housing"
        ):
            parts.append(
                housing
                .replace(
                    "_",
                    " ",
                )
            )

        if not parts:
            return (
                "all respondents"
            )

        return ", ".join(
            parts
        )


    def reset_filters(
        self,
    ) -> None:
        self.schedule_combo.setCurrentIndex(
            0
        )

        self.occupation_combo.setCurrentIndex(
            0
        )

        self.housing_combo.setCurrentIndex(
            0
        )

        self.missing_only.setChecked(
            False
        )

        self.refresh()


    def refresh(
        self,
    ) -> None:
        field = (
            self.selected_field()
        )

        subgroup = (
            self.subgroup_df()
        )

        displayed = (
            self.displayed_df()
        )

        subgroup_n = len(
            subgroup
        )

        displayed_n = len(
            displayed
        )

        total_n = len(
            self.main
            .missing_df
        )

        field_label = (
            FIELD_LABELS[field]
        )

        # -------------------------------------------------------------
        # Field information
        # -------------------------------------------------------------

        self.about_label.setText(
            field_label
        )

        self.about_field.setText(
            f"{FIELD_DESCRIPTIONS[field]}<br><br>"
            f"<b>{subgroup_n} of {total_n}</b> records are in "
            "the current comparison group."
        )

        # -------------------------------------------------------------
        # Missing-data summary
        # -------------------------------------------------------------

        report = (
            missing_report(
                subgroup
            )
        )

        self.missing_model = (
            MissingSummaryModel(
                report,
                focus_field_label=field_label,
            )
        )

        self.missing_table.setModel(
            self.missing_model
        )

        self.missing_table.resizeColumnsToContents()

        self.missing_table.setColumnWidth(
            0,
            150,
        )

        self.missing_table.setColumnWidth(
            1,
            90,
        )

        self.missing_table.setColumnWidth(
            2,
            100,
        )

        self.missing_table.horizontalHeader().setStretchLastSection(
            False
        )

        self.missing_table.resizeRowsToContents()

        self.summary_label.setText(
            self.subgroup_description()
        )

        # -------------------------------------------------------------
        # Survey records
        # -------------------------------------------------------------

        preview = (
            displayed[
                DISPLAY_COLUMNS
            ]
            .copy()
        )

        self.table_model = (
            SurveyTableModel(
                preview,
                focus_field=field,
            )
        )

        self.table.setModel(
            self.table_model
        )

        self.table.resizeColumnsToContents()

        self.table.horizontalHeader().setStretchLastSection(
            False
        )

        if (
            self.missing_only
            .isChecked()
        ):
            self.preview_label.setText(
                f"Survey records: "
                f"{displayed_n} records with missing "
                f"{field_label.lower()} shown"
            )

        elif (
            subgroup_n
            == total_n
        ):
            self.preview_label.setText(
                f"Survey records: "
                f"{total_n} shown"
            )

        else:
            self.preview_label.setText(
                f"Survey records: "
                f"{subgroup_n} of "
                f"{total_n} shown"
            )

        # -------------------------------------------------------------
        # Thinking prompt
        # -------------------------------------------------------------

        self.prompt.setText(
            f"<b>Look closely:</b> "
            f"{FIELD_QUESTIONS[field]}"
        )


# ---------------------------------------------------------------------
# Tab 2: Create a Test Gap
# ---------------------------------------------------------------------


class CreateTestGapPage(
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
        # Explanation
        # -------------------------------------------------------------

        intro = QLabel(
            "<b>Create a Test Gap</b><br>"
            "Ravi cannot test an estimate against a value that was never "
            "recorded because he does not know the true value. To evaluate "
            "an estimation method, he needs a case where the answer is known. "
            "CobberHumImpute creates that test by temporarily hiding some "
            "observed values while keeping their true values as an answer key."
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
        # Controls
        # -------------------------------------------------------------

        controls_box = QGroupBox(
            "Create a Test Gap"
        )

        controls = QHBoxLayout(
            controls_box
        )

        field_label = QLabel(
            "Field to test:"
        )

        field_label.setStyleSheet(
            "font-weight: bold;"
        )

        self.field_combo = QComboBox()

        self.field_combo.addItems(
            [
                FIELD_LABELS[field]
                for field
                in CORE_FIELDS
            ]
        )

        self.field_combo.setFixedWidth(
            180
        )

        fraction_label = QLabel(
            "<b>Test size:</b> "
            "20% of currently observed values"
        )

        self.create_btn = QPushButton(
            "Create Test Gap"
        )

        self.create_btn.setFixedWidth(
            150
        )

        self.create_btn.setStyleSheet(
            "background-color: #6C1D45; "
            "color: white; "
            "font-weight: bold; "
            "border: 1px solid #6C1D45; "
            "border-radius: 4px; "
            "padding: 6px 10px;"
        )

        self.clear_btn = QPushButton(
            "Clear Test Gap"
        )

        self.clear_btn.setFixedWidth(
            140
        )

        controls.addWidget(
            field_label
        )

        controls.addWidget(
            self.field_combo
        )

        controls.addSpacing(
            20
        )

        controls.addWidget(
            fraction_label
        )

        controls.addStretch()

        controls.addWidget(
            self.create_btn
        )

        controls.addWidget(
            self.clear_btn
        )

        layout.addWidget(
            controls_box
        )

        # -------------------------------------------------------------
        # Mechanism context
        # -------------------------------------------------------------

        self.mechanism_context = QLabel()

        self.mechanism_context.setWordWrap(
            True
        )

        self.mechanism_context.setStyleSheet(
            "background-color: #F7F7F7; "
            "border-left: 4px solid #3E6990; "
            "padding: 8px;"
        )

        layout.addWidget(
            self.mechanism_context
        )

        # -------------------------------------------------------------
        # Real gap versus Test Gap
        # -------------------------------------------------------------

        distinction = QLabel(
            "<b>Do not confuse the two kinds of gaps.</b> "
            "<span style='color:#6C1D45;'><b>Real gaps</b></span> "
            "were already missing from the survey. "
            "<span style='color:#3E6990;'><b>Test Gaps</b></span> "
            "contain values Ravi actually knows, but the app temporarily "
            "hides them for the experiment."
        )

        distinction.setWordWrap(
            True
        )

        distinction.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 8px;"
        )

        layout.addWidget(
            distinction
        )

        # -------------------------------------------------------------
        # Experiment status
        # -------------------------------------------------------------

        status_box = QGroupBox(
            "Experiment status"
        )

        status_layout = QHBoxLayout(
            status_box
        )

        self.observed_status = QLabel()
        self.real_status = QLabel()
        self.test_status = QLabel()

        status_style = (
            "background-color: #FFFFFF; "
            "border: 1px solid #CFCFCF; "
            "border-radius: 4px; "
            "padding: 10px;"
        )

        self.observed_status.setStyleSheet(
            status_style
        )

        self.real_status.setStyleSheet(
            status_style
        )

        self.test_status.setStyleSheet(
            status_style
        )

        status_layout.addWidget(
            self.observed_status
        )

        status_layout.addWidget(
            self.real_status
        )

        status_layout.addWidget(
            self.test_status
        )

        status_layout.addStretch()

        layout.addWidget(
            status_box
        )

        # -------------------------------------------------------------
        # What changed?
        # -------------------------------------------------------------

        self.change_summary = QLabel()

        self.change_summary.setWordWrap(
            True
        )

        self.change_summary.setStyleSheet(
            "background-color: #F7F7F7; "
            "border-left: 4px solid #3E6990; "
            "padding: 8px;"
        )

        layout.addWidget(
            self.change_summary
        )

        # -------------------------------------------------------------
        # Survey records
        # -------------------------------------------------------------

        self.table_label = QLabel(
            "Survey records"
        )

        self.table_label.setStyleSheet(
            "font-weight: bold; "
            "color: #6C1D45;"
        )

        layout.addWidget(
            self.table_label
        )

        self.table = QTableView()

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

        layout.addWidget(
            self.table,
            stretch=1,
        )

        # -------------------------------------------------------------
        # Thinking prompt
        # -------------------------------------------------------------

        prompt = QLabel(
            "<b>Think about it:</b> Why does a Test Gap give Ravi "
            "something the real gaps cannot?"
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

        # -------------------------------------------------------------
        # Signals
        # -------------------------------------------------------------

        self.field_combo.currentTextChanged.connect(
            self.refresh
        )

        self.create_btn.clicked.connect(
            self.create_gap
        )

        self.clear_btn.clicked.connect(
            self.clear_gap
        )

        self.refresh()


    def selected_field(
        self,
    ) -> str:
        label = (
            self.field_combo
            .currentText()
        )

        for field in CORE_FIELDS:
            if (
                FIELD_LABELS[field]
                == label
            ):
                return field

        return CORE_FIELDS[0]


    def set_selected_field(
        self,
        field: str,
    ) -> None:
        if (
            field
            not in CORE_FIELDS
        ):
            return

        index = (
            self.field_combo
            .findText(
                FIELD_LABELS[field]
            )
        )

        if (
            index
            >= 0
        ):
            self.field_combo.setCurrentIndex(
                index
            )


    def create_gap(
        self,
    ) -> None:
        field = (
            self.selected_field()
        )

        if (
            self.main
            .test_gap_field
            is not None
        ):
            current_label = (
                FIELD_LABELS[
                    self.main
                    .test_gap_field
                ]
            )

            reply = QMessageBox.question(
                self,
                "Replace current Test Gap?",
                (
                    f"A Test Gap already exists for {current_label}. "
                    f"Replace it with a Test Gap for "
                    f"{FIELD_LABELS[field]}?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if (
                reply
                != QMessageBox.StandardButton.Yes
            ):
                return

            self.main.clear_test_gap()

        source = (
            self.main
            .missing_df
        )

        observed_indices = (
            source.index[
                source[field]
                .notna()
            ]
            .to_numpy()
        )

        n_hide = max(
            1,
            int(
                round(
                    TEST_FRACTION
                    * len(
                        observed_indices
                    )
                )
            ),
        )

        rng = np.random.default_rng(
            TEST_SEEDS[field]
        )

        test_indices = np.sort(
            rng.choice(
                observed_indices,
                size=n_hide,
                replace=False,
            )
        )

        truth = (
            source.loc[
                test_indices,
                [
                    "respondent_id",
                    field,
                ],
            ]
            .copy()
        )

        truth = truth.rename(
            columns={
                field:
                    "true_value",
            }
        )

        truth[
            "field"
        ] = field

        truth[
            "source_index"
        ] = test_indices

        truth = (
            truth[
                [
                    "source_index",
                    "respondent_id",
                    "field",
                    "true_value",
                ]
            ]
            .reset_index(
                drop=True
            )
        )

        self.main.working_df = (
            self.main
            .missing_df
            .copy(
                deep=True
            )
        )

        self.main.working_df.loc[
            test_indices,
            field,
        ] = np.nan

        self.main.test_gap_field = (
            field
        )

        self.main.test_gap_indices = {
            int(index)
            for index
            in test_indices
        }

        self.main.test_truth = (
            truth
        )

        self.refresh()

        QMessageBox.information(
            self,
            "Test Gap created",
            (
                f"{n_hide} known "
                f"{FIELD_LABELS[field].lower()} values "
                "are now hidden for testing. "
                "Their true values are stored internally."
            ),
        )


    def clear_gap(
        self,
    ) -> None:
        if (
            self.main
            .test_gap_field
            is None
        ):
            QMessageBox.information(
                self,
                "No Test Gap",
                "There is no Test Gap to clear.",
            )
            return

        self.main.clear_test_gap()

        self.refresh()


    def refresh(
        self,
    ) -> None:
        field = (
            self.selected_field()
        )

        self.mechanism_context.setText(
            MECHANISM_CONTEXT[field]
        )

        source = (
            self.main
            .missing_df
        )

        total_n = len(
            source
        )

        real_missing_n = int(
            source[field]
            .isna()
            .sum()
        )

        original_observed_n = (
            total_n
            - real_missing_n
        )

        active_for_field = (
            self.main
            .test_gap_field
            == field
        )

        if active_for_field:
            test_n = len(
                self.main
                .test_gap_indices
            )

            still_visible_n = (
                original_observed_n
                - test_n
            )

        else:
            test_n = 0

            still_visible_n = (
                original_observed_n
            )

        self.observed_status.setText(
            "<b>Observed values still visible</b><br>"
            f"{still_visible_n}"
        )

        self.real_status.setText(
            "<span style='color:#6C1D45;'>"
            "<b>Real gaps</b></span><br>"
            f"{real_missing_n}"
        )

        self.test_status.setText(
            "<span style='color:#3E6990;'>"
            "<b>Test Gaps</b></span><br>"
            f"{test_n}"
        )

        if active_for_field:
            self.change_summary.setText(
                f"<b>What changed?</b> "
                f"{test_n} known {FIELD_LABELS[field].lower()} values "
                "are now hidden for testing. Their true values are still "
                "stored internally. The real gaps remain different because "
                "their true values are unknown."
            )

        else:
            self.change_summary.setText(
                "<b>What will change?</b> "
                "When you create a Test Gap, the app will temporarily hide "
                "some known values while keeping their true values internally "
                "for later comparison."
            )

        if active_for_field:
            working_df = (
                self.main
                .working_df
            )

            test_indices = (
                self.main
                .test_gap_indices
            )

            self.table_label.setText(
                f"Survey records: "
                f"{FIELD_LABELS[field]} with "
                f"{real_missing_n} real gaps and "
                f"{test_n} Test Gaps"
            )

        else:
            working_df = (
                self.main
                .missing_df
            )

            test_indices = set()

            if (
                self.main
                .test_gap_field
                is None
            ):
                self.table_label.setText(
                    f"Survey records: "
                    f"{FIELD_LABELS[field]} before "
                    "a Test Gap is created"
                )

            else:
                other_field = (
                    FIELD_LABELS[
                        self.main
                        .test_gap_field
                    ]
                )

                self.table_label.setText(
                    f"Survey records: "
                    f"{FIELD_LABELS[field]} "
                    f"(the active Test Gap is for "
                    f"{other_field})"
                )

        columns = (
            TEST_TABLE_COLUMNS[
                field
            ]
        )

        self.table_model = (
            TestGapTableModel(
                working_df=working_df,
                source_missing_df=self.main.missing_df,
                columns=columns,
                test_field=field,
                test_indices=test_indices,
            )
        )

        self.table.setModel(
            self.table_model
        )

        self.table.resizeColumnsToContents()

        self.table.horizontalHeader().setStretchLastSection(
            False
        )

        self.clear_btn.setEnabled(
            self.main
            .test_gap_field
            is not None
        )



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
        # Current Test Gap
        # -------------------------------------------------------------

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
                "<b>Observed values still available</b>"
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

        layout.addWidget(
            status_box
        )

        # -------------------------------------------------------------
        # Test button
        # -------------------------------------------------------------

        button_row = QHBoxLayout()

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

        button_row.addWidget(
            self.test_btn
        )

        button_row.addStretch()

        layout.addLayout(
            button_row
        )

        # -------------------------------------------------------------
        # Result cards
        # -------------------------------------------------------------

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

        layout.addWidget(
            result_box
        )

        # -------------------------------------------------------------
        # MAE explanation
        # -------------------------------------------------------------

        mae_box = QLabel(
            "<b>What does MAE mean?</b> "
            "For each Test Gap, the app finds how far the mean estimate "
            "is from the hidden true value. Mean Absolute Error, or MAE, "
            "is the average of those absolute errors. A smaller MAE means "
            "the estimates were closer to the values that were hidden."
        )

        mae_box.setWordWrap(
            True
        )

        mae_box.setStyleSheet(
            "background-color: #F7F7F7; "
            "border-left: 4px solid #3E6990; "
            "padding: 8px;"
        )

        layout.addWidget(
            mae_box
        )

        # -------------------------------------------------------------
        # Comparison table
        # -------------------------------------------------------------

        self.table_label = QLabel(
            "Individual Test Gap estimates"
        )

        self.table_label.setStyleSheet(
            "font-weight: bold; "
            "color: #6C1D45;"
        )

        layout.addWidget(
            self.table_label
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

        layout.addWidget(
            self.table,
            stretch=1,
        )

        # -------------------------------------------------------------
        # Thinking prompt
        # -------------------------------------------------------------

        prompt = QLabel(
            "<b>Look closely:</b> Every Test Gap received exactly the "
            "same estimate. What information about individual respondents "
            "did the mean ignore?"
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

        mean_value = float(
            observed.mean()
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

        self.table_model.set_data(
            result[
                "comparison"
            ],
            field,
        )

        self.table.resizeColumnsToContents()

        self.table.horizontalHeader().setStretchLastSection(
            False
        )


# ---------------------------------------------------------------------
# Tab 4 result table models
# ---------------------------------------------------------------------


class ModelComparisonTableModel(
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
                    "method",
                    "mae",
                    "comparison",
                ]
            )
        )

        self.field = field
        self.columns = [
            "method",
            "mae",
            "comparison",
        ]

        self.headers = {
            "method": "Method",
            "mae": "MAE",
            "comparison": "Compared with baseline",
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
            if column == "method":
                return str(value)

            if column == "comparison":
                return str(value)

            if pd.isna(value):
                return "—"

            if self.field == "age":
                return f"{float(value):.2f} years"

            if self.field == "hourly_wage":
                return f"${float(value):,.2f}/hour"

            if self.field == "household_income":
                return f"${float(value):,.0f}"

            return f"{float(value):,.2f}"

        if (
            role == Qt.ItemDataRole.TextAlignmentRole
            and column == "mae"
        ):
            return int(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

        if (
            role == Qt.ItemDataRole.FontRole
            and str(row["method"]) == "Mean baseline"
        ):
            font = QFont()
            font.setBold(True)
            return font

        if (
            role == Qt.ItemDataRole.BackgroundRole
            and str(row["method"]) == "Mean baseline"
        ):
            return QBrush(QColor("#F5F0E8"))

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


class GroupMAETableModel(
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
                    "group",
                    "n",
                    "mae",
                ]
            )
        )

        self.field = field
        self.columns = [
            "group",
            "n",
            "mae",
        ]

        self.headers = {
            "group": "Group",
            "n": "Test cases",
            "mae": "MAE",
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
            if column == "group":
                return str(value).replace("_", " ")

            if column == "n":
                return str(int(value))

            if self.field == "age":
                return f"{float(value):.2f} years"

            if self.field == "hourly_wage":
                return f"${float(value):,.2f}/hour"

            if self.field == "household_income":
                return f"${float(value):,.0f}"

            return f"{float(value):,.2f}"

        if (
            role == Qt.ItemDataRole.TextAlignmentRole
            and column in ("n", "mae")
        ):
            return int(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )

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


class PredictionScatterCanvas(
    QWidget
):

    def __init__(
        self,
    ):
        super().__init__()

        self.comparison = pd.DataFrame()
        self.field: Optional[str] = None
        self.method_label = ""
        self.message = "Test a model to see its estimates."

        self.setMinimumHeight(235)


    def clear_plot(
        self,
        message: str = "Test a model to see its estimates.",
    ) -> None:
        self.comparison = pd.DataFrame()
        self.field = None
        self.method_label = ""
        self.message = message
        self.update()


    def show_result(
        self,
        comparison: pd.DataFrame,
        field: str,
        method_label: str,
    ) -> None:
        self.comparison = comparison.copy()
        self.field = field
        self.method_label = method_label
        self.message = ""
        self.update()


    def paintEvent(
        self,
        event,
    ) -> None:
        painter = QPainter(self)
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )

        painter.fillRect(
            self.rect(),
            QColor("#FFFFFF"),
        )

        if self.comparison.empty:
            painter.setPen(
                QColor("#555555")
            )
            painter.drawText(
                self.rect(),
                int(
                    Qt.AlignmentFlag.AlignCenter
                    | Qt.AlignmentFlag.TextWordWrap
                ),
                self.message,
            )
            return

        truth = self.comparison[
            "hidden_truth"
        ].to_numpy(dtype=float)

        estimate = self.comparison[
            "estimate"
        ].to_numpy(dtype=float)

        lower = float(
            min(
                np.min(truth),
                np.min(estimate),
            )
        )

        upper = float(
            max(
                np.max(truth),
                np.max(estimate),
            )
        )

        if upper <= lower:
            upper = lower + 1.0

        padding = max(
            (upper - lower) * 0.06,
            1.0,
        )

        lower -= padding
        upper += padding

        left = 62
        right = 20
        top = 32
        bottom = 46

        width = max(
            10,
            self.width() - left - right,
        )

        height = max(
            10,
            self.height() - top - bottom,
        )

        def x_position(value: float) -> float:
            return (
                left
                + (value - lower)
                / (upper - lower)
                * width
            )

        def y_position(value: float) -> float:
            return (
                top
                + height
                - (value - lower)
                / (upper - lower)
                * height
            )

        # Title
        title_font = QFont(self.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#222222"))
        painter.drawText(
            left,
            20,
            self.method_label,
        )

        # Axes
        painter.setPen(
            QPen(
                QColor("#555555"),
                1,
            )
        )
        painter.drawLine(
            left,
            top + height,
            left + width,
            top + height,
        )
        painter.drawLine(
            left,
            top,
            left,
            top + height,
        )

        # Perfect-estimate diagonal
        diagonal_pen = QPen(
            QColor(COBBER_MAROON),
            1.5,
        )
        diagonal_pen.setStyle(
            Qt.PenStyle.DashLine
        )
        painter.setPen(diagonal_pen)
        painter.drawLine(
            int(x_position(lower)),
            int(y_position(lower)),
            int(x_position(upper)),
            int(y_position(upper)),
        )

        # Points
        point_brush = QBrush(
            QColor(INFO_BLUE)
        )
        painter.setBrush(point_brush)
        painter.setPen(
            QPen(
                QColor(INFO_BLUE),
                1,
            )
        )

        for true_value, estimated_value in zip(
            truth,
            estimate,
        ):
            x = x_position(
                float(true_value)
            )
            y = y_position(
                float(estimated_value)
            )

            painter.drawEllipse(
                int(x - 3),
                int(y - 3),
                6,
                6,
            )

        # Axis labels
        painter.setPen(
            QColor("#333333")
        )

        if self.field == "age":
            x_label = "Hidden truth (years)"
            y_label = "Estimate (years)"

        elif self.field == "hourly_wage":
            x_label = "Hidden truth ($/hour)"
            y_label = "Estimate ($/hour)"

        elif self.field == "household_income":
            x_label = "Hidden truth ($)"
            y_label = "Estimate ($)"

        else:
            x_label = "Hidden truth"
            y_label = "Estimate"

        painter.drawText(
            left,
            top + height + 30,
            width,
            20,
            int(Qt.AlignmentFlag.AlignCenter),
            x_label,
        )

        painter.save()
        painter.translate(
            18,
            top + height / 2,
        )
        painter.rotate(-90)
        painter.drawText(
            -height / 2,
            -8,
            height,
            20,
            int(Qt.AlignmentFlag.AlignCenter),
            y_label,
        )
        painter.restore()

        # Low/high endpoint labels give the scale without clutter.
        painter.setPen(
            QColor("#666666")
        )

        if self.field == "household_income":
            low_text = f"${lower:,.0f}"
            high_text = f"${upper:,.0f}"

        elif self.field == "hourly_wage":
            low_text = f"${lower:,.0f}"
            high_text = f"${upper:,.0f}"

        else:
            low_text = f"{lower:,.0f}"
            high_text = f"{upper:,.0f}"

        painter.drawText(
            left,
            top + height + 14,
            low_text,
        )

        painter.drawText(
            left + width - 55,
            top + height + 14,
            55,
            18,
            int(Qt.AlignmentFlag.AlignRight),
            high_text,
        )


# ---------------------------------------------------------------------
# Tab 4: Can a Model Beat the Baseline?
# ---------------------------------------------------------------------


class CompareModelsPage(
    QWidget
):

    def __init__(
        self,
        main: "CobberHumImputeApp",
    ):
        super().__init__(main)

        self.main = main
        self.current_method_key: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # -------------------------------------------------------------
        # Introduction
        # -------------------------------------------------------------

        intro = QLabel(
            "<b>Can a Model Beat the Baseline?</b><br>"
            "The mean gave Ravi a simple benchmark. Now he can test methods "
            "that use information elsewhere in each record. A more complicated "
            "method only earns its place if the evidence shows that it improves "
            "the estimates."
        )

        intro.setWordWrap(True)
        intro.setStyleSheet(
            "background-color: #FAFAFA; "
            "border: 1px solid #D6D6D6; "
            "border-radius: 5px; "
            "padding: 10px;"
        )

        layout.addWidget(intro)

        # -------------------------------------------------------------
        # Experiment controls
        # -------------------------------------------------------------

        controls_box = QGroupBox(
            "Test a model"
        )

        controls = QGridLayout(
            controls_box
        )

        controls.setColumnStretch(1, 1)

        controls.addWidget(
            QLabel("<b>Field under investigation</b>"),
            0,
            0,
        )

        self.field_label = QLabel("—")
        controls.addWidget(
            self.field_label,
            0,
            1,
        )

        controls.addWidget(
            QLabel("<b>Mean baseline MAE</b>"),
            1,
            0,
        )

        self.baseline_label = QLabel("—")
        self.baseline_label.setStyleSheet(
            "color: #6C1D45; font-weight: bold;"
        )
        controls.addWidget(
            self.baseline_label,
            1,
            1,
        )

        controls.addWidget(
            QLabel("<b>Method</b>"),
            2,
            0,
        )

        method_row = QHBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItem(
            MODEL_LABELS["linear_regression"],
            "linear_regression",
        )
        self.method_combo.addItem(
            MODEL_LABELS["knn"],
            "knn",
        )
        self.method_combo.addItem(
            MODEL_LABELS["random_forest"],
            "random_forest",
        )
        self.method_combo.setFixedWidth(190)

        self.test_btn = QPushButton(
            "Test This Method"
        )
        self.test_btn.setFixedWidth(150)
        self.test_btn.setStyleSheet(
            "background-color: #6C1D45; "
            "color: white; "
            "font-weight: bold; "
            "border: 1px solid #6C1D45; "
            "border-radius: 4px; "
            "padding: 6px 10px;"
        )

        method_row.addWidget(self.method_combo)
        method_row.addSpacing(12)
        method_row.addWidget(self.test_btn)
        method_row.addStretch()

        controls.addLayout(
            method_row,
            2,
            1,
        )

        self.method_description = QLabel()
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet(
            "color: #555555;"
        )
        controls.addWidget(
            self.method_description,
            3,
            1,
        )

        layout.addWidget(controls_box)

        # -------------------------------------------------------------
        # Comparison table
        # -------------------------------------------------------------

        comparison_box = QGroupBox(
            "Compare with the baseline"
        )

        comparison_layout = QVBoxLayout(
            comparison_box
        )

        self.comparison_table = QTableView()
        self.comparison_model = ModelComparisonTableModel()
        self.comparison_table.setModel(
            self.comparison_model
        )
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.comparison_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setMinimumHeight(135)
        self.comparison_table.setMaximumHeight(165)

        comparison_layout.addWidget(
            self.comparison_table
        )

        layout.addWidget(comparison_box)

        # -------------------------------------------------------------
        # Visual + subgroup evidence
        # -------------------------------------------------------------

        evidence_grid = QGridLayout()
        evidence_grid.setColumnStretch(0, 3)
        evidence_grid.setColumnStretch(1, 2)
        evidence_grid.setHorizontalSpacing(12)

        graph_box = QGroupBox(
            "Hidden truth and model estimate"
        )
        graph_layout = QVBoxLayout(
            graph_box
        )

        graph_help = QLabel(
            "Points closer to the diagonal line have estimates closer to the hidden truth."
        )
        graph_help.setWordWrap(True)
        graph_help.setStyleSheet(
            "color: #555555;"
        )
        graph_layout.addWidget(graph_help)

        self.scatter = PredictionScatterCanvas()
        graph_layout.addWidget(
            self.scatter,
            stretch=1,
        )

        evidence_grid.addWidget(
            graph_box,
            0,
            0,
        )

        group_box = QGroupBox(
            "Does the error look the same for every group?"
        )
        group_layout = QVBoxLayout(
            group_box
        )

        group_row = QHBoxLayout()
        group_row.addWidget(
            QLabel("<b>Compare by:</b>")
        )

        self.group_combo = QComboBox()
        self.group_combo.addItems(
            list(GROUP_FIELDS.keys())
        )
        self.group_combo.setFixedWidth(155)
        group_row.addWidget(
            self.group_combo
        )
        group_row.addStretch()

        group_layout.addLayout(
            group_row
        )

        self.group_table = QTableView()
        self.group_model = GroupMAETableModel()
        self.group_table.setModel(
            self.group_model
        )
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.group_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.group_table.setAlternatingRowColors(True)

        group_layout.addWidget(
            self.group_table,
            stretch=1,
        )

        evidence_grid.addWidget(
            group_box,
            0,
            1,
        )

        layout.addLayout(
            evidence_grid,
            stretch=1,
        )

        # -------------------------------------------------------------
        # Individual estimates
        # -------------------------------------------------------------

        individual_box = QGroupBox(
            "Individual Test Gap estimates"
        )
        individual_layout = QVBoxLayout(
            individual_box
        )

        self.prediction_table = QTableView()
        self.prediction_model = ModelPredictionTableModel()
        self.prediction_table.setModel(
            self.prediction_model
        )
        self.prediction_table.verticalHeader().setVisible(False)
        self.prediction_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.prediction_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.prediction_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.prediction_table.setAlternatingRowColors(True)
        self.prediction_table.setMinimumHeight(145)

        individual_layout.addWidget(
            self.prediction_table
        )

        layout.addWidget(
            individual_box,
            stretch=1,
        )

        # -------------------------------------------------------------
        # Thinking prompt
        # -------------------------------------------------------------

        self.prompt = QLabel(
            "<b>Think about it:</b> A model is more complicated than the mean. "
            "Did it actually earn that extra complexity?"
        )
        self.prompt.setWordWrap(True)
        self.prompt.setStyleSheet(
            "background-color: #FAFAFA; "
            "border-left: 4px solid #6C1D45; "
            "padding: 8px;"
        )

        layout.addWidget(
            self.prompt
        )

        # -------------------------------------------------------------
        # Signals
        # -------------------------------------------------------------

        self.method_combo.currentIndexChanged.connect(
            self.method_changed
        )
        self.test_btn.clicked.connect(
            self.test_method
        )
        self.group_combo.currentTextChanged.connect(
            self.refresh_group_table
        )

        self.method_changed()
        self.refresh()


    def format_metric(
        self,
        value: float,
        field: str,
    ) -> str:
        if field == "age":
            return f"{value:.2f} years"

        if field == "hourly_wage":
            return f"${value:,.2f}/hour"

        if field == "household_income":
            return f"${value:,.0f}"

        return f"{value:,.2f}"


    def selected_method_key(
        self,
    ) -> str:
        value = self.method_combo.currentData()

        if value is None:
            return "linear_regression"

        return str(value)


    def method_changed(
        self,
    ) -> None:
        method_key = self.selected_method_key()
        self.method_description.setText(
            MODEL_DESCRIPTIONS[method_key]
        )

        field = self.main.test_gap_field

        if field is None:
            return

        result = (
            self.main.model_results
            .get(field, {})
            .get(method_key)
        )

        if (
            result is not None
            and result["test_gap_indices"]
            == self.main.test_gap_indices
        ):
            self.show_method_result(result)


    def valid_baseline(
        self,
        field: str,
    ):
        result = self.main.baseline_results.get(field)

        if result is None:
            return None

        if (
            result["test_gap_indices"]
            != self.main.test_gap_indices
        ):
            return None

        return result


    def refresh(
        self,
    ) -> None:
        field = self.main.test_gap_field

        if (
            field is None
            or self.main.test_truth.empty
        ):
            self.field_label.setText(
                "Create a Test Gap first."
            )
            self.baseline_label.setText("—")
            self.test_btn.setEnabled(False)
            self.comparison_model.set_data(
                pd.DataFrame(
                    columns=[
                        "method",
                        "mae",
                        "comparison",
                    ]
                ),
                None,
            )
            self.prediction_model.set_data(
                pd.DataFrame(),
                None,
            )
            self.group_model.set_data(
                pd.DataFrame(),
                None,
            )
            self.scatter.clear_plot(
                "Create a Test Gap and establish its baseline first."
            )
            return

        self.field_label.setText(
            FIELD_LABELS[field]
        )

        baseline = self.valid_baseline(field)

        if baseline is None:
            self.baseline_label.setText(
                "Establish the mean baseline on the previous tab."
            )
            self.test_btn.setEnabled(False)
            self.update_comparison_table(field)
            self.prediction_model.set_data(
                pd.DataFrame(),
                field,
            )
            self.group_model.set_data(
                pd.DataFrame(),
                field,
            )
            self.scatter.clear_plot(
                "Establish the baseline first."
            )
            return

        self.baseline_label.setText(
            self.format_metric(
                baseline["mae"],
                field,
            )
        )
        self.test_btn.setEnabled(True)
        self.update_comparison_table(field)

        method_key = self.selected_method_key()
        result = (
            self.main.model_results
            .get(field, {})
            .get(method_key)
        )

        if (
            result is not None
            and result["test_gap_indices"]
            == self.main.test_gap_indices
        ):
            self.show_method_result(result)

        else:
            self.current_method_key = None
            self.prediction_model.set_data(
                pd.DataFrame(),
                field,
            )
            self.group_model.set_data(
                pd.DataFrame(),
                field,
            )
            self.scatter.clear_plot()


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
            for column in MODEL_FEATURES
            if column != field
        ]

        categorical = [
            column
            for column in CATEGORICAL_FEATURES
            if column in features
        ]

        numeric = [
            column
            for column in features
            if column not in categorical
        ]

        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]

        if scale_numeric:
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
                    SimpleImputer(strategy="most_frequent"),
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

        return preprocessor, features


    def make_model(
        self,
        field: str,
        method_key: str,
    ):
        if method_key == "linear_regression":
            preprocessor, features = self.make_preprocessor(
                field,
                scale_numeric=True,
            )

            estimator = LinearRegression()

        elif method_key == "knn":
            preprocessor, features = self.make_preprocessor(
                field,
                scale_numeric=True,
            )

            estimator = KNeighborsRegressor(
                n_neighbors=5
            )

        elif method_key == "random_forest":
            preprocessor, features = self.make_preprocessor(
                field,
                scale_numeric=False,
            )

            estimator = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                random_state=1704,
                n_jobs=-1,
            )

        else:
            raise ValueError(
                f"Unknown method: {method_key}"
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

        return model, features


    def test_method(
        self,
    ) -> None:
        field = self.main.test_gap_field

        if field is None:
            QMessageBox.information(
                self,
                "Create a Test Gap First",
                "Create a Test Gap before testing a model.",
            )
            return

        baseline = self.valid_baseline(field)

        if baseline is None:
            QMessageBox.information(
                self,
                "Establish a Baseline First",
                "Test the mean on the previous tab before testing a model.",
            )
            return

        method_key = self.selected_method_key()
        model, features = self.make_model(
            field,
            method_key,
        )

        working = self.main.working_df.copy()

        train_mask = working[field].notna()
        test_indices = sorted(
            self.main.test_gap_indices
        )

        X_train = working.loc[
            train_mask,
            features,
        ]
        y_train = pd.to_numeric(
            working.loc[
                train_mask,
                field,
            ],
            errors="coerce",
        )

        X_test = working.loc[
            test_indices,
            features,
        ]

        try:
            model.fit(
                X_train,
                y_train,
            )

            estimates = model.predict(
                X_test
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Model error",
                str(exc),
            )
            return

        truth = (
            self.main.test_truth
            .set_index("source_index")
            .loc[test_indices]
            .reset_index()
        )

        comparison = pd.DataFrame(
            {
                "source_index": test_indices,
                "respondent_id": truth["respondent_id"].astype(str),
                "hidden_truth": pd.to_numeric(
                    truth["true_value"],
                    errors="coerce",
                ),
                "estimate": estimates,
            }
        )

        comparison["absolute_error"] = (
            comparison["hidden_truth"]
            - comparison["estimate"]
        ).abs()

        comparison = (
            comparison
            .dropna(
                subset=[
                    "hidden_truth",
                    "estimate",
                    "absolute_error",
                ]
            )
            .reset_index(drop=True)
        )

        mae = float(
            comparison["absolute_error"].mean()
        )

        baseline_mae = float(
            baseline["mae"]
        )

        difference_pct = (
            100.0
            * (mae - baseline_mae)
            / baseline_mae
        )

        if mae < baseline_mae:
            comparison_text = (
                f"{abs(difference_pct):.1f}% lower error"
            )
        elif mae > baseline_mae:
            comparison_text = (
                f"{abs(difference_pct):.1f}% higher error"
            )
        else:
            comparison_text = "Same error as baseline"

        if field not in self.main.model_results:
            self.main.model_results[field] = {}

        result = {
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

        self.main.model_results[field][method_key] = result

        self.show_method_result(result)
        self.update_comparison_table(field)


    def update_comparison_table(
        self,
        field: str,
    ) -> None:
        baseline = self.valid_baseline(field)

        rows = []

        if baseline is not None:
            rows.append(
                {
                    "method": "Mean baseline",
                    "mae": baseline["mae"],
                    "comparison": "Baseline",
                }
            )

        field_results = self.main.model_results.get(
            field,
            {}
        )

        for method_key in (
            "linear_regression",
            "knn",
            "random_forest",
        ):
            result = field_results.get(method_key)

            if (
                result is not None
                and result["test_gap_indices"]
                == self.main.test_gap_indices
            ):
                rows.append(
                    {
                        "method": result["method_label"],
                        "mae": result["mae"],
                        "comparison": result["comparison_text"],
                    }
                )

        frame = pd.DataFrame(
            rows,
            columns=[
                "method",
                "mae",
                "comparison",
            ],
        )

        self.comparison_model.set_data(
            frame,
            field,
        )

        self.comparison_table.resizeColumnsToContents()
        self.comparison_table.horizontalHeader().setStretchLastSection(
            True
        )


    def show_method_result(
        self,
        result,
    ) -> None:
        field = result["field"]
        self.current_method_key = result["method_key"]

        self.prediction_model.set_data(
            result["comparison"],
            field,
        )
        self.prediction_table.resizeColumnsToContents()
        self.prediction_table.horizontalHeader().setStretchLastSection(
            False
        )

        self.scatter.show_result(
            result["comparison"],
            field,
            result["method_label"],
        )

        baseline_mae = result["baseline_mae"]
        model_mae = result["mae"]

        self.prompt.setText(
            "<b>Think about it:</b> "
            f"{result['method_label']} has an MAE of "
            f"{self.format_metric(model_mae, field)}, compared with "
            f"{self.format_metric(baseline_mae, field)} for the mean baseline. "
            "Did this method earn its extra complexity?"
        )

        self.refresh_group_table()


    def refresh_group_table(
        self,
    ) -> None:
        field = self.main.test_gap_field

        if (
            field is None
            or self.current_method_key is None
        ):
            self.group_model.set_data(
                pd.DataFrame(),
                field,
            )
            return

        result = (
            self.main.model_results
            .get(field, {})
            .get(self.current_method_key)
        )

        if (
            result is None
            or result["test_gap_indices"]
            != self.main.test_gap_indices
        ):
            self.group_model.set_data(
                pd.DataFrame(),
                field,
            )
            return

        group_field = GROUP_FIELDS[
            self.group_combo.currentText()
        ]

        comparison = result["comparison"].copy()

        lookup = (
            self.main.missing_df[
                [group_field]
            ]
            .copy()
        )
        lookup["source_index"] = lookup.index

        comparison = comparison.merge(
            lookup,
            on="source_index",
            how="left",
        )

        rows = (
            comparison
            .groupby(
                group_field,
                dropna=False,
            )
            .agg(
                n=("absolute_error", "size"),
                mae=("absolute_error", "mean"),
            )
            .reset_index()
            .rename(
                columns={
                    group_field: "group",
                }
            )
        )

        rows["group"] = (
            rows["group"]
            .fillna("missing")
            .astype(str)
        )

        rows = rows.sort_values(
            "mae",
            ascending=True,
        ).reset_index(drop=True)

        self.group_model.set_data(
            rows,
            field,
        )

        self.group_table.resizeColumnsToContents()
        self.group_table.horizontalHeader().setStretchLastSection(
            False
        )


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

        self.tabs.currentChanged.connect(
            self.tab_changed
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
            if (
                self.test_gap_field
                is None
            ):
                self.test_gap_page.set_selected_field(
                    self.explore_page.selected_field()
                )

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
