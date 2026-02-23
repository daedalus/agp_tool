import argparse
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from agp.metrics import compute_all_metrics
from agp.plot import build_agp_profile, format_date_range, generate_agp_plot


def _make_plot_args(**overrides):
    defaults = dict(
        output="test_output.png",
        heatmap=False,
        heatmap_cmap="RdYlGn_r",
        verbose=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def df_with_roc(glucose_df):
    """glucose_df with a minimal ROC column added."""
    df = glucose_df.copy()
    df["ROC"] = 0.0
    return df


@pytest.fixture
def report_header():
    return {
        "patient_name": "Test Patient",
        "patient_id": "001",
        "doctor": "",
        "report_date": "2024-01-08",
        "notes": "",
        "data_source": "test.csv",
    }


def _run_plot(df_with_roc, cfg, report_header, heatmap):
    result = build_agp_profile(df_with_roc, cfg)
    metrics = compute_all_metrics(df_with_roc, cfg)
    args = _make_plot_args(heatmap=heatmap)

    with (
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.close"),
    ):
        generate_agp_plot(df_with_roc, result, metrics, cfg, args, report_header)

    return plt.gcf()


def test_heatmap_subplot_absent_when_disabled(df_with_roc, cfg, report_header):
    """Without --heatmap, the figure should have 4 axes (no heatmap row)."""
    fig = _run_plot(df_with_roc, cfg, report_header, heatmap=False)
    assert len(fig.get_axes()) == 4


def test_heatmap_subplot_present_when_enabled(df_with_roc, cfg, report_header):
    """With --heatmap, the figure should have 6 axes (heatmap + colorbar added)."""
    fig = _run_plot(df_with_roc, cfg, report_header, heatmap=True)
    assert len(fig.get_axes()) == 6


# --- format_date_range tests ---


def test_format_date_range_normal(glucose_df):
    """Normal multi-day dataset returns first and last date in YYYY-MM-DD \u2013 YYYY-MM-DD format."""
    result = format_date_range(glucose_df)
    first = glucose_df["Time"].min().strftime("%Y-%m-%d")
    last = glucose_df["Time"].max().strftime("%Y-%m-%d")
    assert result == f"{first} \u2013 {last}"


def test_format_date_range_single_row():
    """Single-row dataset returns the same date for both start and end."""
    df = pd.DataFrame(
        {
            "Time": [pd.Timestamp("2024-03-15")],
            "Sensor Reading(mg/dL)": [100.0],
        }
    )
    assert format_date_range(df) == "2024-03-15 \u2013 2024-03-15"


def test_format_date_range_empty():
    """Empty dataset returns 'N/A' without raising."""
    df = pd.DataFrame(
        {
            "Time": pd.Series(dtype="datetime64[ns]"),
            "Sensor Reading(mg/dL)": pd.Series(dtype=float),
        }
    )
    assert format_date_range(df) == "N/A"
