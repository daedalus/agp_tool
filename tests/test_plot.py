import argparse
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from agp.metrics import compute_all_metrics
from agp.plot import (
    build_agp_profile,
    format_date_range,
    generate_agp_plot,
    generate_daily_plot,
)


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


# --- generate_daily_plot tests ---


def test_daily_plot_returns_figure(df_with_roc, cfg, report_header):
    """generate_daily_plot returns a matplotlib Figure."""
    args = _make_plot_args()
    with (
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.close"),
    ):
        fig = generate_daily_plot(df_with_roc, cfg, args, report_header)
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure)


def test_daily_plot_one_line_per_day(df_with_roc, cfg, report_header):
    """generate_daily_plot draws one line per unique calendar day."""
    args = _make_plot_args()
    with (
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.close"),
    ):
        fig = generate_daily_plot(df_with_roc, cfg, args, report_header)

    ax = fig.get_axes()[0]
    expected_days = df_with_roc["Time"].dt.date.nunique()
    # Day lines carry many data points; threshold/band lines carry ≤2
    day_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 10]
    assert len(day_lines) == expected_days


def test_daily_plot_distinct_colors(df_with_roc, cfg, report_header):
    """Each day line in generate_daily_plot uses a distinct color."""
    import matplotlib.colors as mcolors

    args = _make_plot_args()
    with (
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.close"),
    ):
        fig = generate_daily_plot(df_with_roc, cfg, args, report_header)

    ax = fig.get_axes()[0]
    day_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 10]
    colors = [mcolors.to_rgba(ln.get_color()) for ln in day_lines]
    assert len(colors) == len(set(colors)), "Day lines must have distinct colors"
