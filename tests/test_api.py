"""Tests for the public generate_report() API."""

import os

import matplotlib
import matplotlib.figure
import pytest

matplotlib.use("Agg")

from agp import generate_report  # noqa: E402


@pytest.fixture
def glucose_csv(tmp_path, glucose_df):
    """Write the shared glucose_df fixture to a temporary CSV file."""
    # Add the ROC-independent columns required by load_and_preprocess
    csv_path = tmp_path / "glucose.csv"
    glucose_df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_generate_report_returns_figure(glucose_csv, capsys):
    """generate_report returns a matplotlib Figure when no_plot=False."""
    fig = generate_report(glucose_csv, no_plot=False, show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_generate_report_returns_none_when_no_plot(glucose_csv):
    """generate_report returns None when no_plot=True."""
    fig = generate_report(glucose_csv, no_plot=True)
    assert fig is None


def test_generate_report_accepts_all_defaults(glucose_csv):
    """generate_report can be called with only input_file."""
    fig = generate_report(glucose_csv, show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_generate_report_custom_thresholds(glucose_csv):
    """generate_report accepts custom threshold parameters."""
    fig = generate_report(
        glucose_csv,
        low_threshold=65,
        high_threshold=200,
        show=False,
        close=False,
    )
    assert isinstance(fig, matplotlib.figure.Figure)


def test_generate_report_heatmap(glucose_csv):
    """generate_report with heatmap=True returns a figure with extra axes."""
    fig = generate_report(glucose_csv, heatmap=True, show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    # heatmap adds 2 extra axes (heatmap + colorbar) → 6 total
    assert len(fig.get_axes()) == 6


def test_generate_report_no_show_by_default(glucose_csv, monkeypatch):
    """generate_report does not call plt.show() by default (library use)."""
    import matplotlib.pyplot as plt

    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    generate_report(glucose_csv, show=False, close=False)
    assert show_called == [], "plt.show() must not be called by default"


def test_generate_report_saves_png(glucose_csv, tmp_path):
    """generate_report saves a PNG to the specified output path."""
    out = str(tmp_path / "out.png")
    generate_report(glucose_csv, output=out, show=False, close=False)
    assert os.path.exists(out)


def test_generate_report_with_config_override(glucose_csv, tmp_path):
    """generate_report applies JSON config overrides."""
    import json

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"low_threshold": 65}))
    fig = generate_report(glucose_csv, config=str(cfg_path), show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_generate_report_export_json(glucose_csv, tmp_path):
    """generate_report exports metrics to JSON when export is set."""
    import json

    export_path = str(tmp_path / "metrics.json")
    generate_report(
        glucose_csv,
        no_plot=True,
        export=export_path,
    )
    assert os.path.exists(export_path)
    with open(export_path) as f:
        data = json.load(f)
    assert "tir" in data
