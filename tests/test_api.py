"""Tests for the public generate_report() API and ReportGenerator class."""

import os

import matplotlib
import matplotlib.figure
import pytest

matplotlib.use("Agg")

from agp import ReportGenerator, generate_report  # noqa: E402


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


# ---------------------------------------------------------------------------
# ReportGenerator class tests
# ---------------------------------------------------------------------------


@pytest.fixture
def report(glucose_csv):
    """Return a ReportGenerator instance built from the shared fixture CSV."""
    return ReportGenerator(glucose_csv)


def test_report_generator_instantiates(glucose_csv):
    """ReportGenerator can be constructed with just an input file."""
    rg = ReportGenerator(glucose_csv)
    assert isinstance(rg, ReportGenerator)


def test_report_generator_metrics_property_returns_dict(report):
    """metrics property returns a non-empty dict."""
    m = report.metrics
    assert isinstance(m, dict)
    assert len(m) > 0


def test_report_generator_get_metrics_equals_metrics_property(report):
    """get_metrics() returns the same object as the metrics property."""
    assert report.get_metrics() is report.metrics


def test_report_generator_individual_metric_tir(report):
    """TIR is accessible as a direct attribute."""
    assert 0.0 <= report.tir <= 100.0


def test_report_generator_individual_metric_mean_glucose(report):
    """mean_glucose is accessible as a direct attribute."""
    assert report.mean_glucose > 0


def test_report_generator_individual_metric_gri(report):
    """GRI is accessible as a direct attribute."""
    assert 0.0 <= report.gri <= 100.0


def test_report_generator_unknown_attribute_raises(report):
    """Accessing an unknown attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        _ = report.nonexistent_metric_xyz


def test_report_generator_plot_agp_returns_figure(report):
    """plot_agp() returns a matplotlib Figure."""
    fig = report.plot_agp(output="", show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_report_generator_plot_daily_returns_figure(report):
    """plot_daily() returns a matplotlib Figure."""
    fig = report.plot_daily(output="", show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_report_generator_plot_agp_saves_file(report, tmp_path):
    """plot_agp() saves to the specified output path."""
    out = str(tmp_path / "agp.png")
    report.plot_agp(output=out, show=False, close=False)
    assert os.path.exists(out)


def test_report_generator_plot_daily_saves_file(report, tmp_path):
    """plot_daily() saves to the specified output path."""
    out = str(tmp_path / "daily.png")
    report.plot_daily(output=out, show=False, close=False)
    assert os.path.exists(out)


def test_report_generator_print_summary_produces_output(report, capsys):
    """print_summary() writes to stdout."""
    report.print_summary()
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_report_generator_export_json(report, tmp_path):
    """export() writes a valid JSON file when given a .json path."""
    import json

    out = str(tmp_path / "metrics.json")
    report.export(out)
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert "tir" in data


def test_report_generator_export_csv(report, tmp_path):
    """export() writes a CSV file when given a .csv path."""
    out = str(tmp_path / "metrics.csv")
    report.export(out)
    assert os.path.exists(out)


def test_report_generator_custom_thresholds(glucose_csv):
    """ReportGenerator respects custom threshold parameters."""
    rg = ReportGenerator(glucose_csv, low_threshold=65, high_threshold=200)
    assert rg.metrics is not None


def test_report_generator_config_override(glucose_csv, tmp_path):
    """ReportGenerator applies JSON config overrides."""
    import json

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"low_threshold": 65}))
    rg = ReportGenerator(glucose_csv, config=str(cfg_path))
    assert rg.metrics is not None


def test_report_generator_heatmap_plot(glucose_csv):
    """plot_agp() with heatmap=True produces a figure with extra axes."""
    rg = ReportGenerator(glucose_csv, heatmap=True)
    fig = rg.plot_agp(output="", show=False, close=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Heatmap adds 2 extra axes (heatmap panel + colorbar) → 6 total
    # (vs. 4 for non-heatmap: distribution bar, AGP main, ROC twin, raw series)
    assert len(fig.get_axes()) == 6


def test_report_generator_metrics_keys_complete(report):
    """metrics dict contains all expected top-level keys."""
    expected_keys = {
        "tir",
        "titr",
        "tbr",
        "gri",
        "mean_glucose",
        "median_glucose",
        "cv_percent",
        "mage",
        "modd",
        "lbgi",
        "hbgi",
        "days_of_data",
        "wear_percentage",
        "trend_direction",
    }
    for key in expected_keys:
        assert key in report.metrics, f"Missing metric key: {key}"
