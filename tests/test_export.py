import json

import pandas as pd

from agp.export import export_metrics

SAMPLE_METRICS = {
    "days_of_data": 7.0,
    "readings_per_day": 288.0,
    "wear_percentage": 100.0,
    "mean_glucose": 120.0,
    "median_glucose": 118.0,
    "std_glucose": 30.0,
    "cv_percent": 25.0,
    "day_cv": 24.0,
    "night_cv": 26.0,
    "gmi": 6.1,
    "skew_glucose": 0.2,
    "j_index": 22.5,
    "tir": 70.0,
    "titr": 50.0,
    "tatr": 10.0,
    "tar": 15.0,
    "tar_level1": 10.0,
    "tar_level2": 5.0,
    "tbr": 5.0,
    "tbr_level1": 3.0,
    "tbr_level2": 2.0,
    "very_low_pct": 2.0,
    "low_pct": 3.0,
    "tight_target_pct": 50.0,
    "above_tight_pct": 10.0,
    "high_pct": 10.0,
    "very_high_pct": 5.0,
    "mage": 45.0,
    "modd": 20.0,
    "conga": 18.0,
    "lbgi": 1.5,
    "hbgi": 2.0,
    "gri": 15.0,
    "gri_txt": "Low Risk",
    "adrr": 10.0,
    "time_weighted_avg": 121.0,
    "exposure_severity_to_hyperglycemia_pct": 5.0,
    "exposure_severity_to_hypoglycemia_pct": 2.0,
    "exposure_severity_to_severe_hypoglycemia_pct": 0.5,
    "trend_direction": "STABLE",
    "trend_slope": 0.0,
    "trend_arrow": "→",
    "trend_color": "steelblue",
    "severe_hypo_count": 0,
    "severe_hypo_per_week": 0.0,
    "hours_of_data": 168.0,
}

EXPECTED_EXPORT_KEYS = [
    "days_of_data",
    "mean_glucose",
    "tir",
    "tbr",
    "tar",
    "gmi",
    "cv_percent",
    "lbgi",
    "hbgi",
    "gri",
    "trend_direction",
]


def test_export_csv_creates_file(tmp_path):
    path = str(tmp_path / "metrics.csv")
    export_metrics(SAMPLE_METRICS, path)
    assert (tmp_path / "metrics.csv").exists()


def test_export_csv_contains_expected_keys(tmp_path):
    path = str(tmp_path / "metrics.csv")
    export_metrics(SAMPLE_METRICS, path)
    df = pd.read_csv(path)
    for key in EXPECTED_EXPORT_KEYS:
        assert key in df.columns, f"Missing column: {key}"


def test_export_json_creates_file(tmp_path):
    path = str(tmp_path / "metrics.json")
    export_metrics(SAMPLE_METRICS, path)
    assert (tmp_path / "metrics.json").exists()


def test_export_json_contains_expected_keys(tmp_path):
    path = str(tmp_path / "metrics.json")
    export_metrics(SAMPLE_METRICS, path)
    with open(path) as f:
        data = json.load(f)
    for key in EXPECTED_EXPORT_KEYS:
        assert key in data, f"Missing key: {key}"


def test_export_xlsx_creates_readable_file(tmp_path):
    path = str(tmp_path / "metrics.xlsx")
    export_metrics(SAMPLE_METRICS, path)
    assert (tmp_path / "metrics.xlsx").exists()
    df = pd.read_excel(path)
    assert len(df) == 1
    for key in EXPECTED_EXPORT_KEYS:
        assert key in df.columns, f"Missing column: {key}"


def test_export_unknown_format_does_not_create_file(tmp_path, capsys):
    path = str(tmp_path / "metrics.txt")
    export_metrics(SAMPLE_METRICS, path)
    assert not (tmp_path / "metrics.txt").exists()
    captured = capsys.readouterr()
    assert "Warning" in captured.out
