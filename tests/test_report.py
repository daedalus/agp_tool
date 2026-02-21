import argparse
import pytest

from agp.report import create_report_header, print_clinical_summary


def _make_args(**overrides):
    defaults = dict(
        patient_name="Jane Doe",
        patient_id="P001",
        doctor="Dr. Smith",
        notes="Follow-up visit",
        input_file="glucose.xlsx",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


SAMPLE_METRICS = {
    'tir': 70.0,
    'titr': 50.0,
    'tbr': 3.0,
    'cv_percent': 25.0,
    'trend_arrow': '→',
    'trend_direction': 'STABLE',
    'trend_slope': 0.0,
    'gri': 15.0,
    'gri_txt': 'Low Risk',
    'days_of_data': 7.0,
    'hours_of_data': 168.0,
    'readings_per_day': 288.0,
    'wear_percentage': 100.0,
    'skew_glucose': 0.2,
    'skew_interpretation': 'Approximately symmetric',
    'mean_glucose': 120.0,
    'median_glucose': 118.0,
    'very_low_pct': 2.0,
    'low_pct': 3.0,
    'tight_target_pct': 50.0,
    'above_tight_pct': 10.0,
    'high_pct': 10.0,
    'very_high_pct': 5.0,
}


def test_create_report_header_returns_dict():
    header = create_report_header(_make_args())
    assert isinstance(header, dict)


def test_create_report_header_expected_keys():
    header = create_report_header(_make_args())
    for key in ("patient_name", "patient_id", "doctor", "notes", "report_date"):
        assert key in header, f"Missing key: {key}"


def test_create_report_header_values():
    header = create_report_header(_make_args(patient_name="Alice", patient_id="A42"))
    assert header["patient_name"] == "Alice"
    assert header["patient_id"] == "A42"


def test_print_clinical_summary_produces_output(capsys):
    header = create_report_header(_make_args())
    print_clinical_summary(SAMPLE_METRICS, header)
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_print_clinical_summary_contains_patient_name(capsys):
    header = create_report_header(_make_args(patient_name="Bob"))
    print_clinical_summary(SAMPLE_METRICS, header)
    captured = capsys.readouterr()
    assert "Bob" in captured.out


def test_print_clinical_summary_contains_tir(capsys):
    header = create_report_header(_make_args())
    print_clinical_summary(SAMPLE_METRICS, header)
    captured = capsys.readouterr()
    assert "70.0" in captured.out
