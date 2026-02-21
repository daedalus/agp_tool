import numpy as np
import pandas as pd
import pytest

from agp.metrics import (
    compute_mage,
    compute_modd,
    compute_adrr,
    compute_conga,
    compute_risk_indices,
    compute_gri,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# compute_mage
# ---------------------------------------------------------------------------

def test_mage_returns_nan_for_short_series():
    """Series with ≤4 values yields NaN after rolling(3) leaves <3 points."""
    series = pd.Series([100, 120, 90, 110])
    result = compute_mage(series)
    assert np.isnan(result)


def test_mage_returns_nan_for_two_values():
    """Fewer than 3 values always returns NaN."""
    result = compute_mage(pd.Series([100, 200]))
    assert np.isnan(result)


def test_mage_returns_positive_float_for_oscillating_series():
    """Clear alternating high/low pattern should produce a positive MAGE."""
    values = [100, 200] * 20   # strong oscillation
    series = pd.Series(values)
    result = compute_mage(series)
    assert isinstance(result, float)
    assert result > 0


def test_mage_returns_float_or_nan_for_fixture(glucose_df):
    """MAGE on realistic fixture data returns a float (not exception)."""
    result = compute_mage(glucose_df["Sensor Reading(mg/dL)"])
    assert isinstance(result, float) or np.isnan(result)


# ---------------------------------------------------------------------------
# compute_modd
# ---------------------------------------------------------------------------

def test_modd_returns_non_negative_for_multi_day(glucose_df):
    result = compute_modd(glucose_df)
    assert not np.isnan(result)
    assert result >= 0


def test_modd_returns_nan_for_single_day():
    """Only one day of data → no consecutive-day pairs → NaN."""
    rng = pd.date_range("2024-01-01", periods=288, freq="5min")
    df = pd.DataFrame({
        "Time": rng,
        "Sensor Reading(mg/dL)": np.full(288, 120.0),
    })
    result = compute_modd(df)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# compute_adrr
# ---------------------------------------------------------------------------

def test_adrr_returns_non_negative_for_multi_day(glucose_df):
    values = glucose_df["Sensor Reading(mg/dL)"]
    dates = glucose_df["Time"].dt.date
    result = compute_adrr(values, dates)
    assert not np.isnan(result)
    assert result >= 0


def test_adrr_returns_nan_for_insufficient_readings():
    """Fewer than 12 readings/day → NaN."""
    rng = pd.date_range("2024-01-01", periods=10, freq="2h")
    values = pd.Series(np.full(10, 120.0))
    dates = pd.Series(rng.date)
    result = compute_adrr(values, dates)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# compute_conga
# ---------------------------------------------------------------------------

def test_conga_returns_non_negative(glucose_df):
    result = compute_conga(glucose_df)
    assert isinstance(result, float)
    assert result >= 0


def test_conga_is_zero_for_constant_glucose():
    """Constant glucose → no variability → CONGA ≈ 0."""
    rng = pd.date_range("2024-01-01", periods=200, freq="5min")
    df = pd.DataFrame({
        "Time": rng,
        "Sensor Reading(mg/dL)": np.full(200, 110.0),
    })
    result = compute_conga(df)
    assert result == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_risk_indices
# ---------------------------------------------------------------------------

def test_risk_indices_returns_non_negative_tuple():
    values = pd.Series([80, 100, 120, 140, 160])
    lbgi, hbgi = compute_risk_indices(values)
    assert lbgi >= 0
    assert hbgi >= 0


def test_lbgi_higher_for_low_glucose():
    low_values = pd.Series([40, 45, 50, 55, 60])
    high_values = pd.Series([200, 220, 240, 260, 280])
    lbgi_low, _ = compute_risk_indices(low_values)
    lbgi_high, _ = compute_risk_indices(high_values)
    assert lbgi_low > lbgi_high


def test_hbgi_higher_for_high_glucose():
    low_values = pd.Series([60, 65, 70, 75, 80])
    high_values = pd.Series([200, 220, 240, 260, 280])
    _, hbgi_low = compute_risk_indices(low_values)
    _, hbgi_high = compute_risk_indices(high_values)
    assert hbgi_high > hbgi_low


# ---------------------------------------------------------------------------
# compute_gri
# ---------------------------------------------------------------------------

def test_gri_zero_for_all_zeros():
    assert compute_gri(0, 0, 0, 0) == 0.0


def test_gri_correct_weighted_sum():
    # 3.0*2 + 2.4*3 + 1.6*4 + 0.8*5 = 6 + 7.2 + 6.4 + 4.0 = 23.6
    result = compute_gri(2, 3, 4, 5)
    assert result == pytest.approx(23.6)


def test_gri_capped_at_100():
    result = compute_gri(20, 20, 20, 20)
    assert result == 100.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

EXPECTED_KEYS = [
    "tir", "titr", "tar", "tbr", "mage", "modd", "conga",
    "lbgi", "hbgi", "gri", "gri_txt", "adrr",
    "mean_glucose", "cv_percent", "gmi",
    "days_of_data", "wear_percentage",
]


def test_all_metrics_returns_expected_keys(glucose_df, cfg):
    metrics = compute_all_metrics(glucose_df, cfg)
    for key in EXPECTED_KEYS:
        assert key in metrics, f"Missing key: {key}"


def test_all_metrics_percentages_sum_to_100(glucose_df, cfg):
    m = compute_all_metrics(glucose_df, cfg)
    total = m["very_low_pct"] + m["low_pct"] + m["tight_target_pct"] + \
            m["above_tight_pct"] + m["high_pct"] + m["very_high_pct"]
    assert total == pytest.approx(100.0, abs=1e-6)


def test_all_metrics_mean_glucose_in_range(glucose_df, cfg):
    m = compute_all_metrics(glucose_df, cfg)
    assert 20 <= m["mean_glucose"] <= 600


def test_all_metrics_days_of_data(glucose_df, cfg):
    m = compute_all_metrics(glucose_df, cfg)
    # 2016 readings × 5 minutes = 10080 min = 7 days
    assert m["days_of_data"] == pytest.approx(7.0, abs=0.1)
