import numpy as np
import pandas as pd
import pytest

from agp.data import load_and_preprocess


def _write_excel(path, df):
    df.to_excel(path, index=False)


def _valid_df(n=300):
    """Return a minimal valid glucose DataFrame with n rows."""
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    glucose = np.random.default_rng(0).uniform(80, 160, size=n)
    return pd.DataFrame({"Time": rng, "Sensor Reading(mg/dL)": glucose})


def test_load_returns_dataframe_with_expected_columns(tmp_path, cfg):
    path = str(tmp_path / "glucose.xlsx")
    _write_excel(path, _valid_df())
    df = load_and_preprocess(path, cfg)
    assert "Time" in df.columns
    assert "Sensor Reading(mg/dL)" in df.columns
    assert "ROC" in df.columns


def test_load_raises_for_missing_columns(tmp_path, cfg):
    bad = pd.DataFrame({"Timestamp": pd.date_range("2024-01-01", periods=5, freq="5min")})
    path = str(tmp_path / "bad.xlsx")
    _write_excel(path, bad)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_preprocess(path, cfg)


def test_load_raises_for_no_valid_rows(tmp_path, cfg):
    df = pd.DataFrame({
        "Time": ["not-a-date"] * 5,
        "Sensor Reading(mg/dL)": [None] * 5,
    })
    path = str(tmp_path / "empty.xlsx")
    _write_excel(path, df)
    with pytest.raises(ValueError, match="No valid glucose data"):
        load_and_preprocess(path, cfg)


def test_load_clips_glucose_to_physiological_range(tmp_path, cfg):
    df = _valid_df(20)
    df.loc[0, "Sensor Reading(mg/dL)"] = 5    # below 20
    df.loc[1, "Sensor Reading(mg/dL)"] = 700  # above 600
    path = str(tmp_path / "clip.xlsx")
    _write_excel(path, df)
    result = load_and_preprocess(path, cfg)
    assert result["Sensor Reading(mg/dL)"].min() >= 20
    assert result["Sensor Reading(mg/dL)"].max() <= 600


def test_load_drops_duplicate_timestamps(tmp_path, cfg):
    df = _valid_df(10)
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # introduce 3 duplicates
    path = str(tmp_path / "dup.xlsx")
    _write_excel(path, df)
    result = load_and_preprocess(path, cfg)
    assert result["Time"].duplicated().sum() == 0


def test_load_computes_roc_clipped(tmp_path, cfg):
    path = str(tmp_path / "roc.xlsx")
    _write_excel(path, _valid_df())
    result = load_and_preprocess(path, cfg)
    roc = result["ROC"].dropna()
    assert (roc.abs() <= cfg["ROC_CLIP"]).all()
    assert roc.apply(np.isfinite).all()
