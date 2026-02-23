import shutil

import numpy as np
import pandas as pd
import pytest

from agp.data import _sniff_format, load_and_preprocess


def _write_excel(path, df):
    df.to_excel(path, index=False)


def _write_csv(path, df):
    df.to_csv(path, index=False)


def _write_ods(path, df):
    df.to_excel(path, index=False, engine="odf")


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
    bad = pd.DataFrame(
        {"Timestamp": pd.date_range("2024-01-01", periods=5, freq="5min")}
    )
    path = str(tmp_path / "bad.xlsx")
    _write_excel(path, bad)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_preprocess(path, cfg)


def test_load_raises_for_no_valid_rows(tmp_path, cfg):
    df = pd.DataFrame(
        {
            "Time": ["not-a-date"] * 5,
            "Sensor Reading(mg/dL)": [None] * 5,
        }
    )
    path = str(tmp_path / "empty.xlsx")
    _write_excel(path, df)
    with pytest.raises(ValueError, match="No valid glucose data"):
        load_and_preprocess(path, cfg)


def test_load_clips_glucose_to_physiological_range(tmp_path, cfg):
    df = _valid_df(20)
    df.loc[0, "Sensor Reading(mg/dL)"] = 5  # below 20
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


def test_load_csv_returns_expected_columns(tmp_path, cfg):
    path = str(tmp_path / "glucose.csv")
    _write_csv(path, _valid_df())
    df = load_and_preprocess(path, cfg)
    assert "Time" in df.columns
    assert "Sensor Reading(mg/dL)" in df.columns
    assert "ROC" in df.columns


def test_load_ods_returns_expected_columns(tmp_path, cfg):
    path = str(tmp_path / "glucose.ods")
    _write_ods(path, _valid_df())
    df = load_and_preprocess(path, cfg)
    assert "Time" in df.columns
    assert "Sensor Reading(mg/dL)" in df.columns
    assert "ROC" in df.columns


def test_unsupported_extension_raises_value_error(tmp_path, cfg):
    # Create a dummy file so the extension dispatch is reached
    path_obj = tmp_path / "glucose.txt"
    path_obj.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_and_preprocess(str(path_obj), cfg)


# ---------------------------------------------------------------------------
# Content-sniffing tests
# ---------------------------------------------------------------------------


def test_sniff_format_returns_xlsx_for_xlsx_file(tmp_path):
    path = str(tmp_path / "glucose.xlsx")
    _write_excel(path, _valid_df())
    assert _sniff_format(path) == "xlsx"


def test_sniff_format_returns_ods_for_ods_file(tmp_path):
    path = str(tmp_path / "glucose.ods")
    _write_ods(path, _valid_df())
    assert _sniff_format(path) == "ods"


def test_sniff_format_returns_none_for_csv(tmp_path):
    path = str(tmp_path / "glucose.csv")
    _write_csv(path, _valid_df())
    assert _sniff_format(path) is None


def test_xlsx_content_with_xls_extension_loads_correctly(tmp_path, cfg):
    """An xlsx file given a misleading .xls extension must still load via content sniffing."""
    xlsx_path = str(tmp_path / "glucose.xlsx")
    _write_excel(xlsx_path, _valid_df())
    misleading_path = str(tmp_path / "glucose_misleading.xls")
    shutil.copy(xlsx_path, misleading_path)

    result = load_and_preprocess(misleading_path, cfg)
    assert "Time" in result.columns
    assert "Sensor Reading(mg/dL)" in result.columns
    assert "ROC" in result.columns
