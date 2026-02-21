import argparse
import pytest

from agp.config import build_config


def _make_args(**overrides):
    """Return a minimal Namespace matching the defaults from build_parser."""
    defaults = dict(
        very_low_threshold=54,
        low_threshold=70,
        high_threshold=180,
        very_high_threshold=250,
        tight_low=70,
        tight_high=140,
        bin_minutes=5,
        min_samples=5,
        sensor_interval=5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_returns_all_expected_keys():
    cfg = build_config(_make_args())
    expected = {
        "VERY_LOW", "LOW", "HIGH", "VERY_HIGH",
        "TIGHT_LOW", "TIGHT_HIGH", "BIN_MINUTES",
        "MIN_SAMPLES_PER_BIN", "SENSOR_INTERVAL", "ROC_CLIP",
    }
    assert expected == set(cfg.keys())


def test_build_config_default_values():
    cfg = build_config(_make_args())
    assert cfg["VERY_LOW"] == 54
    assert cfg["LOW"] == 70
    assert cfg["HIGH"] == 180
    assert cfg["VERY_HIGH"] == 250
    assert cfg["TIGHT_LOW"] == 70
    assert cfg["TIGHT_HIGH"] == 140
    assert cfg["BIN_MINUTES"] == 5
    assert cfg["MIN_SAMPLES_PER_BIN"] == 5
    assert cfg["SENSOR_INTERVAL"] == 5
    assert cfg["ROC_CLIP"] == 10


def test_build_config_bin_minutes_defaults_to_1_when_zero():
    cfg = build_config(_make_args(bin_minutes=0))
    assert cfg["BIN_MINUTES"] == 1


def test_build_config_bin_minutes_defaults_to_1_when_negative():
    cfg = build_config(_make_args(bin_minutes=-3))
    assert cfg["BIN_MINUTES"] == 1


def test_build_config_sensor_interval_defaults_to_5_when_zero():
    cfg = build_config(_make_args(sensor_interval=0))
    assert cfg["SENSOR_INTERVAL"] == 5


def test_build_config_sensor_interval_defaults_to_5_when_negative():
    cfg = build_config(_make_args(sensor_interval=-1))
    assert cfg["SENSOR_INTERVAL"] == 5


def test_build_config_roc_clip_always_10():
    for val in [0, -5, 100]:
        cfg = build_config(_make_args(bin_minutes=val))
        assert cfg["ROC_CLIP"] == 10


def test_build_config_custom_thresholds():
    cfg = build_config(_make_args(
        very_low_threshold=60,
        low_threshold=80,
        high_threshold=160,
        very_high_threshold=200,
    ))
    assert cfg["VERY_LOW"] == 60
    assert cfg["LOW"] == 80
    assert cfg["HIGH"] == 160
    assert cfg["VERY_HIGH"] == 200
