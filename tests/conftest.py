import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cfg():
    """Default config matching build_config defaults."""
    return {
        'VERY_LOW': 54,
        'LOW': 70,
        'HIGH': 180,
        'VERY_HIGH': 250,
        'TIGHT_LOW': 70,
        'TIGHT_HIGH': 140,
        'BIN_MINUTES': 5,
        'MIN_SAMPLES_PER_BIN': 5,
        'SENSOR_INTERVAL': 5,
        'ROC_CLIP': 10,
    }


@pytest.fixture
def glucose_df():
    """7 days of 5-minute CGM data with realistic mixed glucose values."""
    rng = pd.date_range("2024-01-01", periods=2016, freq="5min")

    # 24-hour sinusoidal pattern centered at 120 mg/dL, amplitude 50
    t = np.arange(len(rng))
    glucose = 120 + 50 * np.sin(2 * np.pi * t / (24 * 12))

    # Sprinkle in high, very-high, low and very-low readings
    glucose[100:120] = 210   # high
    glucose[200:210] = 260   # very high
    glucose[300:310] = 62    # low
    glucose[400:410] = 45    # very low

    return pd.DataFrame({"Time": rng, "Sensor Reading(mg/dL)": glucose})
