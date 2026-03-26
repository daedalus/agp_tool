import json
from dataclasses import dataclass, field
from typing import Optional


def apply_config_overrides(obj, config_path, verbose=False):
    """Apply JSON config file overrides to *obj* in-place.

    Opens *config_path*, loads it as JSON, and for every key that matches an
    attribute on *obj* sets that attribute to the corresponding value.  When
    *verbose* is ``True`` a confirmation message is printed.  Any exception is
    caught and printed rather than propagated.

    Args:
        obj: Any object that supports ``hasattr``/``setattr``.
        config_path (str): Path to a JSON file containing override key/value
            pairs.
        verbose (bool): Print a confirmation message on success.  Default:
            ``False``.

    Returns:
        The same *obj* (modified in-place).
    """
    try:
        with open(config_path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        if verbose:
            print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading config file: {e}")
    return obj


@dataclass
class ReportConfig:
    """Configuration dataclass for :class:`~agp.api.ReportGenerator`.

    Holds all configurable parameters with the same defaults as the
    :class:`~agp.api.ReportGenerator` constructor, providing a clean
    alternative to ``argparse.Namespace`` for programmatic use.
    """

    input_file: str = field(default="")
    output: str = "ambulatory_glucose_profile.png"
    very_low_threshold: int = 54
    low_threshold: int = 70
    high_threshold: int = 180
    very_high_threshold: int = 250
    tight_low: int = 70
    tight_high: int = 140
    bin_minutes: int = 5
    sensor_interval: int = 5
    min_samples: int = 5
    verbose: bool = False
    config: Optional[str] = None
    patient_name: str = "Unknown"
    patient_id: str = "N/A"
    doctor: str = ""
    notes: str = ""
    heatmap: bool = False
    heatmap_cmap: str = "RdYlGn_r"
    dark_mode: bool = False


def build_config(args):
    """Return a dict of all threshold/config constants derived from parsed args.

    Raises ``ValueError`` if the supplied values violate basic invariants.
    """
    very_low = args.very_low_threshold
    low = args.low_threshold
    high = args.high_threshold
    very_high = args.very_high_threshold
    tight_low = args.tight_low
    tight_high = args.tight_high
    bin_minutes = args.bin_minutes
    sensor_interval = args.sensor_interval
    min_samples = args.min_samples

    if not (very_low <= low <= high <= very_high):
        raise ValueError(
            f"Thresholds must satisfy very_low <= low <= high <= very_high, "
            f"got {very_low} <= {low} <= {high} <= {very_high}"
        )
    if tight_low >= tight_high:
        raise ValueError(
            f"tight_low must be less than tight_high, got {tight_low} >= {tight_high}"
        )
    if bin_minutes <= 0:
        raise ValueError(f"bin_minutes must be positive, got {bin_minutes}")
    if sensor_interval <= 0:
        raise ValueError(f"sensor_interval must be positive, got {sensor_interval}")
    if min_samples <= 0:
        raise ValueError(f"min_samples must be positive, got {min_samples}")

    return {
        "VERY_LOW": very_low,
        "LOW": low,
        "HIGH": high,
        "VERY_HIGH": very_high,
        "TIGHT_LOW": tight_low,
        "TIGHT_HIGH": tight_high,
        "BIN_MINUTES": bin_minutes,
        "MIN_SAMPLES_PER_BIN": min_samples,
        "SENSOR_INTERVAL": sensor_interval,
        "ROC_CLIP": 10,  # Physiological constant
    }
