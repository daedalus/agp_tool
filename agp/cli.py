import argparse
import json
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agp")
except PackageNotFoundError:
    __version__ = "1.0.0"


def build_parser():
    """Construct and return the ArgumentParser for the AGP tool."""
    parser = argparse.ArgumentParser(
        description="Generate Ambulatory Glucose Profile from sensor data"
    )
    parser.add_argument(
        "input_file", help="Path to glucose data file (.xlsx, .xls, .csv, .ods)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="ambulatory_glucose_profile.png",
        help="Output PNG filename (default: ambulatory_glucose_profile.png)",
    )

    parser.add_argument(
        "--very-low-threshold",
        type=int,
        default=54,
        help="Very low glucose threshold in mg/dL (default: 54)",
    )
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=70,
        help="Low glucose threshold in mg/dL (default: 70)",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=180,
        help="High glucose threshold in mg/dL (default: 180)",
    )
    parser.add_argument(
        "--very-high-threshold",
        type=int,
        default=250,
        help="Very high glucose threshold in mg/dL (default: 250)",
    )
    parser.add_argument(
        "--tight-low",
        type=int,
        default=70,
        help="Tight range lower limit in mg/dL (default: 70)",
    )
    parser.add_argument(
        "--tight-high",
        type=int,
        default=140,
        help="Tight range upper limit in mg/dL (default: 140)",
    )
    parser.add_argument(
        "--bin-minutes",
        type=int,
        default=5,
        help="Time bin size in minutes for AGP (default: 5)",
    )
    parser.add_argument(
        "--sensor-interval",
        type=int,
        default=5,
        help="CGM Sensor interval (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples per bin (default: 5)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Calculate metrics only, do not generate plot",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed metrics during execution",
    )
    parser.add_argument(
        "--export",
        "-e",
        default="",
        help="Export metrics to file. Use .csv or .json extension (e.g. metrics.json)",
    )
    parser.add_argument("--config", "-c", help="Configuration file with parameters")

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--patient-name", "-n", default="Unknown", help="Patient name for report header"
    )
    parser.add_argument(
        "--patient-id", "-id", default="N/A", help="Patient ID for report header"
    )
    parser.add_argument(
        "--doctor", "-d", default="", help="Doctor name for report header"
    )
    parser.add_argument(
        "--notes", "-note", default="", help="Additional notes for report header"
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Enable the circadian glucose heatmap (disabled by default)",
    )
    parser.add_argument(
        "--heatmap-cmap",
        default="RdYlGn_r",
        help="Colormap for circadian heatmap (default: RdYlGn_r, requires --heatmap)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also produce a PDF file with the PNG embedded as an image and metadata copied from the PNG",
    )

    return parser


def parse_args():
    """Parse command-line arguments and apply any config file overrides."""
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        try:
            with open(args.config) as f:
                config = json.load(f)
                for key, value in config.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                if args.verbose:
                    print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")

    return args
