# SPEC.md — agp_tool

## Purpose
agp_tool is a Python library and CLI tool for generating Ambulatory Glucose Profiles (AGP) from continuous glucose monitoring (CGM) data. It computes clinical metrics (TIR, TITR, TAR, TBR, variability indices, risk indices) and produces visualizations including AGP plots, daily overlay graphs, and circadian heatmaps.

## Scope
- **In scope**: CGM data parsing (xlsx, xls, csv, ods), clinical metric computation, AGP visualization, PDF export, CLI interface, library API
- **Not in scope**: Real-time data processing, cloud integration, mobile app support, medical device validation

## Public API / Interface

### CLI (`agp_tool` or `agp`)
```
agp_tool <input_file> [options]
```
- `input_file`: Path to glucose data (.xlsx, .xls, .csv, .ods)
- Options: `--output`, `--very-low-threshold`, `--low-threshold`, `--high-threshold`, `--very-high-threshold`, `--tight-low`, `--tight-high`, `--bin-minutes`, `--sensor-interval`, `--min-samples`, `--no-plot`, `--verbose`, `--export`, `--config`, `--patient-name`, `--patient-id`, `--doctor`, `--notes`, `--heatmap`, `--heatmap-cmap`, `--pdf`, `--daily-plot`, `--dark-mode`

### Library API
- `ReportGenerator` class: Instantiate with input file, access metrics as attributes, generate plots via methods
- `generate_report` function: Thin wrapper around ReportGenerator, backward compatible

### Module Structure
- `agp/__init__.py`: Exports `__version__`, `generate_report`, `ReportGenerator`
- `agp/__main__.py`: CLI entry point (calls `agp.main:main`)
- `agp/cli.py`: Click-based CLI
- `agp/api.py`: Public library API (`ReportGenerator`, `generate_report`)
- `agp/data.py`: Data loading and preprocessing
- `agp/metrics.py`: Metric computation
- `agp/plot.py`: Plot generation
- `agp/config.py`: Configuration handling
- `agp/report.py`: Report header and summary
- `agp/export.py`: Metrics export
- `agp/pdf.py`: PDF generation

## Data Formats
- **Input**: Excel (.xlsx, .xls), CSV, OpenDocument (.ods)
- **Required columns**: datetime, glucose (mg/dL)
- **Output**: PNG, PDF, JSON, CSV, XLSX

## Edge Cases
1. Empty input file: Raise `ValueError("No data found in file")`
2. Missing required columns: Raise `ValueError("Missing required column: ...")`
3. All glucose values below/above thresholds: Handle zero division in percentage calculations
4. Single day of data: Compute metrics with limited data quality warnings
5. Duplicate timestamps: Keep first occurrence, warn if duplicates found
6. Gaps >2 hours: May affect MODD/ADRR; issue warning
7. Invalid glucose values (negative, extremely high): Filter out and warn
8. File not found: Raise `FileNotFoundError`

## Performance & Constraints
- O(n log n) due to sorting by time
- Memory: Load entire dataset into pandas DataFrame
- Python >= 3.9 required
- Dependencies: pandas, numpy, matplotlib, openpyxl, xlrd, odfpy, Pillow, fpdf2, click