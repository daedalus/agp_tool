import os
import sys
import zipfile

import numpy as np
import pandas as pd

# Magic-byte signatures for binary spreadsheet formats.
_OLE2_MAGIC = (
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # Legacy .xls (OLE2 compound document)
)
_ZIP_MAGIC = b"PK\x03\x04"  # ZIP container (.xlsx, .ods)

_READERS = {
    ".xlsx": lambda p: pd.read_excel(p, engine="openpyxl"),
    ".xls": lambda p: pd.read_excel(p, engine="xlrd"),
    ".ods": lambda p: pd.read_excel(p, engine="odf"),
    ".csv": lambda p: pd.read_csv(p),
}


def _sniff_format(path: str):
    """Detect binary spreadsheet format by reading magic bytes.

    Returns one of ``'xls'``, ``'xlsx'``, ``'ods'``, or ``None`` when the
    format cannot be determined from content alone.  CSV detection is
    intentionally omitted because sniffing is unreliable for plain-text files.
    """
    if "../" in path or "..\\" in path:
        raise Exception("Invalid file path")
    with open(path, "rb") as f:
        header = f.read(8)

    # OLE2 compound document → legacy .xls
    if header == _OLE2_MAGIC:
        return "xls"

    # ZIP container → could be .xlsx or .ods; inspect the archive to decide
    if header[:4] == _ZIP_MAGIC:
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
                # OpenDocument: has a 'mimetype' entry with an ODF MIME-type prefix
                if "mimetype" in names:
                    with zf.open("mimetype") as mt:
                        mime = mt.read(64).decode("ascii", errors="replace")
                    if mime.startswith("application/vnd.oasis.opendocument"):
                        return "ods"
                # Office Open XML (.xlsx): contains [Content_Types].xml and/or xl/ namespace
                if "[Content_Types].xml" in names or any(
                    n.startswith("xl/") for n in names
                ):
                    return "xlsx"
        except (zipfile.BadZipFile, KeyError):
            pass

    return None


def _read_input(input_file: str) -> pd.DataFrame:
    """Dispatch to the correct pandas reader.

    Prefers magic-byte/content sniffing for binary spreadsheet formats
    (.xls, .xlsx, .ods) so that files with misleading extensions are handled
    correctly (e.g. a file named ``.xls`` that is actually an ``.xlsx``).
    CSV files are always handled by extension because plain-text sniffing is
    unreliable.  Extension-based dispatch is used as a fallback when sniffing
    is inconclusive.
    """
    ext = os.path.splitext(input_file)[1].lower()

    # CSV: sniffing unreliable for plain-text formats; always use extension.
    if ext == ".csv":
        return pd.read_csv(input_file)

    # Attempt content sniffing for binary spreadsheet formats.
    fmt = _sniff_format(input_file)
    if fmt == "xls":
        return pd.read_excel(input_file, engine="xlrd")
    if fmt == "xlsx":
        return pd.read_excel(input_file, engine="openpyxl")
    if fmt == "ods":
        return pd.read_excel(input_file, engine="odf")

    # Sniffing inconclusive – fall back to extension-based dispatch.
    reader = _READERS.get(ext)
    if reader is not None:
        return reader(input_file)

    supported = ", ".join(sorted(_READERS))
    raise ValueError(
        f"Unsupported file format (detected: unrecognized, extension: '{ext}'). "
        f"Supported formats: {supported}"
    )


def load_and_preprocess(input_file, cfg, verbose=False):
    """Load data file (xlsx/xls/csv/ods), validate columns, parse datetimes, deduplicate, compute ROC.

    Returns a cleaned DataFrame with an added 'ROC' column.
    """
    if verbose:
        print(f"Loading data from: {input_file}")
        print(
            f"Glucose thresholds: Low={cfg['LOW']}, High={cfg['HIGH']}, "
            f"Tight={cfg['TIGHT_LOW']}-{cfg['TIGHT_HIGH']}"
        )

    try:
        df = _read_input(input_file)
        if verbose:
            print(f"Successfully loaded {len(df)} rows from {input_file}")
    except ValueError:
        raise
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        sys.exit(1)

    required = ["Time", "Sensor Reading(mg/dL)"]
    if not all(col in df.columns for col in required):
        raise ValueError("Missing required columns.")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=required)
    df = df.sort_values("Time").drop_duplicates(subset=["Time"])

    if len(df) == 0:
        raise ValueError("No valid glucose data found.")

    # Filter physiologically impossible values
    glucose = df["Sensor Reading(mg/dL)"].astype(float)
    glucose = glucose.clip(20, 600)  # Physiological limits
    df["Sensor Reading(mg/dL)"] = glucose

    # Rate of Change (ROC)
    df["delta_glucose"] = glucose.diff()
    df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0
    df["ROC"] = df["delta_glucose"] / df["delta_minutes"]
    df.loc[df["delta_minutes"] <= 0, "ROC"] = np.nan
    df["ROC"] = df["ROC"].clip(-cfg["ROC_CLIP"], cfg["ROC_CLIP"])

    return df
