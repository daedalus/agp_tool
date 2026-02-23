"""Public library API for the AGP tool.

Example::

    import matplotlib
    matplotlib.use("Agg")  # use non-interactive backend when no display is available

    from agp import generate_report

    fig = generate_report("data.csv", patient_name="Jane Doe", verbose=True)
    if fig is not None:
        fig.savefig("report.png", dpi=150, bbox_inches="tight")
"""

import argparse
import json

from .config import build_config
from .data import load_and_preprocess
from .export import export_metrics
from .metrics import compute_all_metrics
from .pdf import png_to_pdf
from .plot import build_agp_profile, generate_agp_plot
from .report import create_report_header, print_clinical_summary


def generate_report(
    input_file,
    output="ambulatory_glucose_profile.png",
    very_low_threshold=54,
    low_threshold=70,
    high_threshold=180,
    very_high_threshold=250,
    tight_low=70,
    tight_high=140,
    bin_minutes=5,
    sensor_interval=5,
    min_samples=5,
    no_plot=False,
    verbose=False,
    export="",
    config=None,
    patient_name="Unknown",
    patient_id="N/A",
    doctor="",
    notes="",
    heatmap=False,
    heatmap_cmap="RdYlGn_r",
    pdf=False,
    show=False,
    close=False,
):
    """Run the full AGP pipeline and return the matplotlib Figure.

    This function mirrors every option available in the CLI (except
    ``--version``) and executes the same pipeline: data loading,
    preprocessing, metric computation, plot generation, clinical summary,
    optional metrics export, and optional PDF creation.

    Args:
        input_file (str): Path to glucose data file (.xlsx, .xls, .csv, .ods).
        output (str): Output PNG filename.
            Default: ``"ambulatory_glucose_profile.png"``.
        very_low_threshold (int): Very low glucose threshold in mg/dL. Default: 54.
        low_threshold (int): Low glucose threshold in mg/dL. Default: 70.
        high_threshold (int): High glucose threshold in mg/dL. Default: 180.
        very_high_threshold (int): Very high glucose threshold in mg/dL. Default: 250.
        tight_low (int): Tight range lower limit in mg/dL. Default: 70.
        tight_high (int): Tight range upper limit in mg/dL. Default: 140.
        bin_minutes (int): Time bin size in minutes for AGP. Default: 5.
        sensor_interval (int): CGM sensor reading interval in minutes. Default: 5.
        min_samples (int): Minimum samples per bin. Default: 5.
        no_plot (bool): When ``True``, skip plot generation and return ``None``.
            Default: ``False``.
        verbose (bool): Print detailed metrics during execution. Default: ``False``.
        export (str): Export metrics to a file path.  Use ``.csv`` or ``.json``
            extension.  Empty string disables export. Default: ``""``.
        config (str | None): Path to a JSON configuration file.  Keys that
            match parameter names override the corresponding arguments (same
            behaviour as the CLI ``--config`` option). Default: ``None``.
        patient_name (str): Patient name for the report header.
            Default: ``"Unknown"``.
        patient_id (str): Patient ID for the report header. Default: ``"N/A"``.
        doctor (str): Doctor name for the report header. Default: ``""``.
        notes (str): Additional notes for the report header. Default: ``""``.
        heatmap (bool): Enable the circadian glucose heatmap. Default: ``False``.
        heatmap_cmap (str): Colormap name for the circadian heatmap.
            Default: ``"RdYlGn_r"``.
        pdf (bool): Also produce a PDF file alongside the PNG. Default: ``False``.
        show (bool): Call ``plt.show()`` after building the figure.
            Set to ``True`` only when running interactively.  Default: ``False``.
        close (bool): Call ``plt.close()`` after building the figure.
            Default: ``False``.

    Returns:
        matplotlib.figure.Figure | None: The completed AGP figure, or ``None``
        when *no_plot* is ``True``.
    """
    # Build an argparse Namespace so the existing helpers (build_config,
    # create_report_header) continue to work without modification.
    args = argparse.Namespace(
        input_file=input_file,
        output=output,
        very_low_threshold=very_low_threshold,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        very_high_threshold=very_high_threshold,
        tight_low=tight_low,
        tight_high=tight_high,
        bin_minutes=bin_minutes,
        sensor_interval=sensor_interval,
        min_samples=min_samples,
        no_plot=no_plot,
        verbose=verbose,
        export=export,
        config=config,
        patient_name=patient_name,
        patient_id=patient_id,
        doctor=doctor,
        notes=notes,
        heatmap=heatmap,
        heatmap_cmap=heatmap_cmap,
        pdf=pdf,
    )

    # Apply JSON config file overrides (mirrors CLI behaviour).
    if config:
        try:
            with open(config, "r") as f:
                cfg_overrides = json.load(f)
            for key, value in cfg_overrides.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            if verbose:
                print(f"Loaded configuration from {config}")
        except Exception as e:
            print(f"Error loading config file: {e}")

    cfg = build_config(args)
    report_header = create_report_header(args)
    df = load_and_preprocess(args.input_file, cfg, verbose=verbose)
    metrics = compute_all_metrics(df, cfg)

    fig = None
    if not no_plot:
        result = build_agp_profile(df, cfg)
        fig = generate_agp_plot(
            df,
            result,
            metrics,
            cfg,
            args,
            report_header,
            show=show,
            close=close,
        )
        if pdf:
            pdf_path = output.rsplit(".", 1)[0] + ".pdf"
            png_to_pdf(output, pdf_path)
            if verbose:
                print(f"PDF saved to: {pdf_path}")
    elif verbose:
        print("Plot generation skipped (no_plot=True)")

    print_clinical_summary(metrics, report_header, cfg)

    if export:
        export_metrics(metrics, export, report_header=report_header, verbose=verbose)

    return fig
