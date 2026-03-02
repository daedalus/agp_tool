"""Public library API for the AGP tool.

Example::

    import matplotlib
    matplotlib.use("Agg")  # use non-interactive backend when no display is available

    from agp import generate_report, ReportGenerator

    # Function-based API (backward compatible)
    fig = generate_report("data.csv", patient_name="Jane Doe", verbose=True)
    if fig is not None:
        fig.savefig("report.png", dpi=150, bbox_inches="tight")

    # Class-based API
    report = ReportGenerator("data.csv", patient_name="Jane Doe")
    print(report.tir)           # Time in Range value
    print(report.metrics)       # full metrics dict
    fig = report.plot_agp()     # generate the AGP figure
    fig = report.plot_daily()   # generate the daily overlay figure
    report.print_summary()      # print clinical summary to stdout
    report.export("out.json")   # export metrics to file
"""

import argparse
import json

from .config import build_config
from .data import load_and_preprocess
from .export import export_metrics
from .metrics import compute_all_metrics
from .pdf import png_to_pdf
from .plot import build_agp_profile, generate_agp_plot, generate_daily_plot
from .report import create_report_header, print_clinical_summary


class ReportGenerator:
    """Instantiable AGP report generator.

    Loads and preprocesses the glucose data file at construction time and
    computes all metrics immediately, making them available as instance
    attributes.  Graph generation and export are available as methods so
    they can be invoked on demand.

    Args:
        input_file (str): Path to glucose data file (.xlsx, .xls, .csv, .ods).
        output (str): Default output PNG filename used by :meth:`plot_agp`.
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
        verbose (bool): Print detailed progress during data loading. Default: ``False``.
        config (str | None): Path to a JSON configuration file.  Keys that
            match parameter names override the corresponding arguments.
            Default: ``None``.
        patient_name (str): Patient name for the report header.
            Default: ``"Unknown"``.
        patient_id (str): Patient ID for the report header. Default: ``"N/A"``.
        doctor (str): Doctor name for the report header. Default: ``""``.
        notes (str): Additional notes for the report header. Default: ``""``.
        heatmap (bool): Enable the circadian glucose heatmap in
            :meth:`plot_agp`. Default: ``False``.
        heatmap_cmap (str): Colormap name for the circadian heatmap.
            Default: ``"RdYlGn_r"``.

    Attributes:
        metrics (dict): Full computed metrics dictionary.  Individual metric
            values are also accessible directly as instance attributes, e.g.
            ``report.tir``, ``report.mean_glucose``.

    Example::

        report = ReportGenerator("data.csv", patient_name="Jane Doe")

        # Access individual metrics
        print(f"TIR: {report.tir:.1f}%")
        print(f"Mean glucose: {report.mean_glucose:.1f} mg/dL")

        # Access full metrics dict
        m = report.metrics
        print(m["gri"])

        # Generate plots
        fig_agp   = report.plot_agp(output="agp.png")
        fig_daily = report.plot_daily(output="daily.png")

        # Print clinical summary
        report.print_summary()

        # Export metrics
        report.export("metrics.json")
    """

    def __init__(
        self,
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
        verbose=False,
        config=None,
        patient_name="Unknown",
        patient_id="N/A",
        doctor="",
        notes="",
        heatmap=False,
        heatmap_cmap="RdYlGn_r",
    ):
        # Build an argparse Namespace so the existing helpers (build_config,
        # create_report_header, generate_agp_plot) continue to work unchanged.
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
            verbose=verbose,
            config=config,
            patient_name=patient_name,
            patient_id=patient_id,
            doctor=doctor,
            notes=notes,
            heatmap=heatmap,
            heatmap_cmap=heatmap_cmap,
        )

        # Apply JSON config file overrides (mirrors CLI behaviour).
        if config:
            try:
                with open(config) as f:
                    cfg_overrides = json.load(f)
                for key, value in cfg_overrides.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                if verbose:
                    print(f"Loaded configuration from {config}")
            except Exception as e:
                print(f"Error loading config file: {e}")

        self._args = args
        self._cfg = build_config(args)
        self._report_header = create_report_header(args)
        self._df = load_and_preprocess(args.input_file, self._cfg, verbose=verbose)
        self._metrics = compute_all_metrics(self._df, self._cfg)

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        """Return the full metrics dictionary."""
        return self._metrics

    def get_metrics(self):
        """Return the full metrics dictionary (callable form of :attr:`metrics`)."""
        return self._metrics

    def __getattr__(self, name):
        # Delegate unknown attribute access to the metrics dict so that
        # individual metrics are accessible directly, e.g. ``report.tir``.
        # __getattr__ is only called when normal attribute lookup fails, so
        # this does not shadow any real instance/class attributes.
        try:
            return self.__dict__["_metrics"][name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def plot_agp(self, output=None, show=False, close=False):
        """Generate the full AGP figure and return it.

        Args:
            output (str | None): Save the figure to this path.  When
                ``None`` the *output* value supplied at construction time is
                used.  Pass an empty string ``""`` to skip saving.
            show (bool): Call ``plt.show()`` after building. Default: ``False``.
            close (bool): Call ``plt.close()`` after building. Default: ``False``.

        Returns:
            matplotlib.figure.Figure: The completed AGP figure.
        """
        result = build_agp_profile(self._df, self._cfg)
        _output = output if output is not None else self._args.output
        return generate_agp_plot(
            self._df,
            result,
            self._metrics,
            self._cfg,
            self._args,
            self._report_header,
            output_path=_output,
            show=show,
            close=close,
        )

    def plot_daily(self, output=None, show=False, close=False):
        """Generate the daily overlay figure and return it.

        Each calendar day in the dataset is drawn as a separate colored
        line so that day-to-day glucose patterns can be compared directly.

        Args:
            output (str | None): Save the figure to this path.  When
                ``None`` a ``_daily`` suffix is appended to the base output
                filename supplied at construction time.  Pass an empty
                string ``""`` to skip saving.
            show (bool): Call ``plt.show()`` after building. Default: ``False``.
            close (bool): Call ``plt.close()`` after building. Default: ``False``.

        Returns:
            matplotlib.figure.Figure: The completed daily overlay figure.
        """
        if output is None:
            base_output = self._args.output
            if "." in base_output:
                base, ext = base_output.rsplit(".", 1)
            else:
                base, ext = base_output, "png"
            output = f"{base}_daily.{ext}"
        return generate_daily_plot(
            self._df,
            self._cfg,
            self._args,
            self._report_header,
            output_path=output,
            show=show,
            close=close,
        )

    # ------------------------------------------------------------------
    # Summary and export
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print the formatted clinical summary to stdout."""
        print_clinical_summary(self._metrics, self._report_header, self._cfg)

    def export(self, path, verbose=None):
        """Export metrics to *path* (``.json``, ``.csv``, or ``.xlsx``).

        Args:
            path (str): Destination file path.  The extension determines the
                format: ``.json``, ``.csv``, or ``.xlsx``.
            verbose (bool | None): Override the verbosity level set at
                construction time.  ``None`` uses the construction-time value.
        """
        _verbose = self._args.verbose if verbose is None else verbose
        export_metrics(
            self._metrics, path, report_header=self._report_header, verbose=_verbose
        )


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
    daily_plot=False,
    show=False,
    close=False,
):
    """Run the full AGP pipeline and return the matplotlib Figure.

    This function mirrors every option available in the CLI (except
    ``--version``) and executes the same pipeline: data loading,
    preprocessing, metric computation, plot generation, clinical summary,
    optional metrics export, and optional PDF creation.

    Internally this is a thin wrapper around :class:`ReportGenerator`.

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
        daily_plot (bool): Generate an additional daily overlay plot where each
            day is shown as a separate colored line.  The figure is saved next
            to *output* with ``_daily`` appended before the extension.
            Default: ``False``.
        show (bool): Call ``plt.show()`` after building the figure.
            Set to ``True`` only when running interactively.  Default: ``False``.
        close (bool): Call ``plt.close()`` after building the figure.
            Default: ``False``.

    Returns:
        matplotlib.figure.Figure | None: The completed AGP figure, or ``None``
        when *no_plot* is ``True``.
    """
    report = ReportGenerator(
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
        verbose=verbose,
        config=config,
        patient_name=patient_name,
        patient_id=patient_id,
        doctor=doctor,
        notes=notes,
        heatmap=heatmap,
        heatmap_cmap=heatmap_cmap,
    )

    fig = None
    if not no_plot:
        fig = report.plot_agp(show=show, close=close)
        if pdf:
            pdf_path = output.rsplit(".", 1)[0] + ".pdf"
            png_to_pdf(output, pdf_path)
            if verbose:
                print(f"PDF saved to: {pdf_path}")

    if daily_plot and not no_plot:
        report.plot_daily(show=show, close=close)
    elif verbose and no_plot:
        print("Plot generation skipped (no_plot=True)")

    report.print_summary()

    if export:
        report.export(export)

    return fig
