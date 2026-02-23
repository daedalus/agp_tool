import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def format_date_range(df):
    """Return 'YYYY-MM-DD \u2013 YYYY-MM-DD' covering the mg/dL data in *df*.

    Returns 'N/A' when *df* has no rows.
    """
    if df.empty:
        return "N/A"
    first = df["Time"].min().strftime("%Y-%m-%d")
    last = df["Time"].max().strftime("%Y-%m-%d")
    return f"{first} \u2013 {last}"


def build_agp_profile(df, cfg):
    """Perform the circadian binning and return the result DataFrame."""
    BIN_MINUTES = cfg["BIN_MINUTES"]
    MIN_SAMPLES_PER_BIN = cfg["MIN_SAMPLES_PER_BIN"]

    df_work = df.copy()
    df_work["seconds"] = (
        df_work["Time"].dt.hour * 3600
        + df_work["Time"].dt.minute * 60
        + df_work["Time"].dt.second
    )
    df_work["bin"] = (df_work["seconds"] // (BIN_MINUTES * 60)).astype(int)

    grouped = df_work.groupby("bin")["Sensor Reading(mg/dL)"]

    result = grouped.agg(
        count="count",
        p5=lambda x: np.percentile(x, 5),
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        median="median",
        mean="mean",
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
        p95=lambda x: np.percentile(x, 95),
    ).reset_index()

    roc_profile = df_work.groupby("bin")["ROC"].mean().reset_index()
    roc_profile.rename(columns={"ROC": "roc_mean"}, inplace=True)

    result = result.merge(roc_profile, on="bin", how="left")

    result.loc[
        result["count"] < MIN_SAMPLES_PER_BIN,
        result.columns.difference(["bin", "count"]),
    ] = np.nan

    result = result.sort_values("bin")
    result["minutes"] = result["bin"] * BIN_MINUTES

    return result


def generate_agp_plot(df, result, metrics, cfg, args, report_header, *, output_path=None, show=False, close=False):
    """Render the full AGP figure and return it.

    Args:
        df: Preprocessed glucose DataFrame.
        result: AGP profile DataFrame from :func:`build_agp_profile`.
        metrics: Metrics dict from :func:`compute_all_metrics`.
        cfg: Configuration dict from :func:`build_config`.
        args: argparse Namespace (or None) supplying ``heatmap``,
            ``heatmap_cmap``, ``output``, and ``verbose`` attributes.
        report_header: Report header dict from :func:`create_report_header`.
        output_path: If given, save the figure to this path (overrides
            ``args.output``).  Pass ``None`` (and ensure ``args.output`` is
            also ``None`` / falsy) to skip saving entirely.
        show: If ``True`` call ``plt.show()`` after building the figure.
        close: If ``True`` call ``plt.close()`` after building the figure.

    Returns:
        matplotlib.figure.Figure: The completed AGP figure.
    """
    VERY_LOW = cfg["VERY_LOW"]
    LOW = cfg["LOW"]
    HIGH = cfg["HIGH"]
    VERY_HIGH = cfg["VERY_HIGH"]
    TIGHT_LOW = cfg["TIGHT_LOW"]
    TIGHT_HIGH = cfg["TIGHT_HIGH"]

    # Unpack metrics
    tir = metrics["tir"]
    titr = metrics["titr"]
    tatr = metrics["tatr"]
    tar = metrics["tar"]
    tar_level1 = metrics["tar_level1"]
    tar_level2 = metrics["tar_level2"]
    tbr = metrics["tbr"]
    tbr_level1 = metrics["tbr_level1"]
    tbr_level2 = metrics["tbr_level2"]
    very_low_pct = metrics["very_low_pct"]
    low_pct = metrics["low_pct"]
    tight_target_pct = metrics["tight_target_pct"]
    above_tight_pct = metrics["above_tight_pct"]
    high_pct = metrics["high_pct"]
    very_high_pct = metrics["very_high_pct"]
    mean_glucose = metrics["mean_glucose"]
    median_glucose = metrics["median_glucose"]
    std_glucose = metrics["std_glucose"]
    mode_str = metrics["mode_str"]
    skew_glucose = metrics["skew_glucose"]
    skew_interpretation = metrics["skew_interpretation"]
    gmi = metrics["gmi"]
    cv_percent = metrics["cv_percent"]
    day_cv = metrics["day_cv"]
    night_cv = metrics["night_cv"]
    j_index = metrics["j_index"]
    mage = metrics["mage"]
    modd = metrics["modd"]
    conga = metrics["conga"]
    lbgi = metrics["lbgi"]
    hbgi = metrics["hbgi"]
    gri = metrics["gri"]
    gri_txt = metrics["gri_txt"]
    adrr = metrics["adrr"]
    time_weighted_avg = metrics["time_weighted_avg"]
    exposure_severity_to_hyperglycemia_pct = metrics[
        "exposure_severity_to_hyperglycemia_pct"
    ]
    exposure_severity_to_hypoglycemia_pct = metrics[
        "exposure_severity_to_hypoglycemia_pct"
    ]
    exposure_severity_to_severe_hypoglycemia_pct = metrics[
        "exposure_severity_to_severe_hypoglycemia_pct"
    ]
    days_of_data = metrics["days_of_data"]
    readings_per_day = metrics["readings_per_day"]
    wear_percentage = metrics["wear_percentage"]
    severe_hypo_per_week = metrics["severe_hypo_per_week"]
    trend_arrow = metrics["trend_arrow"]
    trend_color = metrics["trend_color"]

    # Create color-coded data series for raw readings
    df = df.copy()
    df["glucose_range"] = pd.cut(
        df["Sensor Reading(mg/dL)"],
        bins=[0, VERY_LOW, TIGHT_LOW, TIGHT_HIGH, HIGH, VERY_HIGH, 1000],
        labels=["Very Low", "Low", "Tight Target", "Above Tight", "High", "Very High"],
    )

    def fmt(v, decimals=1):
        return f"{v:.{decimals}f}" if not np.isnan(v) else "N/A"

    # --------------------------------------------------
    # Create figure with GridSpec for custom layout
    # --------------------------------------------------
    if getattr(args, "heatmap", False):
        fig = plt.figure(figsize=(18, 17))
        gs = GridSpec(
            3, 12, figure=fig, height_ratios=[3, 1.5, 1.5], hspace=0.35, wspace=0.3
        )
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(
            2, 12, figure=fig, height_ratios=[3, 1.5], hspace=0.35, wspace=0.3
        )

    # --- TOP ROW ---
    ax_bar = fig.add_subplot(gs[0, :2])
    ax1 = fig.add_subplot(gs[0, 2:])

    # Stacked bar chart of glucose distribution
    percentages = [
        very_low_pct,
        low_pct,
        tight_target_pct,
        above_tight_pct,
        high_pct,
        very_high_pct,
    ]
    colors = ["darkred", "red", "limegreen", "yellowgreen", "orange", "darkorange"]
    labels = [
        f"Very Low (<{VERY_LOW})",
        f"Low ({VERY_LOW}-{LOW - 1})",
        f"Tight Target ({TIGHT_LOW}-{TIGHT_HIGH})",
        f"Above Tight ({TIGHT_HIGH + 1}-{HIGH})",
        f"High ({HIGH + 1}-{VERY_HIGH})",
        f"Very High (>{VERY_HIGH})",
    ]

    bottoms = np.zeros(1)
    bars = []
    for _, (pct, color) in enumerate(zip(percentages, colors)):
        if pct > 0:
            bar = ax_bar.bar(
                0,
                pct,
                bottom=bottoms[0],
                color=color,
                edgecolor="white",
                linewidth=1,
                width=0.5,
            )
            bars.append(bar)

            if pct > 3:
                y_pos = bottoms[0] + pct / 2
                ax_bar.text(
                    0,
                    y_pos,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9,
                )
            bottoms[0] += pct

    ax_bar.set_ylim(0, 100)
    ax_bar.set_xlim(-0.5, 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("Percentage of Time (%)", fontsize=10)
    ax_bar.set_title("Glucose Distribution\nby Range", fontsize=11, pad=10)

    ax_bar.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax_bar.set_axisbelow(True)

    legend_elements = []
    for _, (label, color, pct) in enumerate(zip(labels, colors, percentages)):
        if pct > 0:
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white")
            )

    if legend_elements:
        ax_bar.legend(
            legend_elements,
            [label for label, pct in zip(labels, percentages) if pct > 0],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=1,
            fontsize=8,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="gray",
        )

    # Main AGP plot
    x = result["minutes"]

    ax1.axhspan(
        TIGHT_LOW,
        TIGHT_HIGH,
        alpha=0.15,
        color="limegreen",
        label=f"Tight Target ({TIGHT_LOW}-{TIGHT_HIGH})",
    )
    ax1.axhspan(
        TIGHT_HIGH,
        HIGH,
        alpha=0.20,
        color="darkgreen",
        label=f"Above Tight ({TIGHT_HIGH}-{HIGH})",
    )
    ax1.axhspan(HIGH, 600, alpha=0.1, color="orange", label=f"Above Range (>{HIGH})")
    ax1.axhspan(20, LOW, alpha=0.1, color="red", label=f"Below Range (<{LOW})")

    ax1.axhline(
        mean_glucose,
        linestyle="-.",
        linewidth=2,
        color="purple",
        alpha=0.7,
        label="Overall Mean",
    )

    ax1.fill_between(
        x, result["p5"], result["p95"], alpha=0.15, color="blue", label="5–95%"
    )
    ax1.fill_between(
        x, result["p25"], result["p75"], alpha=0.35, color="blue", label="IQR"
    )
    ax1.plot(x, result["median"], linewidth=2.5, color="darkblue", label="Median")
    ax1.plot(
        x, result["mean"], linestyle="--", linewidth=1.5, color="navy", label="Mean"
    )

    ax1.axhline(LOW, linestyle=":", linewidth=1, color="darkred", alpha=0.5)
    ax1.axhline(HIGH, linestyle=":", linewidth=1, color="darkred", alpha=0.5)
    ax1.axhline(TIGHT_HIGH, linestyle=":", linewidth=1, color="darkgreen", alpha=0.5)

    ax1.axvspan(22 * 60, 24 * 60, alpha=0.05, color="gray")
    ax1.axvspan(0, 6 * 60, alpha=0.05, color="gray", label="Night Hours")

    ax1.set_xlabel("Time of Day", fontsize=12)
    ax1.set_ylabel("Glucose (mg/dL)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        result["roc_mean"],
        linestyle=":",
        linewidth=2,
        label="ROC (mg/dL/min)",
        color="orange",
    )
    ax2.set_ylabel("Rate of Change (mg/dL/min)", fontsize=12)

    xticks = np.arange(0, 1441, 120)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])

    current_ylim = ax1.get_ylim()
    ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.15)

    textstr = (
        f"TIME IN RANGE\n"
        f"TIR ({LOW}-{HIGH}): {tir:.1f}%\n"
        f"TITR ({TIGHT_LOW}-{TIGHT_HIGH}): {titr:.1f}%  ← Tight Target\n"
        f"TATR ({TIGHT_HIGH}-{HIGH}): {tatr:.1f}%\n"
        f"TAR >{HIGH}: {tar:.1f}% ({HIGH + 1}-{VERY_HIGH}: {tar_level1:.1f}%, >{VERY_HIGH}: {tar_level2:.1f}%)\n"
        f"TBR <{LOW}: {tbr:.1f}% ({VERY_LOW}-{LOW - 1}: {tbr_level1:.1f}%, <{VERY_LOW}: {tbr_level2:.1f}%)\n\n"
        f"GLUCOSE STATS\n"
        f"Mean: {mean_glucose:.1f} mg/dL, median: {median_glucose:.1f} mg/dL\n"
        f"Std: {std_glucose:.1f} mg/dL, mode: {mode_str} mg/dL\n"
        f"skew: {skew_glucose:.1f} ({skew_interpretation})\n"
        f"GMI: {gmi:.2f}%\n"
        f"CV: {cv_percent:.1f}% {'(Stable)' if cv_percent < 36 else '(Unstable)'}\n"
        f"CV: Day: {fmt(day_cv)}%, Night: {fmt(night_cv)}%\n"
        f"J-Index: {j_index:.1f}\n\n"
        f"VARIABILITY\n"
        f"MAGE: {fmt(mage)}\n"
        f"MODD: {fmt(modd)}\n"
        f"CONGA(1h): {fmt(conga)}\n\n"
        f"RISK\n"
        f"LBGI: {lbgi:.2f}\n"
        f"HBGI: {hbgi:.2f}\n"
        f"GRI: {gri:.1f} ({gri_txt})\n"
        f"ADRR: {fmt(adrr)}\n\n"
        f"AUC\n"
        f"Time-weighted avg: {fmt(time_weighted_avg)} mg/dL\n"
        f"Hyperglycemia exposure severity: {exposure_severity_to_hyperglycemia_pct:.1f}%\n"
        f"Hypoglycemia exposure severiry: {exposure_severity_to_hypoglycemia_pct:.1f}%\n"
        f"Severe hypoglycemia exposure severiry: {exposure_severity_to_severe_hypoglycemia_pct:.1f}%\n\n"
        f"DATA QUALITY\n"
        f"Days: {days_of_data:.1f}\n"
        f"Readings/day: {readings_per_day:.0f}\n"
        f"Wear time: {wear_percentage:.1f}%\n"
        f"Severe hypo/week: {fmt(severe_hypo_per_week, 2)}"
    )

    plt.gcf().text(
        0.75,
        0.92,
        textstr,
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.49,
            edgecolor="gray",
            linewidth=1,
            pad=0.8,
        ),
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax1.transAxes,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=9,
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
    )

    ax1.text(
        0.60,
        0.97,
        f"Overall Trend: {trend_arrow}",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        color=trend_color,
        va="top",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.7,
            edgecolor=trend_color,
            linewidth=1.5,
        ),
    )

    # --- BOTTOM ROW: Raw Data Series Chart ---
    ax3 = fig.add_subplot(gs[1, :])

    range_colors = {
        "Very Low": "darkred",
        "Low": "red",
        "Tight Target": "limegreen",
        "Above Tight": "yellowgreen",
        "High": "orange",
        "Very High": "darkorange",
    }

    tight_target_data = df[df["glucose_range"] == "Tight Target"]
    if not tight_target_data.empty:
        ax3.scatter(
            tight_target_data["Time"],
            tight_target_data["Sensor Reading(mg/dL)"],
            c=range_colors["Tight Target"],
            s=8,
            alpha=0.4,
            label=f"Tight Target ({TIGHT_LOW}-{TIGHT_HIGH}): {len(tight_target_data)} pts",
            edgecolors="none",
        )

    above_tight_data = df[df["glucose_range"] == "Above Tight"]
    if not above_tight_data.empty:
        ax3.scatter(
            above_tight_data["Time"],
            above_tight_data["Sensor Reading(mg/dL)"],
            c=range_colors["Above Tight"],
            s=10,
            alpha=0.5,
            label=f"Above Tight ({TIGHT_HIGH + 1}-{HIGH}): {len(above_tight_data)} pts",
            edgecolors="none",
        )

    high_data = df[df["glucose_range"] == "High"]
    if not high_data.empty:
        ax3.scatter(
            high_data["Time"],
            high_data["Sensor Reading(mg/dL)"],
            c=range_colors["High"],
            s=12,
            alpha=0.6,
            label=f"High ({HIGH + 1}-{VERY_HIGH}): {len(high_data)} pts",
            edgecolors="none",
        )

    very_high_data = df[df["glucose_range"] == "Very High"]
    if not very_high_data.empty:
        ax3.scatter(
            very_high_data["Time"],
            very_high_data["Sensor Reading(mg/dL)"],
            c=range_colors["Very High"],
            s=12,
            alpha=0.7,
            label=f"Very High (>{VERY_HIGH}): {len(very_high_data)} pts",
            edgecolors="none",
        )

    low_data = df[df["glucose_range"] == "Low"]
    if not low_data.empty:
        ax3.scatter(
            low_data["Time"],
            low_data["Sensor Reading(mg/dL)"],
            c=range_colors["Low"],
            s=15,
            alpha=0.8,
            label=f"Low ({VERY_LOW}-{LOW - 1}): {len(low_data)} pts",
            edgecolors="black",
            linewidth=0.5,
        )

    very_low_data = df[df["glucose_range"] == "Very Low"]
    if not very_low_data.empty:
        ax3.scatter(
            very_low_data["Time"],
            very_low_data["Sensor Reading(mg/dL)"],
            c=range_colors["Very Low"],
            s=20,
            alpha=1.0,
            label=f"Very Low (<{VERY_LOW}): {len(very_low_data)} pts",
            edgecolors="black",
            linewidth=0.8,
        )

    ax3.axhspan(TIGHT_LOW, TIGHT_HIGH, alpha=0.1, color="limegreen")
    ax3.axhspan(TIGHT_HIGH, HIGH, alpha=0.07, color="green")
    ax3.axhspan(HIGH, 600, alpha=0.07, color="orange")
    ax3.axhspan(20, LOW, alpha=0.07, color="red")

    ax3.axhline(LOW, linestyle=":", linewidth=1, color="darkred", alpha=0.4)
    ax3.axhline(HIGH, linestyle=":", linewidth=1, color="darkred", alpha=0.4)
    ax3.axhline(TIGHT_HIGH, linestyle=":", linewidth=1, color="darkgreen", alpha=0.4)

    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
    ax3.xaxis.set_major_locator(
        plt.matplotlib.dates.DayLocator(interval=max(1, days_of_data // 7))
    )
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    if days_of_data <= 3:
        ax3.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=6))
        ax3.xaxis.set_minor_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
        plt.setp(ax3.xaxis.get_minorticklabels(), rotation=45, ha="right", fontsize=8)

    ax3.set_ylabel("Glucose (mg/dL)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.set_title("Raw Glucose Data Series (Color-coded by Range)", fontsize=12, pad=10)
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
    ax3.set_ylim(20, 400)

    ax3.text(
        0.01,
        0.97,
        f"Overall Trend: {trend_arrow}",
        transform=ax3.transAxes,
        fontsize=11,
        fontweight="bold",
        color=trend_color,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.7,
            edgecolor=trend_color,
            linewidth=1.5,
        ),
    )

    # --- HEATMAP ROW: Circadian Glucose Heatmap ---
    if getattr(args, "heatmap", False):
        ax_heat = fig.add_subplot(gs[2, :])

        heat_df = df[["Time", "Sensor Reading(mg/dL)"]].copy()
        heat_df["hour"] = heat_df["Time"].dt.hour
        heat_df["date"] = heat_df["Time"].dt.date
        heat_data = heat_df.pivot_table(
            index="date", columns="hour", values="Sensor Reading(mg/dL)", aggfunc="mean"
        )
        heat_data = heat_data.reindex(columns=range(24))

        heat_img = ax_heat.imshow(
            heat_data.values,
            aspect="auto",
            cmap=args.heatmap_cmap,
            vmin=40,
            vmax=300,
            interpolation="nearest",
        )

        cbar = plt.colorbar(heat_img, ax=ax_heat, pad=0.01)
        cbar.set_label("Glucose (mg/dL)", fontsize=10)

        ax_heat.set_xticks(range(24))
        ax_heat.set_xticklabels(
            [f"{h:02d}:00" for h in range(24)], fontsize=8, rotation=45, ha="right"
        )
        ax_heat.set_xlabel("Hour of Day", fontsize=10)

        date_labels = [str(d) for d in heat_data.index]
        ax_heat.set_yticks(range(len(date_labels)))
        ax_heat.set_yticklabels(date_labels, fontsize=8)
        ax_heat.set_ylabel("Date", fontsize=10)

        WAKE_HOUR = 6
        SLEEP_HOUR = 22
        ax_heat.axvline(
            x=WAKE_HOUR - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.8
        )
        ax_heat.axvline(
            x=SLEEP_HOUR - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.8
        )

        ax_heat.set_title(
            "Circadian Glucose Heatmap (Mean mg/dL per Hour)", fontsize=12, pad=10
        )

    # Header
    date_range_str = format_date_range(df)
    header_text = (
        f"Patient: {report_header['patient_name']} | ID: {report_header['patient_id']}"
    )
    if report_header["doctor"]:
        header_text += f" | Dr: {report_header['doctor']}"
    header_text += f" | Report Date: {report_header['report_date']}"

    plt.figtext(
        0.5,
        0.96,
        header_text,
        ha="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3),
    )

    if report_header["notes"]:
        plt.figtext(
            0.5,
            0.94,
            f"Notes: {report_header['notes']} | Source: {report_header['data_source']} | Data range: {date_range_str}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )
    else:
        plt.figtext(
            0.5,
            0.94,
            f"Source: {report_header['data_source']} | Data range: {date_range_str}",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

    if getattr(args, "heatmap", False):
        plt.suptitle(
            "Ambulatory Glucose Profile with Time in Tight Range (TITR), Raw Data Series and Circadian Heatmap",
            fontsize=14,
            y=0.92,
        )
    else:
        plt.suptitle(
            "Ambulatory Glucose Profile with Time in Tight Range (TITR) and Raw Data Series",
            fontsize=14,
            y=0.92,
        )
    plt.tight_layout()

    metadata = {
        "Description": "Ambulatory Glucose Profile generated by AGP tool",
        "Source": "https://github.com/daedalus/agp",
        "Copyright": "Copyright 2026 Darío Clavijo",
        "License": "MIT License",
    }

    plt.figtext(
        0.5,
        0.02,
        f"{metadata['Source']}\n{metadata['Copyright']}\n{metadata['License']}",
        ha="center",
        fontsize=9,
        color="gray",
        style="italic",
        alpha=0.7,
    )

    _save_path = output_path if output_path is not None else getattr(args, "output", None)
    if _save_path:
        plt.savefig(_save_path, dpi=300, bbox_inches="tight", metadata=metadata)
        if getattr(args, "verbose", False):
            print(f"Plot saved to: {_save_path}")
    if show:
        plt.show()
    if close:
        plt.close()
    return fig
