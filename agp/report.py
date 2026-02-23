from datetime import datetime


def create_report_header(args):
    """Create a formatted header dictionary with patient and report information."""
    header = {
        "patient_name": args.patient_name,
        "patient_id": args.patient_id,
        "doctor": args.doctor,
        "notes": args.notes,
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "report_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": args.input_file,
    }
    return header


def print_clinical_summary(metrics, report_header, cfg):
    """Print the formatted clinical summary to stdout."""
    VERY_LOW = cfg["VERY_LOW"]
    LOW = cfg["LOW"]
    HIGH = cfg["HIGH"]
    VERY_HIGH = cfg["VERY_HIGH"]
    TIGHT_LOW = cfg["TIGHT_LOW"]
    TIGHT_HIGH = cfg["TIGHT_HIGH"]
    tir = metrics["tir"]
    titr = metrics["titr"]
    tbr = metrics["tbr"]
    cv_percent = metrics["cv_percent"]
    trend_arrow = metrics["trend_arrow"]
    trend_direction = metrics["trend_direction"]
    trend_slope = metrics["trend_slope"]
    gri = metrics["gri"]
    gri_txt = metrics["gri_txt"]
    days_of_data = metrics["days_of_data"]
    hours_of_data = metrics["hours_of_data"]
    readings_per_day = metrics["readings_per_day"]
    wear_percentage = metrics["wear_percentage"]
    skew_glucose = metrics["skew_glucose"]
    skew_interpretation = metrics["skew_interpretation"]
    mean_glucose = metrics["mean_glucose"]
    median_glucose = metrics["median_glucose"]
    very_low_pct = metrics["very_low_pct"]
    low_pct = metrics["low_pct"]
    tight_target_pct = metrics["tight_target_pct"]
    above_tight_pct = metrics["above_tight_pct"]
    high_pct = metrics["high_pct"]
    very_high_pct = metrics["very_high_pct"]

    print("\n" + "=" * 60)
    print(
        f"PATIENT: {report_header['patient_name']} (ID: {report_header['patient_id']})"
    )
    print(f"REPORT DATE: {report_header['report_date']}")
    print(f"Hours of data: {hours_of_data:.1f}")
    print("\n" + "=" * 60)
    print("CLINICAL SUMMARY")
    print("=" * 60)
    print(
        f"Time in Range ({LOW}-{HIGH}): {tir:.1f}% - {'Target met (≥70%)' if tir >= 70 else 'Below target'}"
    )
    print(
        f"Time in Tight Range ({TIGHT_LOW}-{TIGHT_HIGH}): {titr:.1f}% - {'Excellent' if titr >= 50 else 'Room for improvement'}"
    )
    print(
        f"Time Below Range: {tbr:.1f}% - {'Target met (<4%)' if tbr < 4 else 'Above target'}"
    )
    print(
        f"Glucose Variability (CV): {cv_percent:.1f}% - {'Stable (<36%)' if cv_percent < 36 else 'Unstable (≥36%)'}"
    )
    print(
        f"Overall Trend: {trend_arrow} {trend_direction} (slope: {trend_slope:.1f} mg/dL/day)"
    )
    print(f"Glycemia Risk Index (GRI): {gri:.1f} - {gri_txt}\n")
    print("-" * 60)

    if days_of_data < 5:
        print(
            f"Warning: Only {days_of_data:.1f} days of data. AGP typically requires ≥5 days for reliability."
        )
    if readings_per_day < 24:
        print(
            f"Warning: Low reading frequency ({readings_per_day:.0f} readings/day). Continuous glucose monitor expected."
        )
    if wear_percentage < 70:
        print(
            f"Warning: Low sensor wear time ({wear_percentage:.1f}%). Results may not be representative."
        )

    print(f"Distribution Shape: skew = {skew_glucose:.2f} - {skew_interpretation}")
    if 1.0 < skew_glucose < 1.5:
        print(
            f"  → Note: In this 'gray zone', the mean ({mean_glucose:.1f}) exceeds the median ({median_glucose:.1f})"
        )
        print(f"  → The median better represents typical glucose exposure")
    print("-" * 60)
    print("\nGlucose Distribution Summary:")
    print(f"  Very Low (<{VERY_LOW} mg/dL): {very_low_pct:.1f}%")
    print(f"  Low ({VERY_LOW}-{LOW - 1} mg/dL): {low_pct:.1f}%")
    print(f"  Tight Target ({TIGHT_LOW}-{TIGHT_HIGH} mg/dL): {tight_target_pct:.1f}%")
    print(f"  Above Tight ({TIGHT_HIGH + 1}-{HIGH} mg/dL): {above_tight_pct:.1f}%")
    print(f"  High ({HIGH + 1}-{VERY_HIGH} mg/dL): {high_pct:.1f}%")
    print(f"  Very High (>{VERY_HIGH} mg/dL): {very_high_pct:.1f}%")
