import numpy as np
import pandas as pd

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def compute_mage(series):
    s = series.rolling(3, center=True).mean().dropna().values
    if len(s) < 3:
        return np.nan

    sd = np.std(s, ddof=0)
    diffs = np.diff(s)
    signs = np.sign(diffs)

    turning = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0:
            turning.append(i)

    excursions = []
    for i in range(1, len(turning)):
        delta = abs(s[turning[i]] - s[turning[i - 1]])
        if delta > sd:
            excursions.append(delta)

    return np.mean(excursions) if excursions else np.nan


def compute_modd(df):
    """Calculate Mean of Daily Differences - day-to-day glucose variability."""
    df_daily = df.copy()
    df_daily["date"] = df_daily["Time"].dt.date
    df_daily["time"] = df_daily["Time"].dt.time

    # Create time index for matching across days
    df_daily["time_seconds"] = (
        df_daily["Time"].dt.hour * 3600
        + df_daily["Time"].dt.minute * 60
        + df_daily["Time"].dt.second
    )

    # Pivot to get glucose by time across days
    pivot = df_daily.pivot_table(
        values="Sensor Reading(mg/dL)",
        index="time_seconds",
        columns="date",
        aggfunc="mean",
    )

    # Calculate differences between consecutive days
    diffs = []
    dates = sorted(pivot.columns)
    for i in range(len(dates) - 1):
        day1 = pivot[dates[i]].dropna()
        day2 = pivot[dates[i + 1]].dropna()

        # Find common time points
        common_times = day1.index.intersection(day2.index)
        if len(common_times) > 0:
            day_diff = np.abs(day1.loc[common_times] - day2.loc[common_times])
            diffs.extend(day_diff.dropna().values)

    return np.mean(diffs) if diffs else np.nan


def compute_adrr(values, dates):
    """Calculate Average Daily Risk Range."""

    def risk_function(g):
        g = np.clip(g, 20, 600)
        return 10 * (1.509 * ((np.log(g)) ** 1.084 - 5.381)) ** 2

    # Group by date
    df_risk = pd.DataFrame({"glucose": values, "date": dates})
    daily_risk_range = []

    for _, group in df_risk.groupby("date"):
        if len(group) >= 12:  # At least 12 readings per day (every 2 hours)
            risks = risk_function(group["glucose"].values)
            daily_range = np.max(risks) - np.min(risks)
            daily_risk_range.append(daily_range)

    return np.mean(daily_risk_range) if daily_risk_range else np.nan


def compute_conga(df, lag_minutes=60, tolerance=5):
    df_temp = df.set_index("Time")
    target_index = df_temp.index - pd.Timedelta(minutes=lag_minutes)

    lagged = df_temp["Sensor Reading(mg/dL)"].reindex(
        target_index, method="nearest", tolerance=pd.Timedelta(minutes=tolerance)
    )

    diff = df_temp["Sensor Reading(mg/dL)"].values - lagged.values
    return np.nanstd(diff, ddof=0)


def compute_risk_indices(values):
    g = np.clip(values.values, 1, None)
    f = 1.509 * ((np.log(g) ** 1.084) - 5.381)
    risk = 10 * (f**2)

    lbgi = np.mean(risk[f < 0]) if np.any(f < 0) else 0
    hbgi = np.mean(risk[f > 0]) if np.any(f > 0) else 0
    return lbgi, hbgi


def compute_gri(very_low_pct, low_pct, very_high_pct, high_pct):
    raw = (
        (3.0 * very_low_pct)
        + (2.4 * low_pct)
        + (1.6 * very_high_pct)
        + (0.8 * high_pct)
    )
    return min(raw, 100.0)


def compute_core_metrics(df, cfg):
    """Compute core descriptive glucose statistics."""
    glucose = df["Sensor Reading(mg/dL)"]

    mean_glucose = glucose.mean()
    median_glucose = glucose.median()
    std_glucose = glucose.std(ddof=0)
    mode_glucose = glucose.mode()
    skew_glucose = glucose.skew()
    cv_percent = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else np.nan

    mode_str = f"{mode_glucose.iloc[0]:.1f}" if not mode_glucose.empty else "N/A"

    # skew interpretation
    if skew_glucose < -1:
        skew_interpretation = "Left-skewed (hypoglycemia tendency)"
        skew_clinical = "Review hypoglycemia patterns"
    elif skew_glucose > 1.5:
        skew_interpretation = "Highly right-skewed (significant hyperglycemia)"
        skew_clinical = "High diabetes burden in population"
    elif skew_glucose > 1.0:
        skew_interpretation = "Moderately right-skewed (hyperglycemia present)"
        skew_clinical = "Mean exceeds median - use median for typical value"
    elif skew_glucose > 0.5:
        skew_interpretation = "Mildly right-skewed (expected pattern)"
        skew_clinical = "Typical glucose distribution"
    else:
        skew_interpretation = "Approximately symmetric"
        skew_clinical = "Normal distribution pattern"

    # Day/Night CV
    day_mask = (df["Time"].dt.hour >= 6) & (df["Time"].dt.hour < 22)
    night_mask = ~day_mask
    if day_mask.any() and glucose[day_mask].mean() > 0:
        day_cv = glucose[day_mask].std() / glucose[day_mask].mean() * 100
    else:
        day_cv = np.nan
    if night_mask.any() and glucose[night_mask].mean() > 0:
        night_cv = glucose[night_mask].std() / glucose[night_mask].mean() * 100
    else:
        night_cv = np.nan

    # GMI and J-Index
    gmi = 3.31 + (0.02392 * mean_glucose)
    j_index = 0.001 * (mean_glucose + std_glucose) ** 2

    return {
        "mean_glucose": mean_glucose,
        "median_glucose": median_glucose,
        "std_glucose": std_glucose,
        "mode_str": mode_str,
        "skew_glucose": skew_glucose,
        "skew_interpretation": skew_interpretation,
        "skew_clinical": skew_clinical,
        "gmi": gmi,
        "cv_percent": cv_percent,
        "day_cv": day_cv,
        "night_cv": night_cv,
        "j_index": j_index,
    }


def compute_variability_metrics(df, cfg):
    """Compute glucose variability metrics."""
    glucose = df["Sensor Reading(mg/dL)"]

    mage = compute_mage(glucose)
    modd = compute_modd(df)
    adrr = compute_adrr(glucose, df["Time"].dt.date)
    conga = compute_conga(df)
    lbgi, hbgi = compute_risk_indices(glucose)

    return {
        "mage": mage,
        "modd": modd,
        "adrr": adrr,
        "conga": conga,
        "lbgi": lbgi,
        "hbgi": hbgi,
    }


def compute_time_in_range_metrics(df, cfg):
    """Compute time-in-range, time-above-range, and time-below-range metrics."""
    VERY_LOW = cfg["VERY_LOW"]
    LOW = cfg["LOW"]
    HIGH = cfg["HIGH"]
    VERY_HIGH = cfg["VERY_HIGH"]
    TIGHT_LOW = cfg["TIGHT_LOW"]
    TIGHT_HIGH = cfg["TIGHT_HIGH"]

    glucose = df["Sensor Reading(mg/dL)"]
    total = len(glucose)

    very_low_pct = (glucose < VERY_LOW).sum() / total * 100
    low_pct = ((glucose >= VERY_LOW) & (glucose < TIGHT_LOW)).sum() / total * 100
    tight_target_pct = (
        ((glucose >= TIGHT_LOW) & (glucose <= TIGHT_HIGH)).sum() / total * 100
    )
    above_tight_pct = ((glucose > TIGHT_HIGH) & (glucose <= HIGH)).sum() / total * 100
    high_pct = ((glucose > HIGH) & (glucose <= VERY_HIGH)).sum() / total * 100
    very_high_pct = (glucose > VERY_HIGH).sum() / total * 100

    tir = ((glucose >= LOW) & (glucose <= HIGH)).sum() / total * 100
    titr = tight_target_pct

    tar_level1 = high_pct
    tar_level2 = very_high_pct
    tar = tar_level1 + tar_level2
    tatr = above_tight_pct

    tbr_level2 = very_low_pct
    tbr_level1 = low_pct
    tbr = tbr_level1 + tbr_level2

    gri = compute_gri(very_low_pct, low_pct, very_high_pct, high_pct)
    if gri < 20:
        gri_txt = "Low Risk"
    elif gri < 40:
        gri_txt = "Moderate Risk"
    elif gri < 60:
        gri_txt = "High Risk"
    elif gri < 80:
        gri_txt = "Very High Risk"
    else:
        gri_txt = "Extremely High Risk"

    return {
        "tir": tir,
        "titr": titr,
        "tatr": tatr,
        "tar": tar,
        "tar_level1": tar_level1,
        "tar_level2": tar_level2,
        "tbr": tbr,
        "tbr_level1": tbr_level1,
        "tbr_level2": tbr_level2,
        "very_low_pct": very_low_pct,
        "low_pct": low_pct,
        "tight_target_pct": tight_target_pct,
        "above_tight_pct": above_tight_pct,
        "high_pct": high_pct,
        "very_high_pct": very_high_pct,
        "gri": gri,
        "gri_txt": gri_txt,
    }


def compute_auc_metrics(df, cfg):
    """Compute area-under-the-curve glucose exposure metrics."""
    LOW = cfg["LOW"]
    HIGH = cfg["HIGH"]
    VERY_LOW = cfg["VERY_LOW"]

    df_auc = df.sort_values("Time").copy()
    df_auc["time_minutes"] = (
        df_auc["Time"] - df_auc["Time"].iloc[0]
    ).dt.total_seconds() / 60.0

    times = df_auc["time_minutes"].values
    values = df_auc["Sensor Reading(mg/dL)"].values

    auc_total = trapz(values, times)
    auc_high = trapz(np.maximum(values - HIGH, 0), times)
    auc_low = trapz(np.maximum(LOW - values, 0), times)
    auc_very_low = trapz(np.maximum(VERY_LOW - values, 0), times)

    time_weighted_avg = auc_total / times[-1] if times[-1] > 0 else np.nan
    exposure_severity_to_hyperglycemia_pct = (
        (auc_high / auc_total) * 100 if auc_total > 0 else 0
    )
    exposure_severity_to_hypoglycemia_pct = (
        (auc_low / auc_total) * 100 if auc_total > 0 else 0
    )
    exposure_severity_to_severe_hypoglycemia_pct = (
        (auc_very_low / auc_total) * 100 if auc_total > 0 else 0
    )

    return {
        "time_weighted_avg": time_weighted_avg,
        "exposure_severity_to_hyperglycemia_pct": exposure_severity_to_hyperglycemia_pct,
        "exposure_severity_to_hypoglycemia_pct": exposure_severity_to_hypoglycemia_pct,
        "exposure_severity_to_severe_hypoglycemia_pct": exposure_severity_to_severe_hypoglycemia_pct,
    }


def compute_data_quality_metrics(df, cfg):
    """Compute data completeness and quality metrics."""
    SENSOR_INTERVAL = cfg["SENSOR_INTERVAL"]

    glucose = df["Sensor Reading(mg/dL)"]

    days_of_data = (df["Time"].max() - df["Time"].min()).total_seconds() / 86400.0
    days_of_data = max(days_of_data, 1 / 24)  # Minimum 1 hour to avoid division by zero
    hours_of_data = (df["Time"].max() - df["Time"].min()).total_seconds() / 3600
    readings_per_day = len(df) / days_of_data if days_of_data > 0 else len(df)

    total_possible_readings = days_of_data * (24 * 60 / SENSOR_INTERVAL)
    wear_percentage = (
        (len(df) / total_possible_readings) * 100
        if total_possible_readings > 0
        else np.nan
    )

    severe_hypo_count = (glucose < 40).sum()
    severe_hypo_per_week = (
        (severe_hypo_count / days_of_data) * 7 if days_of_data > 0 else np.nan
    )

    return {
        "days_of_data": days_of_data,
        "hours_of_data": hours_of_data,
        "readings_per_day": readings_per_day,
        "wear_percentage": wear_percentage,
        "severe_hypo_count": severe_hypo_count,
        "severe_hypo_per_week": severe_hypo_per_week,
    }


def compute_overall_glucose_trend(df, cfg):
    """Compute the overall glucose trend direction and slope."""
    time_days = (df["Time"] - df["Time"].min()).dt.total_seconds() / 86400.0
    if time_days.max() > 0 and len(time_days) >= 2:
        trend_slope, _ = np.polyfit(time_days, df["Sensor Reading(mg/dL)"], 1)
    else:
        trend_slope = 0.0

    if trend_slope > 1:
        trend_direction = "UP"
        trend_arrow = "↑"
        trend_color = "orangered"
    elif trend_slope < -1:
        trend_direction = "DOWN"
        trend_arrow = "↓"
        trend_color = "mediumseagreen"
    else:
        trend_direction = "STABLE"
        trend_arrow = "→"
        trend_color = "steelblue"

    return {
        "trend_slope": trend_slope,
        "trend_direction": trend_direction,
        "trend_arrow": trend_arrow,
        "trend_color": trend_color,
    }


def compute_all_metrics(df, cfg):
    """Orchestrate all metric computations and return a comprehensive dict."""
    metrics = {}
    metrics.update(compute_core_metrics(df, cfg))
    metrics.update(compute_variability_metrics(df, cfg))
    metrics.update(compute_time_in_range_metrics(df, cfg))
    metrics.update(compute_auc_metrics(df, cfg))
    metrics.update(compute_data_quality_metrics(df, cfg))
    metrics.update(compute_overall_glucose_trend(df, cfg))
    return metrics
