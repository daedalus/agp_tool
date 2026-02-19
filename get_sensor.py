# Robust CGM Circadian Profile with Clinical Metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

LOW = 70
HIGH = 180
BIN_MINUTES = 5
MIN_SAMPLES_PER_BIN = 5
ROC_CLIP = 10  # mg/dL/min physiological guardrail

# --------------------------------------------------
# 1) Read XLS
# --------------------------------------------------
df = pd.read_excel(sys.argv[1])

if "Time" not in df.columns or "Sensor Reading(mg/dL)" not in df.columns:
    raise ValueError("Missing required columns.")

# --------------------------------------------------
# 2) Clean
# --------------------------------------------------
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time", "Sensor Reading(mg/dL)"])
df = df.sort_values("Time")
df = df.drop_duplicates(subset=["Time"])

if len(df) == 0:
    raise ValueError("No valid glucose data found.")

# --------------------------------------------------
# 3) ROC (safe)
# --------------------------------------------------
df["delta_glucose"] = df["Sensor Reading(mg/dL)"].diff()
df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0

df["ROC"] = df["delta_glucose"] / df["delta_minutes"]
df.loc[df["delta_minutes"] <= 0, "ROC"] = np.nan
df["ROC"] = df["ROC"].clip(-ROC_CLIP, ROC_CLIP)

# --------------------------------------------------
# 4) Variability Metrics
# --------------------------------------------------
glucose = df["Sensor Reading(mg/dL)"].astype(float)

mean_glucose = glucose.mean()
std_glucose = glucose.std(ddof=0)
cv_percent = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else np.nan

# ---------- Smoothed MAGE ----------
def compute_mage(series):
    s = series.rolling(3, center=True).mean().dropna().values
    if len(s) < 3:
        return np.nan

    sd = np.std(s, ddof=0)
    diffs = np.diff(s)
    signs = np.sign(diffs)

    turning_points = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1] and signs[i] != 0:
            turning_points.append(i)

    excursions = []
    for i in range(1, len(turning_points)):
        delta = abs(s[turning_points[i]] - s[turning_points[i - 1]])
        if delta > sd:
            excursions.append(delta)

    return np.mean(excursions) if excursions else np.nan

mage = compute_mage(glucose)

# ---------- Proper CONGA ----------
def compute_conga(df, lag_minutes=60, tolerance=5):
    df_temp = df.set_index("Time")
    target_index = df_temp.index - pd.Timedelta(minutes=lag_minutes)

    lagged = df_temp["Sensor Reading(mg/dL)"].reindex(
        target_index,
        method="nearest",
        tolerance=pd.Timedelta(minutes=tolerance)
    )

    diff = df_temp["Sensor Reading(mg/dL)"].values - lagged.values
    return np.nanstd(diff, ddof=0)

conga = compute_conga(df)

# ---------- Risk indices ----------
def compute_risk_indices(glucose_values):
    g = np.clip(glucose_values.values, 1, None)
    f = 1.509 * ((np.log(g) ** 1.084) - 5.381)
    risk = 10 * (f ** 2)

    lbgi = np.mean(risk[f < 0]) if np.any(f < 0) else 0
    hbgi = np.mean(risk[f > 0]) if np.any(f > 0) else 0
    return lbgi, hbgi

lbgi, hbgi = compute_risk_indices(glucose)

# --------------------------------------------------
# 5) TIR / TAR / TBR
# --------------------------------------------------
total = len(glucose)

tir = ((glucose >= LOW) & (glucose <= HIGH)).sum() / total * 100
tar = (glucose > HIGH).sum() / total * 100
tbr = (glucose < LOW).sum() / total * 100

# --------------------------------------------------
# 6) Circadian bins
# --------------------------------------------------
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

df["bin"] = (df["seconds"] // (BIN_MINUTES * 60)).astype(int)

# --------------------------------------------------
# 7) Aggregate statistics
# --------------------------------------------------
def percentile(x, q):
    return np.percentile(x, q)

grouped = df.groupby("bin")["Sensor Reading(mg/dL)"]

result = grouped.agg(
    mean="mean",
    std=lambda x: np.std(x, ddof=0),
    median="median",
    min="min",
    max="max",
    count="count",
    p10=lambda x: percentile(x, 10),
    p25=lambda x: percentile(x, 25),
    p75=lambda x: percentile(x, 75),
    p90=lambda x: percentile(x, 90),
).reset_index()

roc_profile = df.groupby("bin")["ROC"].mean().reset_index()
roc_profile.rename(columns={"ROC": "roc_mean"}, inplace=True)

result = result.merge(roc_profile, on="bin", how="left")

result.loc[result["count"] < MIN_SAMPLES_PER_BIN,
           result.columns.difference(["bin", "count"])] = np.nan

result = result.sort_values("bin")
result["minutes"] = result["bin"] * BIN_MINUTES

# --------------------------------------------------
# 8) Plot
# --------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 7))

x = result["minutes"]

ax1.plot(x, result["median"], linewidth=2, label="Median")
ax1.plot(x, result["mean"], linestyle="--", linewidth=1.5, label="Mean")

ax1.fill_between(x, result["p25"], result["p75"], alpha=0.30, label="IQR")
ax1.fill_between(x, result["p10"], result["p90"], alpha=0.15, label="10–90%")

ax1.axhline(LOW, linestyle=":", linewidth=1)
ax1.axhline(HIGH, linestyle=":", linewidth=1)

ax1.set_xlabel("Time of Day")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.grid(True)

# ROC axis
ax2 = ax1.twinx()
ax2.plot(x, result["roc_mean"], linestyle=":", linewidth=2, label="ROC (mg/dL/min)")
ax2.set_ylabel("Rate of Change (mg/dL/min)")

# X ticks
xticks = np.arange(0, 1441, 120)
xtick_labels = [f"{int(t//60):02d}:00" for t in xticks]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Metrics box
textstr = (
    f"TIR ({LOW}-{HIGH}): {tir:.1f}%\n"
    f"TAR (> {HIGH}): {tar:.1f}%\n"
    f"TBR (< {LOW}): {tbr:.1f}%\n\n"
    f"Mean: {mean_glucose:.1f}\n"
    f"CV: {cv_percent:.1f}%\n"
    f"MAGE: {mage:.1f}\n"
    f"CONGA(1h): {conga:.1f}\n"
    f"LBGI: {lbgi:.2f}\n"
    f"HBGI: {hbgi:.2f}"
)

plt.gcf().text(0.80, 0.72, textstr, fontsize=10,
               bbox=dict(boxstyle="round", alpha=0.2))

plt.title("Circadian Glucose Profile (Robust Clinical Version)")
plt.tight_layout()
plt.savefig("average_daily_profile.png", dpi=300)
plt.show()
plt.close()

