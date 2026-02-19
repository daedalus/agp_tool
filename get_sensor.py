# Full AGP + GMI + Robust Metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

LOW = 70
HIGH = 180
BIN_MINUTES = 5
MIN_SAMPLES_PER_BIN = 5
ROC_CLIP = 10

# --------------------------------------------------
# 1) Read Data
# --------------------------------------------------
df = pd.read_excel(sys.argv[1])

required = ["Time", "Sensor Reading(mg/dL)"]
if not all(col in df.columns for col in required):
    raise ValueError("Missing required columns.")

df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=required)
df = df.sort_values("Time").drop_duplicates(subset=["Time"])

if len(df) == 0:
    raise ValueError("No valid glucose data.")

# --------------------------------------------------
# 2) ROC
# --------------------------------------------------
df["delta_glucose"] = df["Sensor Reading(mg/dL)"].diff()
df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0
df["ROC"] = df["delta_glucose"] / df["delta_minutes"]
df.loc[df["delta_minutes"] <= 0, "ROC"] = np.nan
df["ROC"] = df["ROC"].clip(-ROC_CLIP, ROC_CLIP)

glucose = df["Sensor Reading(mg/dL)"].astype(float)

# --------------------------------------------------
# 3) Core Metrics
# --------------------------------------------------
mean_glucose = glucose.mean()
std_glucose = glucose.std(ddof=0)
cv_percent = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else np.nan

# GMI
gmi = 3.31 + (0.02392 * mean_glucose)

# --------------------------------------------------
# 4) MAGE (smoothed)
# --------------------------------------------------
def compute_mage(series):
    s = series.rolling(3, center=True).mean().dropna().values
    if len(s) < 3:
        return np.nan

    sd = np.std(s, ddof=0)
    diffs = np.diff(s)
    signs = np.sign(diffs)

    turning = []
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1] and signs[i] != 0:
            turning.append(i)

    excursions = []
    for i in range(1, len(turning)):
        delta = abs(s[turning[i]] - s[turning[i-1]])
        if delta > sd:
            excursions.append(delta)

    return np.mean(excursions) if excursions else np.nan

mage = compute_mage(glucose)

# --------------------------------------------------
# 5) CONGA
# --------------------------------------------------
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

# --------------------------------------------------
# 6) Risk Indices
# --------------------------------------------------
def compute_risk_indices(values):
    g = np.clip(values.values, 1, None)
    f = 1.509 * ((np.log(g) ** 1.084) - 5.381)
    risk = 10 * (f ** 2)

    lbgi = np.mean(risk[f < 0]) if np.any(f < 0) else 0
    hbgi = np.mean(risk[f > 0]) if np.any(f > 0) else 0
    return lbgi, hbgi

lbgi, hbgi = compute_risk_indices(glucose)

# --------------------------------------------------
# 7) TIR / TAR / TBR
# --------------------------------------------------
total = len(glucose)
tir = ((glucose >= LOW) & (glucose <= HIGH)).sum() / total * 100
tar = (glucose > HIGH).sum() / total * 100
tbr = (glucose < LOW).sum() / total * 100

# --------------------------------------------------
# 8) Circadian Binning
# --------------------------------------------------
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

df["bin"] = (df["seconds"] // (BIN_MINUTES * 60)).astype(int)

def pct(x, q):
    return np.percentile(x, q)

grouped = df.groupby("bin")["Sensor Reading(mg/dL)"]

result = grouped.agg(
    count="count",
    p5=lambda x: pct(x, 5),
    p10=lambda x: pct(x, 10),
    p25=lambda x: pct(x, 25),
    median="median",
    p75=lambda x: pct(x, 75),
    p90=lambda x: pct(x, 90),
    p95=lambda x: pct(x, 95),
).reset_index()

roc_profile = df.groupby("bin")["ROC"].mean().reset_index()
roc_profile.rename(columns={"ROC": "roc_mean"}, inplace=True)

result = result.merge(roc_profile, on="bin", how="left")

result.loc[result["count"] < MIN_SAMPLES_PER_BIN,
           result.columns.difference(["bin", "count"])] = np.nan

result = result.sort_values("bin")
result["minutes"] = result["bin"] * BIN_MINUTES

# --------------------------------------------------
# 9) Plot AGP
# --------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 7))
x = result["minutes"]

# 5–95%
ax1.fill_between(x, result["p5"], result["p95"], alpha=0.15, label="5–95%")

# 25–75% (IQR)
ax1.fill_between(x, result["p25"], result["p75"], alpha=0.35, label="25–75% (IQR)")

# Median
ax1.plot(x, result["median"], linewidth=2.5, label="Median")

ax1.axhline(LOW, linestyle=":", linewidth=1)
ax1.axhline(HIGH, linestyle=":", linewidth=1)

ax1.set_xlabel("Time of Day")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.grid(True)

# ROC secondary axis
ax2 = ax1.twinx()
ax2.plot(x, result["roc_mean"], linestyle="--", linewidth=1.5,
         label="ROC (mg/dL/min)")
ax2.set_ylabel("Rate of Change (mg/dL/min)")

xticks = np.arange(0, 1441, 120)
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# --------------------------------------------------
# Metrics Box
# --------------------------------------------------
textstr = (
    f"TIR: {tir:.1f}%\n"
    f"TAR: {tar:.1f}%\n"
    f"TBR: {tbr:.1f}%\n\n"
    f"Mean: {mean_glucose:.1f}\n"
    f"GMI: {gmi:.2f}%\n"
    f"CV: {cv_percent:.1f}%\n\n"
    f"MAGE: {mage:.1f}\n"
    f"CONGA(1h): {conga:.1f}\n"
    f"LBGI: {lbgi:.2f}\n"
    f"HBGI: {hbgi:.2f}"
)

plt.gcf().text(0.80, 0.70, textstr, fontsize=10,
               bbox=dict(boxstyle="round", alpha=0.2))

plt.title("Ambulatory Glucose Profile (Full AGP + GMI)")
plt.tight_layout()
plt.savefig("agp_profile.png", dpi=300)
plt.show()
plt.close()

