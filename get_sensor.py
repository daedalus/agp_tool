# Takes in sibionics gs1 sensor data
# Outputs an image with a circadian glucose profile including variability bands

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# -------------------------
# 1) Read XLS file
# -------------------------
df = pd.read_excel(sys.argv[1])

# -------------------------
# 2) Clean & parse time
# -------------------------
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.dropna(subset=["Time", "Sensor Reading(mg/dL)"])

# -------------------------
# 3) Seconds from midnight
# -------------------------
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

# -------------------------
# 4) 5-minute bins
# -------------------------
bin_size = 300
df["bin"] = (df["seconds"] / bin_size).round().astype(int)

# -------------------------
# 5) Aggregate statistics
# -------------------------
result = df.groupby("bin")["Sensor Reading(mg/dL)"].agg(
    mean="mean",
    std="std",
    median="median",
    min="min",
    max="max",
    count="count"
).reset_index()

# Optional: remove bins with very few samples
min_samples = 5
result.loc[result["count"] < min_samples,
           ["mean", "std", "median", "min", "max"]] = np.nan

# Sort & convert to minutes
result = result.sort_values("bin")
result["minutes"] = result["bin"] * 5

# -------------------------
# 6) Plot
# -------------------------
plt.figure(figsize=(14, 7))

x = result["minutes"]
mean = result["mean"]
std = result["std"]
median = result["median"]
minv = result["min"]
maxv = result["max"]

# Mean line
plt.plot(x, mean, linewidth=2, label="Mean")

# Median line
plt.plot(x, median, linestyle="--", linewidth=1.5, label="Median")

# Mean ± Std band
plt.fill_between(
    x,
    mean - std,
    mean + std,
    alpha=0.25,
    label="Mean ± Std"
)

# Min–Max envelope
plt.fill_between(
    x,
    minv,
    maxv,
    alpha=0.12,
    label="Min–Max"
)

# Format X axis (HH:MM every 2 hours)
xticks = np.arange(0, 1441, 120)
xtick_labels = [f"{int(t//60):02d}:00" for t in xticks]
plt.xticks(xticks, xtick_labels)

plt.xlabel("Time of Day")
plt.ylabel("Glucose (mg/dL)")
plt.title("Circadian Glucose Profile with Variability")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("average_daily_profile.png", dpi=300)
plt.show()
plt.close()

