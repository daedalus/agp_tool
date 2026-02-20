# Unified Robust AGP + GMI + AUC + Full Clinical Metrics + TITR + Raw Data Series (Bottom)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import timedelta
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

LOW = 70
HIGH = 180
TIGHT_LOW = 70
TIGHT_HIGH = 140  # Tight range upper limit
BIN_MINUTES = 5
MIN_SAMPLES_PER_BIN = 5
ROC_CLIP = 10  # mg/dL/min physiological guardrail

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
    raise ValueError("No valid glucose data found.")

# Filter physiologically impossible values
glucose = df["Sensor Reading(mg/dL)"].astype(float)
glucose = glucose.clip(20, 600)  # Physiological limits
df["Sensor Reading(mg/dL)"] = glucose

# --------------------------------------------------
# 2) Rate of Change (ROC)
# --------------------------------------------------
df["delta_glucose"] = glucose.diff()
df["delta_minutes"] = df["Time"].diff().dt.total_seconds() / 60.0
df["ROC"] = df["delta_glucose"] / df["delta_minutes"]
df.loc[df["delta_minutes"] <= 0, "ROC"] = np.nan
df["ROC"] = df["ROC"].clip(-ROC_CLIP, ROC_CLIP)

# --------------------------------------------------
# 3) Core Metrics
# --------------------------------------------------
mean_glucose = glucose.mean()
std_glucose = glucose.std(ddof=0)
cv_percent = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else np.nan

# GMI
gmi = 3.31 + (0.02392 * mean_glucose)

# J-Index (combines mean and variability)
j_index = 0.001 * (mean_glucose + std_glucose) ** 2

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
        if signs[i] != signs[i - 1] and signs[i] != 0:
            turning.append(i)

    excursions = []
    for i in range(1, len(turning)):
        delta = abs(s[turning[i]] - s[turning[i - 1]])
        if delta > sd:
            excursions.append(delta)

    return np.mean(excursions) if excursions else np.nan

mage = compute_mage(glucose)

# --------------------------------------------------
# 5) MODD (Mean of Daily Differences)
# --------------------------------------------------
def compute_modd(df):
    """Calculate Mean of Daily Differences - day-to-day glucose variability"""
    df_daily = df.copy()
    df_daily['date'] = df_daily['Time'].dt.date
    df_daily['time'] = df_daily['Time'].dt.time
    
    # Create time index for matching across days
    df_daily['time_seconds'] = (
        df_daily['Time'].dt.hour * 3600 +
        df_daily['Time'].dt.minute * 60 +
        df_daily['Time'].dt.second
    )
    
    # Pivot to get glucose by time across days
    pivot = df_daily.pivot_table(
        values='Sensor Reading(mg/dL)',
        index='time_seconds',
        columns='date',
        aggfunc='mean'
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

modd = compute_modd(df)

# --------------------------------------------------
# 6) ADRR (Average Daily Risk Range)
# --------------------------------------------------
def compute_adrr(values, dates):
    """Calculate Average Daily Risk Range"""
    def risk_function(g):
        # Convert glucose to risk score
        g = np.clip(g, 20, 600)
        return 10 * (1.509 * ((np.log(g)) ** 1.084 - 5.381)) ** 2
    
    # Group by date
    df_risk = pd.DataFrame({'glucose': values, 'date': dates})
    daily_risk_range = []
    
    for date, group in df_risk.groupby('date'):
        if len(group) >= 12:  # At least 12 readings per day (every 2 hours)
            risks = risk_function(group['glucose'].values)
            daily_range = np.max(risks) - np.min(risks)
            daily_risk_range.append(daily_range)
    
    return np.mean(daily_risk_range) if daily_risk_range else np.nan

adrr = compute_adrr(glucose, df['Time'].dt.date)

# --------------------------------------------------
# 7) CONGA
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
# 8) Risk Indices
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
# 9) TIR / TAR / TBR / TITR with levels
# --------------------------------------------------
total = len(glucose)

# Calculate percentages for each glucose range
very_low_pct = (glucose < 54).sum() / total * 100
low_pct = ((glucose >= 54) & (glucose < LOW)).sum() / total * 100
target_pct = ((glucose >= LOW) & (glucose <= HIGH)).sum() / total * 100
high_pct = ((glucose > HIGH) & (glucose <= 250)).sum() / total * 100
very_high_pct = (glucose > 250).sum() / total * 100

# Time in Range (Standard: 70-180)
tir = target_pct

# Time in Tight Range (70-140) - NEW METRIC
titr = ((glucose >= TIGHT_LOW) & (glucose <= TIGHT_HIGH)).sum() / total * 100

# Time Above Range (with levels)
tar_level1 = high_pct
tar_level2 = very_high_pct
tar = tar_level1 + tar_level2

# Time Above Tight Range (140-180) - supplementary
tatr = ((glucose > TIGHT_HIGH) & (glucose <= HIGH)).sum() / total * 100

# Time Below Range (with levels)
tbr_level2 = very_low_pct
tbr_level1 = low_pct
tbr = tbr_level1 + tbr_level2

# --------------------------------------------------
# 10) AUC Metrics
# --------------------------------------------------
df_auc = df.sort_values("Time").copy()
df_auc["time_minutes"] = (
    (df_auc["Time"] - df_auc["Time"].iloc[0]).dt.total_seconds() / 60.0
)

times = df_auc["time_minutes"].values
values = df_auc["Sensor Reading(mg/dL)"].values

auc_total = np.trapezoid(values, times)
auc_high = np.trapezoid(np.maximum(values - HIGH, 0), times)
auc_low = np.trapezoid(np.maximum(LOW - values, 0), times)

time_weighted_avg = auc_total / times[-1]  # mg/dL (time-weighted average)
time_in_hyperglycemia_pct = (auc_high / auc_total) * 100 if auc_total > 0 else 0
time_in_hypoglycemia_pct = (auc_low / auc_total) * 100 if auc_total > 0 else 0

# --------------------------------------------------
# 11) Data Quality Metrics
# --------------------------------------------------
days_of_data = (df['Time'].max() - df['Time'].min()).days
hours_of_data = (df['Time'].max() - df['Time'].min()).total_seconds() / 3600
readings_per_day = len(df) / days_of_data if days_of_data > 0 else len(df)

# Sensor wear time percentage
total_possible_readings = days_of_data * (24 * 60 / BIN_MINUTES)  # Theoretical max based on bin size
wear_percentage = (len(df) / total_possible_readings) * 100 if total_possible_readings > 0 else np.nan

# Severe hypoglycemia events
severe_hypo_count = (glucose < 40).sum()
severe_hypo_per_week = (severe_hypo_count / days_of_data) * 7 if days_of_data > 0 else np.nan

# --------------------------------------------------
# 12) Circadian Binning
# --------------------------------------------------
df["seconds"] = (
    df["Time"].dt.hour * 3600 +
    df["Time"].dt.minute * 60 +
    df["Time"].dt.second
)

df["bin"] = (df["seconds"] // (BIN_MINUTES * 60)).astype(int)

grouped = df.groupby("bin")["Sensor Reading(mg/dL)"]

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

roc_profile = df.groupby("bin")["ROC"].mean().reset_index()
roc_profile.rename(columns={"ROC": "roc_mean"}, inplace=True)

result = result.merge(roc_profile, on="bin", how="left")

result.loc[result["count"] < MIN_SAMPLES_PER_BIN,
           result.columns.difference(["bin", "count"])] = np.nan

result = result.sort_values("bin")
result["minutes"] = result["bin"] * BIN_MINUTES

# --------------------------------------------------
# 13) Create color-coded data series for raw readings
# --------------------------------------------------
# Create a categorical column for glucose ranges
df['glucose_range'] = pd.cut(df['Sensor Reading(mg/dL)'], 
                              bins=[0, 54, 70, 180, 250, 1000],
                              labels=['Very Low', 'Low', 'Target', 'High', 'Very High'])

# --------------------------------------------------
# 14) Plot AGP with Internal Metrics Box, TITR Band, and Distribution Stacked Bar (ORIGINAL LAYOUT)
#     PLUS Raw Data Series at the Bottom
# --------------------------------------------------
# Create figure with GridSpec for custom layout - now with 2 rows (original + bottom)
fig = plt.figure(figsize=(18, 12))
# Create 12 columns for top row, and full width for bottom row
gs = GridSpec(2, 12, figure=fig, 
              height_ratios=[3, 1.5],  # Original AGP:distribution : raw data series
              hspace=0.3, wspace=0.3)

# --- TOP ROW: EXACTLY THE ORIGINAL LAYOUT (unchanged) ---
# Left subplot for stacked bar chart (spanning first 2 columns)
ax_bar = fig.add_subplot(gs[0, :2])

# Right subplot for AGP (spanning remaining 10 columns)
ax1 = fig.add_subplot(gs[0, 2:])

# Create stacked bar chart of glucose distribution (EXACTLY AS ORIGINAL)
categories = ['Very Low\n(<54)', 'Low\n(54-69)', 'Target\n(70-180)', 
              'High\n(181-250)', 'Very High\n(>250)']
percentages = [very_low_pct, low_pct, target_pct, high_pct, very_high_pct]
colors = ['darkred', 'red', 'limegreen', 'orange', 'darkorange']
labels = ['Very Low (<54)', 'Low (54-69)', 'Target (70-180)', 
          'High (181-250)', 'Very High (>250)']

# Create vertical stacked bar
bottoms = np.zeros(1)
bars = []
for i, (pct, color) in enumerate(zip(percentages, colors)):
    if pct > 0:
        bar = ax_bar.bar(0, pct, bottom=bottoms[0], color=color, edgecolor='white', 
                        linewidth=1, width=0.5)
        bars.append(bar)
        
        # Add percentage label if segment is large enough
        if pct > 3:
            y_pos = bottoms[0] + pct/2
            ax_bar.text(0, y_pos, f'{pct:.1f}%', ha='center', va='center', 
                       color='white', fontweight='bold', fontsize=9)
        bottoms[0] += pct

# Customize bar chart
ax_bar.set_ylim(0, 100)
ax_bar.set_xlim(-0.5, 0.5)
ax_bar.set_xticks([])
ax_bar.set_ylabel('Percentage of Time (%)', fontsize=10)
ax_bar.set_title('Glucose Distribution\nby Range', fontsize=11, pad=10)

# Add gridlines
ax_bar.yaxis.grid(True, alpha=0.3, linestyle='--')
ax_bar.set_axisbelow(True)

# Add legend INSIDE the stacked chart at the bottom
legend_elements = []
for i, (label, color, pct) in enumerate(zip(labels, colors, percentages)):
    if pct > 0:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white'))

# Position legend at the bottom of the bar chart
if legend_elements:
    ax_bar.legend(legend_elements, 
                 [label for label, pct in zip(labels, percentages) if pct > 0],
                 loc='lower center', bbox_to_anchor=(0.5, 0.05), 
                 ncol=1, fontsize=8, frameon=True, fancybox=True, shadow=True,
                 facecolor='white', edgecolor='gray')

# Now create the main AGP plot on ax1 (EXACTLY AS ORIGINAL)
x = result["minutes"]

# Add target zones with distinct colors and alpha
ax1.axhspan(TIGHT_LOW, TIGHT_HIGH, alpha=0.15, color='limegreen', 
            label=f'Tight Target ({TIGHT_LOW}-{TIGHT_HIGH})')
ax1.axhspan(HIGH, 600, alpha=0.1, color='orange', 
            label='Above Range (>180)')
ax1.axhspan(20, LOW, alpha=0.1, color='red', 
            label='Below Range (<70)')

# Add the standard target range as a lighter overlay to show the full target
ax1.axhspan(HIGH, TIGHT_HIGH, alpha=0.1, color='yellowgreen')  # 140-180 zone

# Add trend line
ax1.axhline(mean_glucose, linestyle='-.', linewidth=2, color='purple', alpha=0.7, label='Trend')

# Main AGP elements
ax1.fill_between(x, result["p5"], result["p95"], alpha=0.15, color='blue', label="5–95%")
ax1.fill_between(x, result["p25"], result["p75"], alpha=0.35, color='blue', label="IQR")
ax1.plot(x, result["median"], linewidth=2.5, color='darkblue', label="Median")
ax1.plot(x, result["mean"], linestyle="--", linewidth=1.5, color='navy', label="Mean")

# Reference lines
ax1.axhline(LOW, linestyle=":", linewidth=1, color='darkred', alpha=0.5)
ax1.axhline(HIGH, linestyle=":", linewidth=1, color='darkred', alpha=0.5)
ax1.axhline(TIGHT_HIGH, linestyle=":", linewidth=1, color='darkgreen', alpha=0.5)

# Shade nighttime hours (10pm - 6am)
ax1.axvspan(22*60, 24*60, alpha=0.05, color='gray')
ax1.axvspan(0, 6*60, alpha=0.05, color='gray', label='Night Hours')

ax1.set_xlabel("Time of Day", fontsize=12)
ax1.set_ylabel("Glucose (mg/dL)", fontsize=12)
ax1.grid(True, alpha=0.3)

# ROC axis
ax2 = ax1.twinx()
ax2.plot(x, result["roc_mean"], linestyle=":", linewidth=2,
         label="ROC (mg/dL/min)", color='orange')
ax2.set_ylabel("Rate of Change (mg/dL/min)", fontsize=12)

xticks = np.arange(0, 1441, 120)
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"{int(t//60):02d}:00" for t in xticks])

# Set y-axis limits to leave space at top for metrics box
current_ylim = ax1.get_ylim()
ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.15)  # Add 15% headroom

# Internal Metrics Box (EXACTLY AS ORIGINAL, positioned inside plot)
textstr = (
    f"TIME IN RANGE\n"
    f"TIR (70-180): {tir:.1f}%\n"
    f"TITR (70-140): {titr:.1f}%  ← Tight Target\n"
    f"TATR (140-180): {tatr:.1f}%\n"
    f"TAR >180: {tar:.1f}% (181-250: {tar_level1:.1f}%, >250: {tar_level2:.1f}%)\n"
    f"TBR <70: {tbr:.1f}% (54-69: {tbr_level1:.1f}%, <54: {tbr_level2:.1f}%)\n\n"
    f"GLUCOSE STATS\n"
    f"Mean: {mean_glucose:.1f} mg/dL\n"
    f"GMI: {gmi:.2f}%\n"
    f"CV: {cv_percent:.1f}% {'(Stable)' if cv_percent < 36 else '(Unstable)'}\n"
    f"J-Index: {j_index:.1f}\n\n"
    f"VARIABILITY\n"
    f"MAGE: {mage:.1f}\n"
    f"MODD: {modd:.1f}\n"
    f"CONGA(1h): {conga:.1f}\n\n"
    f"RISK\n"
    f"LBGI: {lbgi:.2f}\n"
    f"HBGI: {hbgi:.2f}\n"
    f"ADRR: {adrr:.1f}\n\n"
    f"AUC\n"
    f"Time-weighted avg: {time_weighted_avg:.1f} mg/dL\n"
    f"Hyperglycemia exposure: {time_in_hyperglycemia_pct:.1f}%\n"
    f"Hypoglycemia exposure: {time_in_hypoglycemia_pct:.1f}%\n\n"
    f"DATA QUALITY\n"
    f"Days: {days_of_data:.1f}\n"
    f"Readings/day: {readings_per_day:.0f}\n"
    f"Wear time: {wear_percentage:.1f}%\n"
    f"Severe hypo/week: {severe_hypo_per_week:.2f}"
)

# Position the text box with proper margins from plot edges
plt.gcf().text(0.75, 0.92, textstr, fontsize=9,
               bbox=dict(boxstyle="round", facecolor='white', alpha=0.49,
                        edgecolor='gray', linewidth=1, pad=0.8),
               verticalalignment='top',
               horizontalalignment='left',
               transform=ax1.transAxes)

# Adjust legend position to avoid overlapping with metrics box
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, 
          bbox_to_anchor=(0.02, 0.98), ncol=2)

# --- BOTTOM ROW: Raw Data Series Chart (spans all 12 columns) ---
ax3 = fig.add_subplot(gs[1, :])

# Define color mapping for glucose ranges
range_colors = {
    'Very Low': 'darkred',
    'Low': 'red',
    'Target': 'limegreen',
    'High': 'orange',
    'Very High': 'darkorange'
}

# Plot points in order (target first for better visibility of extremes)
# Plot target points first (most numerous, less critical)
target_data = df[df['glucose_range'] == 'Target']
if not target_data.empty:
    ax3.scatter(target_data['Time'], target_data['Sensor Reading(mg/dL)'], 
               c=range_colors['Target'], s=8, alpha=0.4, 
               label=f'Target (70-180): {len(target_data)} pts', edgecolors='none')

# Plot high and very high
high_data = df[df['glucose_range'] == 'High']
if not high_data.empty:
    ax3.scatter(high_data['Time'], high_data['Sensor Reading(mg/dL)'], 
               c=range_colors['High'], s=12, alpha=0.6, 
               label=f'High (181-250): {len(high_data)} pts', edgecolors='none')

very_high_data = df[df['glucose_range'] == 'Very High']
if not very_high_data.empty:
    ax3.scatter(very_high_data['Time'], very_high_data['Sensor Reading(mg/dL)'], 
               c=range_colors['Very High'], s=12, alpha=0.7, 
               label=f'Very High (>250): {len(very_high_data)} pts', edgecolors='none')

# Plot low and very low (most critical, plot last to be on top)
low_data = df[df['glucose_range'] == 'Low']
if not low_data.empty:
    ax3.scatter(low_data['Time'], low_data['Sensor Reading(mg/dL)'], 
               c=range_colors['Low'], s=15, alpha=0.8, 
               label=f'Low (54-69): {len(low_data)} pts', edgecolors='black', linewidth=0.5)

very_low_data = df[df['glucose_range'] == 'Very Low']
if not very_low_data.empty:
    ax3.scatter(very_low_data['Time'], very_low_data['Sensor Reading(mg/dL)'], 
               c=range_colors['Very Low'], s=20, alpha=1.0, 
               label=f'Very Low (<54): {len(very_low_data)} pts', edgecolors='black', linewidth=0.8)

# Add target zone backgrounds (lighter than main plot)
ax3.axhspan(TIGHT_LOW, TIGHT_HIGH, alpha=0.1, color='limegreen')
ax3.axhspan(HIGH, 600, alpha=0.07, color='orange')
ax3.axhspan(20, LOW, alpha=0.07, color='red')

# Add reference lines
ax3.axhline(LOW, linestyle=":", linewidth=1, color='darkred', alpha=0.4)
ax3.axhline(HIGH, linestyle=":", linewidth=1, color='darkred', alpha=0.4)
ax3.axhline(TIGHT_HIGH, linestyle=":", linewidth=1, color='darkgreen', alpha=0.4)

# Format x-axis to show dates nicely
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
ax3.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=max(1, days_of_data//7)))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

# Add minor ticks for hours if data span is short
if days_of_data <= 3:
    ax3.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=6))
    ax3.xaxis.set_minor_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.setp(ax3.xaxis.get_minorticklabels(), rotation=45, ha='right', fontsize=8)

ax3.set_ylabel("Glucose (mg/dL)", fontsize=11)
ax3.set_xlabel("Date", fontsize=11)
ax3.set_title("Raw Glucose Data Series (Color-coded by Range)", fontsize=12, pad=10)
ax3.grid(True, alpha=0.2)
ax3.legend(loc='upper right', ncol=3, fontsize=8, framealpha=0.9)

# Set y-axis limits for consistency
ax3.set_ylim(20, 400)

plt.suptitle("Ambulatory Glucose Profile with Time in Tight Range (TITR) and Raw Data Series", 
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("ambulatory_glucose_profile.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Print clinical interpretations and warnings (EXACTLY AS ORIGINAL)
print("\n" + "="*60)
print("CLINICAL SUMMARY")
print("="*60)
print(f"Time in Range (70-180): {tir:.1f}% - {'Target met (≥70%)' if tir >= 70 else 'Below target'}")
print(f"Time in Tight Range (70-140): {titr:.1f}% - {'Excellent' if titr >= 50 else 'Room for improvement'}")
print(f"Time Below Range: {tbr:.1f}% - {'Target met (<4%)' if tbr < 4 else 'Above target'}")
print(f"Glucose Variability (CV): {cv_percent:.1f}% - {'Stable (<36%)' if cv_percent < 36 else 'Unstable (≥36%)'}")
print("="*60)

if days_of_data < 5:
    print(f"Warning: Only {days_of_data:.1f} days of data. AGP typically requires ≥5 days for reliability.")
if readings_per_day < 24:
    print(f"Warning: Low reading frequency ({readings_per_day:.0f} readings/day). Continuous glucose monitor expected.")
if wear_percentage < 70:
    print(f"Warning: Low sensor wear time ({wear_percentage:.1f}%). Results may not be representative.")

print("\nGlucose Distribution Summary:")
print(f"  Very Low (<54 mg/dL): {very_low_pct:.1f}%")
print(f"  Low (54-69 mg/dL): {low_pct:.1f}%")
print(f"  Target (70-180 mg/dL): {target_pct:.1f}%")
print(f"  High (181-250 mg/dL): {high_pct:.1f}%")
print(f"  Very High (>250 mg/dL): {very_high_pct:.1f}%")
