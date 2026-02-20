# Unified Robust AGP + GMI + AUC + Full Clinical Metrics + TITR + Raw Data Series (Bottom)

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import timedelta
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# Add argument parser
parser = argparse.ArgumentParser(description='Generate Ambulatory Glucose Profile from sensor data')
parser.add_argument('input_file', help='Path to Excel file with glucose data')
parser.add_argument('--output', '-o', default='ambulatory_glucose_profile.png', 
                    help='Output PNG filename (default: ambulatory_glucose_profile.png)')
parser.add_argument('--low-threshold', type=int, default=70, 
                    help='Low glucose threshold in mg/dL (default: 70)')
parser.add_argument('--high-threshold', type=int, default=180, 
                    help='High glucose threshold in mg/dL (default: 180)')
parser.add_argument('--tight-low', type=int, default=70, 
                    help='Tight range lower limit in mg/dL (default: 70)')
parser.add_argument('--tight-high', type=int, default=140, 
                    help='Tight range upper limit in mg/dL (default: 140)')
parser.add_argument('--bin-minutes', type=int, default=5, 
                    help='Time bin size in minutes for AGP (default: 5)')
parser.add_argument('--min-samples', type=int, default=5, 
                    help='Minimum samples per bin (default: 5)')
parser.add_argument('--no-plot', action='store_true', 
                    help='Calculate metrics only, do not generate plot')
parser.add_argument('--verbose', '-v', action='store_true', 
                    help='Print detailed metrics during execution')
parser.add_argument('--export', '-e', default='', 
                    help='Export metrics to file. Use .csv or .json extension (e.g. metrics.json)')
parser.add_argument('--config', '-c', help='Configuration file with parameters')

parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

parser.add_argument('--patient-name', '-n', default='Unknown', 
                    help='Patient name for report header')
parser.add_argument('--patient-id', '-id', default='N/A', 
                    help='Patient ID for report header')
parser.add_argument('--doctor', '-d', default='', 
                    help='Doctor name for report header')
parser.add_argument('--notes', '-note', default='', 
                    help='Additional notes for report header')

args = parser.parse_args()


from datetime import datetime

def create_report_header(args):
    """Create a formatted header dictionary with patient and report information"""
    header = {
        'patient_name': args.patient_name,
        'patient_id': args.patient_id,
        'doctor': args.doctor,
        'notes': args.notes,
        'report_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'report_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': args.input_file
    }
    return header

# Create header
report_header = create_report_header(args)


# Load config file if specified
if args.config:
    import json
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            # Override args with config values (if they exist)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            if args.verbose:
                print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading config file: {e}")

# Replace hardcoded constants with argparse values
LOW = args.low_threshold
HIGH = args.high_threshold
TIGHT_LOW = args.tight_low
TIGHT_HIGH = args.tight_high
BIN_MINUTES = args.bin_minutes
MIN_SAMPLES_PER_BIN = args.min_samples
ROC_CLIP = 10  # Keep this hardcoded as it's a physiological constant

if args.verbose:
    print(f"Loading data from: {args.input_file}")
    print(f"Glucose thresholds: Low={LOW}, High={HIGH}, Tight={TIGHT_LOW}-{TIGHT_HIGH}")


# --------------------------------------------------
# 1) Read Data
# --------------------------------------------------
try:
    df = pd.read_excel(args.input_file)
    if args.verbose:
        print(f"Successfully loaded {len(df)} rows from {args.input_file}")
except Exception as e:
    print(f"Error loading file {args.input_file}: {e}")
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
median_glucose = glucose.median()
std_glucose = glucose.std(ddof=0)
mode_glucose = glucose.mode()
skew_glucose = glucose.skew()
cv_percent = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else np.nan

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
day_mask = (df['Time'].dt.hour >= 6) & (df['Time'].dt.hour < 22)
night_mask = ~day_mask
if day_mask.any() and glucose[day_mask].mean() > 0:
    day_cv = glucose[day_mask].std() / glucose[day_mask].mean() * 100
else:
    day_cv = np.nan
if night_mask.any() and glucose[night_mask].mean() > 0:
    night_cv = glucose[night_mask].std() / glucose[night_mask].mean() * 100
else:
    night_cv = np.nan

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


def compute_gri(lbgi, hbgi):
    # GRI = (3.0 * LBGI) + (1.6 * HBGI)
    return (3.0 * lbgi) + (1.6 * hbgi)


gri = compute_gri(lbgi, hbgi)
gri_txt = 'Low Risk' if gri < 20 else 'Moderate' if gri < 40 else 'High Risk' 

# --------------------------------------------------
# 9) TIR / TAR / TBR / TITR with levels
# --------------------------------------------------
total = len(glucose)

# Calculate percentages for each glucose range
very_low_pct = (glucose < 54).sum() / total * 100
low_pct = ((glucose >= 54) & (glucose < LOW)).sum() / total * 100
tight_target_pct = ((glucose >= LOW) & (glucose <= TIGHT_HIGH)).sum() / total * 100  # 70-140
above_tight_pct = ((glucose > TIGHT_HIGH) & (glucose <= HIGH)).sum() / total * 100   # 141-180
high_pct = ((glucose > HIGH) & (glucose <= 250)).sum() / total * 100                 # 181-250
very_high_pct = (glucose > 250).sum() / total * 100

# Time in Range (Standard: 70-180) — correctly includes 70-140 AND 141-180
tir = ((glucose >= LOW) & (glucose <= HIGH)).sum() / total * 100

# Time in Tight Range (70-140)
titr = tight_target_pct


# Time Above Range (with levels)
tar_level1 = high_pct        # 181-250
tar_level2 = very_high_pct   # >250
tar = tar_level1 + tar_level2

# Time Above Tight Range (140-180) - supplementary
tatr = above_tight_pct

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
days_of_data = (df['Time'].max() - df['Time'].min()).total_seconds() / 86400.0  # Use float days, not .days
days_of_data = max(days_of_data, 1/24)  # Minimum 1 hour to avoid division by zero
hours_of_data = (df['Time'].max() - df['Time'].min()).total_seconds() / 3600
readings_per_day = len(df) / days_of_data if days_of_data > 0 else len(df)

# Sensor wear time percentage
total_possible_readings = days_of_data * (24 * 60 / BIN_MINUTES)  # Theoretical max based on bin size
wear_percentage = (len(df) / total_possible_readings) * 100 if total_possible_readings > 0 else np.nan

# Severe hypoglycemia events
severe_hypo_count = (glucose < 40).sum()
severe_hypo_per_week = (severe_hypo_count / days_of_data) * 7 if days_of_data > 0 else np.nan

# --------------------------------------------------
# 12) Overall Glucose Trend
# --------------------------------------------------
time_days = (df['Time'] - df['Time'].min()).dt.total_seconds() / 86400.0
if time_days.max() > 0 and len(time_days) >= 2:
    trend_slope, _ = np.polyfit(time_days, df['Sensor Reading(mg/dL)'], 1)
else:
    trend_slope = 0.0

# Thresholds: ±1 mg/dL/day represents clinically meaningful directional change
# while minimizing noise from short-term fluctuations

if trend_slope > 1:
    trend_direction = 'UP'
    trend_arrow = '↑'
    trend_color = 'orangered'
elif trend_slope < -1:
    trend_direction = 'DOWN'
    trend_arrow = '↓'
    trend_color = 'mediumseagreen'
else:
    trend_direction = 'STABLE'
    trend_arrow = '→'
    trend_color = 'steelblue'

# --------------------------------------------------
# 13) Circadian Binning
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
# 14) Create color-coded data series for raw readings
# --------------------------------------------------
# Create a categorical column for glucose ranges (6 bands matching corrected metrics)
df['glucose_range'] = pd.cut(df['Sensor Reading(mg/dL)'], 
                              bins=[0, 54, 70, 140, 180, 250, 1000],
                              labels=['Very Low', 'Low', 'Tight Target', 'Above Tight', 'High', 'Very High'])

# --------------------------------------------------
# 15) Plot AGP with Internal Metrics Box, TITR Band, and Distribution Stacked Bar (ORIGINAL LAYOUT)
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

# Create stacked bar chart of glucose distribution (6 segments matching corrected metrics)
percentages = [very_low_pct, low_pct, tight_target_pct, above_tight_pct, high_pct, very_high_pct]
colors = ['darkred', 'red', 'limegreen', 'yellowgreen', 'orange', 'darkorange']
labels = ['Very Low (<54)', 'Low (54-69)', 'Tight Target (70-140)',
          'Above Tight (141-180)', 'High (181-250)', 'Very High (>250)']

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
#ax1.plot(x, result["mode"], linestyle="--", linewidth=1.5, color='steelblue', label="Mode")


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
    f"Mean: {mean_glucose:.1f} mg/dL, median: {median_glucose:.1f} mg/dL\n" 
    f"Std: {std_glucose:.1f} mg/dL\n"
    f"skew: {skew_glucose:.1f} ({skew_interpretation})\n"
    f"GMI: {gmi:.2f}%\n"
    f"CV: {cv_percent:.1f}% {'(Stable)' if cv_percent < 36 else '(Unstable)'}\n"
    f"CV: Day: {day_cv:.1f}%, Night: {night_cv:.1f}%\n"
    f"J-Index: {j_index:.1f}\n\n"
    f"VARIABILITY\n"
    f"MAGE: {mage:.1f}\n"
    f"MODD: {modd:.1f}\n"
    f"CONGA(1h): {conga:.1f}\n\n"
    f"RISK\n"
    f"LBGI: {lbgi:.2f}\n"
    f"HBGI: {hbgi:.2f}\n"
    f"GRI: {gri:.1f} ({gri_txt})\n"
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

# Trend annotation on AGP chart (positioned between legend and metrics box)
ax1.text(0.60, 0.97, f"Overall Trend: {trend_arrow}",
         transform=ax1.transAxes, fontsize=12, fontweight='bold',
         color=trend_color, va='top', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7,
                   edgecolor=trend_color, linewidth=1.5))

# --- BOTTOM ROW: Raw Data Series Chart (spans all 12 columns) ---
ax3 = fig.add_subplot(gs[1, :])

# Define color mapping for glucose ranges (6 bands)
range_colors = {
    'Very Low': 'darkred',
    'Low': 'red',
    'Tight Target': 'limegreen',
    'Above Tight': 'yellowgreen',
    'High': 'orange',
    'Very High': 'darkorange'
}

# Plot points in order (tight target first for better visibility of extremes)
tight_target_data = df[df['glucose_range'] == 'Tight Target']
if not tight_target_data.empty:
    ax3.scatter(tight_target_data['Time'], tight_target_data['Sensor Reading(mg/dL)'],
               c=range_colors['Tight Target'], s=8, alpha=0.4,
               label=f'Tight Target (70-140): {len(tight_target_data)} pts', edgecolors='none')

above_tight_data = df[df['glucose_range'] == 'Above Tight']
if not above_tight_data.empty:
    ax3.scatter(above_tight_data['Time'], above_tight_data['Sensor Reading(mg/dL)'],
               c=range_colors['Above Tight'], s=10, alpha=0.5,
               label=f'Above Tight (141-180): {len(above_tight_data)} pts', edgecolors='none')

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

# Trend annotation on raw data series chart (upper-left, clear of upper-right legend)
ax3.text(0.01, 0.97, f"Overall Trend: {trend_arrow}",
         transform=ax3.transAxes, fontsize=11, fontweight='bold',
         color=trend_color, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7,
                   edgecolor=trend_color, linewidth=1.5))


# Add discrete header with patient information at the top of the figure
header_text = f"Patient: {report_header['patient_name']} | ID: {report_header['patient_id']}"
if report_header['doctor']:
    header_text += f" | Dr: {report_header['doctor']}"
header_text += f" | Report Date: {report_header['report_date']}"

plt.figtext(0.5, 0.96, header_text, 
            ha="center", fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))

# Add data source and notes as smaller text below header
if report_header['notes']:
    plt.figtext(0.5, 0.94, f"Notes: {report_header['notes']} | Source: {report_header['data_source']}", 
                ha="center", fontsize=8, style='italic', color='gray')
else:
    plt.figtext(0.5, 0.94, f"Source: {report_header['data_source']}", 
                ha="center", fontsize=8, style='italic', color='gray')

plt.suptitle("Ambulatory Glucose Profile with Time in Tight Range (TITR) and Raw Data Series", 
             fontsize=14, y=0.92)
plt.tight_layout()


metadata = {"Description": "Ambulatory Glucose Profile generated by AGP tool",
            "Source":"https://github.com/daedalus/agp",
            "Copyright":"Copyright 2026 Darío Clavijo",
            "License":"MIT License"
} 

plt.figtext(0.5, 0.02, f"{metadata['Source']}\n{metadata['Copyright']}\n{metadata['License']}", 
            ha="center", fontsize=9, color='gray', 
            style='italic', alpha=0.7)

if not args.no_plot:
    plt.savefig(args.output, dpi=300, bbox_inches='tight', metadata=metadata)
    if args.verbose:
        print(f"Plot saved to: {args.output}")
    plt.show()
    plt.close()
else:
    if args.verbose:
        print("Plot generation skipped (--no-plot flag used)")
    plt.close()

print("\n" + "="*60)
print(f"PATIENT: {report_header['patient_name']} (ID: {report_header['patient_id']})")
print(f"REPORT DATE: {report_header['report_date']}")

# Print clinical interpretations and warnings (EXACTLY AS ORIGINAL)
print("\n" + "="*60)
print("CLINICAL SUMMARY")
print("="*60)
print(f"Time in Range (70-180): {tir:.1f}% - {'Target met (≥70%)' if tir >= 70 else 'Below target'}")
print(f"Time in Tight Range (70-140): {titr:.1f}% - {'Excellent' if titr >= 50 else 'Room for improvement'}")
print(f"Time Below Range: {tbr:.1f}% - {'Target met (<4%)' if tbr < 4 else 'Above target'}")
print(f"Glucose Variability (CV): {cv_percent:.1f}% - {'Stable (<36%)' if cv_percent < 36 else 'Unstable (≥36%)'}")
print(f"Overall Trend: {trend_arrow} {trend_direction} (slope: {trend_slope:.1f} mg/dL/day)")
print(f"Glycemia Risk Index (GRI): {gri:.1f} - {gri_txt}\n")
print("-"*60)

if days_of_data < 5:
    print(f"Warning: Only {days_of_data:.1f} days of data. AGP typically requires ≥5 days for reliability.")
if readings_per_day < 24:
    print(f"Warning: Low reading frequency ({readings_per_day:.0f} readings/day). Continuous glucose monitor expected.")
if wear_percentage < 70:
    print(f"Warning: Low sensor wear time ({wear_percentage:.1f}%). Results may not be representative.")

print(f"Distribution Shape: skew = {skew_glucose:.2f} - {skew_interpretation}")
if 1.0 < skew_glucose < 1.5:
    print(f"  → Note: In this 'gray zone', the mean ({mean_glucose:.1f}) exceeds the median ({median_glucose:.1f})")
    print(f"  → The median better represents typical glucose exposure")
print("-"*60)
print("\nGlucose Distribution Summary:")
print(f"  Very Low (<54 mg/dL): {very_low_pct:.1f}%")
print(f"  Low (54-69 mg/dL): {low_pct:.1f}%")
print(f"  Tight Target (70-140 mg/dL): {tight_target_pct:.1f}%")
print(f"  Above Tight (141-180 mg/dL): {above_tight_pct:.1f}%")
print(f"  High (181-250 mg/dL): {high_pct:.1f}%")
print(f"  Very High (>250 mg/dL): {very_high_pct:.1f}%")

# --------------------------------------------------
# Export metrics if requested
# --------------------------------------------------
if args.export:
    metrics = {
        'patient_name': report_header['patient_name'],
        'patient_id': report_header['patient_id'],
        'doctor': report_header['doctor'],
        'report_date': report_header['report_date'],
        'days_of_data': round(days_of_data, 2),
        'readings_per_day': round(readings_per_day, 1),
        'wear_percentage': round(wear_percentage, 1),
        'mean_glucose': round(mean_glucose, 1),
        'median_glucose': round(median_glucose, 1),
        'std_glucose': round(std_glucose, 1),
        'cv_percent': round(cv_percent, 1),
        'day_cv': round(day_cv, 1) if not np.isnan(day_cv) else None,
        'night_cv': round(night_cv, 1) if not np.isnan(night_cv) else None,
        'gmi': round(gmi, 2),
        'skew': round(skew_glucose, 2),
        'j_index': round(j_index, 1),
        'tir': round(tir, 1),
        'titr': round(titr, 1),
        'tatr': round(tatr, 1),
        'tar': round(tar, 1),
        'tbr': round(tbr, 1),
        'very_low_pct': round(very_low_pct, 1),
        'low_pct': round(low_pct, 1),
        'high_pct': round(high_pct, 1),
        'very_high_pct': round(very_high_pct, 1),
        'mage': round(mage, 1) if not np.isnan(mage) else None,
        'modd': round(modd, 1) if not np.isnan(modd) else None,
        'conga': round(conga, 1) if not np.isnan(conga) else None,
        'lbgi': round(lbgi, 2),
        'hbgi': round(hbgi, 2),
        'gri': round(gri, 1),
        'adrr': round(adrr, 1) if not np.isnan(adrr) else None,
        'auc_time_weighted_avg': round(time_weighted_avg, 1),
        'auc_hyperglycemia_pct': round(time_in_hyperglycemia_pct, 1),
        'auc_hypoglycemia_pct': round(time_in_hypoglycemia_pct, 1),
        'trend_direction': trend_direction,
        'trend_slope_mg_per_day': round(trend_slope, 2),
        'severe_hypo_per_week': round(severe_hypo_per_week, 2) if not np.isnan(severe_hypo_per_week) else None,
    }
    try:
        if args.export.endswith('.json'):
            import json
            with open(args.export, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif args.export.endswith('.csv'):
            pd.DataFrame([metrics]).to_csv(args.export, index=False)
        else:
            print(f"Warning: Unrecognized export format for '{args.export}'. Use .json or .csv")
            args.export = ''
        if args.export:
            print(f"\nMetrics exported to: {args.export}")
    except Exception as e:
        print(f"Error exporting metrics: {e}")

