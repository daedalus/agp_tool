Ambulatory glucose profile analysis tool

## DISCLAIMER ##

This tool is for research and educational purposes only. It is NOT a medical device and has NOT been validated for clinical diagnosis or treatment decisions.

Do not adjust medications, change diet, or make health decisions based solely on this output. Exercise caution and always consult a qualified healthcare professional for interpretation of glucose data and any treatment adjustments.

### Overview ###

This tool generates a comprehensive Ambulatory Glucose Profile (AGP) with extended clinical metrics, including Time in Tight Range (TITR) metric. It processes continuous glucose monitoring (CGM) data and produces both a visual AGP plot and detailed statistical analysis.

### Features ###

#### General  ####

- Full AGP visualization with percentile curves (5-95%, IQR)
- Rate of Change (ROC) profile
- Circadian binning for time-of-day patterns

#### Time in Range metrics with level breakdowns: ####

- TIR (70-180 mg/dL)
- TITR (70-140 mg/dL) - Tight target
- TAR with Level 1 (181-250) and Level 2 (>250)
- TBR with Level 1 (54-69) and Level 2 (<54)

#### Advanced variability metrics: ####

- MAGE (Mean Amplitude of Glycemic Excursions)
- MODD (Mean of Daily Differences)
- CONGA (Continuous Overall Net Glycemic Action)

#### Risk indices: ####

- LBGI (Low Blood Glucose Index)
- HBGI (High Blood Glucose Index)
- ADRR (Average Daily Risk Range)

### AUC analysis: ####

- total, above range, below range

### Data quality assessment: ####

- wear time, reading frequency

### Requirements ###

#### Packages ####

`pip install -r requirements.txt`

#### Input Data Format ####

- Excel file with two columns:
- Time: datetime (e.g., "2024-01-01 08:00:00")
- Sensor Reading(mg/dL): numeric glucose values

### Usage ###

bash

`python get_sensor.py your_glucose_data.xlsx`


## Example ##

![Example](files/ambulatory_glucose_profile.png)

### Metrics Explained ###

#### Core Metrics ####

Metric	Description	Clinical Target

- TIR	Time in Range (70-180 mg/dL)	≥70%
- TITR	Time in Tight Range (70-140 mg/dL)	≥50%
- TBR	Time Below Range (<70 mg/dL)	<4%
- CV	Coefficient of Variation	<36%
- GMI	Glucose Management Indicator	below 7%
- J-Index	Combined mean + variability	n/a

#### Variability Metrics ####

Metric	Description

- MAGE	Mean amplitude of glycemic excursions
- MODD	Day-to-day glucose variability
- CONGA	Intra-day glycemic variability (1h lag)

### Risk Metrics ####

Metric	Description

- LBGI	Low Blood Glucose Index
- HBGI	High Blood Glucose Index
- ADRR	Average Daily Risk Range

#### Data Quality ####

Metric	Description	Warning Threshold

- Readings/day	Average daily readings	<24 readings/day
- Wear time	% of possible readings	<70%
- Severe hypo/week	Events <40 mg/dL per week	n/a

### Configuration ###

Key parameters at script top:

- LOW = 70           # Lower bound standard range
- HIGH = 180         # Upper bound standard range
- TIGHT_LOW = 70     # Lower bound tight range
- TIGHT_HIGH = 140   # Upper bound tight range
- BIN_MINUTES = 5    # Time bin size for AGP
- ROC_CLIP = 10      # Rate of change physiological limit

### Limitations & Warnings ###

- Minimum data: AGP typically requires ≥5 days for reliability
- Data gaps: Long gaps (>2 hours) may affect MODD and ADRR calculations
- Sensor accuracy: Assumes CGM-grade data; fingerstick data may have limitations
- MAGE calculation: Uses smoothed data; may differ from manual calculation

### Interpretation Tips ###

- TITR >50% suggests acceptable glycemic control
- TBR >4% indicates need for hypoglycemia prevention
- CV >36% suggests unstable glucose, consider variability-reducing therapy
- Nighttime patterns: Shaded area helps identify nocturnal hypoglycemia; however, a sharp "V-shaped" drop with a rapid recovery may indicate sensor compression artifact rather than true hypoglycemia
- ROC spikes indicate rapid changes; correlate with meals/exercise
- GMI < 7.0%: Suggests acceptable glycemic control (Note: GMI estimates A1c from CGM data, but may differ from lab A1c).

### License ###

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
