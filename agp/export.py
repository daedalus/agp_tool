import numpy as np
import pandas as pd


def export_metrics(metrics, export_path, report_header=None, verbose=False):
    """Export metrics dict to .json or .csv file."""
    export_data = {}

    if report_header is not None:
        export_data['patient_name'] = report_header['patient_name']
        export_data['patient_id'] = report_header['patient_id']
        export_data['doctor'] = report_header['doctor']
        export_data['report_date'] = report_header['report_date']

    day_cv = metrics['day_cv']
    night_cv = metrics['night_cv']
    mage = metrics['mage']
    modd = metrics['modd']
    conga = metrics['conga']
    adrr = metrics['adrr']
    severe_hypo_per_week = metrics['severe_hypo_per_week']

    export_data.update({
        'days_of_data': round(metrics['days_of_data'], 2),
        'readings_per_day': round(metrics['readings_per_day'], 1),
        'wear_percentage': round(metrics['wear_percentage'], 1),
        'mean_glucose': round(metrics['mean_glucose'], 1),
        'median_glucose': round(metrics['median_glucose'], 1),
        'std_glucose': round(metrics['std_glucose'], 1),
        'cv_percent': round(metrics['cv_percent'], 1),
        'day_cv': round(day_cv, 1) if not np.isnan(day_cv) else None,
        'night_cv': round(night_cv, 1) if not np.isnan(night_cv) else None,
        'gmi': round(metrics['gmi'], 2),
        'skew': round(metrics['skew_glucose'], 2),
        'j_index': round(metrics['j_index'], 1),
        'tir': round(metrics['tir'], 1),
        'titr': round(metrics['titr'], 1),
        'tatr': round(metrics['tatr'], 1),
        'tar': round(metrics['tar'], 1),
        'tbr': round(metrics['tbr'], 1),
        'very_low_pct': round(metrics['very_low_pct'], 1),
        'low_pct': round(metrics['low_pct'], 1),
        'high_pct': round(metrics['high_pct'], 1),
        'very_high_pct': round(metrics['very_high_pct'], 1),
        'mage': round(mage, 1) if not np.isnan(mage) else None,
        'modd': round(modd, 1) if not np.isnan(modd) else None,
        'conga': round(conga, 1) if not np.isnan(conga) else None,
        'lbgi': round(metrics['lbgi'], 2),
        'hbgi': round(metrics['hbgi'], 2),
        'gri': round(metrics['gri'], 1),
        'adrr': round(adrr, 1) if not np.isnan(adrr) else None,
        'auc_time_weighted_avg': round(metrics['time_weighted_avg'], 1),
        'auc_hyperglycemia_pct': round(metrics['exposure_severity_to_hyperglycemia_pct'], 1),
        'auc_hypoglycemia_pct': round(metrics['exposure_severity_to_hypoglycemia_pct'], 1),
        'trend_direction': metrics['trend_direction'],
        'trend_slope_mg_per_day': round(metrics['trend_slope'], 2),
        'severe_hypo_per_week': round(severe_hypo_per_week, 2) if not np.isnan(severe_hypo_per_week) else None,
    })

    try:
        if export_path.endswith('.json'):
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif export_path.endswith('.csv'):
            pd.DataFrame([export_data]).to_csv(export_path, index=False)
        elif export_path.endswith('.xlsx'):
            pd.DataFrame([export_data]).to_excel(export_path, index=False)
        else:
            print(f"Warning: Unrecognized export format for '{export_path}'. Use .json, .csv, or .xlsx")
            return
        print(f"\nMetrics exported to: {export_path}")
    except Exception as e:
        print(f"Error exporting metrics: {e}")
