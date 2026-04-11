# KS-test drift detection across all 15+ numeric and encoded features.

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import plotly.graph_objects as go
from src.ml_logic import preprocess, FEATURE_COLS

DRIFT_THRESHOLD = 0.05


class DriftDetector:
    def __init__(self, reference_df):
        self.reference = preprocess(reference_df)[FEATURE_COLS]

    def run_check(self, production_df):
        production = preprocess(production_df)[FEATURE_COLS]
        results = []
        for col in FEATURE_COLS:
            stat, p_value = ks_2samp(self.reference[col], production[col])
            results.append({
                "feature": col,
                "ks_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "drifted": p_value < DRIFT_THRESHOLD
            })
        details = pd.DataFrame(results)
        drifted_count = details['drifted'].sum()
        drift_share = drifted_count / len(FEATURE_COLS)
        overall_drift = drift_share > 0.2
        return overall_drift, drift_share, details

    def get_plot(self, details):
        colors = ['#ef4444' if d else '#22c55e' for d in details['drifted']]
        fig = go.Figure(go.Bar(
            x=details['feature'],
            y=details['p_value'],
            marker_color=colors,
            text=details['ks_stat'],
            textposition='outside'
        ))
        fig.add_hline(y=DRIFT_THRESHOLD, line_dash="dash", line_color="orange",
                      annotation_text="Drift threshold (p=0.05)")
        fig.update_layout(
            title="KS Test p-values per Feature",
            xaxis_tickangle=-45,
            yaxis_title="p-value",
            height=450
        )
        return fig