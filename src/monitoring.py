import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ks_2samp, chi2_contingency

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.p_value_threshold = 0.05
        
        # Identify columns
        self.num_features = reference_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_features = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target if present
        if 'Churn' in self.num_features: self.num_features.remove('Churn')
        if 'Churn' in self.cat_features: self.cat_features.remove('Churn')

    def run_check(self, current_data: pd.DataFrame):
        """
        Performs statistical tests to detect drift.
        - Numerical: Kolmogorov-Smirnov (KS) Test
        - Categorical: Chi-Square Test
        """
        drifted_features = {}
        
        # 1. Check Numerical Features (KS Test)
        for feature in self.num_features:
            ref_data = self.reference_data[feature].dropna()
            cur_data = current_data[feature].dropna()
            
            # KS Test checks if two samples are drawn from the same distribution
            statistic, p_value = ks_2samp(ref_data, cur_data)
            
            if p_value < self.p_value_threshold:
                drifted_features[feature] = {'test': 'KS-Test', 'p_value': p_value, 'drift': True}
            else:
                drifted_features[feature] = {'test': 'KS-Test', 'p_value': p_value, 'drift': False}

    
        # 2. Check Categorical Features (Chi-Square)
        # Note: We align categories to ensure arrays match for contingency table
        for feature in self.cat_features:
            # Get value counts as probabilities
            ref_dist = self.reference_data[feature].value_counts(normalize=True)
            cur_dist = current_data[feature].value_counts(normalize=True)
            
            # Align indices
            all_cats = list(set(ref_dist.index) | set(cur_dist.index))
            ref_freq = [ref_dist.get(cat, 0) for cat in all_cats]
            cur_freq = [cur_dist.get(cat, 0) for cat in all_cats]
            
            # Simple Chi-Square on frequencies (scaled to 100 for simplicity)
            # (Strictly this is a simplified heuristic for batch monitoring)
            stat, p_value = ks_2samp(ref_freq, cur_freq) 
            
            if p_value < self.p_value_threshold:
                drifted_features[feature] = {'test': 'Distribution Check', 'p_value': p_value, 'drift': True}
            else:
                drifted_features[feature] = {'test': 'Distribution Check', 'p_value': p_value, 'drift': False}

        # 3. Calculate Global Metrics
        total_features = len(drifted_features)
        drift_count = sum(1 for v in drifted_features.values() if v['drift'])
        drift_share = drift_count / total_features if total_features > 0 else 0
        drift_detected = drift_share > 0.5 # Threshold: if >50% of features drift
        
        return drift_detected, drift_share, drifted_features

    def get_plot(self, drift_results):
        """
        Generates a Plotly heatmap of p-values to visualize drift.
        """
        features = list(drift_results.keys())
        p_values = [v['p_value'] for v in drift_results.values()]
        is_drift = [v['drift'] for v in drift_results.values()]
        colors = ['red' if d else 'green' for d in is_drift]
        
        fig = go.Figure(go.Bar(
            x=p_values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f"p={p:.4f}" for p in p_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Feature Drift Analysis (P-Values)",
            xaxis_title="P-Value (Lower = More Drift)",
            yaxis_title="Features",
            shapes=[dict(
                type="line", x0=0.05, x1=0.05, y0=-1, y1=len(features),
                line=dict(color="black", dash="dash")
            )],
            height=600
        )
        return fig