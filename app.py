# Main Streamlit dashboard: Live Metrics, Drift Monitor, MLOps Pipeline, Explainability.

import streamlit as st
import pandas as pd
import time
import os

from src import ml_logic, monitoring, mlops_engine, explainability, prediction_logger

st.set_page_config(page_title="Churn MLOps System", layout="wide")

DATA_PATH = "data/reference_data.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}. Download Telco Customer Churn from Kaggle.")
    st.stop()

if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None
if 'drift_report' not in st.session_state:
    st.session_state.drift_report = None
if 'drift_details' not in st.session_state:
    st.session_state.drift_details = None

df_baseline = ml_logic.load_data(DATA_PATH)

registry = mlops_engine.ModelRegistry()
model = registry.load_latest_model()

if model is None:
    st.info("No model found. Training initial model...")
    with st.spinner("Training V1..."):
        ver, metrics = mlops_engine.retrain_pipeline(df_baseline, note="Initial V1")
    model = registry.load_latest_model()
    st.rerun()

# Sidebar
st.sidebar.title("MLOps Console")
st.sidebar.subheader("Simulate Traffic")
sim_type = st.sidebar.selectbox("Traffic Type", [
    "Normal", "Drifted (High Churn)", "Drifted (Income Change)"
])

if st.sidebar.button("Generate Traffic Batch"):
    batch = df_baseline.sample(500, replace=True).copy()
    if sim_type == "Drifted (High Churn)":
        batch['MonthlyCharges'] = batch['MonthlyCharges'] * 1.6
        batch['tenure'] = batch['tenure'] * 0.4
        batch['Contract'] = 'Month-to-month'
        batch['TechSupport'] = 'No'
        batch['OnlineSecurity'] = 'No'
        st.sidebar.warning("Drifted batch generated")
    elif sim_type == "Drifted (Income Change)":
        batch['TotalCharges'] = batch['TotalCharges'] * 1.8
        batch['MonthlyCharges'] = batch['MonthlyCharges'] * 1.3
        st.sidebar.warning("Drifted batch generated")
    else:
        st.sidebar.success("Normal batch generated")

    history = registry.get_history()
    threshold = history[-1].get('threshold', 0.5) if history else 0.5
    df_proc = ml_logic.preprocess(batch)
    X = df_proc[ml_logic.FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    prediction_logger.log_predictions(batch, probs, threshold=threshold)
    st.session_state.sim_data = batch

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Live Metrics", "Drift Monitor", "MLOps Pipeline", "Explainability"])

# TAB 1: Live Metrics
with tab1:
    st.title("Live Prediction Metrics")
    history = registry.get_history()
    current = history[-1] if history else {}

    summary = prediction_logger.get_summary_metrics()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Predictions", f"{summary['total']:,}")
    col2.metric("Weekly Predictions", f"{summary['weekly']:,}")
    col3.metric("Churn Rate", f"{summary['churn_rate']}%")
    col4.metric("Avg Churn Probability", f"{summary['avg_probability']}")
    col5.metric("Model Version", f"v{current.get('version', 'N/A')}")

    st.subheader("Current Model Performance")
    perf_cols = st.columns(3)
    perf_cols[0].metric("F1 Score", current.get('f1_score', 'N/A'))
    perf_cols[1].metric("AUC-ROC", current.get('auc_roc', 'N/A'))
    perf_cols[2].metric("Accuracy", current.get('accuracy', 'N/A'))

    log = prediction_logger.get_log()
    if not log.empty:
        st.subheader("Churn Prediction Over Time")
        log['timestamp'] = pd.to_datetime(log['timestamp'])
        trend = log.set_index('timestamp')['prediction'].resample('h').mean().reset_index()
        trend.columns = ['time', 'churn_rate']
        st.line_chart(trend.set_index('time'))

# TAB 2: Drift Monitor
with tab2:
    st.header("Drift Detection — KS Test Across All Features")
    if st.session_state.sim_data is None:
        st.info("Generate a traffic batch from the sidebar first.")
    else:
        detector = monitoring.DriftDetector(df_baseline)
        if st.button("Run Drift Check"):
            drifted, share, details = detector.run_check(st.session_state.sim_data)
            st.session_state.drift_report = detector.get_plot(details)
            st.session_state.drift_details = details

            col1, col2, col3 = st.columns(3)
            col1.metric("Drift Detected", "YES" if drifted else "NO")
            col2.metric("Features Drifted", f"{details['drifted'].sum()} / {len(details)}")
            col3.metric("Drift Share", f"{share * 100:.1f}%")

            if drifted:
                st.error("Significant drift detected. Retraining recommended.")
            else:
                st.success("Distribution is stable.")

        if st.session_state.drift_report:
            st.plotly_chart(st.session_state.drift_report, width='stretch')

        if st.session_state.drift_details is not None:
            with st.expander("Per-Feature KS Statistics"):
                st.dataframe(st.session_state.drift_details, width='stretch')

# TAB 3: MLOps Pipeline
with tab3:
    st.header("Retraining Pipeline")
    history = registry.get_history()

    if history:
        df_hist = pd.DataFrame(history).drop(columns=['model_path'], errors='ignore')
        if 'pre_retrain_f1' in df_hist.columns and df_hist['pre_retrain_f1'].notna().any():
            df_hist['f1_improvement'] = (
                df_hist['f1_score'] - df_hist['pre_retrain_f1']
            ).round(4)
        st.subheader("Model Lineage")
        st.dataframe(df_hist.sort_values('version', ascending=False), width='stretch')

    st.divider()
    if st.button("Trigger Retraining"):
        if st.session_state.sim_data is None:
            st.error("No new data batch available. Generate traffic first.")
        else:
            with st.status("Running retraining pipeline..."):
                st.write("Evaluating current model on drifted batch...")
                combined = pd.concat([df_baseline, st.session_state.sim_data])
                time.sleep(1)
                st.write("Retraining XGBoost on combined dataset...")
                new_ver, new_metrics = mlops_engine.retrain_pipeline(
                    combined,
                    note=f"Retrained on +{len(st.session_state.sim_data)} samples",
                    drifted_batch=st.session_state.sim_data
                )
                st.write(f"Promoting v{new_ver} to production...")
                time.sleep(1)
            st.success(
                f"v{new_ver} live — F1: {new_metrics['f1_score']}, "
                f"AUC-ROC: {new_metrics['auc_roc']}, Accuracy: {new_metrics['accuracy']}"
            )
            time.sleep(1)
            st.rerun()

# TAB 4: Explainability
with tab4:
    st.header("Model Explainability")
    sample = df_baseline.sample(min(1000, len(df_baseline)), random_state=42)

    xai_tab1, xai_tab2 = st.tabs(["SHAP (Global)", "LIME (Instance)"])

    with xai_tab1:
        st.subheader("SHAP Feature Importance — Global Summary")
        st.caption(f"Computed on {min(1000, len(df_baseline))} predictions from the test set.")
        if st.button("Generate SHAP Summary"):
            with st.spinner("Computing SHAP values..."):
                n = explainability.plot_shap_summary(model, sample)
            st.caption(f"SHAP values computed across {n} predictions.")

        st.subheader("SHAP Waterfall — Single Prediction")
        idx = st.slider("Select prediction index", 0, len(sample) - 1, 0)
        if st.button("Show Waterfall"):
            explainability.plot_shap_waterfall(model, sample, idx=idx)

    with xai_tab2:
        st.subheader("LIME — Local Instance Explanation")
        lime_idx = st.slider("Select instance index", 0, min(99, len(sample) - 1), 0)
        if st.button("Explain with LIME"):
            with st.spinner("Running LIME..."):
                explainability.plot_lime_explanation(model, sample, idx=lime_idx)