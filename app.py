import streamlit as st
import pandas as pd
import time
import os

# Import modules from our src package
from src import ml_logic, ui_components, monitoring, mlops_engine

st.set_page_config(page_title="MLOps Production System", page_icon="🏭", layout="wide")

# --- Session State ---
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None
if 'drift_report' not in st.session_state:
    st.session_state.drift_report = None

# --- Sidebar: Simulation ---
st.sidebar.title("🛠️ MLOps Console")
DATA_PATH = "data/reference_data.csv"

# Load Data
if not os.path.exists(DATA_PATH):
    st.error(f"File not found: {DATA_PATH}. Please move 'customerchurn.csv' to 'data/reference_data.csv'")
    st.stop()

df_baseline = ml_logic.load_data(DATA_PATH)

st.sidebar.subheader("1. Data Simulation")
sim_type = st.sidebar.selectbox("Simulate Traffic", ["Normal", "Drifted (High Churn)", "Drifted (Income Change)"])

if st.sidebar.button("Generate Traffic Batch"):
    # Generate synthetic data based on baseline
    batch = df_baseline.sample(300, replace=True).copy()
    
    if sim_type == "Drifted (High Churn)":
        # Simulate drift: People with lower tenure are paying more
        batch['MonthlyCharges'] = batch['MonthlyCharges'] * 1.5
        batch['tenure'] = batch['tenure'] * 0.5
        st.sidebar.warning("Generated DRIFTED batch")
    else:
        st.sidebar.success("Generated NORMAL batch")
    
    st.session_state.sim_data = batch

# --- Main Interface ---
tab1, tab2, tab3 = st.tabs(["Production Model", "Drift Monitor", "MLOps Pipeline"])

# --- TAB 1: Prediction & XAI ---
with tab1:
    st.title("Customer Churn Prediction")
    
    # Load Production Model
    registry = mlops_engine.ModelRegistry()
    model = registry.load_latest_model()
    
    if model is None:
        st.info("No model in registry. Training initial V1...")
        v, acc = mlops_engine.retrain_pipeline(df_baseline, "Initial V1")
        # FIX: Replaced experimental_rerun with rerun
        st.rerun()
        
    # Get latest version from history
    history = registry.get_history()
    current_version = history[-1]['version'] if history else 0
    st.success(f"Serving Model Version: v{current_version}")
    
    # Use existing UI components for data/XAI
    st.markdown("### Model Performance (Reference)")
    ui_components.render_data_exploration(df_baseline.sample(100))

# --- TAB 2: Monitoring (Updated for Plotly) ---
with tab2:
    st.header("Drift Detection Service")
    
    if st.session_state.sim_data is None:
        st.info("Please generate traffic in the sidebar first.")
    else:
        st.write("Analyzing current batch vs baseline using Kolmogorov-Smirnov Test...")
        detector = monitoring.DriftDetector(df_baseline)
        
        if st.button("Run Drift Check"):
            drifted, share, details = detector.run_check(st.session_state.sim_data)
            
            # Save chart to state
            st.session_state.drift_report = detector.get_plot(details)
            
            col1, col2 = st.columns(2)
            col1.metric("Drift Detected?", "YES" if drifted else "NO", delta_color="inverse" if drifted else "normal")
            col2.metric("Drifted Features %", f"{share*100:.1f}%")
            
            if drifted:
                st.error("Critical Data Drift Detected! Retraining recommended.")
            else:
                st.success("Data distribution is stable.")

        # Display Plotly Chart
        if st.session_state.drift_report:
            st.plotly_chart(st.session_state.drift_report, use_container_width=True)
            
            with st.expander("See Statistical Details"):
                st.write("Drift Threshold: p-value < 0.05")

# --- TAB 3: Retraining ---
with tab3:
    st.header("Automated Retraining Engine")
    
    # Registry History
    st.subheader("Model Lineage")
    history = registry.get_history()
    st.dataframe(pd.DataFrame(history).sort_values("version", ascending=False), use_container_width=True)
    
    st.divider()
    
    if st.button("Trigger Automated Retraining"):
        if st.session_state.sim_data is None:
            st.error("No new data available to retrain on.")
        else:
            with st.status("MLOps Pipeline Running..."):
                st.write("Ingesting new data batch...")
                # Combine baseline + new data
                new_dataset = pd.concat([df_baseline, st.session_state.sim_data])
                time.sleep(1)
                
                st.write("Retraining XGBoost Classifier...")
                new_ver, new_acc = mlops_engine.retrain_pipeline(new_dataset, f"Retrained on +300 samples")
                
                st.write(f"Promoting V{new_ver} to Production...")
                time.sleep(1)
                
            st.balloons()
            st.success(f"Pipeline Success! Model V{new_ver} is now live (Accuracy: {new_acc:.2%})")
            time.sleep(2)
            st.rerun()