import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

# --- Data Exploration ---
def render_data_exploration(df):
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(names=churn_counts.index, values=churn_counts.values, hole=0.4,
                     color_discrete_sequence=['#EF553B', '#636EFA'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Payment Methods")
        payment_counts = df['PaymentMethod'].value_counts()
        st.bar_chart(payment_counts)

# --- XAI Tabs ---
def render_xai_tabs(model, shap_values, lime_explainer, X_test_processed, y_test, class_names, feature_names):
    
    tab1, tab2 = st.tabs(["Global & Local (SHAP)", "Local (LIME)"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        st.markdown("Which features impact the model the most across ALL customers?")
        
        # SHAP Summary Plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig)
        
        st.divider()
        st.subheader("Local Explanation (Waterfall)")
        idx = st.slider("Select Customer Index for SHAP", 0, len(X_test_processed)-1, 0)
        
        fig2, ax2 = plt.subplots()
        # Explanation object for the waterfall plot
        exp = shap.Explanation(
            values=shap_values[idx], 
            base_values=shap_values.mean(), # Approximation for TreeExplainer
            data=X_test_processed[idx], 
            feature_names=feature_names
        )
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig2)

    with tab2:
        st.subheader("LIME Local Explanation")
        idx_lime = st.number_input("Select Customer Index for LIME", 0, len(X_test_processed)-1, 0)
        
        # Because we are inside a pipeline, LIME needs the raw data structure
        # Ideally, pass the predict_proba function of the pipeline
        st.write(f"Explaining prediction for Customer #{idx_lime}")
        
        # Render the HTML directly.
        exp = lime_explainer.explain_instance(
            X_test_processed[idx_lime], 
            model.named_steps['classifier'].predict_proba, 
            num_features=10
        )
        st.components.v1.html(exp.as_html(), height=800, scrolling=True)