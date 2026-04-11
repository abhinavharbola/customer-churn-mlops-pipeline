# SHAP (global + local) and LIME explanations for model predictions.

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit as st
from src.ml_logic import preprocess, FEATURE_COLS


def get_shap_explainer(model, X_train):
    return shap.TreeExplainer(model)


def plot_shap_summary(model, df):
    df_proc = preprocess(df)[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_proc)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, df_proc, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close()
    return len(df_proc)


def plot_shap_waterfall(model, df, idx=0):
    df_proc = preprocess(df)[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    explanation = explainer(df_proc)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(explanation[idx], show=False)
    st.pyplot(fig)
    plt.close()


def plot_lime_explanation(model, df, idx=0):
    df_proc = preprocess(df)[FEATURE_COLS]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=df_proc.values,
        feature_names=FEATURE_COLS,
        class_names=['No Churn', 'Churn'],
        mode='classification'
    )
    instance = df_proc.iloc[idx].values
    exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 5)
    st.pyplot(fig)
    plt.close()