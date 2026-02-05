import pandas as pd
import numpy as np
import shap
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- Data Loading ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Basic cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    return df

@st.cache_data
def get_raw_splits(df):
    X = df.drop('Churn', axis=1)
    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Stratified split ensures churn proportion is maintained
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --- Preprocessor ---
def create_preprocessor(X_train):
    """
    Creates the sklearn ColumnTransformer.
    Now a standalone function so the MLOps engine can call it.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

# --- Training ---
@st.cache_resource
def train_models(df):
    """
    Trains baseline (Logistic) and Production (XGBoost) models.
    """
    X_train, X_test, y_train, y_test = get_raw_splits(df)
    preprocessor = create_preprocessor(X_train)
    
    # XGBoost Pipeline
    pipeline_xgb = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    pipeline_xgb.fit(X_train, y_train)
    
    # Logistic Regression (Baseline)
    from sklearn.linear_model import LogisticRegression
    pipeline_log = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression())
    ])
    pipeline_log.fit(X_train, y_train)
    
    return pipeline_log, pipeline_xgb

# --- XAI Helpers ---
@st.cache_resource
def get_shap_explainer(_model, X_train_processed):
    classifier = _model.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    return explainer

@st.cache_data
def get_shap_values(_explainer, X_test_processed):
    # SHAP TreeExplainer usually expects numpy array if using a pipeline
    shap_values = _explainer.shap_values(X_test_processed)
    return shap_values