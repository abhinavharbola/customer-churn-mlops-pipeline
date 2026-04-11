# Handles data loading, preprocessing, and XGBoost model training with full metrics.

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_curve

FEATURE_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges',
    'charges_per_tenure', 'is_new_customer', 'has_support'
]
TARGET_COL = 'Churn'


def load_data(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df


def preprocess(df):
    df = df.copy()
    df = engineer_features(df)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def engineer_features(df):
    df = df.copy()
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['is_new_customer'] = (df['tenure'] < 12).astype(int)
    df['has_support'] = ((df['TechSupport'] == 1) | (df['OnlineSecurity'] == 1)).astype(int)
    return df

def train_model(df):
    df = preprocess(df)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg/pos,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Find threshold that maximises F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1s)]
    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
        "threshold": round(float(best_thresh), 4)
    }
    return model, X_test, y_test, metrics


def evaluate_on_batch(model, batch_df):
    batch_df = preprocess(batch_df)
    X = batch_df[FEATURE_COLS]
    y = batch_df[TARGET_COL]
    y_pred = model.predict(X)
    return round(f1_score(y, y_pred, zero_division=0), 4)
