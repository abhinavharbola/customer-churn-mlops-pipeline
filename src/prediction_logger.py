# Logs every prediction with timestamp, features, and result to prediction_log.csv.

import os
import pandas as pd
from datetime import datetime

LOG_PATH = "data/prediction_log.csv"


def log_predictions(input_df, probabilities, threshold=0.5):
    df = input_df.copy()
    df['churn_probability'] = probabilities
    df['prediction'] = (probabilities >= threshold).astype(int)
    df['timestamp'] = datetime.now().isoformat(timespec='seconds')

    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


def get_log():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    return pd.read_csv(LOG_PATH, parse_dates=['timestamp'])


def get_summary_metrics():
    log = get_log()
    if log.empty:
        return {"total": 0, "weekly": 0, "churn_rate": 0.0, "avg_probability": 0.0}

    log['timestamp'] = pd.to_datetime(log['timestamp'])
    week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
    weekly = log[log['timestamp'] >= week_ago]

    return {
        "total": len(log),
        "weekly": len(weekly),
        "churn_rate": round(log['prediction'].mean() * 100, 1),
        "avg_probability": round(log['churn_probability'].mean(), 3)
    }