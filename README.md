Self-Healing MLOps System: Customer Churn

A production-grade machine learning system that not only predicts churn but also monitors its own health and automatically retrains when data drift is detected.

Architecture

```bash
graph LR
    A[Data Stream] --> B{Drift Detector};
    B -- Stable --> C[Prediction API];
    B -- Drift! --> D[Retraining Engine];
    D --> E[Model Registry];
    E --> C;
```

Key Features

Production Model Registry: Local version control for ML models (saves artifacts as model_v1.joblib, model_v2.joblib, etc.).

Automated Drift Detection: Uses Evidently AI to compare incoming production traffic against training data using statistical tests (KS-test).

Auto-Retraining Pipeline: When drift is detected, the system ingests the new data, retrains the XGBoost model, and hot-swaps the production model without downtime.

Explainability (XAI): Integrated SHAP and LIME for local and global model interpretation.


Project Structure

```bash
├── app.py                 # Main Streamlit Dashboard
├── src/                   # Core Logic
│   ├── mlops_engine.py    # Retraining & Registry Logic
│   ├── monitoring.py      # Evidently AI Drift Detection
│   └── ml_logic.py        # Model Architecture
├── models/                # Artifact Store (JSON + .joblib)
└── data/                  # Data Store
```

Setup & Run

Install Requirements

```bash
pip install -r requirements.txt
```

Run the App

```bash
streamlit run app.py
```

Simulate MLOps Workflow

Go to Sidebar -> Select "Drifted (High Churn)".

Click Generate Traffic.

Go to Drift Monitor tab -> Click Run Check (See the red alert).

Go to MLOps Pipeline tab -> Click Trigger Retraining.

Watch the Model Version update automatically!