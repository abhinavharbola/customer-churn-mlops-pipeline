import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src import ml_logic

MODEL_DIR = "models"
REGISTRY_FILE = os.path.join(MODEL_DIR, "model_registry.json")

class ModelRegistry:
    def __init__(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        if not os.path.exists(REGISTRY_FILE):
            self._init_registry()

    def _init_registry(self):
        init_data = {"current_version": 0, "history": []}
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(init_data, f)

    def save_model(self, pipeline, metrics, note=""):
        with open(REGISTRY_FILE, 'r') as f:
            reg = json.load(f)
            
        new_ver = reg["current_version"] + 1
        fname = f"model_v{new_ver}.joblib"
        fpath = os.path.join(MODEL_DIR, fname)
        
        joblib.dump(pipeline, fpath)
        
        entry = {
            "version": new_ver,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "path": fpath,
            "metrics": metrics,
            "note": note
        }
        
        reg["current_version"] = new_ver
        reg["history"].append(entry)
        
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(reg, f, indent=4)
            
        return new_ver

    def load_latest_model(self):
        with open(REGISTRY_FILE, 'r') as f:
            reg = json.load(f)
            
        if reg["current_version"] == 0:
            return None
            
        # Find entry for current version
        latest_entry = next(item for item in reg["history"] if item["version"] == reg["current_version"])
        return joblib.load(latest_entry["path"])
    
    def get_history(self):
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)["history"]

def retrain_pipeline(full_data, note="Auto-Retrain"):
    """
    Simulates the retraining of the model
    """
    registry = ModelRegistry()
    
    # 1. Split
    X_train, X_test, y_train, y_test = ml_logic.get_raw_splits(full_data)
    
    # 2. Pipeline
    preprocessor = ml_logic.create_preprocessor(X_train)
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    
    # 3. Train
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 5. Register
    version = registry.save_model(pipeline, {"accuracy": round(acc, 4)}, note)
    return version, acc