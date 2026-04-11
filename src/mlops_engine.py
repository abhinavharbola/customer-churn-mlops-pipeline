# Model registry, versioning, and retraining pipeline with before/after metric tracking.

import os
import json
import joblib
from datetime import datetime
from src.ml_logic import train_model, evaluate_on_batch, load_data

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
os.makedirs(MODELS_DIR, exist_ok=True)


class ModelRegistry:
    def _read(self):
        if not os.path.exists(REGISTRY_PATH):
            return []
        with open(REGISTRY_PATH) as f:
            return json.load(f)

    def _write(self, history):
        with open(REGISTRY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    def save(self, model, metrics, note="", pre_retrain_f1=None):
        history = self._read()
        version = len(history) + 1
        path = os.path.join(MODELS_DIR, f"model_v{version}.joblib")
        joblib.dump(model, path)
        entry = {
            "version": version,
            "trained_at": datetime.now().isoformat(timespec='seconds'),
            "model_path": path,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "auc_roc": metrics["auc_roc"],
            "threshold": metrics.get("threshold", 0.5),
            "pre_retrain_f1": pre_retrain_f1,
            "note": note
        }
        history.append(entry)
        self._write(history)
        return version

    def load_latest_model(self):
        history = self._read()
        if not history:
            return None
        path = history[-1]["model_path"]
        return joblib.load(path) if os.path.exists(path) else None

    def get_history(self):
        return self._read()


def retrain_pipeline(df, note="", drifted_batch=None):
    registry = ModelRegistry()
    current_model = registry.load_latest_model()

    pre_retrain_f1 = None
    if current_model is not None and drifted_batch is not None:
        try:
            pre_retrain_f1 = evaluate_on_batch(current_model, drifted_batch)
        except Exception:
            pre_retrain_f1 = None

    model, X_test, y_test, metrics = train_model(df)
    version = registry.save(model, metrics, note=note, pre_retrain_f1=pre_retrain_f1)
    return version, metrics
