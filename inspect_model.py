import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(".")
models_dir = ROOT / "models"
model_path = models_dir / "best_model.pkl"

if not model_path.exists():
    print(f"Error: {model_path} not found")
else:
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    if hasattr(model, 'feature_importances_'):
         print("Feature importances available")
         print(model.feature_importances_)
    if hasattr(model, 'feature_names_in_'):
         print("Feature names:", model.feature_names_in_)
    elif hasattr(model, 'feature_names'):
         print("Feature names:", model.feature_names)
