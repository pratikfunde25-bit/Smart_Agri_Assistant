import joblib
import json
from pathlib import Path
import numpy as np

roots = Path(".")
model_path = roots / "models" / "best_model.pkl"
classes_path = roots / "models" / "class_labels.json"

if model_path.exists():
    model = joblib.load(model_path)
    
    # Extract core importance
    if hasattr(model, 'steps'):
        # It's a pipeline
        final_step = model.steps[-1][1]
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        if hasattr(final_step, 'feature_importances_'):
            importances = final_step.feature_importances_
            importance_map = dict(zip(feature_names, [float(x) for x in importances]))
            
            # Save metadata
            metadata = {
                "feature_importance": importance_map,
                "model_type": str(type(final_step)),
                "features": list(feature_names)
            }
            with open("data/model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print("Model metadata saved to data/model_metadata.json")
        else:
            print("Final estimator doesn't have feature_importances_")
    else:
        print("Model is not a pipeline")
else:
    print("Model pkl not found")
