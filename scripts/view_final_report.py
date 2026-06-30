import json
from pathlib import Path
import pandas as pd

def main():
    ROOT = Path(__file__).resolve().parents[1]
    metrics_path = ROOT / "reports" / "disease_joint" / "evaluation_metrics.json"
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    print("="*60)
    print("      FINAL MODEL CLASSIFICATION REPORT (CROP + DISEASE)")
    print("="*60)
    print(f"Trained At: {data.get('trained_at', 'N/A')}")
    print(f"Backbone:   {data.get('backbone', 'N/A')}")
    print(f"Total Test Samples: {data.get('test_samples', 'N/A')}")
    print("-"*60)
    
    report = data.get("classification_report", {})
    
    # Convert to DataFrame for pretty printing
    # Filter out accuracy, macro avg, weighted avg for the main table
    rows = []
    for cls_name, metrics in report.items():
        if cls_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        rows.append({
            "Class": cls_name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1-score"],
            "Support": metrics["support"]
        })
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    
    print("-"*60)
    # Summary metrics
    acc = data.get("accuracy", 0)
    crop_acc = data.get("crop_accuracy", 0)
    print(f"Overall Accuracy (Joint): {acc:.4f}")
    print(f"Crop Detection Accuracy:  {crop_acc:.4f}")
    
    if "macro avg" in report:
        m = report["macro avg"]
        print(f"Macro Avg F1-Score:      {m['f1-score']:.4f}")
        print(f"Macro Avg Precision:     {m['precision']:.4f}")
        print(f"Macro Avg Recall:        {m['recall']:.4f}")

    print("="*60)

if __name__ == "__main__":
    main()
