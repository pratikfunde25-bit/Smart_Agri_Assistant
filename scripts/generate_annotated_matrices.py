import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set base directories
BASE_DIR = r"d:\Major Project sem 6 spit - Copy\Smart_Agri_Assistant"
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Models to process
models = {
    "Crop Recommendation (XGBoost)": {
        "metrics_file": os.path.join(REPORTS_DIR, "metrics.json"),
        "cm_image": "crop_recommendation_cm_annotated.png",
        "has_cm_in_json": True,
        "is_hybrid": False
    },
    "Disease Detection (MobileNetV2)": {
        "metrics_file": os.path.join(REPORTS_DIR, "disease", "evaluation_metrics.json"),
        "cm_csv": os.path.join(REPORTS_DIR, "disease", "confusion_matrix.csv"),
        "cm_image": "disease_detection_cm_annotated.png",
        "has_cm_in_json": False,
        "is_hybrid": False
    },
    "Hybrid/Joint Disease & Crop Model (DenseNet121)": {
        "metrics_file": os.path.join(REPORTS_DIR, "disease_joint", "evaluation_metrics.json"),
        "cm_csv": os.path.join(REPORTS_DIR, "disease_joint", "confusion_matrix.csv"),
        "cm_image": "joint_disease_cm_annotated.png",
        "has_cm_in_json": False,
        "is_hybrid": True
    }
}

report_md = "# Comprehensive Detailed Report: Model Performances\n\n"
report_md += "This report contains the evaluation metrics (Accuracy, Precision, Recall, F1-Score) and Annotated Confusion Matrices for all the machine learning models developed in the Smart Agri Assistant project.\n\n"

def plot_cm(cm_data, class_names, title, save_path):
    plt.figure(figsize=(24, 20))
    # If the matrix is large (like 38 classes), annotate might be small, so adjust font size
    annot_kws = {"size": 8}
    
    sns.heatmap(cm_data, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws=annot_kws)
    plt.title(title, fontsize=24)
    plt.ylabel('Actual Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

for model_name, info in models.items():
    if not os.path.exists(info["metrics_file"]):
        print(f"Skipping {model_name}: Metrics file not found.")
        continue
        
    with open(info["metrics_file"], "r") as f:
        metrics = json.load(f)
        
    report_md += f"## {model_name}\n\n"
    
    # Extract overall metrics
    # XGBoost format vs NN format
    if "macro avg" in metrics:
        # NN format or sklearn report format
        acc = metrics.get("accuracy", "N/A")
        if isinstance(acc, float):
            acc = f"{acc:.4f}"
            
        prec = metrics["macro avg"].get("precision", "N/A")
        rec = metrics["macro avg"].get("recall", "N/A")
        f1 = metrics["macro avg"].get("f1-score", "N/A")
    elif "report" in metrics:
        # XGBoost format
        acc = f"{metrics.get('accuracy', 0):.4f}"
        report_data = metrics.get("report", {})
        prec = report_data.get("macro avg", {}).get("precision", "N/A")
        rec = report_data.get("macro avg", {}).get("recall", "N/A")
        f1 = report_data.get("macro avg", {}).get("f1-score", "N/A")
    else:
        acc = f"{metrics.get('accuracy', 0):.4f}"
        prec = metrics.get("pair_precision_macro", metrics.get("precision_macro", "N/A"))
        rec = metrics.get("pair_recall_macro", metrics.get("recall_macro", "N/A"))
        f1 = metrics.get("pair_f1_macro", metrics.get("f1_macro", "N/A"))
    
    if isinstance(prec, float): prec = f"{prec:.4f}"
    if isinstance(rec, float): rec = f"{rec:.4f}"
    if isinstance(f1, float): f1 = f"{f1:.4f}"
        
    report_md += "### Overall Performance\n"
    report_md += f"- **Accuracy**: {acc}\n"
    report_md += f"- **Precision (Macro)**: {prec}\n"
    report_md += f"- **Recall (Macro)**: {rec}\n"
    report_md += f"- **F1 Score (Macro)**: {f1}\n\n"
    
    if info.get("is_hybrid"):
        report_md += f"- **Auxiliary Crop Head Accuracy**: {metrics.get('crop_accuracy', 'N/A'):.4f}\n\n"
    
    # Plot Annotated Confusion Matrix
    cm_image_path = os.path.join(DOCS_DIR, info["cm_image"])
    
    cm_data = None
    class_names = []
    
    if info["has_cm_in_json"]:
        cm_data = np.array(metrics["confusion_matrix"])
        # If it's the crop model, class names might just be indices 0 to N
        class_names = [str(i) for i in range(cm_data.shape[0])]
        plot_cm(cm_data, class_names, f"Annotated Confusion Matrix: {model_name}", cm_image_path)
    else:
        if os.path.exists(info["cm_csv"]):
            cm_df = pd.read_csv(info["cm_csv"], index_col=0)
            cm_data = cm_df.values
            class_names = cm_df.index.tolist()
            plot_cm(cm_data, class_names, f"Annotated Confusion Matrix: {model_name}", cm_image_path)
            
    if cm_data is not None:
        report_md += f"### Annotated Confusion Matrix\n"
        report_md += f"The confusion matrix is annotated with exact counts of True Positives, True Negatives, False Positives, and False Negatives per class.\n\n"
        report_md += f"![{model_name} Confusion Matrix]({info['cm_image']})\n\n"
        
    # Class-wise report
    report_md += "### Class-wise Detailed Report\n"
    report_md += "| Class Name | Precision | Recall | F1-Score | Support (TP + FN) |\n"
    report_md += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    class_report = metrics.get("classification_report", metrics.get("report", {}))
    for cls_name, cls_metrics in class_report.items():
        if cls_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        try:
            p = f"{cls_metrics['precision']:.4f}"
            r = f"{cls_metrics['recall']:.4f}"
            f1_val = f"{cls_metrics['f1-score']:.4f}"
            s = str(int(cls_metrics['support']))
            report_md += f"| {cls_name} | {p} | {r} | {f1_val} | {s} |\n"
        except TypeError:
            pass
            
    report_md += "\n---\n\n"

# Save markdown report
report_path = os.path.join(DOCS_DIR, "detailed_model_report.md")
with open(report_path, "w") as f:
    f.write(report_md)

print("Generated detailed report and annotated confusion matrices successfully.")
