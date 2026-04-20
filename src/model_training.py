import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from .data_pipeline import PROCESSED_DIR, preprocess


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
VIS_DIR = ROOT / "visualizations"

NUMERIC_COLS = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
CATEGORICAL_COLS = ["soil_type", "season", "region"]


def load_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "merged_clean.csv"
    if not path.exists():
        return preprocess()
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def model_search_spaces():
    spaces = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            "params": {
                "model__n_estimators": [200, 400, 800],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
            },
            "n_iter": 8,
        },
        "SVM": {
            "model": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
            "params": {
                "model__C": [1, 5, 10],
                "model__gamma": ["scale", 0.01, 0.1],
            },
            "n_iter": 6,
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [5, 15, 25, 35],
                "model__weights": ["uniform", "distance"],
            },
            "n_iter": 6,
        },
    }
    if XGBClassifier is not None:
        spaces["XGBoost"] = {
            "model": XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",
            ),
            "params": {
                "model__n_estimators": [300, 600],
                "model__max_depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            "n_iter": 12,
        }
    return spaces


def save_plot(fig, filename: str) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(VIS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_eda_plots(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    order = df["crop_label"].value_counts().index
    sns.countplot(data=df, y="crop_label", order=order, ax=ax, palette="viridis")
    ax.set_title("Crop Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Crop")
    save_plot(fig, "crop_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[NUMERIC_COLS].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")

    regional = (
        df.groupby(["region", "crop_label"]).size().reset_index(name="count")
        .sort_values(["region", "count"], ascending=[True, False])
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=regional.head(20), x="count", y="crop_label", hue="region", ax=ax)
    ax.set_title("Regional Crop Patterns")
    save_plot(fig, "regional_patterns.png")


def create_diagnostics(df: pd.DataFrame, y_encoded: np.ndarray) -> dict:
    counts = df["crop_label"].value_counts().to_dict()
    duplicate_rate = float(df[NUMERIC_COLS + CATEGORICAL_COLS + ["crop_label"]].duplicated().mean())
    return {
        "row_count": int(len(df)),
        "class_count": int(df["crop_label"].nunique()),
        "class_support": counts,
        "duplicate_rate": duplicate_rate,
        "encoded_class_count": int(len(np.unique(y_encoded))),
    }


def save_comparison(results: list) -> None:
    comparison = pd.DataFrame(
        [
            {
                "model": item["model"],
                "accuracy": item["accuracy"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1_macro": item["f1_macro"],
            }
            for item in results
        ]
    ).sort_values(["f1_macro", "accuracy"], ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_plot(fig, "confusion_matrix.png")


def save_feature_importance(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    else:
        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
        scores = perm.importances_mean

    ranking = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    ranking.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=ranking, x="importance", y="feature", ax=ax, palette="crest")
    ax.set_title("Top Feature Importance")
    save_plot(fig, "feature_importance.png")


def train_and_select() -> dict:
    df = load_processed()
    save_eda_plots(df)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y_raw = df["crop_label"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    diagnostics = create_diagnostics(df, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor()
    all_results = []
    best_bundle = None

    for name, cfg in model_search_spaces().items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", cfg["model"]),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=cfg["params"],
            n_iter=cfg["n_iter"],
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        result = {
            "model": name,
            "best_params": search.best_params_,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        all_results.append(result)

        if best_bundle is None or (result["f1_macro"], result["accuracy"]) > (
            best_bundle["metrics"]["f1_macro"],
            best_bundle["metrics"]["accuracy"],
        ):
            best_bundle = {"metrics": result, "model": best_model, "y_pred": y_pred}

        print(f"{name} done: F1-macro={result['f1_macro']:.3f}, Acc={result['accuracy']:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    best_metrics = dict(best_bundle["metrics"])
    best_metrics["diagnostics"] = diagnostics
    best_metrics["class_labels"] = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_))).tolist()

    joblib.dump(best_bundle["model"], MODELS_DIR / "best_model.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    with open(MODELS_DIR / "class_labels.json", "w", encoding="utf-8") as file:
        json.dump(best_metrics["class_labels"], file, indent=2)

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=2)

    with open(REPORTS_DIR / "diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)

    save_comparison(all_results)
    save_confusion(y_test, best_bundle["y_pred"], best_metrics["class_labels"], best_metrics["model"])
    save_feature_importance(best_bundle["model"], X_test, y_test)

    return best_metrics


if __name__ == "__main__":
    train_and_select()
