from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .disease_data import DatasetBundle, build_dataset_bundle
from .disease_keras import get_backbone_preprocess, load_disease_model
from .disease_taxonomy import build_metadata_payload


ROOT = Path(__file__).resolve().parents[1]

BACKBONE_REGISTRY = {
    "mobilenetv2": {
        "factory": tf.keras.applications.MobileNetV2,
        "preprocess": get_backbone_preprocess("mobilenetv2"),
        "reason": "MobileNetV2 gives the best latency-to-accuracy tradeoff for mobile and edge inference.",
    },
    "efficientnetb0": {
        "factory": tf.keras.applications.EfficientNetB0,
        "preprocess": get_backbone_preprocess("efficientnetb0"),
        "reason": "EfficientNetB0 is a strong accuracy baseline when a little more compute is acceptable.",
    },
    "resnet50": {
        "factory": tf.keras.applications.ResNet50,
        "preprocess": get_backbone_preprocess("resnet50"),
        "reason": "ResNet50 is heavier but useful as a reference model for offline benchmarking.",
    },
}


@dataclass
class DiseaseTrainingConfig:
    dataset_dirs: Sequence[str | Path]
    image_size: int = 224
    batch_size: int = 32
    backbone: str = "mobilenetv2"
    random_state: int = 42
    max_images_per_class: int | None = None
    head_epochs: int = 6
    fine_tune_epochs: int = 10
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    dropout_rate: float = 0.35
    fine_tune_layers: int = 30
    model_dir: Path = field(default_factory=lambda: ROOT / "models" / "disease")
    report_dir: Path = field(default_factory=lambda: ROOT / "reports" / "disease")

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.image_size, self.image_size)

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "leaf_disease_classifier.keras"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "leaf_disease_metadata.json"

    @property
    def history_path(self) -> Path:
        return self.report_dir / "training_history.json"

    @property
    def metrics_path(self) -> Path:
        return self.report_dir / "evaluation_metrics.json"


def _ensure_output_dirs(config: DiseaseTrainingConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)


def _build_model(config: DiseaseTrainingConfig, num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    key = config.backbone.lower()
    if key not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(f"Unsupported backbone '{config.backbone}'. Choose from: {available}")

    backbone_cfg = BACKBONE_REGISTRY[key]
    base_model = backbone_cfg["factory"](
        include_top=False,
        weights="imagenet",
        input_shape=(config.image_size, config.image_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(config.image_size, config.image_size, 3), name="leaf_image")
    x = tf.keras.layers.Rescaling(255.0, name="restore_pixel_scale")(inputs)
    x = tf.keras.layers.Lambda(backbone_cfg["preprocess"], name=f"{key}_preprocess")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Activation("linear", name="feature_maps")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout_head")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate / 2.0, name="dropout_classifier")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{key}_leaf_disease_classifier")
    return model, base_model


def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def _build_callbacks(config: DiseaseTrainingConfig, suffix: str) -> List[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(config.report_dir / f"training_{suffix}.csv"), append=False),
    ]


def _unfreeze_backbone(base_model: tf.keras.Model, fine_tune_layers: int) -> None:
    base_model.trainable = True
    if fine_tune_layers <= 0:
        return

    split_index = max(len(base_model.layers) - fine_tune_layers, 0)
    for layer in base_model.layers[:split_index]:
        layer.trainable = False
    for layer in base_model.layers[split_index:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def _merge_histories(*histories: tf.keras.callbacks.History) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def _save_history_plots(history: Dict[str, List[float]], config: DiseaseTrainingConfig) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("accuracy", []), label="train")
    axes[0].plot(history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("loss", []), label="train")
    axes[1].plot(history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.report_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    config: DiseaseTrainingConfig,
) -> None:
    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(matrix, cmap="YlGnBu", ax=ax)
    ax.set_title("Leaf Disease Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(config.report_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        config.report_dir / "confusion_matrix.csv"
    )


def _evaluate_classifier(
    model: tf.keras.Model,
    bundle: DatasetBundle,
    config: DiseaseTrainingConfig,
) -> Dict:
    probabilities = model.predict(bundle.test_ds, verbose=0)
    predictions = probabilities.argmax(axis=1)
    labels = bundle.test_df["label_idx"].to_numpy()

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=bundle.class_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    _save_confusion_matrix(labels, predictions, bundle.class_names, config)

    top_predictions = pd.DataFrame(
        {
            "filepath": bundle.test_df["filepath"],
            "actual_class": bundle.test_df["class_name"],
            "predicted_class": [bundle.class_names[idx] for idx in predictions],
            "confidence": probabilities.max(axis=1),
        }
    )
    top_predictions.to_csv(config.report_dir / "test_predictions.csv", index=False)
    return metrics


def _save_metadata(bundle: DatasetBundle, config: DiseaseTrainingConfig) -> None:
    metadata = build_metadata_payload(bundle.class_names)
    metadata.update(
        {
            "backbone": config.backbone.lower(),
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "crop_prediction_integration": (
                "At inference time, disease logits are reweighted using the crop prior from the CSV model."
            ),
            "backbone_reason": BACKBONE_REGISTRY[config.backbone.lower()]["reason"],
        }
    )

    with open(config.metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def train_disease_classifier(config: DiseaseTrainingConfig) -> Dict:
    _ensure_output_dirs(config)

    bundle = build_dataset_bundle(
        dataset_dirs=config.dataset_dirs,
        image_size=config.image_shape,
        batch_size=config.batch_size,
        random_state=config.random_state,
        max_images_per_class=config.max_images_per_class,
    )
    bundle.train_df.to_csv(config.report_dir / "train_split.csv", index=False)
    bundle.val_df.to_csv(config.report_dir / "val_split.csv", index=False)
    bundle.test_df.to_csv(config.report_dir / "test_split.csv", index=False)

    model, base_model = _build_model(config=config, num_classes=len(bundle.class_names))
    _compile_model(model, learning_rate=config.learning_rate)

    warmup_history = model.fit(
        bundle.train_ds,
        validation_data=bundle.val_ds,
        epochs=config.head_epochs,
        class_weight=bundle.class_weights,
        callbacks=_build_callbacks(config, suffix="warmup"),
        verbose=1,
    )

    _unfreeze_backbone(base_model, fine_tune_layers=config.fine_tune_layers)
    _compile_model(model, learning_rate=config.fine_tune_learning_rate)

    fine_tune_history = model.fit(
        bundle.train_ds,
        validation_data=bundle.val_ds,
        epochs=config.head_epochs + config.fine_tune_epochs,
        initial_epoch=config.head_epochs,
        class_weight=bundle.class_weights,
        callbacks=_build_callbacks(config, suffix="finetune"),
        verbose=1,
    )

    best_model = load_disease_model(config.checkpoint_path, backbone=config.backbone.lower(), compile=True)
    history = _merge_histories(warmup_history, fine_tune_history)
    _save_history_plots(history, config)
    _save_metadata(bundle, config)

    with open(config.history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    metrics = _evaluate_classifier(best_model, bundle, config)
    metrics.update(
        {
            "backbone": config.backbone.lower(),
            "image_size": config.image_size,
            "train_samples": int(len(bundle.train_df)),
            "val_samples": int(len(bundle.val_df)),
            "test_samples": int(len(bundle.test_df)),
            "num_classes": int(len(bundle.class_names)),
            "class_weights": bundle.class_weights,
            "backbone_reason": BACKBONE_REGISTRY[config.backbone.lower()]["reason"],
            "max_images_per_class": config.max_images_per_class,
        }
    )

    with open(config.metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
