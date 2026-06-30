from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .disease_data import DatasetBundle, build_joint_tf_dataset, build_dataset_bundle
from .disease_keras import get_backbone_preprocess, load_disease_model
from .disease_taxonomy import build_metadata_payload


ROOT = Path(__file__).resolve().parents[1]

BACKBONE_REGISTRY = {
    "densenet121": {
        "factory": tf.keras.applications.DenseNet121,
        "preprocess": get_backbone_preprocess("densenet121"),
        "reason": "DenseNet121 is a strong fine-grained visual backbone for leaf texture, lesion pattern, and vein-level cues.",
    },
    "resnet50v2": {
        "factory": tf.keras.applications.ResNet50V2,
        "preprocess": get_backbone_preprocess("resnet50v2"),
        "reason": "ResNet50V2 is a reliable transfer-learning baseline with stable optimization for agricultural images.",
    },
    "xception": {
        "factory": tf.keras.applications.Xception,
        "preprocess": get_backbone_preprocess("xception"),
        "reason": "Xception is often strong on plant pathology datasets because depthwise separable filters preserve lesion detail well.",
    },
}


@dataclass
class JointDiseaseTrainingConfig:
    dataset_dirs: Sequence[str | Path]
    image_size: int = 224
    batch_size: int = 16
    backbone: str = "densenet121"
    random_state: int = 42
    max_images_per_class: int | None = None
    head_epochs: int = 8
    fine_tune_epochs: int = 18
    learning_rate: float = 8e-4
    fine_tune_learning_rate: float = 7e-5
    dropout_rate: float = 0.35
    crop_loss_weight: float = 0.3
    class_loss_weight: float = 1.0
    fine_tune_layers: int = 80
    label_smoothing: float = 0.04
    l2_regularization: float = 2e-4
    model_dir: Path = field(default_factory=lambda: ROOT / "models" / "disease")
    report_dir: Path = field(default_factory=lambda: ROOT / "reports" / "disease_joint")

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.image_size, self.image_size)

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "leaf_disease_joint_classifier.keras"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "leaf_disease_joint_metadata.json"

    @property
    def history_path(self) -> Path:
        return self.report_dir / "training_history.json"

    @property
    def metrics_path(self) -> Path:
        return self.report_dir / "evaluation_metrics.json"


def _ensure_output_dirs(config: JointDiseaseTrainingConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)


def _build_model(
    config: JointDiseaseTrainingConfig,
    num_classes: int,
    num_crops: int,
) -> tuple[tf.keras.Model, tf.keras.Model]:
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

    regularizer = tf.keras.regularizers.l2(config.l2_regularization)

    def dense_block(
        x: tf.Tensor,
        units: int,
        name: str,
        dropout_rate: float,
    ) -> tf.Tensor:
        x = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=regularizer, name=f"{name}_dense")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = tf.keras.layers.Activation("gelu", name=f"{name}_act")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
        return x

    inputs = tf.keras.layers.Input(shape=(config.image_size, config.image_size, 3), name="leaf_image")
    x = tf.keras.layers.Rescaling(255.0, name="restore_pixel_scale")(inputs)
    x = tf.keras.layers.Lambda(backbone_cfg["preprocess"], name=f"{key}_preprocess")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.Activation("linear", name="feature_maps")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="shared_batch_norm")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="shared_dropout")(x)
    shared = dense_block(x, units=640, name="shared_block_1", dropout_rate=config.dropout_rate)
    shared = dense_block(shared, units=384, name="shared_block_2", dropout_rate=config.dropout_rate / 1.2)

    class_branch = dense_block(shared, units=320, name="class_block_1", dropout_rate=config.dropout_rate / 1.35)
    class_branch = dense_block(class_branch, units=192, name="class_block_2", dropout_rate=config.dropout_rate / 1.5)
    class_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(class_branch)

    crop_branch = dense_block(shared, units=224, name="crop_block_1", dropout_rate=config.dropout_rate / 1.7)
    crop_branch = dense_block(crop_branch, units=96, name="crop_block_2", dropout_rate=config.dropout_rate / 1.9)
    crop_output = tf.keras.layers.Dense(num_crops, activation="softmax", name="crop_output")(crop_branch)

    model = tf.keras.Model(inputs=inputs, outputs=[class_output, crop_output], name=f"{key}_joint_leaf_classifier")
    return model, base_model


def _compile_model(model: tf.keras.Model, config: JointDiseaseTrainingConfig, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-5),
        loss={
            "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            "crop_output": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "class_output": config.class_loss_weight,
            "crop_output": config.crop_loss_weight,
        },
        metrics={
            "class_output": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            ],
            "crop_output": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
            ],
        },
    )


def _build_callbacks(config: JointDiseaseTrainingConfig, suffix: str) -> List[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.checkpoint_path),
            monitor="val_class_output_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_class_output_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_class_output_loss",
            mode="min",
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


def _save_history_plots(history: Dict[str, List[float]], config: JointDiseaseTrainingConfig) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history.get("class_output_accuracy", []), label="train")
    axes[0].plot(history.get("val_class_output_accuracy", []), label="val")
    axes[0].set_title("Pair Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("crop_output_accuracy", []), label="train")
    axes[1].plot(history.get("val_crop_output_accuracy", []), label="val")
    axes[1].set_title("Crop Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history.get("loss", []), label="train")
    axes[2].plot(history.get("val_loss", []), label="val")
    axes[2].set_title("Total Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(config.report_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    config: JointDiseaseTrainingConfig,
) -> None:
    matrix = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(matrix, cmap="YlGnBu", ax=ax)
    ax.set_title("Joint Crop Disease Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(config.report_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        config.report_dir / "confusion_matrix.csv"
    )


def _class_meta_lookup(bundle: DatasetBundle) -> tuple[Dict[int, str], Dict[int, str]]:
    crop_lookup: Dict[int, str] = {}
    disease_lookup: Dict[int, str] = {}
    for index, meta in enumerate(bundle.class_metadata):
        crop_lookup[index] = meta["crop_name"]
        disease_lookup[index] = meta["disease_name"]
    return crop_lookup, disease_lookup


def _save_metadata(bundle: DatasetBundle, config: JointDiseaseTrainingConfig) -> None:
    metadata = build_metadata_payload(bundle.class_names)
    metadata.update(
        {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "backbone": config.backbone.lower(),
            "image_size": config.image_size,
            "batch_size": config.batch_size,
            "model_type": "joint_crop_disease",
            "class_output_name": "class_output",
            "crop_output_name": "crop_output",
            "crop_names": bundle.crop_names,
            "num_pair_classes": len(bundle.class_names),
            "num_crops": len(bundle.crop_names),
            "backbone_reason": BACKBONE_REGISTRY[config.backbone.lower()]["reason"],
            "notes": (
                "Primary head predicts the exact crop+disease class. Auxiliary crop head stabilizes crop identity "
                "and improves confidence consistency for deployment. Training includes heavier augmentation and "
                "extra intermediate dense blocks to reduce overfitting to crop-specific visual shortcuts."
            ),
        }
    )
    with open(config.metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def _evaluate_classifier(model: tf.keras.Model, bundle: DatasetBundle, config: JointDiseaseTrainingConfig) -> Dict:
    raw_outputs = model.predict(
        build_joint_tf_dataset(
            bundle.test_df,
            image_size=config.image_shape,
            batch_size=config.batch_size,
            training=False,
        ),
        verbose=0,
    )
    class_probs, crop_probs = raw_outputs[0], raw_outputs[1]
    class_predictions = class_probs.argmax(axis=1)
    crop_predictions = crop_probs.argmax(axis=1)

    class_labels = bundle.test_df["label_idx"].to_numpy()
    crop_labels = bundle.test_df["crop_idx"].to_numpy()
    crop_lookup, disease_lookup = _class_meta_lookup(bundle)

    predicted_diseases = np.array([disease_lookup[int(index)] for index in class_predictions], dtype=object)
    actual_diseases = bundle.test_df["disease_name"].to_numpy(dtype=object)

    metrics = {
        "accuracy": float(accuracy_score(class_labels, class_predictions)),
        "pair_accuracy": float(accuracy_score(class_labels, class_predictions)),
        "crop_accuracy": float(accuracy_score(crop_labels, crop_predictions)),
        "crop_accuracy_from_pair_head": float(
            accuracy_score(
                bundle.test_df["crop_name"].to_numpy(dtype=object),
                np.array([crop_lookup[int(index)] for index in class_predictions], dtype=object),
            )
        ),
        "disease_accuracy_from_pair_head": float(accuracy_score(actual_diseases, predicted_diseases)),
        "pair_top3_accuracy": float(
            np.mean(
                [
                    true_label in np.argsort(probabilities)[-3:]
                    for true_label, probabilities in zip(class_labels, class_probs, strict=False)
                ]
            )
        ),
        "pair_precision_macro": float(precision_score(class_labels, class_predictions, average="macro", zero_division=0)),
        "pair_recall_macro": float(recall_score(class_labels, class_predictions, average="macro", zero_division=0)),
        "pair_f1_macro": float(f1_score(class_labels, class_predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            class_labels,
            class_predictions,
            target_names=bundle.class_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    _save_confusion_matrix(class_labels, class_predictions, bundle.class_names, config)

    top_predictions = pd.DataFrame(
        {
            "filepath": bundle.test_df["filepath"],
            "actual_class": bundle.test_df["class_name"],
            "actual_crop": bundle.test_df["crop_name"],
            "actual_disease": bundle.test_df["disease_name"],
            "predicted_class": [bundle.class_names[idx] for idx in class_predictions],
            "predicted_crop_from_pair": [crop_lookup[int(idx)] for idx in class_predictions],
            "predicted_disease_from_pair": [disease_lookup[int(idx)] for idx in class_predictions],
            "predicted_crop_aux": [bundle.crop_names[idx] for idx in crop_predictions],
            "pair_confidence": class_probs.max(axis=1),
            "crop_confidence": crop_probs.max(axis=1),
        }
    )
    top_predictions.to_csv(config.report_dir / "test_predictions.csv", index=False)
    return metrics


def train_joint_disease_classifier(config: JointDiseaseTrainingConfig) -> Dict:
    _ensure_output_dirs(config)
    tf.keras.utils.set_random_seed(config.random_state)

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

    train_ds = build_joint_tf_dataset(
        bundle.train_df,
        image_size=config.image_shape,
        batch_size=config.batch_size,
        training=True,
        class_weights=bundle.class_weights,
        crop_weights=bundle.crop_weights,
    )
    val_ds = build_joint_tf_dataset(
        bundle.val_df,
        image_size=config.image_shape,
        batch_size=config.batch_size,
        training=False,
        class_weights=bundle.class_weights,
        crop_weights=bundle.crop_weights,
    )

    model, base_model = _build_model(
        config=config,
        num_classes=len(bundle.class_names),
        num_crops=len(bundle.crop_names),
    )
    _compile_model(model, config=config, learning_rate=config.learning_rate)

    warmup_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.head_epochs,
        callbacks=_build_callbacks(config, suffix="warmup"),
        verbose=1,
    )

    _unfreeze_backbone(base_model, fine_tune_layers=config.fine_tune_layers)
    _compile_model(model, config=config, learning_rate=config.fine_tune_learning_rate)

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.head_epochs + config.fine_tune_epochs,
        initial_epoch=config.head_epochs,
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
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "backbone": config.backbone.lower(),
            "image_size": config.image_size,
            "train_samples": int(len(bundle.train_df)),
            "val_samples": int(len(bundle.val_df)),
            "test_samples": int(len(bundle.test_df)),
            "num_pair_classes": int(len(bundle.class_names)),
            "num_crops": int(len(bundle.crop_names)),
            "class_weights": bundle.class_weights,
            "crop_weights": bundle.crop_weights,
            "backbone_reason": BACKBONE_REGISTRY[config.backbone.lower()]["reason"],
            "max_images_per_class": config.max_images_per_class,
        }
    )

    with open(config.metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
