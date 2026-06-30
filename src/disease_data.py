from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .disease_taxonomy import build_class_metadata


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_AUGMENTER = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.2),
    ],
    name="leaf_train_augmenter",
)


@dataclass
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_names: List[str]
    crop_names: List[str]
    class_weights: Dict[int, float]
    crop_weights: Dict[int, float]
    class_metadata: List[dict]


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> tf.Tensor:
    axis = tf.range(-(size // 2), size // 2 + 1, dtype=tf.float32)
    kernel_1d = tf.exp(-(axis**2) / (2.0 * sigma**2))
    kernel_1d /= tf.reduce_sum(kernel_1d)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]
    return tf.tile(kernel_2d, [1, 1, 3, 1])


GAUSSIAN_KERNEL = _gaussian_kernel()


def collect_plant_disease_samples(dataset_dirs: Sequence[str | Path]) -> pd.DataFrame:
    records = []

    for dataset_dir in dataset_dirs:
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        class_dirs = [path for path in root.iterdir() if path.is_dir()]
        for class_dir in sorted(class_dirs):
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                records.append({"filepath": str(image_path.resolve()), "class_name": class_dir.name})

    if not records:
        joined = ", ".join(str(Path(path)) for path in dataset_dirs)
        raise ValueError(f"No image files were found inside: {joined}")

    samples = pd.DataFrame(records).drop_duplicates().reset_index(drop=True)
    class_names = sorted(samples["class_name"].unique().tolist())
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    samples["label_idx"] = samples["class_name"].map(class_to_idx)
    metadata = build_class_metadata(class_names)
    class_meta = {item.class_name: item for item in metadata}
    samples["crop_name"] = samples["class_name"].map(lambda name: class_meta[name].crop_name)
    samples["disease_name"] = samples["class_name"].map(lambda name: class_meta[name].disease_name)
    samples["is_healthy"] = samples["class_name"].map(lambda name: class_meta[name].is_healthy)
    return samples


def sample_per_class(
    samples: pd.DataFrame,
    max_images_per_class: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if not max_images_per_class or max_images_per_class <= 0:
        return samples.reset_index(drop=True)

    sampled_frames = []
    for _, frame in samples.groupby("class_name", sort=True):
        take = min(len(frame), max_images_per_class)
        sampled_frames.append(frame.sample(n=take, random_state=random_state))
    return pd.concat(sampled_frames, ignore_index=True)


def split_samples(
    samples: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        samples,
        test_size=0.30,
        stratify=samples["label_idx"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label_idx"],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_balanced_class_weights(labels: Iterable[int]) -> Dict[int, float]:
    labels = np.asarray(list(labels))
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _apply_random_blur(image: tf.Tensor, probability: float = 0.25) -> tf.Tensor:
    def blurred() -> tf.Tensor:
        expanded = tf.expand_dims(image, axis=0)
        convolved = tf.nn.depthwise_conv2d(
            expanded,
            filter=GAUSSIAN_KERNEL,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        return tf.squeeze(convolved, axis=0)

    return tf.cond(tf.random.uniform(()) < probability, blurred, lambda: image)


def augment_leaf_image(image: tf.Tensor) -> tf.Tensor:
    augmented = TRAIN_AUGMENTER(tf.expand_dims(image, axis=0), training=True)
    image = tf.squeeze(augmented, axis=0)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = _apply_random_blur(image)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.03, dtype=tf.float32)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image


def _decode_image(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def build_tf_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["filepath"].tolist(), frame["label_idx"].astype(np.int32).tolist())
    )

    if training:
        dataset = dataset.shuffle(len(frame), reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_image(path, image_size=image_size)
        if training:
            image = augment_leaf_image(image)
        return image, label

    dataset = dataset.map(_load, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def build_joint_tf_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool = False,
    class_weights: Dict[int, float] | None = None,
    crop_weights: Dict[int, float] | None = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            frame["filepath"].tolist(),
            frame["label_idx"].astype(np.int32).tolist(),
            frame["crop_idx"].astype(np.int32).tolist(),
        )
    )

    if training:
        dataset = dataset.shuffle(len(frame), reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, class_label: tf.Tensor, crop_label: tf.Tensor):
        image = _decode_image(path, image_size=image_size)
        if training:
            image = augment_leaf_image(image)

        labels = {
            "class_output": class_label,
            "crop_output": crop_label,
        }

        return image, labels

    dataset = dataset.map(_load, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def build_dataset_bundle(
    dataset_dirs: Sequence[str | Path],
    image_size: tuple[int, int],
    batch_size: int,
    random_state: int = 42,
    max_images_per_class: int | None = None,
) -> DatasetBundle:
    samples = collect_plant_disease_samples(dataset_dirs)
    samples = sample_per_class(samples, max_images_per_class=max_images_per_class, random_state=random_state)
    class_names = sorted(samples["class_name"].unique().tolist())
    crop_names = sorted(samples["crop_name"].unique().tolist())
    crop_to_idx = {crop_name: idx for idx, crop_name in enumerate(crop_names)}
    samples["crop_idx"] = samples["crop_name"].map(crop_to_idx).astype(np.int32)
    train_df, val_df, test_df = split_samples(samples, random_state=random_state)
    class_metadata = build_class_metadata(class_names)

    return DatasetBundle(
        train_ds=build_tf_dataset(train_df, image_size=image_size, batch_size=batch_size, training=True),
        val_ds=build_tf_dataset(val_df, image_size=image_size, batch_size=batch_size, training=False),
        test_ds=build_tf_dataset(test_df, image_size=image_size, batch_size=batch_size, training=False),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        class_names=class_names,
        crop_names=crop_names,
        class_weights=compute_balanced_class_weights(train_df["label_idx"].tolist()),
        crop_weights=compute_balanced_class_weights(train_df["crop_idx"].tolist()),
        class_metadata=[item.__dict__ for item in class_metadata],
    )
