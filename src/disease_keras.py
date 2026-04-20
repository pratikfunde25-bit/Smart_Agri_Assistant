from __future__ import annotations

from pathlib import Path

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="smart_agri")
def mobilenetv2_preprocess(x):
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)


@tf.keras.utils.register_keras_serializable(package="smart_agri")
def efficientnetb0_preprocess(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)


@tf.keras.utils.register_keras_serializable(package="smart_agri")
def resnet50_preprocess(x):
    return tf.keras.applications.resnet50.preprocess_input(x)


def get_backbone_preprocess(backbone: str):
    key = backbone.lower()
    mapping = {
        "mobilenetv2": mobilenetv2_preprocess,
        "efficientnetb0": efficientnetb0_preprocess,
        "resnet50": resnet50_preprocess,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported backbone '{backbone}' for preprocessing.")
    return mapping[key]


def get_custom_objects(backbone: str | None = None) -> dict:
    objects = {
        "mobilenetv2_preprocess": mobilenetv2_preprocess,
        "efficientnetb0_preprocess": efficientnetb0_preprocess,
        "resnet50_preprocess": resnet50_preprocess,
    }

    if backbone:
        preprocess = get_backbone_preprocess(backbone)
        objects["preprocess_input"] = preprocess
    return objects


def load_disease_model(model_path: str | Path, backbone: str | None = None, compile: bool = False):
    return tf.keras.models.load_model(
        model_path,
        compile=compile,
        custom_objects=get_custom_objects(backbone),
    )
