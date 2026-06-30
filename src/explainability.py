from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int | None = None,
    feature_layer_name: str = "feature_maps",
) -> np.ndarray:
    # Handle multi-task models where model.output might be a list
    model_outputs = model.output
    if isinstance(model_outputs, (list, tuple)):
        # We target the first output branch (usually the disease classification head)
        target_output = model_outputs[0]
    else:
        target_output = model_outputs

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(feature_layer_name).output, target_output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            # predictions has shape (1, num_classes)
            # We must specify axis=-1 to get the index across classes
            class_index_tensor = tf.argmax(predictions[0], axis=-1)
            class_index = int(tf.reshape(class_index_tensor, []))
        
        # Ensure we pick the right score for the requested class
        target_score = predictions[:, class_index]

    gradients = tape.gradient(target_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    
    # Use tf.reduce_max and convert to float safely
    max_val = tf.reduce_max(heatmap)
    if float(np.asarray(max_val)) > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap_on_image(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    resized_heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)


def save_gradcam_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap_on_image(image_rgb=image_rgb, heatmap=heatmap)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path
