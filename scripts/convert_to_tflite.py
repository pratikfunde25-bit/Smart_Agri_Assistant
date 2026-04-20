from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the trained disease classifier to TensorFlow Lite.")
    parser.add_argument(
        "--model-path",
        default=str(Path("models") / "disease" / "leaf_disease_classifier.keras"),
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path("models") / "disease" / "leaf_disease_classifier.tflite"),
        help="Destination path for the .tflite file.",
    )
    parser.add_argument(
        "--quantization",
        default="float16",
        choices=["none", "dynamic", "float16"],
        help="Quantization scheme for mobile deployment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"TensorFlow Lite model saved to {output_path}")


if __name__ == "__main__":
    main()
