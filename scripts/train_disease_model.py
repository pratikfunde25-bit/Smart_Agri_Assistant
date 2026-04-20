from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.disease_model import DiseaseTrainingConfig, train_disease_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid crop disease classifier on PlantVillage images.")
    parser.add_argument(
        "--dataset-dir",
        dest="dataset_dirs",
        action="append",
        required=True,
        help="Path to a dataset root where each subfolder is a PlantVillage-style class.",
    )
    parser.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0", "resnet50"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images-per-class", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DiseaseTrainingConfig(
        dataset_dirs=[Path(path) for path in args.dataset_dirs],
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_images_per_class=args.max_images_per_class or None,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        fine_tune_layers=args.fine_tune_layers,
    )
    metrics = train_disease_classifier(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
