from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.joint_disease_model import JointDiseaseTrainingConfig, train_joint_disease_classifier


def _existing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _default_dataset_dirs() -> list[Path]:
    workspace_defaults = [
        ROOT / "data" / "external" / "plantvillage dataset" / "color",
        ROOT / "data" / "external" / "maharashtra_leaf_dataset",
        ROOT / "data" / "external" / "field_leaf_images",
    ]

    sibling_root = ROOT.parent
    sibling_defaults = [
        sibling_root / "Banana Disease Recognition Dataset" / "Original Images" / "Original Images",
        sibling_root / "Cotton Disease" / "train",
        sibling_root / "Rice_Leaf_AUG",
        sibling_root / "rice_leaf_diseases",
        sibling_root / "Sugarcane Disease Dataset",
        sibling_root / "Wheat Disease Dataset",
    ]
    return _existing_paths(workspace_defaults + sibling_defaults)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Maharashtra-focused joint crop+disease model. "
            "This script is designed for multi-dataset training with early stopping and fine-tuning."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        dest="dataset_dirs",
        action="append",
        help="Add one or more dataset roots where each subfolder is a Crop___Disease class.",
    )
    parser.add_argument("--backbone", default="densenet121", choices=["densenet121", "resnet50v2", "xception"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images-per-class", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=6e-5)
    parser.add_argument("--crop-loss-weight", type=float, default=0.25)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--fine-tune-layers", type=int, default=100)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if a complete joint model package already exists.",
    )
    parser.add_argument(
        "--print-default-datasets",
        action="store_true",
        help="Print the auto-detected default dataset directories and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_dirs = _default_dataset_dirs()
    if args.print_default_datasets:
        print(json.dumps([str(path) for path in default_dirs], indent=2))
        return

    dataset_dirs = [Path(path) for path in args.dataset_dirs] if args.dataset_dirs else default_dirs
    if not dataset_dirs:
        raise FileNotFoundError(
            "No dataset directories were found. Pass one or more --dataset-dir values with Crop___Disease folders."
        )

    config = JointDiseaseTrainingConfig(
        dataset_dirs=dataset_dirs,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_images_per_class=args.max_images_per_class or None,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        crop_loss_weight=args.crop_loss_weight,
        class_loss_weight=args.class_loss_weight,
        fine_tune_layers=args.fine_tune_layers,
    )

    if config.checkpoint_path.exists() and config.metadata_path.exists() and not args.force_retrain:
        print(
            json.dumps(
                {
                    "status": "already_trained",
                    "message": "A complete joint model package already exists. Pass --force-retrain to rebuild it.",
                    "model_path": str(config.checkpoint_path),
                    "metadata_path": str(config.metadata_path),
                },
                indent=2,
            )
        )
        return

    print("Training with datasets:")
    for dataset_dir in dataset_dirs:
        print(f" - {dataset_dir}")

    metrics = train_joint_disease_classifier(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
