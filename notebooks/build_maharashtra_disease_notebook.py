from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "maharashtra_joint_disease_training.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> None:
    cells = [
        markdown_cell(
            "# Maharashtra Joint Crop Disease Training\n"
            "This notebook trains the upgraded joint crop+disease model for Smart Agri Assistant.\n\n"
            "Goal: improve disease and crop recognition for Maharashtra-focused crops using PlantVillage plus local datasets."
        ),
        markdown_cell("## Setup"),
        code_cell(
            "import sys, json\n"
            "from pathlib import Path\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.joint_disease_model import JointDiseaseTrainingConfig, train_joint_disease_classifier\n"
            "from src.disease_data import collect_plant_disease_samples\n"
            "ROOT\n"
        ),
        markdown_cell("## Choose datasets"),
        code_cell(
            "dataset_dirs = [\n"
            "    ROOT / 'data' / 'external' / 'plantvillage dataset' / 'color',\n"
            "    # ROOT / 'data' / 'external' / 'maharashtra_leaf_dataset',\n"
            "    # Path(r'D:/Major Project sem 6 spit/Banana Disease Recognition Dataset/Original Images/Original Images'),\n"
            "    # Path(r'D:/Major Project sem 6 spit/Cotton Disease/train'),\n"
            "    # Path(r'D:/Major Project sem 6 spit/Rice_Leaf_AUG'),\n"
            "    # Path(r'D:/Major Project sem 6 spit/rice_leaf_diseases'),\n"
            "]\n"
            "dataset_dirs = [path for path in dataset_dirs if path.exists()]\n"
            "dataset_dirs\n"
        ),
        markdown_cell("## Inspect class coverage"),
        code_cell(
            "samples = collect_plant_disease_samples(dataset_dirs)\n"
            "print('Images:', len(samples))\n"
            "print('Pair classes:', samples['class_name'].nunique())\n"
            "print('Crops:', samples['crop_name'].nunique())\n"
            "samples[['crop_name', 'disease_name']].head()\n"
        ),
        markdown_cell("## Train the model"),
        code_cell(
            "config = JointDiseaseTrainingConfig(\n"
            "    dataset_dirs=dataset_dirs,\n"
            "    backbone='densenet121',\n"
            "    image_size=224,\n"
            "    batch_size=16,\n"
            "    head_epochs=10,\n"
            "    fine_tune_epochs=24,\n"
            "    learning_rate=8e-4,\n"
            "    fine_tune_learning_rate=6e-5,\n"
            "    crop_loss_weight=0.25,\n"
            "    class_loss_weight=1.0,\n"
            "    fine_tune_layers=100,\n"
            ")\n"
            "metrics = train_joint_disease_classifier(config)\n"
            "print(json.dumps({\n"
            "    'pair_accuracy': metrics['pair_accuracy'],\n"
            "    'crop_accuracy': metrics['crop_accuracy'],\n"
            "    'disease_accuracy_from_pair_head': metrics['disease_accuracy_from_pair_head'],\n"
            "    'pair_top3_accuracy': metrics['pair_top3_accuracy'],\n"
            "}, indent=2))\n"
        ),
        markdown_cell("## Review saved outputs"),
        code_cell(
            "report_dir = ROOT / 'reports' / 'disease_joint'\n"
            "list(report_dir.iterdir()) if report_dir.exists() else 'No reports yet'\n"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build()
