from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability import make_gradcam_heatmap, save_gradcam_overlay
from src.hybrid_disease_predictor import HybridDiseasePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM explanation for a leaf image.")
    parser.add_argument("--image-path", required=True, help="Path to the leaf image.")
    parser.add_argument(
        "--output-path",
        default=str(Path("reports") / "disease" / "gradcam_overlay.png"),
        help="Where the Grad-CAM overlay should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = HybridDiseasePredictor().load()
    image_batch, image_rgb = predictor.prepare_image(args.image_path)
    heatmap = make_gradcam_heatmap(predictor.model, image_batch)
    output_path = save_gradcam_overlay(image_rgb=image_rgb, heatmap=heatmap, output_path=args.output_path)
    print(f"Grad-CAM saved to {output_path}")


if __name__ == "__main__":
    main()
