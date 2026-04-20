from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.explainability import make_gradcam_heatmap, save_gradcam_overlay
from src.hybrid_disease_predictor import HybridDiseasePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time crop disease detection from a webcam feed.")
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--features-json",
        default="",
        help="Optional JSON file containing structured crop prediction features.",
    )
    parser.add_argument("--provided-crop", default="", help="Optional crop name if it is already known.")
    parser.add_argument("--frame-stride", type=int, default=6, help="Run inference every N frames.")
    parser.add_argument(
        "--gradcam-dir",
        default=str(Path("reports") / "disease" / "live_gradcam"),
        help="Folder for saved Grad-CAM snapshots when you press G.",
    )
    return parser.parse_args()


def load_crop_features(path: str) -> dict | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def draw_prediction(frame, result: dict | None) -> None:
    if not result:
        cv2.putText(frame, "Scanning leaf...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return

    color = (40, 220, 40) if result["is_healthy"] else (20, 20, 230)
    lines = [
        f"Crop: {result['crop_name']} ({result['crop_confidence']:.2%})",
        f"Disease: {result['disease_name']} ({result['disease_confidence']:.2%})",
        f"Treatment: {result['suggested_treatment'][:75]}",
    ]
    if result.get("fertilizer_recommendation"):
        lines.append(f"Fertilizer Hint: {result['fertilizer_recommendation']}")

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28

    cv2.putText(
        frame,
        "Keys: Q quit | G save Grad-CAM snapshot",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def main() -> None:
    args = parse_args()
    crop_features = load_crop_features(args.features_json)
    predictor = HybridDiseasePredictor().load()
    capture = cv2.VideoCapture(args.camera_id)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera_id}")

    latest_result = None
    latest_frame_rgb = None
    frame_counter = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_counter += 1
            if frame_counter % max(args.frame_stride, 1) == 0:
                latest_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                latest_result = predictor.predict_from_image(
                    latest_frame_rgb,
                    crop_features=crop_features,
                    provided_crop=args.provided_crop or None,
                )

            draw_prediction(frame, latest_result)
            cv2.imshow("Hybrid Crop Disease Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g") and latest_frame_rgb is not None:
                image_batch, image_rgb = predictor.prepare_image(latest_frame_rgb)
                heatmap = make_gradcam_heatmap(predictor.model, image_batch)
                output_name = datetime.now().strftime("gradcam_%Y%m%d_%H%M%S.jpg")
                output_path = Path(args.gradcam_dir) / output_name
                saved_path = save_gradcam_overlay(image_rgb=image_rgb, heatmap=heatmap, output_path=output_path)
                print(f"Saved Grad-CAM snapshot to {saved_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
