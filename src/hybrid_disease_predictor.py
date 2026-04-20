from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image

from .disease_keras import load_disease_model
from .disease_taxonomy import suggest_treatment
from .predictor import Predictor


ROOT = Path(__file__).resolve().parents[1]


class HybridDiseasePredictor:
    def __init__(
        self,
        vision_model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        crop_prior_weight: float = 0.45,
    ) -> None:
        self.vision_model_path = Path(vision_model_path or ROOT / "models" / "disease" / "leaf_disease_classifier.keras")
        self.metadata_path = Path(metadata_path or ROOT / "models" / "disease" / "leaf_disease_metadata.json")
        self.crop_prior_weight = float(np.clip(crop_prior_weight, 0.0, 0.95))
        self.crop_predictor = Predictor()

        self.model: tf.keras.Model | None = None
        self.metadata: Dict[str, Any] | None = None
        self.class_names: List[str] = []
        self.class_metadata: List[Dict[str, Any]] = []
        self.image_size: int = 224

    def load(self) -> "HybridDiseasePredictor":
        if self.model is None:
            if not self.vision_model_path.exists():
                raise FileNotFoundError(
                    f"Vision model not found at {self.vision_model_path}. Train the disease model first."
                )
            backbone = None
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as file:
                    preview_metadata = json.load(file)
                backbone = preview_metadata.get("backbone")
                self.metadata = preview_metadata
            self.model = load_disease_model(self.vision_model_path, backbone=backbone, compile=False)

        if self.metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found at {self.metadata_path}. Train the disease model first."
                )
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        self.class_names = list(self.metadata.get("class_names", []))
        self.class_metadata = list(self.metadata.get("class_metadata", []))
        self.image_size = int(self.metadata.get("image_size", 224))

        if not self.class_metadata:
            raise ValueError(
                f"Disease metadata at {self.metadata_path} does not contain any class definitions."
            )

        return self

    def prepare_image(self, image_source: str | Path | np.ndarray | Image.Image) -> tuple[np.ndarray, np.ndarray]:
        self.load()

        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source).convert("RGB")
            image_rgb = np.array(image)
        elif isinstance(image_source, Image.Image):
            image_rgb = np.array(image_source.convert("RGB"))
        else:
            image_rgb = np.asarray(image_source)
            if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
                raise ValueError("Expected an RGB image array with shape [H, W, 3].")
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        resized = Image.fromarray(image_rgb).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        batch = np.asarray(resized, dtype=np.float32) / 255.0
        batch = np.expand_dims(batch, axis=0)
        return batch, image_rgb

    def _crop_prior_from_features(self, crop_features: Dict[str, Any] | None) -> tuple[Dict[str, float], bool, str | None]:
        if not crop_features:
            return {}, False, None

        ranked, rule_used = self.crop_predictor.predict_distribution(crop_features, apply_rules=True)
        fertilizer = self.crop_predictor.fertilizer_recommendation(
            crop_label=ranked[0][0],
            soil_type=crop_features.get("soil_type", "Other"),
            region=crop_features.get("region", "Unknown"),
        )
        return {crop: score for crop, score in ranked}, rule_used, fertilizer

    def _build_crop_mask(
        self,
        provided_crop: str | None,
        crop_prior: Dict[str, float],
    ) -> np.ndarray:
        if provided_crop:
            provided = provided_crop.strip().lower()
            return np.array(
                [1.0 if item["crop_name"].lower() == provided else 1e-3 for item in self.class_metadata],
                dtype=np.float32,
            )

        if crop_prior:
            return np.array(
                [max(float(crop_prior.get(item["crop_name"], 0.0)), 1e-6) for item in self.class_metadata],
                dtype=np.float32,
            )

        return np.ones(len(self.class_metadata), dtype=np.float32)

    def _blend_probabilities(self, image_probs: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        reweighted = image_probs * ((1.0 - self.crop_prior_weight) + self.crop_prior_weight * crop_mask)
        total = reweighted.sum()
        if total <= 0:
            return image_probs
        return reweighted / total

    def _aggregate_crop_scores(self, class_probs: np.ndarray) -> Dict[str, float]:
        crop_scores: Dict[str, float] = {}
        for probability, meta in zip(class_probs, self.class_metadata):
            crop_scores[meta["crop_name"]] = crop_scores.get(meta["crop_name"], 0.0) + float(probability)
        return dict(sorted(crop_scores.items(), key=lambda item: item[1], reverse=True))

    def predict_from_image(
        self,
        image_source: str | Path | np.ndarray | Image.Image,
        crop_features: Dict[str, Any] | None = None,
        provided_crop: str | None = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        self.load()
        image_batch, _ = self.prepare_image(image_source)
        raw_probs = self.model.predict(image_batch, verbose=0)[0]

        crop_prior, rule_used, fertilizer = self._crop_prior_from_features(crop_features)
        crop_mask = self._build_crop_mask(provided_crop=provided_crop, crop_prior=crop_prior)
        hybrid_probs = self._blend_probabilities(raw_probs, crop_mask)
        crop_scores = self._aggregate_crop_scores(hybrid_probs)

        top_indices = np.argsort(hybrid_probs)[::-1][:top_k]
        top_predictions = []
        for index in top_indices:
            meta = self.class_metadata[int(index)]
            top_predictions.append(
                {
                    "class_name": meta["class_name"],
                    "crop_name": meta["crop_name"],
                    "disease_name": meta["disease_name"],
                    "is_healthy": bool(meta["is_healthy"]),
                    "confidence": float(hybrid_probs[index]),
                    "suggested_treatment": suggest_treatment(meta["disease_name"], crop_name=meta["crop_name"]),
                }
            )

        best = top_predictions[0]
        return {
            "crop_name": best["crop_name"],
            "crop_confidence": float(crop_scores.get(best["crop_name"], best["confidence"])),
            "disease_name": best["disease_name"],
            "disease_confidence": float(best["confidence"]),
            "is_healthy": bool(best["is_healthy"]),
            "suggested_treatment": best["suggested_treatment"],
            "fertilizer_recommendation": fertilizer,
            "rule_based_crop_hint_used": rule_used,
            "crop_prior": crop_scores,
            "top_predictions": top_predictions,
        }
