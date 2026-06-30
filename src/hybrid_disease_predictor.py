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
        crop_prior_weight: float = 0.3,
    ) -> None:
        default_model_path = ROOT / "models" / "disease" / "leaf_disease_joint_classifier.keras"
        default_metadata_path = ROOT / "models" / "disease" / "leaf_disease_joint_metadata.json"
        legacy_model_path = ROOT / "models" / "disease" / "leaf_disease_classifier.keras"
        legacy_metadata_path = ROOT / "models" / "disease" / "leaf_disease_metadata.json"
        joint_ready = default_model_path.exists() and default_metadata_path.exists()

        self.vision_model_path = Path(
            vision_model_path
            or (default_model_path if joint_ready else legacy_model_path)
        )
        self.metadata_path = Path(
            metadata_path
            or (default_metadata_path if joint_ready else legacy_metadata_path)
        )
        self.crop_prior_weight = float(np.clip(crop_prior_weight, 0.0, 0.95))
        self.crop_predictor = Predictor()

        self.model: tf.keras.Model | None = None
        self.metadata: Dict[str, Any] | None = None
        self.class_names: List[str] = []
        self.class_metadata: List[Dict[str, Any]] = []
        self.crop_names: List[str] = []
        self.image_size: int = 224
        self.minimum_confidence_for_auto_accept = 0.45
        self.minimum_margin_for_auto_accept = 0.12

    def _fallback_legacy_paths(self) -> tuple[Path, Path]:
        return (
            ROOT / "models" / "disease" / "leaf_disease_classifier.keras",
            ROOT / "models" / "disease" / "leaf_disease_metadata.json",
        )

    def load(self) -> "HybridDiseasePredictor":
        if self.model is None:
            candidate_model_path = self.vision_model_path
            candidate_metadata_path = self.metadata_path
            try:
                if not candidate_model_path.exists():
                    raise FileNotFoundError(
                        f"Vision model not found at {candidate_model_path}. Train the disease model first."
                    )
                backbone = None
                if candidate_metadata_path.exists():
                    with open(candidate_metadata_path, "r", encoding="utf-8") as file:
                        preview_metadata = json.load(file)
                    backbone = preview_metadata.get("backbone")
                    self.metadata = preview_metadata
                self.model = load_disease_model(candidate_model_path, backbone=backbone, compile=False)
            except Exception:
                legacy_model_path, legacy_metadata_path = self._fallback_legacy_paths()
                if candidate_model_path == legacy_model_path:
                    raise
                if not legacy_model_path.exists() or not legacy_metadata_path.exists():
                    raise
                self.vision_model_path = legacy_model_path
                self.metadata_path = legacy_metadata_path
                self.metadata = None
                with open(legacy_metadata_path, "r", encoding="utf-8") as file:
                    preview_metadata = json.load(file)
                backbone = preview_metadata.get("backbone")
                self.metadata = preview_metadata
                self.model = load_disease_model(legacy_model_path, backbone=backbone, compile=False)

        if self.metadata is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found at {self.metadata_path}. Train the disease model first."
                )
            with open(self.metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        self.class_names = list(self.metadata.get("class_names", []))
        self.class_metadata = list(self.metadata.get("class_metadata", []))
        self.crop_names = list(self.metadata.get("crop_names", []))
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

    def _predict_model_outputs(self, image_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        raw_outputs = self.model.predict(image_batch, verbose=0)
        
        # Handle multiple outputs (list or tuple)
        if isinstance(raw_outputs, (list, tuple)):
            # If it's a list/tuple, it means we have multiple output branches
            class_probs = np.asarray(raw_outputs[0], dtype=np.float32)
            crop_probs = np.asarray(raw_outputs[1], dtype=np.float32) if len(raw_outputs) > 1 else None
        else:
            # Single output model
            class_probs = np.asarray(raw_outputs, dtype=np.float32)
            crop_probs = None
            
        # Squeeze ALL singleton dimensions to get 1D vectors (e.g., [1, N] or [1, 1, N] -> [N])
        # We only squeeze if it's not a single scalar (ndim > 0)
        if class_probs.ndim > 1:
            class_probs = np.squeeze(class_probs)
        if crop_probs is not None and crop_probs.ndim > 1:
            crop_probs = np.squeeze(crop_probs)
            
        # Final safety check: if it's still not 1D (e.g. more than one sample in batch somehow), 
        # force take the first sample's flattened vector
        if class_probs.ndim > 1:
            class_probs = class_probs.reshape(-1)
        if crop_probs is not None and crop_probs.ndim > 1:
            crop_probs = crop_probs.reshape(-1)
            
        return class_probs, crop_probs

    def _crop_prior_from_features(self, crop_features: Dict[str, Any] | None) -> tuple[Dict[str, float], bool, str | None]:
        if not crop_features:
            return {}, False, None

        ranked, rule_used = self.crop_predictor.predict_distribution(crop_features, apply_rules=True)
        if not ranked:
            return {}, rule_used, None

        primary_crop = ranked[0]["crop"]
        fertilizer = self.crop_predictor.fertilizer_recommendation(
            crop_label=primary_crop,
            soil_type=crop_features.get("soil_type", "Other"),
            region=crop_features.get("region", "Unknown"),
        )
        return {item["crop"]: float(item["confidence"]) for item in ranked}, rule_used, fertilizer

    def _build_crop_mask(
        self,
        provided_crop: str | None,
        crop_prior: Dict[str, float],
    ) -> tuple[np.ndarray, bool]:
        if provided_crop:
            provided = provided_crop.strip().lower()
            mask = np.array(
                [1.0 if item["crop_name"].lower() == provided else 1e-3 for item in self.class_metadata],
                dtype=np.float32,
            )
            return mask, bool(np.any(mask > 0.5))

        if crop_prior:
            return np.array(
                [max(float(crop_prior.get(item["crop_name"], 0.0)), 1e-6) for item in self.class_metadata],
                dtype=np.float32,
            ), True

        return np.ones(len(self.class_metadata), dtype=np.float32), True

    def _to_scalar(self, value: Any) -> float:
        """Extremely robust conversion of any array/tensor to a Python float."""
        if value is None:
            return 0.0
        try:
            if hasattr(value, "numpy"):
                value = value.numpy()
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return 0.0
                return float(value.reshape(-1)[0])
            return float(value)
        except (TypeError, ValueError, IndexError):
            return 0.0

    def _blend_probabilities(self, image_probs: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        reweighted = image_probs * ((1.0 - self.crop_prior_weight) + self.crop_prior_weight * crop_mask)
        total = reweighted.sum()
        if total <= 0:
            return image_probs
        return reweighted / total

    def _aggregate_crop_scores(self, class_probs: np.ndarray) -> Dict[str, float]:
        crop_scores: Dict[str, float] = {}
        for probability, meta in zip(class_probs, self.class_metadata):
            crop_scores[meta["crop_name"]] = crop_scores.get(meta["crop_name"], 0.0) + self._to_scalar(probability)
        return dict(sorted(crop_scores.items(), key=lambda item: item[1], reverse=True))

    def _blend_crop_scores(
        self,
        pair_crop_scores: Dict[str, float],
        crop_probs: np.ndarray | None,
    ) -> Dict[str, float]:
        if crop_probs is None or not self.crop_names:
            return pair_crop_scores

        combined: Dict[str, float] = {}
        for crop_name, score in pair_crop_scores.items():
            combined[crop_name] = combined.get(crop_name, 0.0) + self._to_scalar(score) * 0.7
        for crop_name, score in zip(self.crop_names, crop_probs, strict=False):
            combined[crop_name] = combined.get(crop_name, 0.0) + self._to_scalar(score) * 0.3

        total = sum(combined.values())
        if total > 0:
            combined = {key: value / total for key, value in combined.items()}
        return dict(sorted(combined.items(), key=lambda item: item[1], reverse=True))

    def _apply_crop_consistency(
        self,
        class_probs: np.ndarray,
        crop_probs: np.ndarray | None,
    ) -> np.ndarray:
        if crop_probs is None or not self.crop_names:
            return class_probs

        crop_lookup = {crop_name: self._to_scalar(prob) for crop_name, prob in zip(self.crop_names, crop_probs, strict=False)}
        adjusted = np.array(
            [
                self._to_scalar(probability) * (0.55 + 0.45 * crop_lookup.get(meta["crop_name"], 0.0))
                for probability, meta in zip(class_probs, self.class_metadata, strict=False)
            ],
            dtype=np.float32,
        )
        total = adjusted.sum()
        if total <= 0:
            return class_probs
        return adjusted / total

    def _prediction_review_signal(
        self,
        probabilities: np.ndarray,
        crop_scores: Dict[str, float],
    ) -> tuple[bool, str | None]:
        top_two = np.sort(probabilities)[::-1][:2]
        top_confidence = self._to_scalar(top_two[0]) if len(top_two) else 0.0
        margin = self._to_scalar(top_two[0] - top_two[1]) if len(top_two) > 1 else top_confidence
        best_crop_confidence = self._to_scalar(next(iter(crop_scores.values()), 0.0))

        if top_confidence < self.minimum_confidence_for_auto_accept:
            return True, "Low disease confidence. Capture a clearer leaf image in natural light."
        if margin < self.minimum_margin_for_auto_accept:
            return True, "The top disease candidates are very close. Manual review is recommended."
        if best_crop_confidence < 0.50:
            return True, "The detected crop is uncertain, so the disease label may also be unreliable."
        return False, None

    def _is_likely_leaf_by_color(self, image_rgb: np.ndarray) -> tuple[bool, float]:
        """Heuristic check for leaf-like color distribution (Green/Yellow/Brown)."""
        try:
            # Resize for performance
            img = Image.fromarray(image_rgb).resize((64, 64), Image.Resampling.NEAREST)
            arr = np.array(img).astype(np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            
            # Green: G is dominant
            is_green = (g > r * 1.05) & (g > b * 1.1) & (g > 30)
            
            # Yellow/Brown: R and G are high, B is low
            is_yellow_brown = (r > b * 1.2) & (g > b * 1.1) & (np.abs(r - g) < 60) & (r > 50)
            
            # Flesh tones: R > G > B and significant R-G difference (to catch people)
            is_flesh = (r > g + 15) & (g > b + 10) & (r > 80) & (r < 240)
            
            leaf_pixels = np.sum(is_green | is_yellow_brown)
            flesh_pixels = np.sum(is_flesh)
            total = 64 * 64
            
            leaf_ratio = leaf_pixels / total
            flesh_ratio = flesh_pixels / total
            
            # If it's mostly flesh tones and very little leaf colors, it's likely a person
            if flesh_ratio > 0.35 and leaf_ratio < 0.15:
                return False, leaf_ratio
                
            return (leaf_ratio > 0.08), leaf_ratio
        except Exception:
            return True, 1.0

    def predict_from_image(
        self,
        image_source: str | Path | np.ndarray | Image.Image,
        crop_features: Dict[str, Any] | None = None,
        provided_crop: str | None = None,
        top_k: int = 3,
        validate_leaf: bool = True,
    ) -> Dict[str, Any]:
        self.load()
        image_batch, image_rgb = self.prepare_image(image_source)
        
        if validate_leaf:
            is_leaf, leaf_ratio = self._is_likely_leaf_by_color(image_rgb)
            if not is_leaf:
                raise ValueError(
                    "The uploaded image does not appear to be a crop leaf. "
                    "Please upload a clear photo of a single leaf against a plain background."
                )

        raw_probs, crop_head_probs = self._predict_model_outputs(image_batch)
        raw_probs = self._apply_crop_consistency(raw_probs, crop_head_probs)

        # Confidence validation for OOD
        top_conf = self._to_scalar(np.max(raw_probs))
        if validate_leaf and top_conf < 0.22 and leaf_ratio < 0.2:
             raise ValueError(
                 "Uncertain image content. This image does not strongly match any supported crop leaves. "
                 "Please ensure the leaf is well-lit and fills most of the frame."
             )

        crop_prior, rule_used, fertilizer = self._crop_prior_from_features(crop_features)
        crop_mask, crop_hint_supported = self._build_crop_mask(provided_crop=provided_crop, crop_prior=crop_prior)
        hybrid_probs = self._blend_probabilities(raw_probs, crop_mask)
        crop_scores = self._blend_crop_scores(self._aggregate_crop_scores(hybrid_probs), crop_head_probs)
        review_required, review_reason = self._prediction_review_signal(hybrid_probs, crop_scores)
        if provided_crop and not crop_hint_supported:
            review_required = True
            review_reason = (
                f"The provided crop '{provided_crop}' is not covered by the current disease model. "
                "Use one of the supported crops or retrain with additional datasets."
            )

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
                    "confidence": self._to_scalar(hybrid_probs[index]),
                    "suggested_treatment": suggest_treatment(meta["disease_name"], crop_name=meta["crop_name"]),
                }
            )

        best = top_predictions[0]
        best_crop_confidence = self._to_scalar(crop_scores.get(best["crop_name"], best["confidence"]))
        return {
            "crop_name": best["crop_name"],
            "crop_confidence": best_crop_confidence,
            "disease_name": best["disease_name"],
            "disease_confidence": self._to_scalar(best["confidence"]),
            "is_healthy": bool(best["is_healthy"]),
            "suggested_treatment": best["suggested_treatment"],
            "fertilizer_recommendation": fertilizer,
            "rule_based_crop_hint_used": rule_used,
            "crop_prior": crop_scores,
            "top_predictions": top_predictions,
            "review_required": review_required,
            "review_reason": review_reason,
            "supported_crops": self.crop_names,
        }
