import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from .rule_engine import blend_predictions, rule_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "merged_clean.csv"
METADATA_PATH = DATA_DIR / "crop_metadata.json"
MODEL_METADATA_PATH = DATA_DIR / "model_metadata.json"


class Predictor:
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or MODELS_DIR / "best_model.pkl"
        self.classes_path = MODELS_DIR / "class_labels.json"
        self.model = None
        self.class_labels = None
        self.lookup_df = None
        self.crop_metadata = None
        self.model_metadata = None

    def _ensure_loaded(self) -> None:
        if self.model is None:
            if not self.model_path.exists() or not self.classes_path.exists():
                raise FileNotFoundError(
                    "Model artifacts are missing. Run training first to create best_model.pkl and class_labels.json."
                )
            self.model = joblib.load(self.model_path)
            with open(self.classes_path, "r", encoding="utf-8") as file:
                self.class_labels = json.load(file)
        
        if self.lookup_df is None:
            if PROCESSED_PATH.exists():
                self.lookup_df = pd.read_csv(PROCESSED_PATH)
        
        if self.crop_metadata is None:
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    self.crop_metadata = json.load(f)
            else:
                self.crop_metadata = {"excluded_crops": [], "crop_durations": {}}

        if self.model_metadata is None:
            if MODEL_METADATA_PATH.exists():
                with open(MODEL_METADATA_PATH, "r") as f:
                    self.model_metadata = json.load(f)

    def get_insights(self) -> Dict:
        self._ensure_loaded()
        return self.model_metadata or {}

    def _build_dataframe(self, features: Dict) -> pd.DataFrame:
        # Standardize feature names to match what the model expects
        return pd.DataFrame([features])

    def predict_distribution(self, features: Dict, apply_rules: bool = True, duration_months: Optional[float] = None) -> Tuple[List[Dict], bool]:
        self._ensure_loaded()
        df = self._build_dataframe(features)
        
        # Get probabilities
        proba = self.model.predict_proba(df)[0]
        rule_used = False

        if apply_rules:
            rules = rule_score(df.iloc[0], self.class_labels)
            proba = blend_predictions(proba, rules)
            rule_used = bool(rules.sum() > 0)

        # Mapping results
        results = []
        excluded = self.crop_metadata.get("excluded_crops", [])
        durations = self.crop_metadata.get("crop_durations", {})

        for i, score in enumerate(proba):
            crop_name = self.class_labels[i]
            
            # 1. Skip excluded crops (Stage 1: Data Cleaning)
            if crop_name in excluded:
                continue
            
            # 2. Duration Filtering (Stage 3: Logic Upgrade)
            duration_match = True
            duration_info = durations.get(crop_name)
            
            if duration_months is not None and duration_info:
                # Add 0.5 month (~15 days) slack as requested
                slack = 0.5
                if duration_months < (duration_info["min"] - slack):
                    duration_match = False
                elif duration_months > (duration_info["max"] + slack * 2): # more slack for longer periods
                    # usually users don't mind if it takes less time, but if they want exactly X months, 
                    # we keep it relative.
                    pass 

            results.append({
                "crop": crop_name,
                "confidence": float(score),
                "duration_match": duration_match,
                "duration_range": f"{duration_info['min']}-{duration_info['max']} months" if duration_info else "Unknown"
            })

        # Sort by confidence
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        
        # Apply strict duration filter if there are matches, otherwise show all with warnings
        if duration_months is not None:
             matches = [r for r in results if r["duration_match"]]
             if matches:
                 # If we have matches, we can still show non-matches at the bottom or filter them
                 # For "Clean Output" we'll just prioritize matches
                 results = matches + [r for r in results if not r["duration_match"]]

        return results, rule_used

    def fertilizer_recommendation(self, crop_label: str, soil_type: str, region: str) -> str:
        self._ensure_loaded()
        if self.lookup_df is None:
            return "Reference data for fertilizer is currently unavailable."
            
        candidates = self.lookup_df[
            (self.lookup_df["crop_label"] == crop_label)
            & (self.lookup_df["soil_type"] == soil_type)
            & (self.lookup_df["fertilizer"].notna())
        ]
        if region and region != "Unknown":
            regional = candidates[candidates["region"] == region]
            if not regional.empty:
                candidates = regional
        if candidates.empty:
            candidates = self.lookup_df[
                (self.lookup_df["crop_label"] == crop_label)
                & (self.lookup_df["fertilizer"].notna())
            ]
        if candidates.empty:
            return "No specific fertilizer recommendation found in our knowledge base."
        return str(candidates["fertilizer"].mode().iat[0])

    def predict_topk(self, features: Dict, k: int = 3, duration_months: Optional[float] = None) -> Dict:
        results, rule_used = self.predict_distribution(features, apply_rules=True, duration_months=duration_months)
        
        top_results = results[:k]
        primary_crop = top_results[0]["crop"] if top_results else "Unknown"
        
        fertilizer = self.fertilizer_recommendation(
            crop_label=primary_crop,
            soil_type=features.get("soil_type", "Other"),
            region=features.get("region", "Unknown"),
        )
        
        return {
            "predictions": top_results,
            "rule_based_override": rule_used,
            "fertilizer": fertilizer,
            "primary_crop": primary_crop
        }
