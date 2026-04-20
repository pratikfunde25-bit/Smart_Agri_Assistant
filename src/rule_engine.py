import numpy as np
import pandas as pd


def rule_score(row: pd.Series, class_labels) -> np.ndarray:
    """Generate a rule-based score vector aligned with model classes."""
    scores = np.zeros(len(class_labels))
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    def add_score(label, val):
        idx = label_to_index.get(label)
        if idx is not None:
            scores[idx] += val

    rainfall = row.get("rainfall", 0)
    humidity = row.get("humidity", 0)
    ph = row.get("pH", 7)
    soil = row.get("soil_type", "").lower()
    region = row.get("region", "").lower()

    # Rain + humidity -> rice
    if rainfall > 150 and humidity > 80:
        add_score("Rice", 0.2)

    # High pH + low rainfall -> wheat
    if ph > 7.5 and rainfall < 120:
        add_score("Wheat", 0.15)

    # Black soil, moderate K -> cotton
    if "black" in soil and row.get("K", 0) < 60:
        add_score("Cotton", 0.15)

    # Konkan + heavy rain -> rice/arecanut preference
    if region == "konkan" and rainfall > 120:
        add_score("Rice", 0.1)

    return scores


def blend_predictions(model_proba: np.ndarray, rule_scores: np.ndarray, alpha: float = 0.8):
    """Weighted blend of model probabilities and rule scores."""
    # Normalize rule scores
    if rule_scores.sum() > 0:
        rule_scores = rule_scores / rule_scores.sum()
    blended = alpha * model_proba + (1 - alpha) * rule_scores
    return blended
