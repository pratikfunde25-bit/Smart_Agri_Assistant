from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class DiseaseClassMeta:
    class_name: str
    crop_name: str
    disease_name: str
    is_healthy: bool


def _prettify_label(value: str) -> str:
    cleaned = str(value).replace("_", " ").replace("  ", " ").strip()
    return " ".join(cleaned.split()).title()


def parse_plantvillage_label(class_name: str) -> DiseaseClassMeta:
    if "___" in class_name:
        crop_raw, disease_raw = class_name.split("___", maxsplit=1)
    else:
        crop_raw, disease_raw = class_name, "healthy"

    crop_name = _prettify_label(crop_raw)
    disease_name = _prettify_label(disease_raw)
    is_healthy = disease_name.lower() == "healthy"
    return DiseaseClassMeta(
        class_name=class_name,
        crop_name=crop_name,
        disease_name=disease_name,
        is_healthy=is_healthy,
    )


def build_class_metadata(class_names: Iterable[str]) -> List[DiseaseClassMeta]:
    return [parse_plantvillage_label(name) for name in class_names]


def build_metadata_payload(class_names: Iterable[str]) -> Dict:
    metadata = build_class_metadata(class_names)
    crop_to_classes: Dict[str, List[str]] = {}

    for item in metadata:
        crop_to_classes.setdefault(item.crop_name, []).append(item.class_name)

    return {
        "class_names": list(class_names),
        "class_metadata": [asdict(item) for item in metadata],
        "crop_to_classes": crop_to_classes,
    }


def suggest_treatment(disease_name: str, crop_name: str | None = None) -> str:
    disease = disease_name.lower().strip()

    if disease == "healthy":
        return (
            "Leaf appears healthy. Continue field scouting, keep irrigation balanced, "
            "and maintain preventive nutrition and sanitation."
        )
    if "early blight" in disease or "late blight" in disease or "blight" in disease:
        return (
            "Remove infected leaves, avoid overhead irrigation, improve airflow, and use "
            "a locally recommended fungicide such as copper or mancozeb if symptoms spread."
        )
    if "rust" in disease:
        return (
            "Remove heavily infected foliage, reduce leaf wetness, and apply a labeled "
            "rust fungicide after checking local advisory guidance."
        )
    if "mildew" in disease or "leaf mold" in disease:
        return (
            "Increase ventilation, avoid dense canopy moisture, and use sulfur- or "
            "bicarbonate-based management where appropriate."
        )
    if "bacterial" in disease or "canker" in disease:
        return (
            "Sanitize tools, avoid working in wet fields, remove infected tissue, and "
            "consider copper-based bactericide support if advised locally."
        )
    if "virus" in disease or "mosaic" in disease or "leaf curl" in disease:
        return (
            "Rogue infected plants early, control vectors such as whiteflies and aphids, "
            "and use clean nursery material for the next cycle."
        )
    if "spot" in disease or "scab" in disease or "septoria" in disease:
        return (
            "Remove infected leaves, improve spacing and airflow, and follow a preventive "
            "fungicide schedule if the disease pressure continues."
        )
    if "mite" in disease:
        return (
            "Inspect the underside of leaves, reduce dust stress, and use neem-based or "
            "recommended miticide treatment if infestation increases."
        )
    if "wilt" in disease or "rot" in disease:
        return (
            "Improve drainage, avoid overwatering, remove badly affected plants, and "
            "review soil-borne disease management before replanting."
        )

    crop_hint = f" for {crop_name}" if crop_name else ""
    return (
        f"Apply integrated disease management{crop_hint}: remove infected tissue, "
        "improve field hygiene, and confirm the treatment with a local agronomist or KVK."
    )
