from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "data" / "maharashtra_disease_targets.json"
DEFAULT_METADATA_PATH = ROOT / "models" / "disease" / "leaf_disease_metadata.json"
JOINT_METADATA_PATH = ROOT / "models" / "disease" / "leaf_disease_joint_metadata.json"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def split_target_class_name(value: str) -> tuple[str, str]:
    if "___" in value:
        crop_name, disease_name = value.split("___", maxsplit=1)
    else:
        crop_name, disease_name = value, "Healthy"
    return crop_name, disease_name


def crop_name_matches(target_keys: list[str], covered_keys: list[str]) -> bool:
    for target_key in target_keys:
        for covered_key in covered_keys:
            if target_key == covered_key or target_key in covered_key or covered_key in target_key:
                return True
    return False


def pick_metadata_path() -> Path:
    if JOINT_METADATA_PATH.exists():
        return JOINT_METADATA_PATH
    return DEFAULT_METADATA_PATH


def main() -> None:
    targets = load_json(TARGETS_PATH)
    metadata_path = pick_metadata_path()
    metadata = load_json(metadata_path)

    covered_crops = {normalize(item["crop_name"]): item["crop_name"] for item in metadata.get("class_metadata", [])}
    covered_crop_keys = list(covered_crops.keys())
    covered_pairs = {
        (normalize(item["crop_name"]), normalize(item["disease_name"])): item["class_name"]
        for item in metadata.get("class_metadata", [])
    }

    total_crops = 0
    covered_crop_count = 0
    total_target_classes = 0
    covered_target_class_count = 0

    print(f"Coverage audit using metadata: {metadata_path}")
    print(f"Target region: {targets.get('region', 'Unknown')}")
    print()

    for crop in targets.get("priority_crops", []):
        total_crops += 1
        crop_name = crop["crop_name"]
        aliases = crop.get("aliases", [])
        crop_keys = [normalize(crop_name), *(normalize(alias) for alias in aliases)]

        target_classes = crop.get("target_classes", [])
        matched_classes = []
        for target_class in target_classes:
            target_crop, target_disease = split_target_class_name(target_class)
            crop_variants = [normalize(target_crop), *(normalize(alias) for alias in aliases)]
            disease_key = normalize(target_disease.replace("_", " "))
            if any((crop_key, disease_key) in covered_pairs for crop_key in crop_variants):
                matched_classes.append(target_class)

        crop_supported = crop_name_matches(crop_keys, covered_crop_keys) or bool(matched_classes)
        if crop_supported:
            covered_crop_count += 1

        total_target_classes += len(target_classes)
        covered_target_class_count += len(matched_classes)

        print(f"{crop_name}: {'SUPPORTED' if crop_supported else 'MISSING'}")
        print(f"  Priority: {crop.get('priority', 'unknown')}")
        print(f"  Covered target classes: {len(matched_classes)}/{len(target_classes)}")
        if matched_classes:
            print(f"  Matched: {', '.join(matched_classes)}")
        missing = [name for name in target_classes if name not in matched_classes]
        if missing:
            print(f"  Missing: {', '.join(missing)}")
        print()

    crop_coverage = (covered_crop_count / total_crops * 100.0) if total_crops else 0.0
    class_coverage = (covered_target_class_count / total_target_classes * 100.0) if total_target_classes else 0.0

    print("Summary")
    print(f"  Crop coverage: {covered_crop_count}/{total_crops} ({crop_coverage:.1f}%)")
    print(
        f"  Target class coverage: {covered_target_class_count}/{total_target_classes} ({class_coverage:.1f}%)"
    )


if __name__ == "__main__":
    main()
