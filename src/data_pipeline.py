from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SEASON_MAP = {
    "W": "Winter",
    "Sp": "Spring",
    "Su": "Summer",
    "Au": "Autumn",
}

REGION_MAP = {
    "kolhapur": "Konkan",
    "pune": "Western Maharashtra",
    "sangli": "Western Maharashtra",
    "satara": "Western Maharashtra",
    "solapur": "Western Maharashtra",
}

SOIL_FIXES = {
    "red": "Red",
    "black": "Black",
    "dark brown": "Black",
    "medium brown": "Red",
    "light brown": "Alluvial",
    "reddish brown": "Red",
    "brown": "Red",
    "gray": "Alluvial",
    "dark gray": "Alluvial",
    "yellowish brown": "Alluvial",
    "dark reddish brown": "Red",
    "other": "Other",
}


def normalize_crop_label(value: str) -> str:
    if pd.isna(value):
        return value
    return " ".join(str(value).strip().split()).title()


def normalize_soil_type(value: str) -> str:
    if pd.isna(value):
        return "Other"
    cleaned = " ".join(str(value).strip().lower().split())
    if cleaned in SOIL_FIXES:
        return SOIL_FIXES[cleaned]
    if "black" in cleaned:
        return "Black"
    if "red" in cleaned:
        return "Red"
    if "alluv" in cleaned:
        return "Alluvial"
    if "brown" in cleaned:
        return "Red"
    return "Other"


def map_region(district: str) -> str:
    if pd.isna(district):
        return "Unknown"
    cleaned = " ".join(str(district).strip().lower().split())
    return REGION_MAP.get(cleaned, "Unknown")


def infer_fertilizer_season(row: pd.Series) -> str:
    rainfall = float(row["rainfall"])
    temperature = float(row["temperature"])
    if rainfall >= 900:
        return "Autumn"
    if rainfall >= 700:
        return "Summer"
    if temperature <= 20:
        return "Winter"
    return "Spring"


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    crop_path = RAW_DIR / "Crop Recommendation using Soil Properties and Weather Prediction (1).csv"
    fert_path = RAW_DIR / "Crop and fertilizer dataset.csv"
    return pd.read_csv(crop_path), pd.read_csv(fert_path)


def melt_crop_by_season(df: pd.DataFrame) -> pd.DataFrame:
    melted_rows = []
    for short, season_name in SEASON_MAP.items():
        max_col = f"T2M_MAX-{short}"
        min_col = f"T2M_MIN-{short}"
        humidity_col = f"QV2M-{short}"
        rainfall_col = f"PRECTOTCORR-{short}"
        if not all(col in df.columns for col in [max_col, min_col, humidity_col, rainfall_col]):
            continue
        season_df = pd.DataFrame(
            {
                "N": df["N"],
                "P": df["P"],
                "K": df["K"],
                "pH": df["Ph"],
                "temperature": (df[max_col] + df[min_col]) / 2.0,
                "humidity": df[humidity_col],
                "rainfall": df[rainfall_col],
                "soil_type": df["Soilcolor"].apply(normalize_soil_type),
                "season": season_name,
                "crop_label": df["label"].apply(normalize_crop_label),
                "region": "Unknown",
                "district": "Unknown",
                "fertilizer": np.nan,
                "source": "crop_recommendation",
            }
        )
        melted_rows.append(season_df)
    if not melted_rows:
        raise ValueError("Seasonal weather columns are missing from the crop recommendation dataset.")
    return pd.concat(melted_rows, ignore_index=True)


def prepare_fertilizer_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "N": df["Nitrogen"],
            "P": df["Phosphorus"],
            "K": df["Potassium"],
            "pH": df["pH"],
            "temperature": df["Temperature"],
            "humidity": np.nan,
            "rainfall": df["Rainfall"],
            "soil_type": df["Soil_color"].apply(normalize_soil_type),
            "crop_label": df["Crop"].apply(normalize_crop_label),
            "district": df["District_Name"].astype(str).str.strip(),
            "fertilizer": df["Fertilizer"].astype(str).str.strip(),
            "source": "fertilizer",
        }
    )
    out["region"] = out["district"].apply(map_region)
    out["season"] = out[["rainfall", "temperature"]].apply(infer_fertilizer_season, axis=1)
    return out


def winsorize_iqr(df: pd.DataFrame, numeric_cols) -> pd.DataFrame:
    capped = df.copy()
    for col in numeric_cols:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        capped[col] = capped[col].clip(lower, upper)
    return capped


def validate_processed(df: pd.DataFrame) -> None:
    numeric_cols = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
    categorical_cols = ["soil_type", "season", "region", "district", "crop_label", "source"]

    if len(df) < 19000:
        raise ValueError(f"Processed dataset has only {len(df)} rows; expected at least 19000.")
    if df["crop_label"].nunique() < 12:
        raise ValueError("Processed dataset retains fewer than 12 crop classes.")
    if df[numeric_cols + categorical_cols].isna().sum().sum() > 0:
        raise ValueError("Processed dataset still has nulls in required columns.")


def preprocess() -> pd.DataFrame:
    crop_raw, fert_raw = load_raw()

    crop_long = melt_crop_by_season(crop_raw)
    fert_std = prepare_fertilizer_df(fert_raw)

    # Only remove exact source duplicates before feature-space transforms.
    crop_long = crop_long.drop_duplicates()
    fert_std = fert_std.drop_duplicates()

    merged = pd.concat([crop_long, fert_std], ignore_index=True)

    numeric_cols = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
    categorical_cols = ["soil_type", "season", "region", "district", "crop_label", "source"]

    for col in numeric_cols:
        merged[col] = merged[col].fillna(merged[col].median())
    for col in categorical_cols:
        merged[col] = merged[col].fillna(merged[col].mode().iat[0])

    merged = winsorize_iqr(merged, numeric_cols)

    # Remove tiny classes that are not robust enough for CV.
    crop_counts = merged["crop_label"].value_counts()
    valid_crops = crop_counts[crop_counts >= 10].index
    merged = merged[merged["crop_label"].isin(valid_crops)].reset_index(drop=True)

    validate_processed(merged)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "merged_clean.csv"
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == "__main__":
    df = preprocess()
    print(f"Processed dataset saved to {PROCESSED_DIR / 'merged_clean.csv'}")
    print(df.shape)
