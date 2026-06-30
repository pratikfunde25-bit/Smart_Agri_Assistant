from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import os
import uuid
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from src.predictor import Predictor
from src.weather_service import OpenWeatherService, WeatherServiceError, weather_lookup_available
from src.crop_advisor import CropAdvisor
from src.dynamic_engine_v3 import DynamicAdvisorEngine


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
REPORTS_DIR = ROOT / "reports"
VIS_DIR = ROOT / "visualizations"
MODELS_DIR = ROOT / "models"
DISEASE_REPORTS_DIR = REPORTS_DIR / "disease"
JOINT_DISEASE_REPORTS_DIR = REPORTS_DIR / "disease_joint"

GENERATED_DIR = APP_DIR / "static" / "generated"
UPLOADS_DIR = GENERATED_DIR / "uploads"
HEATMAPS_DIR = GENERATED_DIR / "heatmaps"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SOIL_OPTIONS = ["Black", "Red", "Alluvial", "Other"]
SEASON_OPTIONS = ["Winter", "Spring", "Summer", "Autumn"]
REGION_OPTIONS = ["Konkan", "Western Maharashtra", "Vidarbha", "Unknown"]

DEFAULT_FORM_VALUES = {
    "N": "",
    "P": "",
    "K": "",
    "pH": "",
    "temperature": "",
    "humidity": "",
    "rainfall": "",
    "soil_type": "Black",
    "season": "Summer",
    "region": "Unknown",
}

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "smart-agri-assistant-dev")

crop_predictor = Predictor()
crop_advisor = CropAdvisor()
advisor_v2 = DynamicAdvisorEngine()
_hybrid_predictor = None
_hybrid_predictor_error = None
_gradcam_tools = None
_gradcam_error = None


def ensure_generated_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)


def has_tensorflow_runtime() -> bool:
    return find_spec("tensorflow") is not None


def disease_model_files_exist() -> bool:
    joint_ready = (MODELS_DIR / "disease" / "leaf_disease_joint_classifier.keras").exists() and (
        MODELS_DIR / "disease" / "leaf_disease_joint_metadata.json"
    ).exists()
    legacy_ready = (MODELS_DIR / "disease" / "leaf_disease_classifier.keras").exists() and (
        MODELS_DIR / "disease" / "leaf_disease_metadata.json"
    ).exists()
    return joint_ready or legacy_ready


def get_disease_artifact_status() -> Dict[str, Any]:
    joint_model = MODELS_DIR / "disease" / "leaf_disease_joint_classifier.keras"
    joint_metadata = MODELS_DIR / "disease" / "leaf_disease_joint_metadata.json"
    legacy_model = MODELS_DIR / "disease" / "leaf_disease_classifier.keras"
    legacy_metadata = MODELS_DIR / "disease" / "leaf_disease_metadata.json"

    if joint_model.exists():
        status = {
            "active_model": "joint",
            "model_path": str(joint_model),
            "metadata_path": str(joint_metadata),
            "ready": joint_metadata.exists(),
            "metadata_missing": not joint_metadata.exists(),
            "display_name": "Joint crop+disease model",
        }
        if joint_metadata.exists():
            status.update(load_json_file(joint_metadata))
        return status

    if legacy_model.exists():
        status = {
            "active_model": "legacy",
            "model_path": str(legacy_model),
            "metadata_path": str(legacy_metadata),
            "ready": legacy_metadata.exists(),
            "metadata_missing": not legacy_metadata.exists(),
            "display_name": "Legacy disease-only model",
        }
        if legacy_metadata.exists():
            status.update(load_json_file(legacy_metadata))
        return status

    return {
        "active_model": None,
        "ready": False,
        "metadata_missing": False,
        "display_name": "No disease model saved yet",
    }


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_crop_comparison() -> list[Dict[str, Any]]:
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_dashboard_payload() -> Dict[str, Any]:
    crop_metrics = load_json_file(REPORTS_DIR / "metrics.json")
    disease_metrics = load_json_file(JOINT_DISEASE_REPORTS_DIR / "evaluation_metrics.json")
    if not disease_metrics:
        disease_metrics = load_json_file(DISEASE_REPORTS_DIR / "evaluation_metrics.json")
    disease_status = get_disease_artifact_status()

    return {
        "crop_metrics": crop_metrics,
        "crop_comparison": load_crop_comparison(),
        "disease_metrics": disease_metrics,
        "disease_status": disease_status,
        "weather_ready": weather_lookup_available(),
        "crop_ready": (MODELS_DIR / "best_model.pkl").exists() and (MODELS_DIR / "class_labels.json").exists(),
        "disease_ready": disease_status.get("ready", False) and has_tensorflow_runtime(),
        "disease_runtime_note": None
        if has_tensorflow_runtime()
        else "TensorFlow is not installed in this environment yet, so disease inference routes will stay unavailable.",
    }


def get_hybrid_predictor():
    global _hybrid_predictor, _hybrid_predictor_error

    if _hybrid_predictor is not None:
        return _hybrid_predictor
    if _hybrid_predictor_error is not None:
        raise RuntimeError(_hybrid_predictor_error)

    try:
        from src.hybrid_disease_predictor import HybridDiseasePredictor

        _hybrid_predictor = HybridDiseasePredictor().load()
        return _hybrid_predictor
    except Exception as exc:
        _hybrid_predictor_error = str(exc)
        raise RuntimeError(_hybrid_predictor_error) from exc


def get_gradcam_tools():
    global _gradcam_tools, _gradcam_error

    if _gradcam_tools is not None:
        return _gradcam_tools
    if _gradcam_error is not None:
        raise RuntimeError(_gradcam_error)

    try:
        from src.explainability import make_gradcam_heatmap, save_gradcam_overlay

        _gradcam_tools = (make_gradcam_heatmap, save_gradcam_overlay)
        return _gradcam_tools
    except Exception as exc:
        _gradcam_error = str(exc)
        raise RuntimeError(_gradcam_error) from exc


def payload_from_request() -> Dict[str, Any]:
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


def parse_float(payload: Dict[str, Any], key: str, label: str) -> float:
    raw_value = payload.get(key, "")
    if raw_value in (None, ""):
        raise ValueError(f"{label} is required.")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def parse_crop_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "N": parse_float(payload, "N", "Nitrogen"),
        "P": parse_float(payload, "P", "Phosphorus"),
        "K": parse_float(payload, "K", "Potassium"),
        "pH": parse_float(payload, "pH", "Soil pH"),
        "temperature": parse_float(payload, "temperature", "Temperature"),
        "humidity": parse_float(payload, "humidity", "Humidity"),
        "rainfall": parse_float(payload, "rainfall", "Rainfall"),
        "soil_type": str(payload.get("soil_type") or "Other"),
        "season": str(payload.get("season") or "Summer"),
        "region": str(payload.get("region") or "Unknown"),
    }


def allowed_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def open_image_bytes(raw_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Please upload a valid JPG, PNG, or WEBP leaf image.") from exc


def extract_image_from_request() -> tuple[Image.Image, str]:
    uploaded = request.files.get("leaf_image")
    if uploaded and uploaded.filename:
        filename = secure_filename(uploaded.filename)
        if not allowed_extension(filename):
            raise ValueError("Unsupported file type. Please upload JPG, PNG, or WEBP.")
        return open_image_bytes(uploaded.read()), filename

    payload = payload_from_request()
    data_url = payload.get("captured_image_data", "")
    if data_url:
        if "," not in data_url:
            raise ValueError("Captured image data is malformed.")
        header, encoded = data_url.split(",", 1)
        extension = ".png" if "png" in header.lower() else ".jpg"
        try:
            return open_image_bytes(base64.b64decode(encoded)), f"camera_capture{extension}"
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Unable to decode captured camera image.") from exc

    raise ValueError("Please upload a leaf image or capture one with the webcam.")


def save_preview_image(image: Image.Image, filename_hint: str) -> str:
    ensure_generated_dirs()
    stem = secure_filename(Path(filename_hint).stem) or "leaf"
    filename = f"{stem}_{uuid.uuid4().hex[:12]}.jpg"
    destination = UPLOADS_DIR / filename
    image.save(destination, format="JPEG", quality=92)
    return url_for("static", filename=f"generated/uploads/{filename}")


def save_heatmap_image(image: Image.Image, filename_hint: str) -> str:
    predictor = get_hybrid_predictor()
    make_gradcam_heatmap, save_gradcam_overlay = get_gradcam_tools()

    ensure_generated_dirs()
    image_batch, image_rgb = predictor.prepare_image(image)
    heatmap = make_gradcam_heatmap(predictor.model, image_batch)

    stem = secure_filename(Path(filename_hint).stem) or "leaf"
    filename = f"{stem}_{uuid.uuid4().hex[:12]}_gradcam.jpg"
    destination = HEATMAPS_DIR / filename
    save_gradcam_overlay(image_rgb=image_rgb, heatmap=heatmap, output_path=destination)
    return url_for("static", filename=f"generated/heatmaps/{filename}")


def serialize_crop_predictions(predictions: list[tuple[str, float]]) -> list[Dict[str, Any]]:
    return [
        {
            "label": label,
            "score": float(score),
            "confidence_pct": round(float(score) * 100.0, 2),
        }
        for label, score in predictions
    ]


def build_field_tips(result: Dict[str, Any], humidity: float | None = None, rainfall: float | None = None) -> list[str]:
    disease_name = str(result.get("disease_name", "Unknown"))
    is_healthy = bool(result.get("is_healthy", False))
    tips = []

    if is_healthy:
        tips.append("The leaf looks healthy, but continue weekly scouting to catch symptoms early.")
    else:
        tips.append(f"Isolate heavily affected leaves and monitor nearby plants for {disease_name} spread.")

    if result.get("review_required"):
        tips.append(str(result.get("review_reason") or "This prediction needs manual review before treatment action."))

    if humidity is not None and humidity >= 80:
        tips.append("High humidity can accelerate fungal disease pressure, so improve spacing and airflow if possible.")

    if rainfall is not None and rainfall > 0:
        tips.append("Recent rainfall increases leaf wetness. Avoid overhead irrigation until the canopy dries out.")

    fertilizer_note = result.get("fertilizer_recommendation")
    if fertilizer_note:
        tips.append(f"Crop support hint: {fertilizer_note}")

    if len(tips) < 3:
        tips.append("Capture images under natural light and a plain background for more stable predictions.")

    return tips[:4]


def json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.errorhandler(413)
def file_too_large(_error):
    return json_error("Uploaded file is too large. Please keep images under 10 MB.", status=413)


@app.route("/", methods=["GET"])
def home():
    dashboard = load_dashboard_payload()
    return render_template("home.html", active_page="home", dashboard=dashboard)


@app.route("/crop", methods=["GET"])
def crop_page():
    return render_template(
        "crop.html",
        active_page="crop",
        form_defaults=DEFAULT_FORM_VALUES,
        soil_options=SOIL_OPTIONS,
        season_options=SEASON_OPTIONS,
        region_options=REGION_OPTIONS,
        weather_ready=weather_lookup_available(),
    )


@app.route("/disease", methods=["GET"])
def disease_page():
    dashboard = load_dashboard_payload()
    return render_template("disease.html", active_page="disease", dashboard=dashboard)


@app.route("/advisor", methods=["GET"])
def advisor_page():
    supported_crops = crop_advisor.get_supported_crops_metadata()
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Filter out excluded crops from advisor selection too
    # Assuming crop_advisor has access to the same metadata or we pass it
    # For now, we'll let it be, but Stage 1 suggests consistency.
    
    return render_template(
        "crop_advisor.html", 
        active_page="advisor", 
        supported_crops=advisor_v2.get_crops(),
        current_date=current_date
    )

@app.route("/api/insights")
def api_insights():
    return jsonify(crop_predictor.get_insights())


@app.route("/get_crop_guidance", methods=["POST"])
def get_crop_guidance():
    payload = request.get_json(silent=True) or {}
    crop = payload.get("crop")
    sowing_date = payload.get("sowing_date")
    stage_idx = payload.get("stage_idx")
    
    if not crop:
        return json_error("Crop selection is required.")
        
    result = crop_advisor.get_guidance(crop, sowing_date, stage_idx)
    return jsonify(result)

@app.route("/api/advisor/query", methods=["POST"])
def advisor_query():
    payload = request.get_json(silent=True) or {}
    crop = payload.get("crop")
    sowing_date = payload.get("sowing_date")
    manual_day = payload.get("manual_day")
    lat = payload.get("latitude")
    lon = payload.get("longitude")
    lang = payload.get("lang", "en")
    
    if not crop:
        return json_error("Crop name is required.")
        
    # Ensure manual_day is an integer if provided
    try:
        if manual_day is not None:
            manual_day = int(manual_day)
    except (ValueError, TypeError):
        manual_day = None

    result = advisor_v2.query(
        crop_name=crop,
        sowing_date=sowing_date,
        manual_day=manual_day,
        lat=lat,
        lon=lon,
        lang=lang
    )
    return jsonify(result)

@app.route("/api/advisor/timeline", methods=["POST"])
def advisor_timeline():
    payload = request.get_json(silent=True) or {}
    crop = payload.get("crop")
    if not crop:
        return json_error("Crop name is required.")
    return jsonify(advisor_v2.get_timeline(crop))


@app.route("/hybrid", methods=["GET"])
def hybrid_page():
    dashboard = load_dashboard_payload()
    return render_template(
        "hybrid.html",
        active_page="hybrid",
        form_defaults=DEFAULT_FORM_VALUES,
        soil_options=SOIL_OPTIONS,
        season_options=SEASON_OPTIONS,
        region_options=REGION_OPTIONS,
        weather_ready=weather_lookup_available(),
        dashboard=dashboard,
    )


@app.route("/insights", methods=["GET"])
def insights():
    dashboard = load_dashboard_payload()
    visuals = [
        "crop_distribution.png",
        "correlation_heatmap.png",
        "regional_patterns.png",
        "feature_importance.png",
        "confusion_matrix.png",
    ]
    available_visuals = [name for name in visuals if (VIS_DIR / name).exists()]
    return render_template("insights.html", active_page="insights", dashboard=dashboard, visuals=available_visuals)


@app.route("/get_weather", methods=["POST"])
def get_weather():
    payload = request.get_json(silent=True) or {}
    try:
        latitude = parse_float(payload, "latitude", "Latitude")
        longitude = parse_float(payload, "longitude", "Longitude")
        service = OpenWeatherService(api_key=os.getenv("OPENWEATHER_API_KEY", ""))
        snapshot = service.fetch_current_weather(latitude=latitude, longitude=longitude)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except WeatherServiceError as exc:
        return json_error(str(exc), status=503)

    return jsonify(
        {
            "weather": snapshot.to_dict(),
            "autofill": {
                "temperature": snapshot.temperature,
                "humidity": snapshot.humidity,
                "rainfall": snapshot.rainfall,
            },
        }
    )


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    payload = payload_from_request()
    duration = payload.get("duration")
    duration_months = float(duration) if duration and str(duration).strip() else None

    try:
        features = parse_crop_features(payload)
        result = crop_predictor.predict_topk(features, k=5, duration_months=duration_months)
        
        top_predictions = result["predictions"]
        rule_used = result["rule_based_override"]
        fertilizer = result["fertilizer"]
    except FileNotFoundError as exc:
        return json_error(str(exc), status=503)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except Exception as exc:
        return json_error(f"Crop prediction failed: {exc}", status=500)

    return jsonify(
        {
            "predicted_crop": result["primary_crop"],
            "confidence": round(float(top_predictions[0]["confidence"]) * 100.0, 2) if top_predictions else 0,
            "predictions": top_predictions,
            "rule_used": rule_used,
            "fertilizer_recommendation": fertilizer,
            "inputs": features,
            "duration_input": duration_months
        }
    )


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    payload = payload_from_request()
    provided_crop = str(payload.get("provided_crop") or "").strip() or None

    try:
        image, filename = extract_image_from_request()
        predictor = get_hybrid_predictor()
        result = predictor.predict_from_image(image, provided_crop=provided_crop)
        preview_url = save_preview_image(image, filename)
        heatmap_url = save_heatmap_image(image, filename)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except RuntimeError as exc:
        return json_error(f"Disease inference is unavailable: {exc}", status=503)
    except Exception as exc:
        return json_error(f"Disease prediction failed: {exc}", status=500)

    result["preview_url"] = preview_url
    result["heatmap_url"] = heatmap_url
    result["tips"] = build_field_tips(result)
    return jsonify(result)


@app.route("/hybrid_predict", methods=["POST"])
def hybrid_predict():
    payload = payload_from_request()
    provided_crop = str(payload.get("provided_crop") or "").strip() or None
    duration = payload.get("duration")
    duration_months = float(duration) if duration and str(duration).strip() else None

    try:
        features = parse_crop_features(payload)
        image, filename = extract_image_from_request()
        
        # 1. Structured crop prediction with duration constraint
        crop_result = crop_predictor.predict_topk(features, k=3, duration_months=duration_months)
        
        # 2. Image-based disease detection
        predictor_hybrid = get_hybrid_predictor()
        disease_result = predictor_hybrid.predict_from_image(
            image,
            crop_features=features,
            provided_crop=provided_crop,
        )
        disease_result["tips"] = build_field_tips(
            disease_result,
            humidity=features.get("humidity"),
            rainfall=features.get("rainfall"),
        )
        
        preview_url = save_preview_image(image, filename)
        heatmap_url = save_heatmap_image(image, filename)
    except ValueError as exc:
        return json_error(str(exc), status=400)
    except RuntimeError as exc:
        return json_error(f"Hybrid inference is unavailable: {exc}", status=503)
    except FileNotFoundError as exc:
        return json_error(str(exc), status=503)
    except Exception as exc:
        return json_error(f"Hybrid prediction failed: {exc}", status=500)

    return jsonify({
        "disease": disease_result,
        "crop": crop_result,
        "preview_url": preview_url,
        "heatmap_url": heatmap_url,
        "inputs": features,
        "duration_input": duration_months
    })

@app.route("/visualizations/<path:filename>", methods=["GET"])
def visualizations(filename):
    return send_from_directory(VIS_DIR, filename)


if __name__ == "__main__":
    ensure_generated_dirs()
    app.run(host="0.0.0.0", port=5000, debug=True)
