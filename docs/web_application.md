# Smart Agri Assistant Web Application

## What This App Includes

- crop prediction from structured agronomic inputs
- browser geolocation plus OpenWeather autofill for temperature, humidity, and rainfall
- disease detection from uploaded or webcam-captured leaf images
- hybrid crop plus disease inference
- Grad-CAM explainability overlays
- fertilizer hints, remedies, and field tips

## Routes

- `GET /` home page
- `GET /crop` crop prediction page
- `GET /disease` disease detection page
- `GET /hybrid` hybrid result page
- `GET /insights` model report page
- `POST /get_weather` weather lookup from latitude and longitude
- `POST /predict_crop` crop prediction API
- `POST /predict_disease` disease prediction API
- `POST /hybrid_predict` hybrid prediction API

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
copy .env.example .env
```

Set `OPENWEATHER_API_KEY` to your OpenWeather key.

3. Make sure the model artifacts exist:

- crop model: `models/best_model.pkl`, `models/class_labels.json`
- disease model: `models/disease/leaf_disease_classifier.keras`, `models/disease/leaf_disease_metadata.json`

4. Start the Flask app:

```bash
python app/app.py
```

5. Open:

```text
http://127.0.0.1:5000
```

## Notes

- browser geolocation requires a secure context in production, so deploy over HTTPS
- on localhost, geolocation usually works without extra setup
- the weather autofill uses live weather as a convenience and remains editable for field correction
- generated uploads and Grad-CAM overlays are stored under `app/static/generated/`
