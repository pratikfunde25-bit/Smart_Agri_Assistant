# 🌱 Smart Agri Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Pratik/Smart_Agri_Assistant/graphs/commit-activity)

**Smart Agri Assistant** is a data-driven decision support platform designed specifically for Indian agriculture. It leverages Machine Learning, Explainable AI (XAI), and real-time weather data to provide high-accuracy crop recommendations, disease detection, and actionable analytics for farmers.

---

## 🚀 Key Features

- **📍 Location-Based Recommendations**: Automatically fetches weather data (temperature, humidity, rainfall) based on the user's location to provide hyper-local advice.
- **🛡️ Disease Prediction**: Advanced deep learning models to identify crop diseases and suggest preventive measures.
- **🧠 Explainable AI (XAI)**: Provides insights into *why* a specific crop was recommended, making the AI transparent and trustworthy.
- **📊 Interactive Dashboard**: Visualizes soil health metrics and regional trends for better planning.
- **🇮🇳 Tailored for India**: Focuses on Indian soil types, weather patterns, and crop varieties (e.g., Sugarcane, Cotton, Rice, etc.).

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow/Keras
- **Explainability**: SHAP, LIME
- **Data Processing**: Pandas, NumPy
- **Weather API**: OpenWeatherMap (or similar)
- **Frontend**: HTML5, Vanilla CSS, JavaScript (Dynamic UI)

---

## 📁 Project Structure

```text
Smart_Agri_Assistant/
├── app/                # Flask application (app.py, templates, static)
├── src/                # Modular core logic (Predictors, Data Pipeline, Weather)
├── data/               # Data storage (Raw, Processed, External)
├── models/             # Pre-trained ML/DL models
├── notebooks/          # Research and EDA (Jupyter Notebooks)
├── docs/               # Documentation and project reports
├── scripts/            # Utility scripts for data processing
├── .env.example        # Environment variable template
├── requirements.txt    # Python dependencies
└── README.md           # This file!
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Pratik/Smart_Agri_Assistant.git
   cd Smart_Agri_Assistant
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

5. **Run the Application**:
   ```bash
   python app/app.py
   ```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

**Pratik** - [GitHub](https://github.com/Pratik)

Project Link: [https://github.com/Pratik/Smart_Agri_Assistant](https://github.com/Pratik/Smart_Agri_Assistant)

---
*Developed for Major Project Sem 6 @ SPIT*