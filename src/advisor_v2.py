import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.weather_service import OpenMeteoService, AdvisoryWeatherContext

class AdvisorEngineV2:
    def __init__(self):
        self.data_path = Path(__file__).resolve().parents[1] / "data" / "crop_knowledge_v2.json"
        self.knowledge = self._load_knowledge()
        self.weather_service = OpenMeteoService()

    def _load_knowledge(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            return {}
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_crops(self) -> List[Dict[str, Any]]:
        crops = []
        for name, data in self.knowledge.get("crops", {}).items():
            crops.append({
                "name": name,
                "emoji": data.get("emoji", "🌱"),
                "category": data.get("category", "General"),
                "readiness": data.get("readiness", "starter")
            })
        return sorted(crops, key=lambda x: (x["readiness"] != "production", x["name"]))

    def get_timeline(self, crop_name: str) -> Dict[str, Any]:
        crop_data = self.knowledge.get("crops", {}).get(crop_name)
        if not crop_data:
            return {"error": f"Crop {crop_name} not found"}
        
        return {
            "crop": crop_name,
            "micro_stages": crop_data.get("micro_stages", []),
            "total_days": crop_data.get("crop_profile", {}).get("duration_days", 100)
        }

    def query(self, 
              crop_name: str, 
              sowing_date: Optional[str] = None, 
              manual_day: Optional[int] = None,
              lat: Optional[float] = None,
              lon: Optional[float] = None,
              lang: str = 'en') -> Dict[str, Any]:
        
        crop_data = self.knowledge.get("crops", {}).get(crop_name)
        if not crop_data:
            return {"error": f"Crop {crop_name} not found"}

        # 1. Resolve Day
        current_day = 1
        if manual_day is not None:
            current_day = manual_day
        elif sowing_date:
            try:
                sowing_obj = datetime.strptime(sowing_date, "%Y-%m-%d").date()
                delta = date.today() - sowing_obj
                current_day = max(1, delta.days + 1)
            except Exception:
                current_day = 1

        # 2. Resolve Micro-Stage
        micro_stages = crop_data.get("micro_stages", [])
        current_stage = next((s for s in micro_stages if s["start_day"] <= current_day <= s["end_day"]), 
                             micro_stages[-1] if micro_stages else {})
        
        # 3. Resolve Daily Window Advice
        daily_windows = crop_data.get("daily_windows", [])
        window_advice = next((w for w in daily_windows if w["start_day"] <= current_day <= w["end_day"]), 
                             daily_windows[-1] if daily_windows else {})

        # 4. Fetch Weather Context
        weather_context = None
        if lat is not None and lon is not None:
            try:
                weather_context = self.weather_service.fetch_advisory_weather(lat, lon)
            except Exception:
                pass

        # 5. Build Response (Use copies to prevent mutating the shared knowledge base)
        result = {
            "crop": crop_name,
            "day": current_day,
            "stage": dict(current_stage), # Copy to avoid cross-language contamination
            "focus": window_advice.get("focus", ""),
            "today_actions": list(window_advice.get("today_actions", [])),
            "next_5_day_plan": list(window_advice.get("next_5_day_plan", [])),
            "risks": [dict(r) for r in window_advice.get("base_risks", [])],
            "alerts": list(window_advice.get("alerts", [])),
            "ai_recommendation": window_advice.get("ai_recommendation", ""),
            "why_this_matters": window_advice.get("why_this_matters", ""),
            "weather": weather_context.to_dict() if weather_context else None,
            "progress_pct": min(100, round((current_day / crop_data["crop_profile"]["duration_days"]) * 100, 1))
        }

        # 6. Apply Decision Rules & Safety Guardrails
        if weather_context:
            self._apply_weather_logic(result, crop_data, weather_context)

        # 7. Localize to Marathi if requested
        if lang == 'mr':
            self._localize_to_marathi(result, crop_name)

        return result

    def _localize_to_marathi(self, result: Dict[str, Any], crop_name: str):
        # Farmer-friendly Marathi dictionary for Maize
        maize_mr = {
            "stages": {
                "VE": "रोपांची उगवण", "V1": "पहिले पान", "V2": "दुसरे पान", "V3": "तिसरे पान",
                "V4": "चौथे पान", "V5": "पाचवे पान (वाढ सुरू)", "V6": "गुडघाभर उंची", "V8": "जोमदार वाढ",
                "V10": "भरपूर पाने", "V12": "तुरा येण्यापूर्वीची तयारी", "VT": "तुरा येणे",
                "R1": "कणीस बाहेर येणे (सिल्किंग)", "R2": "कणीस भरण्यास सुरुवात", "R3": "दुधिया अवस्था",
                "R4": "चिकाची अवस्था", "R5": "दाणे घट्ट होणे", "R6": "कापणीची वेळ (परिपक्वता)"
            },
            "families": {
                "establishment": "उगवण अवस्था", "vegetative": "वाढीची अवस्था", "rapid_growth": "जोमदार वाढ",
                "reproductive_critical": "फुलोरा (महत्वाचा टप्पा)", "grain_set": "दाणे भरणे",
                "grain_fill": "दाण्यांची वाढ", "maturity": "कापणीची वेळ"
            }
        }

        # Localize Stage Text
        if crop_name == "Maize":
            stage_code = result["stage"].get("code")
            if stage_code in maize_mr["stages"]:
                result["stage"]["label"] = maize_mr["stages"][stage_code]
            
            family_code = result["stage"].get("stage_family")
            if family_code in maize_mr["families"]:
                result["stage"]["stage_family_label"] = maize_mr["families"][family_code]

        # Localize Actions (Full translation mapping for all common agronomy phrases)
        translations = {
            "irrigation": "गरजेनुसार पिकाला पाणी द्या.",
            "water": "पिकाला वेळेवर पाणी द्या.",
            "nitrogen": "पिकाला नत्र खताचा (युरिया) डोस द्या.",
            "fertilizer": "खत व्यवस्थापन करा.",
            "weed": "शेत तणमुक्त ठेवा, कोळपणी करा.",
            "pest": "किडींचा प्रादुर्भाव तपासा.",
            "scout": "शेताची पाहणी करा.",
            "harvest": "पिकाची कापणी करा.",
            "moisture": "मातीतील ओलावा तपासा.",
            "nutrient": "पिकाचे पोषण आणि खत नियोजन करा.",
            "sowing": "पेरणीची तयारी करा.",
            "germination": "रोपांची उगवण तपासा.",
            "stand": "रोपांची संख्या कायम ठेवा.",
            "canopy": "पानांची वाढ तपासा.",
            "lodging": "पीक लोळणार नाही याची काळजी घ्या.",
            "tasseling": "तुरा येण्याच्या काळात काळजी घ्या.",
            "silking": "कणीस बाहेर येताना पाणी द्या.",
            "grain": "दाणे भरण्याच्या अवस्थेत ओलावा ठेवा.",
            "maturity": "पीक काढणीसाठी तयार आहे."
        }

        def translate_phrase(text):
            low = text.lower()
            for key, val in translations.items():
                if key in low: return val
            return text

        result["today_actions"] = [translate_phrase(a) for a in result["today_actions"]]
        result["next_5_day_plan"] = [translate_phrase(a) for a in result["next_5_day_plan"]]
        
        # Override Focus
        result["focus"] = translate_phrase(result["focus"])
        
        # AI Recommendation & Why
        if "irrigation" in result["ai_recommendation"].lower() or "water" in result["ai_recommendation"].lower():
            result["ai_recommendation"] = "योग्य ओलावा टिकवण्यासाठी पाणी देण्याचे नियोजन करा. ताण पडू देऊ नका."
        else:
            result["ai_recommendation"] = "पिकाची उत्तम वाढ होण्यासाठी वेळेवर निगा राखा."

        result["why_this_matters"] = "उत्पादन वाढवण्यासाठी आणि पिकाची गुणवत्ता टिकवण्यासाठी हा टप्पा अतिशय महत्त्वाचा आहे."
        
        # Alerts
        mr_alerts = []
        for alert in result["alerts"]:
            if "rain" in alert.lower():
                mr_alerts.append("🚨 आज पाऊस येण्याची शक्यता आहे, पाणी देणे टाळा.")
            elif "stress" in alert.lower() or "irrigation" in alert.lower():
                mr_alerts.append("⚠️ पिकाला पाण्याचा ताण पडतोय, पाणी द्या.")
            else:
                mr_alerts.append("✅ पिकाची नियमित तपासणी करा.")
        result["alerts"] = mr_alerts

        # Risks (Full Marathi, no English fragments)
        for risk in result["risks"]:
            if risk.get("severity") == "high":
                risk["severity"] = "उच्च धोका"
            elif risk.get("severity") == "critical":
                risk["severity"] = "अतिशय धोकादायक"
            else:
                risk["severity"] = "मध्यम धोका"
            
            risk["reason"] = translate_phrase(risk.get("reason", ""))
            risk["recommended_response"] = translate_phrase(risk.get("recommended_response", ""))

        result["confidence_note"] = "🌟 तुम्ही खूप मेहनत करत आहात! आम्ही तुम्हाला योग्य दिशा दाखवत राहू."

    def _apply_weather_logic(self, result: Dict[str, Any], crop_data: Dict[str, Any], weather: AdvisoryWeatherContext):
        rules = crop_data.get("decision_rules", {})
        signals = weather.signals
        
        # Irrigation Suppression
        irr_rules = rules.get("irrigation", {})
        rain_threshold = irr_rules.get("default_suppress_if_rain_mm_24h", 10)
        
        if result["stage"].get("stage_family") in ["reproductive_critical"]:
            rain_threshold = irr_rules.get("critical_stage_suppress_if_rain_mm_24h", 15)

        if signals.get("rain_expected_24h_mm", 0) >= rain_threshold:
            # Filter out irrigation actions or add a blocking note
            result["today_actions"] = [a for a in result["today_actions"] if "irrigation" not in a.lower() and "water" not in a.lower()]
            result["alerts"].append(f"Rainfall of {signals['rain_expected_24h_mm']}mm expected. Postpone irrigation to avoid waterlogging.")
        
        # Fertilizer Delay
        fert_rules = rules.get("fertilizer", {})
        if signals.get("rain_expected_24h_mm", 0) >= fert_rules.get("delay_if_rain_mm_24h", 20):
            result["alerts"].append("Heavy rain forecast. Delay fertilizer application to prevent nutrient loss.")

        # Heat Stress
        if signals.get("heat_stress"):
            result["risks"].append({
                "severity": "high",
                "reason": "High temperature detected. Increased evaporation and plant stress risk.",
                "recommended_response": "Ensure soil remains moist and avoid afternoon field operations."
            })

        # Confidence Note
        result["confidence_note"] = "Live weather integrated advice."
