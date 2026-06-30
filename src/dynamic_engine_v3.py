import math
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from src.weather_service import OpenMeteoService, AdvisoryWeatherContext

# --- DYNAMIC KNOWLEDGE BASE ---
# This replaces static JSON with a rule-based engine covering major crops.
CROP_DB = {
    "Maize": {
        "category": "Cereals", "emoji": "🌽", "duration": 110,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Seedling (V1-V3)", "focus": "Root establishment and weed control"},
            "vegetative": {"pct": (15, 45), "label": "Vegetative Growth (V4-VT)", "focus": "Rapid canopy expansion and nutrient uptake"},
            "flowering": {"pct": (45, 60), "label": "Silking & Tasseling (R1)", "focus": "Pollination and critical moisture management"},
            "yield_formation": {"pct": (60, 85), "label": "Grain Filling (R2-R4)", "focus": "Kernel weight accumulation"},
            "maturity": {"pct": (85, 100), "label": "Maturity (R5-R6)", "focus": "Dry down and harvest preparation"}
        },
        "inputs": {
            "fertilizer": "Basal: 50% N, 100% P & K. Top dress: 25% N at knee-high (V6), 25% N at tasseling (VT).",
            "herbicide": "Pre-emergence: Atrazine. Post-emergence (15-20 days): Tembotrione (Laudis) or Topramezone (Tynzer).",
            "pesticide": "Fall Armyworm (FAW): Emamectin Benzoate 5% SG or Spinetoram 11.7% SC (Delegate)."
        }
    },
    "Soybean": {
        "category": "Oilseeds", "emoji": "🌿", "duration": 105,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Germination (V1-V2)", "focus": "Ensure uniform plant stand"},
            "vegetative": {"pct": (15, 40), "label": "Branching (V3-Vn)", "focus": "Nodule formation and weed management"},
            "flowering": {"pct": (40, 55), "label": "Flowering (R1-R2)", "focus": "Protect from moisture stress"},
            "yield_formation": {"pct": (55, 85), "label": "Pod Development (R3-R6)", "focus": "Seed filling and pest protection"},
            "maturity": {"pct": (85, 100), "label": "Maturity (R7-R8)", "focus": "Leaf drop and harvest"}
        },
        "inputs": {
            "fertilizer": "Basal: 20:60:40 NPK kg/ha + Sulphur (Bentonite). Foliar spray 2% Urea/DAP at flowering.",
            "herbicide": "Pre-emergence: Pendimethalin. Post-emergence: Imazethapyr (Pursuit) or Propaquizafop.",
            "pesticide": "Whitefly/Girdle Beetle: Thiamethoxam + Lambda-cyhalothrin (Alika). Spodoptera: Flubendiamide."
        }
    },
    "Cotton": {
        "category": "Commercial", "emoji": "☁️", "duration": 160,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Seedling", "focus": "Protect against early sucking pests"},
            "vegetative": {"pct": (15, 35), "label": "Square Formation", "focus": "Encourage sympodial branches"},
            "flowering": {"pct": (35, 60), "label": "Flowering & Boll Setting", "focus": "Prevent square drop, nutrient supply"},
            "yield_formation": {"pct": (60, 85), "label": "Boll Development", "focus": "Protect from Pink Bollworm"},
            "maturity": {"pct": (85, 100), "label": "Boll Opening", "focus": "Clean picking"}
        },
        "inputs": {
            "fertilizer": "Split Nitrogen at 30, 60, 90 days. Spray 2% DAP or KNO3 at boll development.",
            "herbicide": "Pre-emergence: Pendimethalin. Post-emergence: Pyrithiobac-sodium + Quizalofop-p-ethyl (Hitweed Maxx).",
            "pesticide": "Sucking Pests: Flonicamid (Ullala). Pink Bollworm: Profenofos or Emamectin Benzoate."
        }
    },
    "Sugarcane": {
        "category": "Commercial", "emoji": "🎋", "duration": 360,
        "phases": {
            "establishment": {"pct": (0, 10), "label": "Germination", "focus": "Settt treatment and moisture"},
            "vegetative": {"pct": (10, 35), "label": "Tillering", "focus": "Maximize tiller production"},
            "flowering": {"pct": (35, 70), "label": "Grand Growth", "focus": "Irrigation and earthing up"},
            "yield_formation": {"pct": (70, 90), "label": "Sugar Accumulation", "focus": "Potassium application"},
            "maturity": {"pct": (90, 100), "label": "Ripening", "focus": "Stop irrigation before harvest"}
        },
        "inputs": {
            "fertilizer": "Heavy feeder. Split N into 4 doses. Use Zinc & Ferrous Sulphate to prevent chlorosis.",
            "herbicide": "Pre-emergence: Atrazine. Post-emergence: 2,4-D for broadleaf weeds.",
            "pesticide": "Early Shoot Borer: Chlorantraniliprole (Coragen). White Grub: Fipronil granules."
        }
    },
    "Wheat": {
        "category": "Cereals", "emoji": "🌾", "duration": 120,
        "phases": {
            "establishment": {"pct": (0, 18), "label": "CRI (Crown Root Initiation)", "focus": "First critical irrigation"},
            "vegetative": {"pct": (18, 45), "label": "Tillering to Jointing", "focus": "Top dressing nitrogen"},
            "flowering": {"pct": (45, 65), "label": "Booting to Heading", "focus": "Protect flag leaf"},
            "yield_formation": {"pct": (65, 85), "label": "Milking to Dough stage", "focus": "Irrigation to avoid shriveled grains"},
            "maturity": {"pct": (85, 100), "label": "Hard Dough to Maturity", "focus": "Harvesting"}
        },
        "inputs": {
            "fertilizer": "First irrigation (CRI stage): Top dress Nitrogen. Spray NPK 19:19:19 if deficient.",
            "herbicide": "Post-emergence (30-35 days): Sulfosulfuron + Metsulfuron (Total) or Clodinafop-propargyl.",
            "pesticide": "Aphids: Imidacloprid. Rust (Yellow/Brown): Propiconazole (Tilt) or Tebuconazole."
        }
    },
    "Tomato": {
        "category": "Vegetables", "emoji": "🍅", "duration": 140,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Transplanting", "focus": "Seedling establishment"},
            "vegetative": {"pct": (15, 30), "label": "Vegetative Growth", "focus": "Staking and pruning"},
            "flowering": {"pct": (30, 50), "label": "Flowering & Fruit Set", "focus": "Micronutrients (Boron, Calcium)"},
            "yield_formation": {"pct": (50, 80), "label": "Fruit Enlargement", "focus": "Potassium and pest control"},
            "maturity": {"pct": (80, 100), "label": "Harvesting", "focus": "Multiple pickings"}
        },
        "inputs": {
            "fertilizer": "Water soluble fertilizers (19:19:19, 0:52:34, 0:0:50) via drip irrigation. Calcium Nitrate for fruit quality.",
            "herbicide": "Pre-emergence: Pendimethalin. Manual weeding preferred later.",
            "pesticide": "Fruit Borer: Spinosad (Tracer). Blight: Mancozeb or Cymoxanil + Mancozeb (Curzate)."
        }
    },
    "Gram (Chana)": {
        "category": "Pulses", "emoji": "🌱", "duration": 110,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Seedling", "focus": "Wilt prevention"},
            "vegetative": {"pct": (20, 45), "label": "Branching (Nipper stage)", "focus": "Nipping to encourage branches"},
            "flowering": {"pct": (45, 65), "label": "Flowering", "focus": "Critical irrigation"},
            "yield_formation": {"pct": (65, 85), "label": "Pod Formation", "focus": "Pod borer management"},
            "maturity": {"pct": (85, 100), "label": "Maturity", "focus": "Harvesting"}
        },
        "inputs": {
            "fertilizer": "Basal: 20:40:20 NPK. Seed treatment with Rhizobium is critical.",
            "herbicide": "Pre-emergence: Pendimethalin. Usually rainfed, so mechanical weeding is common.",
            "pesticide": "Pod Borer (Helicoverpa): Emamectin Benzoate or Indoxacarb. Wilt: Seed treatment with Trichoderma."
        }
    },
    "Onion": {
        "category": "Vegetables", "emoji": "🧅", "duration": 120,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Transplanting", "focus": "Establishment in main field"},
            "vegetative": {"pct": (20, 50), "label": "Vegetative Growth", "focus": "Nitrogen application and weed control"},
            "flowering": {"pct": (50, 75), "label": "Bulb Initiation", "focus": "Stop nitrogen, increase potassium"},
            "yield_formation": {"pct": (75, 90), "label": "Bulb Development", "focus": "Maintain uniform moisture"},
            "maturity": {"pct": (90, 100), "label": "Neck Fall & Curing", "focus": "Stop irrigation"}
        },
        "inputs": {
            "fertilizer": "Split N applications. High Sulphur requirement (use SSP). K is crucial for bulb size.",
            "herbicide": "Pre-emergence: Oxyfluorfen (Goal). Post-emergence: Quizalofop for grassy weeds.",
            "pesticide": "Thrips: Fipronil or Spinetoram. Purple Blotch: Mancozeb + Carbendazim (Saaf)."
        }
    },
    "Rice": {
        "category": "Cereals", "emoji": "🌾", "duration": 130,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Nursery / Transplanting", "focus": "Seedling recovery and rooting"},
            "vegetative": {"pct": (15, 45), "label": "Active Tillering", "focus": "Nutrient uptake and water management"},
            "flowering": {"pct": (45, 70), "label": "Panicle Initiation", "focus": "Critical water and pest protection"},
            "yield_formation": {"pct": (70, 90), "label": "Flowering & Milk Stage", "focus": "Protect from blast and stem borer"},
            "maturity": {"pct": (90, 100), "label": "Dough Stage & Ripening", "focus": "Field drainage for harvest"}
        },
        "inputs": {
            "fertilizer": "Zinc Sulphate (25kg/ha) is critical. Apply Nitrogen in 3 splits (Basal, Tillering, Panicle).",
            "herbicide": "Pre-emergence: Pretilachlor. Post-emergence: Bispyribac-sodium (Nominee Gold).",
            "pesticide": "Stem Borer: Cartap Hydrochloride or Fipronil. Blast: Tricyclazole (Beam)."
        }
    },
    "Jowar": {
        "category": "Cereals", "emoji": "🌾", "duration": 110,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Seedling", "focus": "Establishment"},
            "vegetative": {"pct": (15, 50), "label": "Growth Phase", "focus": "Shoot fly management"},
            "flowering": {"pct": (50, 75), "label": "Booting & Flowering", "focus": "Moisture at grain set"},
            "yield_formation": {"pct": (75, 90), "label": "Grain Filling", "focus": "Protect from birds and midge"},
            "maturity": {"pct": (90, 100), "label": "Maturity", "focus": "Harvesting"}
        },
        "inputs": {
            "fertilizer": "Apply NPK at sowing. Top dress Urea at 30 days.",
            "herbicide": "Pre-emergence: Atrazine.",
            "pesticide": "Shoot Fly: Seed treatment with Thiamethoxam. Stem Borer: Carbofuran granules."
        }
    },
    "Bajra": {
        "category": "Cereals", "emoji": "🌾", "duration": 90,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Seedling", "focus": "Thinning and gap filling"},
            "vegetative": {"pct": (20, 50), "label": "Tillering", "focus": "Inter-culturing"},
            "flowering": {"pct": (50, 70), "label": "Heading", "focus": "Downy mildew protection"},
            "yield_formation": {"pct": (70, 90), "label": "Grain Development", "focus": "Moisture at earhead stage"},
            "maturity": {"pct": (90, 100), "label": "Ripening", "focus": "Prompt harvest"}
        },
        "inputs": {
            "fertilizer": "Responsive to Nitrogen. Apply in 2 splits.",
            "herbicide": "Atrazine at pre-emergence.",
            "pesticide": "Downy Mildew: Metalaxyl (Apron) seed treatment. Ergot: Brine solution treatment."
        }
    },
    "Tur (Arhar)": {
        "category": "Pulses", "emoji": "🌿", "duration": 180,
        "phases": {
            "establishment": {"pct": (0, 15), "label": "Seedling", "focus": "Slow early growth phase"},
            "vegetative": {"pct": (15, 45), "label": "Branching", "focus": "Weed management"},
            "flowering": {"pct": (45, 75), "label": "Flowering", "focus": "Protect from pod borer"},
            "yield_formation": {"pct": (75, 90), "label": "Pod Filling", "focus": "Maintain moisture"},
            "maturity": {"pct": (90, 100), "label": "Maturity", "focus": "Drying and harvesting"}
        },
        "inputs": {
            "fertilizer": "DAP (100kg/ha) at sowing. Use Sulphur for better yield.",
            "herbicide": "Pendimethalin (Pre-em). Imazethapyr (Post-em).",
            "pesticide": "Pod Borer: Indoxacarb or Chlorantraniliprole (Coragen). Wilt: Trichoderma."
        }
    },
    "Groundnut": {
        "category": "Oilseeds", "emoji": "🥜", "duration": 115,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Germination", "focus": "Seed treatment for root rot"},
            "vegetative": {"pct": (20, 40), "label": "Flowering", "focus": "Avoid disturbance at pegging"},
            "flowering": {"pct": (40, 60), "label": "Pegging", "focus": "Apply Gypsum"},
            "yield_formation": {"pct": (60, 90), "label": "Pod Development", "focus": "Irrigation during pod set"},
            "maturity": {"pct": (90, 100), "label": "Maturity", "focus": "Check shell color for harvest"}
        },
        "inputs": {
            "fertilizer": "Gypsum (500kg/ha) at 45 days is vital for pod filling. NPK: 25:50:0.",
            "herbicide": "Pendimethalin. Post-emergence: Imazethapyr.",
            "pesticide": "White Grub: Chlorpyrifos soil drenching. Tikka disease: Carbendazim + Mancozeb."
        }
    },
    "Mustard": {
        "category": "Oilseeds", "emoji": "🌱", "duration": 110,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Seedling", "focus": "Thinning and spacing"},
            "vegetative": {"pct": (20, 45), "label": "Rosette Phase", "focus": "Irrigation and Nitrogen"},
            "flowering": {"pct": (45, 70), "label": "Flowering & Siliqua", "focus": "Aphid management"},
            "yield_formation": {"pct": (70, 90), "label": "Seed Filling", "focus": "Protect from frost"},
            "maturity": {"pct": (90, 100), "label": "Maturity", "focus": "Harvest at 75% siliqua yellowing"}
        },
        "inputs": {
            "fertilizer": "High Sulphur requirement (use SSP). Split Nitrogen.",
            "herbicide": "Pendimethalin (Pre-em).",
            "pesticide": "Aphids: Thiamethoxam or Oxydemeton-methyl. Alternaria: Mancozeb."
        }
    },
    "Grapes": {
        "category": "Fruits", "emoji": "🍇", "duration": 150,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Pruning & Sprouting", "focus": "Disease prevention at bud break"},
            "vegetative": {"pct": (20, 45), "label": "Vegetative Growth", "focus": "Canopy management and thinning"},
            "flowering": {"pct": (45, 60), "label": "Flowering & Fruit Set", "focus": "Gibberellic Acid application"},
            "yield_formation": {"pct": (60, 85), "label": "Berry Development", "focus": "Nutrient sprays and water"},
            "maturity": {"pct": (85, 100), "label": "Ripening (Veraison)", "focus": "Sugar accumulation and Brix check"}
        },
        "inputs": {
            "fertilizer": "Drip fertigation with 19:19:19, 0:52:34, and Sulphate of Potash (0:0:50).",
            "herbicide": "Manual weeding and mulching preferred.",
            "pesticide": "Downy Mildew: Metalaxyl + Mancozeb (Ridomil). Mealy Bug: Buprofezin."
        }
    },
    "Chili": {
        "category": "Vegetables", "emoji": "🌶️", "duration": 160,
        "phases": {
            "establishment": {"pct": (0, 20), "label": "Transplanting", "focus": "Seedling establishment"},
            "vegetative": {"pct": (20, 45), "label": "Vegetative Growth", "focus": "Control thrips and mites"},
            "flowering": {"pct": (45, 70), "label": "Flowering & Fruit Set", "focus": "Apply NAA to prevent flower drop"},
            "yield_formation": {"pct": (70, 90), "label": "Fruit Development", "focus": "Potassium and Calcium supply"},
            "maturity": {"pct": (90, 100), "label": "Harvesting", "focus": "Multiple pickings (Green/Red)"}
        },
        "inputs": {
            "fertilizer": "NPK in splits. Calcium Nitrate + Boron foliar spray for fruit quality.",
            "herbicide": "Pendimethalin (Pre-em).",
            "pesticide": "Thrips/Mites: Fipronil or Abamectin. Fruit Borer: Emamectin Benzoate."
        }
    }
}


class DynamicAdvisorEngine:
    def __init__(self):
        self.weather_service = OpenMeteoService()

    def get_crops(self) -> List[Dict[str, Any]]:
        crops = []
        for name, data in CROP_DB.items():
            crops.append({
                "name": name,
                "emoji": data["emoji"],
                "category": data["category"],
                "readiness": "production" if name in ["Maize", "Soybean", "Cotton", "Wheat", "Sugarcane"] else "starter"
            })
        return sorted(crops, key=lambda x: (x["readiness"] != "production", x["name"]))

    def _get_phase_days(self, crop_data: Dict) -> Dict[str, Dict[str, int]]:
        dur = crop_data["duration"]
        prev_end = 0
        phase_days = {}
        phase_keys = list(crop_data["phases"].keys())
        for i, phase_key in enumerate(phase_keys):
            start_day = prev_end + 1
            end_day = int((crop_data["phases"][phase_key]["pct"][1] / 100) * dur)
            if end_day < start_day:
                end_day = start_day
            if i == len(phase_keys) - 1:
                end_day = dur
                
            phase_days[phase_key] = {"start_day": start_day, "end_day": end_day}
            prev_end = end_day
        return phase_days

    def get_timeline(self, crop_name: str) -> Dict[str, Any]:
        crop_data = CROP_DB.get(crop_name)
        if not crop_data:
            return {"error": f"Crop {crop_name} not found"}
            
        dur = crop_data["duration"]
        micro_stages = []
        phase_days = self._get_phase_days(crop_data)
        
        for phase_key, phase_data in crop_data["phases"].items():
            micro_stages.append({
                "code": phase_key.upper()[:3],
                "label": phase_data["label"],
                "start_day": phase_days[phase_key]["start_day"],
                "end_day": phase_days[phase_key]["end_day"]
            })
            
        return {
            "crop": crop_name,
            "micro_stages": micro_stages,
            "total_days": dur
        }

    def query(self, crop_name: str, sowing_date: Optional[str] = None, manual_day: Optional[int] = None,
              lat: Optional[float] = None, lon: Optional[float] = None, lang: str = 'en') -> Dict[str, Any]:
        
        crop_data = CROP_DB.get(crop_name)
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

        if current_day > crop_data["duration"]:
            current_day = crop_data["duration"]

        # 2. Determine Phase
        phase_days = self._get_phase_days(crop_data)
        phase_key = "maturity"  # default
        for pk, days in phase_days.items():
            if days["start_day"] <= current_day <= days["end_day"]:
                phase_key = pk
                break
                
        phase_data = crop_data["phases"][phase_key]

        stage_obj = {
            "code": phase_key.upper()[:3],
            "label": phase_data["label"],
            "start_day": phase_days[phase_key]["start_day"],
            "end_day": phase_days[phase_key]["end_day"],
            "stage_family": phase_key
        }

        # 3. Dynamic Base Generation (AI Logic Rules)
        actions = []
        alerts = []
        risks = []
        
        if phase_key == "establishment":
            actions = ["Ensure optimal soil moisture for establishment.", "Monitor for early cutworms or soil-borne pests.", "Apply pre-emergence herbicide if not already done."]
            risks.append({"severity": "high", "reason": "Damping off or poor germination", "recommended_response": "Ensure proper drainage."})
            ai_rec = "Focus strictly on achieving a uniform plant stand. Gaps now will permanently lower yield."
        elif phase_key == "vegetative":
            actions = ["Top dress nitrogen fertilizer.", "Scout for foliar pests.", "Maintain weed-free environment."]
            risks.append({"severity": "medium", "reason": "Weed competition", "recommended_response": "Apply post-emergence herbicide or manual weeding."})
            ai_rec = "This is the canopy building phase. Nutrient availability here directly drives the plant's factory size."
        elif phase_key == "flowering":
            actions = ["Ensure absolutely NO moisture stress.", "Apply protective fungicide/insecticide if pest pressure is high.", "Stop heavy mechanical operations."]
            risks.append({"severity": "critical", "reason": "Moisture stress during pollination", "recommended_response": "Apply irrigation immediately if soil is dry."})
            ai_rec = "TREAT AS CRITICAL. Any stress during flowering causes irreversible yield loss due to poor fruit/grain setting."
        elif phase_key == "yield_formation":
            actions = ["Maintain adequate moisture for grain/fruit filling.", "Monitor for fruit borers or pod bugs.", "Apply foliar potassium (K) if recommended."]
            risks.append({"severity": "high", "reason": "Pest attack on developing fruit/grain", "recommended_response": "Spray target-specific insecticide."})
            ai_rec = "Focus shifts to moving energy from leaves to the harvestable part. Potassium is highly beneficial here."
        else: # maturity
            actions = ["Stop irrigation to allow drying.", "Prepare harvesting equipment.", "Monitor for weather delays."]
            risks.append({"severity": "medium", "reason": "Late rains causing quality deterioration", "recommended_response": "Harvest promptly when physiological maturity is reached."})
            ai_rec = "Crop has reached physiological maturity. Rapid and safe harvesting is the only priority."

        # 4. Fetch Weather Context
        weather_context = None
        if lat is not None and lon is not None:
            try:
                weather_context = self.weather_service.fetch_advisory_weather(lat, lon)
            except Exception:
                pass

        # 5. Build Base Response
        result = {
            "crop": crop_name,
            "day": current_day,
            "stage": stage_obj,
            "focus": phase_data["focus"],
            "today_actions": actions,
            "next_5_day_plan": ["Continue phase-specific monitoring.", "Check weather forecast for sudden changes."],
            "risks": risks,
            "alerts": alerts,
            "ai_recommendation": ai_rec,
            "why_this_matters": "Stage-specific management is the core of precision agriculture.",
            "inputs": crop_data["inputs"],
            "weather": weather_context.to_dict() if weather_context else None,
            "progress_pct": min(100, round((current_day / crop_data["duration"]) * 100, 1))
        }

        # 6. Apply Weather Modifications
        if weather_context:
            self._apply_weather_logic(result, weather_context)

        # 7. Localization
        if lang == 'mr':
            self._localize_to_marathi(result)

        return result

    def _apply_weather_logic(self, result: Dict, weather: AdvisoryWeatherContext):
        signals = weather.signals
        rain_mm = signals.get("rain_expected_24h_mm", 0)
        
        # Rain Logic
        if rain_mm > 10:
            result["alerts"].append(f"Heavy rainfall ({rain_mm}mm) expected. Do not irrigate. Postpone fertilizer/pesticide sprays.")
            # Remove irrigation actions
            result["today_actions"] = [a for a in result["today_actions"] if "irrigat" not in a.lower() and "moisture" not in a.lower()]
            result["today_actions"].insert(0, "Ensure field drainage is clear to prevent waterlogging.")
        elif rain_mm > 2:
            result["alerts"].append("Light rain expected. Good for top-dressing fertilizer, but avoid foliar sprays.")

        # Heat Logic
        if signals.get("heat_stress"):
            result["risks"].append({
                "severity": "high",
                "reason": "Heat Stress",
                "recommended_response": "Apply light irrigation in the evening to cool the micro-climate."
            })
            result["alerts"].append("Extreme heat detected. Plants may show temporary wilting.")

        result["confidence_note"] = "Live weather integrated advice."

    def _localize_to_marathi(self, result: Dict):
        # Universal Marathi Mapping
        # We use full sentences as keys to avoid "mixed" language results
        mr_dict = {
            # Categories
            "Cereals": "धान्ये", "Oilseeds": "गळित धान्ये (तेलबीया)", "Commercial": "नगदी पिके",
            "Vegetables": "भाजीपाला", "Pulses": "कडधान्ये", "Fruits": "फळे", "Spices": "मसाले",

            # Phase Labels (Universal)
            "Seedling (V1-V3)": "रोपावस्था (V1-V3)",
            "Vegetative Growth (V4-VT)": "शाकीय वाढ (V4-VT)",
            "Silking & Tasseling (R1)": "तुरा आणि सिल्क येणे (R1)",
            "Grain Filling (R2-R4)": "दाणे भरणे (R2-R4)",
            "Maturity (R5-R6)": "परिपक्वता (R5-R6)",
            "Germination": "उगवण अवस्था",
            "Tillering": "फुटवे फुटण्याची अवस्था",
            "Grand Growth": "जोमदार वाढीचा काळ",
            "Sugar Accumulation": "साखर साठवण काळ",
            "Ripening": "परिपक्वता / पक्व होणे",
            "CRI (Crown Root Initiation)": "मुकुट मूळ सुरूवात (CRI)",
            "Tillering to Jointing": "फुटवे ते कांडी अवस्था",
            "Booting to Heading": "बोंड येणे ते तुरा येणे",
            "Milking to Dough stage": "चिकाची अवस्था",
            "Hard Dough to Maturity": "दाणे कडक होणे ते काढणी",
            "Transplanting": "पुनर्लागवड",
            "Vegetative Growth": "शाकीय वाढ",
            "Flowering & Fruit Set": "फुलोरा आणि फळधारणा",
            "Fruit Enlargement": "फळांची वाढ",
            "Harvesting": "काढणी",
            "Branching (Nipper stage)": "फांद्या फुटणे (शेंडा खुडणे)",
            "Pod Formation": "घाटे आणि शेंगा लागणे",
            "Bulb Initiation": "कांदा बसणे (सुरुवात)",
            "Bulb Development": "कांद्याचा आकार वाढणे",
            "Neck Fall & Curing": "मान पडणे आणि सुकवणे",
            "Nursery / Transplanting": "रोपवाटिका आणि पुनर्लागवड",
            "Active Tillering": "फुटवे फुटणे (सक्रिय)",
            "Panicle Initiation": "लोंब्या येण्याची सुरुवात",
            "Flowering & Milk Stage": "फुलोरा आणि दुभती अवस्था",
            "Dough Stage & Ripening": "चिकाची अवस्था आणि पक्व होणे",
            "Growth Phase": "वाढीचा टप्पा",
            "Booting & Flowering": "बोंड आणि फुलोरा",
            "Heading": "तुरा येणे",
            "Grain Development": "दाणे भरणे",
            "Branching": "फांद्या फुटणे",
            "Pod Filling": "शेंगा भरणे",
            "Pruning & Sprouting": "छाटणी आणि फुटवे",
            "Berry Development": "मण्यांची वाढ",
            "Ripening (Veraison)": "रंग बदलणे आणि पक्व होणे",

            # Focus Areas
            "Root establishment and weed control": "मुळांची वाढ आणि तण नियंत्रण",
            "Rapid canopy expansion and nutrient uptake": "पानांची वाढ आणि अन्नद्रव्ये शोषण",
            "Pollination and critical moisture management": "परागीकरण आणि पाणी व्यवस्थापन",
            "Kernel weight accumulation": "दाण्यांचे वजन वाढणे",
            "Dry down and harvest preparation": "पीक सुकवणे आणि काढणीची तयारी",
            "Ensure uniform plant stand": "समान उगवण सुनिश्चित करा",
            "Nodule formation and weed management": "मुळांवरील गाठी आणि तण व्यवस्थापन",
            "Protect from moisture stress": "पाण्याच्या ताणापासून संरक्षण",
            "Seed filling and pest protection": "दाणे भरणे आणि कीड संरक्षण",
            "Leaf drop and harvest": "पाने गळणे आणि काढणी",

            # Actions (Full Sentences)
            "Ensure optimal soil moisture for establishment.": "उगवण होण्यासाठी जमिनीत पुरेसा ओलावा ठेवा.",
            "Monitor for early cutworms or soil-borne pests.": "सुरुवातीच्या किडींवर (कटवर्म) आणि जमिनीतील किडींवर लक्ष ठेवा.",
            "Apply pre-emergence herbicide if not already done.": "तणनाशकाची (Pre-emergence) फवारणी केली नसल्यास त्वरित करा.",
            "Top dress nitrogen fertilizer.": "नत्र खताचा (Urea) दुसरा हप्ता द्या.",
            "Scout for foliar pests.": "पानांवरील किडींची पाहणी (Scouting) करा.",
            "Maintain weed-free environment.": "शेत तणमुक्त ठेवा (कोळपणी किंवा निंदणी करा).",
            "Ensure absolutely NO moisture stress.": "पिकाला पाण्याचा अजिबात ताण पडू देऊ नका.",
            "Apply protective fungicide/insecticide if pest pressure is high.": "कीड जास्त असल्यास प्रतिबंधात्मक बुरशीनाशक/कीटकनाशक फवारा.",
            "Stop heavy mechanical operations.": "शेतात मोठी यांत्रिक कामे (उदा. कोळपणी) आता टाळा.",
            "Apply irrigation immediately if soil is dry.": "जमीन कोरडी असल्यास त्वरित पाणी (Irrigation) द्या.",
            "Maintain adequate moisture for grain/fruit filling.": "दाणे/फळ भरण्यासाठी योग्य ओलावा ठेवा.",
            "Monitor for fruit borers or pod bugs.": "फळ पोखरणाऱ्या अळीवर किंवा शेंगावरील ढेकणांवर लक्ष ठेवा.",
            "Apply foliar potassium (K) if recommended.": "शिफारशीनुसार पोटॅशयुक्त (K) खताची फवारणी करा.",
            "Spray target-specific insecticide.": "शिफारस केलेले योग्य कीटकनाशक फवारा.",
            "Stop irrigation to allow drying.": "पीक काढणीसाठी सुकण्यासाठी पाणी देणे थांबवा.",
            "Prepare harvesting equipment.": "काढणीच्या यंत्रांची (Harvesting equipment) जुळवाजुळव करा.",
            "Monitor for weather delays.": "हवामानाचा अंदाज घेऊन काढणीचे नियोजन करा.",
            "Harvest promptly when physiological maturity is reached.": "पीक पक्व झाल्यावर त्वरित काढणी करा.",
            "Ensure field drainage is clear to prevent waterlogging.": "पाणी साचू नये म्हणून पाण्याचा निचरा (Drainage) करा.",
            "Continue phase-specific monitoring.": "पीक वाढीच्या टप्प्यानुसार देखरेख सुरू ठेवा.",
            "Check weather forecast for sudden changes.": "हवामानातील बदलांकडे लक्ष द्या.",
            
            # AI Recommendations
            "Focus strictly on achieving a uniform plant stand. Gaps now will permanently lower yield.": "समान उगवण (Uniform stand) मिळवण्यावर भर द्या. आता पडलेली नांगे नंतर भरून निघत नाहीत आणि उत्पादनात घट होते.",
            "This is the canopy building phase. Nutrient availability here directly drives the plant's factory size.": "हा पिकाच्या पानांची आणि फांद्यांची वाढीचा काळ आहे. इथे दिलेले खत पिकाचे भविष्य ठरवते.",
            "TREAT AS CRITICAL. Any stress during flowering causes irreversible yield loss due to poor fruit/grain setting.": "अतिशय महत्त्वाचा टप्पा! फुलोऱ्याच्या काळात ताण पडल्यास परागीकरण नीट होत नाही आणि उत्पादनात मोठी घट होते.",
            "Focus shifts to moving energy from leaves to the harvestable part. Potassium is highly beneficial here.": "आता पिकाची ऊर्जा दाणे/फळ भरण्यात जाते. या काळात पोटॅश (Potassium) खताचा वापर फायदेशीर ठरतो.",
            "Crop has reached physiological maturity. Rapid and safe harvesting is the only priority.": "पीक पूर्ण पक्व झाले आहे. आता सुरक्षित आणि लवकर काढणी करणे हेच मुख्य उद्दिष्ट आहे.",
            
            # Risks & Reasons
            "Damping off or poor germination": "रोपे कुजणे किंवा कमी उगवण होणे",
            "Ensure proper drainage.": "पाण्याचा निचरा व्यवस्थित असल्याची खात्री करा.",
            "Weed competition": "तणांशी स्पर्धा",
            "Apply post-emergence herbicide or manual weeding.": "उगवणीनंतरचे तणनाशक फवारा किंवा निंदणी करा.",
            "Moisture stress during pollination": "परागीकरणाच्या वेळी पाण्याचा ताण",
            "Pest attack on developing fruit/grain": "वाढणाऱ्या फळांवर/दाण्यांवर किडींचा हल्ला",
            "Late rains causing quality deterioration": "उशिरा पाऊस पडल्यास प्रत खराब होण्याचा धोका",
            "Heat Stress": "उष्णतेचा ताण",
            "Apply light irrigation in the evening to cool the micro-climate.": "पीक थंड ठेवण्यासाठी सायंकाळी हलके पाणी द्या.",
            "Stage-specific management is the core of precision agriculture.": "पिकाच्या अवस्थेनुसार व्यवस्थापन करणे हेच अचूक शेतीचे सूत्र आहे.",

            # Alerts
            "Extreme heat detected. Plants may show temporary wilting.": "☀️ कडक ऊन! झाडे कोमेजण्याची शक्यता आहे. हलके पाणी द्या.",
            "Light rain expected. Good for top-dressing fertilizer, but avoid foliar sprays.": "🌦️ हलक्या पावसाची शक्यता. खते देण्यास योग्य वेळ, पण फवारणी टाळा.",
            "Heavy rainfall expected. Do not irrigate. Postpone fertilizer/pesticide sprays.": "🚨 मुसळधार पाऊस अपेक्षित! पाणी देऊ नका आणि फवारणी पुढे ढकला.",
            "Rainfall": "पाऊस",

            # Input Sections Labels (Handled in frontend but keys might be used)
            "fertilizer": "खत व्यवस्थापन",
            "herbicide": "तणनाशक (औषध)",
            "pesticide": "कीटकनाशक / बुरशीनाशक"
        }

        def translate(text):
            if not text: return text
            # 1. Try exact match first
            if text in mr_dict:
                return mr_dict[text]
            
            # 2. Try substring replacements for more complex items
            processed = text
            # Sort by length descending to match longest phrases first
            for eng in sorted(mr_dict.keys(), key=len, reverse=True):
                if eng in processed:
                    processed = processed.replace(eng, mr_dict[eng])
            return processed

        # Apply translations
        result["stage"]["label"] = translate(result["stage"]["label"])
        result["stage"]["stage_family_label"] = mr_dict.get(result["stage"].get("stage_family"), "अवस्था")
        
        result["today_actions"] = [translate(a) for a in result["today_actions"]]
        result["next_5_day_plan"] = [translate(a) for a in result["next_5_day_plan"]]
        result["focus"] = translate(result["focus"])
        result["ai_recommendation"] = translate(result["ai_recommendation"])
        result["why_this_matters"] = translate(result["why_this_matters"])
        
        for risk in result["risks"]:
            risk["severity"] = mr_dict.get(risk.get("severity"), risk.get("severity"))
            risk["reason"] = translate(risk.get("reason", ""))
            risk["recommended_response"] = translate(risk.get("recommended_response", ""))

        mr_alerts = []
        for alert in result["alerts"]:
            if "Rainfall" in alert or "Heavy rain" in alert:
                mr_alerts.append("🚨 आज पाऊस येण्याची शक्यता आहे, पाणी देणे व फवारणी टाळा.")
            elif "Light rain" in alert:
                mr_alerts.append("🌦️ हलक्या पावसाची शक्यता. खत टाकण्यास योग्य पण फवारणी करू नका.")
            elif "Extreme heat" in alert:
                mr_alerts.append("☀️ कडक ऊन! पिकाला पाण्याचा ताण पडू देऊ नका.")
            else:
                mr_alerts.append(translate(alert))
        result["alerts"] = mr_alerts

        # Inputs Translation
        if "inputs" in result:
            result["inputs"]["fertilizer"] = translate(result["inputs"]["fertilizer"])
            result["inputs"]["herbicide"] = translate(result["inputs"]["herbicide"])
            result["inputs"]["pesticide"] = translate(result["inputs"]["pesticide"])

        result["confidence_note"] = "🌟 तुम्ही खूप मेहनत करत आहात! आम्ही तुम्हाला योग्य दिशा दाखवत राहू."
