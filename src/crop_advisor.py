import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

class CropAdvisor:
    def __init__(self):
        self.data_path = Path(__file__).resolve().parents[1] / "data" / "crop_guidelines.json"
        self.guidelines = self._load_guidelines()

    def _load_guidelines(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            return {}
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_supported_crops_metadata(self) -> List[Dict[str, Any]]:
        """Returns a list of crops with their emoji and category for the UI selection grid."""
        metadata = []
        for name, data in self.guidelines.items():
            metadata.append({
                "name": name,
                "emoji": data.get("emoji", "🌱"),
                "category": data.get("category", "General")
            })
        return sorted(metadata, key=lambda x: (x["category"], x["name"]))

    def get_guidance(self, crop_name: str, sowing_date_str: Optional[str] = None, stage_idx: Optional[int] = None) -> Dict[str, Any]:
        if crop_name not in self.guidelines:
            return {"error": f"Guidance not found for crop: {crop_name}"}

        crop_info = self.guidelines[crop_name]
        total_days = crop_info["total_days"]
        stages = crop_info["stages"]
        
        # Base info
        result = {
            "crop": crop_name,
            "emoji": crop_info.get("emoji", "🌱"),
            "total_days": total_days,
            "all_stages": stages,
            "num_stages": len(stages)
        }

        current_day = -1
        if sowing_date_str:
            try:
                sowing_date_obj = datetime.strptime(sowing_date_str, "%Y-%m-%d").date()
                today = date.today()
                delta = today - sowing_date_obj
                current_day = delta.days + 1
            except Exception as e:
                return {"error": f"Invalid date format: {e}"}

        result["current_day_live"] = current_day

        # Determine which stage to show
        selected_idx = 0
        
        # If stage_idx is provided, use it (Explorer Mode)
        if stage_idx is not None:
            selected_idx = max(0, min(stage_idx, len(stages) - 1))
            result["mode"] = "explorer"
        # If no stage_idx but sowing_date is provided (Live Mode)
        elif current_day != -1:
            result["mode"] = "live"
            if current_day < 0:
                selected_idx = 0 # Pre-sowing or just started
            elif current_day > total_days:
                selected_idx = len(stages) - 1 # Completed
            else:
                for idx, stage in enumerate(stages):
                    if stage["start_day"] <= current_day <= stage["end_day"]:
                        selected_idx = idx
                        break
        
        result["selected_stage_idx"] = selected_idx
        result["current_stage"] = stages[selected_idx]
        
        # Navigation flags
        result["has_next"] = selected_idx < len(stages) - 1
        result["has_prev"] = selected_idx > 0
        
        # Progress calculation for Live Mode
        if current_day != -1:
             result["progress_pct"] = min(100, max(0, round((current_day / total_days) * 100, 1)))
        else:
             result["progress_pct"] = 0

        return result
