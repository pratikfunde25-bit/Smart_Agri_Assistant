# 🌾 Bilingual Agronomy Intelligence: Strategy & Implementation

This plan outlines the transformation of the Smart Agri Assistant into a **Bilingual (English + Marathi)** platform, optimized for rural Marathi-speaking farmers using "Farmer-First" linguistics.

## 1. 🏗️ Architecture: "Marathi-First" Logic
We move away from robotic translation. The system will use two layers of localization:
1.  **Static UI Localization**: Hand-crafted JSON for buttons, headers, and labels.
2.  **Dynamic Intelligence Localization**: The `AdvisorEngineV2` will detect the `lang` parameter and fetch Marathi-optimized advice directly from a bilingual knowledge base.

## 2. 🗣️ Language Style Guide (Marathi)
We follow the **"Smart Shetkari"** tone: short, clear, and actionable.

| English | ❌ Robotic Marathi | ✅ Farmer Marathi |
| :--- | :--- | :--- |
| **Stage: Flowering** | फुलांची अवस्था | **फुलोरा** |
| **Apply Urea** | युरिया लागू करा | **युरिया खत टाका** |
| **Irrigation required** | सिंचन आवश्यक आहे | **पाणी द्या** |
| **Weather Alert: Rain** | पर्जन्यवृष्टीचा इशारा | **आज पाऊस येईल, पाणी देऊ नका** |
| **Next 5 Days** | पुढील पाच दिवस | **पुढचे ५ दिवस** |

## 3. 📂 Data Structure: `mr.json` (Sample)
```json
{
  "ui": {
    "dashboard_title": "आज काय करायचं?",
    "next_5_days": "पुढचे ५ दिवस लक्ष द्या",
    "risks": "धोके",
    "ai_strategy": "सल्ला",
    "toggle": "मराठी | EN"
  },
  "crop_maize": {
    "stages": {
      "VE": "रोपांची उगवण",
      "V6": "गुडघाभर उंची",
      "VT": "तुरा येणे",
      "R1": "कणीस बाहेर येणे (सिल्किंग)",
      "R6": "कापणीची वेळ"
    }
  }
}
```

## 4. 🚀 Implementation Roadmap

### Phase 1: Backend "Bilingual Mode"
*   Update `src/advisor_v2.py` to accept `lang` parameter.
*   Integrate a Marathi mapping for Maize (the production crop).
*   Add logic: `if lang == 'mr'`, overwrite English fields with Marathi equivalents before returning JSON.

### Phase 2: Frontend Language Context
*   Add a **"Language Switcher"** to the `crop_advisor.html` header.
*   Store the preference in `localStorage`.
*   Update the JS `state` to include `lang: 'en'`.

### Phase 3: Dashboard Localizer
*   Implement a `t(key)` function in JavaScript.
*   Update all `textContent` assignments to use the localizer.
*   Ensure the horizontal stepper shows Marathi stage labels in Marathi mode.
