from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class WeatherServiceError(RuntimeError):
    pass


@dataclass
class WeatherSnapshot:
    latitude: float
    longitude: float
    location_name: str
    country_code: str
    temperature: float
    humidity: float
    rainfall: float
    rainfall_window: str
    conditions: str
    weather_group: str
    wind_speed: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_name": self.location_name,
            "country_code": self.country_code,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "rainfall": self.rainfall,
            "rainfall_window": self.rainfall_window,
            "conditions": self.conditions,
            "weather_group": self.weather_group,
            "wind_speed": self.wind_speed,
        }


@dataclass
class ForecastDay:
    date: str
    weather_code: int
    conditions: str
    weather_group: str
    temp_max: float
    temp_min: float
    precipitation_mm: float
    precipitation_probability_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "weather_code": self.weather_code,
            "conditions": self.conditions,
            "weather_group": self.weather_group,
            "temp_max": self.temp_max,
            "temp_min": self.temp_min,
            "precipitation_mm": self.precipitation_mm,
            "precipitation_probability_pct": self.precipitation_probability_pct,
        }


@dataclass
class AdvisoryWeatherContext:
    source: str
    current: Optional[WeatherSnapshot]
    forecast_days: List[ForecastDay]
    signals: Dict[str, Any]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "current": self.current.to_dict() if self.current else None,
            "forecast_days": [item.to_dict() for item in self.forecast_days],
            "signals": self.signals,
            "summary": self.summary,
        }


def weather_lookup_available() -> bool:
    return True


def _request_json(endpoint: str, params: Dict[str, Any], timeout: int = 12) -> Dict[str, Any]:
    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherServiceError(f"Unable to reach the weather service: {exc}") from exc
    return response.json()


def _sum_precipitation(days: List[ForecastDay], count: int) -> float:
    return round(sum(item.precipitation_mm for item in days[:count]), 2)


def _max_probability(days: List[ForecastDay], count: int) -> float:
    if not days:
        return 0.0
    return max(float(item.precipitation_probability_pct) for item in days[:count])


def _build_weather_signals(current: WeatherSnapshot | None, forecast_days: List[ForecastDay]) -> Dict[str, Any]:
    today = forecast_days[0] if forecast_days else None
    tomorrow = forecast_days[1] if len(forecast_days) > 1 else None
    current_temp = current.temperature if current else 0.0
    current_humidity = current.humidity if current else 0.0
    current_wind = current.wind_speed if current else 0.0

    rain_24h = _sum_precipitation(forecast_days, 1)
    rain_48h = _sum_precipitation(forecast_days, 2)
    rain_5d = _sum_precipitation(forecast_days, 5)
    rain_today = bool(today and (today.precipitation_mm >= 5 or today.precipitation_probability_pct >= 60))
    rain_tomorrow = bool(tomorrow and (tomorrow.precipitation_mm >= 5 or tomorrow.precipitation_probability_pct >= 60))
    heavy_rain_24h = rain_24h >= 20
    heavy_rain_48h = rain_48h >= 25
    heat_stress = bool(today and (today.temp_max >= 36 or current_temp >= 36))
    cold_stress = bool(today and (today.temp_min <= 12 or current_temp <= 12))
    humidity_pressure = current_humidity >= 85
    spray_unsuitable = current_wind >= 18 or rain_today

    return {
        "rain_expected_today": rain_today,
        "rain_expected_tomorrow": rain_tomorrow,
        "rain_expected_24h_mm": rain_24h,
        "rain_expected_48h_mm": rain_48h,
        "rain_expected_5d_mm": rain_5d,
        "heavy_rain_24h": heavy_rain_24h,
        "heavy_rain_48h": heavy_rain_48h,
        "max_rain_probability_48h_pct": _max_probability(forecast_days, 2),
        "heat_stress": heat_stress,
        "cold_stress": cold_stress,
        "humidity_pressure": humidity_pressure,
        "spray_unsuitable": spray_unsuitable,
        "irrigation_opportunity_window": not rain_today and not heavy_rain_24h,
    }


def _build_weather_summary(
    latitude: float,
    longitude: float,
    current: WeatherSnapshot | None,
    forecast_days: List[ForecastDay],
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    headline_parts = []
    if signals["rain_expected_today"]:
        headline_parts.append("Rain signal today")
    elif signals["rain_expected_tomorrow"]:
        headline_parts.append("Rain likely by tomorrow")
    else:
        headline_parts.append("No strong rain signal in the next 24 hours")

    if signals["heat_stress"]:
        headline_parts.append("heat stress risk")
    if signals["humidity_pressure"]:
        headline_parts.append("humid canopy pressure")

    return {
        "location_label": current.location_name if current else f"{latitude:.4f}, {longitude:.4f}",
        "headline": ", ".join(headline_parts),
        "current_temperature_c": current.temperature if current else None,
        "current_humidity_pct": current.humidity if current else None,
        "wind_kph": current.wind_speed if current else None,
        "rain_24h_mm": signals["rain_expected_24h_mm"],
        "rain_48h_mm": signals["rain_expected_48h_mm"],
        "forecast_days": [item.to_dict() for item in forecast_days[:5]],
    }


def _describe_weather_code(code: int) -> tuple[str, str]:
    code_map = {
        0: ("Clear Sky", "Clear"),
        1: ("Mainly Clear", "Clear"),
        2: ("Partly Cloudy", "Clouds"),
        3: ("Overcast", "Clouds"),
        45: ("Fog", "Fog"),
        48: ("Depositing Rime Fog", "Fog"),
        51: ("Light Drizzle", "Drizzle"),
        53: ("Moderate Drizzle", "Drizzle"),
        55: ("Dense Drizzle", "Drizzle"),
        56: ("Light Freezing Drizzle", "Drizzle"),
        57: ("Dense Freezing Drizzle", "Drizzle"),
        61: ("Slight Rain", "Rain"),
        63: ("Moderate Rain", "Rain"),
        65: ("Heavy Rain", "Rain"),
        66: ("Light Freezing Rain", "Rain"),
        67: ("Heavy Freezing Rain", "Rain"),
        71: ("Slight Snow Fall", "Snow"),
        73: ("Moderate Snow Fall", "Snow"),
        75: ("Heavy Snow Fall", "Snow"),
        77: ("Snow Grains", "Snow"),
        80: ("Slight Rain Showers", "Rain"),
        81: ("Moderate Rain Showers", "Rain"),
        82: ("Violent Rain Showers", "Rain"),
        85: ("Slight Snow Showers", "Snow"),
        86: ("Heavy Snow Showers", "Snow"),
        95: ("Thunderstorm", "Thunderstorm"),
        96: ("Thunderstorm With Hail", "Thunderstorm"),
        99: ("Severe Thunderstorm With Hail", "Thunderstorm"),
    }
    return code_map.get(code, ("Weather data available", "Unknown"))


class OpenMeteoService:
    endpoint = "https://api.open-meteo.com/v1/forecast"

    def fetch_current_weather(self, latitude: float, longitude: float) -> WeatherSnapshot:
        payload = _request_json(
            self.endpoint,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
        )

        current = payload.get("current") or {}
        if not current:
            raise WeatherServiceError("Weather API did not return current conditions for this location.")

        weather_code = int(current.get("weather_code", -1))
        conditions, weather_group = _describe_weather_code(weather_code)

        return WeatherSnapshot(
            latitude=float(payload.get("latitude", latitude)),
            longitude=float(payload.get("longitude", longitude)),
            location_name=f"{latitude:.4f}, {longitude:.4f}",
            country_code="",
            temperature=float(current.get("temperature_2m", 0.0)),
            humidity=float(current.get("relative_humidity_2m", 0.0)),
            rainfall=float(current.get("precipitation", 0.0)),
            rainfall_window="current",
            conditions=conditions,
            weather_group=weather_group,
            wind_speed=float(current.get("wind_speed_10m", 0.0)),
        )

    def fetch_advisory_weather(self, latitude: float, longitude: float) -> AdvisoryWeatherContext:
        payload = _request_json(
            self.endpoint,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                "forecast_days": 6,
                "timezone": "auto",
            },
        )

        current = self.fetch_current_weather(latitude=latitude, longitude=longitude)
        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        weather_codes = daily.get("weather_code") or []
        temp_max = daily.get("temperature_2m_max") or []
        temp_min = daily.get("temperature_2m_min") or []
        precipitation_sum = daily.get("precipitation_sum") or []
        precipitation_probability = daily.get("precipitation_probability_max") or []

        forecast_days: List[ForecastDay] = []
        for index, date_value in enumerate(times):
            code = int(weather_codes[index]) if index < len(weather_codes) else -1
            conditions, weather_group = _describe_weather_code(code)
            forecast_days.append(
                ForecastDay(
                    date=str(date_value),
                    weather_code=code,
                    conditions=conditions,
                    weather_group=weather_group,
                    temp_max=float(temp_max[index]) if index < len(temp_max) else 0.0,
                    temp_min=float(temp_min[index]) if index < len(temp_min) else 0.0,
                    precipitation_mm=float(precipitation_sum[index]) if index < len(precipitation_sum) else 0.0,
                    precipitation_probability_pct=float(precipitation_probability[index])
                    if index < len(precipitation_probability)
                    else 0.0,
                )
            )

        signals = _build_weather_signals(current=current, forecast_days=forecast_days)
        summary = _build_weather_summary(
            latitude=latitude,
            longitude=longitude,
            current=current,
            forecast_days=forecast_days,
            signals=signals,
        )
        return AdvisoryWeatherContext(
            source="open-meteo",
            current=current,
            forecast_days=forecast_days,
            signals=signals,
            summary=summary,
        )


class OpenWeatherService:
    endpoint = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: str, units: str = "metric") -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.units = units

    def _extract_rainfall(self, payload: Dict[str, Any]) -> tuple[float, str]:
        rain = payload.get("rain") or {}
        if "1h" in rain:
            return float(rain["1h"]), "1h"
        if "3h" in rain:
            return float(rain["3h"]), "3h"
        return 0.0, "none"

    def fetch_current_weather(self, latitude: float, longitude: float) -> WeatherSnapshot:
        if not self.api_key:
            return OpenMeteoService().fetch_current_weather(latitude=latitude, longitude=longitude)

        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": self.api_key,
            "units": self.units,
        }

        payload = _request_json(self.endpoint, params=params)
        if str(payload.get("cod", "200")) != "200":
            raise WeatherServiceError(payload.get("message", "Weather API returned an unexpected response."))

        main = payload.get("main") or {}
        weather = (payload.get("weather") or [{}])[0]
        rainfall, rainfall_window = self._extract_rainfall(payload)

        return WeatherSnapshot(
            latitude=float((payload.get("coord") or {}).get("lat", latitude)),
            longitude=float((payload.get("coord") or {}).get("lon", longitude)),
            location_name=str(payload.get("name") or "Detected location"),
            country_code=str((payload.get("sys") or {}).get("country") or ""),
            temperature=float(main.get("temp", 0.0)),
            humidity=float(main.get("humidity", 0.0)),
            rainfall=float(rainfall),
            rainfall_window=rainfall_window,
            conditions=str(weather.get("description") or "unavailable").title(),
            weather_group=str(weather.get("main") or "Unknown"),
            wind_speed=float((payload.get("wind") or {}).get("speed", 0.0)),
        )

    def fetch_advisory_weather(self, latitude: float, longitude: float) -> AdvisoryWeatherContext:
        fallback = OpenMeteoService().fetch_advisory_weather(latitude=latitude, longitude=longitude)
        if not self.api_key:
            return fallback

        current = self.fetch_current_weather(latitude=latitude, longitude=longitude)
        current.wind_speed = round(current.wind_speed * 3.6, 2)
        signals = _build_weather_signals(current=current, forecast_days=fallback.forecast_days)
        summary = _build_weather_summary(
            latitude=latitude,
            longitude=longitude,
            current=current,
            forecast_days=fallback.forecast_days,
            signals=signals,
        )
        return AdvisoryWeatherContext(
            source="open-weather+open-meteo",
            current=current,
            forecast_days=fallback.forecast_days,
            signals=signals,
            summary=summary,
        )
