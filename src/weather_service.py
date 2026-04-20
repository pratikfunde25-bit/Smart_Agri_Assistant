from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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


def weather_lookup_available() -> bool:
    return True


def _request_json(endpoint: str, params: Dict[str, Any], timeout: int = 12) -> Dict[str, Any]:
    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherServiceError(f"Unable to reach the weather service: {exc}") from exc
    return response.json()


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
