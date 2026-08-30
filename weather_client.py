# weather_client.py
# TASK 1 (2026-08-30) — weather_mult data source. Uses Open-Meteo
# (open-meteo.com), free and no API key required, per spec. Pure HTTP-shape
# fetch only, same house convention as mlb_client.py / odds_client.py: unit
# conversion and the actual scoring-multiplier math live in math_engine.py
# (weather_scoring_multiplier), not here. Venue coordinates + park azimuth
# come from mlb_client.MLBClient.venue() — this module only needs lat/lon
# and a target time, it never touches the MLB Stats API itself.

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from config import settings

logger = logging.getLogger("trckr21.weather_client")

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "trckr21-mlb-quant/1.0"})

    def forecast_at_game_time(
        self, latitude: float, longitude: float, game_time_utc: str | None
    ) -> dict | None:
        """Fetch the hourly temperature/wind forecast nearest to
        `game_time_utc` (an ISO-8601 UTC string — MLB Stats API's
        `gameDate` field is already in this format) for the given venue
        coordinates.

        Returns {"temperature_f": float, "wind_speed_mph": float,
        "wind_direction_deg": float} or None on any fetch/parse failure so
        callers fall back to a neutral weather_mult — same "swallow to a
        safe default, never raise into the request path" convention as
        every other optional signal in this codebase (e.g.
        pitcher_fatigue_and_velocity, hitter_ahead_in_count_avg).
        """
        if not game_time_utc:
            return None
        try:
            target = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            logger.warning("Could not parse game_time_utc=%r for weather forecast.", game_time_utc)
            return None

        try:
            response = self.session.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "timezone": "UTC",
                    "start_date": target.date().isoformat(),
                    "end_date": target.date().isoformat(),
                },
                timeout=settings.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as e:
            logger.warning("🚨 Open-Meteo fetch failed for lat=%s lon=%s: %s", latitude, longitude, e)
            return None
        except ValueError as e:
            logger.warning("🚨 Open-Meteo returned non-JSON for lat=%s lon=%s: %s", latitude, longitude, e)
            return None

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        wind_dirs = hourly.get("wind_direction_10m", [])
        if not times:
            return None

        # Open-Meteo returns each hourly bucket as a naive "YYYY-MM-
        # DDTHH:MM" string in the requested `timezone` (UTC here, matching
        # `target` since we already converted) — pick whichever bucket is
        # closest to game time.
        best_idx, best_diff = 0, None
        for i, t in enumerate(times):
            try:
                bucket = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            diff = abs((bucket - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best_idx, best_diff = i, diff

        try:
            return {
                "temperature_f": float(temps[best_idx]),
                "wind_speed_mph": float(wind_speeds[best_idx]),
                "wind_direction_deg": float(wind_dirs[best_idx]),
            }
        except (IndexError, TypeError, ValueError) as e:
            logger.warning("Open-Meteo forecast payload missing expected fields at idx=%s: %s", best_idx, e)
            return None