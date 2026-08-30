from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


def _get_int(env_key: str, default: int) -> int:
    val = os.getenv(env_key)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(env_key: str, default: float) -> float:
    val = os.getenv(env_key)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _clean_url(url: str) -> str:
    url = url.strip().rstrip("/")
    # Tinatanggal ang accidental Markdown link formatting kung may pumasok
    if "[" in url and "]" in url:
        url = url.split("(")[-1].rstrip(")")
    return url.rstrip("/")


@dataclass(frozen=True)
class Settings:
    mlb_api_base: str = _clean_url(
        os.getenv("MLB_API_BASE", "https://statsapi.mlb.com/api/v1")
    )
    odds_api_base: str = _clean_url(
        os.getenv(
            "ODDS_API_BASE",
            "https://api.the-odds-api.com/v4/sports/baseball_mlb",
        )
    )
    # FIX (2026-08-29): no hardcoded fallback key. A live key checked into
    # source is a leaked secret the moment this repo is shared/pushed
    # anywhere public — rotate the old one on the-odds-api.com if it was
    # ever committed. odds_client.enabled already treats an empty key as
    # "odds feature off" (no crash, market data just isn't attached),
    # so this is a safe, non-breaking change. Set ODDS_API_KEY in your
    # .env file instead.
    odds_api_key: str = os.getenv("ODDS_API_KEY", "").strip()
    database_path: str = os.getenv(
        "DATABASE_PATH",
        "mlb_quant.sqlite3",
    ).strip()
    model_path: str = os.getenv(
        "MODEL_PATH",
        "mlb_calibrated_model.pkl",
    ).strip()
    timezone_name: str = os.getenv(
        "APP_TIMEZONE",
        "America/New_York",
    ).strip()
    
    request_timeout: float = _get_float("REQUEST_TIMEOUT", 10.0)
    max_workers: int = _get_int("MAX_WORKERS", 12)
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv("CORS_ORIGINS", "*").split(",")
        if item.strip()
    )

    @property
    def season(self) -> int:
        configured = os.getenv("MLB_SEASON")
        if configured:
            try:
                return int(configured)
            except ValueError:
                pass
        try:
            return datetime.now(ZoneInfo(self.timezone_name)).year
        except Exception:
            return datetime.now().year


settings = Settings()