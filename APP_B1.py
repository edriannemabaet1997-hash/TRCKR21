from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from math_engine import calibrate_with_statcast_full, prob_to_american
from mlb_client import MLBClient
from odds_client import OddsClient
from prediction_service import PredictionService
from repository import PredictionRepository
from schemas import (
    CalibrationRequest,
    CalibrationResponse,
    GameResponse,
    MatchupResponse,
    PitcherResponse,
    PlayerResponse,
    SlateResponse,
    TrackRecordResponse,
)

app = FastAPI(
    title="trckr21 MLB Quant API",
    version="1.0.0",
    description="Live MLB prediction API for the trckr21 Quant Terminal.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlb_client = MLBClient()
odds_client = OddsClient()
repository = PredictionRepository()

prediction_service = PredictionService(
    mlb=mlb_client,
    odds=odds_client,
    repository=repository,
)


def eastern_today() -> str:
    return datetime.now(ZoneInfo(settings.timezone_name)).date().isoformat()


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def serve_frontend():
    return FileResponse("index.html")


@app.get("/api/info")
def api_info() -> dict:
    return {"name": "trckr21 MLB Quant API", "version": app.version, "status": "online", "docs": "/docs"}


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok", "season": settings.season,
        "oddsConfigured": odds_client.enabled, "timezone": settings.timezone_name,
    }


@app.get("/api/slate", response_model=SlateResponse)
def get_slate(
    date: str | None = Query(default=None, description="Slate date in YYYY-MM-DD format."),
    refresh: bool = Query(default=False, description="Ignore the in-memory cache."),
) -> dict:
    target_date = date or eastern_today()
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Date must use YYYY-MM-DD format.") from exc
    try:
        return prediction_service.build_slate(target_date=target_date, force=refresh)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to build MLB slate: {exc}") from exc


@app.get("/api/players", response_model=list[PlayerResponse])
def get_players(date: str | None = None, refresh: bool = False) -> list[dict]:
    slate = get_slate(date=date, refresh=refresh)
    return slate["players"]


@app.get("/api/players/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, date: str | None = None) -> dict:
    target_date = date or eastern_today()
    slate = prediction_service.build_slate(target_date)
    player = next((item for item in slate["players"] if item["mlbId"] == player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail="Player is not present on this slate.")
    return player


@app.get("/api/games", response_model=list[GameResponse])
def get_games(date: str | None = None, refresh: bool = False) -> list[dict]:
    slate = get_slate(date=date, refresh=refresh)
    return slate["games"]


@app.get("/api/pitchers/{pitcher_id}", response_model=PitcherResponse)
def get_pitcher(pitcher_id: int) -> dict:
    try:
        pitcher = prediction_service.get_pitcher(pitcher_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load pitcher data: {exc}") from exc
    if pitcher is None:
        raise HTTPException(status_code=404, detail="Pitcher not found.")
    return pitcher


# --- NEW: Matchup Verifier endpoint (Issue #2) ---
@app.get("/api/matchup/{batter_id}/{pitcher_id}", response_model=MatchupResponse)
def get_matchup(batter_id: int, pitcher_id: int) -> dict:
    try:
        return prediction_service.get_matchup(batter_id, pitcher_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to build matchup: {exc}") from exc


@app.post("/api/calibrate", response_model=CalibrationResponse)
def calibrate(request: CalibrationRequest) -> dict:
    player = prediction_service.get_indexed_player(request.player_id)
    if player is None:
        slate = prediction_service.build_slate(eastern_today())
        player = next((item for item in slate["players"] if item["mlbId"] == request.player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail="Player is not present on the current slate.")
    prop = player["props"].get(request.prop_type)
    if prop is None:
        raise HTTPException(status_code=404, detail="Requested prop is unavailable.")
    result = calibrate_with_statcast_full(
        original_prob=float(prop["prob"]), xwoba=request.xwoba, xba=request.xba,
        barrel=request.barrel, hard_hit=request.hardhit, whiff=request.whiff,
    )
    return {
        "originalProb": result["original_prob"],
        "modifiers": {
            "xwoba": result["mod_xwoba"], "xba": result["mod_xba"], "barrel": result["mod_barrel"],
            "hardhit": result["mod_hardhit"], "whiff": result["mod_whiff"],
        },
        "totalAdjustment": result["total_adjustment"],
        "finalProb": result["final_prob"],
        "fairOdds": prob_to_american(result["final_prob"]),
    }


@app.get("/api/track-record", response_model=TrackRecordResponse)
def track_record(limit: int = Query(default=100, ge=1, le=1000)) -> dict:
    return repository.track_record(limit=limit)


@app.post("/api/results/sync")
def sync_results() -> dict:
    try:
        return prediction_service.sync_results()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Result synchronization failed: {exc}") from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error.", "error": str(exc)})