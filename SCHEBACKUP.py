from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PropType = Literal["hits", "runs", "homeruns", "rbi", "k", "bb", "er"]
Confidence = Literal["high", "med", "low"]


class PropQuote(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prob: float
    odds: int | None = None
    bookOdds: int | None = None
    noOdds: int | None = None
    conf: Confidence = "low"
    edge: float | None = None
    marketAvailable: bool = False
    modelOnly: bool = True


class PlayerStats(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    xwoba: float | None = None
    xba: float | None = None
    barrel: float | None = None
    hardhit: float | None = None
    whiff: float | None = None
    babip14d: float | None = None
    iso: float | None = None
    plateAppearances: int = 0
    atBats: int = 0


class PlayerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    mlbId: int
    name: str
    team: str
    teamName: str
    gameId: str
    oppPitcher: str | None = None
    oppPitcherId: int | None = None
    order: int | None = None
    bats: str = "R"
    projectedPA: float = 0.0
    props: dict[str, PropQuote]
    stats: PlayerStats


class PitcherSplit(BaseModel):
    baa: float
    k: float
    bb: float
    hr: float
    bf: int


class PitcherLadderLine(BaseModel):
    line: float
    prob: float
    odds: int | None = None


class PitcherResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    team: str
    throws: str
    projK: float
    projER: float
    era: float
    whip: float
    statsSource: str
    splits: dict[str, PitcherSplit]
    propLadder: dict[str, list[PitcherLadderLine]]


class GameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    gamePk: int
    date: str
    venue: str
    away: str
    home: str
    awayName: str
    homeName: str
    awayProb: float
    homeProb: float
    awayOdds: int | None = None
    homeOdds: int | None = None
    awayBookOdds: int | None = None
    homeBookOdds: int | None = None
    awayEdge: float | None = None
    homeEdge: float | None = None
    awayPitcher: str | None = None
    homePitcher: str | None = None
    awayPitcherId: int | None = None
    homePitcherId: int | None = None
    awayXRuns: float
    homeXRuns: float
    # NEW: season-series / recent head-to-head context for the Moneylines
    # sidebar subtitle (replaces the redundant pitcher-vs-pitcher line that
    # already appears on the main game card). All optional — populate from
    # prediction_service/mlb_client once wired; the frontend degrades
    # gracefully when these are absent.
    awaySeasonSeriesWins: int | None = None
    awaySeasonSeriesLosses: int | None = None
    recentH2H: str | None = None


class SlateMeta(BaseModel):
    date: str
    generatedAt: str
    season: int
    playerCount: int
    gameCount: int
    marketDataAvailable: bool


class SlateResponse(BaseModel):
    meta: SlateMeta
    players: list[PlayerResponse]
    games: list[GameResponse]


class CalibrationPoint(BaseModel):
    bucket: str
    predicted: float
    actual: float
    n: int


class PredictionLogItem(BaseModel):
    date: str
    player: str
    stat: str
    predicted: float
    actual: int
    correct: bool


class TrackRecordResponse(BaseModel):
    calibration: list[CalibrationPoint]
    log: list[PredictionLogItem]


class CalibrationRequest(BaseModel):
    player_id: int = Field(gt=0)
    prop_type: PropType = "hits"
    xwoba: float = Field(ge=0.0, le=1.0)
    xba: float = Field(ge=0.0, le=1.0)
    barrel: float = Field(ge=0.0, le=100.0)
    hardhit: float = Field(ge=0.0, le=100.0)
    whiff: float = Field(ge=0.0, le=100.0)


class CalibrationResponse(BaseModel):
    originalProb: float
    modifiers: dict[str, float]
    totalAdjustment: float
    finalProb: float
    fairOdds: int


# --- NEW: Matchup Verifier (Issue #2) ---

class PlatoonLine(BaseModel):
    baa: float
    kPct: float
    bbPct: float
    hrPct: float


class HeadToHead(BaseModel):
    hasData: bool
    atBats: int = 0
    hits: int = 0
    homeRuns: int = 0
    strikeouts: int = 0
    avg: float = 0.0
    obp: float = 0.0


class MatchupResponse(BaseModel):
    batterId: int
    batterName: str
    batterHand: str
    batterAvg: float
    batterKRate: float
    pitcherId: int
    pitcherName: str
    pitcherHand: str
    pitcherBaa: float
    pitcherKRate: float
    platoonVsLeft: PlatoonLine
    platoonVsRight: PlatoonLine
    headToHead: HeadToHead
    advantageScore: float
    verdict: Literal["batter", "pitcher", "neutral"]
    verdictLabel: str