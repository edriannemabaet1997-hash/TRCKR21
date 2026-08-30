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
    # "proxy" = estimated from box-score fields (no free-tier Statcast feed
    # for xwoba/xba/barrel%/hardhit%); "manual" reserved for a future
    # user-entered override path. Frontend should badge non-"proxy" sources
    # differently once/if a real Statcast feed is wired in.
    statsSource: str = "proxy"
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


# --- NEW: consolidation (2026-08-29) — Pitcher Props / Team Matchups /
# Matchup Analyzer, ported from the retired generate_projections.py +
# projections.json contract so index.html's BatterAPI can call these live
# instead of fetching a static file. Field names match the frontend's
# existing `market.history` / `market.model.*` / `matchups[].batters[].*`
# access patterns 1:1 so the only frontend change needed is the fetch
# source (see PROJECTIONS_URL removal note).


class MarketModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    distribution: Literal["poisson", "negative_binomial"]
    mu: float
    observedMu: float
    variance: float
    vmr: float
    overProb: float
    fairOdds: int | None = None
    impliedProbMarket: float
    edgePct: float
    kellyPct: float
    confidence: Literal["high", "medium", "low"]
    sampleSize: int
    fallback: bool = False
    # None for pitcher-props markets (no second projection model to
    # reconcile with); set for team F5/full-game runs markets, where it's
    # the calculate_team_xruns_v2-derived shrinkage anchor.
    anchorMu: float | None = None


class MarketData(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    defaultLine: float
    history: list[int]
    max: int
    model: MarketModelOut


class MatchDetail(BaseModel):
    score: str
    text: str


class PitcherPropsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    sub: str
    opponent: str
    isHome: bool
    matchup: str
    markets: dict[str, MarketData]
    labels: list[str]
    matchDetails: list[MatchDetail]


class TeamMatchupsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    sub: str
    opponent: str
    isHome: bool
    matchup: str
    markets: dict[str, MarketData]
    labels: list[str]
    matchDetails: list[MatchDetail]
    startTimeUTC: str | None = None
    venue: str | None = None


class PitchMixEntry(BaseModel):
    type: str
    code: str
    usagePct: float
    avgVelo: float
    whiffPct: float | None = None
    chasePct: float | None = None
    putAwayPct: float | None = None
    hardHitPct: float | None = None
    lethalitySample: int = 0


class PitchVulnEntry(BaseModel):
    type: str
    sampleSize: int
    whiffPct: float | None = None
    chasePct: float | None = None
    hardHitPct: float | None = None
    confidenceTier: Literal["high", "medium", "low"]


class PlatoonSlash(BaseModel):
    avg: str
    ops: float
    pa: int
    hr: int


class BatterPlatoonSplits(BaseModel):
    vsRHP: PlatoonSlash
    vsLHP: PlatoonSlash


class RecentFormSlash(BaseModel):
    avg: str
    ops: float
    pa: int
    hr: int
    tier: Literal["hot", "cold", "neutral", "limited"]
    days: int


class H2HSlash(BaseModel):
    pa: int
    ab: int
    h: int
    hr: int
    bb: int
    so: int
    avg: str
    obp: str
    slg: str
    ops: float


class MatchupBatter(BaseModel):
    id: int
    name: str
    order: int
    h2h: H2HSlash
    seasonOps: float
    seasonPa: float
    seasonKPct: float
    seasonWhiff: float
    seasonChase: float
    seasonCsw: float
    pitchVulnerability: dict[str, PitchVulnEntry]
    platoonSplits: BatterPlatoonSplits
    recentForm: RecentFormSlash
    credibility: float
    blendedOps: float
    proxyXba: float
    proxyXslg: float
    edgeTier: Literal["elite", "strong", "average", "soft"]


class MatchupAnalyzerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    pitcherId: str
    pitcherName: str
    team: str
    opponent: str
    isHome: bool
    matchup: str
    sub: str
    lineupConfirmed: bool
    pitchMix: list[PitchMixEntry]
    batters: list[MatchupBatter]
    lineupEdgeOps: float
    lineupEdgeTier: Literal["elite", "strong", "average", "soft"]
    totalH2hPa: int
    pitcherEra: float | None = None
    pitcherWhip: float | None = None
    pitcherArsenalWhiffPct: float | None = None


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