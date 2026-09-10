# prediction_service.py — full ABI hit-probability port from the original
# 2000+ line Streamlit backend. Two-stage pipeline, matching the original
# exactly:
#
#   STAGE A (ported from process_abi_single_hitter): Bayesian PA-weighted
#   baseline -> platoon blend -> ahead-in-count boost -> home/away scalar ->
#   wOBA-weighted 5/10/15-game form buckets -> fatigue decay -> weather/park
#   cap -> PA-scaled binomial core -> K/WHIP modifier -> bullpen cap ->
#   platoon DNA -> Poisson decay -> hard clamp [0.02, 0.80].
#
#   STAGE B (ported from the tab_hit post-processing block, "BATAS 1/2/3" +
#   the V8.9 quality modifier): velocity control penalty -> BABIP regression
#   penalty -> lineup order bonus -> quality modifier + pitcher velocity mod
#   -> final clamp [0.0, 1.0].
#
# Two things were deliberately NOT ported (see chat) because they were dead
# code in the original itself, not because they were dropped by accident:
#   - OFC modifier: opposite_pct was a hardcoded 0.26 constant compared
#     against a >=0.28 threshold — always false, so it never fired.
#   - pitcher_avg_fb_velo (BATAS 1's input) was never actually populated
#     anywhere in the original pipeline (always read as 0.0), so the
#     velocity-control penalty never fired either. Here it's wired to a
#     real fetch (pitchArsenal avg fastball velocity) so it actually works.
#
# FIXES (2026-08-29 consolidation pass):
#   1. HR/Run/RBI double-compounding — process_hr_prob/process_run_prob/
#      process_rbi_prob already integrate over the full `pa_proj` plate-
#      appearance projection internally (compute_event_probability splits
#      pa_proj into a starter portion + bullpen portion and returns
#      P(>=1 event) across the whole game — same contract as Stage A's hit
#      binomial core). _build_game was re-applying
#      `1 - (1 - single) ** pa_proj` on top of that already-integrated
#      probability, double-compounding it and materially inflating every
#      HR/Run/RBI quote served by /api/slate. Fixed to use the returned
#      probability directly, exactly like hit_prob.
#   2. Predictions were never persisted — PredictionRepository.upsert_
#      prediction() was fully built (schema, indexes, WAL) but never called
#      anywhere, so the `predictions` table stayed permanently empty. That
#      made /api/track-record always return an empty log/calibration, and
#      sync_results() a permanent no-op (repository.unresolved() always
#      empty). Fixed by upserting each of the 4 props per player at the end
#      of the per-player loop in _build_game.
#   3. _compute_hit_probability_stage_a's early-return branch
#      (projected_pa < 2.0) returned a 3-tuple while the function's normal
#      path — and every caller — unpacks 4 values. projected_pa_from_order()
#      currently never returns <2.0 so this was dead code, not a live crash,
#      but it's a latent ValueError landmine. Fixed to return 4 values.

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

from config import settings
from math_engine import (
    K_TEAM_RUNS,
    LEAGUE_AVG_BA,
    LEAGUE_AVG_BABIP,
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_HR_RATE,
    LEAGUE_AVG_ISO,
    LEAGUE_AVG_K_RATE,
    LEAGUE_AVG_OBP,
    LEAGUE_AVG_OPS_FALLBACK,
    LEAGUE_AVG_RBI_RATE,
    LEAGUE_AVG_RUN_RATE,
    LEAGUE_AVG_SLG,
    LINEUP_STRENGTH_TRUST_CONFIRMED,
    LINEUP_STRENGTH_TRUST_PROJECTED,
    PITCHER_RECENT_FORM_STARTS,
    K_HITS,
    K_HR,
    K_RBI,
    K_RUNS,
    MAX_PROB,
    MIN_PROB,
    STABILIZATION_PA_H2H,
    ahead_in_count_boost,
    analyze_pitcher_split,
    apply_split_effect,
    babip_regression_penalty_points,
    build_count_market,
    build_pitcher_ladder,
    bullpen_fatigue_multiplier,
    calculate_team_xruns_v2,
    calculate_woba_from_stats,
    calibrate_count_market,
    clamp,
    confidence_from_edge,
    credibility_weighted_average,
    fastball_whiff_proxy,
    hit_quality_modifier,
    home_away_scoring_factor,
    lineup_order_bonus_mult,
    lineup_strength_offense_index,
    parse_innings_pitched,
    pitcher_velocity_mod,
    platoon_adjusted_ops,
    poisson_monte_carlo_win_prob,
    prob_to_american,
    process_hr_prob,
    process_rbi_prob,
    process_run_prob,
    projected_pa_from_order,
    remove_vig,
    safe_float,
    shrink_rate,
    starter_quality_index,
    starter_recent_form,
    velocity_control_penalty,
    weather_scoring_multiplier,
    weather_wind_context,
)
from mlb_client import MLBClient
from odds_client import OddsClient
from repository import PredictionRepository
from weather_client import WeatherClient

# LOGGING (2026-08-29): every previously-silent `except: continue` / `except:
# pass` in this file now logs before swallowing — see each site below.
# Deliberately just logging.getLogger(__name__), no basicConfig() here: this
# is a library module, not the process entrypoint, so it shouldn't clobber
# whatever handler/format app.py (or uvicorn) configures at startup. Python's
# default "handler of last resort" still prints WARNING+ to stderr even with
# zero configuration, so these are visible out of the box either way.
logger = logging.getLogger("trckr21.prediction_service")

# NEW (2026-09-09, diagnostic pass) — see build_slate's cache-gate note.
# How long an unconfirmed-lineup slate is trusted before the next request
# pays for a full rebuild again. Short enough that newly-posted lineups
# show up within a reasonable wait, long enough that rapid page
# reloads/refreshes don't each trigger their own cold rebuild.
LINEUP_RECHECK_INTERVAL_SECONDS = 300

# NEW (2026-09-11, diagnostic pass) — see the inflight-join fix in
# _build_slate_blocking / _blocking_list_build. Bounds how long a request
# will wait on a build it's piggybacking on before giving up on that one
# and starting its own.
INFLIGHT_JOIN_TIMEOUT_SECONDS = 180

# NEW (2026-09-11, diagnostic pass) — see _pitcher_quality_snapshot's note.
PITCHER_QUALITY_CACHE_TTL_SECONDS = 1800


# TASK 1 UI (2026-08-31) — carries the weather signal both consumers need:
# the raw scoring multiplier (Poisson Monte Carlo / calculate_team_xruns_v2
# inputs, unchanged from before) AND the display fields for the Moneylines
# weather badge (GameResponse.weatherTempF/weatherSummary/weatherTone/
# weatherDetail). One fetch + one cache entry serves both, instead of a
# second venue/forecast round trip just for display.
@dataclass(frozen=True)
class WeatherContext:
    mult: float = 1.0
    temp_f: int | None = None
    wind_label: str | None = None
    wind_tone: str = "neutral"
    wind_detail: str | None = None

    @property
    def summary(self) -> str:
        if self.temp_f is None or not self.wind_label:
            return "Weather unavailable"
        return f"{self.temp_f}\u00b0F \u00b7 {self.wind_label}"


# ---------------------------------------------------------------------------
# CONSOLIDATION (2026-08-29) — Pitcher Props / Team Matchups / Matchup
# Analyzer, ported from the retired generate_projections.py. Default lines
# unchanged from the script's DEFAULT_LINES.
# ---------------------------------------------------------------------------
DEFAULT_PROP_LINES = {
    "strikeouts": 4.5,
    "earned_runs": 1.5,
    "walks": 2.5,
    "f5_runs": 1.5,
    "total_runs": 4.5,
}

# Full-team-name -> short code, used only for the pitcher-props/team-
# matchups "MM/DD vs OPP" game-log axis labels (ported verbatim from
# generate_projections.py's format_date_label). Separate from
# TEAM_ABBREVIATIONS above, which is keyed by MLB team id, not name.
_TEAM_ABBR_BY_NAME = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def _format_date_label(date_str: str, is_home: bool, opp_name: str | None) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = dt.strftime("%m/%d")
        location = "vs" if is_home else "@"
        if not isinstance(opp_name, str) or not opp_name or opp_name == "OPP":
            return f"{formatted_date} {location} OPP"
        abbr = _TEAM_ABBR_BY_NAME.get(opp_name) or opp_name.split()[0][:3].upper()
        return f"{formatted_date} {location} {abbr}"
    except (ValueError, TypeError) as exc:
        logger.debug("Could not format date label for date_str=%r opp_name=%r: %s", date_str, opp_name, exc)
        return str(date_str) or "—"


def _matchup_label(opponent_name: str, is_home: bool) -> str:
    return f"{'vs' if is_home else '@'} {opponent_name}"


def _edge_tier(blended_ops: float) -> str:
    if blended_ops >= 0.820:
        return "elite"
    if blended_ops >= 0.740:
        return "strong"
    if blended_ops >= 0.660:
        return "average"
    return "soft"


def _recent_form_tier(ops: float, pa: int) -> str:
    if pa < 8:
        return "limited"
    if ops >= 0.900:
        return "hot"
    if ops <= 0.600:
        return "cold"
    return "neutral"


def _pitcher_arsenal_whiff_pct(pitch_mix: list[dict]) -> float | None:
    weighted, total_usage = 0.0, 0.0
    for p in pitch_mix:
        if p.get("whiffPct") is not None:
            weighted += p["whiffPct"] * p["usagePct"]
            total_usage += p["usagePct"]
    return round(weighted / total_usage, 1) if total_usage > 0 else None


def _matchup_multiplier(
    team_obp: float,
    team_slg: float,
    opp_starter_era: float,
    *,
    opp_proj_er: float | None = None,
    opp_k_rate: float | None = None,
    opp_recent_blended_era: float | None = None,
    opp_recent_blended_k_rate: float | None = None,
    opp_arsenal_whiff_pct: float | None = None,
    today_lineup_ops: float | None = None,
    lineup_confidence_weight: float = 0.0,
) -> float:
    # Extracted verbatim from _build_game's local `_matchup_mult` closure —
    # pure dedup, not a formula change — so /api/team-matchups' shrinkage
    # anchor is built from the EXACT SAME offense-vs-pitching multiplier the
    # Moneylines tab already feeds into calculate_team_xruns_v2, instead of
    # a second, drifting reimplementation.
    #
    # TASK (2026-08-31) — extended per chat items #1-#8. Every new arg is
    # optional and keyword-only: a caller that doesn't pass them (there
    # currently isn't one, but this keeps the function safe to call from
    # anywhere else in the future) gets byte-for-byte the old formula.
    #   - offense_index now optionally blends today's actual/projected
    #     lineup strength over the season-wide OBP/SLG average (items
    #     #2/#4/#6/#7/#8) — see lineup_strength_offense_index().
    #   - pitching_index now optionally uses pitcher_profile()'s proj_k/
    #     proj_er/k_rate + recent-form + arsenal whiff% instead of season
    #     ERA alone (items #1/#3/#5) — see starter_quality_index(). Falls
    #     back to the original plain ERA ratio when opp_proj_er/opp_k_rate
    #     aren't supplied.
    offense_index = lineup_strength_offense_index(
        today_lineup_ops=today_lineup_ops,
        lineup_confidence_weight=lineup_confidence_weight,
        season_team_obp=team_obp,
        season_team_slg=team_slg,
    )
    if opp_proj_er is not None and opp_k_rate is not None:
        pitching_index = starter_quality_index(
            era=opp_starter_era,
            proj_er=opp_proj_er,
            k_rate=opp_k_rate,
            recent_blended_era=opp_recent_blended_era,
            recent_blended_k_rate=opp_recent_blended_k_rate,
            arsenal_whiff_pct=opp_arsenal_whiff_pct,
        )
    else:
        # FIX (2026-09-03) — same ERA-direction inversion as everywhere else
        # in this pass: this ratio feeds the Moneylines win-probability
        # model, where >1.0 is documented (see starter_quality_index) as
        # "offense-friendly." LEAGUE_AVG_ERA/opp_starter_era put a low-ERA
        # ace ABOVE 1.0. Flipped to opp_starter_era/LEAGUE_AVG_ERA.
        pitching_index = clamp(opp_starter_era / LEAGUE_AVG_ERA if LEAGUE_AVG_ERA > 0 else 1.0, 0.75, 1.30)
    return offense_index * pitching_index

STADIUM_INDICES = {
    "Coors Field": {"elevation_factor": 1.12, "base_park_factor": 1.15},
    "Yankee Stadium": {"elevation_factor": 1.01, "base_park_factor": 1.04},
    "Fenway Park": {"elevation_factor": 1.00, "base_park_factor": 1.05},
    "Wrigley Field": {"elevation_factor": 1.01, "base_park_factor": 1.02},
    "Dodger Stadium": {"elevation_factor": 1.02, "base_park_factor": 0.96},
    "Oracle Park": {"elevation_factor": 1.00, "base_park_factor": 0.93},
    "T-Mobile Park": {"elevation_factor": 1.00, "base_park_factor": 0.92},
    "Citi Field": {"elevation_factor": 1.00, "base_park_factor": 0.95},
    "Chase Field": {"elevation_factor": 1.04, "base_park_factor": 1.01},
    "Petco Park": {"elevation_factor": 1.00, "base_park_factor": 0.91},
}

TEAM_ABBREVIATIONS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def _ladder_lines(expected: float) -> list[float]:
    base = max(0.5, float(expected))
    center = round(base * 2) / 2.0
    if center == round(center):
        center += 0.5
    return [round(center - 1.0, 1), round(center, 1), round(center + 1.0, 1)]


def _quote_from_market(model_prob: float, sample_size: int, quote: dict | None) -> dict:
    fair_odds = prob_to_american(model_prob)
    if not quote:
        return {
            "prob": model_prob, "odds": fair_odds, "bookOdds": None, "noOdds": None,
            "conf": confidence_from_edge(None, model_prob, sample_size),
            "edge": None, "marketAvailable": False, "modelOnly": True,
        }
    over_odds = quote.get("over")
    under_odds = quote.get("under")
    market_prob_over, _ = remove_vig(over_odds, under_odds)
    edge = None if market_prob_over is None else round(model_prob - market_prob_over, 4)
    return {
        "prob": model_prob, "odds": fair_odds, "bookOdds": over_odds, "noOdds": under_odds,
        "conf": confidence_from_edge(edge, model_prob, sample_size),
        "edge": edge, "marketAvailable": True, "modelOnly": False,
    }


class PredictionService:
    def __init__(
        self,
        mlb: MLBClient,
        odds: OddsClient,
        repository: PredictionRepository,
        weather: WeatherClient | None = None,
    ) -> None:
        self.mlb = mlb
        self.odds = odds
        self.repository = repository
        # TASK 1 (2026-08-30) — optional constructor param (defaults to a
        # plain WeatherClient()) so existing PredictionService(mlb, odds,
        # repository) call sites keep working unchanged.
        self.weather = weather or WeatherClient()
        self._slate_cache: dict[str, dict] = {}
        # NEW (2026-09-09, diagnostic pass): de-dupes concurrent cold
        # builds for the same date — see build_slate()'s note. Without
        # this, a slow first request that a client gives up on (timeout)
        # and a retry a few minutes later — while the FIRST build might
        # still be running server-side — would kick off a second,
        # fully-redundant build racing the first, each competing for the
        # same shared per-build caches (_team_hitting_cache etc., cleared
        # at the start of every build) and making both slower still.
        self._slate_inflight: dict[str, Future] = {}
        self._player_index: dict[int, dict] = {}
        self._pitcher_index: dict[int, dict] = {}
        # CONSOLIDATION — same per-date, in-memory cache pattern as
        # _slate_cache, one dict per new endpoint.
        # FIX (2026-09-10, diagnostic pass): these three used to cache
        # PERMANENTLY until an explicit refresh=True — no staleness check
        # at all, unlike build_slate(). That's why a lineup swap (e.g. a
        # late scratch/call-up change) never showed up here even minutes
        # later: Matchup Analyzer kept serving the exact snapshot it built
        # the first time it was ever requested that day. Each entry is now
        # {"data": [...], "cachedAt": datetime} and goes through the same
        # stale-while-revalidate + in-flight de-dup helpers build_slate()
        # uses (_cached_list_build / _schedule_background_list_rebuild).
        self._pitcher_props_cache: dict[str, dict] = {}
        self._team_matchups_cache: dict[str, dict] = {}
        self._matchup_analyzer_cache: dict[str, dict] = {}
        self._pitcher_props_inflight: dict[str, Future] = {}
        self._team_matchups_inflight: dict[str, Future] = {}
        self._matchup_analyzer_inflight: dict[str, Future] = {}
        self._lock = Lock()

        # PERF (2026-08-29) — team/pitcher-season-level fetch cache, scoped
        # to a single build_slate() call (cleared at the top of build_slate,
        # see below). Before this, team_stats/team_home_away_runs_hits/
        # team_roster/pitcher_profile were re-fetched fresh inside every
        # _build_game() call — harmless for a normal single-game-per-team
        # day, but a real duplicate-fetch bug on doubleheaders (same team_id
        # showing up in two different game threads the same slate build).
        # Separate lock from self._lock so team-cache reads/writes never
        # contend with the slate/player/pitcher-index lock above.
        self._team_hitting_cache: dict[int, dict] = {}
        self._team_pitching_cache: dict[int, dict] = {}
        self._team_home_away_cache: dict[tuple[int, bool], tuple[float, float]] = {}
        self._team_roster_cache: dict[int, list[dict]] = {}
        self._pitcher_profile_cache: dict[int, dict] = {}
        # TASK 1 & 2 (2026-08-30) — same per-slate-build cache pattern as
        # the team caches above. Weather is keyed by (venue_id, game_time)
        # since it's a per-GAME signal, not per-team; bullpen fatigue is
        # keyed by team_id alone, same shape as the other team-level caches.
        self._weather_mult_cache: dict[tuple[int, str], WeatherContext] = {}
        self._bullpen_fatigue_cache: dict[int, float] = {}
        # TASK (2026-08-31) — same per-slate-build cache pattern, for the
        # new starter-quality-index and lineup-strength signals. Keyed by
        # pitcher_id / (game_pk, team_id, opp_pitcher_id) same shape as the
        # caches above.
        self._starter_recent_form_cache: dict[int, dict] = {}
        self._starter_arsenal_whiff_cache: dict[int, float | None] = {}
        self._lineup_signal_cache: dict[tuple[int, int, int], tuple] = {}
        # NEW (2026-09-11, diagnostic pass): _pitcher_quality_snapshot's
        # season-stat fetch and put-away aggregation (which walks several
        # recent starts' worth of pitch-by-pitch data) were re-run from
        # scratch on EVERY Pitcher Props rebuild, including every 5-minute
        # stale-while-revalidate cycle — none of which meaningfully changes
        # within a day. That's real added load on top of everything else
        # this app already fetches, and a likely contributor to slowdowns
        # after hours of continuous use. Cached per pitcher_id for
        # PITCHER_QUALITY_CACHE_TTL_SECONDS instead of every rebuild.
        self._pitcher_quality_cache: dict[int, dict] = {}
        self._team_cache_lock = Lock()

    # ------------------------------------------------------------------
    # PERF — team/pitcher-season-level cache helpers (see __init__ note).
    # Each memoizes exactly the MLBClient call it wraps; nothing here
    # changes what data is fetched or how it's used downstream.
    # ------------------------------------------------------------------

    def _cached_team_stats(self, team_id: int, group: str) -> dict:
        cache = self._team_hitting_cache if group == "hitting" else self._team_pitching_cache
        with self._team_cache_lock:
            cached = cache.get(team_id)
        if cached is not None:
            return cached
        data = self.mlb.team_stats(team_id, group)
        with self._team_cache_lock:
            cache[team_id] = data
        return data

    def _cached_team_home_away(self, team_id: int, is_home: bool) -> tuple[float, float]:
        key = (team_id, is_home)
        with self._team_cache_lock:
            cached = self._team_home_away_cache.get(key)
        if cached is not None:
            return cached
        data = self.mlb.team_home_away_runs_hits(team_id, is_home=is_home)
        with self._team_cache_lock:
            self._team_home_away_cache[key] = data
        return data

    def _cached_team_roster(self, team_id: int) -> list[dict]:
        with self._team_cache_lock:
            cached = self._team_roster_cache.get(team_id)
        if cached is not None:
            return cached
        data = self.mlb.team_roster(team_id)
        with self._team_cache_lock:
            self._team_roster_cache[team_id] = data
        return data

    def _cached_pitcher_profile(self, pitcher_id: int | None) -> dict:
        if not pitcher_id:
            return self.mlb.default_pitcher()
        with self._team_cache_lock:
            cached = self._pitcher_profile_cache.get(pitcher_id)
        if cached is not None:
            return cached
        data = self.mlb.pitcher_profile(pitcher_id)
        with self._team_cache_lock:
            self._pitcher_profile_cache[pitcher_id] = data
        return data

    # ------------------------------------------------------------------
    # TASK (2026-08-31) — Moneylines team-strength signal upgrade, chat
    # items #1/#3/#5. Starter-quality index inputs beyond pitcher_
    # profile()'s own era/proj_er/k_rate: last-2-3-starts recent form
    # (blended toward season via shrink_rate(), same as everywhere else)
    # and Matchup Analyzer's pitcherArsenalWhiffPct. Both are best-effort —
    # on any fetch failure they fall back to a neutral value so a flaky
    # gameLog/pitchArsenal call never takes a whole game off the slate,
    # same convention as _cached_weather_context/_cached_bullpen_fatigue_
    # mult above.
    # ------------------------------------------------------------------

    def _cached_starter_recent_form(
        self, pitcher_id: int | None, season_era: float, season_k_rate: float,
    ) -> dict:
        neutral = {
            "startsSampled": 0, "recentEra": None, "recentKRate": None,
            "blendedEra": season_era, "blendedKRate": season_k_rate,
        }
        if not pitcher_id:
            return neutral
        with self._team_cache_lock:
            cached = self._starter_recent_form_cache.get(pitcher_id)
        if cached is not None:
            return cached

        data = neutral
        try:
            games = self.mlb.pitcher_recent_pitching_log(pitcher_id, limit=PITCHER_RECENT_FORM_STARTS)
            recent_er, recent_ip, recent_k, recent_bf = [], [], [], []
            for g in games:
                s = g.get("stat", {})
                recent_er.append(safe_float(s.get("earnedRuns"), 0.0))
                recent_ip.append(parse_innings_pitched(s.get("inningsPitched")))
                recent_k.append(safe_float(s.get("strikeOuts"), 0.0))
                recent_bf.append(safe_float(s.get("battersFaced"), 0.0))
            data = starter_recent_form(recent_er, recent_ip, recent_k, recent_bf, season_era, season_k_rate)
        except Exception:
            logger.warning(
                "Could not compute recent-form for pitcher_id=%s — starter-quality index will use season "
                "ERA/K-rate only for this signal.",
                pitcher_id, exc_info=True,
            )
            data = neutral

        with self._team_cache_lock:
            self._starter_recent_form_cache[pitcher_id] = data
        return data

    def _cached_starter_arsenal_whiff(self, pitcher_id: int | None) -> float | None:
        if not pitcher_id:
            return None
        _MISS = object()
        with self._team_cache_lock:
            cached = self._starter_arsenal_whiff_cache.get(pitcher_id, _MISS)
        if cached is not _MISS:
            return cached

        whiff = None
        try:
            pitch_mix = self.mlb.pitch_arsenal(pitcher_id)
            recent_pks = self.mlb.pitcher_recent_game_pks(pitcher_id)
            lethality = self.mlb.pitcher_pitch_lethality(pitcher_id, recent_pks) if recent_pks else {}
            for p in pitch_mix:
                stats = lethality.get(p.get("code", "UN"))
                p["whiffPct"] = stats["whiffPct"] if stats else None
            whiff = _pitcher_arsenal_whiff_pct(pitch_mix)
        except Exception:
            logger.warning(
                "Could not compute arsenal whiff%% for pitcher_id=%s — starter-quality index will skip this "
                "signal for this pitcher.",
                pitcher_id, exc_info=True,
            )
            whiff = None

        with self._team_cache_lock:
            self._starter_arsenal_whiff_cache[pitcher_id] = whiff
        return whiff

    # ------------------------------------------------------------------
    # TASK (2026-08-31) — Moneylines team-strength signal upgrade, chat
    # items #2/#4/#6/#7/#8. Reuses the EXACT SAME per-batter pipeline the
    # Matchup Analyzer tab's lineupEdgeOps/lineupEdgeTier already run
    # (_batter_matchup_job below — H2H-blended OPS, credibility, platoon
    # splits) instead of a second, separate per-player projection
    # aggregation off the Hits/HR/RBI/Runs tab's hr_rate/run_rate/rbi_rate
    # pipeline (see chat — item #4 supersedes the original per-player-
    # projection framing of item #2 with "reuse lineupEdgeOps instead").
    # The only NEW math here is the final aggregation step: platoon-adjust
    # each batter's blendedOps toward today's actual starting pitcher's
    # throwing hand (item #7), then combine with a credibility weight
    # instead of a plain mean (item #6) — lineupEdgeOps itself (the plain,
    # unweighted mean shown on the Matchup Analyzer tab) is left untouched.
    # ------------------------------------------------------------------

    def _today_lineup_signal(
        self, game_pk: int | None, team_id: int, opp_pitcher_id: int | None, opp_pitcher_hand: str, target_date: str,
    ) -> tuple[float | None, float, bool]:
        """Returns (today_lineup_ops, lineup_confidence_weight, confirmed).

        today_lineup_ops is the credibility-weighted, platoon-adjusted OPS
        across team_id's probable/confirmed lineup for today's game, or
        None if a lineup couldn't be resolved (no boxscore yet, fetch
        failure, etc.) — callers pass that straight through to
        lineup_strength_offense_index(), which falls back to the
        season-wide OBP/SLG anchor exactly like _matchup_multiplier always
        did before this signal existed.

        lineup_confidence_weight is LINEUP_STRENGTH_TRUST_CONFIRMED or
        LINEUP_STRENGTH_TRUST_PROJECTED depending on probable_lineup()'s
        `confirmed` flag — the dynamic blend from chat item #8, expressed
        as one shrink_rate() input rather than two separate formulas.
        """
        if not game_pk or not opp_pitcher_id:
            return None, 0.0, False

        cache_key = (game_pk, team_id, opp_pitcher_id)
        with self._team_cache_lock:
            cached = self._lineup_signal_cache.get(cache_key)
        if cached is not None:
            return cached

        result = (None, 0.0, False)
        try:
            lineup, confirmed = self.mlb.probable_lineup(game_pk, team_id, target_date)
            if not lineup:
                result = (None, 0.0, confirmed)
            else:
                with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
                    batters = list(pool.map(lambda b: self._batter_matchup_job(b, opp_pitcher_id), lineup))

                platoon_key = "vsLHP" if opp_pitcher_hand == "L" else "vsRHP"
                adjusted_ops, credibilities = [], []
                for b in batters:
                    platoon = (b.get("platoonSplits") or {}).get(platoon_key, {})
                    platoon_ops = safe_float(platoon.get("ops"), 0.0)
                    platoon_pa = int(platoon.get("pa", 0) or 0)
                    adjusted_ops.append(platoon_adjusted_ops(b["blendedOps"], platoon_ops, platoon_pa))
                    credibilities.append(b.get("credibility", 0.0))

                lineup_ops = credibility_weighted_average(adjusted_ops, credibilities)
                if lineup_ops is None:
                    result = (None, 0.0, confirmed)
                else:
                    trust_weight = LINEUP_STRENGTH_TRUST_CONFIRMED if confirmed else LINEUP_STRENGTH_TRUST_PROJECTED
                    result = (lineup_ops, trust_weight, confirmed)
        except Exception:
            logger.warning(
                "Failed to build today's-lineup signal for team_id=%s vs pitcher_id=%s — falling back to "
                "season-wide OBP/SLG for this side's matchup multiplier.",
                team_id, opp_pitcher_id, exc_info=True,
            )
            result = (None, 0.0, False)

        with self._team_cache_lock:
            self._lineup_signal_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # TASK 1 (2026-08-30, extended 2026-08-31) — weather context. venue()
    # gives us lat/lon + park azimuth (already fetched via MLBClient, no new
    # dependency there); WeatherClient.forecast_at_game_time() is the only
    # new I/O. Falls back to a neutral WeatherContext (mult=1.0, "Weather
    # unavailable") on any missing venue/coordinate/forecast data — this is
    # a nice-to-have signal, not something that should ever take a game off
    # the slate.
    #
    # Returns the full WeatherContext (not just the multiplier) so the same
    # cached fetch feeds both the Poisson Monte Carlo model AND the
    # Moneylines weather badge (GameResponse.weatherTempF/weatherSummary/
    # weatherTone/weatherDetail) — see WeatherContext above. Call sites that
    # only need the multiplier read `.mult` off the result.
    # ------------------------------------------------------------------

    def _cached_weather_context(self, venue_id: int | None, game_time_utc: str | None) -> WeatherContext:
        if not venue_id or not game_time_utc:
            return WeatherContext()
        key = (venue_id, game_time_utc)
        with self._team_cache_lock:
            cached = self._weather_mult_cache.get(key)
        if cached is not None:
            return cached

        ctx = WeatherContext()
        try:
            venue_data = self.mlb.venue(venue_id)
            location = venue_data.get("location", {}) if venue_data else {}
            coords = location.get("defaultCoordinates", {})
            lat, lon = coords.get("latitude"), coords.get("longitude")
            if lat is not None and lon is not None:
                forecast = self.weather.forecast_at_game_time(lat, lon, game_time_utc)
                if forecast:
                    temp_f = forecast["temperature_f"]
                    wind_speed = forecast["wind_speed_mph"]
                    wind_dir = forecast["wind_direction_deg"]
                    park_azimuth = location.get("azimuthAngle")
                    mult = weather_scoring_multiplier(
                        temperature_f=temp_f,
                        wind_speed_mph=wind_speed,
                        wind_direction_deg=wind_dir,
                        park_azimuth_deg=park_azimuth,
                    )
                    wind_label, wind_tone, wind_detail = weather_wind_context(
                        wind_speed_mph=wind_speed,
                        wind_direction_deg=wind_dir,
                        park_azimuth_deg=park_azimuth,
                    )
                    ctx = WeatherContext(
                        mult=mult,
                        temp_f=round(temp_f),
                        wind_label=wind_label,
                        wind_tone=wind_tone,
                        wind_detail=wind_detail,
                    )
        except Exception:
            logger.warning(
                "Could not compute weather context for venue_id=%s game_time=%s — using neutral fallback.",
                venue_id, game_time_utc, exc_info=True,
            )
            ctx = WeatherContext()

        with self._team_cache_lock:
            self._weather_mult_cache[key] = ctx
        return ctx

    # ------------------------------------------------------------------
    # TASK 2 (2026-08-30) — bullpen_fatigue_mult. Rolling 2-3 day relief-
    # innings workload from MLB Stats API only (see mlb_client.
    # team_bullpen_relief_innings). Falls back to a neutral 1.0 on any
    # fetch failure, same convention as _cached_weather_context above.
    # ------------------------------------------------------------------

    def _cached_bullpen_fatigue_mult(self, team_id: int | None, target_date: str) -> float:
        if not team_id:
            return 1.0
        with self._team_cache_lock:
            cached = self._bullpen_fatigue_cache.get(team_id)
        if cached is not None:
            return cached

        mult = 1.0
        try:
            relief_ip = self.mlb.team_bullpen_relief_innings(team_id, target_date, days=3)
            mult = bullpen_fatigue_multiplier(relief_ip)
        except Exception:
            logger.warning(
                "Could not compute bullpen_fatigue_mult for team_id=%s on %s — using neutral 1.0.",
                team_id, target_date, exc_info=True,
            )
            mult = 1.0

        with self._team_cache_lock:
            self._bullpen_fatigue_cache[team_id] = mult
        return mult

    # ------------------------------------------------------------------
    # SLATE BUILD
    # ------------------------------------------------------------------

    def build_slate(self, target_date: str, force: bool = False) -> dict:
        # FIX (2026-09-03, diagnostic pass): "missing top players" root
        # cause. app.py's startup hook calls build_slate() the moment the
        # server boots — typically hours before MLB posts today's official
        # lineups. Until a lineup posts, _resolve_hitters() has no real
        # batting order to work with and falls back to an arbitrary 9-name
        # slice of the active roster (roster API order, NOT importance —
        # see _resolve_hitters). That slate then got cached under
        # target_date PERMANENTLY (see the old unconditional cache-hit
        # below) — so once the server warmed up on a placeholder roster
        # slice, it stayed wrong ALL DAY regardless of how many times the
        # page was reloaded, because nothing ever re-checked whether real
        # lineups had shown up since.
        #
        # FIX (2026-09-09, diagnostic pass): "rebuild until every game's
        # lineup is confirmed" meant EVERY single request did a full cold
        # rebuild for as long as even ONE late game hadn't posted its
        # lineup yet — in practice, most of the day. That's why /api/slate
        # kept timing out no matter how long you waited between requests:
        # there was never a cache HIT to benefit from, only ever a cache
        # WRITE that got thrown away before it could help anyone.
        #
        # FIX (2026-09-09, diagnostic pass, part 3 — stale-while-revalidate):
        # even bounding the rebuild to once per LINEUP_RECHECK_INTERVAL_
        # SECONDS still means the unlucky request that lands right when the
        # window expires pays the full rebuild cost and can time out. A
        # request should never have to wait on a rebuild it didn't
        # explicitly ask for. Now: if a cached slate exists at all (even a
        # stale/unconfirmed one), it's returned IMMEDIATELY, and a
        # background rebuild is kicked off (de-duped — see
        # _schedule_background_slate_rebuild) to refresh it for next time.
        # The only path that still blocks is when there's truly nothing
        # cached yet for this date (first request since boot) or an
        # explicit refresh=True — those have nothing safe to serve instead,
        # so they wait on a real build, de-duped the same way a background
        # rebuild is (a second request that lands mid-build joins the one
        # already running instead of starting a redundant one).
        with self._lock:
            cached = self._slate_cache.get(target_date)

        if cached is not None and not force:
            if cached["meta"].get("lineupsConfirmed"):
                return cached
            cached_age = (datetime.now(timezone.utc) - cached["_cachedAt"]).total_seconds()
            if cached_age < LINEUP_RECHECK_INTERVAL_SECONDS:
                return cached
            self._schedule_background_slate_rebuild(target_date)
            return cached

        return self._build_slate_blocking(target_date)

    def _build_slate_blocking(self, target_date: str) -> dict:
        with self._lock:
            inflight = self._slate_inflight.get(target_date)
            if inflight is not None:
                is_builder = False
            else:
                inflight = Future()
                self._slate_inflight[target_date] = inflight
                is_builder = True

        if not is_builder:
            logger.info("build_slate(%s): joining an already-running build instead of starting a redundant one.", target_date)
            # FIX (2026-09-11, diagnostic pass): this used to be an
            # unbounded inflight.result() — if the build it's joining ever
            # genuinely hangs (a stuck network call somewhere, or just a
            # pathologically slow day against a throttled upstream API),
            # every request piles up waiting on that SAME stuck Future
            # forever, with no way out short of restarting the process —
            # matching "nag timeout tapos di na maibalik." Bounded to
            # INFLIGHT_JOIN_TIMEOUT_SECONDS; a request that waits this
            # long gives up on the stuck build and starts its own fresh
            # one instead of hanging alongside it indefinitely.
            try:
                return inflight.result(timeout=INFLIGHT_JOIN_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                logger.warning(
                    "build_slate(%s): the build we were joining has been running for over %ss — starting a fresh one instead of waiting indefinitely.",
                    target_date, INFLIGHT_JOIN_TIMEOUT_SECONDS,
                )
                with self._lock:
                    # Only clear it if it's still the SAME stuck entry —
                    # don't clobber a newer build that may have already
                    # taken its place.
                    if self._slate_inflight.get(target_date) is inflight:
                        self._slate_inflight.pop(target_date, None)
                return self._build_slate_blocking(target_date)

        try:
            slate = self._build_slate_uncached(target_date)
        except Exception as exc:
            inflight.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._slate_inflight.pop(target_date, None)
        inflight.set_result(slate)
        return slate

    def _schedule_background_slate_rebuild(self, target_date: str) -> None:
        with self._lock:
            if target_date in self._slate_inflight:
                return  # a rebuild (blocking or background) is already running for this date
            inflight = self._slate_inflight[target_date] = Future()

        def _run() -> None:
            try:
                slate = self._build_slate_uncached(target_date)
                inflight.set_result(slate)
            except Exception:
                logger.exception("Background slate rebuild failed for %s — will retry on the next stale hit.", target_date)
                inflight.set_exception(RuntimeError("background slate rebuild failed"))
            finally:
                with self._lock:
                    self._slate_inflight.pop(target_date, None)

        threading.Thread(target=_run, daemon=True, name=f"slate-bg-rebuild-{target_date}").start()

    # ------------------------------------------------------------------
    # Generic stale-while-revalidate + in-flight de-dup, for the three
    # simpler "list" endpoints (pitcher-props, team-matchups, matchup-
    # analyzer) — same shape as the slate-specific version above, minus
    # the per-item lineupsConfirmed gate (a flat TTL is a fine, honest
    # trade-off here: it bounds the worst-case staleness the same way
    # without needing each of these three to separately track a
    # slate-style confirmation flag).
    # ------------------------------------------------------------------

    def _cached_list_build(self, cache: dict, inflight: dict, target_date: str, force: bool, build_fn, cache_name: str) -> list[dict]:
        with self._lock:
            entry = cache.get(target_date)
        if entry is not None and not force:
            age = (datetime.now(timezone.utc) - entry["cachedAt"]).total_seconds()
            if age < LINEUP_RECHECK_INTERVAL_SECONDS:
                return entry["data"]
            self._schedule_background_list_rebuild(cache, inflight, target_date, build_fn, cache_name)
            return entry["data"]
        return self._blocking_list_build(cache, inflight, target_date, build_fn, cache_name)

    def _blocking_list_build(self, cache: dict, inflight: dict, target_date: str, build_fn, cache_name: str) -> list[dict]:
        with self._lock:
            fut = inflight.get(target_date)
            if fut is not None:
                is_builder = False
            else:
                fut = Future()
                inflight[target_date] = fut
                is_builder = True

        if not is_builder:
            logger.info("%s(%s): joining an already-running build instead of starting a redundant one.", cache_name, target_date)
            # FIX (2026-09-11, diagnostic pass) — same unbounded-wait fix as
            # _build_slate_blocking above, applied here too.
            try:
                return fut.result(timeout=INFLIGHT_JOIN_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                logger.warning(
                    "%s(%s): the build we were joining has been running for over %ss — starting a fresh one instead of waiting indefinitely.",
                    cache_name, target_date, INFLIGHT_JOIN_TIMEOUT_SECONDS,
                )
                with self._lock:
                    if inflight.get(target_date) is fut:
                        inflight.pop(target_date, None)
                return self._blocking_list_build(cache, inflight, target_date, build_fn, cache_name)

        try:
            data = build_fn(target_date)
            with self._lock:
                cache[target_date] = {"data": data, "cachedAt": datetime.now(timezone.utc)}
        except Exception as exc:
            fut.set_exception(exc)
            with self._lock:
                inflight.pop(target_date, None)
            raise
        with self._lock:
            inflight.pop(target_date, None)
        fut.set_result(data)
        return data

    def _schedule_background_list_rebuild(self, cache: dict, inflight: dict, target_date: str, build_fn, cache_name: str) -> None:
        with self._lock:
            if target_date in inflight:
                return
            fut = inflight[target_date] = Future()

        def _run() -> None:
            try:
                data = build_fn(target_date)
                with self._lock:
                    cache[target_date] = {"data": data, "cachedAt": datetime.now(timezone.utc)}
                fut.set_result(data)
            except Exception:
                logger.exception("Background %s rebuild failed for %s — will retry on the next stale hit.", cache_name, target_date)
                fut.set_exception(RuntimeError("background rebuild failed"))
            finally:
                with self._lock:
                    inflight.pop(target_date, None)

        threading.Thread(target=_run, daemon=True, name=f"{cache_name}-bg-{target_date}").start()

    def _build_slate_uncached(self, target_date: str) -> dict:
        # Fresh team/pitcher cache scope for THIS build: shared by every
        # game thread below (fixes the doubleheader double-fetch), never
        # stale across a later build_slate() call for a new date or an
        # explicit refresh=True.
        with self._team_cache_lock:
            self._team_hitting_cache.clear()
            self._team_pitching_cache.clear()
            self._team_home_away_cache.clear()
            self._team_roster_cache.clear()
            self._pitcher_profile_cache.clear()
            self._weather_mult_cache.clear()
            self._bullpen_fatigue_cache.clear()
            self._starter_recent_form_cache.clear()
            self._starter_arsenal_whiff_cache.clear()
            self._lineup_signal_cache.clear()

        games = self.mlb.schedule(target_date)
        prop_odds = self.odds.player_prop_index()
        moneyline_odds = self.odds.moneyline_index()

        player_rows: list[dict] = []
        game_rows: list[dict] = []
        pitcher_rows: dict[int, dict] = {}
        # See the build_slate() cache note above — tracks whether EVERY
        # game in this slate resolved a real, confirmed batting order
        # (rather than the arbitrary-roster-slice fallback) so the cache
        # gate above knows whether it's safe to trust this slate all day.
        all_lineups_confirmed = True

        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            future_to_game = {
                executor.submit(self._build_game, game, target_date, prop_odds, moneyline_odds): game
                for game in games
            }
            for future in as_completed(future_to_game):
                try:
                    packet = future.result()
                except Exception:
                    game_pk = future_to_game[future].get("gamePk", "unknown")
                    logger.exception(
                        "Failed to build game packet for gamePk=%s on %s — this game will be missing from the slate.",
                        game_pk, target_date,
                    )
                    all_lineups_confirmed = False
                    continue
                player_rows.extend(packet["players"])
                game_rows.append(packet["game"])
                pitcher_rows.update(packet["pitchers"])
                if not packet.get("lineupsConfirmed", False):
                    all_lineups_confirmed = False

        player_rows.sort(key=lambda item: item["props"]["hits"]["prob"], reverse=True)

        slate = {
            "meta": {
                "date": target_date,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "season": settings.season,
                "playerCount": len(player_rows),
                "gameCount": len(game_rows),
                "lineupsConfirmed": all_lineups_confirmed and bool(game_rows),
                "marketDataAvailable": self.odds.enabled,
            },
            "players": player_rows,
            "games": game_rows,
            # Internal bookkeeping only — not part of SlateResponse, stripped
            # automatically by FastAPI's response_model on the way out. See
            # the cache-gate note at the top of build_slate().
            "_cachedAt": datetime.now(timezone.utc),
        }

        with self._lock:
            self._slate_cache[target_date] = slate
            self._player_index = {row["mlbId"]: row for row in player_rows}
            self._pitcher_index.update(pitcher_rows)

        return slate

    # ------------------------------------------------------------------
    # Lineup extraction, hitters-only roster filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _lineup_batting_order(player_id: int, lineup_list: list[dict]) -> int | None:
        for idx, node in enumerate(lineup_list):
            if node.get("id") == player_id:
                return idx + 1
        return None

    def _resolve_hitters(self, team_id: int, lineup_list: list[dict], game_pk: int, target_date: str) -> tuple[list[dict], bool]:
        """Returns (hitters, confirmed). confirmed is True only when the
        9 names came from an actual posted batting order (schedule-hydrate
        lineups, or the boxscore battingOrder via probable_lineup()) —
        False when we had to fall back to an arbitrary roster slice.

        FIX (2026-09-03, diagnostic pass): this used to fall straight to
        `list(non_pitchers.values())[:9]` — the first 9 non-pitchers in
        whatever order MLB's roster endpoint happens to return (roster
        listing order, e.g. jersey number — NOT batting order or
        importance) — any time `lineup_list` (from the schedule's
        hydrate=lineups, only populated once MLB posts the day's lineup)
        was still empty. Since that's the normal state for most of the
        day until ~2-3h before first pitch, and build_slate() used to
        cache whatever it built on first request permanently, this
        silently served a random slice of the 26-man roster — bench bats
        and all — as "today's lineup" for the rest of the day. Now tries
        MLBClient.probable_lineup() first, which itself checks the
        boxscore's battingOrder (sometimes populated slightly earlier
        than the schedule hydrate) before falling back the same way — and
        reports whether either path actually found a real order, so
        build_slate() knows not to treat this result as final.
        """
        roster = self._cached_team_roster(team_id)
        non_pitchers = {
            p.get("person", {}).get("id"): p.get("person", {})
            for p in roster
            if p.get("position", {}).get("code") != "1" and p.get("person", {}).get("id")
        }
        if lineup_list:
            hitters = []
            for node in lineup_list[:9]:
                pid = node.get("id")
                if not pid:
                    continue
                fallback_name = node.get("fullName")
                person = non_pitchers.get(pid, {"id": pid, "fullName": fallback_name})
                hitters.append(person)
            if hitters:
                return hitters, True

        try:
            fallback_lineup, confirmed = self.mlb.probable_lineup(game_pk, team_id, target_date)
        except Exception:
            logger.warning(
                "probable_lineup() failed for game_pk=%s team_id=%s — falling back to raw roster slice.",
                game_pk, team_id, exc_info=True,
            )
            fallback_lineup, confirmed = [], False

        if fallback_lineup:
            hitters = []
            for node in fallback_lineup[:9]:
                pid = node.get("id")
                if not pid:
                    continue
                person = non_pitchers.get(pid, {"id": pid, "fullName": node.get("name")})
                hitters.append(person)
            if hitters:
                return hitters, confirmed

        return list(non_pitchers.values())[:9], False

    # ------------------------------------------------------------------
    # PERF — per-player raw data fetch, split out so it can be submitted to
    # a ThreadPoolExecutor per-hitter (see _build_game PASS 2). Still ~5-6
    # sequential MLB API calls for ONE player (person/season/platoon/
    # game_log/recent/ahead-in-count can't be meaningfully deduped — each
    # is genuinely player-specific), but now many hitters' bundles run
    # concurrently instead of one hitter fully blocking the next.
    # ------------------------------------------------------------------

    def _fetch_hitter_bundle(self, pid: int, opp_pitcher_hand: str, target_date: str) -> dict:
        bat_side = "R"
        try:
            bat_side = self.mlb.person(pid).get("batSide", {}).get("code") or "R"
        except Exception:
            logger.warning("Could not fetch batSide for player_id=%s — defaulting to 'R'.", pid, exc_info=True)
        return {
            "bat_side": bat_side,
            "hitting_stats": self.mlb.player_season_hitting(pid),
            "platoon_stats": self.mlb.hitter_platoon_split(pid, opp_pitcher_hand),
            "game_log": self.mlb.player_game_log(pid, target_date, days=50),
            "recent_14d": self.mlb.player_recent_hitting(pid, target_date, days=14),
            "ahead_avg": self.mlb.hitter_ahead_in_count_avg(pid),
        }

    def _build_game(
        self,
        game: dict,
        target_date: str,
        prop_odds: dict,
        moneyline_events: list[dict],
    ) -> dict:
        game_pk = int(game["gamePk"])
        teams = game.get("teams", {})
        away_node = teams.get("away", {})
        home_node = teams.get("home", {})

        away_id = int(away_node.get("team", {}).get("id", 0))
        home_id = int(home_node.get("team", {}).get("id", 0))
        away_name = away_node.get("team", {}).get("name", "Away")
        home_name = home_node.get("team", {}).get("name", "Home")
        away_abbr = TEAM_ABBREVIATIONS.get(away_id, "MLB")
        home_abbr = TEAM_ABBREVIATIONS.get(home_id, "MLB")

        venue_name = game.get("venue", {}).get("name", "")
        venue_id = game.get("venue", {}).get("id")
        game_date_utc = game.get("gameDate")
        park_factor = STADIUM_INDICES.get(venue_name, {"base_park_factor": 1.00})["base_park_factor"]

        away_pitcher_id = away_node.get("probablePitcher", {}).get("id")
        home_pitcher_id = home_node.get("probablePitcher", {}).get("id")
        away_pitcher = self._cached_pitcher_profile(away_pitcher_id)
        home_pitcher = self._cached_pitcher_profile(home_pitcher_id)

        away_team_hitting = self._cached_team_stats(away_id, "hitting")
        home_team_hitting = self._cached_team_stats(home_id, "hitting")
        away_team_pitching = self._cached_team_stats(away_id, "pitching")
        home_team_pitching = self._cached_team_stats(home_id, "pitching")

        away_team_obp = safe_float(away_team_hitting.get("obp"), LEAGUE_AVG_OBP)
        home_team_obp = safe_float(home_team_hitting.get("obp"), LEAGUE_AVG_OBP)
        away_team_slg = safe_float(away_team_hitting.get("slg"), LEAGUE_AVG_SLG)
        home_team_slg = safe_float(home_team_hitting.get("slg"), LEAGUE_AVG_SLG)
        away_bullpen_era = safe_float(away_team_pitching.get("era"), LEAGUE_AVG_ERA)
        home_bullpen_era = safe_float(home_team_pitching.get("era"), LEAGUE_AVG_ERA)

        # Home/away scoring factor — fetched (now cached) once per team per
        # slate build, not per hitter.
        away_runs, away_hits = self._cached_team_home_away(away_id, is_home=False)
        home_runs, home_hits = self._cached_team_home_away(home_id, is_home=True)
        away_home_away_scalar = home_away_scoring_factor(away_runs, away_hits)
        home_home_away_scalar = home_away_scoring_factor(home_runs, home_hits)

        # TASK (2026-08-31) — Moneylines team-strength signal upgrade (chat
        # items #1-#8). Each side's matchup multiplier now draws on:
        #   - the OPPOSING starter's proj_er/k_rate (already sitting on the
        #     away_pitcher/home_pitcher dicts from pitcher_profile() above)
        #     plus their recent-form (last 2-3 starts) and arsenal whiff%
        #     (items #1/#3/#5);
        #   - THIS side's actual/projected lineup strength for today,
        #     platoon-adjusted to the opposing starter's throwing hand and
        #     credibility-weighted (items #2/#4/#6/#7), dynamically blended
        #     against the season-wide OBP/SLG fallback based on whether
        #     that lineup is confirmed yet (item #8).
        # away_matchup_mult faces home_pitcher, so it's home_pitcher's
        # recent-form/whiff feeding it (and vice versa) — same "opposing"
        # pairing the ERA arg already used.
        away_pitcher_recent_form = self._cached_starter_recent_form(
            home_pitcher_id, home_pitcher.get("era", LEAGUE_AVG_ERA), home_pitcher.get("k_rate", LEAGUE_AVG_K_RATE),
        )
        home_pitcher_recent_form = self._cached_starter_recent_form(
            away_pitcher_id, away_pitcher.get("era", LEAGUE_AVG_ERA), away_pitcher.get("k_rate", LEAGUE_AVG_K_RATE),
        )
        away_faces_whiff = self._cached_starter_arsenal_whiff(home_pitcher_id)
        home_faces_whiff = self._cached_starter_arsenal_whiff(away_pitcher_id)

        away_lineup_ops, away_lineup_weight, _away_lineup_confirmed = self._today_lineup_signal(
            game_pk, away_id, home_pitcher_id, home_pitcher.get("hand", "R"), target_date,
        )
        home_lineup_ops, home_lineup_weight, _home_lineup_confirmed = self._today_lineup_signal(
            game_pk, home_id, away_pitcher_id, away_pitcher.get("hand", "R"), target_date,
        )

        away_matchup_mult = _matchup_multiplier(
            away_team_obp, away_team_slg, home_pitcher.get("era", LEAGUE_AVG_ERA),
            opp_proj_er=home_pitcher.get("proj_er"), opp_k_rate=home_pitcher.get("k_rate"),
            opp_recent_blended_era=away_pitcher_recent_form["blendedEra"],
            opp_recent_blended_k_rate=away_pitcher_recent_form["blendedKRate"],
            opp_arsenal_whiff_pct=away_faces_whiff,
            today_lineup_ops=away_lineup_ops, lineup_confidence_weight=away_lineup_weight,
        )
        home_matchup_mult = _matchup_multiplier(
            home_team_obp, home_team_slg, away_pitcher.get("era", LEAGUE_AVG_ERA),
            opp_proj_er=away_pitcher.get("proj_er"), opp_k_rate=away_pitcher.get("k_rate"),
            opp_recent_blended_era=home_pitcher_recent_form["blendedEra"],
            opp_recent_blended_k_rate=home_pitcher_recent_form["blendedKRate"],
            opp_arsenal_whiff_pct=home_faces_whiff,
            today_lineup_ops=home_lineup_ops, lineup_confidence_weight=home_lineup_weight,
        )

        # TASK 1 (2026-08-30) — Poisson Monte Carlo win% instead of the
        # closed-form Pythagenpat formula. This call happens once per game
        # inside _build_game(), which itself only runs once per game per
        # build_slate() call — build_slate()'s existing self._slate_cache
        # (see build_slate() above) already makes this per-slate, not
        # per-request, with zero extra caching plumbing needed here: a cache
        # hit in build_slate() short-circuits before _build_game() (and
        # therefore this simulation) ever runs again for that date.
        #
        # TASK 3 (2026-08-30) — calculate_team_xruns_v2() is no longer
        # called out here first and handed in as a plain mean; the quality
        # factors (matchup strength, park factor, weather multiplier,
        # opposing bullpen ERA) go straight into
        # poisson_monte_carlo_win_prob(), which now derives each side's
        # lambda internally right before sampling.
        #
        # TASK 1 (2026-08-30) — weather_mult is now real: venue coordinates
        # (via mlb.venue()) + Open-Meteo forecast at game time, converted
        # to a multiplier by weather_scoring_multiplier(). It's a single
        # per-GAME number (not per-side), same as park_factor.
        #
        # TASK 2 (2026-08-30) — bullpen_fatigue_mult_a/b are now real:
        # rolling 2-3 day relief-innings workload (MLB Stats API only, see
        # mlb.team_bullpen_relief_innings()), converted via
        # bullpen_fatigue_multiplier(). Each side's fatigue mult reflects
        # the BULLPEN THAT SIDE'S OFFENSE IS FACING — same "a/b tracks
        # whichever bullpen the side is batting against" contract
        # bullpen_era_a/b already has below.
        #
        # Side "a" = home, side "b" = away (matches the pre-TASK-3
        # home_xruns/away_xruns convention below exactly): bullpen_era_a is
        # the AWAY bullpen's ERA because the HOME team is the one batting
        # against it, and vice versa for bullpen_era_b — bullpen_fatigue_
        # mult_a/b follow the same away/home pairing.
        weather_ctx = self._cached_weather_context(venue_id, game_date_utc)
        weather_mult = weather_ctx.mult
        away_bullpen_fatigue_mult = self._cached_bullpen_fatigue_mult(away_id, target_date)
        home_bullpen_fatigue_mult = self._cached_bullpen_fatigue_mult(home_id, target_date)
        mc_result = poisson_monte_carlo_win_prob(
            matchup_mult_a=home_matchup_mult, matchup_mult_b=away_matchup_mult,
            park_factor=park_factor, weather_mult=weather_mult,
            bullpen_era_a=away_bullpen_era, bullpen_era_b=home_bullpen_era,
            bullpen_fatigue_mult_a=away_bullpen_fatigue_mult, bullpen_fatigue_mult_b=home_bullpen_fatigue_mult,
        )
        home_win_prob, away_win_prob = mc_result.prob_a, mc_result.prob_b
        home_xruns, away_xruns = mc_result.mean_a, mc_result.mean_b

        ml_odds = self.odds.find_moneyline_game(moneyline_events, away_name, home_name)
        away_book, home_book = ml_odds.get("away"), ml_odds.get("home")
        away_market_prob, home_market_prob = remove_vig(away_book, home_book)
        away_edge = None if away_market_prob is None else round(away_win_prob - away_market_prob, 4)
        home_edge = None if home_market_prob is None else round(home_win_prob - home_market_prob, 4)

        game_id = str(game_pk)
        lineups = game.get("lineups", {})
        away_lineup_list = lineups.get("awayPlayers", [])
        home_lineup_list = lineups.get("homePlayers", [])

        players: list[dict] = []
        # PERF (2026-08-29): collected here and flushed ONCE at the end of
        # this game via repository.upsert_predictions_batch(), instead of
        # calling repository.upsert_prediction() (its own sqlite3 connect
        # + commit) for every single prop of every single hitter. See the
        # docstring on upsert_predictions_batch for why that mattered.
        prediction_rows: list[tuple] = []

        team_iter = [
            (away_id, away_abbr, away_name, away_team_obp, away_team_slg, home_pitcher, home_bullpen_era, away_lineup_list, away_home_away_scalar),
            (home_id, home_abbr, home_name, home_team_obp, home_team_slg, away_pitcher, away_bullpen_era, home_lineup_list, home_home_away_scalar),
        ]

        # --- PASS 1: resolve hitters + per-team opposing-pitcher context for
        # BOTH teams into one flat job list. No new I/O here beyond what
        # _resolve_hitters already needed (team roster, now cached).
        hitter_jobs: list[dict] = []
        lineups_confirmed = True
        for team_id, team_abbr, team_name, team_obp, team_slg, opp_pitcher, opp_bullpen_era, lineup_list, home_away_scalar in team_iter:
            hitters, team_lineup_confirmed = self._resolve_hitters(team_id, lineup_list, game_pk, target_date)
            if not team_lineup_confirmed:
                lineups_confirmed = False
            opp_pitcher_hand = opp_pitcher.get("hand", "R")
            opp_pitcher_era = opp_pitcher.get("era", LEAGUE_AVG_ERA)
            opp_pitcher_velo = opp_pitcher.get("avg_fastball_velo", 0.0)
            split_type = analyze_pitcher_split(
                opp_pitcher_hand, opp_pitcher.get("avg_vs_left", 0.250), opp_pitcher.get("avg_vs_right", 0.250)
            )
            is_high_fatigue = opp_pitcher.get("is_high_fatigue", False)
            fatigue_factor = opp_pitcher.get("fatigue_factor", 1.0)

            for idx, person in enumerate(hitters, start=1):
                pid = person.get("id")
                pname = person.get("fullName")
                if not pid or not pname:
                    continue
                order = self._lineup_batting_order(pid, lineup_list) or idx
                hitter_jobs.append({
                    "pid": pid, "pname": pname, "order": order,
                    "team_abbr": team_abbr, "team_name": team_name,
                    "team_obp": team_obp, "team_slg": team_slg,
                    "opp_pitcher": opp_pitcher, "opp_pitcher_hand": opp_pitcher_hand,
                    "opp_pitcher_era": opp_pitcher_era, "opp_pitcher_velo": opp_pitcher_velo,
                    "opp_bullpen_era": opp_bullpen_era, "split_type": split_type,
                    "is_high_fatigue": is_high_fatigue, "fatigue_factor": fatigue_factor,
                    "home_away_scalar": home_away_scalar,
                    # NEW (2026-09-08, diagnostic pass): per-TEAM confirmation
                    # status, not just the per-GAME aggregate — see
                    # PlayerResponse.lineupConfirmed.
                    "team_lineup_confirmed": team_lineup_confirmed,
                })

        # --- PASS 2: fetch every hitter's raw per-player data CONCURRENTLY
        # across the WHOLE GAME (both teams' lineups together), instead of
        # one hitter fully sequential (person -> season -> platoon ->
        # game_log -> recent_14d -> ahead_avg, ~5-6 calls) before starting
        # the next. This is the actual fix for "5+ sequential calls per
        # hitter inside a per-game thread" — per-player fetches now run in
        # parallel, not just per-game. See _fetch_hitter_bundle.
        bundles: dict[int, dict | None] = {}
        if hitter_jobs:
            with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
                future_to_pid = {
                    executor.submit(self._fetch_hitter_bundle, job["pid"], job["opp_pitcher_hand"], target_date): job["pid"]
                    for job in hitter_jobs
                }
                for future in as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        bundles[pid] = future.result()
                    except Exception:
                        logger.exception(
                            "Failed to fetch hitter bundle for player_id=%s in game_id=%s — this player will be "
                            "skipped from the slate.",
                            pid, game_id,
                        )
                        bundles[pid] = None

        # --- PASS 3: pure CPU compute (Stage A/B, HR/Run/RBI) + persistence,
        # sequential — cheap now that all I/O already happened in PASS 2.
        # Formulas below are byte-for-byte unchanged from before this
        # refactor; only the fetch orchestration around them moved.
        for job in hitter_jobs:
            pid, pname, order = job["pid"], job["pname"], job["order"]
            bundle = bundles.get(pid)
            if not bundle:
                continue

            team_abbr, team_name = job["team_abbr"], job["team_name"]
            team_obp, team_slg = job["team_obp"], job["team_slg"]
            opp_pitcher = job["opp_pitcher"]
            opp_pitcher_hand = job["opp_pitcher_hand"]
            opp_pitcher_era = job["opp_pitcher_era"]
            opp_pitcher_velo = job["opp_pitcher_velo"]
            opp_bullpen_era = job["opp_bullpen_era"]
            split_type = job["split_type"]
            is_high_fatigue = job["is_high_fatigue"]
            fatigue_factor = job["fatigue_factor"]
            home_away_scalar = job["home_away_scalar"]
            team_lineup_confirmed = job["team_lineup_confirmed"]

            bat_side = bundle["bat_side"]
            hitting_stats = bundle["hitting_stats"]
            platoon_stats = bundle["platoon_stats"]
            game_log = bundle["game_log"]
            recent_14d = bundle["recent_14d"]
            ahead_avg = bundle["ahead_avg"]

            at_bats = int(hitting_stats.get("atBats", 0) or 0)
            pa_sample = int(hitting_stats.get("plateAppearances", 0) or at_bats)
            season_so = int(hitting_stats.get("strikeOuts", 0) or 0)
            hitter_whiff_proxy = fastball_whiff_proxy(season_so, pa_sample)

            platoon_mult = apply_split_effect(1.0, bat_side, opp_pitcher_hand, split_type)

            stage_a_prob, iso_val, babip_14d, gen_ba = self._compute_hit_probability_stage_a(
                season_stats=hitting_stats,
                platoon_stats=platoon_stats,
                game_log=game_log,
                recent_14d=recent_14d,
                ahead_in_count_avg=ahead_avg,
                batting_order=order,
                park_factor=park_factor,
                home_away_scalar=home_away_scalar,
                is_high_fatigue=is_high_fatigue,
                fatigue_factor=fatigue_factor,
                opp_bullpen_era=opp_bullpen_era,
                opp_pitcher_era=opp_pitcher_era,
                opp_pitcher_k_rate=opp_pitcher.get("k_rate", 0.22),
                opp_pitcher_whip=opp_pitcher.get("whip", 1.32),
                platoon_mult=platoon_mult,
            )

            hit_prob = self._apply_hit_stage_b(
                stage_a_prob=stage_a_prob,
                batting_order=order,
                iso_val=iso_val,
                babip_14d=babip_14d,
                pitcher_velo=opp_pitcher_velo,
                hitter_whiff_pct=hitter_whiff_proxy,
            )

            pa_proj = projected_pa_from_order(order)

            obs_hr_rate = int(hitting_stats.get("homeRuns", 0) or 0) / at_bats if at_bats else LEAGUE_AVG_HR_RATE
            obs_run_rate = int(hitting_stats.get("runs", 0) or 0) / pa_sample if pa_sample else LEAGUE_AVG_RUN_RATE
            obs_rbi_rate = int(hitting_stats.get("rbi", 0) or 0) / pa_sample if pa_sample else LEAGUE_AVG_RBI_RATE
            hr_rate = shrink_rate(obs_hr_rate, at_bats, LEAGUE_AVG_HR_RATE, K_HR)
            run_rate = shrink_rate(obs_run_rate, pa_sample, LEAGUE_AVG_RUN_RATE, K_RUNS)
            rbi_rate = shrink_rate(obs_rbi_rate, pa_sample, LEAGUE_AVG_RBI_RATE, K_RBI)
            iso_mult = clamp(iso_val / LEAGUE_AVG_ISO if LEAGUE_AVG_ISO else 1.0, 0.75, 1.35)

            # --- HR pipeline: fatigue decay + ahead-in-count boost ---
            # Same triggers as Stage A's Pillar 6 (late-inning fatigue) and
            # Pillar 4 (ahead-in-count bias). Previously only the Hits
            # pipeline saw these; platoon DNA (platoon_mult, below) was
            # already shared across Hits/HR since both draw from the same
            # apply_split_effect() call earlier in this loop.
            hr_fatigue_mult = fatigue_factor if (is_high_fatigue and order in (5, 6, 7, 8, 9)) else 1.0
            hr_count_boost_mult = 1.0 + ahead_in_count_boost(ahead_avg, gen_ba)

            # hr_prob/run_prob/rbi_prob are already the full-game P(>=1 event)
            # across pa_proj plate appearances (see FIX 1 note at the top of
            # this file) — use them directly, exactly like hit_prob above.
            # Do NOT re-wrap these in another `1 - (1-p)**pa_proj`.
            hr_prob = process_hr_prob(
                hr_rate, pa_proj, opp_pitcher_era, opp_bullpen_era, park_factor, platoon_mult, iso_val, obs_hr_rate,
                fatigue_mult=hr_fatigue_mult, count_boost_mult=hr_count_boost_mult,
            )
            run_prob = process_run_prob(
                run_rate, pa_proj, opp_pitcher_era, opp_bullpen_era, park_factor, platoon_mult, iso_mult, team_obp, order
            )
            rbi_prob = process_rbi_prob(
                rbi_rate, pa_proj, opp_pitcher_era, opp_bullpen_era, park_factor, platoon_mult, iso_mult, team_obp, order, team_slg
            )

            hit_quote = self.odds.find_player_prop(prop_odds, pname, "hits")
            hr_quote = self.odds.find_player_prop(prop_odds, pname, "homeruns")
            rbi_quote = self.odds.find_player_prop(prop_odds, pname, "rbi")
            run_quote = self.odds.find_player_prop(prop_odds, pname, "runs")

            players.append({
                "id": f"{game_pk}-{pid}",
                "mlbId": pid,
                "name": pname,
                "team": team_abbr,
                "teamName": team_name,
                "gameId": game_id,
                "oppPitcher": opp_pitcher.get("name"),
                "oppPitcherId": opp_pitcher.get("id"),
                "order": order,
                "bats": bat_side,
                "projectedPA": pa_proj,
                # NEW (2026-09-08, diagnostic pass): per-team lineup status —
                # replaces the old whole-slate-only banner (SlateMeta.
                # lineupsConfirmed) with a status that's actually correct
                # when you've filtered the Props table down to one team.
                "lineupConfirmed": team_lineup_confirmed,
                "props": {
                    "hits": _quote_from_market(hit_prob, pa_sample, hit_quote),
                    "homeruns": _quote_from_market(hr_prob, pa_sample, hr_quote),
                    "rbi": _quote_from_market(rbi_prob, pa_sample, rbi_quote),
                    "runs": _quote_from_market(run_prob, pa_sample, run_quote),
                },
                "stats": {
                    "xwoba": None, "xba": None, "barrel": None, "hardhit": None,
                    "whiff": hitter_whiff_proxy,
                    "babip14d": round(babip_14d, 3),
                    "iso": round(iso_val, 3),
                    "plateAppearances": pa_sample,
                    "atBats": at_bats,
                },
            })

            # Queue each prop so /api/track-record and sync_results() have
            # something to grade against once games finish (see FIX 2 note
            # at the top of this file) — upsert so re-building the same
            # slate before first pitch just refreshes the line, not a dupe.
            # Actual DB write happens once for the whole game, below.
            for prop_type, prob_value, quote in (
                ("hits", hit_prob, hit_quote),
                ("homeruns", hr_prob, hr_quote),
                ("rbi", rbi_prob, rbi_quote),
                ("runs", run_prob, run_quote),
            ):
                prediction_rows.append((
                    game_pk,
                    target_date,
                    pid,
                    pname,
                    prop_type,
                    prob_value,
                    prob_to_american(prob_value),
                    (quote or {}).get("over"),
                ))

        try:
            self.repository.upsert_predictions_batch(prediction_rows)
        except Exception:
            # Never let a persistence hiccup break slate building — the
            # live quotes already went into `players` above regardless.
            logger.warning(
                "Failed to persist %d predictions for game_pk=%s — the live quotes still went out in this "
                "response, but these rows won't show up in /api/track-record.",
                len(prediction_rows), game_pk, exc_info=True,
            )

        pitchers_out: dict[int, dict] = {}
        for pitcher, team_abbr in ((away_pitcher, away_abbr), (home_pitcher, home_abbr)):
            pid = pitcher.get("id")
            if not pid:
                continue
            proj_k = pitcher.get("proj_k", 5.0)
            proj_er = pitcher.get("proj_er", 2.5)
            pitchers_out[pid] = {
                "id": pid, "name": pitcher.get("name", "Unknown"), "team": team_abbr,
                "throws": pitcher.get("hand", "R"), "projK": proj_k, "projER": proj_er,
                "era": pitcher.get("era", LEAGUE_AVG_ERA), "whip": pitcher.get("whip", 1.32),
                "statsSource": pitcher.get("stats_source", "league-fallback"),
                "splits": pitcher.get("splits", {}),
                "propLadder": {
                    "k": build_pitcher_ladder(proj_k, _ladder_lines(proj_k)),
                    "bb": build_pitcher_ladder(proj_k * 0.35, _ladder_lines(proj_k * 0.35)),
                    "er": build_pitcher_ladder(proj_er, _ladder_lines(proj_er)),
                },
            }

        game_row = {
            "id": game_id, "gamePk": game_pk, "date": target_date, "venue": venue_name,
            "away": away_abbr, "home": home_abbr, "awayName": away_name, "homeName": home_name,
            "awayProb": round(away_win_prob, 3), "homeProb": round(home_win_prob, 3),
            "awayOdds": prob_to_american(away_win_prob), "homeOdds": prob_to_american(home_win_prob),
            "awayBookOdds": away_book, "homeBookOdds": home_book,
            "awayEdge": away_edge, "homeEdge": home_edge,
            "awayPitcher": away_pitcher.get("name"), "homePitcher": home_pitcher.get("name"),
            "awayPitcherId": away_pitcher.get("id"), "homePitcherId": home_pitcher.get("id"),
            "awayXRuns": round(away_xruns, 2), "homeXRuns": round(home_xruns, 2),
            # TASK 1 UI (2026-08-31) — weather badge fields for the
            # Moneylines card, sourced from the same cached WeatherContext
            # that already feeds weather_mult above (no extra fetch).
            "weatherTempF": weather_ctx.temp_f,
            "weatherSummary": weather_ctx.summary,
            "weatherTone": weather_ctx.wind_tone,
            "weatherDetail": weather_ctx.wind_detail,
            # NEW (2026-09-04) — see schemas.GameResponse.lineupsConfirmed note.
            "lineupsConfirmed": lineups_confirmed,
        }

        return {"game": game_row, "players": players, "pitchers": pitchers_out, "lineupsConfirmed": lineups_confirmed}

    # ------------------------------------------------------------------
    # STAGE A — ported from process_abi_single_hitter. Returns a probability
    # already hard-clamped to [0.02, 0.80], matching the original's `prob =
    # round(max(0.02, min(0.80, prob)), 4)` at the end of that function.
    # ------------------------------------------------------------------

    def _compute_hit_probability_stage_a(
        self,
        season_stats: dict,
        platoon_stats: dict,
        game_log: list[dict],
        recent_14d: dict,
        ahead_in_count_avg: float | None,
        batting_order: int,
        park_factor: float,
        home_away_scalar: float,
        is_high_fatigue: bool,
        fatigue_factor: float,
        opp_bullpen_era: float,
        opp_pitcher_era: float,
        opp_pitcher_k_rate: float,
        opp_pitcher_whip: float,
        platoon_mult: float,
    ) -> tuple[float, float, float, float]:
        league_avg_ba = LEAGUE_AVG_BA
        league_avg_obp = LEAGUE_AVG_OBP

        player_total_pa = int(season_stats.get("plateAppearances", 0) or 0)
        raw_gen_ba = safe_float(season_stats.get("avg"), league_avg_ba)
        raw_gen_obp = safe_float(season_stats.get("obp"), league_avg_obp)

        if player_total_pa < 20:
            weight = player_total_pa / 20.0
            gen_ba = (raw_gen_ba * weight) + (league_avg_ba * (1.0 - weight))
            gen_obp = (raw_gen_obp * weight) + (league_avg_obp * (1.0 - weight))
        else:
            gen_ba, gen_obp = raw_gen_ba, raw_gen_obp

        # --- Platoon blend (Pillars 2 & 3) ---
        plat_ab = int(platoon_stats.get("atBats", 0) or 0)
        if plat_ab <= 10:
            adj_ba, adj_obp, matchup_boost_ratio = gen_ba, gen_obp, 1.0
        else:
            plat_ba = safe_float(platoon_stats.get("avg"), gen_ba)
            plat_obp = safe_float(platoon_stats.get("obp"), gen_obp)
            weight_plat = min(1.0, plat_ab / 60.0)
            adj_ba = (plat_ba * weight_plat) + (gen_ba * (1.0 - weight_plat))
            adj_obp = (plat_obp * weight_plat) + (gen_obp * (1.0 - weight_plat))
            matchup_boost_ratio = clamp(adj_ba / max(gen_ba, 0.200), 0.85, 1.15)

        base_score = adj_ba * (1.0 + ((adj_obp - adj_ba) / 2.0)) * matchup_boost_ratio

        # --- Pillar 4: ahead-in-count bias overlay ---
        count_boost = ahead_in_count_boost(ahead_in_count_avg, gen_ba)
        base_score *= (1.0 + count_boost)

        # --- Pillar 5 (OFC) intentionally skipped — dead code in the
        # original (hardcoded opposite_pct=0.26 never clears the >=0.28
        # gate), see module docstring.

        # --- Home/away scoring factor ---
        base_score *= home_away_scalar

        # --- Recent-form: 5/10/15-game wOBA-weighted buckets ---
        valid_games = [g for g in game_log if safe_float(g.get("stat", {}).get("atBats", 0)) > 0]
        recent_15 = valid_games[-15:]

        def _bucket_woba(games: list[dict]) -> float:
            if not games:
                return base_score
            combined = {
                key: sum(safe_float(g.get("stat", {}).get(key, 0)) for g in games)
                for key in ("atBats", "baseOnBalls", "sacFlies", "hits", "doubles", "triples", "homeRuns")
            }
            return min(0.420, calculate_woba_from_stats(combined, base_score))

        if recent_15:
            recent_15 = list(reversed(recent_15))
            b1 = _bucket_woba(recent_15[0:5])
            b2 = _bucket_woba(recent_15[5:10])
            b3 = _bucket_woba(recent_15[10:15])
            form_value = (b1 * 0.55) + (b2 * 0.325) + (b3 * 0.125)
            raw_score = (base_score * 0.60) + (form_value * 0.40)
        else:
            raw_score = base_score

        # --- Pillar 6: late-inning fatigue decay ---
        if is_high_fatigue and batting_order in (5, 6, 7, 8, 9):
            raw_score *= fatigue_factor

        # --- Weather/park cap (±8%) ---
        weather_mult = clamp(park_factor, 0.92, 1.08)
        raw_score *= weather_mult

        # --- Lineup-position PA scaling (the CORE table — see
        # projected_pa_from_order's docstring for why this one, not the
        # tab_hit display table) ---
        projected_pa = projected_pa_from_order(batting_order)

        if projected_pa < 2.0:
            return round(MIN_PROB, 4), 0.0, LEAGUE_AVG_BABIP, gen_ba

        # --- Binomial core ---
        hit_rate_per_ab = raw_score * 0.85
        raw_hit_prob = 1.0 - math.pow(max(0.0, 1.0 - hit_rate_per_ab), projected_pa)

        # BUGFIX (2026-09-03, diagnostic pass): the original restoration of
        # this term ("BUGFIX 2026-09-03" below, now corrected further) copied
        # math_engine's process_hr_prob/process_run_prob/process_rbi_prob
        # starter_mult formula shape — but that formula itself had the ERA
        # direction inverted: (LEAGUE_AVG_ERA - opp_pitcher_era) makes a
        # low-ERA ACE produce a multiplier ABOVE 1.0, which INCREASES the
        # batter's hit probability against the toughest pitchers on the
        # slate, and a high-ERA replacement-level arm produces a multiplier
        # BELOW 1.0, suppressing it. That's backwards — it's the single
        # biggest reason weak-team bench bats kept out-ranking legitimate
        # stars: better opposing pitching was making the model MORE
        # confident in the hit, not less. Also fixed the matching
        # (LEAGUE_AVG_ERA - opp_bullpen_era) inversion in bullpen_mult below,
        # and the same shape in math_engine.py's process_hit_prob/
        # process_run_prob/process_hr_prob/process_rbi_prob,
        # starter_quality_index, and calculate_team_xruns_v2 (Moneylines).
        starter_era_mult = 1.0 + ((opp_pitcher_era - LEAGUE_AVG_ERA) / 10.0)
        prob = (
            raw_hit_prob
            * starter_era_mult
            * (1.0 - (opp_pitcher_k_rate * 0.22))
            * (1.0 + ((1.20 - opp_pitcher_whip) * 0.05))
        )

        # Bullpen cap — approximated from opposing bullpen ERA vs league avg
        # (the original's calculate_team_bullpen_fatigue_multiplier also
        # factors in innings/games workload ratio, which isn't cheaply
        # available from the free MLB Stats API; ERA is the dominant term).
        bullpen_mult = clamp(1.0 + ((opp_bullpen_era - LEAGUE_AVG_ERA) / 25.0), 0.95, 1.06)
        prob *= bullpen_mult

        # --- Platoon DNA accelerator ---
        prob *= platoon_mult

        # --- Poisson decay for short PA projections ---
        if projected_pa < 3.5:
            prob *= math.exp(projected_pa - 3.5)

        # --- ISO (season) for quality-modifier + reporting ---
        at_bats = safe_float(season_stats.get("atBats", 0))
        hits = safe_float(season_stats.get("hits", 0))
        doubles = safe_float(season_stats.get("doubles", 0))
        triples = safe_float(season_stats.get("triples", 0))
        hr = safe_float(season_stats.get("homeRuns", 0))
        iso_val = 0.0
        if at_bats >= 5:
            singles = hits - (doubles + triples + hr)
            slugging = (singles + 2 * doubles + 3 * triples + 4 * hr) / at_bats
            iso_val = slugging - (hits / at_bats)

        # --- 14-day BABIP (for stage B's regression penalty + quality mod) ---
        babip_14d = LEAGUE_AVG_BABIP
        ab_14d = safe_float(recent_14d.get("atBats", 0))
        hits_14d = safe_float(recent_14d.get("hits", 0))
        hr_14d = safe_float(recent_14d.get("homeRuns", 0))
        so_14d = safe_float(recent_14d.get("strikeOuts", 0))
        sf_14d = safe_float(recent_14d.get("sacFlies", 0))
        bip = ab_14d - so_14d - hr_14d + sf_14d
        if bip >= 10:
            babip_14d = clamp((hits_14d - hr_14d) / bip, 0.0, 0.6)

        # --- THE STAGE-A CLAMP — matches the original's final hard clamp
        # inside process_abi_single_hitter, BEFORE the tab_hit post-
        # processing block ever sees the number. ---
        prob = round(clamp(prob, 0.02, 0.80), 4)

        # gen_ba is returned alongside the usual (prob, iso_val, babip_14d) so
        # the HR pipeline can reuse the same regressed baseline average for its
        # own ahead-in-count boost (see _build_game) instead of recomputing the
        # small-sample shrinkage a second time.
        return prob, iso_val, babip_14d, gen_ba

    # ------------------------------------------------------------------
    # STAGE B — ported from the tab_hit post-processing block ("BATAS
    # 1/2/3" + V8.9 quality modifier). Takes Stage A's clamped [0.02, 0.80]
    # output as its starting point (same as the original: tab_hit read
    # `Hit 1+ %` — Stage A's output — as `base_prob_raw` and built on it),
    # and produces the final [0.0, 1.0] probability.
    # ------------------------------------------------------------------

    def _apply_hit_stage_b(
        self,
        stage_a_prob: float,
        batting_order: int,
        iso_val: float,
        babip_14d: float,
        pitcher_velo: float,
        hitter_whiff_pct: float,
    ) -> float:
        prob = stage_a_prob * 100.0  # original operates in 0-100 space here

        # BATAS 1 — velocity control
        velocity_penalty_pts, _triggered = velocity_control_penalty(pitcher_velo, hitter_whiff_pct)
        prob += velocity_penalty_pts

        # BATAS 2 — BABIP regression (now scaled by ISO — see
        # math_engine.babip_regression_penalty_points's 2026-09-03 note)
        prob += babip_regression_penalty_points(babip_14d, iso_val)

        # BATAS 3 — lineup order bonus
        prob *= lineup_order_bonus_mult(batting_order)

        # V8.9 quality modifier + the second (multiplicative) velocity mod
        quality_mod = hit_quality_modifier(babip_14d, iso_val)
        velo_mod = pitcher_velocity_mod(pitcher_velo)
        final_calibrated_prob = prob * (1.0 + quality_mod + velo_mod)

        return round(clamp(final_calibrated_prob / 100.0, 0.0, 1.0), 4)

    # ------------------------------------------------------------------
    # CONSOLIDATION — Pitcher Props (/api/pitcher-props)
    # ------------------------------------------------------------------

    def _resolve_probable_pitcher(self, team_node: dict, game_pk: int | None, team_id: int | None) -> tuple[int | None, str | None]:
        """FIX (2026-09-10, diagnostic pass): MLB's schedule hydrate only
        populates probablePitcher once it's been announced — a TBD rotation
        slot, a rookie/Triple-A call-up not yet locked in, or just fetching
        early in the day all leave this empty, and the entire matchup
        silently disappeared from Pitcher Props / Team Matchups / Matchup
        Analyzer as a result (no pid meant the job never got queued at
        all). Falls back to that game's boxscore, which is sometimes
        populated with a starter before the schedule hydrate catches up —
        same "try a second real source before giving up" pattern already
        used for lineups (see probable_lineup). Still returns (None, None)
        if genuinely nothing is available anywhere yet — there's no
        legitimate name to show at that point.
        """
        prob_pitcher = team_node.get("probablePitcher") or {}
        pid = prob_pitcher.get("id")
        pname = prob_pitcher.get("fullName")
        if pid:
            return pid, pname
        if not game_pk or not team_id:
            return None, None
        try:
            box = self.mlb.boxscore(game_pk)
            for side in ("home", "away"):
                team_box = box.get("teams", {}).get(side, {})
                if team_box.get("team", {}).get("id") != team_id:
                    continue
                pitcher_ids = team_box.get("pitchers", [])
                if not pitcher_ids:
                    return None, None
                starter_id = pitcher_ids[0]
                return starter_id, self.mlb.person(starter_id).get("fullName")
        except Exception:
            pass
        return None, None

    def build_pitcher_props(self, target_date: str, force: bool = False) -> list[dict]:
        return self._cached_list_build(
            self._pitcher_props_cache, self._pitcher_props_inflight, target_date, force,
            self._build_pitcher_props_uncached, "pitcher-props",
        )

    def _build_pitcher_props_uncached(self, target_date: str) -> list[dict]:
        games = self.mlb.schedule(target_date)
        jobs: list[tuple] = []
        seen_pitcher_ids: set[int] = set()
        for game in games:
            teams = game.get("teams", {})
            for side in ("away", "home"):
                team_node = teams.get(side, {})
                opp_node = teams.get("home" if side == "away" else "away", {})
                team_name = team_node.get("team", {}).get("name", "")
                team_id = team_node.get("team", {}).get("id")
                opponent_name = opp_node.get("team", {}).get("name", "")
                opponent_team_id = opp_node.get("team", {}).get("id")
                is_home = side == "home"
                pid, pname = self._resolve_probable_pitcher(team_node, game.get("gamePk"), team_id)
                if pid and pid not in seen_pitcher_ids:
                    seen_pitcher_ids.add(pid)
                    jobs.append((pid, pname, team_name, opponent_name, is_home, opponent_team_id))

        results: list[dict] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
                futures = {executor.submit(self._build_pitcher_props_entry, *job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        entry = future.result()
                    except Exception:
                        job = futures[future]
                        logger.exception(
                            "Failed to build pitcher-props entry for pitcher_id=%s (%s) on %s — skipping.",
                            job[0], job[1], target_date,
                        )
                        continue
                    if entry:
                        results.append(entry)

        return results

    def _build_pitcher_props_entry(
        self, pitcher_id: int, pitcher_name: str, team_name: str, opponent_name: str, is_home: bool, opponent_team_id: int | None = None,
    ) -> dict | None:
        games = self.mlb.pitcher_recent_pitching_log(pitcher_id, limit=10)
        if not games:
            return None

        history_k, history_er, history_bb, history_outs = [], [], [], []
        labels, match_details = [], []
        for g in games:
            s = g.get("stat", {})
            k = int(s.get("strikeOuts", 0) or 0)
            er = int(s.get("earnedRuns", 0) or 0)
            bb = int(s.get("baseOnBalls", 0) or 0)
            history_k.append(k)
            history_er.append(er)
            history_bb.append(bb)
            history_outs.append(round(parse_innings_pitched(s.get("inningsPitched", "0.0")) * 3))
            labels.append(_format_date_label(g.get("date", ""), g.get("isHome", True), (g.get("opponent") or {}).get("name", "OPP")))
            score_prefix = "W" if g.get("isWin", False) else "L"
            match_details.append({
                "score": score_prefix,
                "text": f"IP: {s.get('inningsPitched', '0.0')} | ER: {er} | BB: {bb} | K: {k}",
            })

        today_matchup = _matchup_label(opponent_name, is_home)
        return {
            "id": str(pitcher_id),
            "name": pitcher_name,
            "sub": f"{team_name} &middot; P &middot; {today_matchup}",
            "opponent": opponent_name,
            "isHome": is_home,
            "matchup": today_matchup,
            "markets": {
                # No anchor_mu here — pitcher K/ER/BB has no second projection
                # model to reconcile with, so this is byte-identical to the
                # retired script's calibrate_market/build_market output.
                "strikeouts": build_count_market("Strikeouts", history_k, DEFAULT_PROP_LINES["strikeouts"]),
                "earned_runs": build_count_market("Earned Runs", history_er, DEFAULT_PROP_LINES["earned_runs"], pad=1),
                "walks": build_count_market("Walks Allowed", history_bb, DEFAULT_PROP_LINES["walks"], pad=1),
            },
            "labels": labels,
            "matchDetails": match_details,
            "pitcherId": pitcher_id,
            "opponentTeamId": opponent_team_id,
            # NEW (2026-09-10, diagnostic pass) — Pitcher Quality Snapshot,
            # for the sidebar hover card. See _pitcher_quality_snapshot for
            # exactly what's real vs. what's honestly omitted.
            "qualitySnapshot": self._pitcher_quality_snapshot(pitcher_id, history_outs),
        }

    # NEW (2026-09-10, diagnostic pass) — Pitcher Quality Snapshot.
    #
    # Every number here is either a standard, real MLB Stats API field, a
    # well-established sabermetric convention (ERA+), or something this
    # codebase already computes elsewhere from real pitch-by-pitch data
    # (putAwayPct — see pitcher_pitch_lethality). Two things were
    # explicitly requested but are NOT included, on purpose, because doing
    # so honestly would mean fabricating them:
    #   - "Line-out rate": the standard season pitching stat line splits
    #     outs into groundOuts/airOuts only — line drives aren't tracked as
    #     their own category there. Rather than invent a split with no
    #     real backing, this exposes the real groundOuts/airOuts (GO/AO)
    #     read instead, which IS a genuine, standard batted-ball-tendency
    #     signal (groundball vs. flyball pitcher).
    #   - "Strikeout rate on 2-strike / 3-2 COUNTS specifically": count-
    #     situation splits aren't in the season stat API at all — getting
    #     this exactly would require parsing every pitch of every start
    #     this season, which isn't a reasonable per-request cost. What IS
    #     already computed from real pitch-by-pitch data in this codebase
    #     is putAwayPct: strikeouts specifically on 2-STRIKE pitches, from
    #     the pitcher's last several starts (pitcher_pitch_lethality) —
    #     the closest real answer to "how good is he at finishing hitters
    #     off once he's got two strikes," aggregated here across his whole
    #     arsenal instead of per-pitch-type.
    def _pitcher_quality_snapshot(self, pitcher_id: int, history_outs: list[int]) -> dict:
        with self._team_cache_lock:
            cached = self._pitcher_quality_cache.get(pitcher_id)
        if cached is not None:
            age = (datetime.now(timezone.utc) - cached["cachedAt"]).total_seconds()
            if age < PITCHER_QUALITY_CACHE_TTL_SECONDS:
                # projectedWorkload is the one part that's cheap to recompute
                # and legitimately DOES shift build-to-build (fresher recent-
                # outs history) — refreshed on every call even on a cache
                # hit, everything else (season stats, put-away rate) reused.
                snapshot = dict(cached["data"])
                snapshot["projectedWorkload"] = self._pitcher_projected_workload(history_outs)
                return snapshot

        snapshot = self._pitcher_quality_snapshot_uncached(pitcher_id)
        with self._team_cache_lock:
            self._pitcher_quality_cache[pitcher_id] = {"data": dict(snapshot), "cachedAt": datetime.now(timezone.utc)}
        snapshot["projectedWorkload"] = self._pitcher_projected_workload(history_outs)
        return snapshot

    def _pitcher_projected_workload(self, history_outs: list[int]) -> dict | None:
        # Same Poisson/NB2 calibration engine used for every other prop in
        # this app (calibrate_count_market), run on this pitcher's own
        # last-10-start outs-recorded history. mu is in OUTS internally
        # (consistent unit for the count-market model); converted to
        # innings (outs / 3) only for display.
        if not history_outs:
            return None
        outs_model = calibrate_count_market(history_outs, 15.5)
        mu_outs = outs_model["mu"]
        sd_outs = math.sqrt(outs_model["variance"]) if outs_model["variance"] > 0 else 0.0
        return {
            "projectedIP": round(mu_outs / 3.0, 1),
            "projectedIPLow": round(max(0.0, mu_outs - sd_outs) / 3.0, 1),
            "projectedIPHigh": round((mu_outs + sd_outs) / 3.0, 1),
            "sampleStarts": outs_model["sampleSize"],
            "distribution": outs_model["distribution"],
        }

    def _pitcher_quality_snapshot_uncached(self, pitcher_id: int) -> dict:
        season_pitching = self.mlb._first_stat(self.mlb.person_stats(pitcher_id, group="pitching", stats="season"))
        era_raw = season_pitching.get("era")
        whip_raw = season_pitching.get("whip")
        era = safe_float(era_raw) if era_raw not in (None, "-", "", "-.--") else None
        whip = safe_float(whip_raw) if whip_raw not in (None, "-", "", "-.--") else None
        k9 = safe_float(season_pitching.get("strikeoutsPer9Inn"), None)
        bb9 = safe_float(season_pitching.get("walksPer9Inn"), None)
        ground_outs = int(season_pitching.get("groundOuts", 0) or 0)
        air_outs = int(season_pitching.get("airOuts", 0) or 0)

        go_ao_ratio = round(ground_outs / air_outs, 2) if air_outs > 0 else None
        if go_ao_ratio is None:
            tendency = "unknown"
        elif go_ao_ratio >= 1.5:
            tendency = "ground_ball"
        elif go_ao_ratio <= 0.9:
            tendency = "fly_ball"
        else:
            tendency = "balanced"

        # ERA+ (era_plus > 100 = better than league-average ERA) — the
        # same standard convention used across baseball media, computed
        # here from this app's own LEAGUE_AVG_ERA constant so it's
        # consistent with every other ERA-relative calc in this codebase.
        era_plus = round(LEAGUE_AVG_ERA / era * 100) if era and era > 0 else None
        if era_plus is None:
            tier, tier_note = "average", "Insufficient ERA sample to grade."
        elif era_plus >= 130:
            tier, tier_note = "elite", f"ERA+ {era_plus} (ERA {era:.2f} vs. league {LEAGUE_AVG_ERA:.2f})"
        elif era_plus >= 110:
            tier, tier_note = "above_average", f"ERA+ {era_plus} (ERA {era:.2f} vs. league {LEAGUE_AVG_ERA:.2f})"
        elif era_plus >= 90:
            tier, tier_note = "average", f"ERA+ {era_plus} (ERA {era:.2f} vs. league {LEAGUE_AVG_ERA:.2f})"
        elif era_plus >= 70:
            tier, tier_note = "below_average", f"ERA+ {era_plus} (ERA {era:.2f} vs. league {LEAGUE_AVG_ERA:.2f})"
        else:
            tier, tier_note = "poor", f"ERA+ {era_plus} (ERA {era:.2f} vs. league {LEAGUE_AVG_ERA:.2f})"

        # Two-strike put-away rate, aggregated across the full arsenal —
        # real pitch-by-pitch data from this pitcher's own recent starts.
        put_away_pct, put_away_sample = None, 0
        try:
            recent_pks = self.mlb.pitcher_recent_game_pks(pitcher_id)
            lethality = self.mlb.pitcher_pitch_lethality(pitcher_id, recent_pks) if recent_pks else {}
            total_two_strike_pitches = sum(v.get("sampleSize", 0) or 0 for v in lethality.values())
            # pitcher_pitch_lethality returns rates, not raw counts, per
            # pitch type — recompute the raw twoStrikePitches/twoStrikeKs
            # totals isn't exposed there, so this reconstructs a
            # sample-size-weighted average from what IS exposed
            # (putAwayPct per pitch type + that pitch's own sample size)
            # rather than re-parsing play-by-play a second time.
            weighted_sum, weight_total = 0.0, 0
            for v in lethality.values():
                if v.get("putAwayPct") is not None and v.get("sampleSize"):
                    weighted_sum += v["putAwayPct"] * v["sampleSize"]
                    weight_total += v["sampleSize"]
            if weight_total > 0:
                put_away_pct = round(weighted_sum / weight_total, 1)
                put_away_sample = weight_total
        except Exception:
            logger.warning("Pitcher quality snapshot: put-away aggregation failed for pitcher_id=%s.", pitcher_id, exc_info=True)

        # Projected workload is computed separately by
        # _pitcher_projected_workload() every call (cheap, and legitimately
        # benefits from the freshest recent-outs history) — not cached here.

        return {
            "era": era, "whip": whip, "k9": k9, "bb9": bb9,
            "battedBallProfile": {"groundOuts": ground_outs, "airOuts": air_outs, "goAoRatio": go_ao_ratio, "tendency": tendency},
            "qualityTier": {"tier": tier, "note": tier_note},
            "twoStrikePutAwayPct": put_away_pct,
            "twoStrikeSampleSize": put_away_sample,
        }

    # ------------------------------------------------------------------
    # CONSOLIDATION — Team Matchups (/api/team-matchups)
    #
    # F5/full-game run history + VMR gate are unchanged from the retired
    # script. The one deliberate change (per your review) is `mu`: instead
    # of the raw last-10-game mean, it's shrink_rate()'d toward
    # calculate_team_xruns_v2's output for this exact matchup — the same
    # number the Moneylines tab's Pythagenpat win prob already uses — so
    # both tabs agree on "expected runs" without erasing the empirical
    # variance signal the NB2 gate depends on. See math_engine.
    # calibrate_count_market() for exactly what is/isn't touched.
    # ------------------------------------------------------------------

    def build_team_matchups(self, target_date: str, force: bool = False) -> list[dict]:
        return self._cached_list_build(
            self._team_matchups_cache, self._team_matchups_inflight, target_date, force,
            self._build_team_matchups_uncached, "team-matchups",
        )

    def _build_team_matchups_uncached(self, target_date: str) -> list[dict]:
        games = self.mlb.schedule(target_date)
        jobs: list[tuple] = []
        seen_team_ids: set[int] = set()
        for game in games:
            teams = game.get("teams", {})
            venue_name = game.get("venue", {}).get("name", "")
            venue_id = game.get("venue", {}).get("id")
            park_factor = STADIUM_INDICES.get(venue_name, {"base_park_factor": 1.00})["base_park_factor"]
            game_date_utc = game.get("gameDate")
            game_pk = game.get("gamePk")
            for side in ("away", "home"):
                team_node = teams.get(side, {})
                opp_node = teams.get("home" if side == "away" else "away", {})
                team_id = int(team_node.get("team", {}).get("id", 0))
                team_name = team_node.get("team", {}).get("name", "")
                opponent_id = int(opp_node.get("team", {}).get("id", 0))
                opponent_name = opp_node.get("team", {}).get("name", "")
                # FIX (2026-09-10, diagnostic pass): see _resolve_probable_pitcher
                # — a TBD/not-yet-announced opposing starter used to leave this
                # card's opponent-pitcher context empty even when the boxscore
                # already had a real name to offer.
                opp_pitcher_id, _opp_pitcher_name = self._resolve_probable_pitcher(opp_node, game_pk, opponent_id)
                is_home = side == "home"
                if team_id and team_id not in seen_team_ids:
                    seen_team_ids.add(team_id)
                    jobs.append((
                        team_id, team_name, opponent_id, opponent_name, is_home,
                        opp_pitcher_id, park_factor, venue_name, venue_id, game_date_utc, target_date,
                        game_pk,
                    ))

        results: list[dict] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
                futures = {executor.submit(self._build_team_matchup_entry, *job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        entry = future.result()
                    except Exception:
                        job = futures[future]
                        logger.exception(
                            "Failed to build team-matchup entry for team_id=%s (%s) on %s — skipping.",
                            job[0], job[1], target_date,
                        )
                        continue
                    if entry:
                        results.append(entry)

        return results

    def _build_team_matchup_entry(
        self,
        team_id: int,
        team_name: str,
        opponent_id: int,
        opponent_name: str,
        is_home: bool,
        opp_pitcher_id: int | None,
        park_factor: float,
        venue_name: str,
        venue_id: int | None,
        game_date_utc: str | None,
        target_date: str,
        game_pk: int | None = None,
    ) -> dict | None:
        games = self.mlb.team_recent_hitting_log(team_id, limit=10)
        if not games:
            return None

        with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
            linescores = list(pool.map(lambda g: (g, self.mlb.linescore(g.get("game", {}).get("gamePk"))), games))

        history_f5, history_full = [], []
        labels, match_details = [], []
        for g, ls_data in linescores:
            is_home_g = g.get("isHome", True)
            side = "home" if is_home_g else "away"
            innings = ls_data.get("innings", [])
            f5_runs = sum(inning.get(side, {}).get("runs", 0) for inning in innings[:5])
            full_runs = int(g.get("stat", {}).get("runs", 0) or 0)
            history_f5.append(f5_runs)
            history_full.append(full_runs)
            labels.append(_format_date_label(g.get("date", ""), is_home_g, (g.get("opponent") or {}).get("name", "OPP")))
            score_prefix = "W" if g.get("isWin", False) else "L"
            match_details.append({"score": score_prefix, "text": f"F5 Runs: {f5_runs} | Full Game Runs: {full_runs}"})

        # --- shrinkage anchor (see class-level note above) ---
        team_hitting = self.mlb.team_stats(team_id, "hitting")
        team_obp = safe_float(team_hitting.get("obp"), LEAGUE_AVG_OBP)
        team_slg = safe_float(team_hitting.get("slg"), LEAGUE_AVG_SLG)
        opp_pitcher = self.mlb.pitcher_profile(opp_pitcher_id) if opp_pitcher_id else self.mlb.default_pitcher()
        opp_pitcher_era = opp_pitcher.get("era", LEAGUE_AVG_ERA)
        opp_team_pitching = self.mlb.team_stats(opponent_id, "pitching") if opponent_id else {}
        opp_bullpen_era = safe_float(opp_team_pitching.get("era"), LEAGUE_AVG_ERA)

        # TASK 1 & 2 (2026-08-30) — same real weather_mult / bullpen_fatigue_
        # mult as _build_game's Monte Carlo path (see the caches + notes
        # there). bullpen_fatigue_mult here tracks `opponent_id` specifically
        # because opp_bullpen_era above is already the OPPONENT's bullpen —
        # the one this team's offense is actually facing today.
        weather_mult = self._cached_weather_context(venue_id, game_date_utc).mult
        opp_bullpen_fatigue_mult = self._cached_bullpen_fatigue_mult(opponent_id, target_date)

        # TASK (2026-08-31) — same team-strength signal upgrade as
        # _build_game's Moneylines path (see the notes there): the
        # shrinkage anchor this F5/full-game market is built from now also
        # draws on opp_pitcher's proj_er/k_rate/recent-form/arsenal whiff%
        # and this team's own today's-lineup strength, instead of just
        # season ERA vs season team OBP/SLG.
        recent_form = self._cached_starter_recent_form(
            opp_pitcher_id, opp_pitcher_era, opp_pitcher.get("k_rate", LEAGUE_AVG_K_RATE),
        )
        arsenal_whiff = self._cached_starter_arsenal_whiff(opp_pitcher_id)
        lineup_ops, lineup_weight, _confirmed = self._today_lineup_signal(
            game_pk, team_id, opp_pitcher_id, opp_pitcher.get("hand", "R"), target_date,
        )

        matchup_mult = _matchup_multiplier(
            team_obp, team_slg, opp_pitcher_era,
            opp_proj_er=opp_pitcher.get("proj_er"), opp_k_rate=opp_pitcher.get("k_rate"),
            opp_recent_blended_era=recent_form["blendedEra"], opp_recent_blended_k_rate=recent_form["blendedKRate"],
            opp_arsenal_whiff_pct=arsenal_whiff,
            today_lineup_ops=lineup_ops, lineup_confidence_weight=lineup_weight,
        )
        full_game_anchor = calculate_team_xruns_v2(
            matchup_mult=matchup_mult, park_factor=park_factor, weather_mult=weather_mult,
            bullpen_era=opp_bullpen_era, bullpen_fatigue_mult=opp_bullpen_fatigue_mult,
        )
        # F5 anchor = full_game_xruns * (5/9) — uniform per-inning run
        # distribution, same assumption already made throughout this
        # pipeline; no new scope.
        f5_anchor = full_game_anchor * (5.0 / 9.0)

        today_matchup = _matchup_label(opponent_name, is_home)
        return {
            "id": f"t_{team_id}",
            "name": team_name,
            "sub": f"Today: {today_matchup}",
            "opponent": opponent_name,
            "isHome": is_home,
            "matchup": today_matchup,
            "markets": {
                "f5_runs": build_count_market(
                    "First 5 Innings (F5) Runs", history_f5, DEFAULT_PROP_LINES["f5_runs"],
                    anchor_mu=f5_anchor, anchor_k=K_TEAM_RUNS, pad=1,
                ),
                "total_runs": build_count_market(
                    "Total Runs (Full Game)", history_full, DEFAULT_PROP_LINES["total_runs"],
                    anchor_mu=full_game_anchor, anchor_k=K_TEAM_RUNS,
                ),
            },
            "labels": labels,
            "matchDetails": match_details,
            "startTimeUTC": game_date_utc,
            "venue": venue_name,
            "teamId": team_id,
            "opponentId": opponent_id,
        }

    # ------------------------------------------------------------------
    # CONSOLIDATION — Matchup Analyzer (/api/matchup-analyzer)
    # ------------------------------------------------------------------

    def build_matchup_analyzer(self, target_date: str, force: bool = False) -> list[dict]:
        return self._cached_list_build(
            self._matchup_analyzer_cache, self._matchup_analyzer_inflight, target_date, force,
            self._build_matchup_analyzer_uncached, "matchup-analyzer",
        )

    def _build_matchup_analyzer_uncached(self, target_date: str) -> list[dict]:
        games = self.mlb.schedule(target_date)
        jobs: list[tuple] = []
        seen_pitcher_ids: set[int] = set()
        for game in games:
            teams = game.get("teams", {})
            game_pk = game.get("gamePk")
            for side in ("away", "home"):
                team_node = teams.get(side, {})
                opp_node = teams.get("home" if side == "away" else "away", {})
                team_id = team_node.get("team", {}).get("id")
                team_name = team_node.get("team", {}).get("name", "")
                opponent_id = int(opp_node.get("team", {}).get("id", 0))
                opponent_name = opp_node.get("team", {}).get("name", "")
                is_home = side == "home"
                # FIX (2026-09-10, diagnostic pass) — see _resolve_probable_pitcher.
                pid, pname = self._resolve_probable_pitcher(team_node, game_pk, team_id)
                if pid and pid not in seen_pitcher_ids:
                    seen_pitcher_ids.add(pid)
                    jobs.append((pid, pname, team_name, opponent_id, opponent_name, is_home, game_pk, target_date))

        results: list[dict] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
                futures = {executor.submit(self._build_matchup_analyzer_entry, *job): job for job in jobs}
                for future in as_completed(futures):
                    try:
                        entry = future.result()
                    except Exception:
                        job = futures[future]
                        logger.exception(
                            "Failed to build matchup-analyzer entry for pitcher_id=%s (%s) on %s — skipping.",
                            job[0], job[1], target_date,
                        )
                        continue
                    if entry:
                        results.append(entry)

        return results

    def _build_matchup_analyzer_entry(
        self,
        pitcher_id: int,
        pitcher_name: str,
        team_name: str,
        opponent_team_id: int,
        opponent_name: str,
        is_home: bool,
        game_pk: int,
        target_date: str,
    ) -> dict | None:
        pitch_mix = self.mlb.pitch_arsenal(pitcher_id)

        recent_pks = self.mlb.pitcher_recent_game_pks(pitcher_id)
        lethality = self.mlb.pitcher_pitch_lethality(pitcher_id, recent_pks) if recent_pks else {}
        for p in pitch_mix:
            stats = lethality.get(p.get("code", "UN"))
            p["whiffPct"] = stats["whiffPct"] if stats else None
            p["chasePct"] = stats["chasePct"] if stats else None
            p["putAwayPct"] = stats["putAwayPct"] if stats else None
            p["hardHitPct"] = stats["hardHitPct"] if stats else None
            p["lethalitySample"] = stats["sampleSize"] if stats else 0

        lineup, confirmed = self.mlb.probable_lineup(game_pk, opponent_team_id, target_date)
        if not lineup:
            return None

        with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
            batters = list(pool.map(lambda b: self._batter_matchup_job(b, pitcher_id), lineup))

        for idx, b in enumerate(batters):
            b["edgeTier"] = _edge_tier(b["blendedOps"])
            b["order"] = idx + 1

        ranked = sorted(batters, key=lambda b: b["blendedOps"], reverse=True)
        lineup_edge_ops = round(sum(b["blendedOps"] for b in batters) / len(batters), 3) if batters else 0.0
        total_h2h_pa = sum(b["h2h"]["pa"] for b in batters)

        season_pitching = self.mlb._first_stat(self.mlb.person_stats(pitcher_id, group="pitching", stats="season"))
        era_raw, whip_raw = season_pitching.get("era"), season_pitching.get("whip")
        pitcher_era = round(safe_float(era_raw), 2) if era_raw not in (None, "-", "", "-.--") else None
        pitcher_whip = round(safe_float(whip_raw), 2) if whip_raw not in (None, "-", "", "-.--") else None

        today_matchup = _matchup_label(opponent_name, is_home)
        pitcher_hand = self.mlb.person(pitcher_id).get("pitchHand", {}).get("code", "R")
        return {
            "id": f"m_{pitcher_id}",
            "pitcherId": str(pitcher_id),
            "pitcherName": pitcher_name,
            "team": team_name,
            "opponent": opponent_name,
            "isHome": is_home,
            "matchup": today_matchup,
            "sub": f"{team_name} &middot; {today_matchup}",
            "lineupConfirmed": confirmed,
            "throws": pitcher_hand,
            "pitchMix": pitch_mix,
            "batters": ranked,
            "lineupEdgeOps": lineup_edge_ops,
            "lineupEdgeTier": _edge_tier(lineup_edge_ops),
            "totalH2hPa": total_h2h_pa,
            "pitcherEra": pitcher_era,
            "pitcherWhip": pitcher_whip,
            "pitcherArsenalWhiffPct": _pitcher_arsenal_whiff_pct(pitch_mix),
        }

    def _batter_season_slash(self, stat: dict) -> dict:
        avg = safe_float(stat.get("avg"), LEAGUE_AVG_BA)
        slg = safe_float(stat.get("slg"), LEAGUE_AVG_SLG)
        ops = safe_float(stat.get("ops"), LEAGUE_AVG_OPS_FALLBACK)
        pa = safe_float(stat.get("plateAppearances"), 0.0)
        so = safe_float(stat.get("strikeOuts"), 0.0)
        k_pct = round((so / pa * 100.0), 1) if pa > 0 else 0.0
        return {
            "avg": avg if avg > 0 else LEAGUE_AVG_BA,
            "slg": slg if slg > 0 else LEAGUE_AVG_SLG,
            "ops": ops if ops > 0 else LEAGUE_AVG_OPS_FALLBACK,
            "pa": pa,
            "kPct": k_pct,
            "whiff": round((k_pct * 1.05) + 2.0, 1) if pa > 0 else 0.0,
            "chase": round((k_pct * 0.95) + 4.0, 1) if pa > 0 else 0.0,
            "csw": round((k_pct * 0.85) + 8.0, 1) if pa > 0 else 0.0,
        }

    def _batter_matchup_job(self, batter: dict, pitcher_id: int) -> dict:
        h2h = self.mlb.batter_vs_pitcher_slash(batter["id"], pitcher_id)
        season_slash = self._batter_season_slash(self.mlb.player_season_hitting(batter["id"]))
        # NEW (2026-09-05, diagnostic pass): position (from the lineup
        # resolution — see MLBClient.probable_lineup) and bats (batting
        # handedness, from person()) for the H2H Matchups table, so the
        # tool doesn't require cross-checking a lineup card elsewhere.
        bats = self.mlb.person(batter["id"]).get("batSide", {}).get("code", "R")

        # H2H credibility shrink — same shape as shrink_rate(), reused
        # directly (see math_engine.STABILIZATION_PA_H2H note).
        credibility = h2h["pa"] / (h2h["pa"] + STABILIZATION_PA_H2H) if (h2h["pa"] + STABILIZATION_PA_H2H) else 0.0
        blended_ops = round(shrink_rate(h2h["ops"], h2h["pa"], season_slash["ops"], STABILIZATION_PA_H2H), 3)
        proxy_xba = round(shrink_rate(safe_float(h2h["avg"]), h2h["pa"], season_slash["avg"], STABILIZATION_PA_H2H), 3)
        proxy_xslg = round(shrink_rate(safe_float(h2h["slg"]), h2h["pa"], season_slash["slg"], STABILIZATION_PA_H2H), 3)

        recent_pks = self.mlb.batter_recent_game_pks(batter["id"])
        pitch_vuln = self.mlb.batter_pitch_vulnerability(batter["id"], recent_pks) if recent_pks else {}
        platoon_splits = self.mlb.batter_platoon_slash(batter["id"])
        recent_form_raw = self.mlb.batter_recent_form_slash(batter["id"])
        recent_form = {
            **recent_form_raw,
            "tier": _recent_form_tier(recent_form_raw["ops"], recent_form_raw["pa"]),
            "days": 14,
        }

        return {
            "id": batter["id"],
            "name": batter.get("name", "Unknown"),
            "position": batter.get("position", ""),
            "bats": bats,
            "h2h": h2h,
            "seasonOps": round(season_slash["ops"], 3),
            "seasonPa": season_slash["pa"],
            "seasonKPct": season_slash["kPct"],
            "seasonWhiff": season_slash["whiff"],
            "seasonChase": season_slash["chase"],
            "seasonCsw": season_slash["csw"],
            "pitchVulnerability": pitch_vuln,
            "platoonSplits": platoon_splits,
            "recentForm": recent_form,
            "credibility": round(credibility, 3),
            "blendedOps": blended_ops,
            "proxyXba": proxy_xba,
            "proxyXslg": proxy_xslg,
        }

    # ------------------------------------------------------------------
    # Matchup Verifier
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Player Card scouting tool (replaces manual Statcast calibration)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Season H2H (Team Matchups) / Pitcher-vs-Team (Pitcher Props) tools
    # — replace the Kelly Stake card on both tabs.
    # ------------------------------------------------------------------

    def get_team_h2h(self, team_id: int, opponent_id: int) -> dict:
        team_info = self.mlb.team_info(team_id)
        opp_info = self.mlb.team_info(opponent_id)
        season = settings.season
        games = self.mlb.team_h2h_games(team_id, opponent_id, season)

        def _line(subset: list[dict]) -> dict:
            wins = sum(1 for g in subset if g["win"])
            return {
                "games": len(subset),
                "wins": wins,
                "losses": len(subset) - wins,
                "runsFor": sum(g["runsFor"] for g in subset),
                "runsAgainst": sum(g["runsAgainst"] for g in subset),
            }

        return {
            "teamId": team_id,
            "teamName": team_info.get("name", ""),
            "opponentId": opponent_id,
            "opponentName": opp_info.get("name", ""),
            "season": season,
            "overall": _line(games),
            "home": _line([g for g in games if g["isHome"]]),
            "away": _line([g for g in games if not g["isHome"]]),
            "day": _line([g for g in games if g["dayNight"] == "day"]),
            "night": _line([g for g in games if g["dayNight"] == "night"]),
        }

    def get_pitcher_vs_team(self, pitcher_id: int, team_id: int) -> dict:
        pitcher_person = self.mlb.person(pitcher_id)
        team_info = self.mlb.team_info(team_id)
        season = settings.season
        games = self.mlb.pitcher_vs_team_games(pitcher_id, team_id)

        def _line(subset: list[dict]) -> dict:
            ip_outs = 0
            er = runs = hits = walks = strikeouts = 0
            for g in subset:
                s = g["stat"]
                ip_str = str(s.get("inningsPitched", "0.0") or "0.0")
                whole, _, frac = ip_str.partition(".")
                ip_outs += int(whole or 0) * 3 + int(frac or 0)
                er += int(s.get("earnedRuns", 0) or 0)
                runs += int(s.get("runs", 0) or 0)
                hits += int(s.get("hits", 0) or 0)
                walks += int(s.get("baseOnBalls", 0) or 0)
                strikeouts += int(s.get("strikeOuts", 0) or 0)
            ip_val = ip_outs / 3.0
            batters_faced_ab = hits + (ip_outs)  # rough AB proxy: outs recorded + hits allowed
            return {
                "games": len(subset),
                "inningsPitched": round(ip_val, 1),
                "er": er,
                "runs": runs,
                "hits": hits,
                "walks": walks,
                "strikeouts": strikeouts,
                "era": round((er * 9.0 / ip_val), 2) if ip_val > 0 else None,
                "whip": round((walks + hits) / ip_val, 2) if ip_val > 0 else None,
                "baa": round(hits / batters_faced_ab, 3) if batters_faced_ab > 0 else None,
            }

        return {
            "pitcherId": pitcher_id,
            "pitcherName": pitcher_person.get("fullName", "Unknown"),
            "teamId": team_id,
            "teamName": team_info.get("name", ""),
            "season": season,
            "overall": _line(games),
            "home": _line([g for g in games if g["isHome"]]),
            "away": _line([g for g in games if not g["isHome"]]),
            "day": _line([g for g in games if g["dayNight"] == "day"]),
            "night": _line([g for g in games if g["dayNight"] == "night"]),
        }

    def get_player_scouting(self, batter_id: int, pitcher_id: int | None) -> dict:
        batter_person = self.mlb.person(batter_id)
        platoon = self.mlb.batter_platoon_slash(batter_id)
        # NEW (2026-09-08, diagnostic pass): single enriched game-log fetch,
        # shared between the recent hand-split and the Last 7 Games log —
        # avoids resolving the same 7 games' boxscores twice.
        recent_log = self.mlb.recent_game_log_enriched(batter_id, games=7)
        recent_hand = self.mlb.recent_hand_splits(batter_id, games=7, log=recent_log)
        last7 = self.mlb.recent_game_log_summary(batter_id, games=7, log=recent_log)

        def _split_line(side: dict) -> dict:
            return {"avg": side.get("avg", ".000"), "ops": side.get("ops", 0.0), "pa": side.get("pa", 0), "hr": side.get("hr", 0)}

        result = {
            "playerId": batter_id,
            "name": batter_person.get("fullName", "Unknown"),
            "seasonVsLHP": _split_line(platoon.get("vsLHP", {})),
            "seasonVsRHP": _split_line(platoon.get("vsRHP", {})),
            "recentVsLHP": recent_hand.get("vsLHP", {"ab": 0, "h": 0, "avg": ".000"}),
            "recentVsRHP": recent_hand.get("vsRHP", {"ab": 0, "h": 0, "avg": ".000"}),
            "recentHandGamesCovered": recent_hand.get("gamesCovered", 0),
            "last7": last7,
            "pitcherId": None,
            "pitcherName": None,
            "pitcherHand": None,
            "h2h": None,
        }

        if pitcher_id:
            pitcher_person = self.mlb.person(pitcher_id)
            h2h = self.mlb.batter_vs_pitcher_slash(batter_id, pitcher_id)
            result["pitcherId"] = pitcher_id
            result["pitcherName"] = pitcher_person.get("fullName", "Unknown")
            result["pitcherHand"] = pitcher_person.get("pitchHand", {}).get("code", "R")
            result["h2h"] = h2h

        return result

    def get_matchup(self, batter_id: int, pitcher_id: int) -> dict:
        batter = self.mlb.person(batter_id)
        pitcher = self.mlb.person(pitcher_id)

        batter_name = batter.get("fullName", "Unknown")
        pitcher_name = pitcher.get("fullName", "Unknown")
        batter_hand = batter.get("batSide", {}).get("code", "R")
        pitcher_hand = pitcher.get("pitchHand", {}).get("code", "R")

        batter_season = self.mlb.player_season_hitting(batter_id)
        batter_avg = safe_float(batter_season.get("avg"), LEAGUE_AVG_BA)
        b_ab = safe_float(batter_season.get("atBats"), 0)
        b_so = safe_float(batter_season.get("strikeOuts"), 0)
        batter_k_rate = (b_so / b_ab * 100.0) if b_ab > 0 else 22.0

        pitcher_profile = self.mlb.pitcher_profile(pitcher_id)
        pitcher_baa = pitcher_profile.get("avg_vs_left" if batter_hand == "L" else "avg_vs_right", 0.245)
        pitcher_k_rate = pitcher_profile.get("k_rate", 0.22) * 100.0

        splits = pitcher_profile.get("splits", {})
        vs_l = splits.get("vsL", {"baa": 0.250, "k": 20.0, "bb": 8.0, "hr": 3.0})
        vs_r = splits.get("vsR", {"baa": 0.250, "k": 22.0, "bb": 8.0, "hr": 3.0})

        h2h_splits = self.mlb.hitter_vs_pitcher_history(batter_id, pitcher_id)
        total_ab = total_hits = total_hr = total_so = total_bb = total_hbp = total_sf = 0
        for split in h2h_splits:
            stat = split.get("stat", {})
            total_ab += int(stat.get("atBats", 0) or 0)
            total_hits += int(stat.get("hits", 0) or 0)
            total_hr += int(stat.get("homeRuns", 0) or 0)
            total_so += int(stat.get("strikeOuts", 0) or 0)
            total_bb += int(stat.get("baseOnBalls", 0) or 0) + int(stat.get("intentionalWalks", 0) or 0)
            total_hbp += int(stat.get("hitByPitch", 0) or 0)
            total_sf += int(stat.get("sacrificeFlies", 0) or 0)

        has_h2h = total_ab > 0
        h2h_avg = (total_hits / total_ab) if has_h2h else 0.0
        h2h_denom = total_ab + total_bb + total_hbp + total_sf
        h2h_obp = ((total_hits + total_bb + total_hbp) / h2h_denom) if h2h_denom > 0 else h2h_avg

        split_advantage = (batter_avg - 0.250) * 400 + (pitcher_baa - 0.240) * 400
        # FIX (2026-09-04, diagnostic pass — "Matchup Verifier ace bias"):
        # k_penalty used to weight the PITCHER's K-rate deviation at 4.0x
        # and the BATTER's own K-rate deviation at only 2.0x — a flat,
        # unjustified 2x thumb on the scale toward "elite/high-K pitcher
        # wins," on top of a pitcher already getting full, uncapped credit
        # for their platoon BAA in split_advantage above. Rebalanced to an
        # equal 2.0x/2.0x weighting — there's no principled reason a
        # pitcher's whiff-generation should count double a batter's own
        # contact skill in a head-to-head read.
        k_penalty = (pitcher_k_rate - 22.0) * 2.0 + (batter_k_rate - 22.0) * 2.0
        raw_advantage = split_advantage - k_penalty

        # FIX (2026-09-04): split_advantage above is a BLUNT signal — it's
        # this pitcher's overall season BAA against the batter's handedness,
        # nothing about how THIS batter's specific pitch-recognition/power
        # profile lines up against THIS pitcher's actual arsenal. Ace-tier
        # pitchers naturally post very low platoon BAA (.150-.200), which
        # swung split_advantage hard toward "pitcher" almost automatically —
        # exactly the "everyone's red against Skubal/Misiorowski regardless
        # of fit" pattern reported after extensive real-world observation.
        # Two changes: (1) cap split_advantage's swing so no single
        # generic season-average signal can dominate the score outright —
        # forces H2H and pitch-arsenal fit (added next) to carry real
        # weight instead of being drowned out; (2) fold in the SAME
        # H2H-blended, arsenal-vulnerability-aware OPS signal that already
        # powers the Matchup Analyzer's edgeTier (_batter_matchup_job) —
        # so a batter who profiles well against this specific pitcher's
        # actual pitches now gets meaningful credit here too, not just in
        # a separate tool that this Verifier never talked to.
        raw_advantage = clamp(raw_advantage, -35.0, 35.0)

        try:
            fit_job = self._batter_matchup_job({"id": batter_id, "name": batter_name}, pitcher_id)
            arsenal_fit_ops = fit_job["blendedOps"]
            arsenal_fit_component = clamp((arsenal_fit_ops - LEAGUE_AVG_OPS_FALLBACK) * 70.0, -25.0, 25.0)
            raw_advantage += arsenal_fit_component
        except Exception:
            logger.warning(
                "Matchup Verifier: pitch-arsenal fit lookup failed for batter=%s pitcher=%s — "
                "falling back to the platoon/K-rate signal only.",
                batter_id, pitcher_id, exc_info=True,
            )

        if has_h2h and total_ab >= 5:
            h2h_weight = min(0.5, total_ab * 0.05)
            h2h_diff = (h2h_avg - 0.250) * 500
            raw_advantage = ((1 - h2h_weight) * raw_advantage) + (h2h_weight * h2h_diff)

        advantage_score = round(clamp(raw_advantage, -100.0, 100.0), 1)

        if advantage_score >= 5.0:
            verdict, verdict_label = "batter", f"Batter advantage (+{advantage_score:.1f}%)"
        elif advantage_score <= -5.0:
            verdict, verdict_label = "pitcher", f"Pitcher advantage ({advantage_score:.1f}%)"
        else:
            verdict, verdict_label = "neutral", "Neutral zone / marginal variance"

        return {
            "batterId": batter_id, "batterName": batter_name, "batterHand": batter_hand,
            "batterAvg": round(batter_avg, 3), "batterKRate": round(batter_k_rate, 1),
            "pitcherId": pitcher_id, "pitcherName": pitcher_name, "pitcherHand": pitcher_hand,
            "pitcherBaa": round(pitcher_baa, 3), "pitcherKRate": round(pitcher_k_rate, 1),
            "platoonVsLeft": {
                "baa": round(vs_l.get("baa", 0.25), 3), "kPct": round(vs_l.get("k", 20.0), 1),
                "bbPct": round(vs_l.get("bb", 8.0), 1), "hrPct": round(vs_l.get("hr", 3.0), 1),
            },
            "platoonVsRight": {
                "baa": round(vs_r.get("baa", 0.25), 3), "kPct": round(vs_r.get("k", 22.0), 1),
                "bbPct": round(vs_r.get("bb", 8.0), 1), "hrPct": round(vs_r.get("hr", 3.0), 1),
            },
            "headToHead": {
                "hasData": has_h2h, "atBats": total_ab, "hits": total_hits, "homeRuns": total_hr,
                "strikeouts": total_so, "avg": round(h2h_avg, 3), "obp": round(h2h_obp, 3),
            },
            "advantageScore": advantage_score,
            "verdict": verdict,
            "verdictLabel": verdict_label,
        }

    # ------------------------------------------------------------------

    def get_slate(self, target_date: str) -> dict:
        return self.build_slate(target_date, force=False)

    def get_indexed_player(self, mlb_id: int) -> dict | None:
        with self._lock:
            return self._player_index.get(mlb_id)

    def get_pitcher(self, pitcher_id: int) -> dict | None:
        with self._lock:
            cached = self._pitcher_index.get(pitcher_id)
        if cached:
            return cached
        profile = self.mlb.pitcher_profile(pitcher_id)
        if not profile.get("id"):
            return None
        proj_k = profile.get("proj_k", 5.0)
        proj_er = profile.get("proj_er", 2.5)
        result = {
            "id": profile.get("id"), "name": profile.get("name", "Unknown"), "team": "MLB",
            "throws": profile.get("hand", "R"), "projK": proj_k, "projER": proj_er,
            "era": profile.get("era", LEAGUE_AVG_ERA), "whip": profile.get("whip", 1.32),
            "statsSource": profile.get("stats_source", "league-fallback"),
            "splits": profile.get("splits", {}),
            "propLadder": {
                "k": build_pitcher_ladder(proj_k, _ladder_lines(proj_k)),
                "bb": build_pitcher_ladder(proj_k * 0.35, _ladder_lines(proj_k * 0.35)),
                "er": build_pitcher_ladder(proj_er, _ladder_lines(proj_er)),
            },
        }
        with self._lock:
            self._pitcher_index[pitcher_id] = result
        return result

    def sync_results(self) -> dict:
        pending = self.repository.unresolved()
        resolved_count = 0
        failed_count = 0
        for row in pending:
            try:
                boxscore = self.mlb._get(f"/game/{row['game_pk']}/boxscore")
                actual = self._extract_actual_stat(boxscore, row["player_id"], row["prop_type"])
                if actual is None:
                    # Not an error — just means the game hasn't finished yet
                    # (or this prop_type/player never appeared in the box
                    # score), so it stays unresolved for the next sync pass.
                    logger.debug(
                        "sync_results: no actual stat yet for game_pk=%s player_id=%s prop_type=%s — "
                        "leaving unresolved.",
                        row.get("game_pk"), row.get("player_id"), row.get("prop_type"),
                    )
                    continue
                self.repository.set_actual(row["game_pk"], row["player_id"], row["prop_type"], actual)
                resolved_count += 1
            except Exception:
                failed_count += 1
                logger.exception(
                    "sync_results failed to resolve game_pk=%s player_id=%s prop_type=%s — will retry on the "
                    "next sync.",
                    row.get("game_pk"), row.get("player_id"), row.get("prop_type"),
                )
                continue
        return {"checked": len(pending), "resolved": resolved_count, "failed": failed_count}

    @staticmethod
    def _extract_actual_stat(boxscore: dict, player_id: int, prop_type: str) -> int | None:
        stat_key = {"hits": "hits", "homeruns": "homeRuns", "rbi": "rbi", "runs": "runs"}.get(prop_type)
        if not stat_key:
            return None
        for side in ("away", "home"):
            players = boxscore.get("teams", {}).get(side, {}).get("players", {})
            player = players.get(f"ID{player_id}")
            if not player:
                continue
            batting = player.get("stats", {}).get("batting", {})
            if stat_key in batting:
                return int(batting.get(stat_key, 0) or 0)
        return None