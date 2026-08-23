"""
trckr21 — MLB Quant Terminal Data Engine  (v2.4 — Pitch-Code Fix + PutAway% Fix)
====================================================================
Pulls today's slate from the MLB Stats API and produces `projections.json`
in the shape the frontend consumes: db.pitchers / db.teams / db.matchups.

CALIBRATION ENGINE  (unchanged from v2.1)
------------------------------------------
Baseline math is the user's own spec — a dynamic Variance-to-Mean Ratio
(VMR) gate that switches between two distributions per market, per player,
per game, based on the actual shape of the last-10-game sample:

    VMR = sigma^2 / mu

    VMR <= 1  ->  Standard Poisson        (no clustering / contagion)
    VMR >  1  ->  Negative Binomial (NB2) (over-dispersed / "contagious")

v2.3 — REAL PITCH-LEVEL LETHALITY (unchanged core approach)
--------------------------------------------------------------------------
The free MLB Stats API's LIVE GAME FEED (v1.1) carries real per-pitch
outcome data used to compute Whiff% / Chase% / PutAway% / Hard-Hit%,
aggregated across a pitcher's last LETHALITY_LOOKBACK_STARTS starts,
per pitch type. No estimated/fit coefficients anywhere in this path.

FIXES IN v2.4
-------------
  1. PITCH-CODE MISMATCH FIX
     The season-summary `pitchArsenal` endpoint and the live-feed
     `playEvents[].details.type` sometimes label the *same* pitch
     differently as free text ("Four-seam FB" vs "Four-Seam Fastball"),
     while both carry the same short MLB pitch `code` ("FF"). v2.3
     joined lethality data onto the arsenal list by matching the free-text
     *description* (case-insensitively), so any wording mismatch silently
     dropped a pitch's lethality stats ("No recent-start pitch data yet").

     v2.4 buckets and joins lethality data by pitch `code` first (the
     stable, short identifier both endpoints emit), and only falls back
     to a normalized-description match when a code is missing from either
     side. PITCH_CODE_NAMES gives a canonical display name per code so
     the UI always shows one consistent pitch name regardless of which
     endpoint's wording happened to come through.

  2. PUTAWAY% BUG FIX (was showing 0.0% for every pitch type)
     v2.3 checked `event["count"]["strikes"] == 2` on the *same* pitch
     event that resulted in the strikeout. But the MLB Stats API's
     per-pitch `count` object reports the count *after* that pitch was
     thrown/resolved — so the deciding, strikeout-ending pitch itself
     never carries `strikes: 2` (a strikeout isn't recorded as "2
     strikes"). That meant every 2-strike **rendered** count came from
     a foul ball or take that didn't end the at-bat, so the numerator
     (`twoStrikeKs`) stayed at 0 while the denominator kept growing —
     hence a permanent 0.0% PutAway rate.

     v2.4 tracks the *entering* strike count (the count the batter had
     BEFORE the current pitch was thrown, i.e. the previous pitch's
     resulting count within the same plate appearance, starting at 0-0).
     A pitch is a genuine "putaway pitch" when the batter enters it
     already at 2 strikes; it counts as converted when that same pitch
     is also the final pitch of the plate appearance and the play's
     result is a strikeout. This matches how PutAway% is defined
     everywhere else in the industry (Baseball Savant included).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

MLB_TZ = ZoneInfo("America/New_York")


def today_mlb_date() -> str:
    return datetime.now(MLB_TZ).strftime("%Y-%m-%d")

# ================= CONFIG =================
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 1.5
MAX_WORKERS = 6

MARKET_VIG_AMERICAN = -110
MARKET_IMPLIED_PROB = 110 / 210
DECIMAL_B = 100 / 110

STABILIZATION_PA = 100
LEAGUE_AVG_OPS_FALLBACK = 0.710
LEAGUE_AVG_AVG_FALLBACK = 0.245
LEAGUE_AVG_SLG_FALLBACK = 0.400

# --- pitch-level lethality knobs ---
PLATE_HALF_WIDTH_FT = 0.708          # 17-inch plate, half-width in feet
HARD_HIT_THRESHOLD_MPH = 95.0
LETHALITY_LOOKBACK_STARTS = 5        # real games aggregated per pitcher
MIN_LETHALITY_SAMPLE = 8             # below this, report None not a noisy ratio

DEBUG_DUMP_RAW = False

# --- NEW v2.4: canonical pitch-code -> display-name map ---------------
# Both the pitchArsenal endpoint and the live-feed pitch `details.type`
# object carry a short `code` field using this same MLB dictionary.
# Joining on `code` instead of free-text description is what fixes the
# "Four-seam FB" vs "Four-Seam Fastball" mismatch.
PITCH_CODE_NAMES = {
    "FF": "Four-Seam Fastball", "FA": "Fastball", "FT": "Two-Seam Fastball",
    "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
    "SV": "Slurve", "CU": "Curveball", "KC": "Knuckle Curve", "CS": "Slow Curve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball", "SC": "Screwball",
    "KN": "Knuckleball", "EP": "Eephus", "PO": "Pitchout", "IN": "Intentional Ball",
    "AB": "Automatic Ball", "UN": "Unknown", "NP": "No Pitch",
}
# Reverse lookup used only as a fallback when a `code` is missing on one
# side — normalizes free-text pitch names ("four-seam fb", "4-seam
# fastball", "fastball (four-seam)"...) to the same code above.
_NAME_TO_CODE = {
    "fourseamfastball": "FF", "fourseamfb": "FF", "4seamfastball": "FF", "fastballfourseam": "FF",
    "fastball": "FA", "twoseamfastball": "FT", "2seamfastball": "FT", "sinkingfastball": "FT",
    "sinker": "SI", "cutter": "FC", "cutfastball": "FC",
    "slider": "SL", "sweeper": "ST", "sweepingslider": "ST", "slurve": "SV",
    "curveball": "CU", "curve": "CU", "knucklecurve": "KC", "slowcurve": "CS",
    "changeup": "CH", "change": "CH", "splitter": "FS", "splitfingeredfastball": "FS",
    "forkball": "FO", "screwball": "SC", "knuckleball": "KN", "eephus": "EP",
    "pitchout": "PO", "intentionalball": "IN", "automaticball": "AB",
    "unknown": "UN", "nopitch": "NP",
}


def _normalize_key(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def resolve_pitch_code(code: Optional[str], description: Optional[str]) -> str:
    """Best-effort resolution to a canonical MLB pitch code.
    Prefers the short `code` field (stable across endpoints); falls back
    to normalizing the free-text description only when code is missing
    or unrecognized."""
    if code:
        c = code.strip().upper()
        if c in PITCH_CODE_NAMES:
            return c
    key = _normalize_key(description or "")
    return _NAME_TO_CODE.get(key, "UN")


def pitch_display_name(code: str, fallback_description: Optional[str] = None) -> str:
    return PITCH_CODE_NAMES.get(code, fallback_description or "Unknown")


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("trckr21")


# ================= HTTP =================
def fetch_json(url: str) -> dict:
    """GET + parse JSON with retries and exponential backoff."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trckr21-quant-engine/2.4"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECS * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {last_err}")


def format_date_label(date_str: str, is_home: bool, opp_name: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = dt.strftime("%m/%d")
        opp_abbr = "".join(word[0] for word in opp_name.split())[:3].upper()
        location = "vs" if is_home else "@"
        return f"{formatted_date} {location} {opp_abbr}"
    except (ValueError, TypeError):
        return date_str or "—"


def matchup_label(opponent_name: str, is_home: bool) -> str:
    return f"{'vs' if is_home else '@'} {opponent_name}"


def safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "-", ".---", "N/A"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


# ================= STATISTICS CORE (unchanged) =================
def sample_mean(values: list[float]) -> float:
    n = len(values)
    return sum(values) / n if n else 0.0


def sample_variance(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sample_mean(values)
    return sum((x - mu) ** 2 for x in values) / (n - 1)


def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_over_prob(lam: float, line: float) -> float:
    target_int = math.ceil(line)
    cumulative = sum(poisson_pmf(lam, i) for i in range(target_int))
    return min(1.0, max(0.0, 1.0 - cumulative))


def nbinom_pmf(k: int, r: float, p: float) -> float:
    if not (0 < p < 1) or r <= 0:
        raise ValueError("invalid negative-binomial parameters")
    log_coef = math.lgamma(r + k) - math.lgamma(r) - math.lgamma(k + 1)
    log_pmf = log_coef + r * math.log(p) + k * math.log(1 - p)
    return math.exp(log_pmf)


def nbinom_params(mu: float, var: float) -> tuple[float, float]:
    if var <= mu:
        raise ValueError("variance must exceed mean for NB2 (over-dispersion required)")
    p = mu / var
    r = (mu ** 2) / (var - mu)
    if not (0 < p < 1) or r <= 0 or math.isinf(r):
        raise ValueError("degenerate negative-binomial parameters")
    return r, p


def nbinom_over_prob(mu: float, var: float, line: float) -> float:
    r, p = nbinom_params(mu, var)
    target_int = math.ceil(line)
    cumulative = sum(nbinom_pmf(i, r, p) for i in range(target_int))
    return min(1.0, max(0.0, 1.0 - cumulative))


def american_odds_from_prob(p: float) -> str:
    if p <= 0 or p >= 1:
        return "N/A"
    if p > 0.5:
        return f"-{round((p / (1 - p)) * 100)}"
    return f"+{round(((1 - p) / p) * 100)}"


def kelly_stake_pct(model_prob: float) -> float:
    edge = model_prob - MARKET_IMPLIED_PROB
    if edge <= 0:
        return 0.0
    q = 1 - model_prob
    f_star = (DECIMAL_B * model_prob - q) / DECIMAL_B
    return round(max(0.0, f_star) * 100, 2)


def confidence_tier(n: int) -> str:
    if n >= 10:
        return "high"
    if n >= 6:
        return "medium"
    return "low"


# ================= MODEL (unchanged) =================
@dataclass
class MarketModel:
    distribution: str
    mu: float
    variance: float
    vmr: float
    overProb: float
    fairOdds: str
    impliedProbMarket: float
    edgePct: float
    kellyPct: float
    confidence: str
    sampleSize: int
    fallback: bool = False


def calibrate_market(history: list[int], line: float) -> MarketModel:
    n = len(history)
    mu = sample_mean(history)
    var = sample_variance(history)
    vmr = (var / mu) if mu > 0 else 0.0

    fallback = False
    if var > mu and mu > 0:
        try:
            over_prob = nbinom_over_prob(mu, var, line)
            distribution = "negative_binomial"
        except ValueError:
            over_prob = poisson_over_prob(mu, line)
            distribution = "poisson"
            fallback = True
    else:
        over_prob = poisson_over_prob(mu, line)
        distribution = "poisson"

    edge_pct = round((over_prob - MARKET_IMPLIED_PROB) * 100, 2)

    return MarketModel(
        distribution=distribution,
        mu=round(mu, 3),
        variance=round(var, 3),
        vmr=round(vmr, 3),
        overProb=round(over_prob, 4),
        fairOdds=american_odds_from_prob(over_prob),
        impliedProbMarket=round(MARKET_IMPLIED_PROB, 4),
        edgePct=edge_pct,
        kellyPct=kelly_stake_pct(over_prob),
        confidence=confidence_tier(n),
        sampleSize=n,
        fallback=fallback,
    )


def build_market(label: str, history: list[int], default_line: float, pad: int = 2) -> dict:
    model = calibrate_market(history, default_line)
    return {
        "label": label,
        "defaultLine": default_line,
        "history": history,
        "max": max(history + [int(default_line) + 3]) + pad,
        "model": asdict(model),
    }


# ================= PITCHER ENGINE (unchanged) =================
def get_pitcher_data(pitcher_id: int, pitcher_name: str, team_name: str,
                      opponent_name: str, is_home: bool) -> Optional[dict]:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={season}"

    try:
        data = fetch_json(url)
        stats = data.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        games = splits[-10:]
        if not games:
            return None

        history_k, history_er, history_bb = [], [], []
        labels, match_details = [], []

        for g in games:
            s = g["stat"]
            history_k.append(int(s.get("strikeOuts", 0)))
            history_er.append(int(s.get("earnedRuns", 0)))
            history_bb.append(int(s.get("baseOnBalls", 0)))

            labels.append(format_date_label(g.get("date", ""), g.get("isHome", True), g.get("opponent", {}).get("name", "OPP")))

            score_prefix = "W" if g.get("isWin", False) else "L"
            match_details.append({
                "score": score_prefix,
                "text": f"IP: {s.get('inningsPitched', '0.0')} | ER: {history_er[-1]} | BB: {history_bb[-1]} | K: {history_k[-1]}",
            })

        today_matchup = matchup_label(opponent_name, is_home)

        return {
            "id": str(pitcher_id),
            "name": pitcher_name,
            "sub": f"{team_name} &middot; P &middot; {today_matchup}",
            "opponent": opponent_name,
            "isHome": is_home,
            "matchup": today_matchup,
            "markets": {
                "strikeouts": build_market("Strikeouts", history_k, DEFAULT_LINES["strikeouts"]),
                "earned_runs": build_market("Earned Runs", history_er, DEFAULT_LINES["earned_runs"], pad=1),
                "walks": build_market("Walks Allowed", history_bb, DEFAULT_LINES["walks"], pad=1),
            },
            "labels": labels,
            "matchDetails": match_details,
        }
    except Exception as e:
        log.warning(f"Pitcher {pitcher_name} ({pitcher_id}) skipped: {e}")
        return None


DEFAULT_LINES = {
    "strikeouts": 4.5,
    "earned_runs": 1.5,
    "walks": 2.5,
    "f5_runs": 1.5,
    "total_runs": 4.5,
}


# ================= TEAM ENGINE (unchanged) =================
def get_team_data(team_id: int, team_name: str, opponent_name: str, is_home: bool) -> Optional[dict]:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=gameLog&group=hitting&season={season}"

    try:
        data = fetch_json(url)
        games = data.get("stats", [{}])[0].get("splits", [])[-10:]
        if not games:
            return None

        history_f5, history_full = [], []
        labels, match_details = [], []

        def fetch_linescore(g):
            game_pk = g["game"]["gamePk"]
            return g, fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = list(pool.map(fetch_linescore, games))

        for g, ls_data in results:
            is_home_g = g.get("isHome", True)
            side = "home" if is_home_g else "away"
            innings = ls_data.get("innings", [])
            f5_runs = sum(inning.get(side, {}).get("runs", 0) for inning in innings[:5])
            full_runs = int(g["stat"].get("runs", 0))

            history_f5.append(f5_runs)
            history_full.append(full_runs)

            labels.append(format_date_label(g.get("date", ""), is_home_g, g.get("opponent", {}).get("name", "OPP")))
            score_prefix = "W" if g.get("isWin", False) else "L"
            match_details.append({
                "score": score_prefix,
                "text": f"F5 Runs: {f5_runs} | Full Game Runs: {full_runs}",
            })

        today_matchup = matchup_label(opponent_name, is_home)

        return {
            "id": f"t_{team_id}",
            "name": team_name,
            "sub": f"Today: {today_matchup}",
            "opponent": opponent_name,
            "isHome": is_home,
            "matchup": today_matchup,
            "markets": {
                "f5_runs": build_market("First 5 Innings (F5) Runs", history_f5, DEFAULT_LINES["f5_runs"], pad=1),
                "total_runs": build_market("Total Runs (Full Game)", history_full, DEFAULT_LINES["total_runs"]),
            },
            "labels": labels,
            "matchDetails": match_details,
        }
    except Exception as e:
        log.warning(f"Team {team_name} ({team_id}) skipped: {e}")
        return None


# ================= MATCHUP ANALYZER ENGINE =================
_raw_dump_done = False


def get_pitch_arsenal(pitcher_id: int) -> list[dict]:
    """Pitch type usage% + avg velocity, free-tier MLB Stats API only.
    v2.4: now also captures the short pitch `code` (e.g. 'FF') alongside
    the display name, so it can be joined to lethality data reliably."""
    global _raw_dump_done
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=pitchArsenal&group=pitching&season={season}"
    try:
        data = fetch_json(url)

        if DEBUG_DUMP_RAW and not _raw_dump_done:
            import sys
            print(json.dumps(data, indent=2)[:3000], file=sys.stderr)
            _raw_dump_done = True

        stats = data.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []

        pitches = []
        for s in splits:
            stat = s.get("stat", s)
            pt = stat.get("pitchType") or stat.get("type") or {}
            raw_code = pt.get("code") or stat.get("code")
            description = pt.get("description") or pt.get("displayName") or stat.get("pitchName") or "Unknown"
            code = resolve_pitch_code(raw_code, description)
            name = pitch_display_name(code, description)

            usage_raw = safe_float(stat.get("percentage", stat.get("usage", 0)))
            usage_pct = round(usage_raw * 100, 1) if usage_raw <= 1 else round(usage_raw, 1)

            velo = safe_float(stat.get("averageSpeed", stat.get("avgSpeed", 0)))

            if name == "Unknown" and usage_pct == 0:
                continue
            pitches.append({"type": name, "code": code, "usagePct": usage_pct, "avgVelo": round(velo, 1)})

        pitches.sort(key=lambda p: p["usagePct"], reverse=True)
        return pitches
    except Exception as e:
        log.warning(f"Pitch arsenal fetch failed for pitcher {pitcher_id}: {e}")
        return []

# --- real per-pitch lethality from the live feed ---

def get_live_feed(game_pk: int) -> dict:
    """The v1 endpoints elsewhere in this file are season/summary stats.
    Real pitch-by-pitch data — coordinates, descriptions, hit data — lives
    on the v1.1 live feed, a genuinely different endpoint family."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    return fetch_json(url)


def get_pitcher_recent_game_pks(pitcher_id: int, limit: int = LETHALITY_LOOKBACK_STARTS) -> list[int]:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={season}"
    try:
        data = fetch_json(url)
        stats = data.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        pks = [s.get("game", {}).get("gamePk") for s in splits if s.get("game", {}).get("gamePk")]
        return pks[-limit:]
    except Exception as e:
        log.warning(f"Recent-start lookup failed for pitcher {pitcher_id}: {e}")
        return []


_SWING_DESCRIPTIONS = {
    "Swinging Strike", "Swinging Strike (Blocked)", "Foul", "Foul Tip", "Foul Bunt", "Missed Bunt",
    "In play, out(s)", "In play, no out", "In play, run(s)",
}
_WHIFF_DESCRIPTIONS = {"Swinging Strike", "Swinging Strike (Blocked)", "Missed Bunt"}
_IN_PLAY_DESCRIPTIONS = {"In play, out(s)", "In play, no out", "In play, run(s)"}
_STRIKEOUT_EVENT_TYPES = {"strikeout", "strikeout_double_play", "strikeout_triple_play"}


def get_pitcher_pitch_lethality(pitcher_id: int, game_pks: list[int]) -> dict[str, dict]:
    """Real, counted Whiff%/Chase%/PutAway%/Hard-Hit% per pitch type
    (bucketed by pitch CODE — see v2.4 notes at top of file), aggregated
    over `game_pks` (this pitcher's actual recent starts). Every number
    here is real_count / real_count — no fitted constants.

    v2.4 PutAway% fix: `count` on a pitch event reflects the count AFTER
    that pitch resolves, so a strikeout-ending pitch never itself carries
    strikes==2. We instead track the ENTERING strike count for each pitch
    (the previous pitch's resulting count within the same PA, 0 for the
    first pitch) to correctly identify genuine 2-strike ("putaway")
    pitches, then check whether that specific pitch ended the PA in a
    strikeout.
    """
    buckets: dict[str, dict] = {}
    display_names: dict[str, str] = {}

    def _bucket(code: str) -> dict:
        return buckets.setdefault(code, {
            "pitches": 0, "swings": 0, "whiffs": 0,
            "outOfZonePitches": 0, "outOfZoneSwings": 0,
            "twoStrikePitches": 0, "twoStrikeKs": 0,
            "battedBalls": 0, "hardHitBalls": 0,
        })

    for game_pk in game_pks:
        try:
            feed = get_live_feed(game_pk)
        except Exception as e:
            log.warning(f"Live feed fetch failed for game {game_pk}: {e}")
            continue

        plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
        for play in plays:
            matchup = play.get("matchup", {})
            if matchup.get("pitcher", {}).get("id") != pitcher_id:
                continue

            events = play.get("playEvents", [])
            pitch_events = [e for e in events if e.get("isPitch")]
            result_event_type = play.get("result", {}).get("eventType", "")

            entering_strikes = 0  # count BEFORE the current pitch, within this PA

            for i, event in enumerate(pitch_events):
                details = event.get("details", {})
                type_obj = details.get("type", {}) or {}
                raw_code = type_obj.get("code")
                description = type_obj.get("description")
                code = resolve_pitch_code(raw_code, description)
                display_names.setdefault(code, pitch_display_name(code, description))

                description_text = details.get("description", "")
                b = _bucket(code)
                b["pitches"] += 1

                is_swing = description_text in _SWING_DESCRIPTIONS
                is_whiff = description_text in _WHIFF_DESCRIPTIONS
                if is_swing:
                    b["swings"] += 1
                if is_whiff:
                    b["whiffs"] += 1

                # Chase% — real zone check with this pitch's own coordinates
                # and this batter's actual strike-zone bounds.
                pitch_data = event.get("pitchData", {})
                coords = pitch_data.get("coordinates", {})
                px, pz = coords.get("pX"), coords.get("pZ")
                sz_top, sz_bot = pitch_data.get("strikeZoneTop"), pitch_data.get("strikeZoneBottom")
                if px is not None and pz is not None and sz_top is not None and sz_bot is not None:
                    out_of_zone = abs(px) > PLATE_HALF_WIDTH_FT or pz < sz_bot or pz > sz_top
                    if out_of_zone:
                        b["outOfZonePitches"] += 1
                        if is_swing:
                            b["outOfZoneSwings"] += 1

                # PutAway% — v2.4: use the ENTERING count (state before this
                # pitch was thrown), not the post-pitch count field.
                is_last_pitch_of_pa = i == len(pitch_events) - 1
                if entering_strikes == 2:
                    b["twoStrikePitches"] += 1
                    if is_last_pitch_of_pa and result_event_type in _STRIKEOUT_EVENT_TYPES:
                        b["twoStrikeKs"] += 1

                # Advance entering_strikes to this pitch's resulting count
                # for the next pitch in the same plate appearance.
                post_count = event.get("count", {})
                post_strikes = post_count.get("strikes")
                if post_strikes is not None:
                    entering_strikes = min(int(post_strikes), 2)

                # Hard-Hit% — real exit velocity off this pitch type.
                if description_text in _IN_PLAY_DESCRIPTIONS:
                    hit_data = event.get("hitData", {})
                    launch_speed = hit_data.get("launchSpeed")
                    if launch_speed is not None:
                        b["battedBalls"] += 1
                        if launch_speed >= HARD_HIT_THRESHOLD_MPH:
                            b["hardHitBalls"] += 1

    results: dict[str, dict] = {}
    for code, b in buckets.items():
        results[code] = {
            "type": display_names.get(code, code),
            "sampleSize": b["pitches"],
            "whiffPct": round(b["whiffs"] / b["swings"] * 100, 1) if b["swings"] >= 3 else None,
            "chasePct": round(b["outOfZoneSwings"] / b["outOfZonePitches"] * 100, 1) if b["outOfZonePitches"] >= 3 else None,
            "putAwayPct": round(b["twoStrikeKs"] / b["twoStrikePitches"] * 100, 1) if b["twoStrikePitches"] >= 3 else None,
            "hardHitPct": round(b["hardHitBalls"] / b["battedBalls"] * 100, 1) if b["battedBalls"] >= 3 else None,
        }
    return results


def get_probable_lineup(game_pk: int, team_id: int) -> tuple[list[dict], bool]:
    try:
        box = fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore")
        for side in ("home", "away"):
            team_box = box.get("teams", {}).get(side, {})
            if team_box.get("team", {}).get("id") == team_id:
                order = team_box.get("battingOrder", [])
                players = team_box.get("players", {})
                if order:
                    batters = []
                    for pid in order:
                        p = players.get(f"ID{pid}", {})
                        person = p.get("person", {})
                        pos = p.get("position", {}).get("abbreviation", "")
                        if pos == "P" or not person.get("id"):
                            continue
                        batters.append({"id": person["id"], "name": person.get("fullName", "Unknown")})
                    if batters:
                        return batters[:9], True
    except Exception as e:
        log.warning(f"Boxscore lineup fetch failed for game {game_pk}: {e}")

    try:
        roster = fetch_json(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/active")
        batters = []
        for entry in roster.get("roster", []):
            if entry.get("position", {}).get("abbreviation") != "P":
                batters.append({"id": entry["person"]["id"], "name": entry["person"]["fullName"]})
        return batters[:9], False
    except Exception as e:
        log.warning(f"Active roster fallback failed for team {team_id}: {e}")
        return [], False


def get_batter_vs_pitcher(batter_id: int, pitcher_id: int) -> dict:
    url = (f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
           f"?stats=vsPlayer&opposingPlayerId={pitcher_id}&group=hitting&sportId=1")
    empty = {"pa": 0, "ab": 0, "h": 0, "hr": 0, "bb": 0, "so": 0,
             "avg": ".000", "obp": ".000", "slg": ".000", "ops": 0.0}
    try:
        data = fetch_json(url)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return empty
        stat = splits[0].get("stat", {})
        return {
            "pa": int(stat.get("plateAppearances", 0) or 0),
            "ab": int(stat.get("atBats", 0) or 0),
            "h": int(stat.get("hits", 0) or 0),
            "hr": int(stat.get("homeRuns", 0) or 0),
            "bb": int(stat.get("baseOnBalls", 0) or 0),
            "so": int(stat.get("strikeOuts", 0) or 0),
            "avg": stat.get("avg", ".000") or ".000",
            "obp": stat.get("obp", ".000") or ".000",
            "slg": stat.get("slg", ".000") or ".000",
            "ops": safe_float(stat.get("ops"), 0.0),
        }
    except Exception as e:
        log.warning(f"vsPlayer fetch failed batter={batter_id} pitcher={pitcher_id}: {e}")
        return empty


def get_batter_season_slash(batter_id: int) -> dict:
    """Real season AVG/SLG/OPS/PA — used both for the existing OPS blend
    and for Proxy xBA/xSLG shrinkage. Mod: Also pulls PA & K% + proxies Whiff/Chase 
    per user request to 'reverse engineer' for the Discipline table."""
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?stats=season&group=hitting&season={season}"
    fallback = {"avg": LEAGUE_AVG_AVG_FALLBACK, "slg": LEAGUE_AVG_SLG_FALLBACK, "ops": LEAGUE_AVG_OPS_FALLBACK, "pa": 0.0, "kPct": 0.0, "whiff": 0.0, "chase": 0.0, "csw": 0.0}
    try:
        data = fetch_json(url)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return fallback
        stat = splits[0].get("stat", {})
        avg = safe_float(stat.get("avg"), fallback["avg"])
        slg = safe_float(stat.get("slg"), fallback["slg"])
        ops = safe_float(stat.get("ops"), fallback["ops"])
        
        pa = safe_float(stat.get("plateAppearances"), 0.0)
        so = safe_float(stat.get("strikeOuts"), 0.0)
        kPct = round((so / pa * 100), 1) if pa > 0 else 0.0
        
        # Proxy calculations based on K% to drive the new K / Discipline UI
        whiff_proxy = round((kPct * 1.05) + 2.0, 1) if pa > 0 else 0.0
        chase_proxy = round((kPct * 0.95) + 4.0, 1) if pa > 0 else 0.0
        csw_proxy = round((kPct * 0.85) + 8.0, 1) if pa > 0 else 0.0
        
        return {
            "avg": avg if avg > 0 else fallback["avg"],
            "slg": slg if slg > 0 else fallback["slg"],
            "ops": ops if ops > 0 else fallback["ops"],
            "pa": pa,
            "kPct": kPct,
            "whiff": whiff_proxy,
            "chase": chase_proxy,
            "csw": csw_proxy,
        }
    except Exception:
        return fallback


def shrink_h2h_ops(h2h_pa: int, h2h_ops: float, season_ops: float) -> dict:
    credibility = h2h_pa / (h2h_pa + STABILIZATION_PA)
    blended = credibility * h2h_ops + (1 - credibility) * season_ops
    return {"credibility": round(credibility, 3), "blendedOps": round(blended, 3)}


def shrink_proxy_xba_xslg(h2h_pa: int, h2h_avg: float, h2h_slg: float, season_avg: float, season_slg: float) -> dict:
    """Proxy xBA / Proxy xSLG — credibility-weighted shrinkage of real
    career H2H AVG/SLG toward real season AVG/SLG. NOT launch-angle-based
    Statcast xBA/xSLG; label it 'shrunk' in the UI to keep that honest."""
    credibility = h2h_pa / (h2h_pa + STABILIZATION_PA)
    proxy_xba = credibility * h2h_avg + (1 - credibility) * season_avg
    proxy_xslg = credibility * h2h_slg + (1 - credibility) * season_slg
    return {"proxyXba": round(proxy_xba, 3), "proxyXslg": round(proxy_xslg, 3)}


def batter_matchup_job(batter: dict, pitcher_id: int) -> dict:
    h2h = get_batter_vs_pitcher(batter["id"], pitcher_id)
    season_slash = get_batter_season_slash(batter["id"])
    shrink = shrink_h2h_ops(h2h["pa"], h2h["ops"], season_slash["ops"])
    proxy = shrink_proxy_xba_xslg(
        h2h["pa"], safe_float(h2h["avg"]), safe_float(h2h["slg"]),
        season_slash["avg"], season_slash["slg"],
    )
    return {
        **batter,
        "h2h": h2h,
        "seasonOps": round(season_slash["ops"], 3),
        "seasonPa": season_slash["pa"],
        "seasonKPct": season_slash["kPct"],
        "seasonWhiff": season_slash["whiff"],
        "seasonChase": season_slash["chase"],
        "seasonCsw": season_slash["csw"],
        **shrink,
        **proxy,
    }


def edge_tier(blended_ops: float) -> str:
    if blended_ops >= 0.820:
        return "elite"
    if blended_ops >= 0.740:
        return "strong"
    if blended_ops >= 0.660:
        return "average"
    return "soft"


def get_matchup_analyzer(pitcher_id: int, pitcher_name: str, team_name: str,
                          opponent_team_id: int, opponent_name: str,
                          is_home: bool, game_pk: int) -> Optional[dict]:
    pitch_mix = get_pitch_arsenal(pitcher_id)

    # --- merge real per-pitch lethality onto each pitch entry, joined by CODE ---
    recent_pks = get_pitcher_recent_game_pks(pitcher_id)
    lethality = get_pitcher_pitch_lethality(pitcher_id, recent_pks) if recent_pks else {}
    for p in pitch_mix:
        stats = lethality.get(p.get("code", "UN"))
        p["whiffPct"] = stats["whiffPct"] if stats else None
        p["chasePct"] = stats["chasePct"] if stats else None
        p["putAwayPct"] = stats["putAwayPct"] if stats else None
        p["hardHitPct"] = stats["hardHitPct"] if stats else None
        p["lethalitySample"] = stats["sampleSize"] if stats else 0

    lineup, confirmed = get_probable_lineup(game_pk, opponent_team_id)
    if not lineup:
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        batters = list(pool.map(lambda b: batter_matchup_job(b, pitcher_id), lineup))

    # `lineup` (and therefore `batters`, via pool.map) is still in real
    # batting-order — capture that as "order" before we re-sort by
    # blendedOps below. The frontend's game-drawer lineup panel needs the
    # true 1-9 batting order, not the OPS-ranked position.
    for idx, b in enumerate(batters):
        b["edgeTier"] = edge_tier(b["blendedOps"])
        b["order"] = idx + 1

    ranked = sorted(batters, key=lambda b: b["blendedOps"], reverse=True)
    lineup_edge_ops = round(sum(b["blendedOps"] for b in batters) / len(batters), 3) if batters else 0.0
    total_h2h_pa = sum(b["h2h"]["pa"] for b in batters)

    return {
        "id": f"m_{pitcher_id}",
        "pitcherId": str(pitcher_id),
        "pitcherName": pitcher_name,
        "team": team_name,
        "opponent": opponent_name,
        "isHome": is_home,
        "matchup": matchup_label(opponent_name, is_home),
        "sub": f"{team_name} &middot; {matchup_label(opponent_name, is_home)}",
        "lineupConfirmed": confirmed,
        "pitchMix": pitch_mix,
        "batters": ranked,
        "lineupEdgeOps": lineup_edge_ops,
        "lineupEdgeTier": edge_tier(lineup_edge_ops),
        "totalH2hPa": total_h2h_pa,
    }


# ================= SLATE RUNNER =================
def get_slate(date_str: str) -> list[dict]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
           f"&hydrate=probablePitcher,team,venue")
    data = fetch_json(url)
    return data.get("dates", [{}])[0].get("games", []) if data.get("dates") else []


def print_edge_board(frontend_db: dict) -> None:
    rows = []
    for bucket in ("pitchers", "teams"):
        for entity in frontend_db[bucket]:
            for market_key, market in entity["markets"].items():
                m = market["model"]
                rows.append((entity["name"], market["label"], market["defaultLine"], m["edgePct"], m["distribution"], m["confidence"]))

    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    log.info("─" * 78)
    log.info(f"{'ENTITY':<24}{'MARKET':<22}{'LINE':>6}{'EDGE':>9}{'DIST':>14}{'CONF':>8}")
    log.info("─" * 78)
    for name, market, line, edge, dist, conf in rows[:15]:
        dist_short = "NB2" if dist == "negative_binomial" else "POI"
        log.info(f"{name:<24}{market:<22}{line:>6}{edge:>+8.1f}%{dist_short:>13}{conf:>9}")
    log.info("─" * 78)

    if frontend_db.get("matchups"):
        log.info("─" * 78)
        log.info(f"{'PITCHER':<22}{'OPP':<20}{'LINEUP':>10}{'EDGE OPS':>12}{'TIER':>10}")
        log.info("─" * 78)
        for m in sorted(frontend_db["matchups"], key=lambda x: x["lineupEdgeOps"], reverse=True)[:15]:
            lineup_tag = "confirmed" if m["lineupConfirmed"] else "projected"
            log.info(f"{m['pitcherName']:<22}{m['opponent']:<20}{lineup_tag:>10}{m['lineupEdgeOps']:>12.3f}{m['lineupEdgeTier']:>10}")
        log.info("─" * 78)


def run(date_str: Optional[str] = None, output_path: str = "projections.json") -> None:
    explicit_date = date_str is not None
    date_str = date_str or today_mlb_date()
    log.info(f"Fetching MLB slate for {date_str} (America/New_York)"
             f"{' [explicit --date]' if explicit_date else ' [today, MLB-local]'}")

    try:
        games = get_slate(date_str)
    except Exception as e:
        log.error(f"Could not fetch schedule: {e}")
        return

    if not games:
        log.warning(f"MLB Stats API returned 0 games for {date_str}.")
        if not explicit_date:
            local_now = datetime.now().strftime("%Y-%m-%d %H:%M %Z") or datetime.now().strftime("%Y-%m-%d %H:%M")
            log.warning(f"(Your machine's local time is {local_now} — if that looks off from "
                        f"{date_str} America/New_York, this is expected, not a bug.)")
        log.warning("If this is unexpected, MLB is likely between series (common on Mondays/"
                    "Thursdays) or it's an All-Star break / offseason date. "
                    "Try: python3 generate_projections.py --date YYYY-MM-DD")
        with open(output_path, "w") as f:
            json.dump({"pitchers": [], "teams": [], "matchups": []}, f, indent=2)
        return

    frontend_db = {"pitchers": [], "teams": [], "matchups": []}
    seen_team_ids: set[int] = set()
    seen_pitcher_ids: set[int] = set()

    jobs: list[tuple[str, tuple]] = []
    pitcher_count_expected = 0
    for game in games:
        away_team = game["teams"]["away"]["team"]
        home_team = game["teams"]["home"]["team"]
        matchup = f"{away_team['name']} @ {home_team['name']}"
        game_pk = game.get("gamePk")
        log.info(f"Queueing: {matchup}")

        for side, opp_side in (("away", "home"), ("home", "away")):
            team_info = game["teams"][side]["team"]
            team_id, team_name = team_info["id"], team_info["name"]
            opponent_info = game["teams"][opp_side]["team"]
            opponent_id, opponent_name = opponent_info["id"], opponent_info["name"]
            is_home = side == "home"

            if team_id not in seen_team_ids:
                seen_team_ids.add(team_id)
                jobs.append(("team", (team_id, team_name, opponent_name, is_home)))

            if "probablePitcher" in game["teams"][side]:
                pid = game["teams"][side]["probablePitcher"]["id"]
                pname = game["teams"][side]["probablePitcher"]["fullName"]
                pitcher_count_expected += 1
                if pid not in seen_pitcher_ids:
                    seen_pitcher_ids.add(pid)
                    jobs.append(("pitcher", (pid, pname, team_name, opponent_name, is_home)))
                    jobs.append(("matchup", (pid, pname, team_name, opponent_id, opponent_name, is_home, game_pk)))

    if pitcher_count_expected == 0:
        log.warning("No probable pitchers were listed for any game on this slate "
                    "(common early in the day before rotations are announced). "
                    "Team Matchups will still populate; re-run closer to game time "
                    "for Pitcher Props and Matchup Analyzer.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for kind, args in jobs:
            if kind == "team":
                fn = get_team_data
            elif kind == "pitcher":
                fn = get_pitcher_data
            else:
                fn = get_matchup_analyzer
            futures[pool.submit(fn, *args)] = kind

        for future in concurrent.futures.as_completed(futures):
            kind = futures[future]
            result = future.result()
            if not result:
                continue
            if kind == "team":
                frontend_db["teams"].append(result)
            elif kind == "pitcher":
                frontend_db["pitchers"].append(result)
            else:
                frontend_db["matchups"].append(result)

    # --- NEW v2.4: attach game start time + venue for the frontend's
    # game-context header (Moneylines tab) ---
    game_context = {}
    for game in games:
        gd = game.get("gameDate")
        venue = (game.get("venue") or {}).get("name")
        away_id = game["teams"]["away"]["team"]["id"]
        home_id = game["teams"]["home"]["team"]["id"]
        ctx = {"startTimeUTC": gd, "venue": venue}
        game_context[f"t_{away_id}"] = ctx
        game_context[f"t_{home_id}"] = ctx
    for team in frontend_db["teams"]:
        ctx = game_context.get(team["id"])
        if ctx:
            team["startTimeUTC"] = ctx["startTimeUTC"]
            team["venue"] = ctx["venue"]

    with open(output_path, "w") as f:
        json.dump(frontend_db, f, indent=2)

    log.info(f"✅ Done — {len(frontend_db['pitchers'])} pitchers, {len(frontend_db['teams'])} teams, "
              f"{len(frontend_db['matchups'])} matchups -> {output_path}")
    if frontend_db["pitchers"] or frontend_db["teams"]:
        print_edge_board(frontend_db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trckr21 quant data engine")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to today)")
    parser.add_argument("--output", type=str, default="projections.json")
    parser.add_argument("--debug-raw", action="store_true", help="Dump one raw pitchArsenal response to stderr")
    args = parser.parse_args()
    if args.debug_raw:
        DEBUG_DUMP_RAW = True
    run(date_str=args.date, output_path=args.output)