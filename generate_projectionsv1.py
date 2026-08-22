"""
trckr21 — MLB Quant Terminal Data Engine  (v2.2 — Matchup Analyzer)
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

Two deliberate upgrades on top of that baseline, both additive:
  1. Unbiased sample variance (n-1 denominator) instead of population
     variance — matters right at the VMR = 1 decision boundary.
  2. Numerically stable NB2 PMF in log-space via math.lgamma, safe for
     any k (the raw rising-factorial form can overflow).

Safety net: if a market's variance collapses to ~mu (or below it), the
engine falls back to Poisson for that single market rather than producing
garbage odds.

NEW IN v2.2 — MATCHUP ANALYZER
--------------------------------
Third data pipeline, `db.matchups`, one entry per today's probable pitcher.
Two parts, both built ONLY from the free MLB Stats API (no Statcast/
Baseball Savant subscription — this is a deliberate constraint, not an
oversight):

  1. PITCH MIX  — `stats=pitchArsenal` for the pitcher: pitch type, usage%,
     avg velocity. This is the free-tier ceiling; whiff%/chase%/CSW% are
     Statcast-derived and are NOT available without Savant, so they are not
     attempted here rather than being faked.

     ⚠ VERIFY-BEFORE-TRUST: I could not hit the live API from this sandbox
     (no network egress) to confirm the exact JSON key names for the
     pitchArsenal split. The parser below tries the field names documented
     in public MLB-StatsAPI usage (`pitchType.description`, `percentage`,
     `averageSpeed`) with defensive fallbacks, but you should log one raw
     response (see `DEBUG_DUMP_RAW` below) and confirm before trusting the
     numbers in production. This mirrors the same class of bug the v2.1
     revision already found and fixed for `strikeOuts` vs `strikeouts`.

  2. LINEUP H2H — for today's opposing lineup (pulled from the live
     boxscore's `battingOrder` once MLB posts it, ~30-90 min pre-game;
     falls back to the team's active-roster hitters, unordered, flagged
     `lineupConfirmed: false`, if the real lineup isn't out yet), each
     batter's career at-bats against THIS specific pitcher via the
     `vsPlayer` stat group.

     H2H sample sizes are frequently tiny (a batter may have 2-3 career
     PA against a given starter), so raw H2H OPS is mostly noise. Rather
     than presenting noise as signal, every batter's H2H OPS is blended
     toward his season OPS using a simple credibility-weighted shrinkage:

         credibility = PA / (PA + K)          K = STABILIZATION_PA
         blendedOps  = credibility * h2hOps + (1 - credibility) * seasonOps

     This is the same philosophy as the VMR gate elsewhere in this file:
     don't trust a statistic more than its sample size earns. K=100 is a
     single flat constant used across all batters for simplicity (a real
     sabermetric stabilization point differs per rate stat — OBP ~460 PA,
     SLG ~320 PA — using one flat K for a blended OPS is a deliberate
     simplification, flag if you want per-component shrinkage instead).

FIXES CARRIED FROM v2.1
-------------------------
  1. Schedule call hydrates `probablePitcher` explicitly (opt-in on the
     MLB Stats API — without it every pitcher job silently disappears).
  2. Every pitcher/team record carries today's actual opponent + a
     human-readable matchup string, folded into the sidebar `sub` line.
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

# MLB schedules run on the US slate date, not the machine's local date.
MLB_TZ = ZoneInfo("America/New_York")


def today_mlb_date() -> str:
    return datetime.now(MLB_TZ).strftime("%Y-%m-%d")

# ================= CONFIG =================
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 1.5
MAX_WORKERS = 6

MARKET_VIG_AMERICAN = -110
MARKET_IMPLIED_PROB = 110 / 210          # p(-110) on both sides of a two-way market
DECIMAL_B = 100 / 110                    # net-profit-per-$1-staked at -110, used in Kelly

# Credibility knob for H2H OPS shrinkage — see docstring above.
STABILIZATION_PA = 100
LEAGUE_AVG_OPS_FALLBACK = 0.710          # used only if a batter has zero season PA logged

# Set True to dump one raw pitchArsenal response to stderr on first call,
# so you can eyeball real field names against what the parser expects.
DEBUG_DUMP_RAW = False

DEFAULT_LINES = {
    "strikeouts": 4.5,
    "earned_runs": 1.5,
    "walks": 2.5,
    "f5_runs": 1.5,
    "total_runs": 4.5,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("trckr21")


# ================= HTTP =================
def fetch_json(url: str) -> dict:
    """GET + parse JSON with retries and exponential backoff."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trckr21-quant-engine/2.2"})
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
    """MLB Stats API frequently returns rate stats as strings (e.g. '.318',
    or '-' for undefined). Centralized so every parser handles it the
    same way instead of re-deriving this edge case per field."""
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


# ================= STATISTICS CORE =================
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


# ================= MODEL =================
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


# ================= PITCHER ENGINE =================
def get_pitcher_data(pitcher_id: int, pitcher_name: str, team_name: str,
                      opponent_name: str, is_home: bool) -> Optional[dict]:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={season}"

    try:
        data = fetch_json(url)
        games = data.get("stats", [{}])[0].get("splits", [])[-10:]
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


# ================= TEAM ENGINE =================
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


# ================= MATCHUP ANALYZER ENGINE (NEW) =================
_raw_dump_done = False


def get_pitch_arsenal(pitcher_id: int) -> list[dict]:
    """Pitch type usage% + avg velocity, free-tier MLB Stats API only.
    See VERIFY-BEFORE-TRUST note in the module docstring."""
    global _raw_dump_done
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=pitchArsenal&group=pitching&season={season}"
    try:
        data = fetch_json(url)

        if DEBUG_DUMP_RAW and not _raw_dump_done:
            import sys
            print(json.dumps(data, indent=2)[:3000], file=sys.stderr)
            _raw_dump_done = True

        splits = data.get("stats", [{}])[0].get("splits", [])
        pitches = []
        for s in splits:
            stat = s.get("stat", s)  # some stat groups nest under "stat", some don't
            pt = stat.get("pitchType") or stat.get("type") or {}
            name = pt.get("description") or pt.get("displayName") or stat.get("pitchName") or "Unknown"

            usage_raw = safe_float(stat.get("percentage", stat.get("usage", 0)))
            # API has returned this as either a 0-1 fraction or an already-
            # scaled 0-100 number across different stat groups historically —
            # normalize defensively rather than assume one or the other.
            usage_pct = round(usage_raw * 100, 1) if usage_raw <= 1 else round(usage_raw, 1)

            velo = safe_float(stat.get("averageSpeed", stat.get("avgSpeed", 0)))

            if name == "Unknown" and usage_pct == 0:
                continue
            pitches.append({"type": name, "usagePct": usage_pct, "avgVelo": round(velo, 1)})

        pitches.sort(key=lambda p: p["usagePct"], reverse=True)
        return pitches
    except Exception as e:
        log.warning(f"Pitch arsenal fetch failed for pitcher {pitcher_id}: {e}")
        return []


def get_probable_lineup(game_pk: int, team_id: int) -> tuple[list[dict], bool]:
    """(batters, confirmed). Tries the live boxscore's posted battingOrder
    first (real, official, usually up ~30-90min pre-game). Falls back to
    the team's active-roster hitters — unordered, confirmed=False — if
    the real lineup isn't posted yet. The frontend must show the
    confirmed/projected distinction; never present a roster guess as a
    confirmed lineup."""
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
    """Career batter-vs-this-pitcher line via the vsPlayer stat group."""
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


def get_batter_season_ops(batter_id: int) -> float:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?stats=season&group=hitting&season={season}"
    try:
        data = fetch_json(url)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return LEAGUE_AVG_OPS_FALLBACK
        ops = safe_float(splits[0].get("stat", {}).get("ops"), LEAGUE_AVG_OPS_FALLBACK)
        return ops if ops > 0 else LEAGUE_AVG_OPS_FALLBACK
    except Exception:
        return LEAGUE_AVG_OPS_FALLBACK


def shrink_h2h_ops(h2h_pa: int, h2h_ops: float, season_ops: float) -> dict:
    """Empirical-Bayes-style credibility blend — see module docstring for
    why raw small-sample H2H OPS is not trustworthy on its own."""
    credibility = h2h_pa / (h2h_pa + STABILIZATION_PA)
    blended = credibility * h2h_ops + (1 - credibility) * season_ops
    return {"credibility": round(credibility, 3), "blendedOps": round(blended, 3)}


def batter_matchup_job(batter: dict, pitcher_id: int) -> dict:
    h2h = get_batter_vs_pitcher(batter["id"], pitcher_id)
    season_ops = get_batter_season_ops(batter["id"])
    shrink = shrink_h2h_ops(h2h["pa"], h2h["ops"], season_ops)
    return {**batter, "h2h": h2h, "seasonOps": round(season_ops, 3), **shrink}


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
    lineup, confirmed = get_probable_lineup(game_pk, opponent_team_id)
    if not lineup:
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        batters = list(pool.map(lambda b: batter_matchup_job(b, pitcher_id), lineup))

    for b in batters:
        b["edgeTier"] = edge_tier(b["blendedOps"])

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
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,team"
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
                    # Matchup Analyzer: this pitcher vs the OPPOSING team's lineup.
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