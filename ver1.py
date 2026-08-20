"""
trckr21 — MLB Quant Terminal Data Engine
==========================================
Pulls today's slate from the MLB Stats API and produces `projections.json`
in the exact shape the frontend consumes (db.pitchers / db.teams).

CALIBRATION ENGINE
-------------------
Baseline math is the user's own spec — a dynamic Variance-to-Mean Ratio
(VMR) gate that switches between two distributions per market, per player,
per game, based on the actual shape of the last-10-game sample:

    VMR = sigma^2 / mu

    VMR <= 1  ->  Standard Poisson        (no clustering / contagion)
    VMR >  1  ->  Negative Binomial (NB2) (over-dispersed / "contagious")

Two deliberate upgrades on top of that baseline, both additive — they
sharpen the same model, they don't replace it:

  1. Unbiased sample variance (n-1 denominator) instead of the population
     variance (n denominator) the spec's own formula uses. With n=10 this
     is a ~11% variance correction, which matters a lot right at the
     VMR = 1 decision boundary — population variance can accidentally
     route a genuinely over-dispersed stat into the plain Poisson bucket.

  2. A numerically stable NB2 PMF computed in log-space with math.lgamma
     instead of the raw rising-factorial product. The rising-factorial
     form in the spec is correct but can overflow/underflow for larger k;
     the log-gamma form is the standard textbook rewrite of the exact
     same PMF and is safe for any k.

  Safety net: if a market's variance collapses to ~mu (or below it) the
  NB2 parameters (r, p) become invalid — the engine automatically falls
  back to Poisson for that single market rather than producing garbage
  odds. Every market's output reports which distribution actually ran
  and how confident we are in it (sample-size based).

Everything else — American-odds conversion, the -110 market-vig
baseline, and Kelly staking — follows the user's formulas exactly.
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
# Running this from PH (UTC+8) or anywhere east of the US on `datetime.now()`
# can silently roll onto a date the US slate hasn't reached yet, or has
# already passed — either way the schedule endpoint comes back with zero
# games for that date string, and every downstream fetch has nothing to do.
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

# Frontend default lines. NOTE: the original backend and frontend disagreed
# on the walks line (1.5 vs 2.5) and the team total-runs line (3.5 vs 4.5).
# Standardized here to the market-realistic values used by the frontend.
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
            req = urllib.request.Request(url, headers={"User-Agent": "trckr21-quant-engine/2.0"})
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


# ================= STATISTICS CORE =================
def sample_mean(values: list[float]) -> float:
    n = len(values)
    return sum(values) / n if n else 0.0


def sample_variance(values: list[float]) -> float:
    """Unbiased (n-1) sample variance. Falls back to population variance
    (n denominator) only when n=1, where n-1 would divide by zero."""
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
    target_int = math.ceil(line)  # line is always x.5, so this is line+0.5
    cumulative = sum(poisson_pmf(lam, i) for i in range(target_int))
    return min(1.0, max(0.0, 1.0 - cumulative))


def nbinom_pmf(k: int, r: float, p: float) -> float:
    """NB2 PMF in log-space via lgamma — the numerically stable rewrite of
    the rising-factorial form: C(r+k-1, k) * p^r * (1-p)^k."""
    if not (0 < p < 1) or r <= 0:
        raise ValueError("invalid negative-binomial parameters")
    log_coef = math.lgamma(r + k) - math.lgamma(r) - math.lgamma(k + 1)
    log_pmf = log_coef + r * math.log(p) + k * math.log(1 - p)
    return math.exp(log_pmf)


def nbinom_params(mu: float, var: float) -> tuple[float, float]:
    """p = mu / sigma^2 ; r = mu^2 / (sigma^2 - mu)"""
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
    """Fraction of bankroll (%) to stake at -110, per the standard Kelly
    criterion f* = (bp - q) / b. Returns 0 when there's no edge."""
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
    distribution: str          # "poisson" | "negative_binomial"
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
    fallback: bool = False      # true if NB2 was attempted but fell back to Poisson


def calibrate_market(history: list[int], line: float) -> MarketModel:
    """The VMR gate. History -> distribution choice -> fair probability -> odds/edge/Kelly."""
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
def get_pitcher_data(pitcher_id: int, pitcher_name: str, team_name: str) -> Optional[dict]:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={season}"

    try:
        data = fetch_json(url)
        games = data.get("stats", [{}])[0].get("splits", [])[-10:]  # last 10, chronological
        if not games:
            return None

        history_k, history_er, history_bb = [], [], []
        labels, match_details = [], []

        for g in games:
            s = g["stat"]
            history_k.append(int(s.get("strikeouts", 0)))
            history_er.append(int(s.get("earnedRuns", 0)))
            history_bb.append(int(s.get("baseOnBalls", 0)))

            labels.append(format_date_label(g.get("date", ""), g.get("isHome", True), g.get("opponent", {}).get("name", "OPP")))

            score_prefix = "W" if g.get("isWin", False) else "L"
            match_details.append({
                "score": score_prefix,
                "text": f"IP: {s.get('inningsPitched', '0.0')} | ER: {history_er[-1]} | BB: {history_bb[-1]} | K: {history_k[-1]}",
            })

        return {
            "id": str(pitcher_id),
            "name": pitcher_name,
            "sub": f"{team_name} &middot; P",
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
def get_team_data(team_id: int, team_name: str) -> Optional[dict]:
    season = datetime.now(MLB_TZ).year
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=gameLog&group=hitting&season={season}"

    try:
        data = fetch_json(url)
        games = data.get("stats", [{}])[0].get("splits", [])[-10:]
        if not games:
            return None

        history_f5, history_full = [], []
        labels, match_details = [], []

        # Fetch linescores concurrently — this is the slow part (one call/game)
        def fetch_linescore(g):
            game_pk = g["game"]["gamePk"]
            return g, fetch_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = list(pool.map(fetch_linescore, games))

        for g, ls_data in results:
            is_home = g.get("isHome", True)
            side = "home" if is_home else "away"
            innings = ls_data.get("innings", [])
            f5_runs = sum(inning.get(side, {}).get("runs", 0) for inning in innings[:5])
            full_runs = int(g["stat"].get("runs", 0))

            history_f5.append(f5_runs)
            history_full.append(full_runs)

            labels.append(format_date_label(g.get("date", ""), is_home, g.get("opponent", {}).get("name", "OPP")))
            score_prefix = "W" if g.get("isWin", False) else "L"
            match_details.append({
                "score": score_prefix,
                "text": f"F5 Runs: {f5_runs} | Full Game Runs: {full_runs}",
            })

        return {
            "id": f"t_{team_id}",
            "name": team_name,
            "sub": "Team Matchup",
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


# ================= SLATE RUNNER =================
def get_slate(date_str: str) -> list[dict]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    data = fetch_json(url)
    return data.get("dates", [{}])[0].get("games", []) if data.get("dates") else []


def print_edge_board(frontend_db: dict) -> None:
    """Console readout of the biggest model-vs-market edges, sorted."""
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


def run(date_str: Optional[str] = None, output_path: str = "projections.json") -> None:
    explicit_date = date_str is not None
    date_str = date_str or today_mlb_date()
    log.info(f"Fetching MLB slate for {date_str} (America/New_York){'':s}"
             f"{' [explicit --date]' if explicit_date else ' [today, MLB-local]'}")

    try:
        games = get_slate(date_str)
    except Exception as e:
        log.error(f"Could not fetch schedule: {e}")
        return

    if not games:
        # Distinguish "the API is telling us the truth" from "something's
        # wrong upstream" — check yesterday/tomorrow so a bad date is obvious
        # in the log instead of just producing an empty file with no context.
        log.warning(f"MLB Stats API returned 0 games for {date_str}.")
        if not explicit_date:
            local_now = datetime.now().strftime("%Y-%m-%d %H:%M %Z") or datetime.now().strftime("%Y-%m-%d %H:%M")
            log.warning(f"(Your machine's local time is {local_now} — if that looks off from "
                        f"{date_str} America/New_York, this is expected, not a bug.)")
        log.warning("If this is unexpected, MLB is likely between series (common on Mondays/"
                    "Thursdays) or it's an All-Star break / offseason date. "
                    "Try: python3 generate_projections.py --date YYYY-MM-DD")
        with open(output_path, "w") as f:
            json.dump({"pitchers": [], "teams": []}, f, indent=2)
        return

    frontend_db = {"pitchers": [], "teams": []}
    seen_team_ids: set[int] = set()
    seen_pitcher_ids: set[int] = set()

    jobs: list[tuple[str, tuple]] = []
    for game in games:
        matchup = f"{game['teams']['away']['team']['name']} @ {game['teams']['home']['team']['name']}"
        log.info(f"Queueing: {matchup}")

        for side in ("away", "home"):
            team_info = game["teams"][side]["team"]
            team_id, team_name = team_info["id"], team_info["name"]

            if team_id not in seen_team_ids:
                seen_team_ids.add(team_id)
                jobs.append(("team", (team_id, team_name)))

            if "probablePitcher" in game["teams"][side]:
                pid = game["teams"][side]["probablePitcher"]["id"]
                pname = game["teams"][side]["probablePitcher"]["fullName"]
                if pid not in seen_pitcher_ids:
                    seen_pitcher_ids.add(pid)
                    jobs.append(("pitcher", (pid, pname, team_name)))

    # Run team/pitcher fetches concurrently — each is network-bound
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for kind, args in jobs:
            fn = get_team_data if kind == "team" else get_pitcher_data
            futures[pool.submit(fn, *args)] = kind

        for future in concurrent.futures.as_completed(futures):
            kind = futures[future]
            result = future.result()
            if result:
                frontend_db["teams" if kind == "team" else "pitchers"].append(result)

    with open(output_path, "w") as f:
        json.dump(frontend_db, f, indent=2)

    log.info(f"✅ Done — {len(frontend_db['pitchers'])} pitchers, {len(frontend_db['teams'])} teams -> {output_path}")
    if frontend_db["pitchers"] or frontend_db["teams"]:
        print_edge_board(frontend_db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trckr21 quant data engine")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to today)")
    parser.add_argument("--output", type=str, default="projections.json")
    args = parser.parse_args()
    run(date_str=args.date, output_path=args.output)