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

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock

from config import settings
from math_engine import (
    LEAGUE_AVG_BA,
    LEAGUE_AVG_BABIP,
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_HR_RATE,
    LEAGUE_AVG_ISO,
    LEAGUE_AVG_OBP,
    LEAGUE_AVG_RBI_RATE,
    LEAGUE_AVG_RUN_RATE,
    LEAGUE_AVG_SLG,
    K_HITS,
    K_HR,
    K_RBI,
    K_RUNS,
    MAX_PROB,
    MIN_PROB,
    ahead_in_count_boost,
    analyze_pitcher_split,
    apply_split_effect,
    babip_regression_penalty_points,
    build_pitcher_ladder,
    calculate_team_xruns_v2,
    calculate_woba_from_stats,
    clamp,
    confidence_from_edge,
    fastball_whiff_proxy,
    hit_quality_modifier,
    home_away_scoring_factor,
    lineup_order_bonus_mult,
    pitcher_velocity_mod,
    prob_to_american,
    process_hr_prob,
    process_rbi_prob,
    process_run_prob,
    projected_pa_from_order,
    pythagenpat_win_prob,
    remove_vig,
    safe_float,
    shrink_rate,
    velocity_control_penalty,
)
from mlb_client import MLBClient
from odds_client import OddsClient
from repository import PredictionRepository

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
    def __init__(self, mlb: MLBClient, odds: OddsClient, repository: PredictionRepository) -> None:
        self.mlb = mlb
        self.odds = odds
        self.repository = repository
        self._slate_cache: dict[str, dict] = {}
        self._player_index: dict[int, dict] = {}
        self._pitcher_index: dict[int, dict] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # SLATE BUILD
    # ------------------------------------------------------------------

    def build_slate(self, target_date: str, force: bool = False) -> dict:
        with self._lock:
            if not force and target_date in self._slate_cache:
                return self._slate_cache[target_date]

        games = self.mlb.schedule(target_date)
        prop_odds = self.odds.player_prop_index()
        moneyline_odds = self.odds.moneyline_index()

        player_rows: list[dict] = []
        game_rows: list[dict] = []
        pitcher_rows: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            futures = [
                executor.submit(self._build_game, game, target_date, prop_odds, moneyline_odds)
                for game in games
            ]
            for future in as_completed(futures):
                try:
                    packet = future.result()
                except Exception:
                    continue
                player_rows.extend(packet["players"])
                game_rows.append(packet["game"])
                pitcher_rows.update(packet["pitchers"])

        player_rows.sort(key=lambda item: item["props"]["hits"]["prob"], reverse=True)

        slate = {
            "meta": {
                "date": target_date,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "season": settings.season,
                "playerCount": len(player_rows),
                "gameCount": len(game_rows),
                "marketDataAvailable": self.odds.enabled,
            },
            "players": player_rows,
            "games": game_rows,
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

    def _resolve_hitters(self, team_id: int, lineup_list: list[dict]) -> list[dict]:
        roster = self.mlb.team_roster(team_id)
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
            return hitters
        return list(non_pitchers.values())[:9]

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
        park_factor = STADIUM_INDICES.get(venue_name, {"base_park_factor": 1.00})["base_park_factor"]

        away_pitcher_id = away_node.get("probablePitcher", {}).get("id")
        home_pitcher_id = home_node.get("probablePitcher", {}).get("id")
        away_pitcher = self.mlb.pitcher_profile(away_pitcher_id)
        home_pitcher = self.mlb.pitcher_profile(home_pitcher_id)

        away_team_hitting = self.mlb.team_stats(away_id, "hitting")
        home_team_hitting = self.mlb.team_stats(home_id, "hitting")
        away_team_pitching = self.mlb.team_stats(away_id, "pitching")
        home_team_pitching = self.mlb.team_stats(home_id, "pitching")

        away_team_obp = safe_float(away_team_hitting.get("obp"), LEAGUE_AVG_OBP)
        home_team_obp = safe_float(home_team_hitting.get("obp"), LEAGUE_AVG_OBP)
        away_team_slg = safe_float(away_team_hitting.get("slg"), LEAGUE_AVG_SLG)
        home_team_slg = safe_float(home_team_hitting.get("slg"), LEAGUE_AVG_SLG)
        away_bullpen_era = safe_float(away_team_pitching.get("era"), LEAGUE_AVG_ERA)
        home_bullpen_era = safe_float(home_team_pitching.get("era"), LEAGUE_AVG_ERA)

        # Home/away scoring factor — fetched once per team per game, not per hitter.
        away_runs, away_hits = self.mlb.team_home_away_runs_hits(away_id, is_home=False)
        home_runs, home_hits = self.mlb.team_home_away_runs_hits(home_id, is_home=True)
        away_home_away_scalar = home_away_scoring_factor(away_runs, away_hits)
        home_home_away_scalar = home_away_scoring_factor(home_runs, home_hits)

        def _matchup_mult(team_obp, team_slg, opp_starter_era):
            offense_index = clamp(((team_obp / LEAGUE_AVG_OBP) + (team_slg / LEAGUE_AVG_SLG)) / 2.0, 0.80, 1.25)
            pitching_index = clamp(LEAGUE_AVG_ERA / opp_starter_era if opp_starter_era > 0 else 1.0, 0.75, 1.30)
            return offense_index * pitching_index

        away_matchup_mult = _matchup_mult(away_team_obp, away_team_slg, home_pitcher.get("era", LEAGUE_AVG_ERA))
        home_matchup_mult = _matchup_mult(home_team_obp, home_team_slg, away_pitcher.get("era", LEAGUE_AVG_ERA))

        away_xruns = calculate_team_xruns_v2(
            matchup_mult=away_matchup_mult, park_factor=park_factor, weather_mult=1.0,
            bullpen_era=home_bullpen_era, bullpen_fatigue_mult=1.0,
        )
        home_xruns = calculate_team_xruns_v2(
            matchup_mult=home_matchup_mult, park_factor=park_factor, weather_mult=1.0,
            bullpen_era=away_bullpen_era, bullpen_fatigue_mult=1.0,
        )
        home_win_prob, away_win_prob = pythagenpat_win_prob(home_xruns, away_xruns)

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

        team_iter = [
            (away_id, away_abbr, away_name, away_team_obp, away_team_slg, home_pitcher, home_bullpen_era, away_lineup_list, away_home_away_scalar),
            (home_id, home_abbr, home_name, home_team_obp, home_team_slg, away_pitcher, away_bullpen_era, home_lineup_list, home_home_away_scalar),
        ]

        for team_id, team_abbr, team_name, team_obp, team_slg, opp_pitcher, opp_bullpen_era, lineup_list, home_away_scalar in team_iter:
            hitters = self._resolve_hitters(team_id, lineup_list)
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

                bat_side = "R"
                try:
                    bat_side = self.mlb.person(pid).get("batSide", {}).get("code") or "R"
                except Exception:
                    pass

                hitting_stats = self.mlb.player_season_hitting(pid)
                platoon_stats = self.mlb.hitter_platoon_split(pid, opp_pitcher_hand)
                game_log = self.mlb.player_game_log(pid, target_date, days=50)
                recent_14d = self.mlb.player_recent_hitting(pid, target_date, days=14)
                ahead_avg = self.mlb.hitter_ahead_in_count_avg(pid)

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

                # Persist each prop so /api/track-record and sync_results() have
                # something to grade against once games finish (see FIX 2 note
                # at the top of this file). upsert so re-building the same
                # slate before first pitch just refreshes the line, not a dupe.
                for prop_type, prob_value, quote in (
                    ("hits", hit_prob, hit_quote),
                    ("homeruns", hr_prob, hr_quote),
                    ("rbi", rbi_prob, rbi_quote),
                    ("runs", run_prob, run_quote),
                ):
                    try:
                        self.repository.upsert_prediction(
                            game_pk=game_pk,
                            game_date=target_date,
                            player_id=pid,
                            player_name=pname,
                            prop_type=prop_type,
                            probability=prob_value,
                            fair_odds=prob_to_american(prob_value),
                            book_odds=(quote or {}).get("over"),
                        )
                    except Exception:
                        # Never let a persistence hiccup break slate building —
                        # the live quote already went into `players` above.
                        pass

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
        }

        return {"game": game_row, "players": players, "pitchers": pitchers_out}

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

        prob = raw_hit_prob * (1.0 - (opp_pitcher_k_rate * 0.22)) * (1.0 + ((1.20 - opp_pitcher_whip) * 0.05))

        # Bullpen cap — approximated from opposing bullpen ERA vs league avg
        # (the original's calculate_team_bullpen_fatigue_multiplier also
        # factors in innings/games workload ratio, which isn't cheaply
        # available from the free MLB Stats API; ERA is the dominant term).
        bullpen_mult = clamp(1.0 + ((LEAGUE_AVG_ERA - opp_bullpen_era) / 25.0), 0.95, 1.06)
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

        # BATAS 2 — BABIP regression
        prob += babip_regression_penalty_points(babip_14d)

        # BATAS 3 — lineup order bonus
        prob *= lineup_order_bonus_mult(batting_order)

        # V8.9 quality modifier + the second (multiplicative) velocity mod
        quality_mod = hit_quality_modifier(babip_14d, iso_val)
        velo_mod = pitcher_velocity_mod(pitcher_velo)
        final_calibrated_prob = prob * (1.0 + quality_mod + velo_mod)

        return round(clamp(final_calibrated_prob / 100.0, 0.0, 1.0), 4)

    # ------------------------------------------------------------------
    # Matchup Verifier
    # ------------------------------------------------------------------

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
        k_penalty = (pitcher_k_rate - 22.0) * 4.0 + (batter_k_rate - 22.0) * 2.0
        raw_advantage = split_advantage - k_penalty

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
        for row in pending:
            try:
                boxscore = self.mlb._get(f"/game/{row['game_pk']}/boxscore")
                actual = self._extract_actual_stat(boxscore, row["player_id"], row["prop_type"])
                if actual is None:
                    continue
                self.repository.set_actual(row["game_pk"], row["player_id"], row["prop_type"], actual)
                resolved_count += 1
            except Exception:
                continue
        return {"checked": len(pending), "resolved": resolved_count}

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