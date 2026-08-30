# mlb_client.py
# Adds three fetches needed to complete the ABI hit-probability port:
#   - pitcher_fatigue_and_velocity(): early/late-inning WHIP split (fatigue)
#     + primary fastball avg velocity (pitchArsenal). The velocity part is a
#     genuine new wire-up: in the 2000+ line original, `pitcher_avg_fb_velo`
#     was referenced but never actually populated anywhere, so BATAS 1
#     (velocity control) was dead code there. This makes it real.
#   - hitter_ahead_in_count_avg(): "Ahead in Count" split AVG, for Pillar 4.
#   - team_home_away_runs_hits(): team-level home/away runs & hits, for
#     get_team_home_away_scoring_factor().
# Everything else is unchanged from your working version.

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import requests

from config import settings
from math_engine import safe_float


class MLBClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "trckr21-mlb-quant/1.0"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        base_url = settings.mlb_api_base.strip().rstrip("/")
        if "[" in base_url and "]" in base_url:
            base_url = base_url.split("(")[-1].rstrip(")")
        try:
            response = self.session.get(f"{base_url}{path}", params=params, timeout=settings.request_timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"🚨 MLB API ERROR: Can't Connect to {base_url}{path}")
            print(f"🚨 Reason: {e}")
            return {}

    def schedule(self, target_date: str) -> list[dict]:
        payload = self._get(
            "/schedule",
            {
                "sportId": 1,
                "date": target_date,
                "hydrate": "lineups,probablePitcher(note),venue,team,game(content(summary))",
            },
        )
        dates = payload.get("dates", [])
        return dates[0].get("games", []) if dates else []

    @lru_cache(maxsize=256)
    def person(self, person_id: int) -> dict:
        payload = self._get(f"/people/{person_id}")
        people = payload.get("people", [])
        return people[0] if people else {}

    def team_roster(self, team_id: int) -> list[dict]:
        payload = self._get(f"/teams/{team_id}/roster", {"rosterType": "active"})
        return payload.get("roster", [])

    def person_stats(self, person_id: int, group: str, stats: str = "season", season: int | None = None, **params) -> dict:
        query = {"stats": stats, "group": group, "season": season or settings.season, **params}
        return self._get(f"/people/{person_id}/stats", query)

    def team_stats(self, team_id: int, group: str, season: int | None = None) -> dict:
        payload = self._get(
            f"/teams/{team_id}/stats",
            {"stats": "season", "group": group, "season": season or settings.season},
        )
        return self._first_stat(payload)

    def team_stats_raw(self, team_id: int, group: str, stats: str = "season", **params) -> dict:
        query = {"stats": stats, "group": group, "season": settings.season, **params}
        return self._get(f"/teams/{team_id}/stats", query)

    def player_season_hitting(self, player_id: int) -> dict:
        payload = self.person_stats(player_id, group="hitting", stats="season")
        return self._first_stat(payload)

    def player_recent_hitting(self, player_id: int, target_date: str, days: int = 14) -> dict:
        try:
            end_date = date.fromisoformat(str(target_date))
        except (ValueError, TypeError):
            end_date = date.today()
        start_date = end_date - timedelta(days=days)
        payload = self.person_stats(
            player_id, group="hitting", stats="byDateRange",
            startDate=start_date.isoformat(), endDate=end_date.isoformat(),
        )
        return self._first_stat(payload)

    def player_game_log(self, player_id: int, target_date: str, days: int = 50) -> list[dict]:
        try:
            end_date = date.fromisoformat(str(target_date))
        except (ValueError, TypeError):
            end_date = date.today()
        start_date = end_date - timedelta(days=days)
        payload = self.person_stats(
            player_id, group="hitting", stats="gameLog",
            startDate=start_date.isoformat(), endDate=end_date.isoformat(),
        )
        stats = payload.get("stats", [])
        return stats[0].get("splits", []) if stats else []

    def hitter_platoon_split(self, player_id: int, pitcher_hand: str) -> dict:
        sit_code = "vl" if pitcher_hand == "L" else "vr"
        payload = self.person_stats(player_id, group="hitting", stats="statSplits", sitCodes=sit_code)
        return self._first_stat(payload)

    # --- Ahead-in-Count split (Pillar 4 count-bias overlay) ---
    def hitter_ahead_in_count_avg(self, player_id: int) -> float | None:
        try:
            payload = self.person_stats(player_id, group="hitting", stats="statSplits")
            for block in payload.get("stats", []):
                for split in block.get("splits", []):
                    if "Ahead in Count" in split.get("split", {}).get("description", ""):
                        return safe_float(split.get("stat", {}).get("avg"), None)
        except Exception:
            pass
        return None

    # --- Team home/away runs & hits (for home_away_scoring_factor) ---
    def team_home_away_runs_hits(self, team_id: int, is_home: bool) -> tuple[float, float]:
        default = (1.0, 1.0)
        if not team_id:
            return default
        try:
            payload = self.team_stats_raw(team_id, group="hitting", stats="statSplits")
            target = "Home" if is_home else "Away"
            for block in payload.get("stats", []):
                for split in block.get("splits", []):
                    if split.get("split", {}).get("description", "") == target:
                        stat = split.get("stat", {})
                        return safe_float(stat.get("runs"), 1.0), safe_float(stat.get("hits"), 1.0)
        except Exception:
            pass
        return default

    def hitter_vs_pitcher_history(self, batter_id: int, pitcher_id: int) -> list[dict]:
        payload = self.person_stats(batter_id, group="hitting", stats="vsPlayer", opposingPlayerId=pitcher_id)
        stats = payload.get("stats", [])
        return stats[0].get("splits", []) if stats else []

    def pitcher_profile(self, pitcher_id: int | None) -> dict:
        if not pitcher_id:
            return self.default_pitcher()

        person = self.person(pitcher_id)
        hand = person.get("pitchHand", {}).get("code", "R")

        season_payload = self.person_stats(pitcher_id, group="pitching", stats="season")
        stat = self._first_stat(season_payload)

        innings = safe_float(stat.get("inningsPitched"), 0.0)
        games = int(stat.get("gamesPitched", 0) or 0)
        strikeouts = int(stat.get("strikeOuts", 0) or 0)
        batters_faced = int(stat.get("battersFaced", 0) or 0)

        avg_ip = innings / games if games else 5.5
        k_per_ip = strikeouts / innings if innings else 0.9

        projected_k = k_per_ip * avg_ip
        projected_er = (safe_float(stat.get("era"), 4.20) / 9.0) * avg_ip

        splits = self.pitcher_platoon_splits(pitcher_id)
        fatigue = self.pitcher_fatigue_and_velocity(pitcher_id)

        return {
            "id": pitcher_id,
            "name": person.get("fullName", "Unknown"),
            "hand": hand,
            "era": safe_float(stat.get("era"), 4.20),
            "whip": safe_float(stat.get("whip"), 1.32),
            "k_rate": strikeouts / batters_faced if batters_faced else 0.22,
            "games_started": int(stat.get("gamesStarted", 0) or 0),
            "innings": innings,
            "proj_k": round(projected_k, 2),
            "proj_er": round(projected_er, 2),
            "avg_vs_left": splits["vsL"]["baa"],
            "avg_vs_right": splits["vsR"]["baa"],
            "splits": splits,
            "fatigue_factor": fatigue["fatigue_factor"],
            "is_high_fatigue": fatigue["is_high_fatigue"],
            "avg_fastball_velo": fatigue["avg_fastball_velo"],
            "stats_source": "current-season" if innings > 0 else "league-fallback",
        }

    # --- Fatigue (Pillar 6, early vs late innings WHIP) + fastball velocity
    # (pitchArsenal). Ported from calculate_abi_pitcher_profile_and_tier's
    # inning-split block; velocity is a NEW real wire-up (see module docstring).
    def pitcher_fatigue_and_velocity(self, pitcher_id: int | None) -> dict:
        result = {"fatigue_factor": 1.0, "is_high_fatigue": False, "avg_fastball_velo": 0.0}
        if not pitcher_id:
            return result

        try:
            payload = self.person_stats(pitcher_id, group="pitching", stats="statSplits")
            early_whip, late_whip = 1.20, 1.20
            for block in payload.get("stats", []):
                for split in block.get("splits", []):
                    desc = split.get("split", {}).get("description", "")
                    stat = split.get("stat", {})
                    if "Innings 1-3" in desc:
                        early_whip = safe_float(stat.get("whip"), 1.20)
                    elif "Innings 4-6" in desc:
                        late_whip = safe_float(stat.get("whip"), 1.20)
            if late_whip > (early_whip * 1.15):
                result["fatigue_factor"] = 1.12
                result["is_high_fatigue"] = True
        except Exception:
            pass

        try:
            arsenal = self.person_stats(pitcher_id, group="pitching", stats="pitchArsenal")
            splits = arsenal.get("stats", [{}])[0].get("splits", [])
            best_usage, best_velo = -1.0, 0.0
            for s in splits:
                stat = s.get("stat", s)
                pt_node = stat.get("pitchType") or {}
                pitch_name = str(pt_node.get("description") or stat.get("pitchName") or "").lower()
                if not any(k in pitch_name for k in ("fastball", "sinker", "four-seam", "two-seam")):
                    continue
                usage_raw = safe_float(stat.get("percentage", stat.get("usage", 0)), 0.0)
                usage = usage_raw * 100 if usage_raw <= 1 else usage_raw
                velo = safe_float(stat.get("averageSpeed", stat.get("avgSpeed", 0)), 0.0)
                if velo > 0 and usage > best_usage:
                    best_usage, best_velo = usage, velo
            result["avg_fastball_velo"] = best_velo
        except Exception:
            pass

        return result

    def pitcher_platoon_splits(self, pitcher_id: int) -> dict:
        result = {
            "vsL": {"baa": 0.250, "k": 22.0, "bb": 8.0, "hr": 3.0, "bf": 0},
            "vsR": {"baa": 0.250, "k": 22.0, "bb": 8.0, "hr": 3.0, "bf": 0},
        }
        for stats_type in ("statSplits", "careerStatSplits"):
            try:
                payload = self.person_stats(pitcher_id, group="pitching", stats=stats_type, sitCodes="vl,vr")
            except requests.RequestException:
                continue
            for block in payload.get("stats", []):
                for split in block.get("splits", []):
                    code = split.get("split", {}).get("code", "").lower()
                    stat = split.get("stat", {})
                    bf = int(stat.get("battersFaced", stat.get("plateAppearances", 0)) or 0)
                    if bf <= 0:
                        continue
                    parsed = {
                        "baa": safe_float(stat.get("avg"), 0.250),
                        "k": int(stat.get("strikeOuts", 0) or 0) / bf * 100,
                        "bb": int(stat.get("baseOnBalls", 0) or 0) / bf * 100,
                        "hr": int(stat.get("homeRuns", 0) or 0) / bf * 100,
                        "bf": bf,
                    }
                    if code == "vl":
                        result["vsL"] = parsed
                    elif code == "vr":
                        result["vsR"] = parsed
            if result["vsL"]["bf"] and result["vsR"]["bf"]:
                break
        return result

    def venue(self, venue_id: int | None) -> dict:
        if not venue_id:
            return {}
        payload = self._get(f"/venues/{venue_id}")
        venues = payload.get("venues", [])
        return venues[0] if venues else {}

    def search_person(self, name: str) -> dict | None:
        payload = self._get("/people/search", {"names": name})
        people = payload.get("people", [])
        return people[0] if people else None

    @staticmethod
    def _first_stat(payload: dict) -> dict:
        stats = payload.get("stats", [])
        if not stats:
            return {}
        splits = stats[0].get("splits", [])
        if not splits:
            return {}
        return splits[0].get("stat", {})

    @staticmethod
    def default_pitcher() -> dict:
        split = {"baa": 0.250, "k": 22.0, "bb": 8.0, "hr": 3.0, "bf": 0}
        return {
            "id": None, "name": "TBD", "hand": "R", "era": 4.20, "whip": 1.32, "k_rate": 0.22,
            "games_started": 0, "innings": 0.0, "proj_k": 5.0, "proj_er": 2.5,
            "avg_vs_left": 0.250, "avg_vs_right": 0.250,
            "splits": {"vsL": split.copy(), "vsR": split.copy()},
            "fatigue_factor": 1.0, "is_high_fatigue": False, "avg_fastball_velo": 0.0,
            "stats_source": "league-fallback",
        }