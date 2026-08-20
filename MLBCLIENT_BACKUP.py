# mlb_client.py — added hitter_vs_pitcher_history() for the Matchup Verifier
# (Issue #2). Everything else is unchanged from your working version.

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import requests

from config import settings
from math_engine import safe_float


class MLBClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "trckr21-mlb-quant/1.0"}
        )

    def _get(self, path: str, params: dict | None = None) -> dict:
        base_url = settings.mlb_api_base.strip().rstrip("/")

        if "[" in base_url and "]" in base_url:
            base_url = base_url.split("(")[-1].rstrip(")")

        try:
            response = self.session.get(
                f"{base_url}{path}",
                params=params,
                timeout=settings.request_timeout,
            )
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
                "hydrate": (
                    "lineups,probablePitcher(note),venue,"
                    "team,game(content(summary))"
                ),
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
        payload = self._get(
            f"/teams/{team_id}/roster",
            {"rosterType": "active"},
        )
        return payload.get("roster", [])

    def person_stats(
        self,
        person_id: int,
        group: str,
        stats: str = "season",
        season: int | None = None,
        **params,
    ) -> dict:
        query = {
            "stats": stats,
            "group": group,
            "season": season or settings.season,
            **params,
        }
        return self._get(
            f"/people/{person_id}/stats",
            query,
        )

    def team_stats(
        self,
        team_id: int,
        group: str,
        season: int | None = None,
    ) -> dict:
        payload = self._get(
            f"/teams/{team_id}/stats",
            {
                "stats": "season",
                "group": group,
                "season": season or settings.season,
            },
        )
        return self._first_stat(payload)

    def player_season_hitting(self, player_id: int) -> dict:
        payload = self.person_stats(
            player_id,
            group="hitting",
            stats="season",
        )
        return self._first_stat(payload)

    def player_recent_hitting(
        self,
        player_id: int,
        target_date: str,
        days: int = 14,
    ) -> dict:
        try:
            end_date = date.fromisoformat(str(target_date))
        except (ValueError, TypeError):
            end_date = date.today()

        start_date = end_date - timedelta(days=days)

        payload = self.person_stats(
            player_id,
            group="hitting",
            stats="byDateRange",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
        )
        return self._first_stat(payload)

    def player_game_log(
        self,
        player_id: int,
        target_date: str,
        days: int = 50,
    ) -> list[dict]:
        try:
            end_date = date.fromisoformat(str(target_date))
        except (ValueError, TypeError):
            end_date = date.today()

        start_date = end_date - timedelta(days=days)

        payload = self.person_stats(
            player_id,
            group="hitting",
            stats="gameLog",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
        )

        stats = payload.get("stats", [])
        return stats[0].get("splits", []) if stats else []

    def hitter_platoon_split(
        self,
        player_id: int,
        pitcher_hand: str,
    ) -> dict:
        sit_code = "vl" if pitcher_hand == "L" else "vr"

        payload = self.person_stats(
            player_id,
            group="hitting",
            stats="statSplits",
            sitCodes=sit_code,
        )

        return self._first_stat(payload)

    # --- NEW: head-to-head batter-vs-pitcher history (Issue #2) ---
    # Ports your original fetch_real_h2h_history(). MLB's vsPlayer stat
    # type returns career at-bats a batter has logged against one specific
    # pitcher — used by the Matchup Verifier's advantage-score gauge.
    def hitter_vs_pitcher_history(
        self,
        batter_id: int,
        pitcher_id: int,
    ) -> list[dict]:
        payload = self.person_stats(
            batter_id,
            group="hitting",
            stats="vsPlayer",
            opposingPlayerId=pitcher_id,
        )
        stats = payload.get("stats", [])
        return stats[0].get("splits", []) if stats else []

    def pitcher_profile(self, pitcher_id: int | None) -> dict:
        if not pitcher_id:
            return self.default_pitcher()

        person = self.person(pitcher_id)
        hand = person.get("pitchHand", {}).get("code", "R")

        season_payload = self.person_stats(
            pitcher_id,
            group="pitching",
            stats="season",
        )
        stat = self._first_stat(season_payload)

        innings = safe_float(stat.get("inningsPitched"), 0.0)
        games = int(stat.get("gamesPitched", 0) or 0)
        strikeouts = int(stat.get("strikeOuts", 0) or 0)
        batters_faced = int(stat.get("battersFaced", 0) or 0)

        avg_ip = innings / games if games else 5.5
        k_per_ip = strikeouts / innings if innings else 0.9

        projected_k = k_per_ip * avg_ip
        projected_er = (
            safe_float(stat.get("era"), 4.20) / 9.0
        ) * avg_ip

        splits = self.pitcher_platoon_splits(pitcher_id)

        return {
            "id": pitcher_id,
            "name": person.get("fullName", "Unknown"),
            "hand": hand,
            "era": safe_float(stat.get("era"), 4.20),
            "whip": safe_float(stat.get("whip"), 1.32),
            "k_rate": (
                strikeouts / batters_faced
                if batters_faced
                else 0.22
            ),
            "games_started": int(stat.get("gamesStarted", 0) or 0),
            "innings": innings,
            "proj_k": round(projected_k, 2),
            "proj_er": round(projected_er, 2),
            "avg_vs_left": splits["vsL"]["baa"],
            "avg_vs_right": splits["vsR"]["baa"],
            "splits": splits,
            "stats_source": (
                "current-season" if innings > 0 else "league-fallback"
            ),
        }

    def pitcher_platoon_splits(self, pitcher_id: int) -> dict:
        result = {
            "vsL": {"baa": 0.250, "k": 22.0, "bb": 8.0, "hr": 3.0, "bf": 0},
            "vsR": {"baa": 0.250, "k": 22.0, "bb": 8.0, "hr": 3.0, "bf": 0},
        }

        for stats_type in ("statSplits", "careerStatSplits"):
            try:
                payload = self.person_stats(
                    pitcher_id,
                    group="pitching",
                    stats=stats_type,
                    sitCodes="vl,vr",
                )
            except requests.RequestException:
                continue

            for block in payload.get("stats", []):
                for split in block.get("splits", []):
                    code = split.get("split", {}).get("code", "").lower()
                    stat = split.get("stat", {})
                    bf = int(
                        stat.get(
                            "battersFaced",
                            stat.get("plateAppearances", 0),
                        )
                        or 0
                    )
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
            "id": None,
            "name": "TBD",
            "hand": "R",
            "era": 4.20,
            "whip": 1.32,
            "k_rate": 0.22,
            "games_started": 0,
            "innings": 0.0,
            "proj_k": 5.0,
            "proj_er": 2.5,
            "avg_vs_left": 0.250,
            "avg_vs_right": 0.250,
            "splits": {"vsL": split.copy(), "vsR": split.copy()},
            "stats_source": "league-fallback",
        }