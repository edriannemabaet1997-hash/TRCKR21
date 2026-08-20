from __future__ import annotations

from difflib import SequenceMatcher

import requests

from config import settings


class OddsClient:
    def __init__(self) -> None:
        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(settings.odds_api_key)

    def events(
        self,
        markets: str,
    ) -> list[dict]:
        if not self.enabled:
            return []

        try:
            response = self.session.get(
                f"{settings.odds_api_base}/odds",
                params={
                    "apiKey": settings.odds_api_key,
                    "regions": "us",
                    "markets": markets,
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                },
                timeout=settings.request_timeout,
            )

            if response.status_code in (401, 422, 429):
                return []

            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return []

    def player_prop_index(self) -> dict:
        market_keys = (
            "batter_hits,"
            "batter_runs_scored,"
            "batter_home_runs,"
            "batter_rbis,"
            "player_hits,"
            "player_runs,"
            "player_home_runs,"
            "player_rbis,"
            "pitcher_strikeouts,"
            "pitcher_walks,"
            "pitcher_earned_runs"
        )
        events = self.events(market_keys)

        aliases = {
            "batter_hits": "hits",
            "player_hits": "hits",
            "batter_runs_scored": "runs",
            "player_runs": "runs",
            "batter_home_runs": "homeruns",
            "player_home_runs": "homeruns",
            "batter_rbis": "rbi",
            "player_rbis": "rbi",
            "pitcher_strikeouts": "k",
            "pitcher_walks": "bb",
            "pitcher_earned_runs": "er",
        }

        index: dict[tuple[str, str], dict] = {}

        for event in events:
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    prop_type = aliases.get(market.get("key"))
                    if not prop_type:
                        continue

                    grouped: dict[str, dict] = {}

                    for outcome in market.get("outcomes", []):
                        player = (
                            outcome.get("description")
                            or outcome.get("participant")
                            or ""
                        ).strip()
                        if not player:
                            continue

                        bucket = grouped.setdefault(
                            player,
                            {
                                "over": None,
                                "under": None,
                                "point": outcome.get("point"),
                            },
                        )

                        side = str(outcome.get("name", "")).lower()
                        if side == "over":
                            bucket["over"] = outcome.get("price")
                        elif side == "under":
                            bucket["under"] = outcome.get("price")

                    for player, quote in grouped.items():
                        key = (self._normalize(player), prop_type)
                        if key not in index:
                            index[key] = quote

        return index

    def moneyline_index(self) -> list[dict]:
        return self.events("h2h")

    def find_player_prop(
        self,
        prop_index: dict,
        player_name: str,
        prop_type: str,
    ) -> dict | None:
        normalized = self._normalize(player_name)
        exact = prop_index.get((normalized, prop_type))
        if exact:
            return exact

        best_match = None
        best_score = 0.0

        for (candidate, candidate_prop), quote in prop_index.items():
            if candidate_prop != prop_type:
                continue

            score = SequenceMatcher(
                None,
                normalized,
                candidate,
            ).ratio()

            if score > best_score:
                best_match = quote
                best_score = score

        return best_match if best_score >= 0.88 else None

    def find_moneyline_game(
        self,
        events: list[dict],
        away_name: str,
        home_name: str,
    ) -> dict[str, int | float | None]:
        for event in events:
            event_away = event.get("away_team", "")
            event_home = event.get("home_team", "")

            if not (
                self._team_match(event_away, away_name)
                and self._team_match(event_home, home_name)
            ):
                continue

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue

                    prices: dict[str, int | float | None] = {"away": None, "home": None}
                    for outcome in market.get("outcomes", []):
                        outcome_name = outcome.get("name", "")
                        price = outcome.get("price")

                        if self._team_match(outcome_name, away_name):
                            prices["away"] = price
                        elif self._team_match(outcome_name, home_name):
                            prices["home"] = price

                    return prices

        return {"away": None, "home": None}

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(
            value.lower()
            .replace(".", "")
            .replace("-", " ")
            .split()
        )

    def _team_match(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_norm = self._normalize(left)
        right_norm = self._normalize(right)
        if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
            return True
        return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.75