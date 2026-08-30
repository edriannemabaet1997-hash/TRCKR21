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

import time
from datetime import date, timedelta
from threading import Lock
from typing import Any, Callable, Optional

import requests

from config import settings
from math_engine import safe_float

# ---------------------------------------------------------------------------
# CONSOLIDATION (2026-08-29) — pitch-code map + resolver, ported verbatim
# from generate_projections.py. Both the season-summary `pitchArsenal`
# endpoint and the live-feed `playEvents[].details.type` object carry a
# short `code` field from this same MLB dictionary; joining on `code` first
# (falling back to a normalized free-text match only when `code` is
# missing) is what avoids the "Four-seam FB" vs "Four-Seam Fastball"
# mismatch the original v2.3 script had.
# ---------------------------------------------------------------------------
PITCH_CODE_NAMES = {
    "FF": "Four-Seam Fastball", "FA": "Fastball", "FT": "Two-Seam Fastball",
    "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
    "SV": "Slurve", "CU": "Curveball", "KC": "Knuckle Curve", "CS": "Slow Curve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball", "SC": "Screwball",
    "KN": "Knuckleball", "EP": "Eephus", "PO": "Pitchout", "IN": "Intentional Ball",
    "AB": "Automatic Ball", "UN": "Unknown", "NP": "No Pitch",
}
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


def _normalize_key(s: Optional[str]) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def resolve_pitch_code(code: Optional[str], description: Optional[str]) -> str:
    if code:
        c = code.strip().upper()
        if c in PITCH_CODE_NAMES:
            return c
    key = _normalize_key(description)
    return _NAME_TO_CODE.get(key, "UN")


def pitch_display_name(code: str, fallback_description: Optional[str] = None) -> str:
    return PITCH_CODE_NAMES.get(code, fallback_description or "Unknown")


# --- pitch-level lethality knobs (unchanged values from the script) -------
PLATE_HALF_WIDTH_FT = 0.708
HARD_HIT_THRESHOLD_MPH = 95.0
LETHALITY_LOOKBACK_STARTS = 5
BATTER_VULN_LOOKBACK_GAMES = 7

_SWING_DESCRIPTIONS = {
    "Swinging Strike", "Swinging Strike (Blocked)", "Foul", "Foul Tip", "Foul Bunt", "Missed Bunt",
    "In play, out(s)", "In play, no out", "In play, run(s)",
}
_WHIFF_DESCRIPTIONS = {"Swinging Strike", "Swinging Strike (Blocked)", "Missed Bunt"}
_IN_PLAY_DESCRIPTIONS = {"In play, out(s)", "In play, no out", "In play, run(s)"}
_STRIKEOUT_EVENT_TYPES = {"strikeout", "strikeout_double_play", "strikeout_triple_play"}


# ---------------------------------------------------------------------------
# FEATURE (2026-08-29) — bounded, thread-safe TTL cache. Replaces the plain
# @lru_cache(maxsize=256) that used to sit on person(): lru_cache has no
# expiry at all, so on a long-running server process a player's cached
# batSide/pitchHand/fullName would never refresh — harmless for 99% of the
# season, but wrong forever after the rare trade/callup/MLB-side data
# correction, until the process restarts. This cache is bounded (like
# lru_cache) AND time-based: an entry older than `ttl_seconds` is treated
# as a miss and refetched, so the data self-heals on its own within one TTL
# window instead of needing a manual cache-bust or a restart.
# ---------------------------------------------------------------------------
class _TTLCache:
    def __init__(self, maxsize: int, ttl_seconds: float) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = Lock()

    def get_or_set(self, key: Any, factory: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                expires_at, value = cached
                if now < expires_at:
                    return value
                # Expired — evict so a failed refetch below doesn't leave a
                # stale entry sitting behind a since-passed TTL.
                del self._store[key]

        value = factory()

        with self._lock:
            if key not in self._store and len(self._store) >= self._maxsize:
                # Bounded like lru_cache — drop the oldest-inserted entry
                # (dicts preserve insertion order in Python 3.7+) rather
                # than growing unbounded.
                oldest_key = next(iter(self._store), None)
                if oldest_key is not None:
                    self._store.pop(oldest_key, None)
            self._store[key] = (now + self._ttl, value)
        return value

    def invalidate(self, key: Any = None) -> None:
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)


class MLBClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "trckr21-mlb-quant/1.0"})
        # FIX (2026-08-29): the default requests.Session() ships with an
        # HTTPAdapter capped at pool_connections=10 / pool_maxsize=10 per
        # host. prediction_service builds a slate with a ThreadPoolExecutor
        # per game AND a nested ThreadPoolExecutor per hitter inside each
        # game (settings.max_workers=12 at both levels) — on a normal
        # ~15-game day that's up to ~100+ threads all hitting
        # statsapi.mlb.com through this ONE shared session at once. With
        # only 10 pooled connections, requests/urllib3 doesn't queue nicely;
        # it opens throwaway connections outside the pool (full TCP+TLS
        # handshake each time) and discards them, which is exactly the kind
        # of silent, invisible slowdown that makes a cold build blow past a
        # 20s client timeout on the first request of the day while looking
        # totally fine in a one-off manual test. Sizing the pool to match
        # the real concurrency, plus a small automatic retry/backoff for
        # transient 429/5xx responses (MLB's API does rate-limit under
        # bursts), fixes both the connection thrashing and the "one flaky
        # call eats the full request_timeout" tail latency.
        _retry = requests.adapters.Retry(
            total=2,
            backoff_factor=0.25,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        _adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=_retry,
        )
        self.session.mount("https://", _adapter)
        self.session.mount("http://", _adapter)
        self._live_feed_cache: dict[int, dict] = {}
        self._person_cache = _TTLCache(maxsize=256, ttl_seconds=settings.person_cache_ttl_seconds)

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

    def _get_absolute(self, url: str, params: dict | None = None) -> dict:
        try:
            response = self.session.get(url, params=params, timeout=settings.request_timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"🚨 MLB API ERROR: Can't Connect to {url}")
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

    def person(self, person_id: int) -> dict:
        return self._person_cache.get_or_set(person_id, lambda: self._fetch_person(person_id))

    def _fetch_person(self, person_id: int) -> dict:
        payload = self._get(f"/people/{person_id}")
        people = payload.get("people", [])
        return people[0] if people else {}

    def invalidate_person_cache(self, person_id: int | None = None) -> None:
        """Manual escape hatch alongside the TTL — e.g. a known trade/
        callup you don't want to wait out the TTL for. Clears one player
        when `person_id` is given, or the whole cache when omitted."""
        self._person_cache.invalidate(person_id)

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

    # ------------------------------------------------------------------
    # CONSOLIDATION (2026-08-29) — fetch methods ported from
    # generate_projections.py for /api/pitcher-props, /api/team-matchups
    # and /api/matchup-analyzer. Pure HTTP-shape fetches only; VMR/Poisson/
    # NB2 calibration and any derived proxies live in math_engine.py /
    # prediction_service.py, not here.
    # ------------------------------------------------------------------

    def pitcher_recent_pitching_log(self, pitcher_id: int, limit: int = 10) -> list[dict]:
        payload = self.person_stats(pitcher_id, group="pitching", stats="gameLog")
        stats = payload.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        return splits[-limit:]

    def pitcher_recent_game_pks(self, pitcher_id: int, limit: int = LETHALITY_LOOKBACK_STARTS) -> list[int]:
        splits = self.pitcher_recent_pitching_log(pitcher_id, limit=limit)
        return [s.get("game", {}).get("gamePk") for s in splits if s.get("game", {}).get("gamePk")]

    def team_recent_hitting_log(self, team_id: int, limit: int = 10) -> list[dict]:
        payload = self.team_stats_raw(team_id, group="hitting", stats="gameLog")
        stats = payload.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        return splits[-limit:]

    def linescore(self, game_pk: int) -> dict:
        return self._get(f"/game/{game_pk}/linescore")

    def boxscore(self, game_pk: int) -> dict:
        return self._get(f"/game/{game_pk}/boxscore")

    def live_feed(self, game_pk: int) -> dict:
        # In-process cache: teammates on the same lineup share most of their
        # recent games, so without this the live-feed call volume multiplies
        # by roughly 9x per lineup for no new data (same reasoning as the
        # original script's module-level _LIVE_FEED_CACHE).
        if game_pk in self._live_feed_cache:
            return self._live_feed_cache[game_pk]
        base = settings.mlb_api_base.strip().rstrip("/")
        v11_base = base[: -len("/v1")] + "/v1.1" if base.endswith("/v1") else base
        feed = self._get_absolute(f"{v11_base}/game/{game_pk}/feed/live")
        self._live_feed_cache[game_pk] = feed
        return feed

    def pitch_arsenal(self, pitcher_id: int) -> list[dict]:
        payload = self.person_stats(pitcher_id, group="pitching", stats="pitchArsenal")
        stats = payload.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []

        pitches: list[dict] = []
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

    def pitcher_pitch_lethality(self, pitcher_id: int, game_pks: list[int]) -> dict[str, dict]:
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
                feed = self.live_feed(game_pk)
            except Exception:
                continue

            plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
            for play in plays:
                matchup = play.get("matchup", {})
                if matchup.get("pitcher", {}).get("id") != pitcher_id:
                    continue

                events = play.get("playEvents", [])
                pitch_events = [e for e in events if e.get("isPitch")]
                result_event_type = play.get("result", {}).get("eventType", "")
                entering_strikes = 0

                for i, event in enumerate(pitch_events):
                    details = event.get("details", {})
                    type_obj = details.get("type", {}) or {}
                    code = resolve_pitch_code(type_obj.get("code"), type_obj.get("description"))
                    display_names.setdefault(code, pitch_display_name(code, type_obj.get("description")))

                    description_text = details.get("description", "")
                    b = _bucket(code)
                    b["pitches"] += 1

                    is_swing = description_text in _SWING_DESCRIPTIONS
                    is_whiff = description_text in _WHIFF_DESCRIPTIONS
                    if is_swing:
                        b["swings"] += 1
                    if is_whiff:
                        b["whiffs"] += 1

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

                    is_last_pitch_of_pa = i == len(pitch_events) - 1
                    if entering_strikes == 2:
                        b["twoStrikePitches"] += 1
                        if is_last_pitch_of_pa and result_event_type in _STRIKEOUT_EVENT_TYPES:
                            b["twoStrikeKs"] += 1

                    post_count = event.get("count", {})
                    post_strikes = post_count.get("strikes")
                    if post_strikes is not None:
                        entering_strikes = min(int(post_strikes), 2)

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

    def batter_recent_game_pks(self, batter_id: int, limit: int = BATTER_VULN_LOOKBACK_GAMES) -> list[int]:
        payload = self.person_stats(batter_id, group="hitting", stats="gameLog")
        stats = payload.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        pks = [s.get("game", {}).get("gamePk") for s in splits if s.get("game", {}).get("gamePk")]
        return pks[-limit:]

    def batter_pitch_vulnerability(self, batter_id: int, game_pks: list[int]) -> dict[str, dict]:
        buckets: dict[str, dict] = {}
        display_names: dict[str, str] = {}

        def _bucket(code: str) -> dict:
            return buckets.setdefault(code, {
                "pitches": 0, "swings": 0, "whiffs": 0,
                "outOfZonePitches": 0, "outOfZoneSwings": 0,
                "battedBalls": 0, "hardHitBalls": 0,
            })

        for game_pk in game_pks:
            try:
                feed = self.live_feed(game_pk)
            except Exception:
                continue

            plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
            for play in plays:
                matchup = play.get("matchup", {})
                if matchup.get("batter", {}).get("id") != batter_id:
                    continue

                for event in play.get("playEvents", []):
                    if not event.get("isPitch"):
                        continue
                    details = event.get("details", {})
                    type_obj = details.get("type", {}) or {}
                    code = resolve_pitch_code(type_obj.get("code"), type_obj.get("description"))
                    display_names.setdefault(code, pitch_display_name(code, type_obj.get("description")))

                    description_text = details.get("description", "")
                    b = _bucket(code)
                    b["pitches"] += 1

                    is_swing = description_text in _SWING_DESCRIPTIONS
                    is_whiff = description_text in _WHIFF_DESCRIPTIONS
                    if is_swing:
                        b["swings"] += 1
                    if is_whiff:
                        b["whiffs"] += 1

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
                "hardHitPct": round(b["hardHitBalls"] / b["battedBalls"] * 100, 1) if b["battedBalls"] >= 3 else None,
                "confidenceTier": ("high" if b["pitches"] >= 10 else "medium" if b["pitches"] >= 6 else "low"),
            }
        return results

    def probable_lineup(self, game_pk: int, team_id: int) -> tuple[list[dict], bool]:
        try:
            box = self.boxscore(game_pk)
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
        except Exception:
            pass

        try:
            roster = self.team_roster(team_id)
            batters = [
                {"id": entry["person"]["id"], "name": entry["person"]["fullName"]}
                for entry in roster
                if entry.get("position", {}).get("code") != "1"
            ]
            return batters[:9], False
        except Exception:
            return [], False

    def batter_vs_pitcher_slash(self, batter_id: int, pitcher_id: int) -> dict:
        empty = {"pa": 0, "ab": 0, "h": 0, "hr": 0, "bb": 0, "so": 0, "avg": ".000", "obp": ".000", "slg": ".000", "ops": 0.0}
        splits = self.hitter_vs_pitcher_history(batter_id, pitcher_id)
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

    def batter_platoon_slash(self, batter_id: int) -> dict:
        empty_side = {"avg": ".000", "ops": 0.0, "pa": 0, "hr": 0}
        fallback = {"vsRHP": dict(empty_side), "vsLHP": dict(empty_side)}
        try:
            payload = self.person_stats(batter_id, group="hitting", stats="statSplits", sitCodes="vr,vl")
            out = dict(fallback)
            for block in payload.get("stats", []):
                for split in block.get("splits", []):
                    code = (split.get("split", {}) or {}).get("code")
                    stat = split.get("stat", {})
                    slash = {
                        "avg": stat.get("avg") or ".000",
                        "ops": safe_float(stat.get("ops"), 0.0),
                        "pa": int(stat.get("plateAppearances", 0) or 0),
                        "hr": int(stat.get("homeRuns", 0) or 0),
                    }
                    if code == "vr":
                        out["vsRHP"] = slash
                    elif code == "vl":
                        out["vsLHP"] = slash
            return out
        except Exception:
            return fallback

    def batter_recent_form_slash(self, batter_id: int, days: int = 14) -> dict:
        fallback = {"avg": ".000", "ops": 0.0, "pa": 0, "hr": 0}
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            payload = self.person_stats(
                batter_id, group="hitting", stats="byDateRange",
                startDate=start_date.isoformat(), endDate=end_date.isoformat(),
            )
            stat = self._first_stat(payload)
            if not stat:
                return fallback
            return {
                "avg": stat.get("avg") or ".000",
                "ops": safe_float(stat.get("ops"), 0.0),
                "pa": int(stat.get("plateAppearances", 0) or 0),
                "hr": int(stat.get("homeRuns", 0) or 0),
            }
        except Exception:
            return fallback

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