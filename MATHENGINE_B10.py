from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

LEAGUE_AVG_XWOBA = 0.320
LEAGUE_AVG_XBA = 0.250
LEAGUE_AVG_BARREL = 8.0
LEAGUE_AVG_HARDHIT = 40.0
LEAGUE_AVG_WHIFF = 25.0

LEAGUE_AVG_BA = 0.245
LEAGUE_AVG_RUN_RATE = 0.122
LEAGUE_AVG_HR_RATE = 0.034
LEAGUE_AVG_RBI_RATE = 0.108
LEAGUE_AVG_OBP = 0.315
LEAGUE_AVG_SLG = 0.410
LEAGUE_AVG_ERA = 4.20
LEAGUE_AVG_XERA = 4.15
LEAGUE_AVG_BABIP = 0.295
LEAGUE_AVG_ISO = LEAGUE_AVG_SLG - LEAGUE_AVG_BA

K_HITS = 60
K_RUNS = 100
K_RBI = 100
K_HR = 300

# ---------------------------------------------------------------------------
# CONSOLIDATION (2026-08-29) — shrinkage anchor for the Team Matchups VMR/
# Poisson-NB2 engine. K_TEAM_RUNS=6 is "moderate": light enough that the
# last-10-game empirical mu still dominates for a team with a full sample,
# heavy enough that the anchor pulls the mean into line with
# calculate_team_xruns_v2 (the same number the Moneylines tab's Pythagenpat
# win prob is built from) when the sample is thin or noisy. See
# calibrate_count_market() below — this only ever touches the mean fed into
# whichever distribution the VMR gate already picked; it never touches the
# gate itself.
K_TEAM_RUNS = 6

# -110 american odds vig-implied probability — the flat "market" reference
# point generate_projections.py's board used for edge%/Kelly on
# pitcher-props & team-matchup markets (moneyline/props from the live odds
# feed already use remove_vig() against real two-sided book odds instead).
MARKET_IMPLIED_PROB = 110.0 / 210.0
DECIMAL_B = 100.0 / 110.0

# Matchup Analyzer H2H credibility shrink (ported from shrink_h2h_ops /
# shrink_proxy_xba_xslg) — same shape as shrink_rate(), just expressed via
# credibility = pa / (pa + STABILIZATION_PA_H2H) for display. Reused through
# shrink_rate() directly in prediction_service.py; no separate function.
STABILIZATION_PA_H2H = 100
LEAGUE_AVG_OPS_FALLBACK = 0.710

MIN_PROB = 0.02
MAX_PROB = 0.95
HOME_FIELD_ADV_RUNS = 0.25


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-", ".---"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def american_to_prob(odds: int | float | None) -> float | None:
    if odds is None:
        return None
    value = float(odds)
    if value == 0:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def prob_to_american(probability: float) -> int:
    probability = clamp(probability, 0.0001, 0.9999)
    if probability >= 0.5:
        return -round((probability / (1.0 - probability)) * 100.0)
    return round(((1.0 - probability) / probability) * 100.0)


def remove_vig(odds_a: int | None, odds_b: int | None) -> tuple[float | None, float | None]:
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    if pa is None or pb is None:
        return None, None
    total = pa + pb
    if total <= 0:
        return None, None
    return pa / total, pb / total


def shrink_rate(observed: float, sample_size: int, league_avg: float, k_factor: int) -> float:
    if sample_size <= 0:
        return league_avg
    return (observed * sample_size + league_avg * k_factor) / (sample_size + k_factor)


# ---------------------------------------------------------------------------
# PA TABLE — this is the table that lives inside process_abi_single_hitter
# in the original backend, i.e. the one that actually drives the binomial
# hit-probability exponent. The original also had a second, DIFFERENT table
# inside the tab_hit UI post-processing block ({1:4.6,2:4.5,3:4.4,4:4.3,
# 5:4.2,6:4.1,7:4.0,8:4.0,9:4.0}) — but that second table only ever
# overwrote a display column ("Projected PA") for the UI; it was never fed
# back into the actual Hit 1+ % math. This is the one that counts.
# ---------------------------------------------------------------------------
def projected_pa_from_order(order: int | None) -> float:
    if order in (1, 2):
        return 4.6
    if order in (3, 4):
        return 4.2
    if order == 5:
        return 3.9
    if order == 6:
        return 3.6
    if order in (7, 8, 9):
        return 3.2
    return 3.8


def analyze_pitcher_split(pitcher_hand: str, avg_vs_left: float, avg_vs_right: float) -> str:
    if pitcher_hand == "R":
        same_side_avg, opposite_side_avg = avg_vs_right, avg_vs_left
    else:
        same_side_avg, opposite_side_avg = avg_vs_left, avg_vs_right
    difference = same_side_avg - opposite_side_avg
    if abs(difference) < 0.015:
        return "NEUTRAL"
    if difference > 0.02:
        return "REVERSE"
    return "NORMAL"


def apply_split_effect(base_probability: float, batter_hand: str, pitcher_hand: str, pitcher_type: str) -> float:
    same_side = batter_hand == pitcher_hand
    modifier = 1.0
    if pitcher_type == "NORMAL":
        modifier = 0.93 if same_side else 1.07
    elif pitcher_type == "REVERSE":
        modifier = 1.07 if same_side else 0.93
    return base_probability * modifier


# ---------------------------------------------------------------------------
# wOBA — ported verbatim from calculate_woba_from_stats(). Used to build the
# 5/10/15-game recent-form buckets from raw game-log counting stats, instead
# of a plain hits/AB average.
# ---------------------------------------------------------------------------
def calculate_woba_from_stats(stat: dict, fallback: float) -> float:
    try:
        ab = safe_float(stat.get("atBats"), 0)
        bb = safe_float(stat.get("baseOnBalls"), 0)
        sf = safe_float(stat.get("sacFlies"), 0)
        h = safe_float(stat.get("hits"), 0)
        d2b = safe_float(stat.get("doubles"), 0)
        d3b = safe_float(stat.get("triples"), 0)
        hr = safe_float(stat.get("homeRuns"), 0)
        singles = h - (d2b + d3b + hr)
        denominator = ab + bb + sf
        if denominator < 3:
            return fallback
        numerator = (0.69 * bb) + (0.88 * singles) + (1.24 * d2b) + (1.56 * d3b) + (2.05 * hr)
        return numerator / denominator
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Home/Away scoring factor — ported from get_team_home_away_scoring_factor().
# Pure function here; the mlb_client fetch supplies raw runs/hits.
# ---------------------------------------------------------------------------
def parse_innings_pitched(value) -> float:
    """MLB's `inningsPitched` field uses baseball's thirds notation, not
    decimal — e.g. "6.1" means 6 and 1/3 innings (6.333...), "6.2" means 6
    and 2/3 (6.667...), NOT 6.1 / 6.2 innings. safe_float() alone (as used
    elsewhere in this codebase for quick ERA/K-rate ratios, where the ~0.3
    IP error washes out) is NOT safe when innings are being SUMMED across
    many appearances, like the rolling bullpen-fatigue workload below —
    the rounding error compounds. This converts to true fractional innings.
    """
    ip = safe_float(value, 0.0)
    whole = math.floor(ip)
    remainder = round(ip - whole, 1)
    if remainder >= 0.2:
        frac = 2.0 / 3.0
    elif remainder >= 0.1:
        frac = 1.0 / 3.0
    else:
        frac = 0.0
    return whole + frac


# ---------------------------------------------------------------------------
# TASK 1 (2026-08-30) — weather_mult, sourced from Open-Meteo (open-meteo.com,
# free / no API key). Pure conversion function — the actual HTTP fetch (venue
# coordinates via mlb_client.venue(), forecast via weather_client.WeatherClient)
# lives in the client modules, same house convention as everything else here
# (fetches in *_client.py, math in math_engine.py).
#
# Simple, bounded model per spec: warmer air is less dense so batted balls
# carry farther (small temperature term); wind blowing OUT toward center
# field helps scoring, wind blowing IN suppresses it (directional wind term,
# only applied when the venue's home-plate orientation, `park_azimuth_deg`,
# is actually known — see mlb_client.venue()'s `location.azimuthAngle`).
# Total effect is capped at +/-10%, deliberately modest relative to the
# ~4.30 base_run_projection this ultimately multiplies in
# calculate_team_xruns_v2.
# ---------------------------------------------------------------------------
WEATHER_NEUTRAL_TEMP_F = 70.0
WEATHER_TEMP_SPAN_F = 30.0
WEATHER_TEMP_MAX_EFFECT = 0.03
WEATHER_WIND_MAX_MPH = 20.0
WEATHER_WIND_MAX_EFFECT = 0.05
WEATHER_MULT_MIN = 0.90
WEATHER_MULT_MAX = 1.10


def weather_scoring_multiplier(
    temperature_f: float,
    wind_speed_mph: float = 0.0,
    wind_direction_deg: float | None = None,
    park_azimuth_deg: float | None = None,
) -> float:
    # --- Temperature term: linear around a 70F neutral point, capped ---
    temp_component = clamp(
        (temperature_f - WEATHER_NEUTRAL_TEMP_F) / WEATHER_TEMP_SPAN_F * WEATHER_TEMP_MAX_EFFECT,
        -WEATHER_TEMP_MAX_EFFECT, WEATHER_TEMP_MAX_EFFECT,
    )

    # --- Wind term: only signed (in vs out) when we know which way the
    # park faces. Open-Meteo's wind_direction_deg is the compass bearing
    # the wind is blowing FROM; park_azimuth_deg is the home-plate ->
    # center-field bearing the park faces TOWARD. cos(0) = wind blowing
    # straight out to center (boost), cos(180) = blowing straight in
    # (penalty), scaled by how strong the wind is (capped at
    # WEATHER_WIND_MAX_MPH for the purposes of this scale). ---
    wind_component = 0.0
    if park_azimuth_deg is not None and wind_direction_deg is not None and wind_speed_mph > 0:
        blowing_toward_deg = (wind_direction_deg + 180.0) % 360.0
        angle_diff = abs((blowing_toward_deg - park_azimuth_deg + 180.0) % 360.0 - 180.0)
        directional_factor = math.cos(math.radians(angle_diff))
        speed_scale = clamp(wind_speed_mph / WEATHER_WIND_MAX_MPH, 0.0, 1.0)
        wind_component = directional_factor * speed_scale * WEATHER_WIND_MAX_EFFECT

    return clamp(1.0 + temp_component + wind_component, WEATHER_MULT_MIN, WEATHER_MULT_MAX)


# ---------------------------------------------------------------------------
# TASK 2 (2026-08-30) — bullpen_fatigue_mult, sourced from a rolling 2-3 day
# team-level relief-innings workload (MLB Stats API only — schedule +
# boxscore, both already used elsewhere in mlb_client.py; no new paid data
# source, see mlb_client.team_bullpen_relief_innings()).
#
# BULLPEN_FATIGUE_BASELINE_IP is a rough "normal" trailing-3-day relief
# workload for an MLB team (~2.5 IP/day of bullpen usage is a reasonable
# league-wide rule of thumb). Innings above that baseline mean the
# opposing bullpen has been worked harder than usual recently — this
# multiplier is applied to the OFFENSE facing that bullpen (bullpen_era_a/b
# in poisson_monte_carlo_win_prob already carries the OPPOSING team's
# bullpen), so more relief innings -> a small boost to the batting team's
# expected runs. Capped at the same order of magnitude as the existing
# ERA-based bullpen_mult in prediction_service.py's Stage A (clamp
# 0.95-1.06) so this new signal nudges rather than dominates.
# ---------------------------------------------------------------------------
BULLPEN_FATIGUE_BASELINE_IP = 7.5
BULLPEN_FATIGUE_IP_STEP = 0.01
BULLPEN_FATIGUE_MULT_MIN = 0.94
BULLPEN_FATIGUE_MULT_MAX = 1.06


def bullpen_fatigue_multiplier(relief_innings_last_n_days: float) -> float:
    delta = relief_innings_last_n_days - BULLPEN_FATIGUE_BASELINE_IP
    return clamp(1.0 + (delta * BULLPEN_FATIGUE_IP_STEP), BULLPEN_FATIGUE_MULT_MIN, BULLPEN_FATIGUE_MULT_MAX)


def home_away_scoring_factor(runs: float, hits: float) -> float:
    if hits <= 0:
        return 1.0
    density = runs / hits
    if density > 0.65:
        return 1.04
    if density < 0.45:
        return 0.96
    return 1.0


# ---------------------------------------------------------------------------
# Ahead-in-count bias overlay — ported from the "PILLAR 4" block inside
# process_abi_single_hitter.
# ---------------------------------------------------------------------------
def ahead_in_count_boost(ahead_avg: float | None, gen_ba: float) -> float:
    if ahead_avg is None:
        return 0.0
    if ahead_avg > gen_ba * 1.10:
        return 0.04
    return 0.0


# ---------------------------------------------------------------------------
# Whiff% proxy — ported from fetch_all_mlb_data_packets_live_v66's
# calculated_whiff formula. Free-tier MLB Stats API has no real Statcast
# whiff%, so the original approximated it from K/PA + a flat 5-point bump.
# ---------------------------------------------------------------------------
def fastball_whiff_proxy(strikeouts: float, plate_appearances: float) -> float:
    if plate_appearances > 10:
        return round((strikeouts / plate_appearances * 100.0) + 5.0, 1)
    return 24.5


# ---------------------------------------------------------------------------
# BATAS 1 — velocity control (tab_hit post-processing). In the original this
# never fired because pitcher_avg_fb_velo was never actually populated
# (always read as 0.0). Wired to a real fetch now — see mlb_client's
# pitcher_fatigue_and_velocity().
# ---------------------------------------------------------------------------
def velocity_control_penalty(pitcher_velo: float, hitter_whiff_pct: float) -> tuple[float, bool]:
    if pitcher_velo > 96.5 and hitter_whiff_pct > 28.0:
        return -7.0, True
    return 0.0, False


# ---------------------------------------------------------------------------
# The OTHER velocity mechanism (tab_hit "PITCHER MOD CALCULATION") — separate
# from BATAS 1, multiplicative, applied at the very end alongside quality_mod.
# ---------------------------------------------------------------------------
def pitcher_velocity_mod(pitcher_velo: float) -> float:
    if pitcher_velo > 95.0:
        return -0.05
    if 0 < pitcher_velo <= 92.0:
        return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# BATAS 2 — BABIP regression penalty, expressed in percentage POINTS
# (matches the original's `prob -= X.0` on a 0-100 scale).
# ---------------------------------------------------------------------------
def babip_regression_penalty_points(babip_14d: float) -> float:
    if 0.300 <= babip_14d <= 0.349:
        return -2.0
    if 0.350 <= babip_14d <= 0.379:
        return -12.0
    if babip_14d >= 0.380:
        return -20.0
    return 0.0


# ---------------------------------------------------------------------------
# BATAS 3 — lineup order bonus (multiplicative, on the 0-100 percentage
# scale, same as the original).
# ---------------------------------------------------------------------------
def lineup_order_bonus_mult(order: int | None) -> float:
    if order in (1, 2):
        return 0.92
    if order in (3, 4, 5):
        return 1.08
    return 1.0


# ---------------------------------------------------------------------------
# V8.9 quality modifier — ISO + BABIP combo, ported verbatim.
# ---------------------------------------------------------------------------
def hit_quality_modifier(babip_14d: float, iso: float) -> float:
    if babip_14d < 0.305 and iso > 0.200:
        return 0.08
    if babip_14d > 0.390 and iso < 0.120:
        return -0.06
    return 0.0


def compute_event_probability(
    base_rate: float,
    projected_pa: float,
    starter_quality: float,
    bullpen_quality: float,
    park_factor: float,
    platoon_mult: float = 1.0,
    iso_mult: float = 1.0,
    min_prob: float = MIN_PROB,
    max_prob: float = MAX_PROB,
) -> float:
    pa_starter = min(2.5, projected_pa)
    pa_bullpen = max(0.0, projected_pa - 2.5)
    rate_starter = base_rate * starter_quality * park_factor * platoon_mult * iso_mult
    rate_bullpen = base_rate * bullpen_quality * park_factor * platoon_mult * iso_mult
    rate_starter = clamp(rate_starter, 0.0, 1.0)
    rate_bullpen = clamp(rate_bullpen, 0.0, 1.0)
    prob_no_starter = math.pow(1.0 - rate_starter, pa_starter)
    prob_no_bullpen = math.pow(1.0 - rate_bullpen, pa_bullpen)
    final_prob = 1.0 - (prob_no_starter * prob_no_bullpen)
    return round(clamp(final_prob, min_prob, max_prob), 4)


# ---------------------------------------------------------------------------
# HR driver — power index from ISO-vs-league ratio blended with the
# hitter's own observed HR rate. Independent of the hits/runs/RBI rate path.
# ---------------------------------------------------------------------------
def hr_power_index(iso_val: float, hr_rate_observed: float) -> float:
    iso_ratio = iso_val / LEAGUE_AVG_ISO if LEAGUE_AVG_ISO > 0 else 1.0
    hr_ratio = hr_rate_observed / LEAGUE_AVG_HR_RATE if LEAGUE_AVG_HR_RATE > 0 else 1.0
    power_index = (iso_ratio * 0.6) + (hr_ratio * 0.4)
    return clamp(power_index, 0.6, 1.8)


# ---------------------------------------------------------------------------
# RBI driver — lineup protection factor. Team SLG relative to league average,
# weighted by how much RISP exposure that batting-order slot actually gets
# (3-6 see the most; 1-2 and 7-9 see less).
# ---------------------------------------------------------------------------
def lineup_protection_factor(order: int | None, team_slg: float) -> float:
    slg_ratio = clamp(team_slg / LEAGUE_AVG_SLG, 0.85, 1.20) if LEAGUE_AVG_SLG > 0 else 1.0
    if order in (3, 4, 5, 6):
        weight = 1.00
    elif order in (1, 2):
        weight = 0.55
    else:
        weight = 0.75
    protection_mult = 1.0 + ((slg_ratio - 1.0) * weight)
    return clamp(protection_mult, 0.80, 1.25)


# ---------------------------------------------------------------------------
# Runs driver — leadoff/table-setter vs power-hitter archetype. #1/#2 score
# mostly by reaching base ahead of the power bats (OBP-driven, ISO dampens
# it slightly); #3-6 score more off their own extra-base pop (ISO-driven).
# ---------------------------------------------------------------------------
def run_scoring_archetype_factor(order: int | None, iso_val: float) -> float:
    iso_ratio = iso_val / LEAGUE_AVG_ISO if LEAGUE_AVG_ISO > 0 else 1.0
    if order in (1, 2):
        return clamp(1.05 - (iso_ratio - 1.0) * 0.05, 0.95, 1.10)
    if order in (3, 4, 5, 6):
        return clamp(0.95 + (iso_ratio - 1.0) * 0.15, 0.85, 1.20)
    return 1.0


# ---------------------------------------------------------------------------
# Statcast proxies — free-tier MLB Stats API has no real xwoba/xba/barrel%/
# hardhit% (those live on Baseball Savant only). Same shape as
# fastball_whiff_proxy above: approximate from always-available box-score
# fields instead of leaving them null, and shrink toward the league average
# rather than trusting small samples at face value.
# ---------------------------------------------------------------------------
def batted_ball_air_rate(ground_outs: float, air_outs: float) -> float:
    total = ground_outs + air_outs
    if total <= 0:
        return 0.45
    return air_outs / total


def xwoba_proxy(computed_woba: float, sample_size: int) -> float:
    return round(shrink_rate(computed_woba, sample_size, LEAGUE_AVG_XWOBA, K_HITS), 3)


def xba_proxy(observed_avg: float, at_bats: int) -> float:
    return round(shrink_rate(observed_avg, at_bats, LEAGUE_AVG_XBA, K_HITS), 3)


def barrel_pct_proxy(iso_val: float, hr_rate_observed: float) -> float:
    power_index = hr_power_index(iso_val, hr_rate_observed)
    return round(clamp(LEAGUE_AVG_BARREL * power_index, 1.0, 25.0), 1)


def hardhit_pct_proxy(iso_val: float, air_rate: float) -> float:
    iso_ratio = iso_val / LEAGUE_AVG_ISO if LEAGUE_AVG_ISO > 0 else 1.0
    scale = clamp(0.5 + (iso_ratio - 1.0) * 0.5 + (air_rate - 0.45) * 0.3, 0.5, 2.0)
    return round(clamp(LEAGUE_AVG_HARDHIT * scale, 15.0, 65.0), 1)


def process_hit_prob(rate, pa, pitcher_era, bullpen_era, park, platoon, iso) -> float:
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    min_prob, max_prob = get_prob_bounds("hits")
    return compute_event_probability(
        rate, pa, starter_mult, bullpen_mult, park, platoon, iso, min_prob=min_prob, max_prob=max_prob
    )


def process_run_prob(rate, pa, pitcher_era, bullpen_era, park, platoon, iso, team_obp, order) -> float:
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    team_factor = clamp(team_obp / 0.315, 0.90, 1.10)
    archetype_mult = run_scoring_archetype_factor(order, iso)
    min_prob, max_prob = get_prob_bounds("runs")
    return compute_event_probability(
        rate * team_factor * archetype_mult,
        pa,
        starter_mult,
        bullpen_mult,
        park,
        platoon,
        iso,
        min_prob=min_prob,
        max_prob=max_prob,
    )


def process_hr_prob(
    rate,
    pa,
    pitcher_era,
    bullpen_era,
    park,
    platoon,
    iso_val,
    hr_rate_observed,
    fatigue_mult: float = 1.0,
    count_boost_mult: float = 1.0,
) -> float:
    # fatigue_mult / count_boost_mult mirror Stage A's Pillar 6 (late-inning
    # fatigue decay) and Pillar 4 (ahead-in-count bias overlay). Both used to
    # only touch the Hits pipeline; wired into HR here too, since power output
    # is arguably at least as fatigue/count sensitive as a generic hit. Default
    # 1.0 keeps this a no-op for any caller that doesn't pass them.
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    power_index = hr_power_index(iso_val, hr_rate_observed)
    min_prob, max_prob = get_prob_bounds("hr")
    adjusted_rate = rate * fatigue_mult * count_boost_mult
    return compute_event_probability(
        adjusted_rate, pa, starter_mult, bullpen_mult, park * 1.1, platoon, power_index,
        min_prob=min_prob, max_prob=max_prob,
    )


def process_rbi_prob(rate, pa, pitcher_era, bullpen_era, park, platoon, iso, team_obp, order, team_slg) -> float:
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    lineup_factor = clamp(team_obp / 0.315, 0.90, 1.15)
    protection_mult = lineup_protection_factor(order, team_slg)
    min_prob, max_prob = get_prob_bounds("rbi")
    return compute_event_probability(
        rate * lineup_factor * protection_mult,
        pa,
        starter_mult,
        bullpen_mult,
        park,
        platoon,
        iso,
        min_prob=min_prob,
        max_prob=max_prob,
    )


# ---------------------------------------------------------------------------
# Per-prop Statcast weight profiles — used by calibrate_with_statcast_full so
# the calibration doesn't treat every prop the same way. HR is a power/quality
# of contact event, so barrel% / hard-hit% should swing it harder. Hits is a
# bat-to-ball / contact event, so xBA / whiff% should swing it harder. Runs
# and RBI stay on the original neutral weighting (they're driven more by
# lineup/context factors elsewhere in the pipeline, not by this calibration
# step).
# ---------------------------------------------------------------------------
WEIGHT_PROFILE_NEUTRAL = {"xwoba": 1.0, "xba": 1.0, "barrel": 1.0, "hardhit": 1.0, "whiff": 1.0}
WEIGHT_PROFILE_HR = {"xwoba": 1.0, "xba": 0.6, "barrel": 1.8, "hardhit": 1.8, "whiff": 0.6}
WEIGHT_PROFILE_HITS = {"xwoba": 1.0, "xba": 1.6, "barrel": 0.5, "hardhit": 0.5, "whiff": 1.6}

# Maps the prop_type strings the frontend/API actually send (see
# index.html's data-prop attrs / State.ui.activeProp) to a weight profile.
PROP_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "homeruns": WEIGHT_PROFILE_HR,
    "hr": WEIGHT_PROFILE_HR,
    "homerun": WEIGHT_PROFILE_HR,
    "hits": WEIGHT_PROFILE_HITS,
    "hit": WEIGHT_PROFILE_HITS,
    "runs": WEIGHT_PROFILE_NEUTRAL,
    "run": WEIGHT_PROFILE_NEUTRAL,
    "rbi": WEIGHT_PROFILE_NEUTRAL,
    "rbis": WEIGHT_PROFILE_NEUTRAL,
}


def get_weight_profile(prop_type: str | None) -> dict[str, float]:
    if not prop_type:
        return WEIGHT_PROFILE_NEUTRAL
    return PROP_WEIGHT_PROFILES.get(prop_type.strip().lower(), WEIGHT_PROFILE_NEUTRAL)


# ---------------------------------------------------------------------------
# Per-prop probability floor/ceiling — HR is a low-frequency power event, so
# its ceiling should sit well below hits (a near-daily contact event). Runs
# and RBI sit in between since they're partly OBP/lineup-context driven
# rather than pure bat-to-ball or power events.
# ---------------------------------------------------------------------------
PROP_PROB_BOUNDS: dict[str, tuple[float, float]] = {
    "homeruns": (0.01, 0.55),
    "hr": (0.01, 0.55),
    "homerun": (0.01, 0.55),
    "hits": (0.15, 0.90),
    "hit": (0.15, 0.90),
    "runs": (0.02, 0.75),
    "run": (0.02, 0.75),
    "rbi": (0.02, 0.70),
    "rbis": (0.02, 0.70),
}


def get_prob_bounds(prop_type: str | None) -> tuple[float, float]:
    if not prop_type:
        return MIN_PROB, MAX_PROB
    return PROP_PROB_BOUNDS.get(prop_type.strip().lower(), (MIN_PROB, MAX_PROB))


def calibrate_with_statcast_full(
    original_prob: float,
    xwoba: float,
    xba: float,
    barrel: float,
    hard_hit: float,
    whiff: float,
    prop_type: str | None = None,
) -> dict:
    weights = get_weight_profile(prop_type)

    dev_xwoba = xwoba - LEAGUE_AVG_XWOBA
    w_xwoba = weights["xwoba"]
    mod_xwoba = clamp(dev_xwoba * 0.35 * w_xwoba, -0.03 * w_xwoba, 0.03 * w_xwoba)

    dev_xba = xba - LEAGUE_AVG_XBA
    w_xba = weights["xba"]
    mod_xba = clamp(dev_xba * 0.40 * w_xba, -0.03 * w_xba, 0.03 * w_xba)

    dev_barrel = barrel - LEAGUE_AVG_BARREL
    w_barrel = weights["barrel"]
    mod_barrel = clamp(dev_barrel * 0.0015 * w_barrel, -0.02 * w_barrel, 0.02 * w_barrel)

    dev_hardhit = hard_hit - LEAGUE_AVG_HARDHIT
    w_hardhit = weights["hardhit"]
    mod_hardhit = clamp(dev_hardhit * 0.0010 * w_hardhit, -0.02 * w_hardhit, 0.02 * w_hardhit)

    dev_whiff = whiff - LEAGUE_AVG_WHIFF
    w_whiff = weights["whiff"]
    mod_whiff = clamp(-(dev_whiff * 0.0015) * w_whiff, -0.02 * w_whiff, 0.02 * w_whiff)

    total_adjustment = clamp(mod_xwoba + mod_xba + mod_barrel + mod_hardhit + mod_whiff, -0.10, 0.10)
    min_prob, max_prob = get_prob_bounds(prop_type)
    final_prob = clamp(original_prob + total_adjustment, min_prob, max_prob)
    return {
        "original_prob": round(original_prob, 4),
        "mod_xwoba": round(mod_xwoba, 4),
        "mod_xba": round(mod_xba, 4),
        "mod_barrel": round(mod_barrel, 4),
        "mod_hardhit": round(mod_hardhit, 4),
        "mod_whiff": round(mod_whiff, 4),
        "total_adjustment": round(total_adjustment, 4),
        "final_prob": round(final_prob, 4),
    }


CONFIDENCE_RANK = {"low": 0, "med": 1, "high": 2}

# ---------------------------------------------------------------------------
# Sample-size confidence CAP — replaces the old hard "sample_size < 20 -> low"
# gate. Below SAMPLE_SIZE_FLOOR the data is too thin to trust at all, so we
# still force "low" outright. Between the floor and SAMPLE_SIZE_FULL_TRUST,
# the sample is thin-but-usable: it caps how high the edge can push
# confidence, instead of zeroing the edge signal out completely. At/above
# SAMPLE_SIZE_FULL_TRUST there's no cap — edge strength alone decides.
# Tune these three knobs (and the edge thresholds below) together if the
# resulting tiers still feel too conservative or too loose.
# ---------------------------------------------------------------------------
SAMPLE_SIZE_FLOOR = 5
SAMPLE_SIZE_FULL_TRUST = 20


def confidence_from_edge(edge: float | None, probability: float, sample_size: int) -> str:
    if sample_size < SAMPLE_SIZE_FLOOR:
        return "low"

    cap = "med" if sample_size < SAMPLE_SIZE_FULL_TRUST else "high"

    if edge is None:
        if probability >= 0.70:
            level = "high"
        elif probability >= 0.55:
            level = "med"
        else:
            level = "low"
    elif edge >= 0.08:
        level = "high"
    elif edge >= 0.03:
        level = "med"
    else:
        level = "low"

    if CONFIDENCE_RANK[level] > CONFIDENCE_RANK[cap]:
        level = cap
    return level


def pythagenpat_win_prob(runs_a: float, runs_b: float) -> tuple[float, float]:
    # Closed-form fallback / reference implementation. Superseded as the
    # live Moneylines source by poisson_monte_carlo_win_prob() below (see
    # prediction_service.py), kept here for parity checks and in case
    # anything else in the codebase still imports it directly.
    if runs_a <= 0 or runs_b <= 0:
        return 0.5, 0.5
    exponent = (runs_a + runs_b) ** 0.285
    prob_a = (runs_a**exponent) / (runs_a**exponent + runs_b**exponent)
    prob_a = clamp(prob_a, MIN_PROB, MAX_PROB)
    return round(prob_a, 4), round(1.0 - prob_a, 4)


# ---------------------------------------------------------------------------
# TASK 1 (2026-08-30) — Poisson Monte Carlo moneyline win%, replacing
# pythagenpat_win_prob() as the live source for the Moneylines tab.
#
# Same Poisson scoring model already used by poisson_tail_at_least() below —
# each team's calculate_team_xruns_v2() output is treated as a Poisson
# lambda — but instead of a closed-form Pythagenpat exponent, we draw
# n_sims independent samples per team (vectorized via numpy.random.poisson,
# runs in low-single-digit milliseconds even at 50k sims) and derive win%
# from the share of simulations where one side out-scores the other.
#
# Baseball can't end in a regulation tie (extra innings), but independent
# Poisson draws for two teams absolutely can land on the same integer — at
# these run-environments that happens on a non-trivial share of sims. Ties
# are broken with a fair coin flip per tied simulation. This is a
# simplification (it doesn't model extra-inning run environment separately
# from regulation), but it doesn't systematically favor either side over a
# large n_sims run, and it's the same shortcut most public Poisson-based
# MLB win-prob models use.
#
# Bonus: the combined (home+away) runs distribution comes back for free
# off the same sims array — this is exactly the "total runs" number the
# F5/full-game over/under lines care about, so no separate simulation is
# needed if/when that gets wired into the Team Matchups totals market.
#
# TASK 3 (2026-08-30) — the function no longer takes a bare runs_a/runs_b
# mean. It now takes the SAME quality-factor inputs calculate_team_xruns_v2
# already accepts (matchup strength, park factor, weather multiplier,
# opposing bullpen ERA, bullpen fatigue) for each side, and calls
# calculate_team_xruns_v2() ITSELF to derive each side's lambda before
# drawing a single sample. Two consequences, both intentional:
#   1. The simulation is provably sampling off the park/weather/bullpen-
#      adjusted mean, not a static league-average number — the adjustment
#      happens inside this function, immediately before the numpy call.
#   2. calculate_team_xruns_v2() now only runs ONCE per side per game
#      (inside here) instead of once in prediction_service.py and then
#      being handed in — mean_a/mean_b come back on the result object so
#      callers that need the point-estimate xRuns (e.g. the awayXRuns/
#      homeXRuns fields on GameResponse) read it off the same number the
#      simulation actually used, with zero chance of the displayed mean
#      drifting from the simulated one.
# ---------------------------------------------------------------------------

MC_DEFAULT_SIMS = 30_000
# Runs above this per side get folded into a single tail bucket in the
# returned total-runs PMF, so the dict stays small regardless of n_sims or
# a freak high-lambda matchup (e.g. Coors Field).
MC_MAX_TOTAL_RUNS_BUCKET = 24


@dataclass(frozen=True)
class MonteCarloWinResult:
    prob_a: float
    prob_b: float
    # Quality-factor-adjusted Poisson lambda actually sampled from for each
    # side (calculate_team_xruns_v2 output) — same numbers previously
    # computed separately in prediction_service.py and handed in as
    # runs_a/runs_b; now derived here so display and simulation can't drift.
    mean_a: float
    mean_b: float
    n_sims: int
    # Empirical PMF of (runs_a + runs_b) across the simulation, keyed by
    # total-runs bucket -> probability. Last key is an overflow bucket for
    # anything >= MC_MAX_TOTAL_RUNS_BUCKET. Bonus output — not yet wired
    # into any response schema (see prediction_service.py TASK 1 note).
    total_runs_dist: dict[int, float]
    mean_total_runs: float


def poisson_monte_carlo_win_prob(
    matchup_mult_a: float,
    matchup_mult_b: float,
    park_factor: float,
    weather_mult: float,
    bullpen_era_a: float,
    bullpen_era_b: float,
    bullpen_fatigue_mult_a: float = 1.0,
    bullpen_fatigue_mult_b: float = 1.0,
    n_sims: int = MC_DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
) -> MonteCarloWinResult:
    # Side "a"/"b" mirror runs_a/runs_b from the pre-TASK-3 signature (and
    # pythagenpat_win_prob's convention) — callers keep whichever side they
    # call "a" mapped to home vs away themselves; this function doesn't
    # care which is which, only that bullpen_era_a is the ERA of the
    # bullpen side "a" is BATTING AGAINST (i.e. the opposing team's pen),
    # same contract calculate_team_xruns_v2 already has.
    mean_a = calculate_team_xruns_v2(
        matchup_mult=matchup_mult_a, park_factor=park_factor, weather_mult=weather_mult,
        bullpen_era=bullpen_era_a, bullpen_fatigue_mult=bullpen_fatigue_mult_a,
    )
    mean_b = calculate_team_xruns_v2(
        matchup_mult=matchup_mult_b, park_factor=park_factor, weather_mult=weather_mult,
        bullpen_era=bullpen_era_b, bullpen_fatigue_mult=bullpen_fatigue_mult_b,
    )

    # calculate_team_xruns_v2 clamps to [1.5, 10.0] so this floor is never
    # actually hit in practice — kept as a defensive fallback (matches the
    # runs_a<=0/runs_b<=0 guard the pre-TASK-3 version had) in case that
    # clamp range ever changes.
    if mean_a <= 0 or mean_b <= 0:
        return MonteCarloWinResult(
            prob_a=0.5, prob_b=0.5, mean_a=mean_a, mean_b=mean_b,
            n_sims=0, total_runs_dist={}, mean_total_runs=0.0,
        )

    generator = rng if rng is not None else np.random.default_rng()
    sims_a = generator.poisson(lam=mean_a, size=n_sims)
    sims_b = generator.poisson(lam=mean_b, size=n_sims)

    a_wins = sims_a > sims_b
    b_wins = sims_b > sims_a
    tie_mask = sims_a == sims_b

    tie_count = int(tie_mask.sum())
    if tie_count:
        a_wins = a_wins.copy()
        b_wins = b_wins.copy()
        tie_idx = np.flatnonzero(tie_mask)
        coin = generator.random(tie_count) < 0.5
        a_wins[tie_idx[coin]] = True
        b_wins[tie_idx[~coin]] = True

    prob_a = clamp(float(a_wins.mean()), MIN_PROB, MAX_PROB)
    prob_b = clamp(1.0 - prob_a, MIN_PROB, MAX_PROB)

    total_runs = sims_a + sims_b
    capped = np.minimum(total_runs, MC_MAX_TOTAL_RUNS_BUCKET)
    counts = np.bincount(capped, minlength=MC_MAX_TOTAL_RUNS_BUCKET + 1)
    total_runs_dist = {
        int(bucket): round(float(count) / n_sims, 5)
        for bucket, count in enumerate(counts)
        if count > 0
    }

    return MonteCarloWinResult(
        prob_a=round(prob_a, 4),
        prob_b=round(prob_b, 4),
        mean_a=round(mean_a, 3),
        mean_b=round(mean_b, 3),
        n_sims=n_sims,
        total_runs_dist=total_runs_dist,
        mean_total_runs=round(float(total_runs.mean()), 3),
    )


def calculate_team_xruns_v2(
    matchup_mult: float, park_factor: float, weather_mult: float, bullpen_era: float, bullpen_fatigue_mult: float
) -> float:
    base_run_projection = 4.30
    bullpen_factor = 1.0 + ((LEAGUE_AVG_ERA - bullpen_era) / 25.0)
    bullpen_factor *= bullpen_fatigue_mult
    final_xruns = base_run_projection * matchup_mult * park_factor * weather_mult * bullpen_factor
    return clamp(final_xruns, 1.5, 10.0)


def poisson_tail_at_least(expected: float, threshold: int) -> float:
    if expected <= 0:
        return 0.0
    cumulative = sum(math.exp(-expected) * (expected**k) / math.factorial(k) for k in range(threshold))
    return clamp(1.0 - cumulative, 0.0, 1.0)


def build_pitcher_ladder(expected: float, lines: list[float]) -> list[dict]:
    output: list[dict] = []
    for line in lines:
        threshold = math.floor(line) + 1
        probability = poisson_tail_at_least(expected, threshold)
        output.append({"line": line, "prob": round(probability, 4), "odds": prob_to_american(probability)})
    return output


# ---------------------------------------------------------------------------
# CONSOLIDATION — ported from generate_projections.py's calibration engine
# (sample_mean/sample_variance/poisson_*/nbinom_*/calibrate_market), now
# shared by /api/pitcher-props and /api/team-matchups instead of living only
# in the retired standalone script + a second copy in index.html's inline JS.
#
# poisson_over_prob/nbinom_over_prob use a *ceil(line)* threshold (P(X >=
# ceil(line))), matching the script's "beat a X.5-style betting line"
# semantics. This is deliberately a separate pair of functions from
# poisson_tail_at_least/build_pitcher_ladder above, which use a
# floor(line)+1 threshold for the existing pitcher K/BB/ER ladder — same
# math family, different contract; ladder callers are untouched.
# ---------------------------------------------------------------------------
def sample_mean(values: list[float]) -> float:
    n = len(values)
    return sum(values) / n if n else 0.0


def sample_variance(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sample_mean(values)
    return sum((x - mu) ** 2 for x in values) / (n - 1)


def poisson_pmf(expected: float, k: int) -> float:
    if expected <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-expected) * (expected**k) / math.factorial(k)


def poisson_over_prob(expected: float, line: float) -> float:
    threshold = math.ceil(line)
    cumulative = sum(poisson_pmf(expected, k) for k in range(threshold))
    return clamp(1.0 - cumulative, 0.0, 1.0)


def nbinom_pmf(k: int, r: float, p: float) -> float:
    if not (0.0 < p < 1.0) or r <= 0:
        raise ValueError("invalid negative-binomial parameters")
    log_coef = math.lgamma(r + k) - math.lgamma(r) - math.lgamma(k + 1)
    log_pmf = log_coef + r * math.log(p) + k * math.log(1.0 - p)
    return math.exp(log_pmf)


def nbinom_params(mu: float, var: float) -> tuple[float, float]:
    if var <= mu:
        raise ValueError("variance must exceed mean for NB2 (over-dispersion required)")
    p = mu / var
    r = (mu**2) / (var - mu)
    if not (0.0 < p < 1.0) or r <= 0 or math.isinf(r):
        raise ValueError("degenerate negative-binomial parameters")
    return r, p


def nbinom_over_prob(mu: float, var: float, line: float) -> float:
    r, p = nbinom_params(mu, var)
    threshold = math.ceil(line)
    cumulative = sum(nbinom_pmf(k, r, p) for k in range(threshold))
    return clamp(1.0 - cumulative, 0.0, 1.0)


def kelly_stake_pct(model_prob: float) -> float:
    edge = model_prob - MARKET_IMPLIED_PROB
    if edge <= 0:
        return 0.0
    q = 1.0 - model_prob
    f_star = (DECIMAL_B * model_prob - q) / DECIMAL_B
    return round(max(0.0, f_star) * 100.0, 2)


def market_sample_confidence(n: int) -> str:
    if n >= 10:
        return "high"
    if n >= 6:
        return "medium"
    return "low"


def calibrate_count_market(
    history: list[float],
    line: float,
    anchor_mu: float | None = None,
    anchor_k: float = 0.0,
) -> dict:
    """VMR-gated Poisson/NB2 calibration for a last-N-game count history
    (pitcher K/ER/BB, team F5/full-game runs), with an OPTIONAL shrinkage
    anchor for the distribution's mean.

    sigma^2 / the VMR gate ITSELF are computed from the raw `history` only —
    `observed_mu`/`var` below — untouched, exactly as in the original script.
    That's the signal that decides Poisson vs NB2 and is not modified by
    this consolidation.

    `mu` — the number actually fed into whichever distribution the gate
    picked — is optionally shrunk toward `anchor_mu` via the SAME shrink_rate()
    used everywhere else in this module: shrink_rate(observed_mu, n,
    anchor_mu, anchor_k). With anchor_mu=None (pitcher props — there is no
    second projection model to reconcile with) this is a no-op and `mu ==
    observed_mu`, i.e. byte-for-byte the pre-consolidation behavior.

    For team runs markets, callers pass anchor_mu=<calculate_team_xruns_v2
    output (or its 5/9 F5 share)> and anchor_k=K_TEAM_RUNS, so a thin/noisy
    last-10-game sample gets pulled toward the same expected-runs number the
    Moneylines tab's Pythagenpat win prob is already built from, without
    ever overwriting the empirical variance that drives the NB2 gate.
    """
    n = len(history)
    observed_mu = sample_mean(history)
    var = sample_variance(history)
    vmr = (var / observed_mu) if observed_mu > 0 else 0.0
    use_nb2 = var > observed_mu and observed_mu > 0

    if anchor_mu is not None and anchor_k > 0:
        mu = shrink_rate(observed_mu, n, anchor_mu, anchor_k)
    else:
        mu = observed_mu

    fallback = False
    if use_nb2:
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

    edge_pct = round((over_prob - MARKET_IMPLIED_PROB) * 100.0, 2)

    return {
        "distribution": distribution,
        "mu": round(mu, 3),
        "observedMu": round(observed_mu, 3),
        "variance": round(var, 3),
        "vmr": round(vmr, 3),
        "overProb": round(over_prob, 4),
        "fairOdds": prob_to_american(over_prob),
        "impliedProbMarket": round(MARKET_IMPLIED_PROB, 4),
        "edgePct": edge_pct,
        "kellyPct": kelly_stake_pct(over_prob),
        "confidence": market_sample_confidence(n),
        "sampleSize": n,
        "fallback": fallback,
        "anchorMu": round(anchor_mu, 3) if anchor_mu is not None else None,
    }


def build_count_market(
    label: str,
    history: list[int],
    default_line: float,
    anchor_mu: float | None = None,
    anchor_k: float = 0.0,
    pad: int = 2,
) -> dict:
    model = calibrate_count_market(history, default_line, anchor_mu=anchor_mu, anchor_k=anchor_k)
    return {
        "label": label,
        "defaultLine": default_line,
        "history": history,
        "max": max(history + [int(default_line) + 3]) + pad,
        "model": model,
    }