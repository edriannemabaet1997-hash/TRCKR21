from __future__ import annotations

import math
from typing import Callable

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
    return round(clamp(final_prob, MIN_PROB, MAX_PROB), 4)


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
    return compute_event_probability(rate, pa, starter_mult, bullpen_mult, park, platoon, iso)


def process_run_prob(rate, pa, pitcher_era, bullpen_era, park, platoon, iso, team_obp, order) -> float:
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    team_factor = clamp(team_obp / 0.315, 0.90, 1.10)
    archetype_mult = run_scoring_archetype_factor(order, iso)
    return compute_event_probability(rate * team_factor * archetype_mult, pa, starter_mult, bullpen_mult, park, platoon, iso)


def process_hr_prob(rate, pa, pitcher_era, bullpen_era, park, platoon, iso_val, hr_rate_observed) -> float:
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    power_index = hr_power_index(iso_val, hr_rate_observed)
    return compute_event_probability(rate, pa, starter_mult, bullpen_mult, park * 1.1, platoon, power_index)


def process_rbi_prob(rate, pa, pitcher_era, bullpen_era, park, platoon, iso, team_obp, order, team_slg) -> float:
    starter_mult = 1.0 + ((4.20 - pitcher_era) / 10.0)
    bullpen_mult = 1.0 + ((4.20 - bullpen_era) / 10.0)
    lineup_factor = clamp(team_obp / 0.315, 0.90, 1.15)
    protection_mult = lineup_protection_factor(order, team_slg)
    return compute_event_probability(rate * lineup_factor * protection_mult, pa, starter_mult, bullpen_mult, park, platoon, iso)


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
    final_prob = clamp(original_prob + total_adjustment, MIN_PROB, MAX_PROB)
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


def confidence_from_edge(edge: float | None, probability: float, sample_size: int) -> str:
    if sample_size < 20:
        return "low"
    if edge is None:
        if probability >= 0.70:
            return "high"
        if probability >= 0.55:
            return "med"
        return "low"
    if edge >= 0.08:
        return "high"
    if edge >= 0.03:
        return "med"
    return "low"


def pythagenpat_win_prob(runs_a: float, runs_b: float) -> tuple[float, float]:
    if runs_a <= 0 or runs_b <= 0:
        return 0.5, 0.5
    exponent = (runs_a + runs_b) ** 0.285
    prob_a = (runs_a**exponent) / (runs_a**exponent + runs_b**exponent)
    prob_a = clamp(prob_a, MIN_PROB, MAX_PROB)
    return round(prob_a, 4), round(1.0 - prob_a, 4)


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