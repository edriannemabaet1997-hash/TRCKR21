# debug_h2h_standalone.py
#
# Isolated, one-off script to inspect the raw MLB `vsPlayer` JSON for two
# different batters against the SAME pitcher — without touching the live
# app, the slate build, or app.py at all. This is deliberately separate
# from mlb_client.py's DEBUG_H2H flag: that flag sits on
# hitter_vs_pitcher_history(), which is also on the hot path for a full
# slate build (batter_vs_pitcher_slash() -> proxy xBA/xSLG, called once
# per batter on the slate, ~270 times, concurrently). Turning debug
# printing on there and then hitting REFRESH floods stdout and can push
# the whole batter model past its 90s timeout. This script makes exactly
# the two calls we actually want to compare, and nothing else — safe to
# run any time, independent of whether the server is even running.

from __future__ import annotations

import json
import sys

from mlb_client import MLBClient

USAGE = (
    "Usage: python debug_h2h_standalone.py BATTER_ID_1 BATTER_ID_2 PITCHER_ID\n"
    "  (all three are plain integers — real MLB person ids, not placeholders)\n"
    "\n"
    "Example (swap in the real ids you see in the app's UI/network tab):\n"
    "  python debug_h2h_standalone.py 682998 596142 594979"
)


def dump_h2h(mlb: MLBClient, batter_id: int, pitcher_id: int) -> None:
    payload = mlb.person_stats(
        batter_id, group="hitting", stats="vsPlayer", opposingPlayerId=pitcher_id
    )
    print(f"\n=== batter_id={batter_id} pitcher_id={pitcher_id} ===")
    print(json.dumps(payload, indent=2))


def main() -> None:
    if len(sys.argv) != 4:
        print(USAGE)
        sys.exit(1)

    try:
        batter_id_1, batter_id_2, pitcher_id = (int(a) for a in sys.argv[1:4])
    except ValueError:
        print("Error: all three arguments must be plain integers (MLB person ids).\n")
        print(USAGE)
        sys.exit(1)

    mlb = MLBClient()
    dump_h2h(mlb, batter_id_1, pitcher_id)
    dump_h2h(mlb, batter_id_2, pitcher_id)

    print(
        "\n--- Compare the two payloads above by eye (or pipe this "
        "script's output through `diff`). If they're byte-for-byte "
        "identical despite different batter_ids, that's the MLB "
        "vsPlayer quirk itself — paste both blocks back so we can see "
        "the exact field names and tighten the validator in "
        "hitter_vs_pitcher_history(). If they differ, the bug is in "
        "get_matchup()'s parsing instead."
    )


if __name__ == "__main__":
    main()