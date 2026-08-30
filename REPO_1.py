from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import settings


class PredictionRepository:
    def __init__(self) -> None:
        self.path = settings.database_path
        self.initialize()

    @contextmanager
    def connection(self):
        # Dagdag na timeout (20s) para iwas database locked errors
        connection = sqlite3.connect(self.path, timeout=20.0)
        connection.row_factory = sqlite3.Row
        try:
            # I-enable ang Write-Ahead Logging para sa multi-threading speed
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute("PRAGMA synchronous=NORMAL;")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    game_pk INTEGER NOT NULL,
                    game_date TEXT NOT NULL,
                    player_id INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    prop_type TEXT NOT NULL,
                    probability REAL NOT NULL,
                    fair_odds INTEGER NOT NULL,
                    book_odds INTEGER,
                    actual INTEGER,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (
                        game_pk,
                        player_id,
                        prop_type
                    )
                )
                """
            )

            # Indexes para sa mabilis na filtering at sorting
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_unresolved
                ON predictions (actual, game_date)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_resolved
                ON predictions (game_date DESC, created_at DESC)
                """
            )

    def upsert_prediction(
        self,
        game_pk: int,
        game_date: str,
        player_id: int,
        player_name: str,
        prop_type: str,
        probability: float,
        fair_odds: int,
        book_odds: int | None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    game_pk,
                    game_date,
                    player_id,
                    player_name,
                    prop_type,
                    probability,
                    fair_odds,
                    book_odds,
                    actual,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT (
                    game_pk,
                    player_id,
                    prop_type
                )
                DO UPDATE SET
                    probability = excluded.probability,
                    fair_odds = excluded.fair_odds,
                    book_odds = excluded.book_odds,
                    created_at = excluded.created_at
                """,
                (
                    game_pk,
                    game_date,
                    player_id,
                    player_name,
                    prop_type,
                    probability,
                    fair_odds,
                    book_odds,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def unresolved(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM predictions
                WHERE actual IS NULL
                ORDER BY game_date ASC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def set_actual(
        self,
        game_pk: int,
        player_id: int,
        prop_type: str,
        actual: int,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE predictions
                SET actual = ?
                WHERE game_pk = ?
                  AND player_id = ?
                  AND prop_type = ?
                """,
                (
                    actual,
                    game_pk,
                    player_id,
                    prop_type,
                ),
            )

    def track_record(self, limit: int = 100) -> dict:
        with self.connection() as connection:
            resolved = connection.execute(
                """
                SELECT *
                FROM predictions
                WHERE actual IS NOT NULL
                ORDER BY game_date DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            calibration_rows = connection.execute(
                """
                SELECT
                    CASE
                        WHEN probability < 0.60 THEN '50-60%'
                        WHEN probability < 0.70 THEN '60-70%'
                        WHEN probability < 0.80 THEN '70-80%'
                        ELSE '80%+'
                    END AS bucket,
                    AVG(probability) AS predicted,
                    AVG(CASE WHEN actual > 0 THEN 1.0 ELSE 0.0 END) AS actual,
                    COUNT(*) AS n
                FROM predictions
                WHERE actual IS NOT NULL
                  AND probability >= 0.50
                GROUP BY bucket
                ORDER BY MIN(probability)
                """
            ).fetchall()

        calibration = [
            {
                "bucket": row["bucket"],
                "predicted": round(row["predicted"] or 0.0, 4),
                "actual": round(row["actual"] or 0.0, 4),
                "n": row["n"],
            }
            for row in calibration_rows
        ]

        log = []
        for row in resolved:
            predicted_yes = row["probability"] >= 0.5
            actual_val = row["actual"] if row["actual"] is not None else 0
            actual_yes = actual_val > 0

            log.append(
                {
                    "date": row["game_date"],
                    "player": row["player_name"],
                    "stat": row["prop_type"],
                    "predicted": row["probability"],
                    "actual": actual_val,
                    "correct": predicted_yes == actual_yes,
                }
            )

        return {
            "calibration": calibration,
            "log": log,
        }