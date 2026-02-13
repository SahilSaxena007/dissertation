"""
SQLite-backed interaction logging for HITL dashboard use.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

VALID_CLASSES = {"SCD", "MCI", "AD"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_default_db_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "artifacts" / "hitl_feedback.db"


def ensure_db(db_path: Path | str | None = None) -> Path:
    db_file = Path(db_path) if db_path else get_default_db_path()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clinician_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                sample_id INTEGER NOT NULL,
                true_label TEXT,
                predicted_label TEXT NOT NULL,
                agree INTEGER NOT NULL,
                corrected_label TEXT NOT NULL,
                clinician_confidence INTEGER NOT NULL,
                notes TEXT,
                decision_seconds REAL,
                viewed_features TEXT,
                escalation_level TEXT,
                escalation_reasons TEXT,
                review_risk_score REAL,
                session_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                session_id TEXT,
                event_type TEXT NOT NULL,
                source TEXT,
                sample_id INTEGER,
                payload_json TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_file


def _validate_feedback(
    predicted_label: str,
    corrected_label: str,
    clinician_confidence: int,
):
    if predicted_label not in VALID_CLASSES:
        raise ValueError(f"predicted_label must be one of {sorted(VALID_CLASSES)}.")
    if corrected_label not in VALID_CLASSES:
        raise ValueError(f"corrected_label must be one of {sorted(VALID_CLASSES)}.")
    if clinician_confidence < 1 or clinician_confidence > 5:
        raise ValueError("clinician_confidence must be in [1, 5].")


def log_clinician_feedback(
    *,
    source: str,
    sample_id: int,
    predicted_label: str,
    agree: bool,
    corrected_label: str,
    clinician_confidence: int,
    true_label: Optional[str] = None,
    notes: str = "",
    decision_seconds: Optional[float] = None,
    viewed_features: Optional[Iterable[str]] = None,
    escalation_level: str = "",
    escalation_reasons: str = "",
    review_risk_score: Optional[float] = None,
    session_id: Optional[str] = None,
    db_path: Path | str | None = None,
) -> None:
    _validate_feedback(
        predicted_label=predicted_label,
        corrected_label=corrected_label,
        clinician_confidence=int(clinician_confidence),
    )

    db_file = ensure_db(db_path)
    viewed_features_text = ",".join(viewed_features) if viewed_features else ""
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(
            """
            INSERT INTO clinician_feedback (
                timestamp_utc, source, sample_id, true_label, predicted_label,
                agree, corrected_label, clinician_confidence, notes,
                decision_seconds, viewed_features, escalation_level,
                escalation_reasons, review_risk_score, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now_iso(),
                source,
                int(sample_id),
                true_label,
                predicted_label,
                int(bool(agree)),
                corrected_label,
                int(clinician_confidence),
                notes,
                float(decision_seconds) if decision_seconds is not None else None,
                viewed_features_text,
                escalation_level,
                escalation_reasons,
                float(review_risk_score) if review_risk_score is not None else None,
                session_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_event(
    *,
    event_type: str,
    source: str = "",
    sample_id: Optional[int] = None,
    payload_json: str = "",
    session_id: Optional[str] = None,
    db_path: Path | str | None = None,
) -> None:
    db_file = ensure_db(db_path)
    conn = sqlite3.connect(db_file)
    try:
        conn.execute(
            """
            INSERT INTO interaction_events (
                timestamp_utc, session_id, event_type, source, sample_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now_iso(),
                session_id,
                event_type,
                source,
                int(sample_id) if sample_id is not None else None,
                payload_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()

