# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

﻿"""
Efficiency metrics from HITL interaction logs.

Saves:
  reports/tables/efficiency_analysis.csv
  reports/tables/efficiency_summary.csv

Usage (from src/):
  python evaluation/efficiency_analysis.py
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _safe_median(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.median())


def run_efficiency_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(TABLES_DIR, exist_ok=True)
    db_path = os.path.join(ARTIFACTS_DIR, "hitl_feedback.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing feedback DB: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        feedback = pd.read_sql_query("SELECT * FROM clinician_feedback", conn)
        events = pd.read_sql_query("SELECT * FROM interaction_events", conn)
    finally:
        conn.close()

    if feedback.empty:
        detail_out = os.path.join(TABLES_DIR, "efficiency_analysis.csv")
        summary_out = os.path.join(TABLES_DIR, "efficiency_summary.csv")
        pd.DataFrame(columns=["metric", "value"]).to_csv(summary_out, index=False)
        pd.DataFrame().to_csv(detail_out, index=False)
        print("No clinician feedback rows yet; wrote empty efficiency outputs.")
        return pd.DataFrame(), pd.DataFrame(columns=["metric", "value"])

    feedback["decision_seconds"] = pd.to_numeric(feedback["decision_seconds"], errors="coerce")
    feedback["clinician_confidence"] = pd.to_numeric(feedback["clinician_confidence"], errors="coerce")
    feedback["agree"] = pd.to_numeric(feedback["agree"], errors="coerce")

    detail_frames = []

    by_level = (
        feedback.groupby("escalation_level", dropna=False)
        .agg(
            n_reviews=("id", "count"),
            median_decision_seconds=("decision_seconds", "median"),
            mean_decision_seconds=("decision_seconds", "mean"),
            agree_rate=("agree", "mean"),
            mean_confidence=("clinician_confidence", "mean"),
        )
        .reset_index()
    )
    by_level["section"] = "by_escalation_level"
    detail_frames.append(by_level)

    by_source = (
        feedback.groupby("source", dropna=False)
        .agg(
            n_reviews=("id", "count"),
            median_decision_seconds=("decision_seconds", "median"),
            mean_decision_seconds=("decision_seconds", "mean"),
            agree_rate=("agree", "mean"),
            mean_confidence=("clinician_confidence", "mean"),
        )
        .reset_index()
        .rename(columns={"source": "group"})
    )
    by_source["section"] = "by_source"
    detail_frames.append(by_source)

    interactions_per_case = (
        feedback.groupby(["source", "sample_id"], dropna=False)
        .size()
        .reset_index(name="n_interactions")
    )
    ipc_summary = interactions_per_case.groupby("source", dropna=False)["n_interactions"].agg(
        median_interactions="median",
        mean_interactions="mean",
        max_interactions="max",
    ).reset_index().rename(columns={"source": "group"})
    ipc_summary["section"] = "interactions_per_case"
    detail_frames.append(ipc_summary)

    conf_dist = (
        feedback.groupby("clinician_confidence", dropna=False)
        .size()
        .reset_index(name="count")
    )
    conf_dist["fraction"] = conf_dist["count"] / max(int(conf_dist["count"].sum()), 1)
    conf_dist["section"] = "confidence_distribution"
    detail_frames.append(conf_dist)

    detail_df = pd.concat(detail_frames, ignore_index=True, sort=False)

    summary_rows = [
        {"metric": "total_reviews", "value": float(len(feedback))},
        {"metric": "total_event_rows", "value": float(len(events))},
        {"metric": "median_decision_seconds", "value": _safe_median(feedback["decision_seconds"].dropna())},
        {"metric": "mean_decision_seconds", "value": _safe_mean(feedback["decision_seconds"].dropna())},
        {"metric": "agree_rate", "value": _safe_mean(feedback["agree"].dropna())},
        {"metric": "mean_clinician_confidence", "value": _safe_mean(feedback["clinician_confidence"].dropna())},
        {
            "metric": "median_interactions_per_case",
            "value": _safe_median(interactions_per_case["n_interactions"]) if not interactions_per_case.empty else float("nan"),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    detail_out = os.path.join(TABLES_DIR, "efficiency_analysis.csv")
    summary_out = os.path.join(TABLES_DIR, "efficiency_summary.csv")
    detail_df.to_csv(detail_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved efficiency analysis to: {detail_out}")
    print(f"Saved efficiency summary to: {summary_out}")
    return detail_df, summary_df


if __name__ == "__main__":
    run_efficiency_analysis()
