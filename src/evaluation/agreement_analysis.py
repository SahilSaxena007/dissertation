# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

﻿"""
AI-human agreement/disagreement analysis.

Saves:
  reports/tables/agreement_analysis.csv
  reports/tables/agreement_summary.csv

Usage (from src/):
  python evaluation/agreement_analysis.py
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TABLES_DIR = os.path.join(PROJECT_ROOT, "reports", "tables")


def run_agreement_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(TABLES_DIR, exist_ok=True)
    db_path = os.path.join(ARTIFACTS_DIR, "hitl_feedback.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing feedback DB: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        fb = pd.read_sql_query("SELECT * FROM clinician_feedback", conn)
    finally:
        conn.close()

    if fb.empty:
        detail_out = os.path.join(TABLES_DIR, "agreement_analysis.csv")
        summary_out = os.path.join(TABLES_DIR, "agreement_summary.csv")
        pd.DataFrame().to_csv(detail_out, index=False)
        pd.DataFrame(columns=["metric", "value"]).to_csv(summary_out, index=False)
        print("No clinician feedback rows yet; wrote empty agreement outputs.")
        return pd.DataFrame(), pd.DataFrame(columns=["metric", "value"])

    fb["agree"] = pd.to_numeric(fb["agree"], errors="coerce")
    fb["clinician_confidence"] = pd.to_numeric(fb["clinician_confidence"], errors="coerce")

    if "review_risk_score" in fb.columns:
        fb["review_risk_score"] = pd.to_numeric(fb["review_risk_score"], errors="coerce")

    sections = []

    by_pred = (
        fb.groupby("predicted_label", dropna=False)
        .agg(
            n=("id", "count"),
            agree_rate=("agree", "mean"),
            mean_confidence=("clinician_confidence", "mean"),
            mean_review_risk_score=("review_risk_score", "mean"),
        )
        .reset_index()
        .rename(columns={"predicted_label": "group"})
    )
    by_pred["section"] = "by_predicted_label"
    sections.append(by_pred)

    by_escalation = (
        fb.groupby("escalation_level", dropna=False)
        .agg(
            n=("id", "count"),
            agree_rate=("agree", "mean"),
            mean_confidence=("clinician_confidence", "mean"),
            mean_review_risk_score=("review_risk_score", "mean"),
        )
        .reset_index()
        .rename(columns={"escalation_level": "group"})
    )
    by_escalation["section"] = "by_escalation_level"
    sections.append(by_escalation)

    disagreement = fb[fb["agree"] == 0].copy()
    if disagreement.empty:
        by_reason = pd.DataFrame(columns=["group", "n", "fraction", "section"])
    else:
        reason_counts = disagreement["escalation_reasons"].fillna("none").value_counts(dropna=False)
        by_reason = reason_counts.reset_index()
        by_reason.columns = ["group", "n"]
        by_reason["fraction"] = by_reason["n"] / max(int(by_reason["n"].sum()), 1)
    by_reason["section"] = "disagreement_by_reason"
    sections.append(by_reason)

    detail_df = pd.concat(sections, ignore_index=True, sort=False)

    summary = pd.DataFrame(
        [
            {"metric": "total_reviews", "value": float(len(fb))},
            {"metric": "overall_agree_rate", "value": float(fb["agree"].mean())},
            {"metric": "overall_disagree_rate", "value": float((1 - fb["agree"]).mean())},
            {
                "metric": "mean_confidence_when_agree",
                "value": float(fb.loc[fb["agree"] == 1, "clinician_confidence"].mean())
                if (fb["agree"] == 1).any()
                else float("nan"),
            },
            {
                "metric": "mean_confidence_when_disagree",
                "value": float(fb.loc[fb["agree"] == 0, "clinician_confidence"].mean())
                if (fb["agree"] == 0).any()
                else float("nan"),
            },
        ]
    )

    detail_out = os.path.join(TABLES_DIR, "agreement_analysis.csv")
    summary_out = os.path.join(TABLES_DIR, "agreement_summary.csv")
    detail_df.to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"Saved agreement analysis to: {detail_out}")
    print(f"Saved agreement summary to: {summary_out}")
    return detail_df, summary


if __name__ == "__main__":
    run_agreement_analysis()
