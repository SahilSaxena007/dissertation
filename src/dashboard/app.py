from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
import __main__
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from escalation.escalation_config import CLASS_NAMES, build_escalation_config
from escalation.escalation_engine import run_batch_escalation_from_probabilities
from escalation.model_stub import create_model as _legacy_create_model
from hitl.feedback_loop import run_active_learning_simulation
from hitl.interaction_logger import ensure_db, get_default_db_path, log_clinician_feedback, log_event


st.set_page_config(page_title="HITL Clinician Dashboard", layout="wide")

# Compatibility shim for legacy joblib artifacts where SciKeras stores
# model build_fn as "__main__.create_model".
__main__.create_model = _legacy_create_model

REFERENCE_RANGES = {
    "AGE": "55-90",
    "PTGENDER": "0/1",
    "TAU": "200-450",
    "PTAU": "20-60",
    "MMSE": "24-30",
    "ABETA": "500-1600",
    "PLASMA_NFL": "5-40",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


_CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def _coerce_predict_proba(raw_probs, n_samples: int) -> np.ndarray:
    """Convert model-specific predict_proba output to a 2D float array."""
    if isinstance(raw_probs, list):
        # Some wrappers may return list[per-sample probs].
        raw_probs = np.array(raw_probs, dtype=object)
        if raw_probs.dtype == object:
            raw_probs = np.vstack([np.asarray(r, dtype=float).reshape(-1) for r in raw_probs])

    arr = np.asarray(raw_probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"predict_proba output must be 2D; got shape={arr.shape}")
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"predict_proba sample count mismatch: expected {n_samples}, got {arr.shape[0]}"
        )
    return arr


def _align_probs_to_classes(probs: np.ndarray, classes_) -> np.ndarray:
    """Align probability columns to CLASS_NAMES order: [SCD, MCI, AD]."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probs, got {probs.shape}")

    n_samples, n_cols = probs.shape
    if n_cols == len(CLASS_NAMES) and classes_ is None:
        return probs

    aligned = np.zeros((n_samples, len(CLASS_NAMES)), dtype=float)

    if classes_ is None:
        if n_cols != len(CLASS_NAMES):
            raise ValueError(
                f"Cannot align probs without classes_: got {n_cols} columns for {len(CLASS_NAMES)} classes"
            )
        return probs

    classes = list(np.asarray(classes_).tolist())
    if len(classes) != n_cols:
        raise ValueError(
            f"classes_ length ({len(classes)}) does not match proba columns ({n_cols})"
        )

    for src_idx, cls in enumerate(classes):
        dst_idx = None
        if isinstance(cls, (int, np.integer)) and 0 <= int(cls) < len(CLASS_NAMES):
            dst_idx = int(cls)
        else:
            key = str(cls)
            if key in _CLASS_INDEX:
                dst_idx = _CLASS_INDEX[key]
        if dst_idx is None:
            continue
        aligned[:, dst_idx] = probs[:, src_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    valid = row_sums.squeeze() > 0
    if np.any(valid):
        aligned[valid] = aligned[valid] / row_sums[valid]
    return aligned


def _predict_ensemble_probs(models: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """Predict and class-align per-model probabilities for ensembling."""
    out: dict[str, np.ndarray] = {}
    n_samples = int(X.shape[0])
    for name, model in models.items():
        try:
            raw = model.predict_proba(X)
            probs_2d = _coerce_predict_proba(raw, n_samples=n_samples)
            probs_aligned = _align_probs_to_classes(probs_2d, getattr(model, "classes_", None))
            out[name] = probs_aligned
        except Exception as e:
            st.warning(f"Skipping model '{name}' for this prediction: {e}")
    if not out:
        raise RuntimeError("No model produced valid class probabilities.")
    return out


def _risk_features_from_df(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - df["ens_max_prob"].to_numpy(dtype=float),
            df["ens_margin"].to_numpy(dtype=float),
            df["ens_entropy"].to_numpy(dtype=float),
            df["disagreement_score"].to_numpy(dtype=float),
            df["missing_fraction"].to_numpy(dtype=float),
            df["critical_missing"].astype(int).to_numpy(dtype=float),
            df["multimodal_mismatch"].astype(int).to_numpy(dtype=float),
        ]
    )


def _escalation_color(level: str) -> str:
    if level == "Clinician-Mandatory":
        return "#d7263d"
    if level == "AI-Assisted":
        return "#ff9f1c"
    return "#2a9d8f"


@st.cache_resource(show_spinner=False)
def load_assets():
    root = _root()
    artifacts = root / "artifacts"
    data_dir = root / "data"

    models = joblib.load(artifacts / "voting_ensemble.pkl")
    if "rf" in models and hasattr(models["rf"], "n_jobs"):
        models["rf"].n_jobs = 1

    scaler = joblib.load(artifacts / "scaler.pkl")
    imputer = joblib.load(artifacts / "imputer.pkl")
    selector = joblib.load(artifacts / "select_k_best.pkl")
    meta_model = joblib.load(artifacts / "step2_meta_model.pkl")
    selected_features = json.load(open(artifacts / "selected_features.json", "r", encoding="utf-8"))
    tau_star = float(json.load(open(artifacts / "step3_best_threshold.json", "r", encoding="utf-8"))["best_threshold"])

    preproc = pd.read_csv(data_dir / "preprocessed_data.csv") if (data_dir / "preprocessed_data.csv").exists() else None
    medians = {}
    if preproc is not None:
        for f in selected_features:
            medians[f] = float(preproc[f].median()) if f in preproc.columns else 0.0
    else:
        medians = {f: 0.0 for f in selected_features}

    # Load SHAP artifacts if available
    shap_artifacts = {}
    for model_name in ["catboost", "rf", "xgb"]:
        for source in ["oof", "holdout"]:
            path = artifacts / f"shap_{source}_{model_name}.npy"
            if path.exists():
                shap_artifacts[f"{source}_{model_name}"] = np.load(path, allow_pickle=True)

    return models, scaler, imputer, selector, meta_model, selected_features, tau_star, preproc, medians, shap_artifacts


@st.cache_data(show_spinner=False)
def load_retrospective_dataset(source: str):
    root = _root()
    tables = root / "reports" / "tables"
    artifacts = root / "artifacts"

    models, _, _, _, meta_model, selected_features, _, preproc, _, shap_artifacts = load_assets()

    if source == "oof":
        df = pd.read_csv(tables / "escalation_table_oof.csv")
        probs = np.load(artifacts / "oof_ens_proba_selected.npy")
        X = np.load(artifacts / "X_oof.npy")
    else:
        df = pd.read_csv(tables / "escalation_table_testset.csv")
        X = np.load(artifacts / "X_test.npy")
        all_probs = _predict_ensemble_probs(models=models, X=X)
        probs = np.mean(np.stack(list(all_probs.values()), axis=0), axis=0)

    if "sample_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "sample_id"})
    if "ai_correct" not in df.columns and {"true_label", "pred_ens"}.issubset(df.columns):
        df["ai_correct"] = (df["true_label"] == df["pred_ens"]).astype(int)

    if "review_risk_score" not in df.columns:
        X_meta = _risk_features_from_df(df)
        df["review_risk_score"] = meta_model.predict_proba(X_meta)[:, 1]

    # Fix raw_features indexing using proper split indices (Issue #7)
    raw_features = None
    if preproc is not None:
        train_indices_path = artifacts / "train_indices.npy"
        holdout_indices_path = artifacts / "holdout_indices.npy"

        if train_indices_path.exists() and holdout_indices_path.exists():
            train_indices = np.load(train_indices_path)
            holdout_indices = np.load(holdout_indices_path)

            if all(c in preproc.columns for c in selected_features):
                if source == "oof":
                    raw_features = preproc.iloc[train_indices][selected_features].reset_index(drop=True)
                else:
                    raw_features = preproc.iloc[holdout_indices][selected_features].reset_index(drop=True)
        else:
            # Fallback to old behaviour
            if all(c in preproc.columns for c in selected_features) and len(preproc) >= len(df):
                raw_features = preproc.loc[: len(df) - 1, selected_features].copy()

    return df, probs, X, raw_features


def _shap_feature_contrib(
    sample_idx: int,
    pred_idx: int,
    feature_names: list[str],
    source: str,
) -> pd.DataFrame | None:
    """Build SHAP-based feature contribution from pre-computed artifacts."""
    _, _, _, _, _, _, _, _, _, shap_artifacts = load_assets()

    source_key = "oof" if source == "oof" else "holdout"
    available_models = []
    for model_name in ["catboost", "rf", "xgb"]:
        key = f"{source_key}_{model_name}"
        if key in shap_artifacts:
            available_models.append((model_name, shap_artifacts[key]))

    if not available_models:
        return None

    # Average SHAP across available models for the predicted class
    shap_contributions = []
    for model_name, shap_arr in available_models:
        if sample_idx >= shap_arr.shape[0]:
            continue
        sample_shap = shap_arr[sample_idx]
        # Handle (F, C) or (C, F) or (F,) shapes
        if sample_shap.ndim == 2:
            if sample_shap.shape[-1] == 3:  # (F, C)
                contrib = sample_shap[:, pred_idx]
            elif sample_shap.shape[0] == 3:  # (C, F)
                contrib = sample_shap[pred_idx, :]
            else:
                contrib = np.mean(sample_shap, axis=-1)
        elif sample_shap.ndim == 1:
            contrib = sample_shap
        else:
            continue
        shap_contributions.append(contrib)

    if not shap_contributions:
        return None

    avg_contrib = np.mean(shap_contributions, axis=0)
    n_features = min(len(feature_names), len(avg_contrib))
    out = pd.DataFrame({
        "feature": feature_names[:n_features],
        "contribution": avg_contrib[:n_features],
    })
    out["abs"] = out["contribution"].abs()
    out = out.sort_values("abs", ascending=False).head(10)
    return out.sort_values("contribution")


def _surrogate_feature_contrib(
    sample_raw: pd.Series,
    df_raw: pd.DataFrame | None,
    pred_idx: int,
    probs_row: np.ndarray,
) -> pd.DataFrame:
    if df_raw is None:
        values = sample_raw.astype(float).values
        contrib = (values - np.nanmean(values)) * float(probs_row[pred_idx])
        feat = sample_raw.index.tolist()
    else:
        class_means = df_raw.mean(numeric_only=True)
        delta = sample_raw.astype(float) - class_means
        contrib = delta.values * float(probs_row[pred_idx])
        feat = delta.index.tolist()
    out = pd.DataFrame({"feature": feat, "contribution": contrib})
    out["abs"] = out["contribution"].abs()
    out = out.sort_values("abs", ascending=False).head(10)
    return out.sort_values("contribution")


def render_patient_queue(df: pd.DataFrame):
    st.subheader("Patient Queue")
    c1, c2, c3, c4 = st.columns(4)
    levels = c1.multiselect("Escalation Level", sorted(df["escalation_level"].dropna().unique()), default=[])
    klass = c2.multiselect("Predicted Class", sorted(df["pred_ens_name"].dropna().unique()), default=[])
    risk_min = float(c3.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.01))
    conf_max = float(c4.slider("Max Confidence", 0.0, 1.0, 1.0, 0.01))

    q = df.copy()
    if levels:
        q = q[q["escalation_level"].isin(levels)]
    if klass:
        q = q[q["pred_ens_name"].isin(klass)]
    q = q[(q["review_risk_score"] >= risk_min) & (q["ens_max_prob"] <= conf_max)]
    q = q.sort_values(["review_risk_score", "ens_entropy"], ascending=[False, False])

    show = q[
        [
            "sample_id",
            "escalation_level",
            "review_risk_score",
            "pred_ens_name",
            "true_label_name",
            "ens_max_prob",
            "ens_entropy",
            "escalation_reasons",
        ]
    ].head(200)

    def _style_row(row):
        return [f"background-color: {_escalation_color(row['escalation_level'])}22"] * len(row)

    st.dataframe(show.style.apply(_style_row, axis=1), use_container_width=True)
    st.caption(f"Queue size after filters: {len(q)}")


def render_patient_detail(
    df: pd.DataFrame,
    probs: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    raw_features: pd.DataFrame | None,
    tau_star: float,
    source: str,
):
    st.subheader("Patient Detail / Review")
    sample_id = st.number_input("Sample ID", min_value=0, max_value=int(df["sample_id"].max()), value=0, step=1)
    row = df[df["sample_id"] == int(sample_id)].iloc[0]
    p = probs[int(sample_id)]

    if st.session_state.get("selected_sample") != int(sample_id):
        st.session_state["selected_sample"] = int(sample_id)
        st.session_state["opened_at"] = time.time()
        log_event(
            event_type="open_patient",
            source=source,
            sample_id=int(sample_id),
            payload_json=json.dumps({"sample_id": int(sample_id)}),
            session_id=st.session_state["session_id"],
        )

    col1, col2 = st.columns([1.1, 1.0])
    with col1:
        st.markdown(f"**Predicted:** `{row['pred_ens_name']}`")
        st.markdown(f"**True label:** `{row.get('true_label_name', 'n/a')}`")
        st.markdown(f"**Escalation:** `{row['escalation_level']}`")
        st.markdown(f"**Risk score:** `{float(row['review_risk_score']):.3f}` | **tau*** `{tau_star:.3f}`")
        st.markdown(f"**Reasons:** `{str(row.get('escalation_reasons', '') or 'none')}`")

        df_prob = pd.DataFrame({"class": CLASS_NAMES, "probability": p})
        fig_prob = px.bar(df_prob, x="class", y="probability", range_y=[0, 1], color="class")
        fig_prob.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

    with col2:
        # Try real SHAP first, fall back to surrogate (Issue #8)
        shap_contrib = _shap_feature_contrib(
            sample_idx=int(sample_id),
            pred_idx=int(row["pred_ens"]),
            feature_names=feature_names,
            source=source,
        )

        if shap_contrib is not None:
            chart_title = "SHAP Feature Attribution"
            contrib = shap_contrib
        else:
            chart_title = "Feature Contribution (Surrogate)"
            if raw_features is not None and int(sample_id) < len(raw_features):
                sample_raw = raw_features.iloc[int(sample_id)]
            else:
                sample_raw = pd.Series(X[int(sample_id)], index=feature_names)
            contrib = _surrogate_feature_contrib(
                sample_raw=sample_raw,
                df_raw=raw_features,
                pred_idx=int(row["pred_ens"]),
                probs_row=p,
            )

        fig_wf = px.bar(
            contrib,
            x="contribution",
            y="feature",
            orientation="h",
            color="contribution",
            color_continuous_scale="RdBu",
            title=chart_title,
        )
        fig_wf.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("**Biomarker Values**")
    if raw_features is not None and int(sample_id) < len(raw_features):
        vals = raw_features.iloc[int(sample_id)].to_dict()
    else:
        vals = {f: float(v) for f, v in zip(feature_names, X[int(sample_id)])}
    bio_df = pd.DataFrame(
        [{"feature": k, "value": vals.get(k, np.nan), "reference_range": REFERENCE_RANGES.get(k, "n/a")} for k in feature_names]
    )
    st.dataframe(bio_df, use_container_width=True)

    st.markdown("**Similar Patients (Nearest Neighbors)**")
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(X)
    _, idxs = nn.kneighbors(X[int(sample_id)].reshape(1, -1), return_distance=True)
    sim_ids = [int(i) for i in idxs[0] if int(i) != int(sample_id)][:5]
    sim = df[df["sample_id"].isin(sim_ids)][
        ["sample_id", "true_label_name", "pred_ens_name", "review_risk_score", "escalation_level"]
    ].copy()
    st.dataframe(sim.sort_values("review_risk_score", ascending=False), use_container_width=True)

    st.markdown("**Clinician Input Form**")
    with st.form(key=f"review_form_{source}_{int(sample_id)}"):
        agree = st.radio("Agree with AI?", ["Agree", "Disagree"], horizontal=True)
        corrected_label = st.selectbox("Correct diagnosis", CLASS_NAMES, index=CLASS_NAMES.index(row["pred_ens_name"]))
        confidence = st.slider("Clinician confidence", 1, 5, 3, 1)
        viewed_features = st.multiselect("Features viewed", feature_names, default=feature_names[:3])
        notes = st.text_area("Free-text notes", height=100)
        submitted = st.form_submit_button("Submit Review")

    if submitted:
        if agree == "Agree":
            corrected_label = row["pred_ens_name"]
        decision_seconds = None
        if "opened_at" in st.session_state:
            decision_seconds = max(0.0, time.time() - float(st.session_state["opened_at"]))

        log_clinician_feedback(
            source=source,
            sample_id=int(sample_id),
            predicted_label=row["pred_ens_name"],
            true_label=row.get("true_label_name", None),
            agree=(agree == "Agree"),
            corrected_label=corrected_label,
            clinician_confidence=int(confidence),
            notes=notes,
            decision_seconds=decision_seconds,
            viewed_features=viewed_features,
            escalation_level=row["escalation_level"],
            escalation_reasons=str(row.get("escalation_reasons", "") or ""),
            review_risk_score=float(row["review_risk_score"]),
            session_id=st.session_state["session_id"],
        )
        st.success("Feedback saved.")


def _load_feedback_df() -> pd.DataFrame:
    db = ensure_db(get_default_db_path())
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query("SELECT * FROM clinician_feedback ORDER BY timestamp_utc DESC", conn)
    finally:
        conn.close()
    return df


def render_analytics(df: pd.DataFrame, raw_features: pd.DataFrame | None):
    st.subheader("Analytics Dashboard")
    feedback = _load_feedback_df()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Escalated Cases", int((df["escalation_level"] != "AI-Autonomous").sum()))
    c2.metric("AI-only Accuracy", f"{df['ai_correct'].mean():.3f}" if "ai_correct" in df.columns else "n/a")
    c3.metric("Cases Reviewed by Clinicians", int(feedback.shape[0]))

    if not feedback.empty:
        fb = feedback.copy()
        fb["final_correct"] = (fb["corrected_label"] == fb["true_label"]).astype(int)
        fb["ai_correct"] = (fb["predicted_label"] == fb["true_label"]).astype(int)
        fb["timestamp_utc"] = pd.to_datetime(fb["timestamp_utc"], utc=True, errors="coerce")
        fb = fb.sort_values("timestamp_utc")
        fb["cum_ai_acc"] = fb["ai_correct"].expanding().mean()
        fb["cum_hitl_acc"] = fb["final_correct"].expanding().mean()
        curve = fb[["timestamp_utc", "cum_ai_acc", "cum_hitl_acc"]].melt(
            id_vars="timestamp_utc", var_name="mode", value_name="accuracy"
        )
        st.plotly_chart(px.line(curve, x="timestamp_utc", y="accuracy", color="mode"), use_container_width=True)

    esc_by_class = (
        df.groupby("true_label_name")["escalation_level"]
        .apply(lambda s: (s != "AI-Autonomous").mean())
        .reset_index(name="escalation_rate")
    )
    st.plotly_chart(px.bar(esc_by_class, x="true_label_name", y="escalation_rate"), use_container_width=True)

    if raw_features is not None and not feedback.empty:
        join = feedback.merge(raw_features.reset_index().rename(columns={"index": "sample_id"}), on="sample_id", how="left")
        if "PTGENDER" in join.columns:
            g = (
                join.assign(final_correct=(join["corrected_label"] == join["true_label"]).astype(int))
                .groupby("PTGENDER", as_index=False)["final_correct"]
                .mean()
            )
            st.plotly_chart(px.bar(g, x="PTGENDER", y="final_correct"), use_container_width=True)

    st.markdown("**Simulated Clinician Experiment**")
    e1, e2, e3, e4 = st.columns(4)
    sim_acc = float(e1.selectbox("Clinician accuracy", [0.85, 0.9, 0.95], index=1))
    batch = int(e2.number_input("Batch size", min_value=5, max_value=100, value=20, step=5))
    rounds = int(e3.number_input("Rounds", min_value=1, max_value=15, value=6, step=1))
    weighted = bool(e4.checkbox("Confidence-weighted retraining", value=True))
    if st.button("Run simulation"):
        curve = run_active_learning_simulation(
            clinician_accuracy=sim_acc,
            batch_size=batch,
            max_rounds=rounds,
            use_confidence_weighting=weighted,
            seed=42,
            fast=True,
        )
        st.plotly_chart(px.line(curve, x="round", y=["test_accuracy", "test_macro_f1"], markers=True), use_container_width=True)
        st.dataframe(curve, use_container_width=True)


def render_feedback_history():
    st.subheader("Feedback History")
    feedback = _load_feedback_df()
    if feedback.empty:
        st.info("No feedback saved yet.")
        return

    filt_agree = st.multiselect("Agreement", ["Agree", "Disagree"], default=["Agree", "Disagree"])
    tmp = feedback.copy()
    tmp["decision"] = tmp["agree"].map({1: "Agree", 0: "Disagree"})
    tmp = tmp[tmp["decision"].isin(filt_agree)]
    st.dataframe(tmp, use_container_width=True)
    st.download_button(
        "Export Feedback CSV",
        data=tmp.to_csv(index=False).encode("utf-8"),
        file_name="clinician_feedback_export.csv",
        mime="text/csv",
    )


def render_new_patient_inference():
    st.subheader("New Patient Inference (Unseen Data)")
    models, scaler, imputer, selector, meta_model, selected_features, tau_star, preproc, medians, _ = load_assets()

    mode = st.radio("Input Mode", ["Manual Entry", "CSV Upload"], horizontal=True)
    patient_df = None
    pre_imp_nan_mask = None

    if mode == "Manual Entry":
        vals = {}
        cols = st.columns(3)
        for i, f in enumerate(selected_features):
            vals[f] = float(cols[i % 3].number_input(f, value=float(medians.get(f, 0.0))))
        patient_df = pd.DataFrame([vals])[selected_features]
        # Manual entry: no missing data
        pre_imp_nan_mask = np.zeros((1, len(selected_features)), dtype=bool)
    else:
        up = st.file_uploader("Upload CSV with feature columns", type=["csv"])
        if up is not None:
            df_up = pd.read_csv(up)

            # Check if CSV has all raw features (needs full pipeline) or just selected 12
            all_raw_features = list(preproc.columns) if preproc is not None else []
            has_all_raw = all_raw_features and all(c in df_up.columns for c in all_raw_features if c != "DX")

            if has_all_raw:
                # Full pipeline: imputer -> selector -> scaler
                raw_cols = [c for c in all_raw_features if c != "DX"]
                X_raw = df_up[raw_cols].values.astype(float)
                pre_imp_nan_mask_raw = np.isnan(X_raw)
                X_imp = imputer.transform(X_raw)
                X_sel = selector.transform(X_imp)
                # Project NaN mask through selector
                selector_support = selector.get_support()
                pre_imp_nan_mask = pre_imp_nan_mask_raw[:, selector_support]
                patient_df = pd.DataFrame(X_sel, columns=selected_features)
                st.caption(f"Loaded {len(patient_df)} patient(s) - applied full imputer->selector pipeline.")
            else:
                missing = [f for f in selected_features if f not in df_up.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    return
                patient_df = df_up[selected_features].copy()
                # Track NaN before median fill
                pre_imp_nan_mask = patient_df.isna().values
                # Median-fill any NaN for 12-feature CSVs
                for f in selected_features:
                    if patient_df[f].isna().any():
                        patient_df[f] = patient_df[f].fillna(medians.get(f, 0.0))
                st.caption(f"Loaded {len(patient_df)} patient(s).")

    if patient_df is None:
        return

    if st.button("Run Prediction + HITL Policy"):
        X_scaled = scaler.transform(patient_df.values.astype(float))
        all_model_probs = _predict_ensemble_probs(models=models, X=X_scaled)
        probs_ens = np.mean(np.stack(list(all_model_probs.values()), axis=0), axis=0)

        # Extract core 3 for escalation engine backward compat
        probs_cat = all_model_probs.get("catboost", list(all_model_probs.values())[0])
        probs_rf = all_model_probs.get("rf", list(all_model_probs.values())[min(1, len(all_model_probs) - 1)])
        probs_nn = all_model_probs.get("nn", list(all_model_probs.values())[min(2, len(all_model_probs) - 1)])

        config = build_escalation_config(selected_features)

        df_case = run_batch_escalation_from_probabilities(
            X=X_scaled,
            y_true=None,
            probs_cat=probs_cat,
            probs_rf=probs_rf,
            probs_nn=probs_nn,
            probs_ens_override=probs_ens,
            class_names=CLASS_NAMES,
            config=config,
            nan_mask=pre_imp_nan_mask,
        ).reset_index()
        X_meta = _risk_features_from_df(df_case)
        df_case["review_risk_score"] = meta_model.predict_proba(X_meta)[:, 1]
        df_case["final_policy"] = np.where(df_case["review_risk_score"] >= tau_star, "ESCALATE", "AI-AUTONOMOUS")

        show_cols = [
            "sample_id",
            "pred_ens_name",
            "ens_max_prob",
            "ens_entropy",
            "escalation_level",
            "escalation_reasons",
            "review_risk_score",
            "final_policy",
        ]
        st.dataframe(df_case[show_cols], use_container_width=True)

        first_probs = pd.DataFrame({"class": CLASS_NAMES, "probability": probs_ens[0]})
        st.plotly_chart(px.bar(first_probs, x="class", y="probability", range_y=[0, 1], color="class"), use_container_width=True)
        st.success(f"Applied Step 2 + Step 3 policy with tau*={tau_star:.3f}.")


def main():
    st.title("HITL Clinician Dashboard")
    st.caption("Supports retrospective analysis and true unseen patient inference.")

    ensure_db(get_default_db_path())
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    models, _, _, _, _, selected_features, tau_star, _, _, _ = load_assets()
    _ = models  # keep models loaded in cache

    mode = st.sidebar.radio("Mode", ["Retrospective Review", "New Patient Inference"], index=1)

    if mode == "New Patient Inference":
        render_new_patient_inference()
        return

    source = st.sidebar.selectbox("Retrospective Source", ["testset", "oof"], index=0)
    df, probs, X, raw_features = load_retrospective_dataset(source=source)
    nav = st.sidebar.radio("View", ["Patient Queue", "Patient Detail / Review", "Analytics Dashboard", "Feedback History"])

    if nav == "Patient Queue":
        render_patient_queue(df)
    elif nav == "Patient Detail / Review":
        render_patient_detail(df, probs, X, selected_features, raw_features, tau_star, source=source)
    elif nav == "Analytics Dashboard":
        render_analytics(df, raw_features)
    else:
        render_feedback_history()


if __name__ == "__main__":
    main()
