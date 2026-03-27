# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Configuration for Step 2 escalation logic.

All values here are simple, interpretable defaults that you can tune
or later optimise in Step 3.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# 1. Class names (must match your y labels order)
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = ["SCD", "MCI", "AD"]


# ---------------------------------------------------------------------------
# 2. Canonical feature-to-modality mapping
# ---------------------------------------------------------------------------

# Maps every possible raw feature name to its modality group and direction.
# direction: +1 = higher values indicate higher severity,
#            -1 = lower values indicate higher severity.
FEATURE_GROUP_MAP: Dict[str, Dict] = {
    # Biomarkers
    "TAU": {"group": "biomarker", "direction": 1},
    "PTAU": {"group": "biomarker", "direction": 1},
    "ABETA": {"group": "biomarker", "direction": -1},
    "PLASMA_NFL": {"group": "biomarker", "direction": 1},
    "FDG": {"group": "biomarker", "direction": -1},
    "AV45": {"group": "biomarker", "direction": 1},
    "ABETA42": {"group": "biomarker", "direction": -1},
    "TAU_ABETA_RATIO": {"group": "biomarker", "direction": 1},
    "PTAU_ABETA_RATIO": {"group": "biomarker", "direction": 1},
    # Cognitive
    "MMSE": {"group": "cognitive", "direction": -1},
    "CDRSB": {"group": "cognitive", "direction": 1},
    "ADAS11": {"group": "cognitive", "direction": 1},
    "ADAS13": {"group": "cognitive", "direction": 1},
    "ADASQ4": {"group": "cognitive", "direction": 1},
    "RAVLT_immediate": {"group": "cognitive", "direction": -1},
    "RAVLT_learning": {"group": "cognitive", "direction": -1},
    "RAVLT_forgetting": {"group": "cognitive", "direction": 1},
    "RAVLT_perc_forgetting": {"group": "cognitive", "direction": 1},
    "FAQ": {"group": "cognitive", "direction": 1},
    "MOCA": {"group": "cognitive", "direction": -1},
    "TRABSCOR": {"group": "cognitive", "direction": 1},
    # Demographic
    "AGE": {"group": "demographic", "direction": 1},
    "PTGENDER": {"group": "demographic", "direction": 0},
    "PTEDUCAT": {"group": "demographic", "direction": -1},
    "PTETHCAT": {"group": "demographic", "direction": 0},
    "PTRACCAT": {"group": "demographic", "direction": 0},
    "APOE4": {"group": "demographic", "direction": 1},
    # Volumetric / imaging
    "Ventricles": {"group": "biomarker", "direction": 1},
    "Hippocampus": {"group": "biomarker", "direction": -1},
    "WholeBrain": {"group": "biomarker", "direction": -1},
    "Entorhinal": {"group": "biomarker", "direction": -1},
    "Fusiform": {"group": "biomarker", "direction": -1},
    "MidTemp": {"group": "biomarker", "direction": -1},
    "ICV": {"group": "biomarker", "direction": 0},
}


def build_multimodal_groups(selected_features: List[str]) -> Dict[str, Dict]:
    """
    Build multimodal feature groups from the list of selected feature names.

    Returns a dict suitable for EscalationConfig.multimodal_feature_groups:
        {"biomarker": {"indices": [...], "direction": 1},
         "cognitive": {"indices": [...], "direction": -1}, ...}
    """
    groups: Dict[str, List[int]] = {}
    directions: Dict[str, List[float]] = {}

    for idx, feat in enumerate(selected_features):
        info = FEATURE_GROUP_MAP.get(feat)
        if info is None:
            continue
        grp = info["group"]
        d = info["direction"]
        groups.setdefault(grp, []).append(idx)
        directions.setdefault(grp, []).append(d)

    result = {}
    for grp, indices in groups.items():
        dirs = directions[grp]
        # Use majority direction; 0 means no directional signal
        nonzero_dirs = [d for d in dirs if d != 0]
        avg_dir = sum(nonzero_dirs) / len(nonzero_dirs) if nonzero_dirs else 1.0
        result[grp] = {"indices": indices, "direction": 1.0 if avg_dir >= 0 else -1.0}

    return result


# ---------------------------------------------------------------------------
# 3. Thresholds for uncertainty + missing-data rules
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyThresholds:
    # if max predicted probability below this -> low confidence
    low_confidence: float = 0.70
    # if gap between top-1 and top-2 probs below this -> ambiguous / small margin
    small_margin: float = 0.20
    # if entropy above this -> high uncertainty
    high_entropy: float = 1.00


@dataclass
class MissingDataThresholds:
    # if fraction of missing features above this -> "critical" missingness
    critical_missing_fraction: float = 0.30


@dataclass
class EscalationConfig:
    uncertainty: UncertaintyThresholds
    missing: MissingDataThresholds
    multimodal_feature_groups: Dict = field(default_factory=dict)


def build_escalation_config(selected_features: List[str] | None = None) -> EscalationConfig:
    """Build an EscalationConfig with populated multimodal groups."""
    groups = {}
    if selected_features is not None:
        groups = build_multimodal_groups(selected_features)
    else:
        # Try to load from artifacts
        features_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "artifacts", "selected_features.json"
        )
        if os.path.exists(features_path):
            with open(features_path, "r", encoding="utf-8") as f:
                selected_features = json.load(f)
            groups = build_multimodal_groups(selected_features)

    return EscalationConfig(
        uncertainty=UncertaintyThresholds(),
        missing=MissingDataThresholds(),
        multimodal_feature_groups=groups,
    )


# Default config (empty groups for backwards compat; use build_escalation_config for populated groups)
ESCALATION_CONFIG = EscalationConfig(
    uncertainty=UncertaintyThresholds(),
    missing=MissingDataThresholds(),
    multimodal_feature_groups={},
)
