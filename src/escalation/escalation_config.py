"""
Configuration for Step 2 escalation logic.

All values here are simple, interpretable defaults that you can tune
or later optimise in Step 3.
"""

from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# 1. Class names (must match your y labels order)
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = ["SCD", "MCI", "AD"]


# ---------------------------------------------------------------------------
# 2. Thresholds for uncertainty + missing-data rules
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyThresholds:
    # if max predicted probability below this → low confidence
    low_confidence: float = 0.70
    # if gap between top-1 and top-2 probs below this → ambiguous / small margin
    small_margin: float = 0.20
    # if entropy above this → high uncertainty
    high_entropy: float = 1.00


@dataclass
class MissingDataThresholds:
    # if fraction of missing features above this → “critical” missingness
    critical_missing_fraction: float = 0.30


@dataclass
class EscalationConfig:
    uncertainty: UncertaintyThresholds
    missing: MissingDataThresholds
    # future multi-modal extension; for now left empty
    multimodal_feature_groups: Dict[str, List[int]]  # e.g. {"biomarker": [...], "cognitive": [...]}


ESCALATION_CONFIG = EscalationConfig(
    uncertainty=UncertaintyThresholds(),
    missing=MissingDataThresholds(),
    multimodal_feature_groups={
        # Fill these lists with feature indices later if you want a real
        # biomarker-vs-cognitive multimodal mismatch check.
        # "biomarker": [0, 1, 2],
        # "cognitive": [3, 4, 5],
    },
)
