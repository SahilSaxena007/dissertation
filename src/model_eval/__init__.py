# Author: Sahil Saxena (11409565)
# University of Manchester, Department of Computer Science, 2025

"""
Evaluation module: Plug-and-play components for model assessment (11-component framework).
"""

from . import metrics
from . import visualizations
from . import uncertainty
from . import statistical_inference
from . import error_taxonomy
from . import explainability
from . import bias_diagnostics
from . import reporting
# DO NOT import performance_analysis here — it imports from sibling modules

__all__ = [
    "metrics",
    "visualizations",
    "uncertainty",
    "statistical_inference",
    "error_taxonomy",
    "explainability",
    "bias_diagnostics",
    "reporting",

]