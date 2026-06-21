"""SHAP explainability.

Global (beeswarm, feature importance) + local (waterfall) explanations.
Powers the /explain API endpoint. See blueprint Bước 5.
"""

from __future__ import annotations


def explain_instance(model, x):
    """Return SHAP values + top factors for a single instance."""
    raise NotImplementedError
