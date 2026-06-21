"""Preprocessing pipeline (Sklearn Pipeline + ColumnTransformer).

All preprocessing lives here so transforms are reproducible and never
fit on the test set (no data leakage). See blueprint Bước 2.
"""

from __future__ import annotations


def build_preprocessor():
    """Build the Sklearn ColumnTransformer / Pipeline.

    Returns a fitted-able preprocessing pipeline covering:
      - numeric imputation + scaling
      - categorical imputation + encoding
    """
    raise NotImplementedError
