from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogTransformAudit:
    """Per-feature counts created while preparing log-space variables."""

    rows: list[dict[str, object]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def log_feature_subset(features: Iterable[str], candidates: Iterable[str]) -> list[str]:
    candidate_set = set(candidates)
    return [feature for feature in features if feature in candidate_set]


def safe_log10(s: pd.Series) -> pd.Series:
    """Convert positive values to log10 and treat non-positive values as missing."""

    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return np.log10(values.where(values > 0))


def apply_log10_transform(
    matrix: pd.DataFrame,
    log_features: Iterable[str],
) -> tuple[pd.DataFrame, LogTransformAudit]:
    """Return a copy with selected positive variables in log10 space.

    Non-positive values in log columns are converted to missing values and
    audited. That keeps impossible physical values from silently entering KNN
    distances.
    """

    transformed = matrix.copy()
    log_set = set(log_features)
    rows: list[dict[str, object]] = []

    for column in transformed.columns:
        values = pd.to_numeric(transformed[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_before = int(values.isna().sum())
        nonpositive_count = 0
        if column in log_set:
            nonpositive = values.notna() & (values <= 0)
            nonpositive_count = int(nonpositive.sum())
            values = safe_log10(values)
        transformed[column] = values
        rows.append(
            {
                "feature": column,
                "log10_applied": column in log_set,
                "missing_before_transform": missing_before,
                "nonpositive_set_missing": nonpositive_count,
                "missing_after_transform": int(transformed[column].isna().sum()),
            }
        )

    return transformed, LogTransformAudit(rows)


def invert_log10_transform(
    matrix: pd.DataFrame,
    log_features: Iterable[str],
) -> pd.DataFrame:
    original = matrix.copy()
    for column in log_features:
        if column in original.columns:
            original[column] = np.power(10.0, pd.to_numeric(original[column], errors="coerce"))
    return original
