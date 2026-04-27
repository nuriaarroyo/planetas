from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


RADIUS_CLASSES = [
    "rocky_size",
    "sub_neptune_size",
    "neptune_or_sub_jovian_size",
    "jovian_size",
    "unknown",
]

DENSITY_CLASSES = [
    "high_density",
    "intermediate_density",
    "low_density",
    "unknown",
]

ORBIT_CLASSES = [
    "short_period",
    "intermediate_period",
    "long_period",
    "unknown",
]

THERMAL_CLASSES = [
    "very_hot",
    "hot",
    "warm",
    "cool",
    "unknown",
]

CANDIDATE_POPULATIONS = [
    "hot_jupiter_candidate",
    "warm_or_cool_giant_candidate",
    "super_earth_candidate",
    "sub_neptune_candidate",
    "rocky_candidate",
    "long_period_giant_candidate",
    "unknown_mixed",
]


def _numeric(series: pd.Series, column: str) -> pd.Series:
    if column not in series.index and not isinstance(series, pd.DataFrame):
        return pd.Series([np.nan])
    return pd.to_numeric(series[column], errors="coerce") if isinstance(series, pd.DataFrame) else pd.to_numeric(series, errors="coerce")


def classify_radius_class(pl_rade: float | None) -> str:
    if pd.isna(pl_rade):
        return "unknown"
    if pl_rade < 1.6:
        return "rocky_size"
    if pl_rade < 4.0:
        return "sub_neptune_size"
    if pl_rade < 8.0:
        return "neptune_or_sub_jovian_size"
    return "jovian_size"


def classify_density_class(pl_dens: float | None) -> str:
    if pd.isna(pl_dens):
        return "unknown"
    if pl_dens >= 5.0:
        return "high_density"
    if pl_dens >= 2.0:
        return "intermediate_density"
    return "low_density"


def classify_orbit_class(pl_orbper: float | None) -> str:
    if pd.isna(pl_orbper):
        return "unknown"
    if pl_orbper < 10:
        return "short_period"
    if pl_orbper < 100:
        return "intermediate_period"
    return "long_period"


def classify_thermal_class(pl_eqt: float | None) -> str:
    if pd.isna(pl_eqt):
        return "unknown"
    if pl_eqt >= 1500:
        return "very_hot"
    if pl_eqt >= 800:
        return "hot"
    if pl_eqt >= 300:
        return "warm"
    return "cool"


def classify_candidate_population(pl_rade: float | None, pl_bmasse: float | None, pl_dens: float | None, pl_orbper: float | None) -> str:
    if pd.notna(pl_rade) and pd.notna(pl_orbper) and pl_rade >= 8.0 and pl_orbper < 10:
        return "hot_jupiter_candidate"
    if pd.notna(pl_rade) and pd.notna(pl_orbper) and pl_rade >= 8.0 and pl_orbper >= 10:
        return "warm_or_cool_giant_candidate"
    if pd.notna(pl_rade) and pd.notna(pl_bmasse) and 1.0 <= pl_rade < 1.6 and pl_bmasse < 10:
        return "super_earth_candidate"
    if pd.notna(pl_rade) and 1.6 <= pl_rade < 4.0:
        return "sub_neptune_candidate"
    if pd.notna(pl_rade) and pd.notna(pl_dens) and pl_rade < 1.6 and pl_dens >= 3.5:
        return "rocky_candidate"
    if pd.notna(pl_rade) and pd.notna(pl_orbper) and pl_rade >= 4.0 and pl_orbper >= 100:
        return "long_period_giant_candidate"
    return "unknown_mixed"


def add_planet_physical_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["radius_class"] = pd.to_numeric(out.get("pl_rade"), errors="coerce").map(classify_radius_class)
    out["density_class"] = pd.to_numeric(out.get("pl_dens"), errors="coerce").map(classify_density_class)
    out["orbit_class"] = pd.to_numeric(out.get("pl_orbper"), errors="coerce").map(classify_orbit_class)
    out["thermal_class"] = pd.to_numeric(out.get("pl_eqt"), errors="coerce").map(classify_thermal_class)
    out["candidate_population"] = [
        classify_candidate_population(r, m, d, p)
        for r, m, d, p in zip(
            pd.to_numeric(out.get("pl_rade"), errors="coerce"),
            pd.to_numeric(out.get("pl_bmasse"), errors="coerce"),
            pd.to_numeric(out.get("pl_dens"), errors="coerce"),
            pd.to_numeric(out.get("pl_orbper"), errors="coerce"),
        )
    ]

    density_source = out.get("pl_dens_source", pd.Series("", index=out.index)).astype("string").fillna("")
    out["density_derived_sensitive"] = (
        density_source.isin(["derived_density", "derived_kepler"])
        | density_source.str.startswith("imputed_", na=False)
    )

    thermal_imputed_cols = [column for column in ["pl_eqt_was_imputed", "pl_insol_was_imputed"] if column in out.columns]
    orbital_imputed_cols = [column for column in ["pl_orbper_was_imputed", "pl_orbsmax_was_imputed"] if column in out.columns]
    out["thermal_imputed_flag"] = (
        out.loc[:, thermal_imputed_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(bool).any(axis=1)
        if thermal_imputed_cols
        else False
    )
    out["orbital_imputed_flag"] = (
        out.loc[:, orbital_imputed_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(bool).any(axis=1)
        if orbital_imputed_cols
        else False
    )
    return out


def label_fraction(frame: pd.DataFrame, column: str, label: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    values = frame[column].astype("string").fillna("unknown")
    return float(values.eq(label).mean())


def dominant_label(frame: pd.DataFrame, column: str) -> tuple[str | None, float]:
    if column not in frame.columns or frame.empty:
        return None, 0.0
    counts = frame[column].astype("string").fillna("unknown").value_counts(normalize=True)
    if counts.empty:
        return None, 0.0
    return str(counts.index[0]), float(counts.iloc[0])


def label_entropy(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    counts = frame[column].astype("string").fillna("unknown").value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())
