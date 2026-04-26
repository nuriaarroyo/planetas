from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EARTH_DENSITY_G_CM3 = 5.514
DENSITY_FORMULA = "pl_dens = 5.514 * pl_bmasse / pl_rade**3"


@dataclass(frozen=True)
class DensityDerivationAudit:
    """Counts that make the density derivation traceable."""

    column_created: bool
    observed_before: int
    missing_before: int
    derived_count: int
    missing_after: int
    formula: str = DENSITY_FORMULA


@dataclass(frozen=True)
class KeplerDerivationAudit:
    """Counts that make the semimajor-axis derivation traceable."""

    observed_before: int
    missing_before: int
    derived_count: int
    missing_after: int
    formula: str = "pl_orbsmax = (st_mass * (pl_orbper / 365.25)**2)**(1/3)"


@dataclass(frozen=True)
class PhysicalDerivationAudit:
    """Audit for physical derivations executed before statistical imputation."""

    density: DensityDerivationAudit
    kepler: KeplerDerivationAudit


def derive_planet_density(df: pd.DataFrame) -> tuple[pd.DataFrame, DensityDerivationAudit]:
    """Fill missing ``pl_dens`` values from mass and radius when possible.

    Existing observed density values are preserved. The returned dataframe always
    includes ``pl_dens`` and ``pl_dens_source``.
    """

    if not {"pl_bmasse", "pl_rade"}.issubset(df.columns):
        missing = sorted({"pl_bmasse", "pl_rade"} - set(df.columns))
        raise KeyError(f"No se puede derivar pl_dens; faltan columnas: {missing}")

    out = df.copy()
    column_created = "pl_dens" not in out.columns
    if column_created:
        out["pl_dens"] = np.nan

    density = pd.to_numeric(out["pl_dens"], errors="coerce")
    mass = pd.to_numeric(out["pl_bmasse"], errors="coerce")
    radius = pd.to_numeric(out["pl_rade"], errors="coerce")

    observed_before = int(density.notna().sum())
    missing_before = int(density.isna().sum())
    valid_physics = np.isfinite(mass) & np.isfinite(radius) & (mass > 0) & (radius > 0)
    derived_mask = density.isna() & valid_physics
    derived_values = EARTH_DENSITY_G_CM3 * mass / radius.pow(3)
    density = density.mask(derived_mask, derived_values)

    source = pd.Series(pd.NA, index=out.index, dtype="object")
    source.loc[density.notna()] = "observed"
    source.loc[derived_mask] = "derived_density"
    source.loc[density.isna()] = pd.NA

    out["pl_dens"] = density
    out["pl_dens_source"] = source
    out.attrs["density_derivation"] = {
        "formula": DENSITY_FORMULA,
        "derived_count": int(derived_mask.sum()),
        "column_created": column_created,
    }

    audit = DensityDerivationAudit(
        column_created=column_created,
        observed_before=observed_before,
        missing_before=missing_before,
        derived_count=int(derived_mask.sum()),
        missing_after=int(density.isna().sum()),
    )
    return out, audit


def derive_semimajor_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, KeplerDerivationAudit]:
    """Fill missing ``pl_orbsmax`` from orbital period and stellar mass.

    Assumes ``pl_orbper`` is in days, ``st_mass`` in solar masses, and returns
    ``pl_orbsmax`` in AU. Planet mass is neglected relative to stellar mass.
    """

    out = df.copy()
    if "pl_orbsmax" not in out.columns:
        out["pl_orbsmax"] = np.nan

    semimajor = pd.to_numeric(out["pl_orbsmax"], errors="coerce")
    observed_before = int(semimajor.notna().sum())
    missing_before = int(semimajor.isna().sum())

    if {"pl_orbper", "st_mass"}.issubset(out.columns):
        period_days = pd.to_numeric(out["pl_orbper"], errors="coerce")
        stellar_mass = pd.to_numeric(out["st_mass"], errors="coerce")
        p_years = period_days / 365.25
        valid = (
            semimajor.isna()
            & np.isfinite(period_days)
            & np.isfinite(stellar_mass)
            & (period_days > 0)
            & (stellar_mass > 0)
        )
        derived_values = np.power(stellar_mass * p_years.pow(2), 1.0 / 3.0)
        semimajor = semimajor.mask(valid, derived_values)
    else:
        valid = pd.Series(False, index=out.index)

    source = pd.Series(pd.NA, index=out.index, dtype="object")
    source.loc[semimajor.notna()] = "observed"
    source.loc[valid] = "derived_kepler"
    source.loc[semimajor.isna()] = pd.NA

    out["pl_orbsmax"] = semimajor
    out["pl_orbsmax_source"] = source
    out.attrs["kepler_derivation"] = {
        "formula": "pl_orbsmax = (st_mass * (pl_orbper / 365.25)**2)**(1/3)",
        "derived_count": int(valid.sum()),
    }

    audit = KeplerDerivationAudit(
        observed_before=observed_before,
        missing_before=missing_before,
        derived_count=int(valid.sum()),
        missing_after=int(semimajor.isna().sum()),
    )
    return out, audit


def apply_physical_derivations(df: pd.DataFrame) -> tuple[pd.DataFrame, PhysicalDerivationAudit]:
    out, density_audit = derive_planet_density(df)
    out, kepler_audit = derive_semimajor_axis(out)
    return out, PhysicalDerivationAudit(density=density_audit, kepler=kepler_audit)
