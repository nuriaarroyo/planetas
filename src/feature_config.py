"""Feature definitions for the ExoData Challenge analysis."""

IDENTIFIER_COLUMNS = [
    "rowid",
    "pl_name",
    "hostname",
    "pl_letter",
    "hd_name",
    "hip_name",
    "tic_id",
    "gaia_dr2_id",
    "gaia_dr3_id",
]

CONTEST_KEY_COLUMNS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "pl_rade",
    "pl_radj",
    "pl_bmasse",
    "pl_bmassj",
    "pl_dens",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "st_teff",
    "st_met",
    "sy_pnum",
    "pl_insol",
    "pl_eqt",
]

CONTEST_NUMERIC_COLUMNS = [
    "pl_rade",
    "pl_bmasse",
    "pl_dens",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "st_teff",
    "st_met",
    "sy_pnum",
    "pl_insol",
    "pl_eqt",
]

MAPPER_PHYS_FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_dens",
]

MAPPER_ORB_FEATURES = [
    "pl_orbper",
    "pl_orbsmax",
    "pl_insol",
    "pl_eqt",
]

MAPPER_ORB_OPTIONAL_FEATURES = [
    "pl_orbeccen",
]

MAPPER_JOINT_FEATURES = (
    MAPPER_PHYS_FEATURES
    + MAPPER_ORB_FEATURES
)

STELLAR_CONTEXT_FEATURES = [
    "st_teff",
    "st_met",
    "st_mass",
    "st_rad",
    "st_lum",
    "sy_pnum",
    "sy_snum",
]

OBSERVATIONAL_CONTROL_COLUMNS = [
    "discoverymethod",
    "disc_year",
    "disc_facility",
    "sy_dist",
    "tran_flag",
    "rv_flag",
    "ima_flag",
    "micro_flag",
]

MAPPER_ORBITAL_FEATURES = MAPPER_ORB_FEATURES
MAPPER_STELLAR_SYSTEM_FEATURES = STELLAR_CONTEXT_FEATURES
MAPPER_HABITABILITY_FEATURES = ["pl_insol", "pl_eqt"]
MAPPER_BIAS_COLUMNS = OBSERVATIONAL_CONTROL_COLUMNS

MAPPER_CORE_FEATURES = (
    *MAPPER_PHYS_FEATURES,
    *MAPPER_ORB_FEATURES,
)

MAPPER_WIDE_FEATURES = [
    *MAPPER_CORE_FEATURES,
    *MAPPER_ORB_OPTIONAL_FEATURES,
    *STELLAR_CONTEXT_FEATURES,
]

IMPUTATION_FEATURE_SETS = {
    "mapper_phys": MAPPER_PHYS_FEATURES,
    "mapper_core": MAPPER_CORE_FEATURES,
    "mapper_wide": MAPPER_WIDE_FEATURES,
}

CLUSTERING_FEATURE_SETS = {
    "mapper_phys": MAPPER_PHYS_FEATURES,
    "mapper_core": MAPPER_CORE_FEATURES,
    "mapper_wide": MAPPER_WIDE_FEATURES,
    "pdf_core_no_mass_density": [
        "pl_rade",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "st_teff",
        "st_met",
        "sy_pnum",
    ],
    "pdf_core_with_mass_density": [
        "pl_rade",
        "pl_bmasse",
        "pl_dens",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "st_teff",
        "st_met",
        "sy_pnum",
    ],
    "habitability": [
        "pl_rade",
        "pl_orbper",
        "pl_orbsmax",
        "st_teff",
        "pl_insol",
        "pl_eqt",
        "sy_pnum",
    ],
    "wide_physical": [
        "pl_rade",
        "pl_bmasse",
        "pl_dens",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "pl_insol",
        "pl_eqt",
        "st_teff",
        "st_met",
        "sy_pnum",
    ],
}

DUPLICATE_UNIT_GROUPS = {
    "planet_radius": ["pl_rade", "pl_radj"],
    "planet_mass": ["pl_bmasse", "pl_bmassj"],
}

NON_MODEL_SUFFIXES = (
    "err1",
    "err2",
    "lim",
    "reflink",
    "refname",
    "systemref",
)

NON_MODEL_SUBSTRINGS = (
    "err1",
    "err2",
    "lim",
    "reflink",
    "refname",
    "systemref",
)

LOG10_FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_dens",
    "pl_orbper",
    "pl_orbsmax",
    "pl_insol",
    "sy_dist",
    "st_mass",
    "st_rad",
    "st_lum",
]

LOG_CANDIDATE_COLUMNS = LOG10_FEATURES

IMPUTATION_VALUE_BOUNDS = {
    "pl_rade": (0.0, None),
    "pl_bmasse": (0.0, None),
    "pl_dens": (0.0, None),
    "pl_orbper": (0.0, None),
    "pl_orbsmax": (0.0, None),
    "pl_orbeccen": (0.0, 1.0),
    "st_teff": (0.0, None),
    "st_mass": (0.0, None),
    "st_rad": (0.0, None),
    "sy_pnum": (1.0, None),
    "sy_snum": (1.0, None),
    "pl_insol": (0.0, None),
    "pl_eqt": (0.0, None),
    "sy_dist": (0.0, None),
}

RADIUS_BINS = [-float("inf"), 1.25, 2.0, 4.0, 10.0, float("inf")]
RADIUS_LABELS = [
    "Tierra (<1.25 R_earth)",
    "Supertierra (1.25-2)",
    "Subneptuno (2-4)",
    "Neptuno/transitorio (4-10)",
    "Gigante gaseoso (>10)",
]
