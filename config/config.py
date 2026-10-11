# config/config.py
# =============================================================================
# SHARED CONFIGURATION - used by Paper 1, Paper 2, and every script
# Edit values here, never in individual scripts.
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (auto-derived from this file's location, works on any machine)
# -----------------------------------------------------------------------------
# This file lives at <repo_root>/config/config.py, so repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR         = REPO_ROOT / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"
EXTRACTED_DIR    = DATA_DIR / "extracted"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
FEATURES_DIR     = DATA_DIR / "features"
SPLITS_DIR       = DATA_DIR / "splits" / "paper1"

DOCS_DIR    = REPO_ROOT / "docs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR     = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results" / "paper1"

# -----------------------------------------------------------------------------
# ERA5 dataset
# -----------------------------------------------------------------------------
ERA5_DATASET = "reanalysis-era5-single-levels"

# -----------------------------------------------------------------------------
# ERA5 variables
# -----------------------------------------------------------------------------
# Canonical short keys (used downstream in the pipeline)
# mapped to the long CDS API names (used when building the request).
# -----------------------------------------------------------------------------
ERA5_VARIABLES = {
    # Radiation
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    # Temperature
    "t2m":  "2m_temperature",
    "d2m":  "2m_dewpoint_temperature",
    # Wind
    "u10":  "10m_u_component_of_wind",
    "v10":  "10m_v_component_of_wind",
    # Rain
    "tp":   "total_precipitation",
    # Cloud cover
    "tcc":  "total_cloud_cover",
    "lcc":  "low_cloud_cover",
    "mcc":  "medium_cloud_cover",
    "hcc":  "high_cloud_cover",
    # Cloud water
    "tclw": "total_column_cloud_liquid_water",
    "tciw": "total_column_cloud_ice_water",
    "tcwv": "total_column_water_vapour",
    # Boundary layer
    "blh":  "boundary_layer_height",
}

# The 15 long names, derived - this is what we send to CDS.
ERA5_VARIABLES_LONG = list(ERA5_VARIABLES.values())

# The 15 short names, derived - this is what downstream scripts expect.
ERA5_VARIABLES_SHORT = list(ERA5_VARIABLES.keys())

# -----------------------------------------------------------------------------
# Cities - bounding boxes in CDS order: North, West, South, East
# -----------------------------------------------------------------------------
CITIES = {
    "muntinlupa": {"north": 14.50, "west": 121.00, "south": 14.25, "east": 121.25},
    "baguio":     {"north": 16.50, "west": 120.50, "south": 16.25, "east": 120.75},
    "cebu":       {"north": 10.50, "west": 123.75, "south": 10.25, "east": 124.00},
    "davao":      {"north":  7.25, "west": 125.50, "south":  7.00, "east": 125.75},
}

# -----------------------------------------------------------------------------
# Years
# -----------------------------------------------------------------------------
YEARS = [2023, 2024, 2025]