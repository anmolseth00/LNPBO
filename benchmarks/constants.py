"""Shared constants for benchmark scripts.

Single source of truth for seeds, batch size, study size thresholds, and
other values used across benchmark runners and analysis scripts.
"""

SEEDS = [42, 123, 456, 789, 2024]

MIN_STUDY_SIZE = 200
SEED_FRACTION = 0.25
BATCH_SIZE = 12
MAX_ROUNDS = 15
MIN_SEED = 30

# Statistical test thresholds
MIN_N_WILCOXON = 6  # Exact Wilcoxon tables require >= 6 non-zero differences
MIN_N_CORRELATION = 5  # Minimum for meaningful correlation (3 df)

# Human-readable labels for metadata dimensions
ASSAY_TYPE_LABELS = {
    "in_vitro_single_formulation": "In Vitro (Single)",
    "in_vitro_barcode_screen": "In Vitro (Barcode)",
    "in_vivo_liver": "In Vivo (Liver)",
    "in_vivo_other": "In Vivo (Other)",
    "unknown": "Unknown",
}
