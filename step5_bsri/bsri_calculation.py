
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CRASH_FILE = PROJECT_ROOT / "data" / "crash_segmented.csv"
SEGMENT_FILE = PROJECT_ROOT / "data" / "nh48_segments.csv"

BLACKSPOT_FILE = (
    PROJECT_ROOT / "step4_blackspot" /
    "results" / "fixed_500m_segment_assessment.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "step5_bsri" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_YEARS = [2023, 2024, 2025]

# Proposed project severity weights
WEIGHTS = {
    "Fatal": 5,
    "Grievous": 4,
    "Minor hospitalized": 3,
    "Minor non-hospitalized": 2,
    "No injury": 1
}

SEVERITY_ORDER = list(WEIGHTS.keys())

# ==================================================
# 2. LOAD DATA
# ==================================================

crashes = pd.read_csv(CRASH_FILE)
segments = pd.read_csv(SEGMENT_FILE)

print("\n========== BSRI CALCULATION ==========")

print("Crash dataset:", crashes.shape)
print("Road segment dataset:", segments.shape)

# ==================================================
# 3. VALIDATE REQUIRED COLUMNS
# ==================================================

required_crash = [
    "year",
    "severity",
    "total_killed",
    "total_grievous_recorded",
    "total_minor_recorded",
    "total_no_injury_recorded",
    "segment_id",
    "possible_duplicate"
]

required_segments = [
    "segment_id",
    "seg_length_m",
    "chain_start_m",
    "chain_end_m"
]

for col in required_crash:
    if col not in crashes.columns:
        raise ValueError(f"Missing crash column: {col}")

for col in required_segments:
    if col not in segments.columns:
        raise ValueError(f"Missing segment column: {col}")

# ==================================================
# 4. PREPARE CRASH DATA
# ==================================================

crashes["year"] = pd.to_numeric(
    crashes["year"], errors="coerce"
)

crashes["severity"] = (
    crashes["severity"].astype(str).str.strip()
)

crashes["total_killed"] = pd.to_numeric(
    crashes["total_killed"], errors="coerce"
).fillna(0)

crashes["possible_duplicate"] = pd.to_numeric(
    crashes["possible_duplicate"], errors="coerce"
).fillna(0)

study = crashes[
    crashes["year"].isin(STUDY_YEARS)
].copy()

print("\nStudy period:", STUDY_YEARS)
print("Study accident records:", len(study))

print("\nSeverity distribution:")
print(study["severity"].value_counts())

print(
    "\nPotential duplicate records:",
    int((study["possible_duplicate"] == 1).sum())
)

# ==================================================
# 5. FUNCTION TO CALCULATE SEGMENT SCORES
# ==================================================

def calculate_segment_scores(crash_data, segment_data):

    # Count accident records by severity and segment
    severity_counts = (
        crash_data
        .groupby(["segment_id", "severity"])
        .size()
        .unstack(fill_value=0)
    )

    severity_counts = severity_counts.reindex(
        columns=SEVERITY_ORDER,
        fill_value=0
    )

    severity_counts = severity_counts.reset_index()

    # Aggregate additional casualty information
    casualty_counts = (
        crash_data
        .groupby("segment_id")
        .agg(
            total_fatalities=("total_killed", "sum"),
            total_grievous_persons=(
                "total_grievous_recorded", "sum"
            ),
            total_minor_persons=(
                "total_minor_recorded", "sum"
            ),
            potential_duplicate_records=(
                "possible_duplicate",
                lambda x: int((x == 1).sum())
            )
        )
        .reset_index()
    )

    # Start with all road segments, including zero-crash segments
    result = segment_data.merge(
        severity_counts,
        on="segment_id",
        how="left"
    )

    result = result.merge(
        casualty_counts,
        on="segment_id",
        how="left"
    )

    # Fill missing counts with zero
    fill_columns = (
        SEVERITY_ORDER +
        [
            "total_fatalities",
            "total_grievous_persons",
            "total_minor_persons",
            "potential_duplicate_records"
        ]
    )

    result[fill_columns] = result[fill_columns].fillna(0)

    # Total accident records
    result["total_accidents"] = (
        result[SEVERITY_ORDER].sum(axis=1)
    )

    # Weighted severity score
    result["weighted_severity_points"] = sum(
        result[severity] * weight
        for severity, weight in WEIGHTS.items()
    )

    # Account for segment length
    result["weighted_points_per_km"] = (
        result["weighted_severity_points"] /
        (result["seg_length_m"] / 1000)
    )

    # Normalise to 0–100
    maximum = result["weighted_points_per_km"].max()

    if maximum > 0:
        result["BSRI"] = (
            result["weighted_points_per_km"] /
            maximum
        ) * 100
    else:
        result["BSRI"] = 0

    result["BSRI"] = result["BSRI"].round(2)

    return result

# ==================================================
# 6. PRIMARY BSRI: ALL RECORDS
# ==================================================

primary = calculate_segment_scores(
    study,
    segments
)

# ==================================================
# 7. SENSITIVITY ANALYSIS
# Exclude records flagged as possible duplicates.
# This is NOT the primary dataset.
# ==================================================

study_without_flagged = study[
    study["possible_duplicate"] != 1
].copy()

sensitivity = calculate_segment_scores(
    study_without_flagged,
    segments
)

sensitivity_scores = sensitivity[
    ["segment_id", "BSRI"]
].rename(
    columns={"BSRI": "BSRI_without_flagged_records"}
)

primary = primary.merge(
    sensitivity_scores,
    on="segment_id",
    how="left"
)

# ==================================================
# 8. ADD STEP 4 BLACKSPOT IDENTIFICATION
# ==================================================

if BLACKSPOT_FILE.exists():

    blackspots = pd.read_csv(BLACKSPOT_FILE)

    primary = primary.merge(
        blackspots[
            ["segment_id", "is_blackspot", "qualification"]
        ],
        on="segment_id",
        how="left"
    )

    primary["is_blackspot"] = (
        primary["is_blackspot"].fillna(False)
    )

else:
    primary["is_blackspot"] = False
    primary["qualification"] = "Not assessed"

# ==================================================
# 9. RANK ALL ROAD SEGMENTS
# ==================================================

primary = primary.sort_values(
    "BSRI",
    ascending=False
).reset_index(drop=True)

primary["BSRI_rank"] = (
    primary["BSRI"].rank(
        ascending=False,
        method="min"
    ).astype(int)
)

# ==================================================
# 10. SAVE RESULTS
# ==================================================

output_file = OUTPUT_DIR / "bsri_all_segments.csv"

primary.to_csv(output_file, index=False)

top_10 = primary.head(10)

top_10_file = OUTPUT_DIR / "bsri_top_10_segments.csv"

top_10.to_csv(top_10_file, index=False)

# ==================================================
# 11. VISUALISE TOP 15 SEGMENTS
# ==================================================

top_15 = primary.head(15).sort_values(
    "BSRI",
    ascending=True
)

plt.figure(figsize=(12, 7))

plt.barh(
    top_15["segment_id"],
    top_15["BSRI"]
)

plt.xlabel("BSRI Score (0–100)")
plt.ylabel("Road Segment")

plt.title(
    "NH-48: Top 15 Road Segments by BSRI"
)

plt.tight_layout()

plot_file = OUTPUT_DIR / "bsri_top_15_plot.png"

plt.savefig(plot_file, dpi=300)
plt.close()

# ==================================================
# 12. DISPLAY RESULTS
# ==================================================

print("\n========== BSRI RESULTS ==========")

print("Total road segments:", len(primary))

print("\nTop 15 highest-risk segments:")

display_columns = [
    "BSRI_rank",
    "segment_id",
    "chain_start_m",
    "chain_end_m",
    "total_accidents",
    "Fatal",
    "Grievous",
    "total_fatalities",
    "weighted_severity_points",
    "BSRI",
    "BSRI_without_flagged_records",
    "is_blackspot"
]

print(
    primary[display_columns]
    .head(15)
    .to_string(index=False)
)

print("\n========== FILES GENERATED ==========")

print(output_file)
print(top_10_file)
print(plot_file)

print("\nBSRI calculation completed successfully!")
