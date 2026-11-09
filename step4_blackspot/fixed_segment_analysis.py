
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CRASH_FILE = PROJECT_ROOT / "data" / "crash_segmented.csv"
SEGMENT_FILE = PROJECT_ROOT / "data" / "nh48_segments.csv"

OUTPUT_DIR = PROJECT_ROOT / "step4_blackspot" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_YEARS = [2023, 2024, 2025]

MIN_FATAL_GRIEVOUS_ACCIDENTS = 5
MIN_FATALITIES = 10

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

crashes = pd.read_csv(CRASH_FILE)
segments = pd.read_csv(SEGMENT_FILE)

print("\n========== FIXED 500 M BLACKSPOT ANALYSIS ==========")

print("Crash records:", len(crashes))
print("Road segments:", len(segments))

# Check required columns
required_crash = [
    "year", "severity", "total_killed", "segment_id"
]

required_segments = [
    "segment_id", "chain_start_m", "chain_end_m",
    "seg_length_m"
]

for col in required_crash:
    if col not in crashes.columns:
        raise ValueError(f"Missing crash column: {col}")

for col in required_segments:
    if col not in segments.columns:
        raise ValueError(f"Missing segment column: {col}")

# --------------------------------------------------
# PREPARE CRASH DATA
# --------------------------------------------------

crashes["year"] = pd.to_numeric(
    crashes["year"], errors="coerce"
)

crashes["total_killed"] = pd.to_numeric(
    crashes["total_killed"], errors="coerce"
).fillna(0)

crashes["severity"] = (
    crashes["severity"].astype(str).str.strip().str.lower()
)

# Use only the three selected calendar years
study_crashes = crashes[
    crashes["year"].isin(STUDY_YEARS)
].copy()

# Fatal or grievous accident records
study_crashes["fatal_grievous"] = (
    study_crashes["severity"].isin(["fatal", "grievous"])
)

print("\nStudy period:", STUDY_YEARS)
print("Accident records:", len(study_crashes))

# --------------------------------------------------
# COUNT ACCIDENTS WITHIN EACH FIXED ROAD SEGMENT
# --------------------------------------------------

counts = (
    study_crashes
    .groupby("segment_id")
    .agg(
        total_accidents=("segment_id", "size"),
        fatal_grievous_accidents=("fatal_grievous", "sum"),
        total_fatalities=("total_killed", "sum")
    )
    .reset_index()
)

# Merge counts with all road segments.
# This retains segments with zero accidents.
result = segments.merge(
    counts,
    on="segment_id",
    how="left"
)

# Fill empty segment counts with zero
count_columns = [
    "total_accidents",
    "fatal_grievous_accidents",
    "total_fatalities"
]

result[count_columns] = result[count_columns].fillna(0)

# --------------------------------------------------
# APPLY MoRTH CRITERIA
# --------------------------------------------------

result["criterion_1_met"] = (
    result["fatal_grievous_accidents"]
    >= MIN_FATAL_GRIEVOUS_ACCIDENTS
)

result["criterion_2_met"] = (
    result["total_fatalities"] >= MIN_FATALITIES
)

result["is_blackspot"] = (
    result["criterion_1_met"] |
    result["criterion_2_met"]
)

result["qualification"] = "Does not qualify"

result.loc[
    result["criterion_1_met"] &
    result["criterion_2_met"],
    "qualification"
] = "Both criteria"

result.loc[
    result["criterion_1_met"] &
    ~result["criterion_2_met"],
    "qualification"
] = "5+ fatal/grievous accidents"

result.loc[
    ~result["criterion_1_met"] &
    result["criterion_2_met"],
    "qualification"
] = "10+ fatalities"

# --------------------------------------------------
# SORT AND SAVE RESULTS
# --------------------------------------------------

result = result.sort_values(
    "fatal_grievous_accidents",
    ascending=False
)

all_file = OUTPUT_DIR / "fixed_500m_segment_assessment.csv"

qualifying = result[result["is_blackspot"]].copy()

qualifying_file = (
    OUTPUT_DIR / "fixed_500m_qualifying_segments.csv"
)

result.to_csv(all_file, index=False)
qualifying.to_csv(qualifying_file, index=False)

# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

print("\n========== RESULTS ==========")

print("Total road segments:", len(result))

print(
    "Qualifying fixed segments:",
    len(qualifying)
)

print("\nQualification distribution:")
print(qualifying["qualification"].value_counts())

print("\nTop qualifying segments:")

display_columns = [
    "segment_id",
    "chain_start_m",
    "chain_end_m",
    "seg_length_m",
    "total_accidents",
    "fatal_grievous_accidents",
    "total_fatalities",
    "qualification"
]

print(
    qualifying[display_columns]
    .to_string(index=False)
)

print("\n========== FILES GENERATED ==========")
print(all_file)
print(qualifying_file)

print("\nFixed-segment assessment completed.")
