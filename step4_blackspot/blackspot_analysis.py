
"""
NH-48 Blackspot Identification
MoRTH 2015 protocol
Study period: 2023–2025
Window length: 500 metres
Sliding step: 50 metres

Original datasets are read-only. All outputs are saved separately.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CRASH_FILE = PROJECT_ROOT / "data" / "crash_segmented.csv"
SEGMENT_FILE = PROJECT_ROOT / "data" / "nh48_segments.csv"

OUTPUT_DIR = PROJECT_ROOT / "step4_blackspot" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_YEARS = [2023, 2024, 2025]

WINDOW_LENGTH_M = 500
SLIDING_STEP_M = 50

# MoRTH blackspot thresholds
MIN_FATAL_GRIEVOUS_ACCIDENTS = 5
MIN_FATALITIES = 10


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

print("\n========== NH-48 BLACKSPOT IDENTIFICATION ==========")

crashes = pd.read_csv(CRASH_FILE)
segments = pd.read_csv(SEGMENT_FILE)

print("\nCrash dataset shape:", crashes.shape)
print("Road segments shape:", segments.shape)


# ---------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------------

required_crash_columns = [
    "accident_id",
    "year",
    "severity",
    "chainage_m",
    "total_killed"
]

required_segment_columns = [
    "chain_start_m",
    "chain_end_m"
]

for column in required_crash_columns:
    if column not in crashes.columns:
        raise ValueError(f"Missing crash dataset column: {column}")

for column in required_segment_columns:
    if column not in segments.columns:
        raise ValueError(f"Missing segment dataset column: {column}")


# ---------------------------------------------------------
# 4. PREPARE THE DATA
# ---------------------------------------------------------

crashes["year"] = pd.to_numeric(
    crashes["year"], errors="coerce"
)

crashes["chainage_m"] = pd.to_numeric(
    crashes["chainage_m"], errors="coerce"
)

crashes["total_killed"] = pd.to_numeric(
    crashes["total_killed"], errors="coerce"
).fillna(0)

crashes["severity"] = (
    crashes["severity"].astype(str).str.strip()
)

# Select the three complete calendar years
study_data = crashes[
    crashes["year"].isin(STUDY_YEARS)
].copy()

# Remove records without usable chainage
study_data = study_data.dropna(
    subset=["chainage_m"]
)

# Identify fatal and grievous injury accidents
study_data["fatal_grievous"] = (
    study_data["severity"].str.lower().isin(
        ["fatal", "grievous"]
    )
)

print("\nStudy period:", STUDY_YEARS)
print("Accidents in selected years:", len(study_data))

print("\nSeverity distribution:")
print(study_data["severity"].value_counts())


# ---------------------------------------------------------
# 5. DETERMINE THE CORRIDOR LIMITS
# ---------------------------------------------------------

corridor_start = float(
    segments["chain_start_m"].min()
)

corridor_end = float(
    segments["chain_end_m"].max()
)

print("\nCorridor start:", corridor_start, "m")
print("Corridor end:", corridor_end, "m")
print(
    "Corridor length:",
    round(corridor_end - corridor_start, 2),
    "m"
)


# ---------------------------------------------------------
# 6. GENERATE 500 M SLIDING WINDOWS
# ---------------------------------------------------------

last_window_start = corridor_end - WINDOW_LENGTH_M

window_starts = np.arange(
    corridor_start,
    last_window_start + 0.001,
    SLIDING_STEP_M
)

# Ensure the final possible window is included
if (
    len(window_starts) == 0
    or not np.isclose(
        window_starts[-1], last_window_start
    )
):
    window_starts = np.append(
        window_starts, last_window_start
    )

results = []

# Sort once for consistent processing
study_data = study_data.sort_values("chainage_m")

for start in window_starts:

    end = start + WINDOW_LENGTH_M

    # Select accidents inside the current 500 m window.
    # The left boundary is included; right boundary excluded.
    window_data = study_data[
        (study_data["chainage_m"] >= start)
        & (study_data["chainage_m"] < end)
    ]

    total_accidents = len(window_data)

    serious_accidents = int(
        window_data["fatal_grievous"].sum()
    )

    fatalities = int(
        window_data["total_killed"].sum()
    )

    # MoRTH: either condition is sufficient
    condition_1 = (
        serious_accidents
        >= MIN_FATAL_GRIEVOUS_ACCIDENTS
    )

    condition_2 = fatalities >= MIN_FATALITIES

    is_blackspot = condition_1 or condition_2

    if condition_1 and condition_2:
        qualification = "Both criteria"

    elif condition_1:
        qualification = "5+ fatal/grievous accidents"

    elif condition_2:
        qualification = "10+ fatalities"

    else:
        qualification = "Does not qualify"

    results.append({
        "window_start_m": round(start, 2),
        "window_end_m": round(end, 2),
        "window_start_km": round(start / 1000, 3),
        "window_end_km": round(end / 1000, 3),
        "total_accidents": total_accidents,
        "fatal_grievous_accidents": serious_accidents,
        "total_fatalities": fatalities,
        "criterion_1_met": condition_1,
        "criterion_2_met": condition_2,
        "is_blackspot": is_blackspot,
        "qualification": qualification
    })


# ---------------------------------------------------------
# 7. CREATE THE RESULTS TABLE
# ---------------------------------------------------------

window_results = pd.DataFrame(results)

candidates = window_results[
    window_results["is_blackspot"]
].copy()

# Sort by severity of the qualifying window
candidates = candidates.sort_values(
    by=[
        "fatal_grievous_accidents",
        "total_fatalities"
    ],
    ascending=False
)

# Save all evaluated windows
all_windows_path = (
    OUTPUT_DIR / "all_sliding_windows.csv"
)

window_results.to_csv(
    all_windows_path, index=False
)

# Save only qualifying windows
candidate_path = (
    OUTPUT_DIR / "qualifying_blackspot_windows.csv"
)

candidates.to_csv(
    candidate_path, index=False
)


# ---------------------------------------------------------
# 8. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n========== ANALYSIS RESULTS ==========")

print("Total windows evaluated:", len(window_results))

print("Qualifying windows:", len(candidates))

if candidates.empty:

    print("\nNo 500 m window satisfies either MoRTH criterion.")

else:

    print("\nQualifying blackspot windows:")

    display_columns = [
        "window_start_km",
        "window_end_km",
        "total_accidents",
        "fatal_grievous_accidents",
        "total_fatalities",
        "qualification"
    ]

    print(
        candidates[display_columns].to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# 9. CREATE A VISUALISATION
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    window_results["window_start_km"],
    window_results["fatal_grievous_accidents"],
    label="Fatal + Grievous Accidents"
)

ax.axhline(
    y=MIN_FATAL_GRIEVOUS_ACCIDENTS,
    linestyle="--",
    label="MoRTH threshold: 5 accidents"
)

ax.set_xlabel("500 m Window Start (km)")
ax.set_ylabel("Fatal + Grievous Accident Count")

ax.set_title(
    "NH-48: Fatal and Grievous Accidents "
    "Across Sliding 500 m Windows (2023–2025)"
)

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()

plot_path = (
    OUTPUT_DIR / "blackspot_sliding_window_plot.png"
)

plt.savefig(plot_path, dpi=300)
plt.close()


# ---------------------------------------------------------
# 10. COMPLETION SUMMARY
# ---------------------------------------------------------

print("\n========== FILES GENERATED ==========")

print("All windows:", all_windows_path)
print("Qualifying windows:", candidate_path)
print("Plot:", plot_path)

print("\nStep 4 initial sliding-window analysis completed.")
