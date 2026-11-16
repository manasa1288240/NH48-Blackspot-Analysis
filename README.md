# NH-48 Blackspot Analysis
### Data-Driven Road Safety Assessment | Kadodara–Kosamba Highway Corridor, Gujarat, India

A data-driven road safety analysis project focused on identifying high-risk accident locations, quantifying segment-level crash risk, studying crash severity, and exploring early-warning approaches along a section of National Highway 48 (NH-48) in Gujarat, India.

---

## 1. Project Overview

Road accidents are a major public safety concern, particularly along national highways where high traffic volumes, mixed road users, and varying road conditions can contribute to severe crashes.

This project analyses accident patterns along an approximately **39 km stretch of NH-48 between Kadodara and Kosamba, Gujarat**.

The objective is to move beyond simply counting accidents and develop a structured approach to understanding:

- Where are accidents concentrated along the highway?
- Which road segments exhibit the highest accident severity?
- Which locations satisfy the selected blackspot identification criteria?
- How can accident severity and location-based risk be quantified?
- Can historical crash patterns help identify segments that may become high-risk in the future?
- How can the results be communicated through an interactive visualisation and scenario analysis tool?

The project combines exploratory data analysis, geospatial segmentation, blackspot identification, a Blackspot Severity Risk Index (BSRI), machine learning models, and an interactive Streamlit dashboard.

## 2. Study Area

| Parameter | Description |
|---|---|
| Highway | National Highway 48 (NH-48) |
| Study corridor | Kadodara–Kosamba |
| Location | Gujarat, India |
| Approximate corridor length | 39 km |
| Road segmentation | Fixed 500 m segments |
| Number of road segments | 77 |

The highway corridor is represented using sequential road segments, allowing crash records to be associated with specific locations and enabling comparison of accident patterns across the study area.

## 3. Project Objectives

1. **Exploratory Data Analysis:** Understand accident frequency, temporal patterns, crash severity, collision types, visibility conditions, and other contributing factors.
2. **Blackspot Identification:** Identify candidate high-risk road stretches using the selected MoRTH blackspot identification criteria.
3. **Blackspot Severity Risk Index:** Develop a segment-level risk score that accounts for accident severity and road segment length.
4. **Crash Prediction and Early Warning:** Investigate whether historical accident patterns can help identify segments likely to meet the blackspot criterion in a subsequent observation window.
5. **Crash Severity Modelling:** Examine the relationship between crash characteristics and the likelihood of fatal or grievous outcomes.
6. **Interactive Dashboard:** Present the analysis results through an interactive interface.
7. **What-if Analysis:** Explore hypothetical road-safety interventions and their estimated effects on risk indicators.

## 4. Project Workflow and Methodology

The project is organised into sequential analytical stages.

### Step 3 — Exploratory Data Analysis (EDA)

**Folder:** `step3_eda/`

The exploratory analysis examines the crash dataset to understand the distribution and characteristics of accidents.

The analysis includes:

- Crashes by year and month.
- Crashes by time of day.
- Distribution of accident severity.
- Collision types and nature of accidents.
- Light and visibility conditions in relation to severity.
- Traffic violations and initial observations.
- Crash distribution across road segments.
- Pedestrian-related crashes along the highway.

The generated visualisations are saved as PNG files for further interpretation and reporting.

### Step 4 — Blackspot Identification

**Folder:** `step4_blackspot/`

This stage identifies candidate blackspot locations using the selected Ministry of Road Transport and Highways (MoRTH) 2015 protocol.

The analysis considers a three-calendar-year study period: **2023–2025**.

The selected identification criteria are:

- **Criterion 1:** Five or more road accidents involving fatalities or grievous injuries within the specified three-year period.
- **Criterion 2:** Ten or more fatalities within the specified three-year period.

A road segment qualifies if it satisfies either criterion.

Two approaches are implemented:

1. **Fixed 500 m segment analysis:** Evaluates each predefined road segment independently.
2. **Sliding-window analysis:** Examines overlapping 500 m stretches using a 50 m sliding interval to identify localised accident concentrations.

The analysis identified **34 candidate qualifying fixed road segments**.

These are project-level analytical findings and should not be interpreted as officially notified government blackspots.

### Step 5 — Blackspot Severity Risk Index (BSRI)

**Folder:** `step5_bsri/`

Blackspot identification indicates whether a location satisfies the selected criteria. However, it does not fully describe the relative severity of risk across all road segments.

To address this, the project develops a Blackspot Severity Risk Index (BSRI).

The index uses project-defined severity weights:

| Accident Severity | Weight |
|---|---:|
| Fatal | 5 |
| Grievous | 4 |
| Minor hospitalized | 3 |
| Minor non-hospitalized | 2 |
| No injury | 1 |

**Weighted Severity Score**

W = (5 × Fatal) + (4 × Grievous) + (3 × Minor hospitalized) + (2 × Minor non-hospitalized) + (1 × No injury)

**Length-adjusted Severity Score**

Weighted Points per km = W / Segment Length (km)

**Normalised BSRI**

BSRI = (Segment Weighted Points per km / Maximum Weighted Points per km) × 100

The resulting index ranges from 0 to 100, with higher values representing greater relative severity-weighted crash concentration per kilometre within the analysed corridor.

The BSRI is a project-developed comparative indicator, not an official MoRTH risk score.

The analysis covers 77 road segments and 474 accident records from 2023–2025.

The highest-ranked segment in the committed results is **S019 (9.0–9.5 km)**, with a BSRI of 100.

### Step 6 — Machine Learning Models

**Folder:** `step6_model/`

This stage explores two distinct modelling problems.

**A. Early-Warning Model**

The early-warning model investigates whether the first two years of crash information can help identify road segments that will meet the fatal-plus-grievous accident criterion over a subsequent three-year observation window.

The implementation includes:

- Logistic regression.
- Historical fatal-plus-grievous crash count baselines.
- Weighted severity-based ranking.
- Temporal backtesting.
- Forward watchlists.
- AUC, precision and recall evaluation.
- Bootstrap-based uncertainty estimates.

The model uses historical information to estimate future blackspot qualification rather than simply ranking segments using the same period being evaluated.

**B. Crash-Severity Model**

The severity model examines whether crash characteristics are associated with fatal or grievous outcomes.

The target variable is:

- 1 — Fatal or grievous crash.
- 0 — Minor or no-injury crash.

Models implemented:

- Base-rate baseline.
- Logistic regression.
- Histogram-based gradient boosting classifier.

The models are evaluated using ROC-AUC and Brier score, with uncertainty intervals also calculated.

Importantly, this is a crash-severity association model, not a causal model or a real-time accident prediction system.

### Interactive Dashboard

**Folder:** `Nh48-dashboard/`

The Streamlit dashboard presents the previously generated analytical results in an interactive format.

Its features include:

- Corridor-level crash risk information.
- BSRI-based segment rankings.
- Interactive road-segment visualisation.
- Blackspot identification information.
- Contributing-factor analysis.
- BSRI sensitivity analysis.
- What-if intervention scenarios.

The dashboard reads precomputed results from its `data/` folder. It does not independently retrain the models or recalculate the original BSRI analysis.

The map requires an internet connection for its online base layer.

---

## 5. Repository Structure

The repository is organised as follows:

```text
NH48-Blackspot-Analysis/
│
├── Nh48-dashboard/
│   ├── app.py
│   ├── whatif.py
│   ├── requirements.txt
│   │
│   └── data/
│       ├── bsri_all_segments.csv
│       ├── bsri_sensitivity_summary.csv
│       ├── bsri_top_10_segments.csv
│       ├── bsri_top_segment_factors.csv
│       ├── factor_intervention_matrix.csv
│       ├── intervention_effects.csv
│       └── nh48_segments.geojson
│
├── step3_eda/
│   ├── eda_charts.py
│   └── reports/
│       └── figures/
│
├── step4_blackspot/
│   ├── blackspot_analysis.py
│   ├── fixed_segment_analysis.py
│   └── results/
│       ├── all_sliding_windows.csv
│       ├── blackspot_sliding_window_plot.png
│       ├── fixed_500m_qualifying_segments.csv
│       ├── fixed_500m_segment_assessment.csv
│       └── qualifying_blackspot_windows.csv
│
├── step5_bsri/
│   ├── bsri_calculation.py
│   └── results/
│       ├── bsri_all_segments.csv
│       ├── bsri_top_10_segments.csv
│       └── bsri_top_15_plot.png
│
├── step6_model/
│   ├── early_warning_model.py
│   ├── severity_model.py
│   └── results/
│       ├── early_warning_metrics.csv
│       ├── ew_backtest_2023_26.csv
│       ├── ew_calibration.csv
│       ├── ew_forward_watchlist_2024_27.csv
│       ├── ew_metrics_no_duplicates.csv
│       ├── ew_metrics_primary.csv
│       ├── severity_metrics.csv
│       └── severity_odds_ratios.csv
│
├── .gitignore
├── README.md
└── requirements.txt
```

*Note: The repository also contains additional generated results and supporting files that are not all expanded in the tree above.*

---

## 6. Dataset Information and Availability

The project uses crash records and road-segment information supplied for the study.

### Primary datasets

| Dataset | Description |
|---|---|
| `crash_clean.csv` | Cleaned accident records |
| `crash_segmented.csv` | Accident records associated with NH-48 road segments |
| `nh48_segments.csv` | Road-segment information, including segment IDs and chainage boundaries |

### Dataset summary

- Full crash dataset: 599 records.
- Records used in the 2023–2025 analysis: 474.
- Total road segments: 77.
- Fixed segment length: generally 500 m, with a shorter final segment.

### Data availability

**The original crash datasets are not included in this public GitHub repository.**

They are shared separately with authorised project members.

To run the full analysis scripts, obtain the approved datasets from the project team and create the following directory in the repository root:

```text
NH48-Blackspot-Analysis/
│
└── data/
    ├── crash_clean.csv
    ├── crash_segmented.csv
    └── nh48_segments.csv
```

Keep the original filenames unchanged.

The dashboard has its own precomputed datasets inside `Nh48-dashboard/data/`. These are included in the repository, so the dashboard can be run without downloading the original crash datasets.

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- Visual Studio Code or another Python-compatible editor.
- Internet access for installing dependencies and loading the dashboard's online map layer.

### Step 1 — Clone the repository

```bash
git clone https://github.com/manasa1288240/NH48-Blackspot-Analysis.git
```

Navigate into the project:

```bash
cd NH48-Blackspot-Analysis
```

### Step 2 — Install dashboard dependencies

To run the interactive dashboard, execute:

```bash
python -m pip install -r Nh48-dashboard/requirements.txt
```

### Step 3 — Install analysis dependencies

To run the analytical Python scripts, install:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

The dashboard dependencies and analysis dependencies can be installed in the same Python environment.

---

## 8. Running the Project

Run commands from the main repository directory.

### A. Launch the interactive dashboard

After installing the dashboard dependencies:

```bash
python -m streamlit run Nh48-dashboard/app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

The dashboard uses the precomputed files in `Nh48-dashboard/data/`.

The following files are required:

- `bsri_all_segments.csv`
- `bsri_top_segment_factors.csv`
- `bsri_sensitivity_summary.csv`

The GeoJSON map and additional what-if analysis files are also included in the repository.

### B. Run Exploratory Data Analysis

First obtain the segmented crash dataset.

Then run:

```bash
python step3_eda/eda_charts.py "data/crash_segmented.csv"
```

The generated figures are saved in:

```text
reports/figures/
```

### C. Run Blackspot Identification

Ensure that the original datasets are available in the root `data/` directory.

Run the fixed-segment analysis:

```bash
python step4_blackspot/fixed_segment_analysis.py
```

Run the sliding-window analysis:

```bash
python step4_blackspot/blackspot_analysis.py
```

Generated results are saved in:

```text
step4_blackspot/results/
```

### D. Run BSRI Calculation

The BSRI calculation uses the crash dataset, road-segment dataset, and the fixed-segment assessment produced in Step 4.

Run:

```bash
python step5_bsri/bsri_calculation.py
```

Generated results are saved in:

```text
step5_bsri/results/
```

### E. Run the Early-Warning Model

Ensure that both the segmented crash dataset and road-segment dataset are available.

Run:

```bash
python step6_model/early_warning_model.py
```

### F. Run the Crash-Severity Model

Ensure that the segmented crash dataset is available.

Run:

```bash
python step6_model/severity_model.py
```

Model outputs are saved in:

```text
step6_model/results/
```

**Important:** Steps 3–6 require the relevant original datasets. The dashboard can be launched independently using its bundled precomputed data.

---

## 9. Results and Key Outputs

The repository contains generated outputs from the different analytical stages.

| Stage | Important outputs |
|---|---|
| EDA | Eight exploratory visualisations |
| Blackspot identification | Fixed-segment assessment and qualifying blackspot results |
| BSRI | Segment-level risk scores, top-ranked segments and visualisation |
| Early warning | Model evaluation metrics, backtesting and forward watchlists |
| Severity modelling | Model metrics and odds-ratio results |
| Dashboard | Interactive maps, rankings and what-if analysis |

### Selected analytical findings

- The fixed-segment blackspot analysis identified 34 candidate qualifying segments.
- The BSRI analysis ranked segment S019 (9.0–9.5 km) highest, with a score of 100.
- The early-warning model was evaluated against simpler historical-count and severity-based ranking approaches.
- The crash-severity analysis compared logistic regression and gradient boosting with a base-rate baseline.

The committed model evaluation results should be consulted directly for detailed performance comparisons. A higher model score does not automatically imply better real-world predictive performance.

---

## 10. Important Assumptions and Limitations

The findings must be interpreted in the context of the available data and project methodology.

1. **Dataset coverage:** The analysis is limited to the crash records made available for the study corridor.
2. **Reporting bias:** Accident datasets may underrepresent certain crash types, particularly less severe incidents.
3. **Study period:** Different analytical stages use different observation windows. The fixed blackspot and BSRI analyses use 2023–2025, while the early-warning and severity models use their explicitly defined temporal splits.
4. **Segment-level analysis:** The results depend on the chosen road segmentation and chainage boundaries.
5. **Project-defined BSRI:** Severity weights are proposed for this project and should not be interpreted as official government weights.
6. **Blackspot classification:** Identified locations are analytical candidates based on the selected criteria, not necessarily officially notified blackspots.
7. **Prediction limitations:** Historical patterns and model evaluation results do not guarantee future crash outcomes.
8. **Association versus causation:** Factors associated with severe crashes should not automatically be interpreted as causes.
9. **What-if scenarios:** Simulated intervention effects depend on the assumptions and parameters used in the simulator. They are not measured real-world intervention outcomes.

---

## 11. Technologies Used

| Technology | Application |
|---|---|
| Python | Data analysis and modelling |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Machine learning |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| GeoJSON | Geospatial road-segment representation |
| Git and GitHub | Version control and collaboration |

---

## 12. Future Scope

Potential extensions of the project include:

- Incorporating additional years of accident data.
- Integrating traffic volume and exposure-based risk measures.
- Incorporating road geometry, lighting, and infrastructure characteristics.
- Evaluating the models on larger and independently collected datasets.
- Improving spatial and temporal validation.
- Integrating real-time traffic and weather information.
- Validating simulated interventions using actual before-and-after safety data.
- Extending the dashboard to support additional highway corridors.

---

## 13. Contributors

This project is a collaborative effort involving students from multiple engineering disciplines.

Contributions include data preparation, exploratory analysis, geospatial processing, blackspot identification, risk-index development, machine learning, dashboard development, and road-safety interpretation.

---
