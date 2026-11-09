
# NH-48 BLACKSPOT ANALYSIS PROJECT
### Kadodara–Kosamba Highway Corridor

Welcome to the NH-48 Blackspot Analysis Project!

This README is the common instruction manual for everyone working on this project. It explains the project structure, datasets, installation, how to run code, where to save files, how to use previous task results, and how to upload work to GitHub.

The project analyses road crashes along the approximately 39 km NH-48 corridor between Kadodara and Kosamba. The work is divided into tasks so that team members can work independently while maintaining consistency across the project.

---

# 1. IMPORTANT PROJECT RULES

Before starting your work, read these rules carefully.

1. Do NOT modify, rename, overwrite, or delete the original CSV files inside the `data/` folder.
2. Upload both your source code AND important generated results to GitHub.
3. Save your Python scripts inside your assigned task folder.
4. Save generated CSV files, graphs, model files, and other outputs inside your task's `results/` folder.
5. Use the existing road `segment_id` values. Do not create a new road segmentation unless approved by the project lead.
6. The established study period for Tasks 4 and 5 is 2023, 2024, and 2025.
7. Do not independently change the severity weights, blackspot criteria, thresholds, or assumptions used in earlier tasks.
8. Do not overwrite another member's scripts or results.
9. If a required file, column, or result is missing, ask the project lead instead of assuming or inventing it.
10. Document your methodology, assumptions, commands, results, and limitations.

IMPORTANT:

GitHub should contain the completed project work, including source code, important output CSVs, graphs, and instructions.

The original datasets are shared separately and are not uploaded to GitHub.

---

# 2. PROJECT FOLDER STRUCTURE

The project should follow this structure:

```text
NH48_Blackspot_Project/
│
├── data/                              # Original datasets (shared separately)
│   ├── crash_clean.csv
│   ├── crash_segmented.csv
│   └── nh48_segments.csv
│
├── step3_eda/
│   └── results/
│
├── step4_blackspot/
│   ├── blackspot_analysis.py
│   ├── fixed_segment_analysis.py
│   └── results/
│       ├── fixed_500m_segment_assessment.csv
│       ├── fixed_500m_qualifying_segments.csv
│       └── other Step 4 outputs
│
├── step5_bsri/
│   ├── bsri_calculation.py
│   └── results/
│       ├── bsri_all_segments.csv
│       ├── bsri_top_10_segments.csv
│       └── bsri_top_15_plot.png
│
├── step6_prediction/
│   └── results/
│
├── step7_dashboard/
│
├── step8_simulator/
│   └── results/
│
├── reports/
│   └── figures/
│
├── .gitignore
└── README.md
```

Some later task folders may not exist yet. If your assigned folder is missing, create it.

## PURPOSE OF EACH FOLDER

| Folder | Purpose |
|---|---|
| `data/` | Original project datasets |
| `step3_eda/` | Exploratory Data Analysis |
| `step4_blackspot/` | Blackspot identification code and results |
| `step5_bsri/` | BSRI calculation code and results |
| `step6_prediction/` | Prediction model code and results |
| `step7_dashboard/` | Streamlit dashboard code |
| `step8_simulator/` | What-if simulator code and results |
| `reports/` | Final project report |
| `reports/figures/` | Figures used in the final report |

## WHERE SHOULD I SAVE MY FILES?

For example, if you are working on Task 6:

- Save your Python scripts inside `step6_prediction/`.
- Save generated CSV files inside `step6_prediction/results/`.
- Save graphs inside `step6_prediction/results/figures/` or `results/`.
- Save model files inside `step6_prediction/results/` if required.

Follow the same structure for your assigned task.

Every task member must upload their scripts and important generated results to GitHub.

---

# 3. HOW TO DOWNLOAD AND OPEN THE PROJECT

Ask the project lead for the correct NH-48 GitHub repository link.

## OPTION A: CLONE USING GIT

1. Install Git if it is not already installed.
2. Open VS Code.
3. Select Terminal → New Terminal.
4. Execute the following command after replacing the placeholder with the actual repository URL:

```bash
git clone <NH48_GITHUB_REPOSITORY_URL>
```

5. Open the downloaded project folder in VS Code.

## OPTION B: DOWNLOAD ZIP

1. Open the NH-48 GitHub repository.
2. Click Code.
3. Select Download ZIP.
4. Extract the downloaded ZIP file.
5. Open the extracted folder in VS Code.

If the project folder is already on your computer, simply open that existing folder. Do not create unnecessary duplicate copies.

---

# 4. DATASET INFORMATION

The project contains three main CSV files.

| Dataset | Description |
|---|---|
| `crash_clean.csv` | Cleaned accident records |
| `crash_segmented.csv` | Accident records mapped to existing road segments |
| `nh48_segments.csv` | Road segment information, including segment IDs and length/geometry attributes |

## HOW TO GET THE DATASETS

The original datasets are NOT uploaded to GitHub.

1. Ask the project lead for the approved datasets.
2. Open the main project folder.
3. Create a folder named `data` if it does not exist.
4. Place all three CSV files inside it.
5. Keep their filenames exactly as given.
6. Do not modify the original files.

Your folder should look like this:

```text
NH48_Blackspot_Project/
└── data/
    ├── crash_clean.csv
    ├── crash_segmented.csv
    └── nh48_segments.csv
```

IMPORTANT:

Cloning the GitHub repository will not automatically download these CSV files. You must obtain them separately from the project lead.

## DATASET FACTS

- Full crash dataset: 599 records.
- Study period used in Tasks 4 and 5: 2023–2025.
- Records in the 2023–2025 study period: 474.
- Total road segments: 77.
- Most segments are 500 metres long.
- The final segment is shorter.

The full-period `crash_count` field should not be confused with `total_accidents` calculated for the 2023–2025 BSRI study period.

## HOW TO INSPECT THE DATASET

Before writing code, inspect the actual column names.

Example:

```python
import pandas as pd

df = pd.read_csv("data/crash_segmented.csv")

print(df.head())
print(df.columns.tolist())
print(df.shape)
```

Never assume a column exists without checking.

If you need to filter or transform a dataset, save a NEW file inside your task's results folder. Never overwrite the original input CSV.

---

# 5. PYTHON INSTALLATION AND LIBRARY SETUP

Python is required to execute the project scripts.

## STEP 1: CHECK PYTHON

Open the VS Code terminal and run:

```bash
python --version
```

If that does not work, try:

```bash
py --version
```

If neither command works, install Python and make sure Add Python to PATH is selected during installation.

## STEP 2: INSTALL COMMON PROJECT LIBRARIES

Execute:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn streamlit
```

If your computer uses `py` instead of `python`, execute:

```bash
py -m pip install pandas numpy matplotlib seaborn scipy scikit-learn streamlit
```

If you encounter:

`ModuleNotFoundError: No module named 'package_name'`

Install the missing package using:

```bash
python -m pip install package_name
```

Replace `package_name` with the actual missing library.

Not every task requires every library.

---

# 6. HOW TO RUN PYTHON FILES

IMPORTANT: Always open the terminal from the MAIN PROJECT FOLDER.

The main project folder is the folder containing:

- data/
- step3_eda/
- step4_blackspot/
- step5_bsri/
- README.md

In VS Code:

1. Open the main project folder.
2. Select Terminal → New Terminal.
3. Make sure the terminal path ends with `NH48_Blackspot_Project`.
4. Run your assigned Python script.

### RUNNING TASK 4

```bash
python step4_blackspot/fixed_segment_analysis.py
```

### RUNNING TASK 5

```bash
python step5_bsri/bsri_calculation.py
```

### RUNNING TASK 6

For example, if your script is named `train_model.py`:

```bash
python step6_prediction/train_model.py
```

Use the actual filename created by the Task 6 member.

### RUNNING TASK 7 DASHBOARD

For example, if the main dashboard file is named `app.py`:

```bash
streamlit run step7_dashboard/app.py
```

### RUNNING TASK 8

The command depends on the actual simulator script filename.

For example:

```bash
python step8_simulator/simulator.py
```

---

# 7. TASK-WISE RESPONSIBILITIES

| Task | Work | Main folder |
|---|---|---|
| Task 1 | Project planning and initial preparation | Project root / relevant folder |
| Task 2 | Data cleaning and road segmentation | Relevant task folder |
| Task 3 | Exploratory Data Analysis | `step3_eda/` |
| Task 4 | Blackspot Identification | `step4_blackspot/` |
| Task 5 | BSRI Calculation | `step5_bsri/` |
| Task 6 | Prediction Model | `step6_prediction/` |
| Task 7 | Streamlit Dashboard | `step7_dashboard/` |
| Task 8 | What-if Simulator | `step8_simulator/` |
| Final | Report preparation and integration | `reports/` |

## TASK 4: BLACKSPOT IDENTIFICATION

Folder:

`step4_blackspot/`

The purpose of this stage is to identify candidate accident blackspots along NH-48.

The analysis uses:

- The 2023–2025 study period.
- Existing 500-metre road segments.
- MoRTH blackspot identification criteria.

The criteria applied are:

1. Five or more accident records involving fatalities or grievous injuries during the previous three calendar years.
2. Ten or more fatalities during the previous three calendar years.

Either criterion qualifies a segment.

Current output files:

```text
step4_blackspot/results/
│
├── fixed_500m_segment_assessment.csv
├── fixed_500m_qualifying_segments.csv
├── all_sliding_windows.csv
├── qualifying_blackspot_windows.csv
└── blackspot_sliding_window_plot.png
```

The fixed-segment analysis identified 34 candidate qualifying segments.

IMPORTANT: These are candidate blackspots identified using the university project dataset. They should not be described as officially notified government blackspots.

## TASK 5: BLACKSPOT RISK INDEX (BSRI)

Folder:

`step5_bsri/`

The purpose of this stage is to calculate a relative risk score for every road segment.

The proposed severity weights are:

| Accident Severity | Weight |
|---|---:|
| Fatal | 5 |
| Grievous | 4 |
| Minor hospitalized | 3 |
| Minor non-hospitalized | 2 |
| No injury | 1 |

Weighted Severity Score:

`W = (5 × Fatal) + (4 × Grievous) + (3 × Minor hospitalized) + (2 × Minor non-hospitalized) + (1 × No injury)`

Length-adjusted score:

`Weighted Points per km = W / Segment Length in km`

Normalised BSRI:

`BSRI = (Segment Weighted Points per km / Maximum Weighted Points per km) × 100`

These are project-defined weights, not official MoRTH weights.

Current output files:

```text
step5_bsri/results/
│
├── bsri_all_segments.csv
├── bsri_top_10_segments.csv
└── bsri_top_15_plot.png
```

The analysis covers 77 road segments and 474 accident records from 2023–2025.

The highest-ranked segment is S019, between 9.0 km and 9.5 km, with a BSRI of 100.

## TASK 6: PREDICTION MODEL

Folder:

`step6_prediction/`

The purpose of this stage is to develop a suitable prediction model based on the approved project brief.

The Task 6 member should:

1. Read the assigned Task 6 instructions.
2. Understand exactly what needs to be predicted.
3. Inspect the available datasets and actual column names.
4. Identify suitable input features.
5. Decide the prediction target based on the task brief and available data.
6. Prepare the data for modelling.
7. Split the data into training and testing sets appropriately.
8. Train a suitable machine learning model.
9. Evaluate the model using appropriate metrics.
10. Compare it with a baseline where appropriate.
11. Generate predictions.
12. Save the code and results in the assigned folder.

Suggested structure:

```text
step6_prediction/
│
├── train_model.py
├── predict.py
├── README.md
│
└── results/
    ├── model_metrics.csv
    ├── predictions.csv
    └── figures/
```

These filenames are suggestions. The Task 6 member may choose suitable names.

IMPORTANT:

Do not assume the prediction target without checking the task brief.

Do not invent data, labels, features, or model performance.

Do not use information from the test set during model training.

If the available dataset does not support the intended prediction target, discuss this with the project lead before proceeding.

## TASK 7: STREAMLIT DASHBOARD

Folder:

`step7_dashboard/`

The purpose of this stage is to create an interactive dashboard presenting the project findings.

The dashboard should use results generated by earlier tasks.

The dashboard member should:

- Read the relevant output CSV files.
- Display accident statistics and visualisations.
- Present blackspot information.
- Present BSRI rankings.
- Keep dashboard code inside `step7_dashboard/`.
- Avoid modifying original datasets and previous task outputs.
- Document the command required to run the dashboard.

## TASK 8: WHAT-IF SIMULATOR

Folder:

`step8_simulator/`

The purpose of this stage is to develop a simulator that allows users to explore hypothetical road-safety scenarios.

The Task 8 member should:

- Identify the parameters that can be changed.
- Define the scenarios to be simulated.
- Clearly distinguish actual data from hypothetical values.
- Explain the assumptions used.
- Generate and present simulation results.
- Save code and important outputs inside `step8_simulator/`.

---

# 8. USING RESULTS FROM PREVIOUS TASKS

IMPORTANT:

Later team members should use the results already generated by earlier tasks.

They do NOT need to rerun every earlier task just to obtain the results.

For example, the Task 6 and Task 7 members can directly use the BSRI output file:

```python
import pandas as pd

bsri = pd.read_csv(
    "step5_bsri/results/bsri_all_segments.csv"
)

print(bsri.head())
print(bsri.columns.tolist())
```

They can also directly read the blackspot results:

```python
blackspots = pd.read_csv(
    "step4_blackspot/results/fixed_500m_qualifying_segments.csv"
)

print(blackspots.head())
```

These examples read the existing files. They do not modify them.

## WHEN SHOULD I RERUN AN EARLIER TASK?

Rerun an earlier task only when:

- You need to reproduce or verify its results.
- The project lead approves a change in methodology or input data.
- The earlier task owner updates the code or results.
- Your task specifically requires a fresh run.

If you rerun an earlier script, check its output path and coordinate with the task owner before overwriting existing results.

---

# 9. WHAT SHOULD BE UPLOADED TO GITHUB?

EVERY TASK MEMBER SHOULD UPLOAD:

1. Source code.
2. Important generated CSV files.
3. Important graphs and visualisations.
4. Model files if required and reasonably sized.
5. A short explanation of how to run the code and understand the outputs.

For example:

```text
step4_blackspot/
│
├── blackspot_analysis.py
├── fixed_segment_analysis.py
│
└── results/
    ├── fixed_500m_segment_assessment.csv
    ├── fixed_500m_qualifying_segments.csv
    ├── all_sliding_windows.csv
    └── blackspot_sliding_window_plot.png
```

```text
step5_bsri/
│
├── bsri_calculation.py
│
└── results/
    ├── bsri_all_segments.csv
    ├── bsri_top_10_segments.csv
    └── bsri_top_15_plot.png
```

## WHAT SHOULD NOT BE UPLOADED?

Do not upload:

- Original protected datasets from `data/`.
- Passwords, API keys, or credentials.
- Python cache files.
- Virtual environments.
- Temporary files that are not needed.
- Very large model files without checking with the project lead.

If an output or model file is too large for normal GitHub storage, discuss another storage method with the project lead.

---

# 10. GITIGNORE AND DATA PROTECTION

The root `.gitignore` should contain:

```gitignore
__pycache__/
*.pyc
.venv/
venv/
.vscode/
data/
.DS_Store
```

IMPORTANT:

Do NOT add `results/` or `*.csv` to `.gitignore`.

Otherwise, important generated results may not be uploaded to GitHub.

The `data/` line keeps the original datasets out of GitHub. The project lead must share those files separately.

If the original datasets were already committed to GitHub before adding this rule, adding `data/` to `.gitignore` alone does not remove previously tracked files. Ask the project lead to handle this carefully.

---

# 11. GITHUB COLLABORATION: HOW TO UPLOAD YOUR WORK

Every team member should work on their own branch.

## STEP 1: GET THE LATEST CHANGES

Open the terminal from the main project folder.

```bash
git pull
```

## STEP 2: CREATE YOUR OWN BRANCH

For example, Task 6:

```bash
git switch -c task6-yourname
```

Replace `yourname` with your name or another unique identifier.

## STEP 3: WORK ONLY IN YOUR ASSIGNED FOLDER

For example:

- Task 6: `step6_prediction/`
- Task 7: `step7_dashboard/`
- Task 8: `step8_simulator/`

## STEP 4: CHECK YOUR CHANGES

```bash
git status
```

Check which files Git reports as modified or untracked.

## STEP 5: ADD ONLY YOUR TASK FOLDER

Task 6:

```bash
git add step6_prediction/
```

Task 7:

```bash
git add step7_dashboard/
```

Task 8:

```bash
git add step8_simulator/
```

## STEP 6: COMMIT YOUR WORK

```bash
git commit -m "Add Task 6 prediction model and results"
```

Change the message to describe your own task.

## STEP 7: PUSH YOUR BRANCH

```bash
git push -u origin task6-yourname
```

Use your actual branch name.

## STEP 8: CREATE A PULL REQUEST

1. Open the project repository on GitHub.
2. Create a Pull Request.
3. Select your branch as the source branch.
4. Select `main` as the destination branch.
5. Describe the files and work you added.
6. Ask the project lead to review and merge it.

IMPORTANT GITHUB RULES:

- Do not blindly use `git add .`.
- Always check `git status` before committing.
- Do not force-push to the shared main branch.
- Do not delete or overwrite another member's files.
- Do not upload protected datasets or credentials.
- If you encounter a merge conflict, stop and ask for help.

---

# 12. COMMON ERRORS AND SOLUTIONS

| Problem | Solution |
|---|---|
| Python is not recognised | Try `py --version`; install Python and enable PATH if needed. |
| ModuleNotFoundError | Install the missing package using pip. |
| FileNotFoundError | Check the terminal's current folder and the file path. |
| KeyError | Check the actual CSV column names and spelling. |
| Script creates no output | Read the script messages and check the results folder. |
| Dataset missing after cloning | Ask the project lead for the CSVs and place them in `data/`. |
| Git merge conflict | Ask the project lead for help. |
| Unsure about your task | Read your task brief and clarify before coding. |

---

# 13. HOW TO USE THIS README AS A CHATGPT PROMPT

Every team member can upload this README to ChatGPT and use the prompt below.

## COPY-PASTE PROMPT

You are helping me work on the NH-48 Blackspot Analysis Project.

I have uploaded the project's README file. Read the entire README carefully before giving me instructions.

MY ASSIGNED TASK IS:

[WRITE YOUR TASK NUMBER AND TASK NAME HERE]

I am a beginner and may not have much programming experience. Explain everything in simple language and guide me step by step.

PROJECT RULES:

1. Follow the folder structure specified in the README.
2. Use the existing project datasets and previous task outputs.
3. Never modify, rename, overwrite, or delete the original CSV files in the data folder.
4. Save my scripts inside my assigned task folder.
5. Save important generated outputs inside my task's results folder.
6. Upload important source code and generated results to GitHub.
7. Use existing segment IDs.
8. Follow the established 2023–2025 study period unless the project lead approves a change.
9. Do not silently change earlier task assumptions, severity weights, thresholds, or results.
10. Do not invent missing data, column names, results, or model performance.
11. Do not overwrite another team member's work.

BEFORE STARTING:

1. Explain what my assigned task means.
2. Explain what I am expected to achieve.
3. Tell me which existing files I need.
4. Tell me which datasets and earlier task outputs I should use.
5. Explain which columns are required.
6. Tell me which folders and files I need to create.
7. Identify missing information and ask me before making assumptions.

WHILE WORKING:

1. Guide me one step at a time.
2. Give me exact filenames and folder paths.
3. Give me copy-paste-ready Python code.
4. Explain important parts of the code in simple language.
5. Give me exact VS Code terminal commands.
6. Explain what successful output should look like.
7. Help me troubleshoot errors one at a time.
8. Do not skip steps assuming I already know how to code.

WHEN FINISHED:

1. Tell me which files were generated.
2. Tell me where each file should be saved.
3. Explain the results in simple language.
4. Explain assumptions and limitations.
5. Give me a checklist of files to hand over.
6. Explain how to upload my code and important results to GitHub safely.

If my task brief conflicts with the README or earlier task outputs, point it out and ask the project lead which instruction to follow.

Your goal is to help me complete my assigned task correctly while maintaining consistency with the rest of the NH-48 project.

---
# FINAL NOTE

GitHub is the shared project workspace.

It should contain:

- Source code.
- Important generated CSV files.
- Graphs and visualisations.
- Model outputs where appropriate.
- README files and instructions.

The original datasets are shared separately and placed in the local `data/` folder.

Later task members should be able to use the committed results from earlier tasks directly. They should not need to rerun every previous task just to obtain those results.

The goal is to make sure every team member can understand what to do, where to save their work, how to run it, and how to hand it over without depending on someone else for every technical step.
