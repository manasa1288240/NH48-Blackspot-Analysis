# NH-48 Black-Spot Dashboard (Kadodara to Kosamba)

Read-only Streamlit dashboard. It only displays the analysis outputs; it never recomputes BSRI.

## Folder layout
```
app.py
requirements.txt
data/
  bsri_all_segments.csv               (required)
  bsri_top_segment_factors.csv        (required)
  bsri_sensitivity_summary.csv        (required)
  nh48_segments.geojson               (optional: enables the map)
  scope_excluded_blackspots.csv       (optional: needs start_m, end_m; shades MoRTH stretches)
  fixed_500m_qualifying_stretches.csv (optional)
```
Rename the uploaded files to these exact names (remove the numeric prefix).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. The map's base layer needs internet access.

## Deploy (free)
Push the folder (including `data/`) to GitHub, then deploy on Streamlit Community Cloud with main file `app.py`.
