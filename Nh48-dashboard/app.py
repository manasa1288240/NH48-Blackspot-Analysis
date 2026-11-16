"""
NH-48 Black-Spot Dashboard (Kadodara to Kosamba, Surat district)
================================================================
READ-ONLY dashboard. It displays numbers produced by the analysis
(teammate's CSVs). It never recomputes BSRI, ranks, EB values or intervals.

Folder layout (see README):
    app.py
    data/bsri_all_segments.csv               (required)
    data/bsri_top_segment_factors.csv        (required)
    data/bsri_sensitivity_summary.csv        (required)
    data/nh48_segments.geojson               (optional: enables the map)
    data/scope_excluded_blackspots.csv       (optional: MoRTH sliding-window stretches)
    data/fixed_500m_qualifying_stretches.csv (optional)

Rule used everywhere: if something is not in a file, it is not shown.
"""
import json
from pathlib import Path
from whatif import render_whatif, render_limits

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

st.set_page_config(page_title="NH-48 Black-Spot Dashboard", layout="wide")
DATA_DIR = Path(__file__).parent / "data"

# ----------------------------------------------------------------------------
# Expected columns. If a required column is missing the app stops and says so.
# ----------------------------------------------------------------------------
SEG_COLS = [
    "segment_id", "chain_start_m", "chain_end_m", "seg_length_m", "short_segment",
    "Fatal", "Grievous", "Minor hospitalized", "Minor non-hospitalized", "No injury",
    "total_accidents", "total_fatalities", "weighted_severity_points", "eb_crashes",
    "BSRI", "BSRI_raw_no_smoothing", "BSRI_rank", "top10_frequency",
    "rank_p05", "rank_p95", "possible_duplicate_records", "is_blackspot",
    "qualification", "near_miss",
]
FACTOR_COLS = ["segment_id", "factor", "crashes_in_segment", "crashes_with_factor",
               "segment_share", "corridor_share", "lift", "low_n"]
SENS_COLS = ["scenario", "top10_overlap_with_primary", "spearman_with_primary"]
SEVERITY = ["Fatal", "Grievous", "Minor hospitalized", "Minor non-hospitalized", "No injury"]


# ----------------------------------------------------------------------------
# Loading (cached, read-only)
# ----------------------------------------------------------------------------
@st.cache_data
def read_csv(name: str):
    path = DATA_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def read_geojson(name: str):
    """Return ({segment_id: [(lon, lat), ...]}, sum of crash_count property) or (None, None)."""
    path = DATA_DIR / name
    if not path.exists():
        return None, None
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lines, total = {}, 0
    for feat in gj["features"]:
        props = feat["properties"]
        lines[str(props["segment_id"])] = [tuple(c[:2]) for c in feat["geometry"]["coordinates"]]
        total += props.get("crash_count", 0) or 0
    return lines, total


def require(name, cols):
    """Required CSV: stop with a clear message if the file or a column is missing."""
    df = read_csv(name)
    if df is None:
        st.error(f"File not found: data/{name}")
        st.stop()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"{name} is missing columns: {missing}. Not guessing; please fix the CSV.")
        st.stop()
    return df.copy()


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


seg = require("bsri_all_segments.csv", SEG_COLS)
factors = require("bsri_top_segment_factors.csv", FACTOR_COLS)
sens = require("bsri_sensitivity_summary.csv", SENS_COLS)
scope = read_csv("scope_excluded_blackspots.csv")           # optional
qual = read_csv("fixed_500m_qualifying_stretches.csv")      # optional
geo, geo_crash_total = read_geojson("nh48_segments.geojson")  # optional

if scope is not None and not {"start_m", "end_m"} <= set(scope.columns):
    st.warning("scope_excluded_blackspots.csv has no start_m/end_m columns, so it is ignored.")
    scope = None

# Display-only preparation. No analysis value is changed.
seg["segment_id"] = seg["segment_id"].astype(str)           # ids are text, e.g. S012
factors["segment_id"] = factors["segment_id"].astype(str)
for c in ["short_segment", "is_blackspot", "near_miss"]:
    seg[c] = to_bool(seg[c])
factors["low_n"] = to_bool(factors["low_n"])
seg["start_km"] = seg["chain_start_m"] / 1000
seg["end_km"] = seg["chain_end_m"] / 1000

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
version = st.sidebar.radio("BSRI version", ["EB-smoothed BSRI (primary)", "Raw (no smoothing)"])
SEL = "BSRI" if version.startswith("EB") else "BSRI_raw_no_smoothing"
min_bsri = st.sidebar.slider("Minimum BSRI", 0, 100, 0)
bs_only = st.sidebar.checkbox("Blackspot segments only")
hide_short = st.sidebar.checkbox("Hide short segments")

mask = seg[SEL] >= min_bsri
if bs_only:
    mask &= seg["is_blackspot"]
if hide_short:
    mask &= ~seg["short_segment"]
view = seg[mask]

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("NH-48 Black-Spot Dashboard: Kadodara to Kosamba")
st.caption("Crash data 2023-2025, fixed 500 m segments. Read-only view of the project's analysis outputs.")

kpis = [
    ("Segments", f"{len(seg)}", None),
    ("Crashes used", f"{int(seg['total_accidents'].sum())}", "Sum of total_accidents over all segments."),
    ("Segments meeting 500 m screening", f"{int(seg['is_blackspot'].sum())}",
     "Count of is_blackspot = True in the segment table."),
]
if scope is not None:
    kpis.append(("MoRTH-criteria stretches", f"{len(scope)}", "Rows in scope_excluded_blackspots.csv."))
for col, (label, value, tip) in zip(st.columns(len(kpis)), kpis):
    col.metric(label, value, help=tip)
st.caption(f"Showing {len(view)} of {len(seg)} segments after filters.")

tabs = st.tabs(["Overview", "Segment detail", "Sensitivity",
                "What-if simulator", "Future blackspots", "About and limits"])

# ============================================================================
# OVERVIEW
# ============================================================================
with tabs[0]:
    if SEL != "BSRI":
        st.info("Raw BSRI is shown. Rank, rank interval and top-10 frequency belong to the "
                "primary (EB-smoothed) BSRI.")

    # ---- BSRI along the corridor (chainage chart) ----
    st.subheader("BSRI along the corridor")

    def group(r):
        if r["short_segment"]:
            return "Short segment (under 500 m)"
        return "Meets 500 m screening" if r["is_blackspot"] else "Other segment"

    plot_df = view.assign(group=view.apply(group, axis=1))
    colors = {"Other segment": "#4c78a8", "Meets 500 m screening": "#d62728",
              "Short segment (under 500 m)": "#f28e2b"}
    fig = go.Figure()
    for name, sub in plot_df.groupby("group"):
        fig.add_trace(go.Bar(
            x=sub["start_km"], y=sub[SEL], width=sub["seg_length_m"] / 1000, offset=0,
            name=name, marker_color=colors[name],
            customdata=sub[["segment_id", "BSRI_rank"]],
            hovertemplate="Segment %{customdata[0]}<br>Rank %{customdata[1]}<br>"
                          "BSRI %{y:.1f}<extra></extra>"))
    if scope is not None:
        for _, r in scope.iterrows():
            fig.add_vrect(x0=r["start_m"] / 1000, x1=r["end_m"] / 1000,
                          fillcolor="red", opacity=0.15, line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="MoRTH-criteria stretch",
                                 marker=dict(symbol="square", size=12, color="rgba(255,0,0,0.3)")))
    fig.update_layout(xaxis_title="Chainage (km)", yaxis_title=version, yaxis_range=[0, 105],
                      bargap=0, height=420, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig)
    if scope is None:
        st.caption("MoRTH sliding-window stretches are not shaded because "
                   "scope_excluded_blackspots.csv was not provided.")

    # ---- Map (only if a geometry file was provided) ----
    st.subheader("Map")
    if geo is None:
        st.info("No geometry file (data/nh48_segments.geojson) found, so no map is drawn.")
    else:
        mp = view[view["segment_id"].isin(geo)]
        fig_m = go.Figure()
        # Context: all segments as thin grey lines (including ones filtered out)
        for sid, pts in geo.items():
            lon, lat = zip(*pts)
            fig_m.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", hoverinfo="skip",
                                             line=dict(width=3, color="#cccccc"), showlegend=False))
        # Filtered segments coloured by the selected BSRI
        scale = sample_colorscale("YlOrRd", [min(max(v, 0), 100) / 100 for v in mp[SEL]])
        mids_lon, mids_lat = [], []
        for (_, r), col in zip(mp.iterrows(), scale):
            pts = geo[r["segment_id"]]
            lon, lat = zip(*pts)
            fig_m.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", hoverinfo="skip",
                                             line=dict(width=6, color=col), showlegend=False))
            mid = pts[len(pts) // 2]          # a point on the segment's own line
            mids_lon.append(mid[0])
            mids_lat.append(mid[1])
        fig_m.add_trace(go.Scattermap(
            lon=mids_lon, lat=mids_lat, mode="markers", showlegend=False,
            marker=dict(size=9, color=mp[SEL], colorscale="YlOrRd", cmin=0, cmax=100,
                        colorbar=dict(title=version.split(" ")[0])),
            customdata=mp[["segment_id", "BSRI_rank", "total_accidents"]],
            hovertemplate="Segment %{customdata[0]}<br>Rank %{customdata[1]}<br>"
                          "Crashes %{customdata[2]}<br>BSRI %{marker.color:.1f}<extra></extra>"))
        all_pts = [p for pts in geo.values() for p in pts]
        fig_m.update_layout(
            map=dict(style="open-street-map",
                        center=dict(lon=sum(p[0] for p in all_pts) / len(all_pts),
                                    lat=sum(p[1] for p in all_pts) / len(all_pts)),
                        zoom=9),
            margin=dict(l=0, r=0, t=0, b=0), height=550)
        st.plotly_chart(fig_m)
        st.caption("Grey = segments hidden by the sidebar filters. The base map needs internet access.")

    # ---- Ranked table ----
    st.subheader("Ranked segments")
    t = view.sort_values("BSRI_rank").copy()
    t["Chainage (km)"] = t.apply(lambda r: f"{r['start_km']:.1f} to {r['end_km']:.1f}", axis=1)
    t["Rank interval (90%)"] = t.apply(
        lambda r: f"{r['rank_p05']:.0f} to {r['rank_p95']:.0f}"
        if pd.notna(r["rank_p05"]) and pd.notna(r["rank_p95"]) else "n/a", axis=1)
    tbl = t[["BSRI_rank", "segment_id", "Chainage (km)", "total_accidents", "total_fatalities",
             SEL, "Rank interval (90%)", "is_blackspot", "qualification", "short_segment"]]
    tbl = tbl.rename(columns={
        "BSRI_rank": "Rank", "segment_id": "Segment", "total_accidents": "Crashes",
        "total_fatalities": "Fatalities", SEL: "BSRI (selected)", "is_blackspot": "Blackspot",
        "qualification": "Qualification", "short_segment": "Short segment"})

    def highlight_short(row):
        return ["background-color:#fff3cd;color:#000" if row["Short segment"] else ""] * len(row)

    st.dataframe(tbl.style.apply(highlight_short, axis=1).format({"BSRI (selected)": "{:.1f}"}),
                 hide_index=True)
    st.caption("Click a column header to sort. Highlighted rows are short segments (under 500 m); "
               "their counts are not directly comparable with full 500 m segments.")

    if scope is not None:
        with st.expander("MoRTH-criteria screening stretches (sliding window)"):
            st.dataframe(scope, hide_index=True)
    if qual is not None:
        with st.expander("Fixed 500 m qualifying stretches"):
            st.dataframe(qual, hide_index=True)

# ============================================================================
# SEGMENT DETAIL
# ============================================================================
with tabs[1]:
    ids = seg.sort_values("BSRI_rank")["segment_id"].tolist()
    sid = st.selectbox("Pick a segment (ordered by rank)", ids)
    r = seg[seg["segment_id"] == sid].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BSRI (EB-smoothed)", f"{r['BSRI']:.1f}")
    c2.metric("Rank", int(r["BSRI_rank"]))
    c3.metric("Rank interval (90%)", f"{r['rank_p05']:.0f} to {r['rank_p95']:.0f}")
    c4.metric("Share of resamples in top 10", f"{r['top10_frequency']:.0%}")
    st.caption(f"Chainage {r['start_km']:.1f} to {r['end_km']:.1f} km | "
               f"qualification: {r['qualification']} | short segment: {r['short_segment']} | "
               f"near miss: {r['near_miss']} | possible duplicate records: "
               f"{int(r['possible_duplicate_records'])}")

    a, b = st.columns(2)
    with a:
        st.markdown("**Severity mix (crash counts)**")
        sev = pd.DataFrame({"Severity": SEVERITY, "Crashes": [r[s] for s in SEVERITY]})
        st.plotly_chart(px.bar(sev, x="Severity", y="Crashes"))
    with b:
        st.markdown("**Observed vs EB-smoothed crash count**")
        eb = pd.DataFrame({"Measure": ["Observed", "EB-smoothed"],
                           "Crashes": [r["total_accidents"], r["eb_crashes"]]})
        st.plotly_chart(px.bar(eb, x="Measure", y="Crashes"))
        st.caption("EB shrinks a segment's count toward the corridor average.")

    st.markdown("**Contributing factors**")
    ff = factors[factors["segment_id"] == sid].copy()
    if ff.empty:
        st.info("Factor data exists only for the top 10 segments; this segment is not in that file.")
    else:
        ff["Factor"] = ff.apply(
            lambda x: x["factor"].replace("_", " ") + (" (low n)" if x["low_n"] else ""), axis=1)
        ff = ff.sort_values("segment_share", ascending=False)
        long = ff.melt(id_vars="Factor", value_vars=["segment_share", "corridor_share"],
                       var_name="Share of", value_name="Share")
        long["Share of"] = long["Share of"].map({"segment_share": "This segment",
                                                 "corridor_share": "Whole corridor"})
        st.plotly_chart(px.bar(long, x="Factor", y="Share", color="Share of", barmode="group"))
        st.dataframe(ff[["factor", "crashes_in_segment", "crashes_with_factor",
                         "segment_share", "corridor_share", "lift", "low_n"]], hide_index=True)
        st.caption("Descriptive, not causal. A crash can have several factors, so shares do not add "
                   "up to 100%. 'low n' marks factors with too few crashes to rely on.")

# ============================================================================
# SENSITIVITY
# ============================================================================
with tabs[2]:
    st.subheader("How stable is the ranking?")
    st.write("Each scenario re-ranks the segments with one assumption changed. Top-10 overlap is how "
             "many of the primary top 10 are also in the scenario's top 10. Spearman is how closely "
             "the full ranking agrees with the primary one (1.0 = identical order).")
    s = sens.copy()
    s["scenario"] = s["scenario"].str.replace("_", " ")
    st.dataframe(s, hide_index=True)
    s1, s2 = st.columns(2)
    s1.plotly_chart(px.bar(s, x="scenario", y="top10_overlap_with_primary",
                           title="Top-10 overlap with primary (out of 10)", range_y=[0, 10]))
    s2.plotly_chart(px.bar(s, x="scenario", y="spearman_with_primary",
                           title="Spearman correlation with primary", range_y=[0, 1]))

# ============================================================================
# PLACEHOLDERS (no fake content)
# ============================================================================
with tabs[3]:
    render_whatif(seg, factors, DATA_DIR)
with tabs[4]:
    st.info("Future-blackspot prediction model: planned for a later phase. No content yet.")

# ============================================================================
# ABOUT AND LIMITS
# ============================================================================
with tabs[5]:
    st.subheader("About and limits")
    fatal_share = seg["Fatal"].sum() / seg["total_accidents"].sum()
    st.markdown(f"""
- **BSRI is relative.** The top segment is 100 by construction. It cannot be compared with other roads or other periods.
- **Severe-crash bias.** The sample leans to severe crashes: {fatal_share:.0%} of the crashes counted here are fatal. BSRI therefore measures severe-crash risk, not all crashes.
- **Severity weights** are APW (Fatal 6, Serious 3, Minor 0.8, PDO 0.2) from published black-spot studies. How the dataset's five severity classes map onto these is the project's own choice.
  *[Teammate: paste the exact citations here.]*
- **Empirical Bayes smoothing** shrinks each segment toward the corridor average. There is no traffic volume, so this is not a traffic-adjusted model.
- **Rank intervals** come from resampling crashes only. They do not include map-matching or recording errors.
- **Contributing-factor shares** are descriptive, not causal, and a crash can have several factors. Factor data covers only the top 10 segments.
- **MoRTH-criteria screening** uses 5 fatal/grievous crashes or 10 deaths per 500 m over 3 years. It is a rule-based screen, not a risk index.
- **Data:** university crash data, calendar years 2023-2025, about 39 km of NH-48, fixed 500 m segments (last segment shorter).
""")
    render_limits()
    if geo_crash_total is not None and geo_crash_total != int(seg["total_accidents"].sum()):
        st.warning(
            f"Data note: the geometry file's crash_count adds up to {geo_crash_total}, while the "
            f"segment table counts {int(seg['total_accidents'].sum())} crashes. The dashboard uses "
            "the segment table only; the geometry file is used for line shapes.")
