"""
whatif.py - What-if simulator tab for the NH-48 Black-Spot Dashboard
--------------------------------------------------------------------
READ-ONLY, like the rest of the dashboard. It only applies published
Crash Modification Factors (CMFs) from the Mech team's CSV files to the crash
counts that are already in the analysis outputs. It never recomputes BSRI,
ranks, EB values or intervals, and it never invents an effect size.

Files read (put both in the "data" folder next to app.py):
    intervention_effects.csv
    factor_intervention_matrix.csv

Use in app.py:
    from whatif import render_whatif, render_limits
    ...
    with tabs[3]:                      # the What-if tab
        render_whatif(seg, factors, DATA_DIR)
    ...
    # inside the About tab, at the end:
        render_limits()
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EFFECTS_FILE = "intervention_effects.csv"
MATRIX_FILE = "factor_intervention_matrix.csv"
MAX_FIXES = 3  # Mech's combination rule: no more than 3 fixes per segment

EFFECT_COLS = ["intervention_id", "intervention", "addresses_factor", "cmf_low",
               "cmf_central", "cmf_high", "applies_to", "source", "source_context",
               "confidence", "notes_and_limits"]
MATRIX_COLS = ["factor", "intervention_id", "priority", "short_reason"]
FACTOR_COLS = ["segment_id", "factor", "crashes_in_segment", "crashes_with_factor", "low_n"]

# Speed power model, from Mech's speed_severity_note.pdf:
#   Crashes_after = Crashes_before x (V_after / V_before) ^ e
# Exponents: Elvik (2009) as tabulated in FHWA-HRT-17-098, Table 1.
# Only the fatal exponents are well supported (the serious-injury interval is very wide),
# so only those two outcomes are shown.
# (label, column in bsri_all_segments.csv, best exponent, 95% low, 95% high)
POWER_MODEL = [
    ("Fatal crashes", "Fatal", 4.1, 2.9, 5.3),
    ("Fatalities", "total_fatalities", 4.6, 4.0, 5.2),
]


# ---------------------------------------------------------------- loading
@st.cache_data
def _read(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _load_checked(data_dir, name, required):
    """Read a CSV and check its columns. Returns None (and shows an error) if not usable."""
    path = Path(data_dir) / name
    if not path.exists():
        st.error(f"File not found: data/{name}")
        return None
    df = _read(str(path)).copy()
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{name} is missing columns: {missing}. Not guessing; please fix the CSV.")
        return None
    return df


def _to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])



def _df(data):
    """st.dataframe that works on old and new Streamlit (use_container_width was deprecated)."""
    try:
        st.dataframe(data, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=True)


def _plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        _plot(fig)


# ---------------------------------------------------------------- main tab
def render_whatif(seg, factors, data_dir):
    """Draw the What-if simulator tab. seg and factors are the dashboard's own dataframes."""
    st.subheader("What-if simulator")
    st.write(
        "Pick a segment and up to three safety fixes. The tool shows the **estimated** number of "
        "crashes that might have been avoided in the 2023-2025 data if the fix had been in place, "
        "as a range, using published Crash Modification Factors (CMFs). This is decision support, "
        "not a prediction. It does not change BSRI or any rank.")

    effects = _load_checked(data_dir, EFFECTS_FILE, EFFECT_COLS)
    matrix = _load_checked(data_dir, MATRIX_FILE, MATRIX_COLS)
    if effects is None or matrix is None:
        return
    missing_f = [c for c in FACTOR_COLS if c not in factors.columns]
    if missing_f:
        st.error(f"The contributing-factor table is missing columns: {missing_f}.")
        return

    for c in ["cmf_low", "cmf_central", "cmf_high"]:
        effects[c] = pd.to_numeric(effects[c], errors="coerce")
    effects["intervention_id"] = effects["intervention_id"].astype(str).str.strip()
    matrix["intervention_id"] = matrix["intervention_id"].astype(str).str.strip()
    factors = factors.copy()
    factors["segment_id"] = factors["segment_id"].astype(str)
    factors["low_n"] = _to_bool(factors["low_n"])
    factors["crashes_with_factor"] = pd.to_numeric(factors["crashes_with_factor"], errors="coerce")
    factors["crashes_in_segment"] = pd.to_numeric(factors["crashes_in_segment"], errors="coerce")

    # Factor data exists only for the top-ranked segments
    with_data = set(factors["segment_id"])
    order = seg.sort_values("BSRI_rank")["segment_id"].astype(str).tolist()
    ids = [s for s in order if s in with_data]
    if not ids:
        st.info("No segment has contributing-factor data, so fixes cannot be matched.")
        return

    sid = st.selectbox("Pick a segment (ordered by rank; only segments with factor data)",
                       ids, key="whatif_segment")
    srow = seg[seg["segment_id"].astype(str) == sid].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rank", int(srow["BSRI_rank"]))
    c2.metric("BSRI (EB-smoothed)", f"{srow['BSRI']:.1f}")
    c3.metric("Crashes in segment (2023-2025)", int(srow["total_accidents"]))

    _fixes_section(sid, factors, effects, matrix)
    st.divider()
    _speed_section(srow)


# ---------------------------------------------------------------- fixes
def _fixes_section(sid, factors, effects, matrix):
    sf = factors[(factors["segment_id"] == sid) & (factors["crashes_with_factor"] > 0)]
    sf = sf.drop_duplicates("factor")
    if sf.empty:
        st.info("This segment has no contributing-factor rows.")
        return
    seg_crashes = float(sf["crashes_in_segment"].iloc[0])

    cand = (matrix.merge(effects, on="intervention_id", how="left")
                  .merge(sf[["factor", "crashes_with_factor", "low_n"]], on="factor", how="inner"))
    cand["has_cmf"] = cand["cmf_central"].notna()
    # A CMF is only applied to the factor it was measured for
    cand["direct"] = cand["has_cmf"] & (cand["factor"] == cand["addresses_factor"])

    def status(r):
        if not r["has_cmf"]:
            return "No evidence found"
        if r["direct"]:
            return "Can be simulated"
        return "Not simulated (CMF not measured for this factor)"

    st.markdown("### Fixes that match this segment's factors")
    unmatched = sorted(set(sf["factor"]) - set(cand["factor"]))
    if cand.empty:
        st.info("None of this segment's factors has a fix in the Mech matrix.")
    else:
        cand["Status"] = cand.apply(status, axis=1)
        show = cand.sort_values(["factor", "priority"])[
            ["factor", "crashes_with_factor", "intervention_id", "intervention", "Status", "short_reason"]
        ].rename(columns={"factor": "Factor", "crashes_with_factor": "Crashes with factor",
                          "intervention_id": "ID", "intervention": "Fix", "short_reason": "Why"})
        _df(show)
    if unmatched:
        st.caption("Factors in this segment with no fix in the matrix: " + ", ".join(unmatched))

    sim = cand[cand["direct"]].copy() if not cand.empty else cand
    if sim.empty:
        st.info("No fix with a published CMF matches this segment's factors.")
        return

    sim["label"] = sim.apply(
        lambda r: f"{r['intervention_id']} | {r['intervention']} "
                  f"(targets {r['factor']}: {int(r['crashes_with_factor'])} crashes)", axis=1)
    picked = st.multiselect(f"Choose up to {MAX_FIXES} fixes to simulate", sim["label"].tolist(),
                            max_selections=MAX_FIXES, key=f"whatif_pick_{sid}")
    if not picked:
        st.caption("Select a fix to see the estimated effect.")
        return

    sel = sim[sim["label"].isin(picked)].head(MAX_FIXES).copy()
    # Mech rule: two fixes on the same factor overlap, so only the more effective one counts
    sel = sel.sort_values("cmf_central")
    counted = sel.drop_duplicates("factor", keep="first").copy()
    skipped = sel.drop(counted.index)
    if not skipped.empty:
        st.warning("Not counted because they overlap with a more effective fix for the same factor: "
                   + ", ".join(skipped["intervention_id"]))

    # Missing low/high -> fall back to central. A lower CMF means a bigger reduction.
    counted["cmf_low"] = counted["cmf_low"].fillna(counted["cmf_central"])
    counted["cmf_high"] = counted["cmf_high"].fillna(counted["cmf_central"])
    n = counted["crashes_with_factor"]
    counted["best"] = n * (1 - counted["cmf_low"])
    counted["central"] = n * (1 - counted["cmf_central"])
    counted["worst"] = n * (1 - counted["cmf_high"])

    # Different factors = different crash groups, so effects are added.
    # Factor groups can overlap, so the total is an upper bound; capped at the segment's crashes.
    tot_best = min(counted["best"].sum(), seg_crashes)
    tot_central = min(counted["central"].sum(), seg_crashes)
    tot_worst = min(counted["worst"].sum(), seg_crashes)

    st.markdown("### Estimated effect")
    m1, m2 = st.columns(2)
    m1.metric("Crashes that might have been avoided (central)", f"{tot_central:.1f}",
              help="Out of the crashes recorded in this segment in 2023-2025.")
    m2.metric("Share of this segment's crashes", f"{tot_central / seg_crashes:.0%}" if seg_crashes else "n/a")
    st.caption(f"Range: {tot_worst:.1f} (cautious) to {tot_best:.1f} (optimistic) crashes. "
               "Treat the total as an upper bound on benefit.")

    if counted["low_n"].any():
        st.warning("Some targeted factors have too few crashes (low n). The estimate is very uncertain.")

    fig = go.Figure(go.Bar(
        y=counted["intervention_id"] + " " + counted["intervention"],
        x=counted["central"], orientation="h",
        error_x=dict(type="data", symmetric=False,
                     array=(counted["best"] - counted["central"]).clip(lower=0),
                     arrayminus=(counted["central"] - counted["worst"]).clip(lower=0)),
        hovertemplate="%{y}<br>central %{x:.1f}<extra></extra>"))
    fig.update_layout(xaxis_title="Crashes that might have been avoided (bars show the range)",
                      yaxis=dict(autorange="reversed"), height=120 + 70 * len(counted),
                      margin=dict(l=10, r=10, t=10, b=10))
    _plot(fig)

    tbl = pd.DataFrame({
        "Fix": counted["intervention_id"] + " " + counted["intervention"],
        "Targets": counted["factor"],
        "Crashes with factor": counted["crashes_with_factor"].astype(int),
        "CMF (optimistic / central / cautious)": counted.apply(
            lambda r: f"{r['cmf_low']:.2f} / {r['cmf_central']:.2f} / {r['cmf_high']:.2f}", axis=1),
        "Avoided (cautious to optimistic)": counted.apply(
            lambda r: f"{r['worst']:.1f} to {r['best']:.1f} (central {r['central']:.1f})", axis=1),
        "Applies to": counted["applies_to"],
        "Confidence": counted["confidence"],
    })
    _df(tbl)
    st.caption(
        "A CMF of 0.80 means an estimated 20% fewer crashes. Each CMF is applied only to the crashes "
        "carrying the factor it targets. The factor count may include crashes outside the fix's exact "
        "scope (for example daytime or non-junction crashes), so figures lean optimistic. "
        "Counts are observed 2023-2025 crashes, not traffic-adjusted.")

    with st.expander("Sources and limits for the selected fixes"):
        for _, r in counted.iterrows():
            st.markdown(f"**{r['intervention_id']} {r['intervention']}**")
            st.markdown(f"- Source: {r['source']}")
            st.markdown(f"- Where the evidence is from: {r['source_context']}")
            st.markdown(f"- Confidence: {r['confidence']}")
            st.markdown(f"- Limits: {r['notes_and_limits']}")

    no_evidence = cand[~cand["has_cmf"]]
    if not no_evidence.empty:
        st.info("Evidence gap, not zero effect: no published CMF was found for "
                + ", ".join(sorted(set(no_evidence["intervention"]))) + ".")


# ---------------------------------------------------------------- speed
def _speed_section(srow):
    st.markdown("### Speed what-if (power model)")
    st.write("If mean traffic speed fell by a chosen percentage, the published power model gives "
             "the expected change in fatal outcomes.")
    cut = st.slider("Mean speed reduction (%)", 0, 20, 10, key="whatif_speed")
    ratio = lambda e: (1 - cut / 100) ** e

    rows = []
    for label, col, e, e_lo, e_hi in POWER_MODEL:
        if col not in srow.index or pd.isna(srow[col]):
            continue
        now = float(srow[col])
        # a higher exponent means a bigger drop
        rows.append({
            "Outcome": label,
            "Recorded (2023-2025)": f"{now:.0f}",
            "After (central)": f"{now * ratio(e):.2f}",
            "After (range)": f"{now * ratio(e_hi):.2f} to {now * ratio(e_lo):.2f}",
            "Change (central)": f"{ratio(e) - 1:.0%}",
            "Exponent (95% interval)": f"{e} ({e_lo} to {e_hi})",
        })
    if rows:
        _df(pd.DataFrame(rows))
    else:
        st.info("This segment has no fatal-crash columns to apply the model to.")
    st.caption(
        "There is no speed data, so this cannot say how much any measure changes mean speed on NH-48. "
        "It is a 'what if mean speed fell by X%' table only. Exponents come from Nordic and other "
        "high-income studies; Indian mixed traffic may differ. It describes averages across many sites, "
        "not one 500 m segment. Do not add it on top of the speed-camera estimate (I04): both describe "
        "the same speed effect.")


# ---------------------------------------------------------------- About tab text
def render_limits():
    """Put this at the end of the About tab. Text comes from the Mech team's three notes."""
    st.markdown("#### What-if simulator: limits")
    with st.expander("What the simulator cannot claim"):
        st.markdown("""
- **Estimated effects, not promises.** Every figure is an expected reduction from past studies. A real segment can do better or worse.
- **Almost all evidence is from outside India** (USA, Europe, Australia). Confidence is Low or Medium for every row.
- **Each effect applies only to the crashes it targets.** A CMF is never applied to total crashes.
- **Ranges matter more than the central value.** Some ranges are wide, and some values are derived from published ranges or standard errors (marked in the source notes).
- **Combined effects are an upper bound on benefit.** At most three fixes per segment.
- **No traffic volume, speed or road geometry data.** The simulator cannot give absolute crashes prevented, the speed change from a camera, or whether a segment has a median.
- **Some fixes only work in some places** (cable barrier needs a divided road with a median, junction lighting needs a junction, friction treatment needs a friction-related problem).
- **No cost or benefit-cost figures.** No cost source was verified.
- **Evidence gaps are not zero effect.** Fog has no source. Daytime side-impact, rear-end, distraction and vehicle-defect factors have no fix in the table.
- **Rankings are not changed.** The simulator does not recompute BSRI and does not predict future crashes.
""")
    with st.expander("How several fixes are combined"):
        st.markdown("""
- Never add percentages. CMFs that act on the same crashes are multiplied (0.80 x 0.90 = 0.72, an estimated 28% reduction, not 30%).
- Two fixes that address the same factor overlap, so only the more effective one is used.
- No more than three fixes per segment (FHWA, Texas DOT and Wisconsin DOT guidance).
- Fixes with no CMF are excluded and shown as "no evidence".
- No published numeric ceiling on total reduction was found. Any percentage cap would be a team assumption.
- Multiplying CMFs can overstate the combined effect, so treat combined results as an upper bound.
- In this dashboard, fixes on different factors act on different crash groups, so their avoided crashes are added and capped at the segment's crash count.
""")
    with st.expander("Speed and severity (power model)"):
        st.markdown("""
**Crashes after = Crashes before x (V after / V before) ^ e**, where V is mean speed and e depends on crash severity.

- Fatal crashes: e = 4.1 (95% interval 2.9 to 5.3). Fatalities: e = 4.6 (4.0 to 5.2). Source: Elvik (2009), as tabulated in FHWA-HRT-17-098, Table 1.
- Needs the change in mean speed. There is no speed data, so it is a "what if" table only.
- Exponents come from high-income countries; Indian mixed traffic may differ.
- The effect also depends on initial speed (Elvik 2013), so the simple model is an approximation.
- Describes averages across many sites, not one segment.
- Not to be added on top of the speed-camera estimate (I04).
""")
    with st.expander("Mech team sources"):
        st.markdown("""
- FHWA, Crash Modification Factors page: multiply no more than 3 CMFs; overestimation warning.
- AASHTO, Highway Safety Manual, 1st ed., 2010, Chapter 3 (section number not confirmed).
- FHWA CMF Clearinghouse FAQ on combining CMFs.
- Vanasse Hangen Brustlin, Investigation of Existing and Alternative Methods for Combining Multiple CMFs, 2011.
- Texas DOT, Roadway Design Manual, Section 5.4.
- Wisconsin DOT, Traffic Engineering, Operations and Safety Manual, 12-3-1, 2018.
- Park J, Abdel-Aty M, Lee C. Accident Analysis and Prevention 70, 2014, 167-177.
- Nilsson G. Traffic safety dimensions and the power model. Lund Bulletin 221, 2004.
- Elvik R. The Power Model of the relationship between speed and road safety. TOI Report 1034/2009.
- Elvik R. A re-parameterisation of the Power Model. Accident Analysis and Prevention, 2013.
- FHWA-HRT-17-098, Table 1.
- SWOV Fact Sheet, The relation between speed and crashes.
- Per-fix CMF sources are listed in the simulator under "Sources and limits for the selected fixes".
""")
