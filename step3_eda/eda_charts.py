"""
Step 3: Exploratory charts for NH-48 crash data (Kadodara to Kosamba).
Reads the segmented crash file and saves PNGs to reports/figures/.
Run: python eda_charts.py "crash_segmented 1.csv"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Settings ----------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "crash_segmented 1.csv"
OUT_DIR = Path("reports/figures")
SEG_LEN_M = 500                    # segment length used in Step 2
MONTH_ORDER = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]  # May-to-April study year
EXCLUDE_FAR_SNAP = True            # drop crashes snapped >300 m from road in along-road charts
SEVERITY_HIGH_RANK_IS_WORSE = True # CHECK: set False if severity_rank 1 = most severe

sns.set_theme(style="whitegrid", context="notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    """Save a figure to reports/figures and close it."""
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  [saved] {OUT_DIR / name}")


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def flag_counts(df, prefix, exclude=()):
    """Count 0/1 multi-valued flag columns sharing a prefix (e.g. cn_, tv_, io_)."""
    cols = [c for c in df.columns
            if c.startswith(prefix) and c not in exclude and not c.endswith("_n_values")
            and not c.endswith("_primary")]
    s = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().astype(int)
    s.index = [c[len(prefix):].replace("_", " ") for c in s.index]
    return s.sort_values(ascending=False)


def stacked_bar(ax, ct, colors, normalize=False, title="", xlabel=""):
    """Stacked bar from a crosstab; normalize=True shows row shares in %."""
    data = ct.div(ct.sum(axis=1), axis=0) * 100 if normalize else ct
    data.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of crashes" if normalize else "Crashes")
    ax.legend(title="Severity", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# ---------------- Load ----------------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {CSV_PATH}")

# Fill categorical gaps so they show up as "Unknown" instead of vanishing
for c in ["severity", "collision_type", "light_condition", "visibility", "study_year"]:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()

# Severity order (worst first) taken from severity_rank
rank_map = df.groupby("severity")["severity_rank"].median()
sev_order = rank_map.sort_values(ascending=not SEVERITY_HIGH_RANK_IS_WORSE).index.tolist()
sev_colors = sns.color_palette("YlOrRd", len(sev_order))[::-1]  # worst = darkest
sev_colors = dict(zip(sev_order, sev_colors))
print("Severity order used (worst first):", sev_order)

study_years = sorted(df["study_year"].unique())

# =====================================================================
# 1. Crashes per study year and per month
# =====================================================================
header("1. Crashes per study year and per month")
per_year = df["study_year"].value_counts().reindex(study_years)
month_year = (df.groupby(["month", "study_year"]).size().unstack(fill_value=0)
                .reindex(MONTH_ORDER, fill_value=0))
per_month = month_year.sum(axis=1)
print("Crashes per study year:\n", per_year.to_string())
print("\nCrashes per month (all years pooled, May..Apr):\n", per_month.to_string())
print(f"Busiest month: {per_month.idxmax()} ({per_month.max()}); "
      f"quietest: {per_month.idxmin()} ({per_month.min()})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(per_year.index, per_year.values, color="steelblue")
axes[0].bar_label(bars)
axes[0].set_title("Crashes per study year (May-Apr)")
axes[0].set_ylabel("Crashes")
month_year.plot(ax=axes[1], marker="o")
axes[1].set_xticks(range(12))
axes[1].set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                         "Jan", "Feb", "Mar", "Apr"])
axes[1].set_title("Crashes per month, by study year")
axes[1].set_ylabel("Crashes")
axes[1].legend(title="Study year", fontsize=8)
save(fig, "01_crashes_by_year_and_month.png")

# =====================================================================
# 2. Crashes by hour of day
# =====================================================================
header("2. Crashes by hour of day")
hr = df["hour"].dropna().astype(int)
by_hour = hr.value_counts().reindex(range(24), fill_value=0)
print(f"Rows with hour: {len(hr)} of {len(df)}")
print("Top 5 hours:\n", by_hour.sort_values(ascending=False).head(5).to_string())
night = by_hour.reindex(list(range(0, 6)) + list(range(19, 24))).sum()
print(f"Crashes 19:00-05:59: {night} ({night / by_hour.sum():.0%})")

hour_sev = (df.dropna(subset=["hour"]).assign(hour=lambda d: d["hour"].astype(int))
              .groupby(["hour", "severity"]).size().unstack(fill_value=0)
              .reindex(index=range(24), columns=sev_order, fill_value=0))
fig, ax = plt.subplots(figsize=(12, 5))
hour_sev.plot(kind="bar", stacked=True, ax=ax, color=[sev_colors[s] for s in sev_order],
              width=0.85, edgecolor="white")
ax.set_title("Crashes by hour of day (stacked by severity)")
ax.set_xlabel("Hour (0-23)")
ax.set_ylabel("Crashes")
ax.legend(title="Severity", fontsize=8)
save(fig, "02_crashes_by_hour.png")

# =====================================================================
# 3. Severity split, overall and by study year
# =====================================================================
header("3. Severity split")
sev_counts = df["severity"].value_counts().reindex(sev_order)
print("Overall:\n", pd.DataFrame({"n": sev_counts,
                                  "pct": (sev_counts / len(df) * 100).round(1)}).to_string())
sev_year = pd.crosstab(df["study_year"], df["severity"]).reindex(columns=sev_order, fill_value=0)
print("\nBy study year (counts):\n", sev_year.to_string())
print("\nBy study year (% of year):\n",
      (sev_year.div(sev_year.sum(axis=1), axis=0) * 100).round(1).to_string())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
bars = axes[0].bar(sev_counts.index, sev_counts.values,
                   color=[sev_colors[s] for s in sev_counts.index])
axes[0].bar_label(bars)
axes[0].set_title("Severity, overall")
axes[0].set_ylabel("Crashes")
plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
cols = [sev_colors[s] for s in sev_order]
stacked_bar(axes[1], sev_year, cols, False, "Severity by study year (counts)", "Study year")
stacked_bar(axes[2], sev_year, cols, True, "Severity by study year (%)", "Study year")
save(fig, "03_severity_split.png")

# =====================================================================
# 4. Collision type and collision nature
# =====================================================================
header("4. Collision type and nature")
ctype = df["collision_type"].value_counts()
cnature = flag_counts(df, "cn_")
print("collision_type:\n", ctype.to_string())
print("\ncollision nature flags (multi-valued, can sum to more than rows):\n",
      cnature.to_string())
print("Rows with more than one nature:",
      int((pd.to_numeric(df["cn_n_values"], errors="coerce") > 1).sum()))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x=ctype.values, y=ctype.index, ax=axes[0], color="slateblue")
axes[0].bar_label(axes[0].containers[0])
axes[0].set_title("Collision type")
axes[0].set_xlabel("Crashes")
sns.barplot(x=cnature.values, y=cnature.index, ax=axes[1], color="teal")
axes[1].bar_label(axes[1].containers[0])
axes[1].set_title("Collision nature (multi-valued flags)")
axes[1].set_xlabel("Crashes mentioning it")
save(fig, "04_collision_type_and_nature.png")

# =====================================================================
# 5. Light condition and visibility vs severity
# =====================================================================
header("5. Light condition and visibility vs severity")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
cols = [sev_colors[s] for s in sev_order]
for i, var in enumerate(["light_condition", "visibility"]):
    ct = pd.crosstab(df[var], df["severity"]).reindex(columns=sev_order, fill_value=0)
    print(f"\n{var} x severity (counts):\n", ct.to_string())
    print(f"\n{var} x severity (% of row):\n",
          (ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())
    stacked_bar(axes[i, 0], ct, cols, False, f"{var} vs severity (counts)", var)
    stacked_bar(axes[i, 1], ct, cols, True, f"{var} vs severity (%)", var)
save(fig, "05_light_visibility_vs_severity.png")

# =====================================================================
# 6. Traffic violation and initial observation counts
# =====================================================================
header("6. Traffic violations and initial observations")
tv = flag_counts(df, "tv_", exclude=("tv_any_violation", "tv_conflict"))
io = flag_counts(df, "io_")
print("Traffic violation flags:\n", tv.to_string())
print("\nInitial observation flags:\n", io.to_string())
print("\nCrashes with any recorded violation:",
      int(pd.to_numeric(df["tv_any_violation"], errors="coerce").fillna(0).sum()),
      "| tv_conflict rows (none + violation both ticked):",
      int(pd.to_numeric(df["tv_conflict"], errors="coerce").fillna(0).sum()))

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
sns.barplot(x=tv.values, y=tv.index, ax=axes[0], color="firebrick")
axes[0].bar_label(axes[0].containers[0])
axes[0].set_title("Traffic violations (multi-valued flags)")
axes[0].set_xlabel("Crashes")
sns.barplot(x=io.values, y=io.index, ax=axes[1], color="darkorange")
axes[1].bar_label(axes[1].containers[0])
axes[1].set_title("Initial observations (multi-valued flags)")
axes[1].set_xlabel("Crashes")
save(fig, "06_violations_and_initial_obs.png")

# =====================================================================
# 7. Crashes per segment along the road (chainage), and per km
# =====================================================================
header("7. Crashes along the road")
geo = df.copy()
if EXCLUDE_FAR_SNAP:
    far = pd.to_numeric(geo["snap_far_300m"], errors="coerce").fillna(0) == 1
    print(f"Excluded {int(far.sum())} crashes snapped >300 m from the road")
    geo = geo[~far]
geo = geo.dropna(subset=["chainage_m"]).copy()
n_end = int(pd.to_numeric(geo["snap_at_line_end"], errors="coerce").fillna(0).sum())
print(f"Crashes used: {len(geo)} | snapped to a line end: {n_end} "
      "(chainage at ends is less reliable)")
road_len = geo["chainage_m"].max()
print(f"Max chainage: {road_len / 1000:.1f} km")


def binned(df_, size_m):
    """Crashes per chainage bin of size_m, stacked by severity, with empty bins kept."""
    b = (df_["chainage_m"] // size_m).astype(int)
    n_bins = int(road_len // size_m) + 1
    ct = (pd.crosstab(b, df_["severity"]).reindex(columns=sev_order, fill_value=0)
            .reindex(range(n_bins), fill_value=0))
    return ct


seg = binned(geo, SEG_LEN_M)
km = binned(geo, 1000)
print(f"\n{SEG_LEN_M} m bins: {len(seg)} | empty: {int((seg.sum(axis=1) == 0).sum())} | "
      f"max in one bin: {int(seg.sum(axis=1).max())}")
print("Top 10 segments (by segment_id):\n",
      geo["segment_id"].value_counts().head(10).to_string())
print("\nTop 5 km blocks:\n",
      km.sum(axis=1).sort_values(ascending=False).head(5).rename("crashes").to_string())

for ct, size, fname in [(seg, SEG_LEN_M, "07a_crashes_per_segment.png"),
                        (km, 1000, "07b_crashes_per_km.png")]:
    fig, ax = plt.subplots(figsize=(15, 5))
    x = (ct.index.values * size) / 1000
    bottom = np.zeros(len(ct))
    for s in sev_order:
        ax.bar(x, ct[s].values, width=size / 1000 * 0.9, bottom=bottom, align="edge",
               color=sev_colors[s], label=s, edgecolor="white", linewidth=0.3)
        bottom += ct[s].values
    ax.set_xlabel("Chainage along NH-48 (km from start of the study line)")
    ax.set_ylabel("Crashes")
    ax.set_title(f"Crashes per {'500 m segment' if size == SEG_LEN_M else 'km'} "
                 "along the road")
    ax.legend(title="Severity", fontsize=8)
    save(fig, fname)

# =====================================================================
# 8. Pedestrian crashes along the road
# =====================================================================
header("8. Pedestrian crashes")
ped_cols = ["pedestrians_killed", "pedestrians_grievous", "pedestrians_minor",
            "pedestrians_no_injury"]
ped = geo[ped_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
geo["ped_involved"] = (ped.sum(axis=1) > 0)
geo["ped_killed"] = ped["pedestrians_killed"] > 0
p = geo[geo["ped_involved"]]
print(f"Pedestrian crashes: {len(p)} of {len(geo)} ({len(p) / len(geo):.0%})")
print(f"Pedestrian-fatal crashes: {int(geo['ped_killed'].sum())} | "
      f"pedestrians killed: {int(ped['pedestrians_killed'].sum())}")
if len(p):
    print("Severity of pedestrian crashes:\n", p["severity"].value_counts().to_string())
    top = (p["chainage_m"] // 1000).astype(int).value_counts().head(5)
    print("Top 5 km blocks for pedestrian crashes (km index: count):\n", top.to_string())

n_km = int(road_len // 1000) + 1
ped_km = ((p["chainage_m"] // 1000).astype(int).value_counts()
          .reindex(range(n_km), fill_value=0))
ped_km_fatal = ((geo[geo["ped_killed"]]["chainage_m"] // 1000).astype(int).value_counts()
                .reindex(range(n_km), fill_value=0))

fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
axes[0].bar(ped_km.index, ped_km.values, width=0.9, align="edge",
            color="lightsteelblue", label="Pedestrian crashes")
axes[0].bar(ped_km_fatal.index, ped_km_fatal.values, width=0.9, align="edge",
            color="crimson", label="Pedestrian killed")
axes[0].set_ylabel("Crashes per km")
axes[0].set_title("Pedestrian crashes along NH-48")
axes[0].legend()
axes[1].scatter(p["chainage_m"] / 1000, np.zeros(len(p)), marker="|", s=300,
                color=np.where(p["ped_killed"], "crimson", "steelblue"))
axes[1].set_yticks([])
axes[1].set_xlabel("Chainage along NH-48 (km)")
axes[1].set_title("Each pedestrian crash (red = pedestrian killed)", fontsize=9)
save(fig, "08_pedestrian_crashes_along_road.png")

print(f"\nDone. Figures in: {OUT_DIR.resolve()}")