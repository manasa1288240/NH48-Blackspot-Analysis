"""
NH-48 early-warning model (v2)
==============================
Question: for a 500 m segment that does NOT yet meet the blackspot criterion
after two observed years, will it meet it by the end of the next 3-year window?

Criterion (same as step 4): >= 5 fatal + grievous crashes in a segment over a
3-year window. Years are May-April study years (column `study_year`).

Time design (no leakage)
  TRAIN window 2022-23..2024-25: features from the first two years only.
  TEST  window 2023-24..2025-26: features from the first two years only,
        label = criterion met over all three years.
  FORWARD watchlist: window 2024-25..2026-27. Features from 2024-25 and
        2025-26 only. The label is unknown; this is a real forecast.

What this script is careful about
  * ties: a rule that gives many segments the same score has no meaningful
    "top 10", so we report (a) tie-aware precision@k (average over random
    tie-breaks) and (b) all methods at the SAME review budget.
  * the model is compared with simple baselines using a PAIRED bootstrap,
    because two AUCs from the same 59 segments are not independent.
  * bootstrap CIs resample segments; neighbouring segments are correlated, so
    treat the intervals as optimistic.

Assumptions that are yours, not learned: the APW severity weights (WEIGHTS) and the
5-crash threshold. Input files are never modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------------------------------------------------------ config
ROOT = Path(__file__).resolve().parents[1]
CRASH_FILE = ROOT / "data" / "crash_segmented.csv"
SEG_FILE = ROOT / "data" / "nh48_segments.csv"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

YEARS = ["2022-23", "2023-24", "2024-25", "2025-26"]
THRESHOLD = 5
N_BOOT, N_TIE = 2000, 1000
SEED = 42
# Same APW weights as bsri_calculation2.py (keep the two scripts in sync).
# Confirm and cite the source of these weights before calling them "published".
WEIGHTS = {"Fatal": 6, "Grievous": 3, "Minor hospitalized": 0.8,
           "Minor non-hospitalized": 0.8, "No injury": 0.2}
FEATS = ["fg_2yr", "trend", "pts_2yr", "nbr_fg"]
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ data
def load(drop_duplicates=False):
    crashes = pd.read_csv(CRASH_FILE)
    segs = pd.read_csv(SEG_FILE)
    rc = crashes["road_classification"].astype(str).str.lower()
    scope = (rc.str.contains("expressway|surat city")
             | (pd.to_numeric(crashes["geocode_says_sh"], errors="coerce").fillna(0) == 1)
             | (crashes["district"].astype(str).str.upper() == "SURAT CITY"))
    crashes = crashes[~scope].copy()
    if drop_duplicates:
        crashes = crashes[crashes["possible_duplicate"] != 1]
    crashes["fg"] = crashes["severity"].isin(["Fatal", "Grievous"]).astype(int)
    crashes["pts"] = crashes["severity"].map(WEIGHTS)
    return crashes, segs, int(scope.sum())


def make_tables(crashes, segs):
    ids = segs["segment_id"].tolist()
    def pv(col):
        t = crashes.pivot_table(index="segment_id", columns="study_year",
                                values=col, aggfunc="sum", fill_value=0)
        return t.reindex(index=ids, columns=YEARS, fill_value=0)
    return ids, pv("fg"), pv("pts")


def neighbours(v):
    """Sum of the two adjacent 500 m segments (spatial context)."""
    v = np.asarray(v, float)
    return np.r_[v[1:], 0] + np.r_[0, v[:-1]]


def build(ids, FG, PTS, y1, y2, y3=None):
    """Features use ONLY years y1, y2. Label (if y3 given) uses y1+y2+y3."""
    df = pd.DataFrame({"segment_id": ids})
    df["fg_y1"], df["fg_y2"] = FG[y1].values, FG[y2].values
    df["fg_2yr"] = df.fg_y1 + df.fg_y2
    df["trend"] = df.fg_y2 - df.fg_y1
    df["pts_2yr"] = PTS[y1].values + PTS[y2].values
    df["nbr_fg"] = neighbours(df.fg_2yr)
    if y3 is not None:
        df["label"] = ((df.fg_2yr + FG[y3].values) >= THRESHOLD).astype(int)
    return df[df.fg_2yr < THRESHOLD].reset_index(drop=True)   # not yet qualifying


def fit_model(train):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    return m.fit(train[FEATS], train["label"])


# ------------------------------------------------------------------ metrics
def order_random_ties(s):
    """Rank high -> low; ties broken at random (so ties cannot flatter a method)."""
    return np.lexsort((rng.random(len(s)), -s))


def tie_aware_at_k(y, s, k):
    """Mean precision and recall of the top-k, averaged over random tie-breaks."""
    p = r = 0.0
    for _ in range(N_TIE):
        top = order_random_ties(s)[:k]
        p += y[top].mean()
        r += y[top].sum() / max(y.sum(), 1)
    return p / N_TIE, r / N_TIE


def paired_auc_ci(y, s_a, s_b):
    """95% CI for AUC(a) - AUC(b), resampling segments."""
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            diffs.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    return np.percentile(diffs, [2.5, 97.5])


def auc_ci(y, s):
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def evaluate(test, label):
    """AUC (+CI), paired comparison and equal-budget precision/recall."""
    y = test["label"].values
    methods = {
        "Logistic model": test["p_model"].values,
        "Rank by FG count (2 yrs)": test["fg_2yr"].values.astype(float),
        "Rank by weighted points (2 yrs)": test["pts_2yr"].values.astype(float),
        "Rank by last year only": test["fg_y2"].values.astype(float),
    }
    n_rule = int((test["fg_2yr"] >= 3).sum())          # budget = size of the "FG>=3" rule
    rows = []
    for name, s in methods.items():
        lo, hi = auc_ci(y, s)
        p, r = tie_aware_at_k(y, s, n_rule)
        rows.append({"method": name, "AUC": round(roc_auc_score(y, s), 3),
                     "AUC_CI_low": round(lo, 3), "AUC_CI_high": round(hi, 3),
                     f"precision@{n_rule}": round(p, 2), f"recall@{n_rule}": round(r, 2)})
    rule = (test["fg_2yr"] >= 3).values
    rows.append({"method": "Rule: FG>=3 (flags %d)" % n_rule, "AUC": np.nan,
                 "AUC_CI_low": np.nan, "AUC_CI_high": np.nan,
                 f"precision@{n_rule}": round(y[rule].mean(), 2),
                 f"recall@{n_rule}": round(y[rule].sum() / y.sum(), 2)})
    res = pd.DataFrame(rows)
    lo, hi = paired_auc_ci(y, methods["Logistic model"], methods["Rank by FG count (2 yrs)"])
    print(f"\n=== {label}: {len(y)} segments, {y.sum()} crossed the threshold ===")
    print(res.to_string(index=False))
    print(f"Paired AUC difference, model minus FG-count: "
          f"{roc_auc_score(y, methods['Logistic model']) - roc_auc_score(y, methods['Rank by FG count (2 yrs)']):+.3f} "
          f"(95% CI {lo:+.3f} to {hi:+.3f})  -> CI containing 0 means no demonstrated difference")
    print(f"Brier: model {brier_score_loss(y, test.p_model):.3f} vs base rate "
          f"{brier_score_loss(y, np.full(len(y), test.p_base.iloc[0])):.3f}")
    return res


def calibration_table(test, bins=(0, 0.2, 0.5, 1.0001)):
    t = test.copy()
    t["bin"] = pd.cut(t.p_model, bins=list(bins), right=False)
    out = t.groupby("bin", observed=True).agg(
        segments=("label", "size"), mean_predicted=("p_model", "mean"),
        observed_rate=("label", "mean")).round(2)
    print("\nCalibration (predicted vs observed crossing rate):")
    print(out.to_string())
    return out


# ------------------------------------------------------------------ main
def main():
    crashes, segs, n_scope = load()
    print(f"Scope rows removed: {n_scope} (expect 13)")
    ids, FG, PTS = make_tables(crashes, segs)

    train = build(ids, FG, PTS, "2022-23", "2023-24", "2024-25")
    test = build(ids, FG, PTS, "2023-24", "2024-25", "2025-26")
    print(f"Train: {len(train)} rows ({train.label.sum()} positives) | "
          f"Test: {len(test)} rows ({test.label.sum()} positives)")

    model = fit_model(train)
    test["p_model"] = model.predict_proba(test[FEATS])[:, 1]
    test["p_base"] = train.label.mean()
    res = evaluate(test, "PRIMARY (scope excluded, duplicates kept)")
    cal = calibration_table(test)
    print("Standardised coefficients:",
          {f: round(float(c), 2) for f, c in zip(FEATS, model[-1].coef_[0])})

    # Sensitivity: drop possible-duplicate rows
    c2, s2, _ = load(drop_duplicates=True)
    ids2, FG2, PTS2 = make_tables(c2, s2)
    tr2 = build(ids2, FG2, PTS2, "2022-23", "2023-24", "2024-25")
    te2 = build(ids2, FG2, PTS2, "2023-24", "2024-25", "2025-26")
    te2["p_model"] = fit_model(tr2).predict_proba(te2[FEATS])[:, 1]
    te2["p_base"] = tr2.label.mean()
    res2 = evaluate(te2, "SENSITIVITY (possible duplicates removed)")

    # Where does the simple rule go wrong?
    fp = test[(test.fg_2yr >= 3) & (test.label == 0)][
        ["segment_id", "fg_y1", "fg_y2", "fg_2yr", "nbr_fg"]]
    print("\nRule false positives (>=3 FG but did not cross):")
    print(fp.to_string(index=False))

    # Forward watchlist: train on BOTH windows whose labels are known by now
    both = pd.concat([train, test[train.columns]], ignore_index=True)
    final = fit_model(both)
    fwd = build(ids, FG, PTS, "2024-25", "2025-26")      # no label: genuine forecast
    fwd["p_cross_next_window"] = final.predict_proba(fwd[FEATS])[:, 1]
    fwd = fwd.merge(segs[["segment_id", "chain_start_m", "chain_end_m", "mid_lat", "mid_lon"]],
                    on="segment_id").sort_values("p_cross_next_window", ascending=False)
    fwd["rule_FG_ge_3"] = (fwd.fg_2yr >= 3).astype(int)
    print("\nFORWARD WATCHLIST (window 2024-25..2026-27; top 10):")
    print(fwd[["segment_id", "chain_start_m", "fg_y1", "fg_y2", "fg_2yr",
               "p_cross_next_window"]].head(10).round(2).to_string(index=False))

    res.to_csv(OUT / "ew_metrics_primary.csv", index=False)
    res2.to_csv(OUT / "ew_metrics_no_duplicates.csv", index=False)
    cal.to_csv(OUT / "ew_calibration.csv")
    test.sort_values("p_model", ascending=False).to_csv(OUT / "ew_backtest_2023_26.csv", index=False)
    fwd.to_csv(OUT / "ew_forward_watchlist_2024_27.csv", index=False)
    print("\nSaved to", OUT)
    print("Reminder: the forward model is trained on 2 overlapping windows (115 rows,"
          " not independent). Treat probabilities as rough ranks, not exact risks.")


if __name__ == "__main__":
    main()