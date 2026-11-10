"""
NH-48 crash-severity model
==========================
Question: given the conditions of a crash, how likely is it to be fatal or
grievous?  (BSRI ranks LOCATIONS; this model describes individual CRASHES.)

Target : 1 = Fatal or Grievous, 0 = Minor / No injury.
Split  : by time. Train on study years 2022-23..2024-25, test on 2025-26.
Models : (a) base rate  (b) logistic regression (interpretable)
         (c) gradient boosting (to see whether non-linearity helps).

Read the results with these limits in mind
  * The sample is skewed to severe crashes (~77% fatal/grievous), and minor
    crashes are probably under-recorded. Probabilities are NOT population risks.
  * Collision type is known only AFTER the crash. This is a risk-FACTOR
    model that describes association, not a forecast and not a causal effect.
  * We deliberately do NOT use casualty columns (killed, grievous counts,
    pedestrian casualties): they are the outcome itself.
  * Odds-ratio intervals resample crashes; they do not capture recording error.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
CRASH_FILE = ROOT / "data" / "crash_segmented.csv"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

TRAIN_YEARS = ["2022-23", "2023-24", "2024-25"]
TEST_YEAR = "2025-26"
N_BOOT = 1000
rng = np.random.default_rng(42)


def load():
    d = pd.read_csv(CRASH_FILE)
    rc = d["road_classification"].astype(str).str.lower()
    scope = (rc.str.contains("expressway|surat city")
             | (pd.to_numeric(d["geocode_says_sh"], errors="coerce").fillna(0) == 1)
             | (d["district"].astype(str).str.upper() == "SURAT CITY"))
    print("Scope rows removed:", int(scope.sum()), "(expect 13)")
    return d[~scope].copy()


def build_features(d):
    """Small, interpretable feature set. Every column is a 0/1 indicator."""
    X = pd.DataFrame(index=d.index)
    ct = d["collision_type"].fillna("").str.lower()
    X["pedestrian_crash"] = ct.str.contains("pedestrian").astype(int)
    X["other_non_vehicle_crash"] = (~ct.str.contains("pedestrian")
                                    & ~ct.str.contains("vehicle to vehicle")).astype(int)
    # reference category = vehicle-to-vehicle
    X["night_20_to_05"] = ((d["hour"] >= 20) | (d["hour"] < 5)).astype(int)
    X["fog_or_mist"] = (pd.to_numeric(d["wx_mist_fog"], errors="coerce").fillna(0) > 0).astype(int)
    X["multi_vehicle"] = (d["n_vehicles"] >= 2).astype(int)
    for c, name in [("tv_high_speed", "high_speed"), ("tv_drunk_driving", "drunk_driving"),
                    ("tv_rash_driving", "rash_driving")]:
        X[name] = (pd.to_numeric(d[c], errors="coerce").fillna(0) > 0).astype(int)
    X["weekend"] = d["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return X


def auc_ci(y, p):
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    d = load()
    d["severe"] = d["severity"].isin(["Fatal", "Grievous"]).astype(int)
    X = build_features(d)
    tr = d["study_year"].isin(TRAIN_YEARS).values
    te = (d["study_year"] == TEST_YEAR).values
    Xtr, ytr, Xte, yte = X[tr], d.loc[tr, "severe"].values, X[te], d.loc[te, "severe"].values
    print(f"Train {tr.sum()} crashes (severe share {ytr.mean():.2f}) | "
          f"Test {te.sum()} crashes (severe share {yte.mean():.2f})")
    print("NOTE: the severe share FELL in the test year, so the base rate shifts over time.")

    # ---- models
    logit = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                        random_state=42).fit(Xtr, ytr)
    preds = {"Base rate (train)": np.full(len(yte), ytr.mean()),
             "Logistic regression": logit.predict_proba(Xte)[:, 1],
             "Gradient boosting": gb.predict_proba(Xte)[:, 1],
             # one-feature baseline: severe rate seen in training for pedestrian / other crashes
             "Pedestrian flag only": np.where(Xte["pedestrian_crash"].values == 1,
                                              ytr[Xtr["pedestrian_crash"].values == 1].mean(),
                                              ytr[Xtr["pedestrian_crash"].values == 0].mean())}

    rows = []
    for name, p in preds.items():
        auc = roc_auc_score(yte, p) if name != "Base rate (train)" else 0.5
        lo, hi = auc_ci(yte, p) if name != "Base rate (train)" else (np.nan, np.nan)
        rows.append({"model": name, "AUC": round(auc, 3), "AUC_CI_low": round(lo, 3),
                     "AUC_CI_high": round(hi, 3), "Brier": round(brier_score_loss(yte, p), 3)})
    res = pd.DataFrame(rows)
    print(f"\nTEST YEAR {TEST_YEAR}")
    print(res.to_string(index=False))
    print("Lower Brier is better. Compare Brier with the base-rate row, not with 0.")

    # ---- odds ratios on ALL data (description, not prediction), bootstrap CI
    full = LogisticRegression(C=1.0, max_iter=1000).fit(X, d["severe"])
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(X), len(X))
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X.iloc[i], d["severe"].values[i])
        boots.append(np.exp(m.coef_[0]))
    boots = np.array(boots)
    orr = pd.DataFrame({
        "factor": X.columns,
        "n_crashes_with_factor": X.sum().values,
        "severe_rate_with_factor": [d.loc[X[c] == 1, "severe"].mean() for c in X.columns],
        "odds_ratio": np.exp(full.coef_[0]),
        "or_ci_low": np.percentile(boots, 2.5, axis=0),
        "or_ci_high": np.percentile(boots, 97.5, axis=0),
    }).round(2).sort_values("odds_ratio", ascending=False)
    orr["interval_excludes_1"] = (orr.or_ci_low > 1) | (orr.or_ci_high < 1)
    print("\nODDS RATIOS (all crashes; reference = vehicle-to-vehicle, daytime, etc.)")
    print("Ridge-penalised (C=1), so values are slightly shrunk toward 1.")
    print("multi_vehicle and pedestrian_crash overlap (pedestrian crashes are almost all single-vehicle),")
    print("so multi_vehicle mainly contrasts multi- vs single-vehicle NON-pedestrian crashes.")
    print(orr.to_string(index=False))

    res.to_csv(OUT / "severity_metrics.csv", index=False)
    orr.to_csv(OUT / "severity_odds_ratios.csv", index=False)
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()