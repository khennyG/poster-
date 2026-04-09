#!/usr/bin/env python3
"""
Phase 2 — Feature Engineering, Data Splitting, and Baseline Models

Capstone research project on predicting toileting events for an individual with
severe autism. This phase:
  1. Defines three feature sets (Schedule/History, Wearable, Combined)
  2. Creates a temporal train / val / test split
  3. Documents and handles missingness
  4. Runs feature-correlation analysis
  5. Builds an hourly-rate baseline to set the performance floor

Outputs (all written to phase2_feature_engineering/):
    feature_sets.json
    baseline_results.json
    Fig14_feature_correlation_heatmap_setA.png
    Fig15_feature_correlation_heatmap_setB.png
    Fig16_baseline_precision_recall_curves.png
    phase2_summary_report.txt
"""

import json
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Global style configuration
# ──────────────────────────────────────────────
sns.set_style("whitegrid")
COLORS = ["#2C6E91", "#E07B54", "#4CA77B", "#8B6CAE", "#D4A84B"]
plt.rcParams["font.family"] = "DejaVu Sans"

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "autism.csv"
OUT_DIR = BASE_DIR / "phase2_feature_engineering"
OUT_DIR.mkdir(exist_ok=True)


def clean_axes(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = pd.to_datetime(df["date"])
print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Feature Set Definitions
# ══════════════════════════════════════════════════════════════════════════════

# Columns to ALWAYS EXCLUDE from modelling
EXCLUDE_COLS = [
    "timestamp",
    "date",
    "bladder_pressure_proxy",  # latent simulation variable — NEVER use
    # Target labels
    "urination_next_15min",
    "urination_next_30min",
    "urination_next_60min",
    "bowel_next_30min",
    "bowel_next_60min",
]

# Feature Set A: Schedule + Routine + History + Intake
# (Everything a caregiver could track WITHOUT a wearable device)
FEATURE_SET_A = [
    # Calendar / time
    "year", "month", "day", "day_of_week", "is_weekend",
    "hour", "minute", "day_of_year", "season",
    # Sleep / routine
    "asleep_flag", "wake_up_event", "bedtime_event",
    "minutes_since_waking", "minutes_until_sleep",
    "sleep_duration_prev_night_hours", "nap_flag",
    # Intake / nutrition
    "meal_event_flag", "meal_type", "meal_size",
    "meal_caloric_load_est", "meal_water_content_score",
    "caffeine_flag", "fiber_score",
    "water_intake_ml_current_window", "water_intake_event_flag",
    "cumulative_water_today_ml",
    "post_meal_flag", "sedentary_after_meal_flag",
    "cumulative_water_since_last_void_ml",
    # Toileting history
    "urination_event", "bowel_event",
    "minutes_since_last_urination", "minutes_since_last_bowel",
    "urinations_so_far_today", "bowels_so_far_today",
    "rolling_avg_intervoid_interval_hours",
    "rolling_avg_bowel_interval_hours",
    "last_24h_urination_count", "last_24h_bowel_count",
    # Activity (observable without wearable — caregiver-observed)
    "activity_level", "activity_score_numeric",
    "recent_activity_30min_mean", "recent_activity_60min_mean",
    "steps_proxy_current_window",
]

# Feature Set B: Wearable Signals Only
FEATURE_SET_B = [
    "heart_rate", "hrv",
    "heart_rate_30min_mean", "heart_rate_60min_mean",
    "hrv_30min_mean", "hrv_60min_mean",
    "heart_rate_trend_30min", "hrv_trend_30min",
]

# Feature Set C: Combined (A + B)
FEATURE_SET_C = FEATURE_SET_A + FEATURE_SET_B

# Targets
TARGETS = [
    "urination_next_15min",
    "urination_next_30min",
    "urination_next_60min",
    "bowel_next_30min",
    "bowel_next_60min",
]

# ── Print summary ──
print("FEATURE SET DEFINITIONS")
print("=" * 60)
print(f"\nFeature Set A (Schedule + History + Intake): {len(FEATURE_SET_A)} features")
for f in FEATURE_SET_A:
    print(f"  - {f}")
print(f"\nFeature Set B (Wearable Only): {len(FEATURE_SET_B)} features")
for f in FEATURE_SET_B:
    print(f"  - {f}")
print(f"\nFeature Set C (Combined): {len(FEATURE_SET_C)} features")
print(f"\nExcluded columns: {EXCLUDE_COLS}")
print(f"\nTarget columns: {TARGETS}")

# Leakage sanity check
for t in TARGETS:
    assert t not in FEATURE_SET_C, f"LEAKAGE: {t} found in feature set!"
assert "bladder_pressure_proxy" not in FEATURE_SET_C, \
    "LEAKAGE: bladder_pressure_proxy in features!"
print("\n✓ No leakage detected in feature sets.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Temporal Data Split
# ══════════════════════════════════════════════════════════════════════════════

df["split"] = "train"
df.loc[df["year"] == 2024, "split"] = "val"
df.loc[df["year"] == 2025, "split"] = "test"

train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

print("TEMPORAL SPLIT SUMMARY")
print("=" * 60)
print(f"\nTraining set:   {len(train):>8,} rows  ({len(train)/len(df)*100:.1f}%)  "
      f"{train['date'].min().date()} to {train['date'].max().date()}")
print(f"Validation set: {len(val):>8,} rows  ({len(val)/len(df)*100:.1f}%)  "
      f"{val['date'].min().date()} to {val['date'].max().date()}")
print(f"Test set:       {len(test):>8,} rows  ({len(test)/len(df)*100:.1f}%)  "
      f"{test['date'].min().date()} to {test['date'].max().date()}")

# ── Target distribution stability ──
print("\nTARGET DISTRIBUTION BY SPLIT")
print("=" * 60)
print(f"\n{'Target':<25} {'Train %':>10} {'Val %':>10} {'Test %':>10}")
print("-" * 55)
for target in TARGETS:
    tr = train[target].mean() * 100
    va = val[target].mean() * 100
    te = test[target].mean() * 100
    print(f"{target:<25} {tr:>9.2f}% {va:>9.2f}% {te:>9.2f}%")

for split_name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
    daily = split_df.groupby("date").agg(
        urinations=("urination_event", "sum"),
        bowels=("bowel_event", "sum"),
    )
    print(f"\n{split_name}: Avg daily urinations = {daily['urinations'].mean():.2f} "
          f"(SD {daily['urinations'].std():.2f}), "
          f"Avg daily bowels = {daily['bowels'].mean():.2f} "
          f"(SD {daily['bowels'].std():.2f})")
print()


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Missingness Handling
# ══════════════════════════════════════════════════════════════════════════════

print("MISSINGNESS ANALYSIS")
print("=" * 60)

all_features = FEATURE_SET_C
missing = df[all_features].isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing values in feature columns:")
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"  {col}: {count:,} ({pct:.2f}%)")

print("\nMISSINGNESS HANDLING STRATEGY:")
print("-" * 40)
print("""
1. minutes_since_waking — Missing during sleep.
   Fill with -1 as sentinel ('currently asleep').
2. minutes_until_sleep — Same logic. Fill with -1.
3. sleep_duration_prev_night_hours — Missing at start.
   Fill with training-set median.
4. rolling_avg_intervoid_interval_hours — Missing at start.
   Fill with training-set median.
5. rolling_avg_bowel_interval_hours — Same. Training median.
6. heart_rate_trend_30min, hrv_trend_30min — Very few rows.
   Fill with 0 (no trend).
""")

# Compute medians from TRAINING set only (no leakage)
train_median_sleep_dur = train["sleep_duration_prev_night_hours"].median()
train_median_ivoid = train["rolling_avg_intervoid_interval_hours"].median()
train_median_bowel_int = train["rolling_avg_bowel_interval_hours"].median()

df["minutes_since_waking"] = df["minutes_since_waking"].fillna(-1)
df["minutes_until_sleep"]  = df["minutes_until_sleep"].fillna(-1)
df["sleep_duration_prev_night_hours"] = df["sleep_duration_prev_night_hours"].fillna(
    train_median_sleep_dur)
df["rolling_avg_intervoid_interval_hours"] = df[
    "rolling_avg_intervoid_interval_hours"].fillna(train_median_ivoid)
df["rolling_avg_bowel_interval_hours"] = df[
    "rolling_avg_bowel_interval_hours"].fillna(train_median_bowel_int)
df["heart_rate_trend_30min"] = df["heart_rate_trend_30min"].fillna(0)
df["hrv_trend_30min"]       = df["hrv_trend_30min"].fillna(0)

remaining_nulls = df[all_features].isnull().sum().sum()
print(f"Remaining null values in feature columns after imputation: {remaining_nulls}")

# ── Categorical encoding ──
print("\nCATEGORICAL ENCODING:")
print("-" * 40)
for col in ["meal_type", "meal_size", "activity_level"]:
    print(f"  {col}: {df[col].unique().tolist()}")

df = pd.get_dummies(df, columns=["meal_type", "meal_size", "activity_level"],
                    drop_first=False)

new_meal_type_cols = sorted([c for c in df.columns if c.startswith("meal_type_")])
new_meal_size_cols = sorted([c for c in df.columns if c.startswith("meal_size_")])
new_activity_cols  = sorted([c for c in df.columns if c.startswith("activity_level_")])

# Update Feature Set A: remove original categorical names, add one-hot versions
for orig, new_cols in [("meal_type", new_meal_type_cols),
                       ("meal_size", new_meal_size_cols),
                       ("activity_level", new_activity_cols)]:
    if orig in FEATURE_SET_A:
        FEATURE_SET_A.remove(orig)
        FEATURE_SET_A.extend(new_cols)

# Rebuild Feature Set C
FEATURE_SET_C = FEATURE_SET_A + FEATURE_SET_B

print(f"\nAfter encoding:")
print(f"  Feature Set A: {len(FEATURE_SET_A)} features")
print(f"  Feature Set B: {len(FEATURE_SET_B)} features")
print(f"  Feature Set C: {len(FEATURE_SET_C)} features")
print(f"  New one-hot columns: "
      f"{new_meal_type_cols + new_meal_size_cols + new_activity_cols}")

# Re-split after transformations
train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

# Save feature lists
feature_sets = {
    "A": FEATURE_SET_A,
    "B": FEATURE_SET_B,
    "C": FEATURE_SET_C,
    "targets": TARGETS,
    "excluded": EXCLUDE_COLS,
}
with open(OUT_DIR / "feature_sets.json", "w") as f:
    json.dump(feature_sets, f, indent=2)
print(f"\n✓ Feature set definitions saved to {OUT_DIR / 'feature_sets.json'}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Feature Correlation Analysis
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.1 Correlation heatmap for Feature Set A ──
print("Fig14  Feature correlation heatmap (Set A) …")

numeric_A = [f for f in FEATURE_SET_A if train[f].dtype in ("int64", "float64",
                                                              "int32", "float32",
                                                              "bool", "uint8")]
corr_A = train[numeric_A].corr()

# Extract top correlated pairs
pairs = []
cols = corr_A.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append({
            "Feature 1": cols[i],
            "Feature 2": cols[j],
            "Correlation": corr_A.iloc[i, j],
        })
pairs_df = pd.DataFrame(pairs)
pairs_df["abs_corr"] = pairs_df["Correlation"].abs()
top_pairs = pairs_df.nlargest(20, "abs_corr")

print("\nTOP 20 CORRELATED FEATURE PAIRS (Set A)")
print("=" * 70)
for _, row in top_pairs.iterrows():
    print(f"  {row['Feature 1']:<45s} x {row['Feature 2']:<45s} "
          f"r = {row['Correlation']:+.3f}")

# Select a compact set of features for the heatmap
top_feat_set = set()
for _, row in top_pairs.head(10).iterrows():
    top_feat_set.add(row["Feature 1"])
    top_feat_set.add(row["Feature 2"])
top_features = sorted(top_feat_set)

fig, ax = plt.subplots(figsize=(14, 11))
corr_sub = train[top_features].corr()
mask = np.triu(np.ones_like(corr_sub, dtype=bool), k=1)

# Human-readable tick labels (replace underscores, truncate)
def _label(c, maxlen=28):
    return c.replace("_", " ").title()[:maxlen]

sns.heatmap(corr_sub, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            ax=ax, annot_kws={"size": 8},
            xticklabels=[_label(c) for c in top_features],
            yticklabels=[_label(c) for c in top_features])
ax.set_title("Feature Correlation: Top Correlated Features (Set A)",
             fontsize=16, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig14_feature_correlation_heatmap_setA.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ── 4.2 Correlation heatmap for Feature Set B ──
print("Fig15  Feature correlation heatmap (Set B) …")

corr_B = train[FEATURE_SET_B].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask_B = np.triu(np.ones_like(corr_B, dtype=bool), k=1)
sns.heatmap(corr_B, mask=mask_B, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            ax=ax, annot_kws={"size": 10},
            xticklabels=[_label(c, 30) for c in FEATURE_SET_B],
            yticklabels=[_label(c, 30) for c in FEATURE_SET_B])
ax.set_title("Feature Correlation: Wearable Signal Features (Set B)",
             fontsize=16, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.tick_params(axis="y", rotation=0, labelsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig15_feature_correlation_heatmap_setB.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Baseline Model (Time-of-Day Hourly Rate)
# ══════════════════════════════════════════════════════════════════════════════

print("BASELINE MODEL: HOURLY RATE")
print("=" * 60)

results = {}

for target in TARGETS:
    # Training-set hourly event rates
    hourly_rates = train.groupby("hour")[target].mean()

    # Map to validation set
    val_pred_proba = val["hour"].map(hourly_rates).values
    val_true = val[target].values

    # Precision–Recall curve and AUPRC
    ap = average_precision_score(val_true, val_pred_proba)

    prec, rec, thresholds = precision_recall_curve(val_true, val_pred_proba)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_f1  = float(f1_scores[best_idx])
    best_prec = float(prec[best_idx])
    best_rec  = float(rec[best_idx])

    try:
        auc_roc = roc_auc_score(val_true, val_pred_proba)
    except Exception:
        auc_roc = float("nan")

    results[target] = {
        "AUPRC": ap,
        "Best F1": best_f1,
        "Precision at Best F1": best_prec,
        "Recall at Best F1": best_rec,
        "Best Threshold": best_threshold,
        "AUROC": auc_roc,
        "Positive Rate": float(val_true.mean()),
    }

    print(f"\n{target}:")
    print(f"  Validation positive rate: {val_true.mean()*100:.2f}%")
    print(f"  AUPRC:           {ap:.4f}")
    print(f"  AUROC:           {auc_roc:.4f}")
    print(f"  Best F1:         {best_f1:.4f}  (threshold={best_threshold:.4f})")
    print(f"  Precision @ F1:  {best_prec:.4f}")
    print(f"  Recall @ F1:     {best_rec:.4f}")

# Save baseline results
with open(OUT_DIR / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Baseline results saved to {OUT_DIR / 'baseline_results.json'}\n")


# ── 5.2 Precision-Recall curves for all targets ──
print("Fig16  Baseline precision-recall curves …")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    ax = axes[i]
    hourly_rates = train.groupby("hour")[target].mean()
    val_pred_proba = val["hour"].map(hourly_rates).values
    val_true = val[target].values

    prec, rec, _ = precision_recall_curve(val_true, val_pred_proba)
    ap = average_precision_score(val_true, val_pred_proba)

    ax.plot(rec, prec, linewidth=2.5, color=COLORS[0],
            label=f"Hourly Rate (AUPRC={ap:.3f})")

    baseline_rate = val_true.mean()
    ax.axhline(y=baseline_rate, color="gray", linestyle="--", linewidth=1.5,
               label=f"No Skill ({baseline_rate:.3f})")

    # Clean title
    clean_name = target.replace("_", " ").replace("next", "Next").title()
    ax.set_title(clean_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, framealpha=0.9)
    clean_axes(ax)

# Hide unused 6th subplot
axes[5].set_visible(False)

plt.suptitle("Baseline Model (Hourly Rate): Precision-Recall Curves on Validation Set",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig16_baseline_precision_recall_curves.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Summary Report
# ══════════════════════════════════════════════════════════════════════════════

print("Generating Phase 2 summary report …")

report = f"""\
PHASE 2: FEATURE ENGINEERING AND BASELINE SUMMARY
{'=' * 60}

FEATURE SET DEFINITIONS
  Feature Set A (Schedule + History + Intake): {len(FEATURE_SET_A)} features
  Feature Set B (Wearable Only):               {len(FEATURE_SET_B)} features
  Feature Set C (Combined A + B):              {len(FEATURE_SET_C)} features

EXCLUDED FROM ALL MODELS
  - bladder_pressure_proxy (latent simulation variable)
  - timestamp, date (index columns)
  - All 5 target columns

TEMPORAL SPLIT
  Training:   {len(train):>8,} rows  ({len(train)/len(df)*100:.1f}%)  2021-01-01 to 2023-12-31
  Validation: {len(val):>8,} rows  ({len(val)/len(df)*100:.1f}%)  2024-01-01 to 2024-12-31
  Test:       {len(test):>8,} rows  ({len(test)/len(df)*100:.1f}%)  2025-01-01 to 2025-12-31

MISSINGNESS HANDLING
  minutes_since_waking:  Filled with -1 (sentinel for 'asleep')
  minutes_until_sleep:   Filled with -1 (sentinel for 'asleep')
  sleep_duration_prev_night_hours: Filled with training median ({train_median_sleep_dur:.2f})
  rolling_avg_intervoid_interval_hours: Filled with training median ({train_median_ivoid:.2f})
  rolling_avg_bowel_interval_hours: Filled with training median ({train_median_bowel_int:.2f})
  heart_rate_trend_30min: Filled with 0
  hrv_trend_30min: Filled with 0

CATEGORICAL ENCODING
  meal_type: One-hot encoded -> {new_meal_type_cols}
  meal_size: One-hot encoded -> {new_meal_size_cols}
  activity_level: One-hot encoded -> {new_activity_cols}

BASELINE MODEL RESULTS (Hourly Rate, Validation Set)
"""

for target, res in results.items():
    report += f"""
  {target}:
    Positive rate: {res['Positive Rate']*100:.2f}%
    AUPRC:  {res['AUPRC']:.4f}
    AUROC:  {res['AUROC']:.4f}
    Best F1: {res['Best F1']:.4f} (threshold={res['Best Threshold']:.4f})
    Precision @ Best F1: {res['Precision at Best F1']:.4f}
    Recall @ Best F1: {res['Recall at Best F1']:.4f}
"""

report += """
INTERPRETATION
  The hourly rate baseline captures only the circadian pattern.
  Any model using features beyond hour-of-day must substantially
  exceed these AUPRC values to justify its added complexity.
  These numbers set the performance floor for Phase 3 modeling.
"""

report_path = OUT_DIR / "phase2_summary_report.txt"
with open(report_path, "w") as fout:
    fout.write(report)
print(report)
print(f"  ✓ report saved to {report_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Final checklist
# ══════════════════════════════════════════════════════════════════════════════
expected = [
    "feature_sets.json",
    "baseline_results.json",
    "Fig14_feature_correlation_heatmap_setA.png",
    "Fig15_feature_correlation_heatmap_setB.png",
    "Fig16_baseline_precision_recall_curves.png",
    "phase2_summary_report.txt",
]

print("=" * 50)
print("OUTPUT CHECKLIST")
print("=" * 50)
all_ok = True
for fname in expected:
    exists = (OUT_DIR / fname).exists()
    status = "✓" if exists else "✗ MISSING"
    if not exists:
        all_ok = False
    print(f"  {status}  {fname}")
print()
if all_ok:
    print("All 6 outputs generated successfully.")
else:
    print("WARNING: some outputs are missing — check errors above.")
print("\nDone.")
