#!/usr/bin/env python3
"""
Phase 3 — Supervised Prediction Modeling

Capstone research project on predicting toileting events for an individual with
severe autism.  This phase trains and evaluates the core prediction models:
  - Logistic Regression (Elastic Net) and LightGBM
  - Across 3 feature sets (A: schedule/history, B: wearable, C: combined)
  - For 5 prediction targets (urination 15/30/60 min, bowel 30/60 min)
  - Total: 30 model configurations

Outputs (all written to phase3_modeling/):
    phase3_results_table.txt
    Fig17_precision_recall_all_models_urination.png
    Fig18_precision_recall_all_models_bowel.png
    Fig19_feature_importance_lightgbm.png
    Fig20_shap_summary_plot.png
    Fig21_shap_beeswarm_plot.png
    Fig22_feature_set_comparison.png
    phase3_summary_report.txt
"""

import json
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────
sns.set_style("whitegrid")
COLORS = ["#2C6E91", "#E07B54", "#4CA77B", "#8B6CAE", "#D4A84B"]
plt.rcParams["font.family"] = "DejaVu Sans"

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "autism.csv"
OUT_DIR = BASE_DIR / "phase3_modeling"
OUT_DIR.mkdir(exist_ok=True)


def clean_axes(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION  (re-run from Phase 2)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading and preparing data …")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = pd.to_datetime(df["date"])

# Temporal split
df["split"] = "train"
df.loc[df["year"] == 2024, "split"] = "val"
df.loc[df["year"] == 2025, "split"] = "test"

# Missingness — training-set medians (no leakage)
train_temp = df[df["split"] == "train"]
train_median_sleep = train_temp["sleep_duration_prev_night_hours"].median()
train_median_ivoid = train_temp["rolling_avg_intervoid_interval_hours"].median()
train_median_bowel = train_temp["rolling_avg_bowel_interval_hours"].median()

df["minutes_since_waking"] = df["minutes_since_waking"].fillna(-1)
df["minutes_until_sleep"]  = df["minutes_until_sleep"].fillna(-1)
df["sleep_duration_prev_night_hours"] = df["sleep_duration_prev_night_hours"].fillna(train_median_sleep)
df["rolling_avg_intervoid_interval_hours"] = df["rolling_avg_intervoid_interval_hours"].fillna(train_median_ivoid)
df["rolling_avg_bowel_interval_hours"] = df["rolling_avg_bowel_interval_hours"].fillna(train_median_bowel)
df["heart_rate_trend_30min"] = df["heart_rate_trend_30min"].fillna(0)
df["hrv_trend_30min"]       = df["hrv_trend_30min"].fillna(0)

# One-hot encode categoricals
df = pd.get_dummies(df, columns=["meal_type", "meal_size", "activity_level"],
                    drop_first=False)

new_meal_type_cols = sorted([c for c in df.columns if c.startswith("meal_type_")])
new_meal_size_cols = sorted([c for c in df.columns if c.startswith("meal_size_")])
new_activity_cols  = sorted([c for c in df.columns if c.startswith("activity_level_")])

# ── Feature set definitions ──
FEATURE_SET_A = [
    "year", "month", "day", "day_of_week", "is_weekend",
    "hour", "minute", "day_of_year", "season",
    "asleep_flag", "wake_up_event", "bedtime_event",
    "minutes_since_waking", "minutes_until_sleep",
    "sleep_duration_prev_night_hours", "nap_flag",
    "meal_event_flag", "meal_caloric_load_est", "meal_water_content_score",
    "caffeine_flag", "fiber_score",
    "water_intake_ml_current_window", "water_intake_event_flag",
    "cumulative_water_today_ml",
    "post_meal_flag", "sedentary_after_meal_flag",
    "cumulative_water_since_last_void_ml",
    "urination_event", "bowel_event",
    "minutes_since_last_urination", "minutes_since_last_bowel",
    "urinations_so_far_today", "bowels_so_far_today",
    "rolling_avg_intervoid_interval_hours",
    "rolling_avg_bowel_interval_hours",
    "last_24h_urination_count", "last_24h_bowel_count",
    "activity_score_numeric",
    "recent_activity_30min_mean", "recent_activity_60min_mean",
    "steps_proxy_current_window",
] + new_meal_type_cols + new_meal_size_cols + new_activity_cols

FEATURE_SET_B = [
    "heart_rate", "hrv",
    "heart_rate_30min_mean", "heart_rate_60min_mean",
    "hrv_30min_mean", "hrv_60min_mean",
    "heart_rate_trend_30min", "hrv_trend_30min",
]

FEATURE_SET_C = FEATURE_SET_A + FEATURE_SET_B

TARGETS = [
    "urination_next_15min", "urination_next_30min", "urination_next_60min",
    "bowel_next_30min", "bowel_next_60min",
]

# Split
train = df[df["split"] == "train"].copy()
val   = df[df["split"] == "val"].copy()
test  = df[df["split"] == "test"].copy()

print(f"  Feature Set A: {len(FEATURE_SET_A)} features")
print(f"  Feature Set B: {len(FEATURE_SET_B)} features")
print(f"  Feature Set C: {len(FEATURE_SET_C)} features")
print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Train All Models (30 total)
# ══════════════════════════════════════════════════════════════════════════════

all_results = {}

feature_sets = {"A": FEATURE_SET_A, "B": FEATURE_SET_B, "C": FEATURE_SET_C}

# Hourly-rate baseline (from Phase 2)
baseline_results = {}
for target in TARGETS:
    hourly_rates = train.groupby("hour")[target].mean()
    val_pred = val["hour"].map(hourly_rates).values
    baseline_results[target] = {
        "val_pred": val_pred,
        "AUPRC": average_precision_score(val[target].values, val_pred),
    }

for fs_name, fs_cols in feature_sets.items():
    print(f"\n{'=' * 60}")
    print(f"FEATURE SET {fs_name} ({len(fs_cols)} features)")
    print(f"{'=' * 60}")

    X_train = train[fs_cols].values
    X_val   = val[fs_cols].values

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    for target in TARGETS:
        y_train = train[target].values
        y_val   = val[target].values

        key = f"{fs_name}_{target}"
        all_results[key] = {}

        # ── Logistic Regression (Elastic Net) ──
        lr = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            C=1.0, max_iter=2000, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict_proba(X_val_scaled)[:, 1]

        lr_auprc = average_precision_score(y_val, lr_pred)
        lr_auroc = roc_auc_score(y_val, lr_pred)

        prec, rec, thresholds = precision_recall_curve(y_val, lr_pred)
        f1s = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1s)
        lr_best_f1     = float(f1s[best_idx])
        lr_best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])

        all_results[key]["LR"] = {
            "AUPRC": lr_auprc,
            "AUROC": lr_auroc,
            "Best_F1": lr_best_f1,
            "Threshold": lr_best_thresh,
            "Precision_at_F1": float(prec[best_idx]),
            "Recall_at_F1": float(rec[best_idx]),
            "val_pred": lr_pred,
        }

        # ── LightGBM ──
        lgb_train    = lgb.Dataset(X_train, label=y_train)
        lgb_val_data = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

        pos_rate = y_train.mean()
        scale_pos = (1 - pos_rate) / pos_rate

        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_pos,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

        lgb_model = lgb.train(
            params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        lgb_pred  = lgb_model.predict(X_val)
        lgb_auprc = average_precision_score(y_val, lgb_pred)
        lgb_auroc = roc_auc_score(y_val, lgb_pred)

        prec, rec, thresholds = precision_recall_curve(y_val, lgb_pred)
        f1s = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1s)
        lgb_best_f1     = float(f1s[best_idx])
        lgb_best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])

        all_results[key]["LightGBM"] = {
            "AUPRC": lgb_auprc,
            "AUROC": lgb_auroc,
            "Best_F1": lgb_best_f1,
            "Threshold": lgb_best_thresh,
            "Precision_at_F1": float(prec[best_idx]),
            "Recall_at_F1": float(rec[best_idx]),
            "val_pred": lgb_pred,
            "model": lgb_model,
            "feature_names": fs_cols,
        }

        base_auprc = baseline_results[target]["AUPRC"]

        print(f"\n  {target} (positive rate: {y_val.mean()*100:.1f}%)")
        print(f"    Baseline AUPRC:  {base_auprc:.4f}")
        print(f"    LR AUPRC:        {lr_auprc:.4f}  "
              f"({(lr_auprc / base_auprc - 1)*100:+.1f}% vs baseline)")
        print(f"    LightGBM AUPRC:  {lgb_auprc:.4f}  "
              f"({(lgb_auprc / base_auprc - 1)*100:+.1f}% vs baseline)")
        print(f"    LR Best F1:      {lr_best_f1:.4f}")
        print(f"    LightGBM F1:     {lgb_best_f1:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Results Summary Table
# ══════════════════════════════════════════════════════════════════════════════

header = (f"{'FS':<5} {'Target':<25} {'Model':<12} "
          f"{'AUPRC':>8} {'AUROC':>8} {'F1':>8} {'vs Base':>10}")
sep = "-" * 100

print(f"\n\nCOMPREHENSIVE RESULTS TABLE")
print("=" * 100)
print(header)
print(sep)

table_lines = []
table_lines.append("PHASE 3: COMPREHENSIVE MODEL COMPARISON RESULTS\n")
table_lines.append("=" * 100 + "\n")
table_lines.append(header + "\n")
table_lines.append(sep + "\n")

for fs_name in ["A", "B", "C"]:
    for target in TARGETS:
        key = f"{fs_name}_{target}"
        base_auprc = baseline_results[target]["AUPRC"]

        if fs_name == "A":
            line = (f"{'Base':<5} {target:<25} {'Hourly Rate':<12} "
                    f"{base_auprc:>8.4f}")
            print(line)
            table_lines.append(line + "\n")

        for model_name in ["LR", "LightGBM"]:
            res = all_results[key][model_name]
            pct = (res["AUPRC"] / base_auprc - 1) * 100
            line = (f"{fs_name:<5} {target:<25} {model_name:<12} "
                    f"{res['AUPRC']:>8.4f} {res['AUROC']:>8.4f} "
                    f"{res['Best_F1']:>8.4f} {pct:>+9.1f}%")
            print(line)
            table_lines.append(line + "\n")
    print()
    table_lines.append("\n")

with open(OUT_DIR / "phase3_results_table.txt", "w") as f:
    f.writelines(table_lines)
print(f"✓ Results table saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 17: Urination targets ──
print("Fig17  PR curves — urination targets …")

urine_targets = ["urination_next_15min", "urination_next_30min",
                 "urination_next_60min"]
panel_titles = ["Urination: Next 15 Minutes",
                "Urination: Next 30 Minutes",
                "Urination: Next 60 Minutes"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (target, title) in enumerate(zip(urine_targets, panel_titles)):
    ax = axes[i]

    # Baseline
    prec_b, rec_b, _ = precision_recall_curve(
        val[target].values, baseline_results[target]["val_pred"])
    ax.plot(rec_b, prec_b, linewidth=2, color="gray", linestyle="--",
            label=f"Baseline ({baseline_results[target]['AUPRC']:.3f})")

    line_styles = {"A": "-", "B": ":", "C": "-"}
    line_widths = {"A": 2.0, "B": 1.5, "C": 2.5}

    for j, fs_name in enumerate(["A", "B", "C"]):
        key = f"{fs_name}_{target}"
        res = all_results[key]["LightGBM"]
        prec, rec, _ = precision_recall_curve(val[target].values, res["val_pred"])
        ax.plot(rec, prec, linewidth=line_widths[fs_name], color=COLORS[j],
                linestyle=line_styles[fs_name],
                label=f"LightGBM Set {fs_name} ({res['AUPRC']:.3f})")

    pos_rate = val[target].mean()
    ax.axhline(y=pos_rate, color="lightgray", linestyle="-", linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    clean_axes(ax)

plt.suptitle("Precision-Recall Comparison: LightGBM Across Feature Sets (Urination)",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig17_precision_recall_all_models_urination.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")

# ── Fig 18: Bowel targets ──
print("Fig18  PR curves — bowel targets …")

bowel_targets = ["bowel_next_30min", "bowel_next_60min"]
panel_titles_b = ["Bowel Movement: Next 30 Minutes",
                  "Bowel Movement: Next 60 Minutes"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, (target, title) in enumerate(zip(bowel_targets, panel_titles_b)):
    ax = axes[i]

    prec_b, rec_b, _ = precision_recall_curve(
        val[target].values, baseline_results[target]["val_pred"])
    ax.plot(rec_b, prec_b, linewidth=2, color="gray", linestyle="--",
            label=f"Baseline ({baseline_results[target]['AUPRC']:.3f})")

    for j, fs_name in enumerate(["A", "B", "C"]):
        key = f"{fs_name}_{target}"
        res = all_results[key]["LightGBM"]
        prec, rec, _ = precision_recall_curve(val[target].values, res["val_pred"])
        ls = "-" if fs_name != "B" else ":"
        lw = 2.5 if fs_name == "C" else 2.0 if fs_name == "A" else 1.5
        ax.plot(rec, prec, linewidth=lw, color=COLORS[j], linestyle=ls,
                label=f"LightGBM Set {fs_name} ({res['AUPRC']:.3f})")

    pos_rate = val[target].mean()
    ax.axhline(y=pos_rate, color="lightgray", linestyle="-", linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    clean_axes(ax)

plt.suptitle("Precision-Recall Comparison: LightGBM Across Feature Sets (Bowel)",
             fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig18_precision_recall_all_models_bowel.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Feature Importance (LightGBM Set C, urination_next_30min)
# ══════════════════════════════════════════════════════════════════════════════

print("Fig19  Feature importance (LightGBM Set C) …")

best_key = "C_urination_next_30min"
best_model = all_results[best_key]["LightGBM"]["model"]
feat_names = all_results[best_key]["LightGBM"]["feature_names"]

importance = best_model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({"Feature": feat_names, "Importance": importance})
feat_imp = feat_imp.sort_values("Importance", ascending=True)

top20 = feat_imp.tail(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top20)), top20["Importance"].values,
        color=COLORS[0], edgecolor="white", linewidth=0.5, height=0.7)
ax.set_yticks(range(len(top20)))
clean_names = [n.replace("_", " ").title()[:40] for n in top20["Feature"].values]
ax.set_yticklabels(clean_names, fontsize=10)
ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
ax.set_title("Top 20 Features: LightGBM (Set C, Urination Next 30 Min)",
             fontsize=16, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, alpha=0.3)
ax.yaxis.grid(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig19_feature_importance_lightgbm.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("\nFEATURE IMPORTANCE RANKING (LightGBM Set C, urination_next_30min)")
print("=" * 60)
for _, row in feat_imp.sort_values("Importance", ascending=False).head(30).iterrows():
    print(f"  {row['Feature']:<50s} {row['Importance']:.0f}")
print("  ✓ saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — SHAP Analysis
# ══════════════════════════════════════════════════════════════════════════════

print("Fig20–21  SHAP analysis …")
import shap

best_model = all_results["C_urination_next_30min"]["LightGBM"]["model"]
feat_names = all_results["C_urination_next_30min"]["LightGBM"]["feature_names"]

# Sample validation data for speed
np.random.seed(42)
sample_idx = np.random.choice(len(val), size=min(5000, len(val)), replace=False)
X_sample = val[feat_names].iloc[sample_idx]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_sample)

# Clean feature names for display
clean_feature_names = [n.replace("_", " ").title()[:35] for n in feat_names]
X_display = X_sample.copy()
X_display.columns = clean_feature_names

# Fig 20: SHAP bar plot
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_display, plot_type="bar",
                  max_display=20, show=False, color=COLORS[0])
plt.title("SHAP Feature Importance: LightGBM (Set C, Urination Next 30 Min)",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig20_shap_summary_plot.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Fig20 saved.")

# Fig 21: SHAP beeswarm plot
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_display, max_display=20, show=False)
plt.title("SHAP Value Distribution: LightGBM (Set C, Urination Next 30 Min)",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig21_shap_beeswarm_plot.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ Fig21 saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Feature Set Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

print("Fig22  Feature-set comparison bar chart …")

targets_short = ["Urine 15m", "Urine 30m", "Urine 60m",
                 "Bowel 30m", "Bowel 60m"]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(TARGETS))
width = 0.18

# Baseline
baseline_vals = [baseline_results[t]["AUPRC"] for t in TARGETS]
ax.bar(x - 2 * width, baseline_vals, width,
       label="Baseline (Hourly Rate)", color="lightgray", edgecolor="white")

# LightGBM A / B / C
for i, fs_name in enumerate(["A", "B", "C"]):
    vals = [all_results[f"{fs_name}_{t}"]["LightGBM"]["AUPRC"] for t in TARGETS]
    ax.bar(x + (i - 1) * width, vals, width,
           label=f"LightGBM Set {fs_name}", color=COLORS[i], edgecolor="white")

# LR Set C
lr_vals = [all_results[f"C_{t}"]["LR"]["AUPRC"] for t in TARGETS]
ax.bar(x + 2 * width, lr_vals, width,
       label="Logistic Regression Set C", color=COLORS[3],
       edgecolor="white", alpha=0.8)

ax.set_xlabel("Prediction Target", fontsize=13)
ax.set_ylabel("AUPRC (Validation Set)", fontsize=13)
ax.set_title("Model Comparison: AUPRC Across Feature Sets and Prediction Targets",
             fontsize=16, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(targets_short, fontsize=11)
ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
clean_axes(ax)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig22_feature_set_comparison.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — Comprehensive Summary Report
# ══════════════════════════════════════════════════════════════════════════════

print("Generating Phase 3 summary report …")

report_header = (f"{'Target':<25} {'Baseline':>10} {'LR-A':>8} {'LR-B':>8} "
                 f"{'LR-C':>8} {'LGB-A':>8} {'LGB-B':>8} {'LGB-C':>8}")

report = f"""\
PHASE 3: SUPERVISED PREDICTION MODELING — SUMMARY REPORT
{'=' * 70}

MODELS TRAINED
  Logistic Regression (Elastic Net, l1_ratio=0.5, class_weight='balanced')
  LightGBM (num_leaves=63, lr=0.05, early stopping 50 rounds)
  Total configurations: 3 feature sets x 5 targets x 2 models = 30 models

FEATURE SETS
  Set A: {len(FEATURE_SET_A)} features (schedule, history, intake)
  Set B: {len(FEATURE_SET_B)} features (wearable only)
  Set C: {len(FEATURE_SET_C)} features (combined A + B)

VALIDATION SET RESULTS (AUPRC)
{'=' * 70}
{report_header}
{'-' * 70}
"""

for target in TARGETS:
    base = baseline_results[target]["AUPRC"]
    lr_a  = all_results[f"A_{target}"]["LR"]["AUPRC"]
    lr_b  = all_results[f"B_{target}"]["LR"]["AUPRC"]
    lr_c  = all_results[f"C_{target}"]["LR"]["AUPRC"]
    lgb_a = all_results[f"A_{target}"]["LightGBM"]["AUPRC"]
    lgb_b = all_results[f"B_{target}"]["LightGBM"]["AUPRC"]
    lgb_c = all_results[f"C_{target}"]["LightGBM"]["AUPRC"]
    report += (f"{target:<25} {base:>10.4f} {lr_a:>8.4f} {lr_b:>8.4f} "
               f"{lr_c:>8.4f} {lgb_a:>8.4f} {lgb_b:>8.4f} {lgb_c:>8.4f}\n")

pct_header = (f"{'Target':<25} {'LR-A':>8} {'LR-B':>8} {'LR-C':>8} "
              f"{'LGB-A':>8} {'LGB-B':>8} {'LGB-C':>8}")

report += f"""
RELATIVE IMPROVEMENT OVER BASELINE (% AUPRC GAIN)
{'=' * 70}
{pct_header}
{'-' * 70}
"""

for target in TARGETS:
    base = baseline_results[target]["AUPRC"]
    vals = []
    for fs in ["A", "B", "C"]:
        for model in ["LR", "LightGBM"]:
            pct = (all_results[f"{fs}_{target}"][model]["AUPRC"] / base - 1) * 100
            vals.append(pct)
    report += (f"{target:<25} {vals[0]:>+7.1f}% {vals[2]:>+7.1f}% "
               f"{vals[4]:>+7.1f}% {vals[1]:>+7.1f}% {vals[3]:>+7.1f}% "
               f"{vals[5]:>+7.1f}%\n")

report += """
FIGURES GENERATED
  Fig17: PR curves - LightGBM across feature sets (urination targets)
  Fig18: PR curves - LightGBM across feature sets (bowel targets)
  Fig19: Feature importance - LightGBM Set C (urination 30min)
  Fig20: SHAP bar plot - LightGBM Set C (urination 30min)
  Fig21: SHAP beeswarm - LightGBM Set C (urination 30min)
  Fig22: Feature set comparison bar chart (AUPRC all targets)
"""

report_path = OUT_DIR / "phase3_summary_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(report)
print(f"  ✓ report saved to {report_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Final checklist
# ══════════════════════════════════════════════════════════════════════════════
expected = [
    "phase3_results_table.txt",
    "Fig17_precision_recall_all_models_urination.png",
    "Fig18_precision_recall_all_models_bowel.png",
    "Fig19_feature_importance_lightgbm.png",
    "Fig20_shap_summary_plot.png",
    "Fig21_shap_beeswarm_plot.png",
    "Fig22_feature_set_comparison.png",
    "phase3_summary_report.txt",
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
    print("All 8 outputs generated successfully.")
else:
    print("WARNING: some outputs are missing — check errors above.")
print("\nDone.")
