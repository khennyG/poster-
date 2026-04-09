#!/usr/bin/env python3
"""
Phase 1B — Exploratory Signal Analysis: Intake, Physiology, and Pre-Event Patterns

Capstone research project analyzing a longitudinal toileting dataset for an
individual with severe autism. Phase 1A established strongly circadian, stable
toileting patterns.  Phase 1B tests whether input signals (fluid intake, heart
rate, HRV, activity) show detectable relationships with toileting events before
any predictive modelling begins.

Outputs (all written to phase1b_signal_analysis/):
    Fig08_intake_urination_lag_response.png
    Fig09_time_to_void_after_drinking.png
    Fig10_pre_event_hr_hrv_trajectories.png
    Fig11_activity_before_events.png
    Fig12_cumulative_intake_vs_event_probability.png
    Fig13_bladder_pressure_pre_event.png
    phase1b_summary_statistics.txt
"""

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ──────────────────────────────────────────────
# Global style configuration
# ──────────────────────────────────────────────
sns.set_style("whitegrid")
COLORS = ["#2C6E91", "#E07B54", "#4CA77B", "#8B6CAE", "#D4A84B"]
plt.rcParams["font.family"] = "DejaVu Sans"

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "autism.csv"
OUT_DIR = BASE_DIR / "phase1b_signal_analysis"
OUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Helper: remove top/right spines, set y-grid
# ──────────────────────────────────────────────
def clean_axes(ax, ygrid=True):
    """Apply standard spine and grid cleanup to an axes object."""
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


# ══════════════════════════════════════════════
# Analysis 8: Fluid Intake to Urination Lag-Response (Cross-Correlation)
# ══════════════════════════════════════════════
print("Fig08  Intake-urination lag-response cross-correlation …")

# Only use waking hours to avoid diluting the signal with sleep
awake = df[df["asleep_flag"] == 0].copy()

intake_series = awake["water_intake_event_flag"].values
urine_series = awake["urination_event"].values

# Cross-correlation at lags 0–180 min (10-min steps)
max_lag = 18  # 180 minutes
lags = np.arange(0, max_lag + 1)
cross_corr = []

for lag in lags:
    if lag == 0:
        corr = np.corrcoef(intake_series, urine_series)[0, 1]
    else:
        corr = np.corrcoef(intake_series[:-lag], urine_series[lag:])[0, 1]
    cross_corr.append(corr)

cross_corr = np.array(cross_corr)
lag_minutes = lags * 10

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(lag_minutes, cross_corr, width=8, color=COLORS[0],
       edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")

# Approximate 95 % CI for null correlation
n = len(intake_series)
ci = 1.96 / np.sqrt(n)
ax.axhline(y=ci, color=COLORS[1], linewidth=1.5, linestyle="--",
           alpha=0.7, label="95% Confidence Bound")
ax.axhline(y=-ci, color=COLORS[1], linewidth=1.5, linestyle="--", alpha=0.7)

# Mark the peak lag
peak_idx = np.argmax(cross_corr)
ax.annotate(
    f"Peak: {lag_minutes[peak_idx]} min\n(r = {cross_corr[peak_idx]:.4f})",
    xy=(lag_minutes[peak_idx], cross_corr[peak_idx]),
    xytext=(lag_minutes[peak_idx] + 25,
            cross_corr[peak_idx] + max(0.003, cross_corr.max() * 0.15)),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="gray", alpha=0.9),
)

ax.set_xlabel("Lag After Fluid Intake (Minutes)", fontsize=13)
ax.set_ylabel("Cross-Correlation with Urination Event", fontsize=13)
ax.set_title("Lag-Response: Fluid Intake to Urination Cross-Correlation",
             fontsize=16, fontweight="bold")
ax.set_xticks(lag_minutes)
ax.set_xticklabels([str(m) for m in lag_minutes], fontsize=9)
ax.legend(fontsize=11, framealpha=0.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig08_intake_urination_lag_response.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 9: Time-to-Next-Void After Drinking (Survival-Style Curve)
# ══════════════════════════════════════════════
print("Fig09  Time-to-void after drinking …")

awake = df[df["asleep_flag"] == 0].copy().reset_index(drop=True)

# --- helper: compute time-to-void for a set of indices ---
def _time_to_next_void(indices, event_col="urination_event", cap=300):
    """Return array of minutes-to-next-void (capped at `cap` minutes)."""
    results = []
    for idx in indices:
        future = awake.loc[idx:, event_col]
        future_events = future[future == 1]
        if len(future_events) > 0:
            delta = (future_events.index[0] - idx) * 10
            if delta <= cap:
                results.append(delta)
    return np.array(results)

# After fluid-intake events
intake_idx = awake.index[awake["water_intake_event_flag"] == 1].tolist()
time_to_void = _time_to_next_void(intake_idx)

# Control: random non-intake waking windows
np.random.seed(42)
no_intake_idx = awake.index[awake["water_intake_event_flag"] == 0].tolist()
control_sample = np.random.choice(no_intake_idx,
                                  size=min(len(intake_idx), len(no_intake_idx)),
                                  replace=False).tolist()
time_to_void_control = _time_to_next_void(control_sample)

# Cumulative event curves
time_points = np.arange(0, 310, 10)
cum_intake = [np.mean(time_to_void <= t) for t in time_points]
cum_control = [np.mean(time_to_void_control <= t) for t in time_points]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_points, cum_intake, linewidth=2.5, color=COLORS[0],
        marker="o", markersize=4, label="After Fluid Intake")
ax.plot(time_points, cum_control, linewidth=2.5, color=COLORS[1],
        marker="s", markersize=4, label="Control (No Intake)")
ax.fill_between(time_points, cum_intake, cum_control, alpha=0.1, color=COLORS[0])

ax.set_xlabel("Minutes After Reference Window", fontsize=13)
ax.set_ylabel("Cumulative Proportion with Urination Event", fontsize=13)
ax.set_title("Time to Next Void: After Fluid Intake vs Control Windows",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=12, framealpha=0.9, loc="lower right")
ax.set_xlim(0, 300)
ax.set_ylim(0, 1.0)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig09_time_to_void_after_drinking.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 10: Pre-Event Heart Rate & HRV Trajectories
# ══════════════════════════════════════════════
print("Fig10  Pre-event HR and HRV trajectories …")

awake = df[df["asleep_flag"] == 0].copy().reset_index(drop=True)
lookback = 12  # 12 windows = 120 minutes

# --- collect trajectories for a given event column ---
def _collect_trajectories(event_col, feature_cols, lookback=12):
    """Return dict of feature -> (n_events, lookback) arrays."""
    indices = awake.index[awake[event_col] == 1].tolist()
    trajs = {f: [] for f in feature_cols}
    for idx in indices:
        if idx >= lookback:
            window = awake.loc[idx - lookback: idx - 1]
            if len(window) == lookback and window["asleep_flag"].sum() == 0:
                for f in feature_cols:
                    trajs[f].append(window[f].values)
    return {f: np.array(v) for f, v in trajs.items()}

# Pre-urination trajectories
event_trajs = _collect_trajectories(
    "urination_event", ["heart_rate", "hrv"])

# Control trajectories (no event in next 60 min)
non_event = awake[(awake["urination_event"] == 0) &
                  (awake["urination_next_60min"] == 0)]
np.random.seed(42)
ctrl_indices = np.random.choice(
    non_event.index.tolist(),
    size=min(len(event_trajs["heart_rate"]), len(non_event)),
    replace=False,
)

ctrl_trajs = {"heart_rate": [], "hrv": []}
for idx in ctrl_indices:
    if idx >= lookback:
        window = awake.loc[idx - lookback: idx - 1]
        if len(window) == lookback and window["asleep_flag"].sum() == 0:
            ctrl_trajs["heart_rate"].append(window["heart_rate"].values)
            ctrl_trajs["hrv"].append(window["hrv"].values)
ctrl_trajs = {f: np.array(v) for f, v in ctrl_trajs.items()}

time_axis = np.arange(-120, 0, 10)  # -120 to -10 min before event

# Means and 95 % CI
def _mean_ci(arr):
    m = arr.mean(axis=0)
    ci = 1.96 * arr.std(axis=0) / np.sqrt(len(arr))
    return m, ci

hr_ev_m, hr_ev_ci = _mean_ci(event_trajs["heart_rate"])
hr_ct_m, hr_ct_ci = _mean_ci(ctrl_trajs["heart_rate"])
hrv_ev_m, hrv_ev_ci = _mean_ci(event_trajs["hrv"])
hrv_ct_m, hrv_ct_ci = _mean_ci(ctrl_trajs["hrv"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Heart-rate panel
ax1.plot(time_axis, hr_ev_m, linewidth=2.5, color=COLORS[0], label="Pre-Urination")
ax1.fill_between(time_axis, hr_ev_m - hr_ev_ci, hr_ev_m + hr_ev_ci,
                 alpha=0.2, color=COLORS[0])
ax1.plot(time_axis, hr_ct_m, linewidth=2.5, color=COLORS[1], label="Control (No Event)")
ax1.fill_between(time_axis, hr_ct_m - hr_ct_ci, hr_ct_m + hr_ct_ci,
                 alpha=0.2, color=COLORS[1])
ax1.set_xlabel("Minutes Before Reference Window", fontsize=13)
ax1.set_ylabel("Heart Rate (bpm)", fontsize=13)
ax1.set_title("Heart Rate Trajectory Before Urination",
              fontsize=14, fontweight="bold")
ax1.legend(fontsize=11, framealpha=0.9)
clean_axes(ax1)

# HRV panel
ax2.plot(time_axis, hrv_ev_m, linewidth=2.5, color=COLORS[0], label="Pre-Urination")
ax2.fill_between(time_axis, hrv_ev_m - hrv_ev_ci, hrv_ev_m + hrv_ev_ci,
                 alpha=0.2, color=COLORS[0])
ax2.plot(time_axis, hrv_ct_m, linewidth=2.5, color=COLORS[1], label="Control (No Event)")
ax2.fill_between(time_axis, hrv_ct_m - hrv_ct_ci, hrv_ct_m + hrv_ct_ci,
                 alpha=0.2, color=COLORS[1])
ax2.set_xlabel("Minutes Before Reference Window", fontsize=13)
ax2.set_ylabel("Heart Rate Variability (ms)", fontsize=13)
ax2.set_title("HRV Trajectory Before Urination",
              fontsize=14, fontweight="bold")
ax2.legend(fontsize=11, framealpha=0.9)
clean_axes(ax2)

plt.tight_layout()
plt.savefig(OUT_DIR / "Fig10_pre_event_hr_hrv_trajectories.png",
            dpi=300, bbox_inches="tight")
plt.close()

# Summary printout
print(f"  Pre-event trajectories collected: {len(event_trajs['heart_rate'])}")
print(f"  Control trajectories collected:   {len(ctrl_trajs['heart_rate'])}")
print(f"\n  Heart Rate (last 30 min before event):")
print(f"    Event mean:   {hr_ev_m[-3:].mean():.2f} bpm")
print(f"    Control mean: {hr_ct_m[-3:].mean():.2f} bpm")
print(f"    Difference:   {hr_ev_m[-3:].mean() - hr_ct_m[-3:].mean():.2f} bpm")
print(f"\n  HRV (last 30 min before event):")
print(f"    Event mean:   {hrv_ev_m[-3:].mean():.2f} ms")
print(f"    Control mean: {hrv_ct_m[-3:].mean():.2f} ms")
print(f"    Difference:   {hrv_ev_m[-3:].mean() - hrv_ct_m[-3:].mean():.2f} ms")
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 11: Activity Level Distribution Before Events vs Non-Events
# ══════════════════════════════════════════════
print("Fig11  Activity level before events …")

awake = df[df["asleep_flag"] == 0].copy().reset_index(drop=True)

pre_event_activity = awake.loc[
    awake["urination_event"] == 1, "recent_activity_30min_mean"
].dropna()

non_event_activity = awake.loc[
    (awake["urination_event"] == 0) & (awake["urination_next_60min"] == 0),
    "recent_activity_30min_mean",
].dropna()

fig, ax = plt.subplots(figsize=(10, 6))
upper = max(pre_event_activity.max(), non_event_activity.max())
bins = np.linspace(0, min(upper, 3.5), 40)
ax.hist(pre_event_activity, bins=bins, density=True, alpha=0.65,
        color=COLORS[0], edgecolor="white", linewidth=0.5,
        label=f"Before Urination (n={len(pre_event_activity):,})")
ax.hist(non_event_activity, bins=bins, density=True, alpha=0.55,
        color=COLORS[1], edgecolor="white", linewidth=0.5,
        label=f"Non-Event Windows (n={len(non_event_activity):,})")
ax.axvline(pre_event_activity.mean(), color=COLORS[0],
           linestyle="--", linewidth=2, alpha=0.8)
ax.axvline(non_event_activity.mean(), color=COLORS[1],
           linestyle="--", linewidth=2, alpha=0.8)

ax.set_xlabel("30-Minute Mean Activity Score", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Activity Level Before Urination Events vs Non-Event Windows",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig11_activity_before_events.png",
            dpi=300, bbox_inches="tight")
plt.close()

# Statistics
u_stat, p_val = stats.mannwhitneyu(pre_event_activity, non_event_activity,
                                   alternative="two-sided")
print(f"  Activity before urination events: mean = {pre_event_activity.mean():.3f}, "
      f"median = {pre_event_activity.median():.3f}")
print(f"  Activity in non-event windows:    mean = {non_event_activity.mean():.3f}, "
      f"median = {non_event_activity.median():.3f}")
print(f"  Difference: {pre_event_activity.mean() - non_event_activity.mean():.3f}")
print(f"  Mann-Whitney U test: U = {u_stat:.0f}, p = {p_val:.2e}")
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 12: Cumulative Fluid Intake Since Last Void vs Event Probability
# ══════════════════════════════════════════════
print("Fig12  Cumulative intake vs event probability …")

# NOTE: cumulative_water_since_last_void_ml resets to 0 at the moment of an
# actual urination event, so comparing it against urination_event in the *same*
# window is meaningless.  Instead we ask: "given the current cumulative intake,
# how likely is a void in the NEXT 30 minutes?" — using the forward-looking
# label urination_next_30min.
awake = df[df["asleep_flag"] == 0].copy()

bins_intake = [0, 50, 100, 200, 300, 400, 600, 800, 1000, 1800]
labels_intake = ["0-50", "50-100", "100-200", "200-300",
                 "300-400", "400-600", "600-800", "800-1000", "1000+"]
awake["intake_bin"] = pd.cut(awake["cumulative_water_since_last_void_ml"],
                             bins=bins_intake, labels=labels_intake, right=True)

event_rates = (awake.groupby("intake_bin", observed=True)
               .agg(event_rate=("urination_next_30min", "mean"),
                    count=("urination_next_30min", "size"))
               .reset_index())

fig, ax1 = plt.subplots(figsize=(11, 6))

bars = ax1.bar(range(len(event_rates)), event_rates["event_rate"] * 100,
               color=COLORS[0], edgecolor="white", linewidth=0.5,
               alpha=0.85, width=0.7)
ax1.set_xlabel("Cumulative Fluid Intake Since Last Void (mL)", fontsize=13)
ax1.set_ylabel("Probability of Urination in Next 30 min (%)",
               fontsize=13, color=COLORS[0])
ax1.set_title(
    "Urination Probability (Next 30 min) by Cumulative Fluid Intake Since Last Void",
    fontsize=15, fontweight="bold")
ax1.set_xticks(range(len(event_rates)))
ax1.set_xticklabels(labels_intake, rotation=30, ha="right", fontsize=10)
ax1.tick_params(axis="y", labelcolor=COLORS[0])

# Secondary axis: sample size
ax2 = ax1.twinx()
ax2.plot(range(len(event_rates)), event_rates["count"],
         color=COLORS[3], marker="D", linewidth=2, markersize=6,
         label="Windows in Bin")
ax2.set_ylabel("Number of Windows in Bin", fontsize=13, color=COLORS[3])
ax2.tick_params(axis="y", labelcolor=COLORS[3])

# Rate labels on bars
y_max = (event_rates["event_rate"] * 100).max()
for i, (rate, count) in enumerate(
        zip(event_rates["event_rate"], event_rates["count"])):
    ax1.text(i, rate * 100 + y_max * 0.02, f"{rate*100:.1f}%",
             ha="center", fontsize=9, fontweight="bold", color=COLORS[0])

ax1.spines["top"].set_visible(False)
ax1.yaxis.grid(True, alpha=0.3)
ax1.xaxis.grid(False)
ax2.spines["top"].set_visible(False)
fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92),
           fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig12_cumulative_intake_vs_event_probability.png",
            dpi=300, bbox_inches="tight")
plt.close()

print("  Urination-next-30min rate by cumulative intake bin:")
for _, row in event_rates.iterrows():
    print(f"    {row['intake_bin']}: {row['event_rate']*100:.2f}%  (n={row['count']:,})")
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 13: Bladder Pressure Proxy Trajectory (Validation Only)
# ══════════════════════════════════════════════
print("Fig13  Bladder pressure proxy trajectory (validation) …")

awake = df[df["asleep_flag"] == 0].copy().reset_index(drop=True)
urine_indices = awake.index[awake["urination_event"] == 1].tolist()
lookback = 12  # 120 min

bp_trajectories = []
for idx in urine_indices:
    if idx >= lookback:
        window = awake.loc[idx - lookback: idx, "bladder_pressure_proxy"].values
        if len(window) == lookback + 1:
            bp_trajectories.append(window)

bp_trajectories = np.array(bp_trajectories)
time_axis_bp = np.arange(-120, 10, 10)  # -120 … 0 (0 = event window)

bp_mean = bp_trajectories.mean(axis=0)
bp_ci = 1.96 * bp_trajectories.std(axis=0) / np.sqrt(len(bp_trajectories))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_axis_bp, bp_mean, linewidth=2.5, color=COLORS[3])
ax.fill_between(time_axis_bp, bp_mean - bp_ci, bp_mean + bp_ci,
                alpha=0.2, color=COLORS[3])
ax.axvline(x=0, color="gray", linestyle=":", linewidth=1.5,
           label="Event Window")
ax.set_xlabel("Minutes Relative to Urination Event", fontsize=13)
ax.set_ylabel("Bladder Pressure Proxy (Latent Variable)", fontsize=13)
ax.set_title("Bladder Pressure Proxy Trajectory Before Urination (Validation Only)",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig13_bladder_pressure_pre_event.png",
            dpi=300, bbox_inches="tight")
plt.close()

print(f"  Bladder pressure proxy at event window:   {bp_mean[-1]:.2f}")
print(f"  Bladder pressure proxy 120 min before:    {bp_mean[0]:.2f}")
print(f"  Rise over 120 min:                        {bp_mean[-1] - bp_mean[0]:.2f}")
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 14: Comprehensive Phase 1B Summary Statistics
# ══════════════════════════════════════════════
print("Generating Phase 1B summary statistics report …")

awake = df[df["asleep_flag"] == 0].copy()

# Intake → urination relationship
intake_events_df = awake[awake["water_intake_event_flag"] == 1]
no_intake_df = awake[awake["water_intake_event_flag"] == 0]
urine_rate_after_intake_60 = intake_events_df["urination_next_60min"].mean()
urine_rate_no_intake_60 = no_intake_df["urination_next_60min"].mean()

# Activity comparison
pre_ev_act = awake.loc[awake["urination_event"] == 1,
                       "recent_activity_30min_mean"].dropna()
non_ev_act = awake.loc[
    (awake["urination_event"] == 0) & (awake["urination_next_60min"] == 0),
    "recent_activity_30min_mean",
].dropna()

# HR / HRV at event vs non-event
event_hr = awake.loc[awake["urination_event"] == 1, "heart_rate"]
nonevent_hr = awake.loc[
    (awake["urination_event"] == 0) & (awake["urination_next_60min"] == 0),
    "heart_rate",
]
event_hrv = awake.loc[awake["urination_event"] == 1, "hrv"]
nonevent_hrv = awake.loc[
    (awake["urination_event"] == 0) & (awake["urination_next_60min"] == 0),
    "hrv",
]

# Cumulative intake at event vs non-event
event_cum = awake.loc[awake["urination_event"] == 1,
                      "cumulative_water_since_last_void_ml"]
nonevent_cum = awake.loc[
    (awake["urination_event"] == 0) & (awake["urination_next_60min"] == 0),
    "cumulative_water_since_last_void_ml",
]

# Safely compute relative increase (avoid division by zero)
if urine_rate_no_intake_60 > 0:
    rel_increase = (urine_rate_after_intake_60 / urine_rate_no_intake_60 - 1) * 100
else:
    rel_increase = float("nan")

report = f"""\
PHASE 1B: SIGNAL ANALYSIS SUMMARY STATISTICS
{'=' * 60}

FLUID INTAKE TO URINATION RELATIONSHIP
  Urination rate within 60 min after fluid intake: {urine_rate_after_intake_60*100:.2f}%
  Urination rate within 60 min (no recent intake): {urine_rate_no_intake_60*100:.2f}%
  Relative increase: {rel_increase:.1f}%

ACTIVITY LEVEL COMPARISON
  Mean 30-min activity before urination events: {pre_ev_act.mean():.3f}
  Mean 30-min activity in non-event windows:    {non_ev_act.mean():.3f}
  Difference: {pre_ev_act.mean() - non_ev_act.mean():.3f}

HEART RATE AT EVENT vs NON-EVENT WINDOWS (Waking Hours)
  Event windows mean HR:     {event_hr.mean():.2f} bpm
  Non-event windows mean HR: {nonevent_hr.mean():.2f} bpm
  Difference:                {event_hr.mean() - nonevent_hr.mean():.2f} bpm

HRV AT EVENT vs NON-EVENT WINDOWS (Waking Hours)
  Event windows mean HRV:     {event_hrv.mean():.2f} ms
  Non-event windows mean HRV: {nonevent_hrv.mean():.2f} ms
  Difference:                 {event_hrv.mean() - nonevent_hrv.mean():.2f} ms

CUMULATIVE FLUID SINCE LAST VOID
  At urination events (mean):   {event_cum.mean():.1f} mL
  At urination events (median): {event_cum.median():.1f} mL
  Non-event windows (mean):     {nonevent_cum.mean():.1f} mL
  Non-event windows (median):   {nonevent_cum.median():.1f} mL

TOTAL FLUID INTAKE EVENTS IN DATASET
  Total intake events (waking):      {intake_events_df.shape[0]:,}
  Total non-intake windows (waking): {no_intake_df.shape[0]:,}

NOTE: bladder_pressure_proxy analysis (Fig13) is for internal
validation only and must NOT be used as a model feature.
"""

report_path = OUT_DIR / "phase1b_summary_statistics.txt"
with open(report_path, "w") as f:
    f.write(report)
print(report)
print(f"  ✓ report saved to {report_path}\n")


# ══════════════════════════════════════════════
# Final checklist
# ══════════════════════════════════════════════
expected = [
    "Fig08_intake_urination_lag_response.png",
    "Fig09_time_to_void_after_drinking.png",
    "Fig10_pre_event_hr_hrv_trajectories.png",
    "Fig11_activity_before_events.png",
    "Fig12_cumulative_intake_vs_event_probability.png",
    "Fig13_bladder_pressure_pre_event.png",
    "phase1b_summary_statistics.txt",
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
    print("All 7 outputs generated successfully.")
else:
    print("WARNING: some outputs are missing — check errors above.")
print("\nDone.")
