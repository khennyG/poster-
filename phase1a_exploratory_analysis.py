#!/usr/bin/env python3
"""
Phase 1A — Exploratory Temporal Analysis: Event Patterns and Circadian Rhythms

Capstone research project analyzing a longitudinal toileting dataset for an
individual with severe autism. This script characterizes temporal patterns in
toileting events before any predictive modelling begins.

Outputs (all written to phase1_exploratory/):
    Fig01_urination_hourly_distribution.png
    Fig02_bowel_hourly_distribution.png
    Fig03_urination_intervoid_intervals.png
    Fig04_bowel_interevent_intervals.png
    Fig05_weekday_vs_weekend_urination.png
    Fig06_urination_circadian_heatmap.png
    Fig07_daily_event_counts_trend.png
    phase1a_summary_statistics.txt
"""

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ──────────────────────────────────────────────
# Global style configuration
# ──────────────────────────────────────────────
sns.set_style("whitegrid")
COLORS = ["#2C6E91", "#E07B54", "#4CA77B", "#8B6CAE", "#D4A84B"]
plt.rcParams["font.family"] = "DejaVu Sans"

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "autism.csv"
OUT_DIR = BASE_DIR / "phase1_exploratory"
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
# Analysis 1: Hourly Distribution of Urination Events
# ══════════════════════════════════════════════
print("Fig01  Urination hourly distribution …")
urine_events = df[df["urination_event"] == 1]

fig, ax = plt.subplots(figsize=(10, 6))
counts = urine_events.groupby("hour").size()
ax.bar(counts.index, counts.values, color=COLORS[0],
       edgecolor="white", linewidth=0.5, width=0.8)
ax.set_xlabel("Hour of Day", fontsize=13)
ax.set_ylabel("Total Urination Events", fontsize=13)
ax.set_title("Distribution of Urination Events by Hour of Day",
             fontsize=16, fontweight="bold")
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   rotation=45, ha="right", fontsize=9)
ax.tick_params(axis="y", labelsize=11)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig01_urination_hourly_distribution.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 2: Hourly Distribution of Bowel Events
# ══════════════════════════════════════════════
print("Fig02  Bowel hourly distribution …")
bowel_events = df[df["bowel_event"] == 1]

fig, ax = plt.subplots(figsize=(10, 6))
counts_b = bowel_events.groupby("hour").size()
ax.bar(counts_b.index, counts_b.values, color=COLORS[1],
       edgecolor="white", linewidth=0.5, width=0.8)
ax.set_xlabel("Hour of Day", fontsize=13)
ax.set_ylabel("Total Bowel Movement Events", fontsize=13)
ax.set_title("Distribution of Bowel Movement Events by Hour of Day",
             fontsize=16, fontweight="bold")
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   rotation=45, ha="right", fontsize=9)
ax.tick_params(axis="y", labelsize=11)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig02_bowel_hourly_distribution.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 3: Inter-Void Interval Distribution (Urination)
# ══════════════════════════════════════════════
print("Fig03  Urination inter-void intervals …")
urine_intervals = df.loc[df["urination_event"] == 1,
                         "minutes_since_last_urination"].dropna()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(urine_intervals, bins=60, color=COLORS[0],
        edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(urine_intervals.median(), color=COLORS[1], linestyle="--",
           linewidth=2,
           label=f"Median: {urine_intervals.median():.0f} min")
ax.axvline(urine_intervals.mean(), color=COLORS[3], linestyle="--",
           linewidth=2,
           label=f"Mean: {urine_intervals.mean():.0f} min")
ax.set_xlabel("Minutes Since Previous Urination", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
ax.set_title("Distribution of Inter-Void Intervals (Urination)",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig03_urination_intervoid_intervals.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 4: Inter-Event Interval Distribution (Bowel)
# ══════════════════════════════════════════════
print("Fig04  Bowel inter-event intervals …")
bowel_intervals = df.loc[df["bowel_event"] == 1,
                         "minutes_since_last_bowel"].dropna()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bowel_intervals, bins=60, color=COLORS[1],
        edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(bowel_intervals.median(), color=COLORS[0], linestyle="--",
           linewidth=2,
           label=f"Median: {bowel_intervals.median():.0f} min")
ax.axvline(bowel_intervals.mean(), color=COLORS[3], linestyle="--",
           linewidth=2,
           label=f"Mean: {bowel_intervals.mean():.0f} min")
ax.set_xlabel("Minutes Since Previous Bowel Movement", fontsize=13)
ax.set_ylabel("Frequency", fontsize=13)
ax.set_title("Distribution of Inter-Event Intervals (Bowel Movements)",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig04_bowel_interevent_intervals.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 5: Weekday vs Weekend Urination Pattern
# ══════════════════════════════════════════════
print("Fig05  Weekday vs weekend urination …")
urine_ev = df[df["urination_event"] == 1].copy()
urine_ev["period"] = urine_ev["is_weekend"].map({0: "Weekday", 1: "Weekend"})

fig, ax = plt.subplots(figsize=(10, 6))
for i, (label, group) in enumerate(urine_ev.groupby("period")):
    hourly = group.groupby("hour").size()
    n_days = df[df["is_weekend"] == (1 if label == "Weekend" else 0)]["date"].nunique()
    avg_per_day = hourly / n_days
    ax.plot(avg_per_day.index, avg_per_day.values,
            marker="o", markersize=5, linewidth=2.5,
            color=COLORS[i], label=label)

ax.set_xlabel("Hour of Day", fontsize=13)
ax.set_ylabel("Average Urination Events per Day", fontsize=13)
ax.set_title("Urination Pattern: Weekday vs Weekend",
             fontsize=16, fontweight="bold")
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   rotation=45, ha="right", fontsize=9)
ax.legend(fontsize=12, framealpha=0.9)
clean_axes(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig05_weekday_vs_weekend_urination.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 6: Circadian Heatmap (Hour × Day of Week)
# ══════════════════════════════════════════════
print("Fig06  Urination circadian heatmap …")
urine_ev2 = df[df["urination_event"] == 1].copy()
heatmap_data = (urine_ev2
                .groupby(["day_of_week", "hour"])
                .size()
                .unstack(fill_value=0))

# Ensure all 24 hours present
for h in range(24):
    if h not in heatmap_data.columns:
        heatmap_data[h] = 0
heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

# Normalize by number of occurrences of each weekday in the dataset
day_counts = df.groupby("day_of_week")["date"].nunique()
heatmap_normalized = heatmap_data.div(day_counts, axis=0)

day_labels = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_normalized, cmap="YlOrRd", annot=False,
            linewidths=0.5,
            xticklabels=[f"{h:02d}:00" for h in range(24)],
            yticklabels=day_labels, ax=ax,
            cbar_kws={"label": "Avg Events per Day"})
ax.set_xlabel("Hour of Day", fontsize=13)
ax.set_ylabel("Day of Week", fontsize=13)
ax.set_title(
    "Circadian Heatmap: Average Urination Events by Hour and Day of Week",
    fontsize=16, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig06_urination_circadian_heatmap.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 7: Daily Event Counts Over Time (Trend)
# ══════════════════════════════════════════════
print("Fig07  Daily event counts trend …")
daily = df.groupby("date").agg(
    urinations=("urination_event", "sum"),
    bowels=("bowel_event", "sum"),
).reset_index()
daily["date"] = pd.to_datetime(daily["date"])

daily["urine_rolling"] = daily["urinations"].rolling(30, center=True).mean()
daily["bowel_rolling"] = daily["bowels"].rolling(30, center=True).mean()

year_min = daily["date"].dt.year.min()
year_max = daily["date"].dt.year.max()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Urination panel
ax1.scatter(daily["date"], daily["urinations"],
            alpha=0.15, s=8, color=COLORS[0], label="Daily Count")
ax1.plot(daily["date"], daily["urine_rolling"],
         color=COLORS[0], linewidth=2.5, label="30-Day Rolling Average")
ax1.set_ylabel("Urination Events per Day", fontsize=13)
ax1.set_title(
    f"Daily Urination Events Over Time ({year_min} to {year_max})",
    fontsize=16, fontweight="bold")
ax1.legend(fontsize=11, framealpha=0.9)
clean_axes(ax1)

# Bowel panel
ax2.scatter(daily["date"], daily["bowels"],
            alpha=0.15, s=8, color=COLORS[1], label="Daily Count")
ax2.plot(daily["date"], daily["bowel_rolling"],
         color=COLORS[1], linewidth=2.5, label="30-Day Rolling Average")
ax2.set_ylabel("Bowel Events per Day", fontsize=13)
ax2.set_xlabel("Date", fontsize=13)
ax2.set_title(
    f"Daily Bowel Movement Events Over Time ({year_min} to {year_max})",
    fontsize=16, fontweight="bold")
ax2.legend(fontsize=11, framealpha=0.9)
clean_axes(ax2)

plt.tight_layout()
plt.savefig(OUT_DIR / "Fig07_daily_event_counts_trend.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("  ✓ saved.\n")


# ══════════════════════════════════════════════
# Analysis 8: Summary Statistics Text Report
# ══════════════════════════════════════════════
print("Generating summary statistics report …")

urine_events = df[df["urination_event"] == 1]
bowel_events = df[df["bowel_event"] == 1]

daily_stats = df.groupby("date").agg(
    urinations=("urination_event", "sum"),
    bowels=("bowel_event", "sum"),
).reset_index()

urine_intervals = df.loc[
    df["urination_event"] == 1, "minutes_since_last_urination"
].dropna()
bowel_intervals = df.loc[
    df["bowel_event"] == 1, "minutes_since_last_bowel"
].dropna()

urine_peak_hour = urine_events.groupby("hour").size().idxmax()
bowel_peak_hour = bowel_events.groupby("hour").size().idxmax()

urine_awake_pct = (urine_events["asleep_flag"] == 0).mean() * 100
bowel_awake_pct = (bowel_events["asleep_flag"] == 0).mean() * 100

daily_we = daily_stats.copy()
daily_we["date"] = pd.to_datetime(daily_we["date"])
daily_we["is_weekend"] = daily_we["date"].dt.dayofweek.isin([5, 6]).astype(int)
weekday_avg = daily_we.loc[daily_we["is_weekend"] == 0, "urinations"].mean()
weekend_avg = daily_we.loc[daily_we["is_weekend"] == 1, "urinations"].mean()

report = f"""\
PHASE 1A: SUMMARY STATISTICS
{'=' * 50}

EVENT TOTALS
  Total urination events: {urine_events.shape[0]:,}
  Total bowel events:     {bowel_events.shape[0]:,}
  Total days in dataset:  {daily_stats.shape[0]:,}

DAILY EVENT RATES
  Urinations per day:      {daily_stats['urinations'].mean():.2f}  (SD: {daily_stats['urinations'].std():.2f})
  Bowel movements per day: {daily_stats['bowels'].mean():.2f}  (SD: {daily_stats['bowels'].std():.2f})

INTER-EVENT INTERVALS (URINATION)
  Median: {urine_intervals.median():.0f} minutes
  IQR:    {urine_intervals.quantile(0.25):.0f} to {urine_intervals.quantile(0.75):.0f} minutes
  Mean:   {urine_intervals.mean():.0f} minutes

INTER-EVENT INTERVALS (BOWEL)
  Median: {bowel_intervals.median():.0f} minutes
  IQR:    {bowel_intervals.quantile(0.25):.0f} to {bowel_intervals.quantile(0.75):.0f} minutes
  Mean:   {bowel_intervals.mean():.0f} minutes

PEAK HOURS
  Peak urination hour:      {urine_peak_hour:02d}:00
  Peak bowel movement hour: {bowel_peak_hour:02d}:00

SLEEP vs WAKE
  Urination events during waking hours: {urine_awake_pct:.1f}%
  Bowel events during waking hours:     {bowel_awake_pct:.1f}%

WEEKDAY vs WEEKEND (URINATION)
  Weekday average: {weekday_avg:.2f} per day
  Weekend average: {weekend_avg:.2f} per day
"""

report_path = OUT_DIR / "phase1a_summary_statistics.txt"
with open(report_path, "w") as f:
    f.write(report)
print(report)
print(f"  ✓ report saved to {report_path}\n")


# ══════════════════════════════════════════════
# Final checklist
# ══════════════════════════════════════════════
expected_files = [
    "Fig01_urination_hourly_distribution.png",
    "Fig02_bowel_hourly_distribution.png",
    "Fig03_urination_intervoid_intervals.png",
    "Fig04_bowel_interevent_intervals.png",
    "Fig05_weekday_vs_weekend_urination.png",
    "Fig06_urination_circadian_heatmap.png",
    "Fig07_daily_event_counts_trend.png",
    "phase1a_summary_statistics.txt",
]

print("=" * 50)
print("OUTPUT CHECKLIST")
print("=" * 50)
all_ok = True
for fname in expected_files:
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
