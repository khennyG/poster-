const PptxGenJS = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

/* ═══════════════════════════════════════════════════════════════════
 *  Capstone Poster — 16 : 9 Widescreen PPTX
 *  Pixel-faithful replica of capstone_poster_16x9.html
 *  Kehinde Adesokan  |  Northeastern University  |  April 2026
 * ═══════════════════════════════════════════════════════════════════ */

const BASE = "/Users/macbookpro/Desktop/CapData";
const pptx = new PptxGenJS();
const W = 13.33, H = 7.5;
pptx.defineLayout({ name: "POSTER16x9", width: W, height: H });
pptx.layout = "POSTER16x9";
pptx.author = "Kehinde Adesokan";
pptx.title = "Personalized Prediction of Toileting Events — 16:9 Poster";

/* ─── PALETTE (matches HTML :root) ──────────────────────────────── */
const NAVY = "1B2A4A", CORAL = "C05746", TEAL = "2C6E91", GREEN = "3D7A5A";
const CREAM = "FAF8F5", WHITE = "FFFFFF", CHARCOAL = "2D2A26";
const MUTED = "7A7570", LIGHT = "F0EDEA", HDSUB = "A8B5CA", BDR = "E8E5E2";

/* ─── GEOMETRY ──────────────────────────────────────────────────── */
const MX = 0.08, CGAP = 0.04;
const CW = (W - 2 * MX - 2 * CGAP) / 3;                 // ≈ 4.363″
const CX = [MX, MX + CW + CGAP, MX + 2 * (CW + CGAP)];  // col x-starts

const HDR_H = 0.55, STRIPE = 0.02;
const CT = HDR_H + STRIPE + 0.03;                         // content top ≈ 0.60
const FTR_H = 0.20, FSTR_Y = H - FTR_H - 0.015;
const FTR_Y = FSTR_Y + 0.015;
const CE = FSTR_Y - 0.02;                                 // content end ≈ 7.265

const SH = 0.18;    // section-header bar
const PAD = 0.05;   // card-body padding
const RG = 0.03;    // gap between cards

/* ─── IMAGES ────────────────────────────────────────────────────── */
const IM = {};
const map = {
  fig01: "phase1_exploratory/Fig01_urination_hourly_distribution.png",
  fig07: "phase1_exploratory/Fig07_daily_event_counts_trend.png",
  fig09: "phase1b_signal_analysis/Fig09_time_to_void_after_drinking.png",
  fig12: "phase1b_signal_analysis/Fig12_cumulative_intake_vs_event_probability.png",
  fig19: "phase3_modeling/Fig19_feature_importance_lightgbm.png",
  fig21: "phase3_modeling/Fig21_shap_beeswarm_plot.png",
  fig22: "phase3_modeling/Fig22_feature_set_comparison.png",
  fig23: "phase3b_test_evaluation/Fig23_test_precision_recall_urination.png",
  fig24: "phase3b_test_evaluation/Fig24_clinical_operating_points.png",
};
for (const [k, v] of Object.entries(map)) {
  IM[k] = path.join(BASE, v);
  if (!fs.existsSync(IM[k])) { console.error(`MISSING ${k}: ${IM[k]}`); process.exit(1); }
}

/* ─── SLIDE ─────────────────────────────────────────────────────── */
const sl = pptx.addSlide();
sl.background = { color: CREAM };

/* ═══════════════════  HELPERS  ═══════════════════════════════════ */
function card(x, y, w, h) {
  sl.addShape(pptx.ShapeType.rect, {
    x: x + 0.005, y: y + 0.005, w, h,
    rectRadius: 0.03, fill: { color: "D6D3D0" }
  });
  sl.addShape(pptx.ShapeType.rect, {
    x, y, w, h, rectRadius: 0.03, fill: { color: WHITE }
  });
}

function shdr(x, y, w, label) {
  sl.addShape(pptx.ShapeType.rect, {
    x, y, w, h: SH, rectRadius: 0.03, fill: { color: CORAL }
  });
  sl.addShape(pptx.ShapeType.rect, {
    x, y: y + SH * 0.5, w, h: SH * 0.5, fill: { color: CORAL }
  });
  sl.addText(label, {
    x: x + 0.06, y, w: w - 0.12, h: SH,
    fontSize: 8.5, fontFace: "Georgia", color: WHITE, bold: true, valign: "middle"
  });
}

function fig(x, y, w, h, p) {
  sl.addImage({ path: p, x, y, w, h, sizing: { type: "contain", w, h } });
}

function cap(x, y, w, t) {
  sl.addText(t, {
    x, y, w, h: 0.13,
    fontSize: 5, fontFace: "Calibri", color: MUTED, italic: true, valign: "top",
    lineSpacingMultiple: 1.1
  });
}

function kf(x, y, w, h, arr, accent = TEAL) {
  sl.addShape(pptx.ShapeType.rect, {
    x, y, w, h, rectRadius: 0.02, fill: { color: LIGHT }
  });
  sl.addShape(pptx.ShapeType.rect, {
    x, y, w: 0.02, h, fill: { color: accent }
  });
  sl.addText(arr, {
    x: x + 0.05, y, w: w - 0.08, h,
    fontSize: 6, fontFace: "Calibri", color: CHARCOAL,
    valign: "middle", lineSpacingMultiple: 1.2
  });
}

/* ═══════════════════  HEADER  ═══════════════════════════════════ */
sl.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: HDR_H, fill: { color: NAVY } });

const LG = 0.38, LGY = (HDR_H - LG) / 2;
[[0.10, "[HIMSS\nLOGO]"], [W - 0.10 - LG, "[NU\nLOGO]"]].forEach(([lx, lt]) => {
  sl.addShape(pptx.ShapeType.rect, {
    x: lx, y: LGY, w: LG, h: LG, rectRadius: 0.04,
    line: { color: "6B7FA0", width: 0.75, dashType: "dash" }
  });
  sl.addText(lt, {
    x: lx, y: LGY, w: LG, h: LG,
    fontSize: 4.5, color: "8899B3", bold: true,
    align: "center", valign: "middle", fontFace: "Calibri"
  });
});

sl.addText(
  "Personalized Prediction of Toileting Events for Caregivers of Individuals\nWith Severe Autism: A Longitudinal Machine Learning Study",
  { x: 0.55, y: 0.02, w: W - 1.10, h: 0.28,
    fontSize: 13, fontFace: "Georgia", color: WHITE, bold: true,
    align: "center", valign: "middle", lineSpacingMultiple: 1.08 }
);
sl.addText("Kehinde Adesokan", {
  x: 0.55, y: 0.30, w: W - 1.10, h: 0.11,
  fontSize: 8, fontFace: "Calibri", color: WHITE, bold: true,
  align: "center", valign: "top"
});
sl.addText(
  "MS Health Informatics  |  Bouvé College of Health Sciences  |  Northeastern University  |  April 2026",
  { x: 0.55, y: 0.40, w: W - 1.10, h: 0.10,
    fontSize: 6, fontFace: "Calibri", color: HDSUB,
    align: "center", valign: "top" }
);
sl.addShape(pptx.ShapeType.rect, { x: 0, y: HDR_H, w: W, h: STRIPE, fill: { color: CORAL } });

/* ═══════════════════  FOOTER  ═══════════════════════════════════ */
sl.addShape(pptx.ShapeType.rect, { x: 0, y: FSTR_Y, w: W, h: 0.012, fill: { color: CORAL } });
sl.addShape(pptx.ShapeType.rect, { x: 0, y: FTR_Y, w: W, h: FTR_H, fill: { color: NAVY } });
sl.addText(
  "k.adesokan@northeastern.edu   |   Bouvé College of Health Sciences, Northeastern University   |   HINF 6215 Health Informatics Capstone   |   New England HIMSS 2026",
  { x: 0.2, y: FTR_Y, w: W - 0.4, h: FTR_H,
    fontSize: 5.5, fontFace: "Calibri", color: HDSUB,
    align: "center", valign: "middle" }
);

/* ═══════════════════════════════════════════════════════════════════
 *  LEFT COLUMN
 * ═══════════════════════════════════════════════════════════════════ */
const L = CX[0], LW = CW;
const lb = L + PAD, lbw = LW - 2 * PAD;
let y = CT;

/* ── Introduction ────────────────────────────────────────────────── */
const introH = 0.52;
card(L, y, LW, introH);
shdr(L, y, LW, "Introduction");
sl.addText([
  { text: "35% of children with autism have not achieved consistent toileting by age 3. For many the challenge persists into adulthood, managed by caregivers with fixed schedules and constant vigilance. This study asks: ",
    options: { fontSize: 7, fontFace: "Calibri", color: CHARCOAL } },
  { text: "can personalized machine learning do better?",
    options: { fontSize: 7, fontFace: "Calibri", color: CHARCOAL, bold: true } }
], { x: lb, y: y + SH + 0.03, w: lbw, h: introH - SH - 0.06, valign: "top", lineSpacingMultiple: 1.22 });
y += introH + RG;

/* ── Methods ─────────────────────────────────────────────────────── */
const statH = 1.30, refH = 0.35;
const methH = (CE - CT) - introH - statH - refH - 3 * RG;  // flex fill
card(L, y, LW, methH);
shdr(L, y, LW, "Methods");

// Flowchart
const fX = L + 0.10, fW = LW - 0.20;
const stepH = 0.46, arH = 0.16;
let fY = y + SH + 0.06;
const steps = [
  ["1", "Data Collection",      "262,944 obs | 10-min windows | 5 yrs (2021–2025) | 60 vars"],
  ["2", "Feature Engineering",   "Set A (54 schedule+intake) | Set B (8 wearable) | Set C (62 combined)"],
  ["3", "Temporal Split",        "Train 2021–2023 | Validate 2024 | Test 2025 (no leakage)"],
  ["4", "Model Training",        "LightGBM + Logistic Regression | 30 models | 5 targets"],
  ["5", "Evaluation",            "AUPRC primary metric | SHAP interpretability | Clinical ops"],
];
steps.forEach(([n, lbl, desc], i) => {
  // box
  sl.addShape(pptx.ShapeType.rect, {
    x: fX, y: fY, w: fW, h: stepH,
    rectRadius: 0.025, line: { color: CORAL, width: 0.75 }, fill: { color: WHITE }
  });
  // number circle
  const cD = 0.14;
  sl.addShape(pptx.ShapeType.ellipse, {
    x: fX + 0.06, y: fY + (stepH - cD) / 2, w: cD, h: cD,
    fill: { color: CORAL }
  });
  sl.addText(n, {
    x: fX + 0.06, y: fY + (stepH - cD) / 2, w: cD, h: cD,
    fontSize: 5.5, fontFace: "Calibri", color: WHITE, bold: true,
    align: "center", valign: "middle"
  });
  // label
  sl.addText(lbl, {
    x: fX + 0.24, y: fY + 0.04, w: fW - 0.30, h: 0.15,
    fontSize: 7, fontFace: "Calibri", color: NAVY, bold: true, valign: "middle"
  });
  // description
  sl.addText(desc, {
    x: fX + 0.24, y: fY + 0.20, w: fW - 0.30, h: 0.22,
    fontSize: 5, fontFace: "Calibri", color: MUTED, valign: "top", lineSpacingMultiple: 1.1
  });
  fY += stepH;
  // arrow
  if (i < steps.length - 1) {
    const mid = fX + fW / 2;
    sl.addShape(pptx.ShapeType.rect, {
      x: mid - 0.005, y: fY, w: 0.01, h: arH * 0.55, fill: { color: CORAL }
    });
    sl.addText("▼", {
      x: mid - 0.04, y: fY + arH * 0.30, w: 0.08, h: arH * 0.65,
      fontSize: 5, color: CORAL, align: "center", valign: "middle"
    });
    fY += arH;
  }
});

// Feature-set boxes
fY += 0.10;
const fbW = (fW - 0.06) / 3, fbH = 0.36;
[
  [TEAL,  "Set A", "54 features\nSchedule, History, Intake"],
  [CORAL, "Set B", "8 features\nHR, HRV, Activity"],
  [GREEN, "Set C", "62 features\nCombined A + B"],
].forEach(([col, ttl, dsc], i) => {
  const bx = fX + i * (fbW + 0.03);
  sl.addShape(pptx.ShapeType.rect, {
    x: bx, y: fY, w: fbW, h: fbH, rectRadius: 0.025, fill: { color: col }
  });
  sl.addText(ttl, {
    x: bx, y: fY + 0.02, w: fbW, h: 0.13,
    fontSize: 6.5, fontFace: "Calibri", color: WHITE, bold: true,
    align: "center", valign: "middle"
  });
  sl.addText(dsc, {
    x: bx + 0.02, y: fY + 0.15, w: fbW - 0.04, h: 0.18,
    fontSize: 4.5, fontFace: "Calibri", color: WHITE,
    align: "center", valign: "top", lineSpacingMultiple: 1.15
  });
});
y += methH + RG;

/* ── Stationarity (Fig 07) ───────────────────────────────────────── */
card(L, y, LW, statH);
shdr(L, y, LW, "Data Stationarity");
const stIY = y + SH + 0.03, stIH = statH - SH - 0.19;
fig(lb, stIY, lbw, stIH, IM.fig07);
cap(lb, stIY + stIH + 0.01, lbw,
  "Fig 1. Daily event counts 2021–2025 with 30-day rolling avg. Urination steady at 6.0/day; bowel 1.5/day — no drift.");
y += statH + RG;

/* ── References ──────────────────────────────────────────────────── */
card(L, y, LW, refH);
sl.addText([
  { text: "References\n",
    options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL, bold: true } },
  { text: "Simon et al. (2022) Res. Autism Spectrum Disord. · Marsack-Topolewski et al. (2021) PLOS ONE · Ali et al. (2022) BMC Med. Inform. · Nan et al. (2024) Sensors · Kline et al. (2022) npj Digital Med. · Lundberg et al. (2020) Nature Mach. Intell. · Full list in capstone paper.",
    options: { fontSize: 4.5, fontFace: "Calibri", color: MUTED, italic: true } }
], { x: lb, y: y + 0.03, w: lbw, h: refH - 0.06, valign: "top", lineSpacingMultiple: 1.2 });

/* ═══════════════════════════════════════════════════════════════════
 *  CENTER COLUMN
 * ═══════════════════════════════════════════════════════════════════ */
const M = CX[1], MW = CW;
const mb = M + PAD, mbw = MW - 2 * PAD;
const mHalf = (mbw - 0.04) / 2;
y = CT;

/* ── Results: Patterns ───────────────────────────────────────────── */
const patH = 2.60;
card(M, y, MW, patH);
shdr(M, y, MW, "Results: Patterns in the Data");
let py = y + SH + 0.03;

// paired: Fig01 + Fig12
const pIH = 0.82;
fig(mb, py, mHalf, pIH, IM.fig01);
fig(mb + mHalf + 0.04, py, mHalf, pIH, IM.fig12);
py += pIH;
cap(mb, py, mHalf, "Fig 2. Urination by hour — 07:00 peak.");
cap(mb + mHalf + 0.04, py, mHalf, "Fig 3. Dose-response: intake vs void probability.");
py += 0.13;

// Fig09 full-width
const f09H = 0.68;
fig(mb, py, mbw, f09H, IM.fig09);
py += f09H;
cap(mb, py, mbw,
  "Fig 4. Time to void: post-intake (blue) vs control (orange). Drinking accelerates voiding by 30–150 min.");
py += 0.14;

// Key finding
kf(mb, py, mbw, 0.30, [
  { text: "97.3%", options: { bold: true, color: NAVY } },
  { text: " of events during waking hours. Circadian peak at 07:00. Cumulative intake drove a ", options: {} },
  { text: "4× increase", options: { bold: true, color: NAVY } },
  { text: " in voiding probability (11.9% → 50.4%). Wearable delta < 0.35 bpm — clinically meaningless.", options: {} }
]);
y += patH + RG;

/* ── Results: Prediction Performance ─────────────────────────────── */
const perfH = CE - y;   // flex fill
card(M, y, MW, perfH);
shdr(M, y, MW, "Results: Prediction Performance");
py = y + SH + 0.03;

// paired: Fig22 + Fig23
fig(mb, py, mHalf, pIH, IM.fig22);
fig(mb + mHalf + 0.04, py, mHalf, pIH, IM.fig23);
py += pIH;
cap(mb, py, mHalf, "Fig 5. AUPRC by feature set & target.");
cap(mb + mHalf + 0.04, py, mHalf, "Fig 6. Test precision-recall (urination).");
py += 0.14;

// Performance table
const hdrOpts = { fill: { color: NAVY }, color: WHITE, bold: true, fontSize: 6, fontFace: "Calibri", align: "left" };
const dataRows = [
  ["Urination 15 min", "0.162", "0.350", "+116%"],
  ["Urination 30 min", "0.243", "0.465", "+91%"],
  ["Urination 60 min", "0.457", "0.685", "+50%"],
  ["Bowel 30 min",     "0.098", "0.147", "+50%"],
  ["Bowel 60 min",     "0.187", "0.255", "+36%"],
];
const tblRows = [
  ["Target", "Baseline", "Test AUPRC", "Gain"].map(t => ({ text: t, options: hdrOpts })),
  ...dataRows.map((r, ri) => r.map((c, ci) => ({
    text: c,
    options: {
      fontSize: 5.5, fontFace: "Calibri",
      color: ci === 3 ? GREEN : CHARCOAL,
      bold: ci === 3,
      fill: { color: ri % 2 === 1 ? LIGHT : WHITE },
      align: "left"
    }
  })))
];
sl.addTable(tblRows, {
  x: mb, y: py, w: mbw,
  colW: [mbw * 0.38, mbw * 0.20, mbw * 0.22, mbw * 0.20],
  rowH: 0.15,
  border: { type: "solid", pt: 0.3, color: BDR },
});
py += 6 * 0.15 + 0.06;

// Wearable banner
const wbH = 0.30;
sl.addShape(pptx.ShapeType.rect, {
  x: mb, y: py, w: mbw, h: wbH,
  rectRadius: 0.025, fill: { color: CHARCOAL }
});
sl.addText("WEARABLE SIGNALS ADDED ZERO PREDICTIVE VALUE", {
  x: mb, y: py + 0.02, w: mbw, h: 0.14,
  fontSize: 7, fontFace: "Georgia", color: CORAL, bold: true,
  align: "center", valign: "middle"
});
sl.addText("Set A ≈ Set C (within 0.003 AUPRC)  |  Set B worse than baseline on every target", {
  x: mb, y: py + 0.16, w: mbw, h: 0.12,
  fontSize: 5, fontFace: "Calibri", color: "B0ADA9",
  align: "center", valign: "top"
});

/* ═══════════════════════════════════════════════════════════════════
 *  RIGHT COLUMN
 * ═══════════════════════════════════════════════════════════════════ */
const R = CX[2], RW = CW;
const rb = R + PAD, rbw = RW - 2 * PAD;
const rHalf = (rbw - 0.04) / 2;
y = CT;

/* ── Clinical Impact ─────────────────────────────────────────────── */
const clinH = 1.50;
card(R, y, RW, clinH);
shdr(R, y, RW, "Clinical Impact");
let ry = y + SH + 0.03;
const ciH = 0.82;
fig(rb, ry, rbw, ciH, IM.fig24);
ry += ciH;
cap(rb, ry, rbw,
  "Fig 7. Clinical operating points: daily alerts (left) & precision vs recall (right).");
ry += 0.13;

kf(rb, ry, rbw, 0.22, [
  { text: "At 60-min horizon: model captures ", options: {} },
  { text: "72% of events", options: { bold: true, color: GREEN, fontFace: "Georgia" } },
  { text: " with 43% false-alert rate → ~", options: {} },
  { text: "25 true + 18 false alerts/day", options: { bold: true, color: NAVY } },
  { text: ".", options: {} }
]);
y += clinH + RG;

/* ── Interpretability ────────────────────────────────────────────── */
const intpH = 1.95;
card(R, y, RW, intpH);
shdr(R, y, RW, "What Drives the Predictions?");
ry = y + SH + 0.03;

// paired: Fig19 + Fig21
const iiH = 0.80;
fig(rb, ry, rHalf, iiH, IM.fig19);
fig(rb + rHalf + 0.04, ry, rHalf, iiH, IM.fig21);
ry += iiH;
cap(rb, ry, rHalf, "Fig 8. Top features by LightGBM gain.");
cap(rb + rHalf + 0.04, ry, rHalf, "Fig 9. SHAP beeswarm plot.");
ry += 0.13;

// Feature list
const feats = [
  [TEAL,  "Hour of Day",              "(313K gain)"],
  [TEAL,  "Min Since Last Urination",  "(244K)"],
  [TEAL,  "Minutes Until Sleep",       "(229K)"],
  [CORAL, "Cumulative Fluid Intake",   "(101K)"],
];
const flH = 0.13;
feats.forEach(([col, nm, gn], i) => {
  const fy = ry + i * (flH + 0.02);
  sl.addShape(pptx.ShapeType.rect, {
    x: rb, y: fy, w: rbw, h: flH,
    rectRadius: 0.02, fill: { color: LIGHT }
  });
  const cD = 0.10;
  sl.addShape(pptx.ShapeType.ellipse, {
    x: rb + 0.03, y: fy + (flH - cD) / 2, w: cD, h: cD,
    fill: { color: col }
  });
  sl.addText([
    { text: nm, options: { bold: true, color: NAVY, fontSize: 6 } },
    { text: "  " + gn, options: { color: MUTED, fontSize: 5 } }
  ], {
    x: rb + 0.16, y: fy, w: rbw - 0.19, h: flH,
    fontFace: "Calibri", valign: "middle"
  });
});
ry += feats.length * (flH + 0.02) + 0.01;

sl.addText("All four trackable by a caregiver or simple mobile app — no wearable required.", {
  x: rb, y: ry, w: rbw, h: 0.12,
  fontSize: 5.5, fontFace: "Calibri", color: TEAL, bold: true,
  align: "center", valign: "middle"
});
y += intpH + RG;

/* ── Fixed vs Personalized ───────────────────────────────────────── */
const cmpH = 0.55;
card(R, y, RW, cmpH);
const csW = (rbw - 0.22) / 2;
const csH = cmpH - 0.06;
const csY = y + 0.03;

// Fixed Schedule
sl.addShape(pptx.ShapeType.rect, {
  x: rb, y: csY, w: csW, h: csH,
  rectRadius: 0.03, fill: { color: "F9EDEB" },
  line: { color: CORAL, width: 0.5 }
});
sl.addText("🕒", { x: rb, y: csY, w: csW, h: 0.12, fontSize: 10, align: "center", valign: "middle" });
sl.addText("Fixed Schedule", {
  x: rb, y: csY + 0.11, w: csW, h: 0.10,
  fontSize: 6.5, fontFace: "Georgia", color: CORAL, bold: true, align: "center", valign: "middle"
});
sl.addText("Same prompt every 2 hrs\nNo adaptation", {
  x: rb, y: csY + 0.21, w: csW, h: 0.14,
  fontSize: 4.5, fontFace: "Calibri", color: CHARCOAL, align: "center", valign: "top", lineSpacingMultiple: 1.1
});
sl.addText("✗ Missed events", {
  x: rb, y: csY + 0.36, w: csW, h: 0.10,
  fontSize: 5.5, fontFace: "Calibri", color: CORAL, bold: true, align: "center", valign: "middle"
});

// VS circle
const vsX = rb + csW, vsW = 0.22;
sl.addShape(pptx.ShapeType.ellipse, {
  x: vsX + (vsW - 0.14) / 2, y: csY + (csH - 0.14) / 2, w: 0.14, h: 0.14,
  fill: { color: NAVY }
});
sl.addText("VS", {
  x: vsX, y: csY + (csH - 0.14) / 2, w: vsW, h: 0.14,
  fontSize: 5, fontFace: "Georgia", color: WHITE, bold: true, align: "center", valign: "middle"
});

// Personalized
const pX = rb + csW + vsW;
sl.addShape(pptx.ShapeType.rect, {
  x: pX, y: csY, w: csW, h: csH,
  rectRadius: 0.03, fill: { color: "EBF5EF" },
  line: { color: GREEN, width: 0.5 }
});
sl.addText("📱", { x: pX, y: csY, w: csW, h: 0.12, fontSize: 10, align: "center", valign: "middle" });
sl.addText("Personalized Prediction", {
  x: pX, y: csY + 0.11, w: csW, h: 0.10,
  fontSize: 6.5, fontFace: "Georgia", color: GREEN, bold: true, align: "center", valign: "middle"
});
sl.addText("Adapts to intake & timing\n72% captured", {
  x: pX, y: csY + 0.21, w: csW, h: 0.14,
  fontSize: 4.5, fontFace: "Calibri", color: CHARCOAL, align: "center", valign: "top", lineSpacingMultiple: 1.1
});
sl.addText("✓ Smarter care", {
  x: pX, y: csY + 0.36, w: csW, h: 0.10,
  fontSize: 5.5, fontFace: "Calibri", color: GREEN, bold: true, align: "center", valign: "middle"
});
y += cmpH + RG;

/* ── Conclusions ─────────────────────────────────────────────────── */
const conclH = CE - y;
card(R, y, RW, conclH);
shdr(R, y, RW, "Conclusions");
ry = y + SH + 0.04;

const concl = [
  [
    { text: "Personalized toileting prediction is feasible using 5 years of caregiver-logged data — AUPRC gains of ",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } },
    { text: "36–116%",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL, bold: true } },
    { text: " over baseline.",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } }
  ],
  [
    { text: "Wearable tech is not required.",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL, bold: true } },
    { text: " Schedule, intake, and void history alone achieve best performance.",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } }
  ],
  [
    { text: "Top 4 features are all caregiver-trackable, enabling deployment in ",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } },
    { text: "resource-limited settings",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL, bold: true } },
    { text: ".",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } }
  ],
  [
    { text: "At the 60-min horizon, the model captures ",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } },
    { text: "72% of urination events",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL, bold: true } },
    { text: ", outperforming fixed-schedule prompting.",
      options: { fontSize: 6, fontFace: "Calibri", color: CHARCOAL } }
  ]
];
concl.forEach((arr, i) => {
  const cY = ry + i * 0.26;
  const cD = 0.11;
  sl.addShape(pptx.ShapeType.ellipse, {
    x: rb, y: cY + 0.01, w: cD, h: cD, fill: { color: CORAL }
  });
  sl.addText(String(i + 1), {
    x: rb, y: cY + 0.01, w: cD, h: cD,
    fontSize: 6, fontFace: "Calibri", color: WHITE, bold: true,
    align: "center", valign: "middle"
  });
  sl.addText(arr, {
    x: rb + 0.15, y: cY, w: rbw - 0.18, h: 0.24,
    valign: "top", lineSpacingMultiple: 1.2
  });
});

// Future work
const fwY = ry + concl.length * 0.26 + 0.06;
sl.addShape(pptx.ShapeType.rect, {
  x: rb, y: fwY - 0.02, w: rbw, h: 0.005, fill: { color: BDR }
});
sl.addText([
  { text: "Future Work: ",
    options: { bold: true, fontSize: 5.5, fontFace: "Calibri", color: MUTED, italic: true } },
  { text: "Multi-participant replication · Reinforcement learning for alert optimization · Mobile app development",
    options: { fontSize: 5.5, fontFace: "Calibri", color: MUTED, italic: true } }
], { x: rb, y: fwY, w: rbw, h: 0.18, valign: "top" });

/* ═══════════════════  EXPORT  ═══════════════════════════════════ */
const OUT = path.join(BASE, "poster_16x9.pptx");
pptx.writeFile({ fileName: OUT })
  .then(() => console.log(`✅ 16:9 poster saved → ${OUT}`))
  .catch(err => { console.error("Error:", err); process.exit(1); });
