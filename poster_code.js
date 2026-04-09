const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaCheckCircle } = require("react-icons/fa");

function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

async function main() {
  let pres = new pptxgen();

  pres.defineLayout({ name: "POSTER_DIGITAL", width: 20, height: 11.25 });
  pres.layout = "POSTER_DIGITAL";
  pres.author = "Kehinde Adesokan";
  pres.title = "Personalized Prediction of Toileting Events";

  let slide = pres.addSlide();

  // === COLORS ===
  const DARK_NAVY = "1B2A4A";
  const MED_NAVY = "2C3E6B";
  const ACCENT_RED = "C0392B";
  const ACCENT_TEAL = "1A8A7D";
  const LIGHT_BG = "F5F6FA";
  const WHITE = "FFFFFF";
  const DARK_TEXT = "1B1B1B";
  const LIGHT_BORDER = "D0D4E0";
  const HIGHLIGHT_GOLD = "E8A838";
  const SUBTLE_BG = "EEF0F5";

  // === FONT SIZES — enlarged ~50% for 100% zoom readability ===
  const TITLE_SIZE = 28;
  const AUTHOR_SIZE = 18;
  const AFFIL_SIZE = 12;
  const SECTION_HEADER_SIZE = 16;
  const BODY_SIZE = 13;
  const RQ_SIZE = 12;
  const METHOD_TITLE_SIZE = 13;
  const METHOD_DESC_SIZE = 10.5;
  const CONCLUSION_SIZE = 11.5;
  const FOOTER_SIZE = 10;
  const REF_SIZE = 8.5;
  const CAPTION_SIZE = 8;
  const TABLE_HEADER_SIZE = 10.5;
  const TABLE_BODY_SIZE = 9.5;
  const FEATURE_LABEL_SIZE = 12;
  const CALLOUT_TITLE_SIZE = 14;
  const CALLOUT_SIZE = 12;

  // === SLIDE DIMENSIONS ===
  const SW = 20;
  const SH = 11.25;

  slide.background = { color: LIGHT_BG };

  // ============================================================
  // HEADER (taller to fit bigger fonts)
  // ============================================================
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: SW, h: 1.50,
    fill: { color: DARK_NAVY }
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.50, w: SW, h: 0.05,
    fill: { color: ACCENT_RED }
  });

  // Logo LEFT
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 0.2, w: 1.1, h: 1.0,
    fill: { color: "243560" },
    line: { color: "FFFFFF", width: 1 }
  });
  slide.addText("INSERT\nHIMSS LOGO", {
    x: 0.3, y: 0.2, w: 1.1, h: 1.0,
    fontSize: 7, color: "8899BB", bold: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });

  // Logo RIGHT
  slide.addShape(pres.shapes.RECTANGLE, {
    x: SW - 1.4, y: 0.2, w: 1.1, h: 1.0,
    fill: { color: "243560" },
    line: { color: "FFFFFF", width: 1 }
  });
  slide.addText("INSERT\nNU LOGO", {
    x: SW - 1.4, y: 0.2, w: 1.1, h: 1.0,
    fontSize: 7, color: "8899BB", bold: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });

  // Title
  slide.addText("Personalized Prediction of Toileting Events for Caregivers\nof Individuals With Severe Autism: A Longitudinal Machine Learning Study", {
    x: 1.6, y: 0.05, w: SW - 3.2, h: 0.88,
    fontSize: TITLE_SIZE, color: WHITE, bold: true,
    align: "center", valign: "middle", fontFace: "Arial Black",
    lineSpacingMultiple: 1.0
  });

  // Author
  slide.addText("Kehinde Adesokan", {
    x: 1.6, y: 0.88, w: SW - 3.2, h: 0.30,
    fontSize: AUTHOR_SIZE, color: HIGHLIGHT_GOLD, bold: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });

  // Affiliation
  slide.addText("MS Health Informatics  |  Bouvé College of Health Sciences  |  Northeastern University  |  April 2026", {
    x: 1.6, y: 1.14, w: SW - 3.2, h: 0.28,
    fontSize: AFFIL_SIZE, color: "B0BDD4", bold: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });

  // ============================================================
  // 3-COLUMN LAYOUT
  // ============================================================
  const MARGIN = 0.25;
  const GAP = 0.22;
  const USABLE_W = SW - 2 * MARGIN;
  const COL_W = (USABLE_W - 2 * GAP) / 3;
  const COL1_X = MARGIN;
  const COL2_X = MARGIN + COL_W + GAP;
  const COL3_X = MARGIN + 2 * (COL_W + GAP);
  const CONTENT_TOP = 1.65;

  // Helper: section header (taller bar)
  function addSectionHeader(x, y, w, text, emoji) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: w, h: 0.38,
      fill: { color: ACCENT_RED }
    });
    slide.addText(emoji + "  " + text, {
      x: x + 0.1, y: y, w: w - 0.2, h: 0.38,
      fontSize: SECTION_HEADER_SIZE, color: WHITE, bold: true,
      align: "left", valign: "middle", fontFace: "Arial Black"
    });
    return y + 0.38;
  }

  // Helper: insert figure with generous sizing + caption below
  function addFigureImage(x, y, w, h, imagePath, caption) {
    const imgH = h - 0.30;
    slide.addImage({
      path: imagePath,
      x: x, y: y, w: w, h: imgH,
      sizing: { type: "contain", w: w, h: imgH }
    });
    if (caption) {
      slide.addText(caption, {
        x: x + 0.05,
        y: y + imgH + 0.02,
        w: w - 0.1,
        h: 0.26,
        fontSize: CAPTION_SIZE, color: "555555", bold: true, italic: true,
        align: "left", valign: "top", fontFace: "Arial"
      });
    }
    return y + h + 0.05;
  }

  // ============================================================
  // LEFT COLUMN
  // ============================================================
  let ly = CONTENT_TOP;

  // INTRODUCTION
  ly = addSectionHeader(COL1_X, ly, COL_W, "Introduction", "📋");
  ly += 0.05;

  slide.addText([
    { text: "Thirty-five percent of children with autism have not achieved consistent toileting by age three. ", options: { bold: true, breakLine: false } },
    { text: "For many, the challenge follows them into adulthood, and it falls on their caregivers to manage it every single day. The tools available to these families have not changed in decades: fixed schedules, rigid prompting, constant vigilance.", options: { bold: false, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 5 } },
    { text: "This study asked a simple question: can we do better?", options: { bold: true, italic: true, color: ACCENT_RED } }
  ], {
    x: COL1_X + 0.1, y: ly, w: COL_W - 0.2, h: 1.10,
    fontSize: BODY_SIZE, color: DARK_TEXT, fontFace: "Arial",
    lineSpacingMultiple: 1.2, valign: "top"
  });
  ly += 1.15;

  // RESEARCH QUESTIONS
  ly = addSectionHeader(COL1_X, ly, COL_W, "Research Questions", "🔬");
  ly += 0.05;

  slide.addText([
    { text: "RQ1: ", options: { bold: true, color: ACCENT_RED, breakLine: false } },
    { text: "Can ML predict when a toileting event is likely, using data a caregiver could realistically collect?", options: { bold: true, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "RQ2: ", options: { bold: true, color: ACCENT_RED, breakLine: false } },
    { text: "Do wearable physiological signals (HR, HRV) add predictive value beyond caregiver-logged features?", options: { bold: true, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "RQ3: ", options: { bold: true, color: ACCENT_RED, breakLine: false } },
    { text: "Can model outputs be translated into clinically actionable caregiver alerts?", options: { bold: true } }
  ], {
    x: COL1_X + 0.1, y: ly, w: COL_W - 0.2, h: 1.20,
    fontSize: RQ_SIZE, color: DARK_TEXT, fontFace: "Arial",
    lineSpacingMultiple: 1.25, valign: "top"
  });
  ly += 1.26;

  // METHODS
  ly = addSectionHeader(COL1_X, ly, COL_W, "Methods", "⚙️");
  ly += 0.05;

  const methodSteps = [
    { num: "1", title: "Data Collection", desc: "262,944 obs  |  10-min windows  |  5 yrs (2021–2025)  |  60 vars" },
    { num: "2", title: "Feature Engineering", desc: "Set A: 54 (schedule+intake)  |  Set B: 8 (wearable)  |  Set C: 62 (combined)" },
    { num: "3", title: "Temporal Split", desc: "Train 2021–2023  |  Validate 2024  |  Test 2025  (no leakage)" },
    { num: "4", title: "Model Training", desc: "LightGBM + Logistic Regression  |  30 models  |  5 targets" },
    { num: "5", title: "Evaluation", desc: "AUPRC primary  |  SHAP interpretability  |  Clinical operating points" }
  ];

  for (let i = 0; i < methodSteps.length; i++) {
    const step = methodSteps[i];
    const stepY = ly + i * 0.48;

    slide.addShape(pres.shapes.RECTANGLE, {
      x: COL1_X + 0.05, y: stepY, w: COL_W - 0.1, h: 0.44,
      fill: { color: WHITE },
      line: { color: LIGHT_BORDER, width: 0.5 }
    });

    slide.addShape(pres.shapes.OVAL, {
      x: COL1_X + 0.12, y: stepY + 0.08, w: 0.28, h: 0.28,
      fill: { color: ACCENT_TEAL }
    });
    slide.addText(step.num, {
      x: COL1_X + 0.12, y: stepY + 0.08, w: 0.28, h: 0.28,
      fontSize: 11, color: WHITE, bold: true,
      align: "center", valign: "middle", fontFace: "Arial Black"
    });

    slide.addText(step.title, {
      x: COL1_X + 0.48, y: stepY + 0.01, w: COL_W - 0.6, h: 0.22,
      fontSize: METHOD_TITLE_SIZE, color: DARK_NAVY, bold: true,
      align: "left", valign: "middle", fontFace: "Arial Black", margin: 0
    });
    slide.addText(step.desc, {
      x: COL1_X + 0.48, y: stepY + 0.22, w: COL_W - 0.6, h: 0.20,
      fontSize: METHOD_DESC_SIZE, color: "444444", bold: true,
      align: "left", valign: "top", fontFace: "Arial", margin: 0
    });
  }
  ly += methodSteps.length * 0.48 + 0.06;

  // FEATURE SETS
  const fsW = (COL_W - 0.2) / 3;
  const fsSets = [
    { label: "Set A", count: "54 features", desc: "Schedule, History,\nIntake", color: ACCENT_TEAL },
    { label: "Set B", count: "8 features", desc: "HR, HRV,\nActivity", color: MED_NAVY },
    { label: "Set C", count: "62 features", desc: "Combined\nA + B", color: ACCENT_RED }
  ];

  for (let i = 0; i < 3; i++) {
    const fs = fsSets[i];
    const fx = COL1_X + 0.05 + i * (fsW + 0.05);

    slide.addShape(pres.shapes.RECTANGLE, {
      x: fx, y: ly, w: fsW, h: 0.85,
      fill: { color: WHITE },
      line: { color: fs.color, width: 1.5 }
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: fx, y: ly, w: fsW, h: 0.24,
      fill: { color: fs.color }
    });
    slide.addText(fs.label, {
      x: fx, y: ly, w: fsW, h: 0.24,
      fontSize: 11, color: WHITE, bold: true,
      align: "center", valign: "middle", fontFace: "Arial Black"
    });
    slide.addText(fs.count, {
      x: fx, y: ly + 0.26, w: fsW, h: 0.20,
      fontSize: 10, color: DARK_NAVY, bold: true,
      align: "center", valign: "middle", fontFace: "Arial"
    });
    slide.addText(fs.desc, {
      x: fx + 0.05, y: ly + 0.46, w: fsW - 0.1, h: 0.35,
      fontSize: 8.5, color: "555555", bold: true,
      align: "center", valign: "top", fontFace: "Arial",
      lineSpacingMultiple: 1.2
    });
  }
  ly += 0.91;

  // DATA STATIONARITY
  ly = addSectionHeader(COL1_X, ly, COL_W, "Data Stationarity", "📊");
  ly += 0.04;
  ly = addFigureImage(COL1_X + 0.05, ly, COL_W - 0.1, 1.55,
    "./phase1_exploratory/Fig07_daily_event_counts_trend.png",
    "Fig 1. Daily event counts 2021–2025. Urination steady at 6.0/day, bowel 1.5/day — no drift.");

  // REFERENCES (compact at bottom)
  slide.addText([
    { text: "References: ", options: { bold: true, color: DARK_TEXT } },
    { text: "Simon et al. (2022) Res. Autism Spectrum Dis. • Marsack-Topolewski et al. (2021) PLOS ONE • Ali et al. (2022) BMC Med. Inform. • Kline et al. (2022) npj Digital Med. • Lundberg et al. (2020) Nature Mach. Intell. Full list in capstone paper.", options: {} }
  ], {
    x: COL1_X + 0.08, y: ly + 0.02, w: COL_W - 0.16, h: 0.28,
    fontSize: REF_SIZE, color: "666666", fontFace: "Arial",
    valign: "top", lineSpacingMultiple: 1.1
  });

  // ============================================================
  // CENTER COLUMN
  // ============================================================
  let cy = CONTENT_TOP;

  // RESULTS: PATTERNS
  cy = addSectionHeader(COL2_X, cy, COL_W, "Results: Patterns in the Data", "📈");
  cy += 0.04;

  cy = addFigureImage(COL2_X + 0.05, cy, COL_W - 0.1, 1.80,
    "./phase1_exploratory/Fig01_urination_hourly_distribution.png",
    "Fig 2. Urination events by hour of day — sharp 07:00 peak; 97.3% during waking hours.");
  cy += 0.04;

  // KEY FINDINGS callout
  slide.addShape(pres.shapes.RECTANGLE, {
    x: COL2_X + 0.05, y: cy, w: COL_W - 0.1, h: 0.34,
    fill: { color: DARK_NAVY }
  });
  slide.addText("KEY:  97.3% waking hours  |  Fluid intake → 4× voiding increase  |  Wearable Δ < 0.35 bpm", {
    x: COL2_X + 0.12, y: cy, w: COL_W - 0.24, h: 0.34,
    fontSize: 10, color: WHITE, bold: true, fontFace: "Arial",
    align: "left", valign: "middle"
  });
  cy += 0.42;

  // RESULTS: PREDICTION
  cy = addSectionHeader(COL2_X, cy, COL_W, "Results: Prediction Performance", "🎯");
  cy += 0.04;

  cy = addFigureImage(COL2_X + 0.05, cy, COL_W - 0.1, 1.80,
    "./phase3_modeling/Fig22_feature_set_comparison.png",
    "Fig 3. AUPRC across feature sets and targets. Set A dominates; Set B worse than baseline.");
  cy += 0.04;

  cy = addFigureImage(COL2_X + 0.05, cy, COL_W - 0.1, 1.80,
    "./phase3b_test_evaluation/Fig23_test_precision_recall_urination.png",
    "Fig 4. Test precision-recall curves (urination). Sets A & C overlap; Set B at bottom.");
  cy += 0.04;

  // TABLE
  slide.addShape(pres.shapes.RECTANGLE, {
    x: COL2_X + 0.05, y: cy, w: COL_W - 0.1, h: 0.28,
    fill: { color: DARK_NAVY }
  });
  slide.addText("Target                           Baseline       Test AUPRC       Gain", {
    x: COL2_X + 0.15, y: cy, w: COL_W - 0.3, h: 0.28,
    fontSize: TABLE_HEADER_SIZE, color: WHITE, bold: true, fontFace: "Consolas",
    align: "left", valign: "middle"
  });

  const tableData = [
    { target: "Urination 15 min", base: "0.162", test: "0.350", gain: "+116%" },
    { target: "Urination 30 min", base: "0.243", test: "0.465", gain: "+91%" },
    { target: "Urination 60 min", base: "0.457", test: "0.685", gain: "+50%" },
    { target: "Bowel 30 min",     base: "0.098", test: "0.147", gain: "+50%" },
    { target: "Bowel 60 min",     base: "0.187", test: "0.255", gain: "+36%" }
  ];

  for (let i = 0; i < tableData.length; i++) {
    const d = tableData[i];
    const rowY = cy + 0.28 + i * 0.22;
    const bgColor = i % 2 === 0 ? "F0F1F6" : WHITE;

    slide.addShape(pres.shapes.RECTANGLE, {
      x: COL2_X + 0.05, y: rowY, w: COL_W - 0.1, h: 0.22,
      fill: { color: bgColor }
    });

    slide.addText(d.target, {
      x: COL2_X + 0.15, y: rowY, w: 2.2, h: 0.22,
      fontSize: TABLE_BODY_SIZE, color: DARK_TEXT, bold: true, fontFace: "Arial",
      align: "left", valign: "middle", margin: 0
    });
    slide.addText(d.base, {
      x: COL2_X + 2.4, y: rowY, w: 1.0, h: 0.22,
      fontSize: TABLE_BODY_SIZE, color: DARK_TEXT, bold: true, fontFace: "Arial",
      align: "center", valign: "middle", margin: 0
    });
    slide.addText(d.test, {
      x: COL2_X + 3.4, y: rowY, w: 1.2, h: 0.22,
      fontSize: TABLE_BODY_SIZE, color: DARK_TEXT, bold: true, fontFace: "Arial",
      align: "center", valign: "middle", margin: 0
    });
    slide.addText(d.gain, {
      x: COL2_X + 4.6, y: rowY, w: 1.0, h: 0.22,
      fontSize: 10.5, color: ACCENT_RED, bold: true, fontFace: "Arial Black",
      align: "center", valign: "middle", margin: 0
    });
  }
  cy += 0.28 + tableData.length * 0.22 + 0.06;

  // WEARABLE CALLOUT
  slide.addShape(pres.shapes.RECTANGLE, {
    x: COL2_X + 0.05, y: cy, w: COL_W - 0.1, h: 0.38,
    fill: { color: ACCENT_RED }
  });
  slide.addText("⚠️  WEARABLE SIGNALS ADDED ZERO PREDICTIVE VALUE", {
    x: COL2_X + 0.05, y: cy, w: COL_W - 0.1, h: 0.38,
    fontSize: 12, color: WHITE, bold: true,
    align: "center", valign: "middle", fontFace: "Arial Black"
  });
  slide.addText("Set A ≈ Set C (within 0.003 AUPRC)  |  Set B worse than baseline on every target", {
    x: COL2_X + 0.05, y: cy + 0.38, w: COL_W - 0.1, h: 0.24,
    fontSize: 9.5, color: ACCENT_RED, bold: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });

  // ============================================================
  // RIGHT COLUMN
  // ============================================================
  let ry = CONTENT_TOP;

  // CLINICAL IMPACT
  ry = addSectionHeader(COL3_X, ry, COL_W, "Clinical Impact", "🏥");
  ry += 0.04;

  ry = addFigureImage(COL3_X + 0.05, ry, COL_W - 0.1, 1.80,
    "./phase3b_test_evaluation/Fig24_clinical_operating_points.png",
    "Fig 5. Clinical operating points: daily alert breakdown and precision vs recall tradeoff.");
  ry += 0.04;

  // Clinical highlight
  slide.addShape(pres.shapes.RECTANGLE, {
    x: COL3_X + 0.05, y: ry, w: COL_W - 0.1, h: 0.58,
    fill: { color: ACCENT_TEAL }
  });
  slide.addText([
    { text: "At the 60-minute urination horizon:\n", options: { bold: true, fontSize: CALLOUT_TITLE_SIZE } },
    { text: "The model captures 72% of events with a 43% false alert rate — ~25 true alerts and 18 false alerts per day.", options: { bold: true, fontSize: CALLOUT_SIZE } }
  ], {
    x: COL3_X + 0.15, y: ry + 0.03, w: COL_W - 0.3, h: 0.52,
    color: WHITE, fontFace: "Arial", valign: "middle",
    lineSpacingMultiple: 1.2
  });
  ry += 0.66;

  // WHAT DRIVES PREDICTIONS
  ry = addSectionHeader(COL3_X, ry, COL_W, "What Drives the Predictions?", "⭐");
  ry += 0.04;

  // Fig21 SHAP beeswarm — FULL WIDTH (was half-width before)
  ry = addFigureImage(COL3_X + 0.05, ry, COL_W - 0.1, 1.80,
    "./phase3_modeling/Fig21_shap_beeswarm_plot.png",
    "Fig 6. SHAP beeswarm — how each feature's value pushes predictions up or down.");
  ry += 0.02;

  // Top 4 features list
  const featureItems = [
    { label: "Hour of Day", gain: "313K gain" },
    { label: "Minutes Since Last Urination", gain: "244K gain" },
    { label: "Minutes Until Sleep", gain: "229K gain" },
    { label: "Cumulative Fluid Intake", gain: "101K gain" }
  ];

  slide.addShape(pres.shapes.RECTANGLE, {
    x: COL3_X + 0.05, y: ry, w: COL_W - 0.1, h: 1.30,
    fill: { color: WHITE },
    line: { color: ACCENT_TEAL, width: 1.5 }
  });

  slide.addText("TOP 4 PREDICTIVE FEATURES", {
    x: COL3_X + 0.15, y: ry + 0.02, w: COL_W - 0.3, h: 0.26,
    fontSize: 13, color: DARK_NAVY, bold: true,
    align: "center", valign: "middle", fontFace: "Arial Black"
  });

  for (let i = 0; i < featureItems.length; i++) {
    const f = featureItems[i];
    const fy = ry + 0.30 + i * 0.24;

    slide.addShape(pres.shapes.OVAL, {
      x: COL3_X + 0.18, y: fy + 0.02, w: 0.22, h: 0.22,
      fill: { color: ACCENT_TEAL }
    });
    slide.addText(String(i + 1), {
      x: COL3_X + 0.18, y: fy + 0.02, w: 0.22, h: 0.22,
      fontSize: 10, color: WHITE, bold: true,
      align: "center", valign: "middle", fontFace: "Arial Black"
    });
    slide.addText(f.label, {
      x: COL3_X + 0.48, y: fy, w: 3.6, h: 0.24,
      fontSize: FEATURE_LABEL_SIZE, color: DARK_TEXT, bold: true,
      align: "left", valign: "middle", fontFace: "Arial Black", margin: 0
    });
    slide.addText(f.gain, {
      x: COL3_X + 4.1, y: fy, w: 1.8, h: 0.24,
      fontSize: 10, color: ACCENT_TEAL, bold: true,
      align: "left", valign: "middle", fontFace: "Arial Black", margin: 0
    });
  }

  slide.addText("All four trackable by a caregiver or simple mobile app. No wearable required.", {
    x: COL3_X + 0.15, y: ry + 1.18, w: COL_W - 0.3, h: 0.18,
    fontSize: 9, color: ACCENT_RED, bold: true, italic: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });
  ry += 1.36;

  // CONCLUSIONS
  ry = addSectionHeader(COL3_X, ry, COL_W, "Conclusions", "✅");
  ry += 0.06;

  const conclusions = [
    "Personalized toileting prediction is feasible using five years of caregiver-logged data, with AUPRC gains of 36–116% over baseline.",
    "Wearable technology is not required. Schedule, intake, and void history alone achieve the best performance.",
    "Top four predictive features are all caregiver-trackable — accessible in resource-limited settings.",
    "At the 60-min horizon, the model captures 72% of urination events, outperforming fixed-schedule prompting."
  ];

  const iconCheck = await iconToBase64Png(FaCheckCircle, "#1A8A7D", 256);

  for (let i = 0; i < conclusions.length; i++) {
    const cly = ry + i * 0.42;

    slide.addImage({
      data: iconCheck,
      x: COL3_X + 0.1, y: cly + 0.04, w: 0.18, h: 0.18
    });
    slide.addText(conclusions[i], {
      x: COL3_X + 0.34, y: cly, w: COL_W - 0.44, h: 0.40,
      fontSize: CONCLUSION_SIZE, color: DARK_TEXT, bold: true, fontFace: "Arial",
      lineSpacingMultiple: 1.15, valign: "top", margin: 0
    });
  }
  ry += conclusions.length * 0.42 + 0.06;

  // Future work
  slide.addShape(pres.shapes.RECTANGLE, {
    x: COL3_X + 0.05, y: ry, w: COL_W - 0.1, h: 0.28,
    fill: { color: SUBTLE_BG },
    line: { color: LIGHT_BORDER, width: 0.5 }
  });
  slide.addText("Future Work:  Multi-participant replication  |  RL for alert optimization  |  Mobile app development", {
    x: COL3_X + 0.12, y: ry, w: COL_W - 0.24, h: 0.28,
    fontSize: 9, color: "555555", bold: true, italic: true,
    align: "left", valign: "middle", fontFace: "Arial"
  });

  // ============================================================
  // FOOTER (taller)
  // ============================================================
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: SH - 0.35, w: SW, h: 0.35,
    fill: { color: DARK_NAVY }
  });
  slide.addText("k.adesokan@northeastern.edu   |   Bouvé College of Health Sciences, Northeastern University   |   HINF 6215 Health Informatics Capstone   |   New England HIMSS 2026", {
    x: 0.3, y: SH - 0.35, w: SW - 0.6, h: 0.35,
    fontSize: FOOTER_SIZE, color: "B0BDD4", bold: true,
    align: "center", valign: "middle", fontFace: "Arial"
  });

  await pres.writeFile({ fileName: "/Users/macbookpro/Desktop/CapData/poster_digital.pptx" });
  console.log("Digital poster created successfully!");
}

main().catch(err => { console.error(err); process.exit(1); });
