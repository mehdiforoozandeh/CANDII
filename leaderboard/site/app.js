/* CANDI imputation leaderboard — vanilla JS over the compiled leaderboard.json.
 * No framework, no chart library, no external requests. The registry travels inside the
 * payload, so nothing about a metric is hard-coded here (LEADERBOARD_PRD.md §5.1). */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
/* The container that is not a regime: the 2019 field, lifted out of the ranked table
 * (BENCHMARK_DESIGN §6). It always sorts last and never ranks beside a regime. */
const ANCHOR_ID = "anchor";
const HEAD_ORDER = ["count", "pval", "peak"];
const HEAD_LABEL = { count: "Count", pval: "P-value", peak: "Peak" };
/* Covariate diagnostics have arm=null in the registry. They sit under Count
 * because `harness.c_block(..., kind=kinds[0])` predicts NB (mu, n) against
 * count truth (`_predictor`, `_c_contexts`). kinds[0] defaults to "impute". */
const COVARIATE_HEAD = "count";
const PVAL_RADAR_EDGES = ["pointwise", "distributional", "peaks"];
const COUNT_RADAR_EDGES = ["count_arm"];
const EN_DASH = "–";
const LINEAGE_LABEL = {
  candi: "CANDI", rival: "retrained rival",
  baseline: "baseline", entrant: "2019 entrant",
};
/* Table names that collide across containers. Slugs stay; the gloss is the label. */
const METHOD_QUALIFIER = {
  avg: "regime baseline · training cells only",
  Average: "2019 anchor · train + validation",
};

/* §7 — the spread device, and its mandatory badge. σ is fit on training-set residuals only:
 * never on V_, never on B_. Every σ that was fit on V_ eval pairs is void under Rule 1 and is
 * being refit (§12.2), so no PIT or coverage figure is quoted here until the refit lands. */
const DIST_ONELINER = "Point-only methods have no distribution, so their prediction is wrapped in a Gaussian whose width σ is measured on training-set residuals only — never on V_, never on B_. A flat bell cannot track a varying truth, so a point-only method loses on PIT and coverage partly by construction. Every distributional cell carries its device badge, native heteroscedastic or fitted flat σ, and that badge is what stops the loss being read as a modelling result. CRPS here is ordering only.";
const DIST_ELI5 = "Most rivals predict one number per bin, not a distribution. To give them CRPS, PIT, and 95% coverage we wrap a Gaussian around that number. The width is one constant per assay, measured on residuals from the method's own training set — never on the validation or test panels, which would leak under Rule 1. In-sample training residuals run narrow, and the overconfidence that follows is the method's own calibration failure and should show up as one. Same width at every bin of that assay: the spread does not grow or shrink with the signal. Rank on CRPS; do not read the absolute as a calibrated score. CANDI's Negative Binomial count head and per-bin Gaussian signal head emit a real distribution and do not use this device. A σ-table on the row means the fitted spread; PIT and coverage without a σ-table mean the method emitted its own. Two σs must not be confused: the marginal σ is the spread of the assay's signal itself and is the marginal baseline; the residual σ is the spread of prediction-minus-truth and is what wraps a point prediction. PIT KS is the Kolmogorov–Smirnov distance of the probability-integral transform from Uniform(0,1) — a calibration companion, never ranked, not a p-value. Coverage 95 is the fraction of bins inside the central 95% predictive interval. Click a column ? for each formula this code computes.";
const PEAK_ONELINER = "The peak arm carries CANDI and the naive baselines only. No rival has a peak head, and the AUPRC they used to carry was a coverage ranking derived from their predicted signal — dropped rather than badged (§7). Under challenge truth the whole arm is greyed out: the 2019 data has no peak calls.";
const PEAK_ELI5 = "A peak score asks whether the positions a method treats as peaks are the positions the truth calls as peaks. Only CANDI and the naive baselines emit one, so only they appear on this arm. Avocado, eDICE, ChromImpute and Lavawizard predict signal and nothing else; ranking their signal against called peaks produces an AUPRC that measures coverage, not peak detection, and §7 drops it rather than badging it. AUPRC is always shown with the peak base rate. Under challenge truth the arm is greyed out entirely: the 2019 data has no peak calls. Click a column ? for the average-precision formula this code computes.";
const POINTWISE_ELI5 = "Four bin-by-bin scores on the concatenation of scored chromosomes. MSE: mean squared gap (lower; scales with mark range; no extra transform). GW Pearson: linear correlation of predicted and true (higher; a constant forecast is absent, not zero). GW Spearman: rank-order correlation, average ranks on ties (higher). MSE top-1% obs: MSE on bins at or above the 1% tallest observed value (ties can admit more than 1%). The same four names score both truths, and the truth toggle is the only honest way to move between them — a number measured against store truth is never rescaled into challenge space; that move costs 12–66% per-experiment error. Click a column ? for the formula this code computes.";
const POSITION_TIP = {
  "position-generalizing": "No genomic-position parameters, or none that pin the eval chromosomes. A genome-wide cell can still be in-sample for other reasons — for example a neighbour table fit on the training chromosome.",
  "position-transductive": "The method fits parameters at genomic positions, including the positions it is scored at when the scope is genome-wide. §4 blanks that cell rather than printing a memorisation score.",
  "position class unrecorded": "Position class is not recorded in this tree. Never invent it.",
};
const LINEAGE_TIP = {
  candi: "A CANDI version. One CANDI row per regime — the current best. Version-over-version arrows use the 0.1195 seed-alone bar, not the cross-method ±0.09 floor.",
  rival: "Retrained by us under this regime's training loci, not a 2019 submission.",
  baseline: "A naive baseline we built from training biosamples. Its fit does not depend on the training loci, so one fit serves both regimes and the same number prints in each.",
  entrant: "A frozen 2019 challenge submission, not retrained. It carries no regime and sits in the anchor block, never in the ranked table. Architecture is not recorded unless the card says otherwise.",
};

const state = {
  data: null,
  help: { methods: {}, combos: {}, metrics: {} },
  outerEval: null,
  /* §1's address, minus the method and the metric. The page opens on the ranked address and
   * cannot show a number until all three are set. */
  truth: "store",
  panel: "V_breadth",
  scope: "held-out",
  midHead: null,
  innerFamily: null,
  radarEval: null,
  climbEval: null,
  showClimb: false,
  showVoid: false,
  openProv: new Set(),
  openHelp: null,
  openCard: null,
};

/* ------------------------------------------------------------- helpers --- */

function h(tag, attrs, ...children) {
  const el = tag.startsWith("svg:")
    ? document.createElementNS(SVG_NS, tag.slice(4))
    : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "onclick") el.addEventListener("click", v);
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

function metricSlot(m) { return (m.arm === "pval" || m.arm === "count") ? m.arm : "diagnostics"; }
function metricId(m) { return `${metricSlot(m)}/${m.key}`; }
function registry() { return state.data.registry; }
function catMetrics(cid, role) {
  return registry().metrics.filter((m) =>
    m.category === cid && (!role || role.includes(m.role)));
}
function fmt(v, m) { return v === null || v === undefined ? null : v.toFixed(m.decimals); }
function rowVal(row, m) {
  const block = row.metrics[metricSlot(m)];
  return block && m.key in block ? block[m.key] : null;
}
function spreadText(spread) {
  return spread[0] === spread[1] ? String(spread[0]) : `${spread[0]}${EN_DASH}${spread[1]}`;
}
function catInfo(cid) { return registry().categories[cid]; }

function headIdOf(m) {
  if (m.category === "peaks") return "peak";
  if (m.category === "covariate_diagnostics") return COVARIATE_HEAD;
  if (m.arm === "count") return "count";
  if (m.arm === "pval") return "pval";
  return null;
}

function derivedHeads() {
  const cats = registry().categories;
  const byId = {};
  for (const hid of HEAD_ORDER) {
    const space = hid === "peak"
      ? ((cats.peaks && cats.peaks.space_label) || "p-value space")
      : hid === "count" ? "count space" : "p-value space";
    byId[hid] = { id: hid, label: HEAD_LABEL[hid], space, families: [] };
  }
  const seen = { count: new Set(), pval: new Set(), peak: new Set() };
  for (const m of registry().metrics) {
    const hid = headIdOf(m);
    if (!hid || !byId[hid] || seen[hid].has(m.category)) continue;
    if (!(m.category in cats)) continue;
    seen[hid].add(m.category);
    byId[hid].families.push(m.category);
  }
  return HEAD_ORDER.map((id) => byId[id]).filter((h) => h.families.length);
}

function overviewMetrics(headId) {
  return registry().metrics.filter((m) =>
    headIdOf(m) === headId
    && m.category !== "loss"
    && m.category !== "covariate_diagnostics"
    && (m.role === "ranked" || m.role === "companion"));
}

function overviewGroups(headId) {
  const byCat = [];
  const seen = new Set();
  for (const m of overviewMetrics(headId)) {
    if (seen.has(m.category)) continue;
    seen.add(m.category);
    byCat.push({
      arm: m.arm, cid: m.category,
      metrics: overviewMetrics(headId).filter((x) => x.category === m.category),
    });
  }
  return byCat;
}

function rankCidForHead(headId) {
  if (headId === "pval") return "summary";
  if (headId === "count") return "count_arm";
  return "peaks";
}
function coded(label, code) {
  return h("span", { title: `Internal code: ${code}` }, label);
}
function helpBtn(id, text) {
  if (!text) return null;
  const open = state.openHelp === id;
  const rich = !!(text && text.nodeType);
  return h("button", {
    class: "help", type: "button",
    "aria-label": "Explain this", "aria-expanded": String(open),
    onclick: (ev) => {
      ev.stopPropagation();
      state.openHelp = open ? null : id;
      render();
    },
  }, "?", open ? h("span", {
    class: rich ? "help-tip help-tip-metric" : "help-tip", role: "tooltip",
  }, text) : null);
}
function mathNode(mathml) {
  const wrap = document.createElement("div");
  wrap.className = "help-math";
  wrap.innerHTML = mathml;
  return wrap;
}
function metricHelpBody(id) {
  const spec = (state.help.metrics || {})[id];
  if (!spec || !spec.question) return null;
  return h("span", { class: "help-metric" },
    h("p", { class: "help-q" }, spec.question),
    spec.formula_mathml ? mathNode(spec.formula_mathml) : null,
    spec.estimator_notes ? h("p", { class: "help-est" }, spec.estimator_notes) : null,
    spec.read_rules ? h("p", { class: "help-rules" }, spec.read_rules) : null);
}
function comboKey() {
  if (!state.outerEval) return null;
  if (state.midHead === "summary") return `${state.outerEval}/summary`;
  if (state.midHead === "radar") return `${state.outerEval}/radar`;
  if (!state.midHead) return null;
  if (!state.innerFamily) return null;
  return `${state.outerEval}/${state.midHead}/${state.innerFamily}`;
}
function comboComplete() {
  if (!state.outerEval) return false;
  if (state.midHead === "summary" || state.midHead === "radar") return true;
  return !!(state.midHead && state.innerFamily);
}
function methodLink(name) {
  return h("button", {
    class: "method-link", type: "button",
    "aria-label": `About ${name}`,
    onclick: (ev) => {
      ev.stopPropagation();
      state.openCard = { kind: "method", id: name };
      render();
    },
  }, name);
}
function comboHelpBtn() {
  const key = comboKey();
  if (!key) return null;
  return h("button", {
    class: "help", type: "button",
    "aria-label": "What this view scores",
    onclick: (ev) => {
      ev.stopPropagation();
      state.openCard = { kind: "combo", id: key };
      render();
    },
  }, "?");
}
/* The URL carries the whole address, so a link to a number is a link to that exact number
 * and to nothing else: #<regime>/<truth>.<panel>.<scope>/<head>/<family>. */
function writeHash() {
  let next = "";
  if (state.showClimb && !state.outerEval) next = "#over-time";
  else if (state.outerEval) {
    const segs = [state.outerEval, viewKeyFor(state.outerEval)];
    if (state.midHead) segs.push(state.midHead);
    if (state.midHead && state.midHead !== "summary" && state.midHead !== "radar"
        && state.innerFamily) {
      segs.push(state.innerFamily);
    }
    next = "#" + segs.join("/");
  }
  if (location.hash !== next) {
    history.replaceState(null, "", next || (location.pathname + location.search));
  }
}
function applyHash() {
  if (!state.data) return;
  const raw = decodeURIComponent((location.hash || "").replace(/^#/, "")).trim();
  const ids = Object.keys(state.data.boards);
  if (!raw) return;
  if (raw === "over-time") {
    state.showClimb = true;
    return;
  }
  const parts = raw.split("/").filter(Boolean);
  if (!ids.includes(parts[0])) return;
  state.outerEval = parts[0];
  let rest = parts.slice(1);
  /* An address segment is three dot-joined vocabulary words. Each is checked against the
   * payload's own vocabulary, so a hand-edited URL cannot invent a truth or a panel. */
  if (rest[0] && rest[0].split(".").length === 3) {
    const [t, p, s] = rest[0].split(".");
    if (t in truths() && p in panels() && s in scopes()) {
      state.truth = t;
      state.panel = p;
      state.scope = s;
    }
    rest = rest.slice(1);
  }
  if (rest[0] === "summary" || rest[0] === "radar") {
    state.midHead = rest[0];
    state.innerFamily = null;
    if (rest[0] === "radar") state.radarEval = parts[0];
    return;
  }
  if (rest[0] && HEAD_ORDER.includes(rest[0])) {
    state.midHead = rest[0];
    state.innerFamily = rest[1] || null;
  }
}
function armClass(m) {
  return m.arm === "count" ? "arm-count" : m.arm === "pval" ? "arm-pval" : "";
}
/* --- §1's address: truth, panel, scope. Three fields of the six; the container supplies
 * the regime, the row supplies the method, the column supplies the metric. --- */

function truths() { return state.data.truths || {}; }
function panels() { return state.data.panels || {}; }
function scopes() { return state.data.scopes || {}; }
function markerSpec(id) { return (state.data.markers || {})[id] || null; }

/* The anchor block exists at one fixed address and it is not the reader's to change; a
 * regime takes all three from the address bar. A null container falls back to the bar too. */
function metaOrNull(bid) {
  const board = bid && state.data.boards[bid];
  return board ? board.meta : null;
}
function truthOf(bid) {
  const meta = metaOrNull(bid);
  return (meta && meta.truth) || state.truth;
}
function panelOf(bid) {
  const meta = metaOrNull(bid);
  return (meta && meta.panel) || state.panel;
}
function scopeOf(bid) {
  const meta = metaOrNull(bid);
  return (meta && meta.scope) || state.scope;
}
function viewKeyFor(bid) {
  return `${truthOf(bid)}.${panelOf(bid)}.${scopeOf(bid)}`;
}
function truthLabel(bid) {
  const t = truths()[truthOf(bid)];
  return t ? t.label : truthOf(bid);
}
/* Which arms this truth can be scored on. Under challenge truth the 2019 data has no counts
 * and no peak calls, so those heads are greyed out rather than silently empty (§7). */
function truthArms(bid) {
  const t = truths()[truthOf(bid)];
  return (t && t.arms) || ["count", "pval", "peak"];
}
function headIsLive(bid, headId) {
  return truthArms(bid).includes(headId);
}
function headDeadReason(bid) {
  const t = truths()[truthOf(bid)];
  return (t && t.greyed_arms_note)
    || "This arm cannot be scored against the truth currently selected.";
}
function addressLine(bid) {
  const meta = metaOrNull(bid);
  const regime = meta && meta.regime_id ? meta.regime_id : "no regime (anchor)";
  const p = panels()[panelOf(bid)], s = scopes()[scopeOf(bid)];
  return `${regime} · truth: ${truthLabel(bid)} · panel: ${p ? p.label : panelOf(bid)}`
    + ` · scope: ${s ? s.label : scopeOf(bid)}`;
}

function truthSpace(bid) {
  return truthOf(bid) === "challenge" ? "challenge −log10 p" : "store −log10 p";
}
function spaceTag(m, bid) {
  let s = "diagnostic";
  if (m.arm === "count") s = "counts";
  else if (m.arm === "pval") s = truthSpace(bid);
  return h("span", { class: "space-tag" }, s);
}
function metricHeaderName(m, bid) {
  if (m.category === "pointwise") return `${m.label} (${truthSpace(bid)})`;
  return m.label;
}

/* §5 and §7 name two row markers. Neither is cosmetic: each says the row's number was
 * produced under a condition the other rows were not, and names what that costs a reader. */
function markerBadges(row) {
  const ids = (row && row.markers) || [];
  return ids.map((id) => {
    const spec = markerSpec(id);
    return h("span", { class: `badge badge-marker marker-${id}`,
      title: spec ? spec.eli5 : id },
      spec ? spec.label : id,
      helpBtn(`marker-${id}-${(row && row.method) || ""}`, spec ? spec.eli5 : null));
  });
}

function cellClassBadge(text) {
  if (!text) return null;
  if (text === "zero-shot cell types") {
    return h("span", { class: "badge",
      title: "Stored class: zero-shot cell types. In this vocabulary that means no learned per-cell embedding, not a cell type with no training experiment. Every eval pair on this board is a mark the target cell has and its paired training cell lacks, so the cell type is one the method has seen (§8)." },
      "no cell embedding");
  }
  return h("span", { class: "badge" }, text);
}

function hasSigmaTable(row) {
  return !!(row && row.provenance && row.provenance.sigma_table);
}
function hasPeakHead(row) {
  if (!row) return false;
  if (row.has_peak_head) return true;
  return !!(row.provenance && row.provenance.has_peak_head);
}
/* §7's mandatory spread badge. The STAMPER decides it: `add` writes badges.sigma from the score
 * json's provenance.sigma_table and refuses a table not fitted on training residuals, so a stamped
 * row is read rather than re-derived. A row stamped before the badge existed falls back to the
 * same rule applied here. */
function spreadDevice(row) {
  if (!row) return null;
  const stamped = row.badges && row.badges.sigma;
  if (stamped) return stamped.indexOf("fitted") === 0 ? "fitted" : "native";
  if (hasSigmaTable(row)) return "fitted";
  const pval = (row.metrics && row.metrics.pval) || {};
  if ("pit_ks" in pval || "coverage_95" in pval) return "native";
  return null;
}
function peakDevice(row) {
  if (!row) return null;
  const pval = (row.metrics && row.metrics.pval) || {};
  if (!("auprc" in pval)) return null;
  return hasPeakHead(row) ? "native" : "coverage";
}
function deviceBadge(kind, device) {
  if (!device) return null;
  if (kind === "spread") {
    return device === "fitted"
      ? h("span", { class: "badge badge-device badge-fitted",
          title: "Gaussian wrapped around a point prediction. One spread per assay, fitted on TRAINING-set residuals (§7 — never on V_, never on B_), reused unchanged. Homoscedastic per assay — the spread does not adapt bin-by-bin, so this method loses on PIT and coverage partly by construction." },
          "fitted flat σ")
      : h("span", { class: "badge badge-device badge-native",
          title: "This method emitted its own per-bin spread. Not the fitted σ-table device." },
          "native heteroscedastic");
  }
  return device === "native"
    ? h("span", { class: "badge badge-device badge-native",
        title: "Producer supplied a peak_score. bernoulli_nll is defined. A Bernoulli peak head, not a coverage ranking." },
        "native peak head")
    : h("span", { class: "badge badge-device badge-fitted",
        title: "No peak_score array. AUPRC ranks bins by predicted signal (or count mean) against called peaks — a coverage ranking, not a peak classifier." },
        "coverage ranking");
}
function deviceNote(cid, row) {
  if (cid === "distributional") {
    const d = spreadDevice(row);
    return d === "fitted" ? "fitted spread" : d === "native" ? "native dist" : "";
  }
  if (cid === "peaks") {
    const d = peakDevice(row);
    return d === "native" ? "native peak" : d === "coverage" ? "coverage ranking" : "";
  }
  return "";
}
function metricHelpBtn(m, id) {
  const mid = metricId(m);
  const spec = (state.help.metrics || {})[mid];
  if (!spec && !metricExplainer(m)) return null;
  return h("button", {
    class: "help", type: "button",
    "aria-label": `Explain ${m.label}`,
    title: (spec && spec.question) || m.label,
    onclick: (ev) => {
      ev.stopPropagation();
      state.openCard = { kind: "metric", id: mid };
      render();
    },
  }, "?");
}
function metricCardBody(id) {
  const spec = (state.help.metrics || {})[id];
  if (!spec || !spec.question) {
    return h("p", null, "No help card recorded for this metric.");
  }
  return h("div", null,
    cardField("What this number asks", spec.question),
    spec.formula_mathml
      ? h("div", null,
          h("div", { class: "card-label" }, "How this code computes it"),
          mathNode(spec.formula_mathml))
      : null,
    cardField("Estimator notes", spec.estimator_notes),
    cardField("How to read it", spec.read_rules));
}
function metricExplainer(m) {
  const body = metricHelpBody(metricId(m));
  if (body) return body;
  if (m.arm === "pval" && (m.key === "crps" || m.key === "pit_ks" || m.key === "coverage_95")) {
    return DIST_ELI5;
  }
  if (m.key === "auprc" || m.key === "peak_base_rate") return PEAK_ELI5;
  return m.eli5 || m.note || metricTitle(m);
}

/* ------------------------------------------------------------- boot --- */

async function boot() {
  const app = document.getElementById("app");
  let payload;
  try {
    payload = await (await fetch("leaderboard.json")).json();
  } catch (err) {
    app.replaceChildren(h("p", { class: "loading" },
      "Could not load leaderboard.json — serve this directory over HTTP ",
      "(python -m http.server) or open the deployed page."));
    return;
  }
  state.data = payload;
  try {
    state.help = await (await fetch("help.json")).json();
  } catch (err) {
    state.help = { methods: {}, combos: {}, metrics: {} };
  }
  /* Open on the ranked address the payload names, not on a guess. */
  const canon = (payload.canonical_view || "store.V_breadth.held-out").split(".");
  if (canon.length === 3) {
    state.truth = canon[0];
    state.panel = canon[1];
    state.scope = canon[2];
  }
  const ids = boardIds();
  if (!(state.radarEval in payload.boards)) state.radarEval = ids[0];
  if (!(state.climbEval in payload.boards)) state.climbEval = ids[0];
  applyHash();
  window.addEventListener("hashchange", () => { applyHash(); render(); });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && state.openCard) {
      state.openCard = null;
      render();
    }
  });
  render();
}

/* Regimes only, in id order. The anchor block is a container too, but it is never a tab
 * beside a regime — it renders underneath the ranked table (§6). */
function boardIds() {
  return Object.keys(state.data.boards)
    .filter((id) => state.data.boards[id].kind !== "anchor")
    .sort((a, b) => a.localeCompare(b));
}
function anchorId() {
  const id = state.data.anchor_id;
  return id && state.data.boards[id] ? id : null;
}

function viewFor(bid) {
  const board = state.data.boards[bid];
  const key = viewKeyFor(bid);
  /* An address a container does not offer is not an error state — it is a reader asking for
   * a cell that does not exist. Fall back to the container's own canonical address. */
  return board.views[key] || board.views[board.canonical_view];
}
function viewExists(bid) {
  return !!state.data.boards[bid].views[viewKeyFor(bid)];
}
function rankingOf(bid) {
  const v = viewFor(bid);
  return (v && v.ranking) || { state: "ranked", reason: "" };
}
function isRanked(bid) { return rankingOf(bid).state === "ranked"; }

function metaOf(bid) { return state.data.boards[bid].meta; }
function pendingOf(bid) { return state.data.boards[bid].pending || []; }

function metricGroups(cid, headId) {
  let all = catMetrics(cid);
  if (headId === "count") all = all.filter((m) => m.arm === "count" || m.arm == null);
  else if (headId === "pval") all = all.filter((m) => m.arm === "pval");
  else if (headId === "peak") all = all.filter((m) => m.category === "peaks");
  if (cid !== "loss") return [{ arm: all[0] ? all[0].arm : null, metrics: all, cid }];
  const pval = all.filter((m) => m.arm === "pval");
  const count = all.filter((m) => m.arm === "count");
  return [
    pval.length ? { arm: "pval", metrics: pval, cid } : null,
    count.length ? { arm: "count", metrics: count, cid } : null,
  ].filter(Boolean);
}

/* ------------------------------------------------------------- render --- */

function render() {
  const app = document.getElementById("app");
  const nodes = [picker()];
  if (comboComplete()) {
    nodes.push(comboView());
    nodes.push(anchorPanel());
    nodes.push(legendPanel());
  }
  nodes.push(state.showClimb ? climbPanel() : climbOptIn());
  nodes.push(voidPanel());
  const overlay = cardOverlay();
  if (overlay) nodes.push(overlay);
  app.replaceChildren(...nodes.filter(Boolean));
  renderFooter();
}

function readerBadges(meta) {
  return (meta.reader_badges || []).map((b, i) =>
    h("span", { class: "badge badge-warn", title: b.eli5 },
      b.label, helpBtn(`badge-${meta.regime_id || meta.label}-${i}`, b.eli5)));
}

function familyTabCaption(cid, bid) {
  if (cid === "summary") return { title: "Summary", sub: "composite rank" };
  const c = catInfo(cid);
  if (!c) return { title: cid, sub: "" };
  let sub = (c.space_label && c.space_label.toLowerCase() !== c.label.toLowerCase())
    ? c.space_label : "";
  if (bid && cid === "pointwise") sub = truthSpace(bid);
  return { title: c.label, sub };
}

function familyEli5(cid) {
  if (cid === "summary") {
    return "Composite is the mean of category mean-ranks for categories at least one method at this address fully covers. A method missing any of those categories is incomplete, not last: the composite cell is a dash, and it is left out of the headline ranking. It still ranks inside every category it has numbers for. Lower is better. Nothing is ranked at all until the noise floor on this panel is measured.";
  }
  if (cid === "distributional") return DIST_ELI5;
  if (cid === "peaks") return PEAK_ELI5;
  if (cid === "pointwise") return POINTWISE_ELI5;
  if (cid === "loss") return LOSS_ELI5;
  const c = catInfo(cid);
  return c && c.eli5;
}

function headEli5(head, bid) {
  if (head.id === "pval") {
    const t = truthOf(bid) === "challenge"
      ? "Challenge truth: the 2019 organizers' processing of the same experiments, loaded onto the store's grid by the same scorer, so only the pipeline differs."
      : "Store truth: −log10 p from CANDI_STORE, the ENCODE4 v1.5.1 processing of Aug–Sep 2020.";
    return `${t} This is the head-to-head arm — every rival predicts signal, so it is the only arm where CANDI meets the published field. Ranking uses the §5.2 composite. Count numbers never appear here. The same metric names score both truths; a number is never rescaled from one into the other.`;
  }
  if (head.id === "count") {
    return "Count space — CANDI and the naive baselines only, because no rival has a count head (§7). Ranking is Negative Binomial CRPS, always with its oracle-scaled / scale-error split and the ±0.09 noise floor. Covariate sensitivity sits here because the C-block re-decodes (μ, n) against count truth. P-value numbers never appear here, and under challenge truth the whole arm is greyed out.";
  }
  return PEAK_ELI5;
}
/* ------------------------------------------------------------- nested tabs --- */

function pick(patch) {
  Object.assign(state, patch);
  writeHash();
  render();
}

function headSummaryCaption(headId) {
  if (headId === "pval") return { title: "Summary", sub: "composite rank" };
  if (headId === "count") return { title: "Summary", sub: "NB CRPS rank" };
  return { title: "Summary", sub: "AUPRC rank" };
}

/* The address bar. It is not a filter over one table — it *is* the table's address, and
 * every one of its three fields must be set before a number is on screen (§1). */
function addressBar(bid) {
  const seg = (label, vocab, key, current, onpick, tip) => {
    const opts = Object.keys(vocab);
    return h("span", { class: "addr-seg" },
      h("span", { class: "addr-label" }, label),
      ...opts.map((id) => {
        const spec = vocab[id];
        const on = id === current;
        return h("button", {
          class: "chip addr-chip" + (on ? " addr-on" : "")
            + (spec.ranked === false ? " addr-unranked" : ""),
          type: "button",
          "aria-pressed": String(on),
          title: spec.eli5 || spec.label || id,
          onclick: () => onpick(id),
        }, spec.label || id,
          spec.subtitle ? h("span", { class: "addr-sub" }, spec.subtitle) : null);
      }),
      helpBtn(`addr-${key}-${bid}`, tip));
  };
  const truthTip = "§6 — the challenge data is not a separate board. It is the same rows, "
    + "measured a second time: the same 363 ENCODE experiments, bridged 1:1 on accession, and "
    + "the only difference is which pipeline produced the truth track. If a method's standing "
    + "holds under both truths, its result is not an artifact of the pipeline that made ours. "
    + "Nowhere else in this field does the same experiment exist twice, processed independently "
    + "by different people years apart. Challenge truth exists on EIC regimes only, and it "
    + "greys out the count and peak arms because the 2019 data has no counts and no peak calls.";
  const panelTip = "§5 — V_ and B_ are different exams, not two views of one thing. Every "
    + "trainable method selected its best checkpoint on V_, so V_ is optimistic by "
    + "construction, identically for everyone. B_ is never read during training and is "
    + "predicted exactly once. V_ matched is the same V_ predictions aggregated over only the "
    + "8 assays B_ has; it is never ranked and exists so the V_→B_ step is readable. Never "
    + "subtract V_ breadth from B_.";
  const scopeTip = "§4 — predictions are produced and scored once, and reported as two "
    + "aggregations of the same pass. held-out is chr20+21+22, where no method's transferable "
    + "parameters were fit: that is the ranked number. genome-wide is all 23 chromosomes, for "
    + "comparability with a literature that scores that way, and is never ranked. A "
    + "genome-wide cell is printed only where the method is out-of-sample somewhere; where it "
    + "is blank the number was never computed, not withheld.";
  return h("div", { class: "addr-bar", id: "address-bar",
      "aria-label": "Address: truth, panel, scope" },
    seg("truth", truths(), "truth", truthOf(bid), (id) => pick({ truth: id }), truthTip),
    seg("panel", panels(), "panel", panelOf(bid), (id) => pick({ panel: id }), panelTip),
    seg("scope", scopes(), "scope", scopeOf(bid), (id) => pick({ scope: id }), scopeTip));
}

function picker() {
  const ids = boardIds();
  if (state.outerEval && !ids.includes(state.outerEval)) {
    state.outerEval = null;
    state.midHead = null;
    state.innerFamily = null;
  }
  const bid = state.outerEval;
  const heads = derivedHeads();
  if (state.midHead && state.midHead !== "summary" && state.midHead !== "radar"
      && !heads.some((hd) => hd.id === state.midHead)) {
    state.midHead = null;
    state.innerFamily = null;
  }
  /* An arm the current truth cannot score is not a tab the reader gets to sit on. */
  if (bid && state.midHead && HEAD_ORDER.includes(state.midHead)
      && !headIsLive(bid, state.midHead)) {
    state.midHead = "summary";
    state.innerFamily = null;
  }
  const head = heads.find((hd) => hd.id === state.midHead) || null;
  const families = head ? head.families : [];
  if (state.innerFamily && state.innerFamily !== "summary"
      && !families.includes(state.innerFamily)) {
    state.innerFamily = null;
  }
  const dataReady = !!bid;
  const headReady = dataReady && state.midHead && state.midHead !== "summary"
    && state.midHead !== "radar";

  const outer = h("div", { class: "tabs outer", id: "eval-tabs", role: "tablist",
    "aria-label": "Regime" },
    ids.map((id) => {
      const m = metaOf(id);
      return h("button", {
        class: "tab outer", type: "button", role: "tab",
        id: `eval-tab-${id}`,
        "aria-selected": String(id === bid),
        "aria-controls": "eval-board",
        title: `${m.eli5} Trained on ${m.train_scope}. Scored on ${m.eval_scope}.`,
        onclick: () => pick({ outerEval: id, radarEval: id }),
      }, m.label,
        h("span", { class: "tab-sub" }, m.subtitle || ""));
    }));

  const midOpts = [
    { id: "summary", label: "Summary", sub: "composite rank", eli5: familyEli5("summary"),
      live: true },
    ...heads.map((hd) => ({
      id: hd.id, label: hd.label,
      sub: (bid && hd.id === "pval") ? truthSpace(bid) : hd.space,
      eli5: headEli5(hd, bid),
      live: !bid || headIsLive(bid, hd.id),
    })),
    { id: "radar", label: "Shape", sub: "per-method radar", live: true,
      eli5: "P-value-space categories share one polygon; count-space rank is a separate figure beside it. Rank 1 is outer (or the top of the count bar); last place is the centre. Ranks are within one address only. Count-space and p-value-space numbers never share an axis." },
  ];
  const mid = h("div", { class: "tabs mid", id: "head-tabs", role: "tablist",
    "aria-label": "Output head" },
    midOpts.map((opt) => {
      const usable = dataReady && opt.live;
      return h("span", { class: "tab-wrap" },
        h("button", {
          class: "tab mid" + (opt.live ? "" : " tab-greyed"), type: "button", role: "tab",
          id: `head-tab-${opt.id}`,
          "aria-selected": String(usable && opt.id === state.midHead),
          "aria-controls": "head-panel",
          "aria-disabled": String(!usable),
          disabled: usable ? null : "disabled",
          title: opt.live ? null : headDeadReason(bid),
          onclick: () => {
            if (!usable) return;
            const keep = (opt.id !== "summary" && opt.id !== "radar"
              && state.innerFamily && (state.innerFamily === "summary"
                || (heads.find((hd) => hd.id === opt.id) || { families: [] })
                  .families.includes(state.innerFamily)))
              ? state.innerFamily : null;
            pick({ midHead: opt.id, innerFamily: keep,
                   radarEval: opt.id === "radar" ? state.outerEval : state.radarEval });
          },
        }, opt.label,
          h("span", { class: "tab-sub" }, opt.live ? opt.sub : "greyed out")),
        helpBtn(`head-${opt.id}`, opt.live ? opt.eli5 : headDeadReason(bid)));
    }));

  const innerOpts = headReady
    ? [{ id: "summary", ...headSummaryCaption(head.id) },
       ...families.map((cid) => ({ id: cid, ...familyTabCaption(cid, bid) }))]
    : [];
  const inner = headReady
    ? h("div", { class: "tabs inner", id: "family-tabs", role: "tablist",
        "aria-label": "Metric family" },
        innerOpts.map((opt) =>
          h("span", { class: "tab-wrap" },
            h("button", {
              class: "tab inner", type: "button", role: "tab",
              id: `fam-tab-${opt.id}`,
              "aria-selected": String(opt.id === state.innerFamily),
              "aria-controls": "family-panel",
              onclick: () => pick({ innerFamily: opt.id }),
            }, opt.title,
              opt.sub ? h("span", { class: "tab-sub" }, opt.sub) : null),
            helpBtn(`fam-${opt.id}`,
              opt.id === "summary" ? headEli5(head, bid) : familyEli5(opt.id)))))
    : h("div", { class: "tabs inner", id: "family-tabs", role: "tablist",
        "aria-label": "Metric family" },
        h("span", { class: "picker-placeholder" },
          dataReady
            ? (state.midHead === "summary" || state.midHead === "radar"
              ? "This pick is complete — a family is not needed."
              : "Pick Count, P-value, or Peak to choose a family — or Summary / Shape above.")
            : "Pick a regime first."));

  const deferred = (state.data.deferred_regimes || []);
  return h("div", { class: "nested-wrap picker" },
    h("p", { class: "picker-hint" },
      "A number on this page is addressed by six fields (§1): method, regime, truth, panel, "
      + "scope, metric. Pick the regime, then the truth, panel and scope, then a head and a "
      + "family. A row missing any of the six never enters the ranked table."),
    outer,
    dataReady ? addressBar(bid) : null,
    dataReady ? h("p", { class: "addr-readout" }, "Showing: ", addressLine(bid)) : null,
    mid, inner,
    deferred.length
      ? h("p", { class: "picker-deferred" },
          "Deferred regimes, named so the axis exists but not yet run: ",
          ...deferred.flatMap((d, i) => [
            i ? " · " : null,
            h("span", { class: "chip-wrap" }, h("code", null, d.id),
              helpBtn(`deferred-${i}`, d.note)),
          ]))
      : null);
}
/* "Here is a number, and it is not ranked" is a state of its own, not an error and not an
 * empty table (§15). It is announced once, at the top, so no reader can miss it. */
function unrankedBanner(bid) {
  if (isRanked(bid)) return null;
  const why = rankingOf(bid).reason;
  return h("div", { class: "unranked-banner", role: "note" },
    h("span", { class: "badge badge-unranked" }, "unranked"),
    h("span", null, " Numbers at this address are reported, not ordered. ", why),
    helpBtn(`unranked-${bid}`,
      "A rank is a claim that one method beat another. That claim needs a resolution band, "
      + "and until the band is measured the honest output is the number without an order. "
      + "Rows go up unranked; nothing is hidden and nothing is ordered on a gap we cannot "
      + "resolve. " + why));
}

function comboView() {
  const bid = state.outerEval;
  const meta = metaOf(bid);
  const heads = derivedHeads();
  const head = heads.find((hd) => hd.id === state.midHead) || null;
  let body;
  if (state.midHead === "summary") {
    body = h("div", { class: "data-summary", id: "data-summary" }, summaryBody(bid));
  } else if (state.midHead === "radar") {
    body = radarPanel();
  } else if (state.innerFamily === "summary") {
    body = h("div", { class: "head-summary" }, headSummaryBody(bid, head));
  } else {
    body = h("div", { id: "family-panel", role: "tabpanel",
      "aria-labelledby": `fam-tab-${state.innerFamily}` },
      familyBody(bid, state.innerFamily, head.id));
  }
  const scopeSpec = scopes()[scopeOf(bid)];
  return h("section", { class: "panel eval-board", id: "eval-board", role: "tabpanel",
    "aria-labelledby": `eval-tab-${bid}` },
    h("h2", null,
      meta.label,
      h("span", { class: "h2-note" }, meta.subtitle),
      helpBtn(`h2-${bid}`, meta.eli5),
      comboHelpBtn(),
      ...readerBadges(meta)),
    h("p", { class: "addr-readout" }, addressLine(bid)),
    unrankedBanner(bid),
    scopeSpec && scopeSpec.blanking_rule
      ? h("p", { class: "sub" }, scopeSpec.blanking_rule)
      : null,
    (meta.caveats || []).length
      ? h("div", { class: "caveats" },
          h("div", { class: "caveats-title" }, "On this regime"),
          h("ul", { style: "margin:4px 0;padding:0" },
            meta.caveats.map((c, i) =>
              h("li", null, c, " ", helpBtn(`tabcav-${bid}-${i}`,
                `${c} This line is a property of ${meta.label}, not of one method. ${meta.eli5}`)))))
      : null,
    body);
}

function familyBody(bid, cid, headId) {
  if (cid === "covariate_diagnostics") return covariateBody(bid);
  if (cid === "loss") return lossBody(bid, headId);
  return familyChartAndTable(bid, cid, headId);
}

function rowHasFamily(row, cid) {
  if (!row || !row.metrics) return false;
  if (cid === "summary") return true;
  if (cid === "covariate_diagnostics") {
    const d = row.metrics.diagnostics;
    return !!(d && Object.keys(d).length);
  }
  return catMetrics(cid).some((m) => rowVal(row, m) !== null);
}

function familyRankSpread(row, cid, view) {
  if (cid === "summary") return row.rank || null;
  if (cid === "count_arm") {
    const sub = ((view.sub_boards.count_arm && view.sub_boards.count_arm.rows) || [])
      .find((r) => r.id === row.id);
    return sub && sub.rank ? sub.rank : null;
  }
  if (row.category_subscores && row.category_subscores[cid]) {
    return row.category_subscores[cid];
  }
  const ranked = catMetrics(cid, ["ranked"]);
  const ranks = ranked.map((m) => row.metric_ranks && row.metric_ranks[metricId(m)]).filter(Boolean);
  if (!ranks.length) return null;
  return [
    ranks.reduce((a, rk) => a + rk[0], 0) / ranks.length,
    ranks.reduce((a, rk) => a + rk[1], 0) / ranks.length,
  ];
}

function boardEntries(bid) {
  const view = viewFor(bid);
  const byName = new Map();
  const ensure = (name, lineage) => {
    if (!byName.has(name)) {
      byName.set(name, { method: name, lineage: lineage || "",
                         row: null, pending: null, unscored: null });
    }
    const e = byName.get(name);
    if (lineage && !e.lineage) e.lineage = lineage;
    return e;
  };
  for (const row of view.rows) ensure(row.method, row.lineage).row = row;
  for (const u of view.unscored || []) ensure(u.method).unscored = u;
  for (const p of pendingOf(bid)) ensure(p.method, p.lineage).pending = p;
  const rankKey = (e) => {
    if (e.row && e.row.rank) return (e.row.rank[0] + e.row.rank[1]) / 2;
    if (e.pending && !e.row) return 1e6;
    return 1e5;
  };
  return [...byName.values()].sort((a, b) =>
    rankKey(a) - rankKey(b) || a.method.localeCompare(b.method));
}

function entriesForFamily(bid, cid) {
  return boardEntries(bid).filter((e) => {
    if (e.pending && !e.row) {
      if (cid === "covariate_diagnostics") return e.lineage === "candi";
      /* §7 — the count and peak arms carry CANDI and the naive baselines only. */
      if (cid === "count_arm" || cid === "peaks") {
        return e.lineage === "baseline" || e.lineage === "candi";
      }
      return true;
    }
    if (cid === "summary") return !!(e.row || e.unscored);
    return e.row && rowHasFamily(e.row, cid);
  });
}

function familyKicker(cid, bid) {
  if (cid === "summary") {
    return isRanked(bid)
      ? "Headline ranking on the p-value-space composite (best at top). Methods with partial coverage have no composite rank — incomplete, not last. Count-space ranking lives under the Count head; peak ranking lives under the Peak head."
      : "Every method with a number at this address, in name order. There is no headline ranking here yet: see the unranked note above for why.";
  }
  if (cid === "count_arm") {
    return "Count space — Negative Binomial CRPS. CANDI and the naive baselines only; no rival has a count head. Count-space and p-value-space numbers never share an axis.";
  }
  if (cid === "distributional") {
    return DIST_ONELINER + " PIT KS is how far the predicted Gaussians' percentiles sit from Uniform(0,1). Coverage 95 is the fraction of bins inside the central 95% interval. Click a column ? for each formula.";
  }
  if (cid === "peaks") return PEAK_ONELINER;
  if (cid === "pointwise") {
    return `Point-wise scores in ${truthSpace(bid)}. MSE: mean squared gap (scales with mark range). GW Pearson: linear correlation (a constant forecast is absent, not zero). GW Spearman: rank-order correlation. MSE top-1% obs: MSE on the tallest 1% of observed bins. The same four names score both truths; the truth toggle above is the only way to move between them, and a number is never rescaled from one into the other. Click a column ? for each formula.`;
  }
  return "";
}

function headKicker(head) {
  if (head.id === "pval") {
    return "P-value head — the head-to-head arm. Every rival predicts signal, so this is the only arm where CANDI meets the published field. Peak metrics are a separate head; count metrics never appear here.";
  }
  if (head.id === "count") {
    return "Count head — Negative Binomial CRPS, with the oracle-scaled / scale-error split and the ±0.09 noise floor. CANDI and the naive baselines only.";
  }
  return PEAK_ONELINER;
}
function summaryBody(bid) {
  const view = viewFor(bid);
  const pending = pendingOf(bid);
  /* When the address does not rank, the chart still lists every row that has a number —
   * it just draws no bars and claims no order. */
  const ranked = isRanked(bid);
  const scored = ranked ? view.rows.filter((r) => r.rank) : view.rows.slice();
  return h("div", null,
    h("p", { class: "chart-kicker" }, familyKicker("summary", bid),
      helpBtn("composite", familyEli5("summary"))),
    rankBarChart({
      aria: `Methods at ${addressLine(bid)}, composite`,
      ranked,
      scored,
      pending,
      rankOf: (r) => (r.rank ? (r.rank[0] + r.rank[1]) / 2 : 999),
      labelOf: (r) => r.method,
      noteOf: (r) => r.composite
        ? (r.composite[0] === r.composite[1]
            ? r.composite[0].toFixed(2)
            : `${r.composite[0].toFixed(2)}${EN_DASH}${r.composite[1].toFixed(2)}`)
        : "",
      lineageOf: (r) => r.lineage,
    }),
    familyTable(bid, "summary", [{ arm: null, metrics: [], cid: "summary" }]));
}

function headSummaryBody(bid, head) {
  const view = viewFor(bid);
  const rankCid = rankCidForHead(head.id);
  const pending = head.id === "count"
    ? pendingOf(bid).filter((p) => p.lineage === "baseline" || p.lineage === "candi")
    : pendingOf(bid);
  const ranked = isRanked(bid);
  const scored = ranked
    ? view.rows.filter((r) => familyRankSpread(r, rankCid, view))
    : view.rows.filter((r) => rowHasFamily(r, rankCid) || rankCid === "summary");
  const primary = head.id === "count"
    ? catMetrics("count_arm", ["ranked"])[0]
    : head.id === "peak"
      ? catMetrics("peaks", ["ranked"])[0]
      : null;
  const groups = overviewGroups(head.id);
  const noteOf = (r) => {
    let note;
    if (head.id === "pval") {
      note = r.composite
        ? (r.composite[0] === r.composite[1]
            ? r.composite[0].toFixed(2)
            : `${r.composite[0].toFixed(2)}${EN_DASH}${r.composite[1].toFixed(2)}`)
        : "";
    } else {
      note = familyNote(r, rankCid, primary, bid);
    }
    const extra = deviceNote(rankCid, r);
    return extra ? (note ? `${note} · ${extra}` : extra) : note;
  };
  return h("div", { class: "head-summary" },
    h("p", { class: "chart-kicker" }, headKicker(head),
      helpBtn(`head-sum-${head.id}`, head.id === "peak" ? PEAK_ELI5 : headEli5(head, bid))),
    rankBarChart({
      aria: `Methods at ${addressLine(bid)}, ${head.label} head`,
      ranked,
      scored,
      pending,
      rankOf: (r) => {
        const s = familyRankSpread(r, rankCid, view);
        return s ? (s[0] + s[1]) / 2 : 999;
      },
      labelOf: (r) => r.method,
      noteOf,
      lineageOf: (r) => r.lineage,
    }),
    familyTable(bid, rankCid, groups, { showComposite: head.id === "pval" }));
}

function familyChartAndTable(bid, cid, headId) {
  const view = viewFor(bid);
  const pending = cid === "count_arm"
    ? pendingOf(bid).filter((p) => p.lineage === "baseline" || p.lineage === "candi")
    : pendingOf(bid);
  const groups = metricGroups(cid, headId);
  const ranked = isRanked(bid);
  const scored = view.rows.filter((r) =>
    rowHasFamily(r, cid) && (!ranked || familyRankSpread(r, cid, view)));
  const primary = catMetrics(cid, ["ranked"]).filter((m) => !headId || headIdOf(m) === headId)[0];
  const kicker = familyKicker(cid, bid);
  const chart = (!scored.length && !pending.length) ? null : rankBarChart({
      aria: `Methods at ${addressLine(bid)}, ${familyTabCaption(cid).title}`,
      ranked,
      scored,
      pending,
      rankOf: (r) => {
        const s = familyRankSpread(r, cid, view);
        return s ? (s[0] + s[1]) / 2 : 999;
      },
      labelOf: (r) => r.method,
      noteOf: (r) => {
        const note = familyNote(r, cid, primary, bid);
        const extra = deviceNote(cid, r);
        return extra ? (note ? `${note} · ${extra}` : extra) : note;
      },
      lineageOf: (r) => r.lineage,
    });
  return h("div", null,
    kicker ? h("p", { class: "chart-kicker" }, kicker,
      helpBtn(`kicker-${cid}`, familyEli5(cid)),
      familyMetricHelps(cid, headId)) : familyMetricHelps(cid, headId),
    chart,
    familyTable(bid, cid, groups));
}

function familyMetricHelps(cid, headId) {
  const groups = metricGroups(cid, headId);
  const ms = groups.flatMap((g) => g.metrics);
  if (!ms.length) return null;
  return h("p", { class: "kicker-metrics" },
    "Metrics: ",
    ...ms.flatMap((m, i) => [
      i ? " · " : null,
      h("span", { class: "chip-wrap" },
        m.label,
        metricHelpBtn(m)),
    ]));
}

function familyNote(row, cid, primary, bid) {
  if (!primary) return "";
  const v = rowVal(row, primary);
  if (v === null || v === undefined) return "";
  let s = v.toFixed(primary.decimals);
  if (primary.floor !== null) s += ` ±${primary.floor}`;
  if (primary.arm === "pval" && primary.key === "crps") s += " · ordering only";
  if (primary.arm === "count" && primary.key === "crps") {
    const osc = rowVal(row, { arm: "count", key: "crps_oracle_scaled", decimals: 4 });
    const se = rowVal(row, { arm: "count", key: "scale_error", decimals: 4 });
    if (osc !== null && osc !== undefined && se !== null && se !== undefined) {
      s += ` · ${osc.toFixed(4)} / ${se.toFixed(4)}`;
    }
  }
  if (primary.key === "auprc") {
    const br = rowVal(row, { arm: "pval", key: "peak_base_rate", decimals: 4 });
    if (br !== null && br !== undefined) s += ` · base ${br.toFixed(4)}`;
  }
  if (cid === "pointwise") s += ` · ${truthSpace(bid)}`;
  return s;
}

function lossBody(bid, headId) {
  const view = viewFor(bid);
  const pending = pendingOf(bid);
  const groups = metricGroups("loss", headId);
  const blocks = groups.map((g) => {
    const m = g.metrics[0];
    const scored = view.rows
      .filter((r) => rowVal(r, m) !== null)
      .slice()
      .sort((a, b) => rowVal(a, m) - rowVal(b, m) || a.method.localeCompare(b.method));
    const space = g.arm === "count" ? "count space" : "p-value space";
    return h("div", { class: "loss-block" },
      h("h3", null, `Loss · ${space}`,
        helpBtn(`loss-${g.arm}`, LOSS_ELI5)),
      familyMetricHelps("loss", headId),
      rankBarChart({
        aria: `Ranked methods on ${metaOf(bid).label}, ${space} loss`,
        scored,
        pending,
        rankOf: (r) => scored.indexOf(r) + 1,
        labelOf: (r) => r.method,
        noteOf: (r) => {
          const v = rowVal(r, m);
          return v === null ? "" : v.toFixed(m.decimals);
        },
        lineageOf: (r) => r.lineage,
      }),
      familyTable(bid, "loss", [g], {
        filterRow: (e) => e.pending && !e.row
          ? true
          : !!(e.row && g.metrics.some((mm) => rowVal(e.row, mm) !== null)),
      }));
  });
  if (!blocks.length) {
    return h("p", { class: "empty-note" }, "No loss numbers at this address.");
  }
  return h("div", { class: "loss-split" }, ...blocks);
}

function covariateBody(bid) {
  const view = viewFor(bid);
  const cat = catInfo("covariate_diagnostics");
  const cover = state.data.covariate_coverage || {};
  const sub = view.sub_boards.candi_lineage || { rows: [] };
  const diags = registry().metrics.filter((m) => m.category === "covariate_diagnostics");
  const has = (sub.rows || []).length > 0;
  const pendingCandi = pendingOf(bid).filter((p) => p.lineage === "candi");
  const nGlobal = cover.n_rows_with_diagnostics || 0;
  const measuredNote = "These numbers are measured on the count head: the C-block changes the prompt and re-decodes the Negative Binomial (μ, n) against count truth (kind = the first eval kind, impute).";

  if (!has) {
    const scorers = cover.scorers || [];
    const lineages = cover.lineages || [];
    if (nGlobal > 0) {
      return h("div", { class: "cov-absent" },
        h("p", { class: "cov-absent-title" },
          "No covariate-sensitivity numbers at this address",
          helpBtn("cov-empty-board", cat.eli5)),
        familyMetricHelps("covariate_diagnostics", "count"),
        h("p", { class: "sub" }, measuredNote),
        h("p", null,
          "These numbers live on CANDI-lineage rows scored by the internal bench. ",
          "They are present at another address, not this one."),
        pendingCandi.length
          ? h("p", { class: "computing-note" },
              pendingCandi.map((p) => p.method).join(", "),
              " — ", pendingCandi[0].note || "results computing")
          : null);
    }
    return h("div", { class: "cov-absent" },
      h("p", { class: "cov-absent-title" },
        "No covariate-sensitivity numbers on any current row",
        helpBtn("cov-empty", cat.eli5)),
      familyMetricHelps("covariate_diagnostics", "count"),
      h("p", { class: "sub" }, measuredNote),
      h("p", null, cat.absent_note),
      h("p", { class: "sub" }, cat.will_populate),
      scorers.length
        ? h("p", { class: "sub" },
            "Scorers on stamped rows: ",
            h("code", null, scorers.join(", ")),
            lineages.length ? ` · lineages present: ${lineages.join(", ")}.` : ".")
        : null,
      pendingCandi.length
        ? h("p", { class: "computing-note" },
            pendingCandi.map((p) => p.method).join(", "),
            " — ", pendingCandi[0].note || "results computing")
        : null);
  }

  const scored = view.rows.filter((r) => rowHasFamily(r, "covariate_diagnostics"));
  const rankedDiag = diags.find((m) => m.role === "diagnostic" && m.direction === "higher");
  const ordered = scored.slice().sort((a, b) => {
    if (!rankedDiag) return a.method.localeCompare(b.method);
    const va = rowVal(a, rankedDiag), vb = rowVal(b, rankedDiag);
    if (va === null && vb === null) return a.method.localeCompare(b.method);
    if (va === null) return 1;
    if (vb === null) return -1;
    const d = rankedDiag.direction === "higher" ? vb - va : va - vb;
    return d || a.method.localeCompare(b.method);
  });
  return h("div", null,
    h("p", { class: "chart-kicker" }, measuredNote, " ", cat.note,
      helpBtn("cov", cat.eli5)),
    familyMetricHelps("covariate_diagnostics", "count"),
    rankBarChart({
      aria: `CANDI versions on ${metaOf(bid).label}, covariate sensitivity`,
      scored: ordered,
      pending: pendingCandi,
      rankOf: (r) => ordered.indexOf(r) + 1,
      labelOf: (r) => `${r.method} ${r.version || ""}`.trim(),
      methodOf: (r) => r.method,
      noteOf: (r) => familyNote(r, "covariate_diagnostics", rankedDiag, bid),
      lineageOf: (r) => r.lineage || "candi",
    }),
    familyTable(bid, "covariate_diagnostics",
      [{ arm: null, metrics: diags, cid: "covariate_diagnostics" }]));
}

function familyTable(bid, cid, groups, opts) {
  const entries = (opts && opts.filterRow)
    ? boardEntries(bid).filter(opts.filterRow)
    : entriesForFamily(bid, cid);
  const view = viewFor(bid);
  const nMetric = groups.reduce((n, g) => n + g.metrics.length, 0);
  const showRank = cid !== "loss";
  const showComposite = (opts && opts.showComposite !== undefined)
    ? opts.showComposite : cid === "summary";
  const nCols = 1 + (showRank ? 1 : 0) + (showComposite ? 1 : 0) + nMetric;

  if (!entries.length) {
    return h("p", { class: "empty-note" },
      cid === "summary"
        ? "No scores stamped at this address."
        : "No method at this address has numbers in this family yet.");
  }

  const RANK_TH_TIP = isRanked(bid)
    ? "Best-to-worst rank in this family at this address. A range like 1–3 is a floor-tied spread, not uncertainty. Lower is better. Bar length on the chart is the same rank (rank 1 longest). Gold / silver / bronze on a row marks an unshared 1 / 2 / 3 — the gap to the next row clears the noise floor."
    : `Nothing at this address is ranked. ${rankingOf(bid).reason}`;
  const METHOD_TH_TIP = "Click the method name for what it is, how it was trained, and how it was scored. A missing card says not recorded — never invent architecture.";
  const head = h("tr", null,
    h("th", { class: "method-col", title: METHOD_TH_TIP },
      h("span", { class: "chip-wrap" }, "method",
        helpBtn(`th-method-${cid}`, METHOD_TH_TIP))),
    showRank ? h("th", { title: RANK_TH_TIP },
      h("span", { class: "chip-wrap" },
        cid === "summary" ? "rank" : "family rank",
        helpBtn(`th-rank-${cid}`, RANK_TH_TIP))) : null,
    showComposite ? h("th", {
      title: "Mean of category sub-scores. Lower is better. A dash is partial coverage, not last place.",
    }, "composite") : null,
    ...groups.flatMap((g) => g.metrics.map((m) =>
      h("th", { class: armClass(m), title: metricTitle(m) },
        h("span", { class: "chip-wrap" },
          metricHeaderName(m, bid),
          m.category === "pointwise" ? null : spaceTag(m, bid),
          metricHelpBtn(m))))));

  const showSpread = cid === "distributional" || groups.some((g) =>
    g.metrics.some((m) => m.key === "pit_ks" || m.key === "coverage_95"));
  const showPeak = cid === "peaks" || groups.some((g) =>
    g.metrics.some((m) => m.key === "auprc"));
  const bodyRows = entries.flatMap((entry) => {
    const pending = entry.pending && !entry.row;
    const cls = pending
      ? "pending-row"
      : (cid === "summary" ? medalClass(entry.row, view) : null);
    const cells = [methodCell(entry, bid, { spread: showSpread, peak: showPeak })];
    if (pending) {
      const rest = nCols - 1;
      cells.push(h("td", { class: "rank-cell cell-pending", colspan: rest,
        title: entry.pending.note || "results computing" },
        entry.pending.note || "results computing"));
    } else if (!entry.row) {
      const title = entry.unscored
        ? (entry.unscored.note || "no scores stamped at this address")
        : "not scored at this address";
      if (showRank) cells.push(h("td", { class: "cell-missing rank-cell", title }, EN_DASH));
      if (showComposite) cells.push(h("td", { class: "cell-missing num", title }, EN_DASH));
      for (const g of groups) {
        for (const m of g.metrics) {
          cells.push(h("td", { class: `cell-missing num ${armClass(m)}`, title }, EN_DASH));
        }
      }
    } else {
      if (showRank) {
        const spread = familyRankSpread(entry.row, cid, view);
        cells.push(spread
          ? rankCell({ rank: spread })
          : isRanked(bid)
            ? h("td", { class: "rank-cell cell-missing" }, EN_DASH)
            : h("td", { class: "rank-cell cell-unranked",
                title: rankingOf(bid).reason }, "unranked"));
      }
      if (showComposite) {
        cells.push(isRanked(bid)
          ? compositeCell(entry.row, bid)
          : h("td", { class: "num cell-unranked", title: rankingOf(bid).reason },
              "unranked"));
      }
      for (const g of groups) {
        for (const m of g.metrics) {
          cells.push(metricCell(entry.row, m, view.rows));
        }
      }
    }
    const tr = h("tr", {
      class: cls,
      title: (cls && String(cls).startsWith("medal-"))
        ? "Unshared rank: the gap to the next row clears the noise floor. Gold / silver / bronze is 1 / 2 / 3."
        : null,
    }, cells);
    const out = [tr];
    if (state.openProv.has(entry.method) && entry.row) {
      out.push(provRow(entry, nCols, bid));
    }
    return out;
  });

  return h("div", { class: "table-scroll" },
    h("table", { class: "board" },
      h("thead", null, head),
      h("tbody", null, bodyRows)));
}

function medalClass(row, view) {
  if (!row || !row.rank || row.rank[0] !== row.rank[1] || row.rank[0] > 3) return null;
  const k = row.rank[0];
  const peers = view.rows.filter((r) => r.rank && r.rank[0] <= k && k <= r.rank[1]);
  return peers.length > 1 ? null : `medal-${k}`;
}

function methodCell(entry, bid, opts) {
  const row = entry.row;
  const b = (row && row.badges) || {};
  const pendingStatus = entry.pending && !row
    ? (entry.pending.note || "results computing")
    : null;
  const spreadBadge = (opts && opts.spread) ? deviceBadge("spread", spreadDevice(row)) : null;
  const peakBadge = (opts && opts.peak) ? deviceBadge("peak", peakDevice(row)) : null;
  const gloss = METHOD_QUALIFIER[entry.method];
  return h("td", { class: "method-col" },
    h("div", { class: "method-line" },
      methodLink(entry.method),
      gloss
        ? h("span", { class: "version-chip",
            title: "avg and Average are different methods. Do not mix them or write “Average vs eDICE” without naming which Average." },
            gloss)
        : null,
      h("span", { class: `badge lineage-${entry.lineage}`,
        title: LINEAGE_TIP[entry.lineage] || entry.lineage },
        LINEAGE_LABEL[entry.lineage] || entry.lineage),
      spreadBadge, peakBadge,
      ...markerBadges(row || entry.pending)),
    h("div", { class: "method-line" },
      row ? h("span", { class: "version-chip" }, `${row.version} · ${row.date}`) : null,
      b.position ? h("span", { class: "badge",
        title: POSITION_TIP[b.position] || b.position }, b.position) : null,
      cellClassBadge(b.cell_types),
      row && row.verified
        ? h("span", { class: "verified", title: "score json resolved when the row was stamped" },
            "✓ verified")
        : row
          ? h("span", { class: "unverified", title: "artifacts not resolved at stamping" },
              "unverified")
          : null,
      pendingStatus
        ? h("span", { class: "badge badge-pending" }, pendingStatus)
        : null,
      row ? h("button", {
        class: "prov-toggle",
        onclick: () => {
          state.openProv.has(entry.method)
            ? state.openProv.delete(entry.method)
            : state.openProv.add(entry.method);
          render();
        },
      }, state.openProv.has(entry.method) ? "hide provenance" : "provenance") : null));
}

function provRow(entry, colspan, bid) {
  return h("tr", { class: "prov-row" }, h("td", { colspan },
    h("div", { class: "prov-board" },
      h("div", { class: "prov-board-title" },
        `${metaOf(bid).label} · ${addressLine(bid)}`),
      provDl(entry.row))));
}

/* ------------------------------------------------------------- ranking bars --- */

/* `ranked` defaults true. When it is false the figure keeps every row and every number and
 * drops the bars, the "#3" and the ordering — there is nothing to draw a length against. */
function rankBarChart({ aria, scored, pending, rankOf, labelOf, noteOf, lineageOf, methodOf,
                        ranked }) {
  const ordered = ranked !== false;
  const n = scored.length;
  const items = scored.slice().sort((a, b) => ordered
    ? (rankOf(a) - rankOf(b) || labelOf(a).localeCompare(labelOf(b)))
    : labelOf(a).localeCompare(labelOf(b)));
  const extra = pending || [];
  if (!items.length && !extra.length) {
    return h("p", { class: "empty-note" }, "No scores stamped yet.");
  }
  const rowH = 26, L = 170, R = 180, T = 6, W = 820;
  const H = T + (items.length + extra.length) * rowH + 8;
  const plotW = W - L - R;
  const svg = h("svg:svg", {
    class: "rank-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": aria,
  });
  const colors = { candi: "var(--candi)", rival: "var(--rival)",
                   baseline: "var(--baseline)", entrant: "var(--entrant)" };
  items.forEach((row, i) => {
    const y = T + i * rowH;
    const rank = rankOf(row, i);
    const t = n <= 1 ? 1 : Math.max(0.02, (n - rank + 1) / n);
    const w = Math.max(6, t * plotW);
    const lin = lineageOf(row);
    const color = colors[lin] || "var(--ink-soft)";
    const rankLabel = typeof rank === "number" && rank === Math.round(rank)
      ? String(rank) : (typeof rank === "number" ? rank.toFixed(1) : String(rank));
    const methodName = methodOf ? methodOf(row) : labelOf(row);
    svg.append(
      h("svg:text", { x: L - 8, y: y + 16, "text-anchor": "end",
        class: "svg-method", "font-size": 12, fill: "var(--ink)",
        onclick: () => {
          state.openCard = { kind: "method", id: methodName };
          render();
        } }, labelOf(row)));
    if (ordered) {
      svg.append(
        h("svg:rect", { x: L, y: y + 6, width: w, height: 14, rx: 3, fill: color }),
        h("svg:text", { x: L + w + 8, y: y + 16, "font-size": 11,
          fill: "var(--ink-soft)" },
          `#${rankLabel}` + (noteOf(row) ? ` · ${noteOf(row)}` : "")));
    } else {
      svg.append(
        h("svg:circle", { cx: L + 7, cy: y + 13, r: 5, fill: color }),
        h("svg:text", { x: L + 20, y: y + 16, "font-size": 11,
          fill: "var(--ink-soft)" },
          "unranked" + (noteOf(row) ? ` · ${noteOf(row)}` : "")));
    }
  });
  extra.forEach((p, i) => {
    const y = T + (items.length + i) * rowH;
    svg.append(
      h("svg:text", { x: L - 8, y: y + 16, "text-anchor": "end",
        class: "svg-method", "font-size": 12, fill: "var(--ink-faint)",
        onclick: () => {
          state.openCard = { kind: "method", id: p.method };
          render();
        } }, p.method),
      h("svg:text", { x: L, y: y + 16, "font-size": 11, fill: "var(--ink-faint)",
        "font-style": "italic" }, p.note || "results computing"));
  });
  return svg;
}

/* ------------------------------------------------------------- cells --- */

function compositeCell(row, bid) {
  if (row.composite) {
    const text = row.composite[0] === row.composite[1]
      ? row.composite[0].toFixed(2)
      : `${row.composite[0].toFixed(2)}${EN_DASH}${row.composite[1].toFixed(2)}`;
    return h("td", { class: "num" }, text, candiDelta(row, bid));
  }
  if (row.partial_coverage) {
    const missing = (row.missing_composite_categories || []).join(", ");
    return h("td", { class: "num cell-missing",
        title: missing
          ? `partial coverage — no ${missing}`
          : "partial coverage — never entered every composite category" },
      EN_DASH, " ",
      h("span", { class: "partial-note" }, "partial coverage"),
      helpBtn(`partial-${row.id}`,
        "This method never entered one or more composite categories, so it has no composite rank. Incomplete, not last. It still ranks inside every category it has numbers for."));
  }
  return h("td", { class: "num cell-missing" }, EN_DASH);
}

function metricTitle(m) {
  const spec = (state.help.metrics || {})[metricId(m)];
  const bits = [];
  if (spec && spec.question) bits.push(spec.question);
  bits.push(`${m.arm ? m.arm + " arm" : "diagnostic"} · `
    + (m.direction ? `${m.direction} is better` : "companion, never ranked"));
  if (m.floor !== null) bits.push(`noise floor ±${m.floor}`);
  if (m.floor_note) bits.push(m.floor_note);
  if (m.calibration_note) bits.push(m.calibration_note);
  if (m.note) bits.push(m.note);
  if (m.eli5) bits.push(m.eli5);
  bits.push("Click ? for the formula this code computes.");
  return bits.join("\n");
}

function rankCell(row) {
  if (!row.rank) return h("td", { class: "rank-cell" }, "—");
  const tied = row.rank[0] !== row.rank[1];
  return h("td", { class: "rank-cell" + (tied ? " rank-tied" : "") },
    spreadText(row.rank));
}

function bestValue(m, rows) {
  let best = null;
  for (const r of rows) {
    const v = rowVal(r, m);
    if (v === null) continue;
    if (best === null) best = v;
    else best = m.direction === "higher" ? Math.max(best, v) : Math.min(best, v);
  }
  return best;
}

function metricCell(row, m, rows) {
  const v = rowVal(row, m);
  if (v === null) {
    return h("td", { class: `num cell-missing ${armClass(m)}`,
      title: "absent — never invented" }, "absent");
  }
  const parts = [fmt(v, m)];
  if (m.floor !== null) parts.push(h("span", { class: "floor-suffix" }, ` ±${m.floor}`));
  if (m.role === "ranked" && m.direction) {
    const best = bestValue(m, rows);
    const d = m.direction === "higher" ? best - v : v - best;
    if (best !== null && d > 0) {
      const sub = m.floor !== null && d < m.floor;
      parts.push(h("span", {
        class: "delta" + (sub ? " subfloor" : ""),
        title: sub ? "gap under the noise floor — never decides a rank" : "gap to the column leader",
      }, `${sub ? "~" : ""}+${d.toFixed(m.decimals)}`));
    }
  }
  if (m.arm === "pval" && m.key === "crps") {
    parts.push(h("span", { class: "order-only",
      title: m.calibration_note || m.eli5 }, "ordering only"));
  }
  return h("td", { class: `num ${armClass(m)}` }, parts);
}

function candiDelta(row, bid) {
  const bar = registry().candi_self_comparison_bar;
  if (row.lineage !== "candi") return null;
  const board = state.data.boards[bid];
  const versions = (board.climb[row.method] || []);
  const idx = versions.findIndex((e) => e.version === row.version);
  if (idx <= 0) return null;
  const rows = viewFor(bid).rows;
  const here = row.metrics[bar.arm] && row.metrics[bar.arm][bar.key];
  const prevRow = rows.find((r) =>
    r.method === row.method && r.version === versions[idx - 1].version);
  const prev = prevRow && prevRow.metrics[bar.arm] && prevRow.metrics[bar.arm][bar.key];
  if (here === undefined || prev === undefined || here === null || prev === null) return null;
  const d = here - prev;
  const clears = Math.abs(d) >= bar.value;
  const arrow = d < 0 ? "▾" : "▴";
  return h("span", {
    class: clears ? (d < 0 ? "verified" : "unverified") : "unverified",
    title: clears
      ? `moved ${d.toFixed(4)} vs ${versions[idx - 1].version} — clears the ${bar.value} seed-alone bar`
      : `moved ${d.toFixed(4)} vs ${versions[idx - 1].version} — under the ${bar.value} seed-alone bar (${bar.note})`,
  }, `${clears ? "" : "~"}${arrow}`);
}

function provDl(row) {
  const p = row.provenance;
  const dl = h("dl", { class: "prov-grid" });
  const put = (k, v) => { if (v) dl.append(h("dt", null, k), h("dd", null, v)); };
  put("score json", p.score_json);
  put("FIR path", p.fir_path);
  put("scoring SHA", p.scoring_sha);
  put("scorer", p.scorer);
  put("regime", p.regime);
  put("store manifest hash", p.store_manifest_hash);
  if (p.sigma_table) put("σ-table", `${p.sigma_table.method} · fitted on ${p.sigma_table.fitted_on}`);
  if (row.has_peak_head || p.has_peak_head) put("peak head", "native (bernoulli_nll present)");
  else if (row.metrics && row.metrics.pval && "auprc" in row.metrics.pval) {
    put("peak head", "absent — AUPRC is a coverage ranking");
  }
  put("flags of record", JSON.stringify(p.flags));
  if (row.missing_metrics && row.missing_metrics.length) {
    put("declared missing", row.missing_metrics.join(", "));
  }
  return dl;
}

/* ------------------------------------------------------------- radar --- */

function categoryRankMid(row, cid, view) {
  if (cid === "count_arm") {
    const sub = view.sub_boards.count_arm.rows.find((r) => r.id === row.id);
    if (!sub || !sub.rank) return null;
    return (sub.rank[0] + sub.rank[1]) / 2;
  }
  if (row.category_subscores && row.category_subscores[cid]) {
    const s = row.category_subscores[cid];
    return (s[0] + s[1]) / 2;
  }
  const ms = catMetrics(cid, ["ranked"]);
  const ranks = ms.map((m) => row.metric_ranks && row.metric_ranks[metricId(m)]).filter(Boolean);
  if (!ranks.length) return null;
  return ranks.reduce((a, rk) => a + (rk[0] + rk[1]) / 2, 0) / ranks.length;
}

function radarEdges(view, ids) {
  return ids.filter((cid) =>
    view.rows.some((r) => categoryRankMid(r, cid, view) !== null));
}

function axisN(view, cid) {
  return view.rows.filter((r) => categoryRankMid(r, cid, view) !== null).length;
}

function edgeSpace(cid) {
  const c = catInfo(cid);
  return (c && c.space_label) || "";
}

function radarPanel() {
  const bid = state.outerEval || state.radarEval;
  const ids = boardIds();
  if (!ids.includes(bid)) {
    return h("p", { class: "empty-note" }, "Pick a data set to see per-method shape.");
  }
  const view = viewFor(bid);
  const meta = metaOf(bid);
  const pvalEdges = radarEdges(view, PVAL_RADAR_EDGES);
  const countEdges = radarEdges(view, COUNT_RADAR_EDGES);
  if (pvalEdges.length < 3 && !countEdges.length) {
    return h("div", null,
      h("p", { class: "sub" },
        `${meta.label} has ${pvalEdges.length} rankable p-value categor${pvalEdges.length === 1 ? "y" : "ies"} — a p-value shape needs three.`));
  }
  const cards = view.rows.map((row) => radarCard(row, view, pvalEdges, countEdges));
  const pending = pendingOf(bid).map((p) =>
    h("div", { class: "radar-card pending-card" },
      h("div", { class: "radar-name" }, methodLink(p.method)),
      h("p", { class: "computing-note" }, p.note || "results computing")));
  return h("div", { id: "radar" },
    h("p", { class: "sub" },
      "P-value-space categories (point-wise, distributional, peaks) share one polygon. ",
      "Count-space rank is a separate figure beside it, labeled count space. ",
      "Rank 1 sits on the outer ring (or the top of the count bar); last place sits at the centre. ",
      "Ranks are computed on ",
      meta.label,
      " — at one address only. Count-space and p-value-space numbers never share an axis."),
    h("div", { class: "radar-grid" }, ...cards, ...pending));
}

function radarPolygon(row, view, edges, color) {
  const space = edges.length ? edgeSpace(edges[0]) : "";
  const oneSpace = edges.filter((cid) => edgeSpace(cid) === space);
  const W = 220, H = 220, cx = 110, cy = 110, R = 78;
  const svg = h("svg:svg", {
    class: "radar-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": `${space} category ranks for ${row.method}`,
  });
  const nAx = oneSpace.length;
  if (nAx < 3) return svg;
  const pt = (i, t) => {
    const ang = -Math.PI / 2 + i * 2 * Math.PI / nAx;
    return [cx + Math.cos(ang) * R * t, cy + Math.sin(ang) * R * t];
  };
  for (const ring of [0.25, 0.5, 0.75, 1]) {
    const pts = oneSpace.map((_, i) => pt(i, ring).join(",")).join(" ");
    svg.append(h("svg:polygon", {
      points: pts, fill: "none", stroke: "var(--line)", "stroke-width": 1,
    }));
  }
  oneSpace.forEach((cid, i) => {
    const [x, y] = pt(i, 1);
    svg.append(h("svg:line", {
      x1: cx, y1: cy, x2: x, y2: y, stroke: "var(--line)", "stroke-width": 1,
    }));
    const [lx, ly] = pt(i, 1.18);
    const c = catInfo(cid);
    const axisTip = (c && c.eli5) || cid;
    svg.append(h("svg:g", null,
      h("svg:title", null, axisTip),
      h("svg:text", {
        x: lx, y: ly, "text-anchor": "middle", "font-size": 9,
        fill: "var(--ink-soft)",
      }, c ? c.label : cid)));
  });
  const verts = [];
  oneSpace.forEach((cid, i) => {
    const rank = categoryRankMid(row, cid, view);
    if (rank === null) return;
    const n = axisN(view, cid) || 1;
    const t = n <= 1 ? 1 : (n - rank + 1) / n;
    verts.push(pt(i, t));
  });
  if (verts.length >= 3) {
    svg.append(h("svg:polygon", {
      points: verts.map((p) => p.join(",")).join(" "),
      fill: color, "fill-opacity": 0.22, stroke: color, "stroke-width": 1.5,
    }));
  }
  verts.forEach(([x, y]) => {
    svg.append(h("svg:circle", { cx: x, cy: y, r: 2.5, fill: color }));
  });
  return svg;
}

function radarCountBar(row, view, cid, color) {
  const W = 88, H = 220, top = 28, bot = 200, x = 34, barW = 20;
  const c = catInfo(cid);
  const space = (c && c.space_label) || "count space";
  const svg = h("svg:svg", {
    class: "radar-count-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": `${space} rank for ${row.method}`,
  });
  svg.append(h("svg:text", {
    x: W / 2, y: 14, "text-anchor": "middle", "font-size": 9,
    fill: "var(--ink-soft)",
  }, c ? c.label : cid));
  svg.append(h("svg:rect", {
    x, y: top, width: barW, height: bot - top, rx: 3,
    fill: "none", stroke: "var(--line)", "stroke-width": 1,
  }));
  svg.append(h("svg:text", {
    x: x + barW + 6, y: top + 8, "font-size": 8, fill: "var(--ink-faint)",
  }, "rank 1"));
  svg.append(h("svg:text", {
    x: x + barW + 6, y: bot, "font-size": 8, fill: "var(--ink-faint)",
  }, "last"));
  const rank = categoryRankMid(row, cid, view);
  if (rank === null) return svg;
  const n = axisN(view, cid) || 1;
  const t = n <= 1 ? 1 : (n - rank + 1) / n;
  const barH = Math.max(4, t * (bot - top));
  svg.append(h("svg:rect", {
    x, y: bot - barH, width: barW, height: barH, rx: 3,
    fill: color,
  }));
  const rankLabel = rank === Math.round(rank) ? String(rank) : rank.toFixed(1);
  svg.append(h("svg:text", {
    x: x + barW / 2, y: bot - barH - 4, "text-anchor": "middle",
    "font-size": 10, fill: "var(--ink-soft)",
  }, rankLabel));
  return svg;
}

function radarCard(row, view, pvalEdges, countEdges) {
  const colors = { candi: "var(--candi)", rival: "var(--rival)",
                   baseline: "var(--baseline)", entrant: "var(--entrant)" };
  const color = colors[row.lineage] || "var(--ink-soft)";
  const figures = [];
  if (pvalEdges.length >= 3) {
    figures.push(h("div", { class: "radar-space" },
      h("div", { class: "space-tag" }, "p-value space"),
      radarPolygon(row, view, pvalEdges, color)));
  }
  if (countEdges.length) {
    const hasCount = countEdges.some((cid) => categoryRankMid(row, cid, view) !== null);
    let countFig;
    if (!hasCount) {
      countFig = h("p", { class: "radar-absent" }, "no count arm");
    } else if (countEdges.length >= 3) {
      countFig = radarPolygon(row, view, countEdges, color);
    } else {
      countFig = h("div", { class: "radar-count-bars" },
        ...countEdges.map((cid) => radarCountBar(row, view, cid, color)));
    }
    figures.push(h("div", { class: "radar-space" },
      h("div", { class: "space-tag" }, "count space"),
      countFig));
  }
  return h("div", { class: "radar-card" },
    h("div", { class: "radar-name" }, methodLink(row.method),
      h("span", { class: "version-chip" }, row.version)),
    h("div", { class: "radar-figures" }, ...figures));
}

/* ------------------------------------------------------------- climb --- */

function climbOptIn() {
  return h("p", { class: "climb-optin" },
    h("button", {
      class: "climb-link", type: "button",
      onclick: () => { state.showClimb = true; render(); },
    }, "CANDI progress over time"));
}

function climbPanel() {
  const hide = h("button", {
    class: "climb-link", type: "button",
    onclick: () => { state.showClimb = false; render(); },
  }, "hide");
  const empty = h("section", { class: "panel panel-secondary", id: "over-time" },
    h("h2", null, "CANDI progress over time"),
    h("p", { class: "empty-note" },
      "no CANDI versions stamped yet; two runs are training"),
    hide);

  const candiClimb = (bid) => {
    const climb = state.data.boards[bid].climb;
    const out = {};
    for (const [method, series] of Object.entries(climb)) {
      const candi = series.filter((e) => e.lineage === "candi" && e.composite);
      if (candi.length) out[method] = candi;
    }
    return out;
  };
  const ids = boardIds().filter((bid) => Object.keys(candiClimb(bid)).length);
  if (!ids.length) return empty;
  if (!ids.includes(state.climbEval)) state.climbEval = ids[0];
  const bid = state.climbEval;
  const climb = candiClimb(bid);
  const methods = Object.keys(climb);
  const W = 900, H = 260, L = 46, R = 150, T = 18, B = 34;
  const entries = methods.flatMap((m) => climb[m]);
  const dates = entries.map((e) => Date.parse(e.date));
  let [d0, d1] = [Math.min(...dates), Math.max(...dates)];
  if (d0 === d1) { d0 -= 864e5 * 7; d1 += 864e5 * 7; }
  const rMax = Math.max(2, Math.ceil(Math.max(...entries.map((e) => e.composite[1]))));
  const x = (t) => L + (t - d0) / (d1 - d0) * (W - L - R);
  const y = (rank) => T + (rank - 1) / (rMax - 1) * (H - T - B);
  const svg = h("svg:svg", { class: "climb-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": "CANDI composite rank over time" });

  for (let r = 1; r <= rMax; r++) {
    svg.append(
      h("svg:line", { x1: L, x2: W - R, y1: y(r), y2: y(r),
        stroke: "var(--line)", "stroke-width": 1 }),
      h("svg:text", { x: L - 8, y: y(r) + 4, "text-anchor": "end",
        "font-size": 11, fill: "var(--ink-faint)" }, r));
  }
  svg.append(h("svg:text", { x: 12, y: T + 10, "font-size": 11,
    fill: "var(--ink-faint)", transform: `rotate(-90 12 ${T + 10})`, "text-anchor": "end" },
    "composite rank (1 = best)"));

  const leader = methods
    .map((m) => climb[m][climb[m].length - 1])
    .filter((e) => e.composite)
    .sort((a, b) => a.composite[0] - b.composite[0])[0];
  if (leader) {
    svg.append(h("svg:rect", { x: L, width: W - L - R,
      y: y(leader.composite[0]),
      height: Math.max(2, y(leader.composite[1]) - y(leader.composite[0])),
      fill: "var(--band)" }));
  }

  const color = "var(--candi)";
  let labelY = [];
  const placeLabel = (yy) => {
    let out = yy;
    while (labelY.some((used) => Math.abs(used - out) < 13)) out += 13;
    labelY.push(out);
    return out;
  };
  for (const method of methods.sort()) {
    const series = climb[method];
    const pts = series.map((e) =>
      [x(Date.parse(e.date)), y((e.composite[0] + e.composite[1]) / 2)]);
    if (pts.length > 1) {
      svg.append(h("svg:polyline", {
        points: pts.map((p) => p.join(",")).join(" "),
        fill: "none", stroke: color, "stroke-width": 2 }));
    }
    for (let i = 0; i < pts.length; i++) {
      svg.append(
        h("svg:circle", { cx: pts[i][0], cy: pts[i][1], r: 3.5, fill: color }),
        h("svg:text", { x: pts[i][0], y: pts[i][1] - 8, "text-anchor": "middle",
          "font-size": 10, fill: color }, series[i].version));
    }
    const last = pts[pts.length - 1];
    svg.append(h("svg:text", {
      class: "svg-method",
      x: W - R + 8, y: placeLabel(last[1] + 4),
      "font-size": 11, "font-weight": 700, fill: color,
      onclick: () => { state.openCard = { kind: "method", id: method }; render(); },
    }, method));
  }
  svg.append(
    h("svg:text", { x: L, y: H - 10, "font-size": 11, fill: "var(--ink-faint)" },
      new Date(d0).toISOString().slice(0, 10)),
    h("svg:text", { x: W - R, y: H - 10, "text-anchor": "end", "font-size": 11,
      fill: "var(--ink-faint)" }, new Date(d1).toISOString().slice(0, 10)));

  const chips = ids.map((id) =>
    h("button", {
      class: "chip", "aria-pressed": String(id === bid),
      title: `${metaOf(id).eli5} Regime: ${metaOf(id).regime_id}.`,
      onclick: () => { state.climbEval = id; render(); },
    }, metaOf(id).label));

  return h("section", { class: "panel panel-secondary", id: "over-time" },
    h("h2", null, "CANDI progress over time",
      helpBtn("climb",
        "Each CANDI version is a dated point on composite rank. Only CANDI-lineage rows are plotted. Version-over-version arrows on the table use the 0.1195 seed-alone bar (a seed change alone moves pooled imputation CRPS by 0.1195 on the full EIC panel) — that bar is never a cross-method floor. The cross-method count CRPS floor is ±0.09.")),
    hide,
    h("div", { class: "controls" },
      h("span", { class: "chip-group" },
        h("span", { class: "chip-label" }, "regime"), ...chips)),
    svg);
}

function cardField(label, text) {
  if (!text) return null;
  return h("div", null,
    h("div", { class: "card-label" }, label),
    Array.isArray(text)
      ? h("ul", null, text.map((c) => h("li", null, c)))
      : h("p", null, text));
}

function methodCardBody(name, info) {
  if (!info) {
    return h("p", null, "No help card recorded for this method. Architecture, training, and scoring notes for this name are not recorded in this tree. Never invent them.");
  }
  const bits = [
    cardField("What it is", info.what),
    cardField("Training data", info.training),
    cardField("Classes", info.classes),
    cardField("How it is scored", info.scoring),
    cardField("Caveats", info.caveats),
  ];
  const bid = state.outerEval || state.radarEval;
  if (bid && state.data.boards[bid]) {
    const view = viewFor(bid);
    const row = view.rows.find((r) => r.method === name);
    const pvalEdges = radarEdges(view, PVAL_RADAR_EDGES);
    const countEdges = radarEdges(view, COUNT_RADAR_EDGES);
    if (row && (pvalEdges.length >= 3 || countEdges.length)) {
      bits.push(
        h("div", { class: "card-label" }, "Shape at this address"),
        radarCard(row, view, pvalEdges, countEdges));
    }
  }
  return h("div", null, ...bits);
}

function comboTitle(key) {
  const parts = (key || "").split("/");
  const labels = {
    anchor: "Anchor — the 2019 field",
    count: "Count", pval: "P-value", peak: "Peak", summary: "Summary",
    count_arm: "Distributional", pointwise: "Point-wise",
    distributional: "Distributional", loss: "Loss",
    covariate_diagnostics: "Covariate sensitivity", peaks: "Peaks",
    radar: "Shape",
  };
  return parts.map((p) => labels[p] || p).join(" · ");
}

function comboCardBody(info) {
  if (!info) {
    return h("p", null, "No help card recorded for this view.");
  }
  return h("div", null,
    cardField("What this view asks", info.question),
    cardField("Truth source", info.truth),
    cardField("Instrument", info.instrument),
    cardField("How each method class is scored", info.devices),
    cardField("Caveats", info.caveats));
}

function cardOverlay() {
  const card = state.openCard;
  if (!card) return null;
  const close = () => { state.openCard = null; render(); };
  let title;
  let body;
  if (card.kind === "method") {
    title = card.id;
    body = methodCardBody(card.id, state.help.methods[card.id]);
  } else if (card.kind === "metric") {
    const m = registry().metrics.find((x) => metricId(x) === card.id);
    title = m ? m.label : card.id;
    body = metricCardBody(card.id);
  } else {
    title = comboTitle(card.id);
    body = comboCardBody(state.help.combos[card.id]);
  }
  return h("div", { class: "card-overlay", onclick: close },
    h("div", { class: "card-sheet", role: "dialog", "aria-label": title,
      onclick: (ev) => ev.stopPropagation() },
      h("div", { class: "card-head" },
        h("h3", null, title),
        h("button", { type: "button", class: "card-close", onclick: close }, "close")),
      body));
}

/* ------------------------------------------------------------- anchor + void --- */

/* §6 — the 2019 field. It sits under the ranked table, not in it: these rows carry no regime
 * because we never trained them, so they never share a ranking denominator with ours. */
function anchorPanel() {
  const aid = anchorId();
  if (!aid) return null;
  const meta = metaOf(aid);
  const view = viewFor(aid);
  const pending = pendingOf(aid);
  const ni = meta.non_independence || {};
  const groups = overviewGroups("pval");
  const nMetric = groups.reduce((n, g) => n + g.metrics.length, 0);
  const head = h("tr", null,
    h("th", { class: "method-col" }, "method"),
    ...groups.flatMap((g) => g.metrics.map((m) =>
      h("th", { class: armClass(m), title: metricTitle(m) },
        h("span", { class: "chip-wrap" }, m.label, metricHelpBtn(m))))));
  const byName = new Map();
  for (const r of view.rows) byName.set(r.method, { method: r.method, row: r });
  for (const p of pending) {
    if (!byName.has(p.method)) byName.set(p.method, { method: p.method, pending: p });
  }
  const entries = [...byName.values()].sort((a, b) => a.method.localeCompare(b.method));
  const body = entries.map((e) => h("tr", { class: e.row ? null : "pending-row" },
    h("td", { class: "method-col" },
      h("div", { class: "method-line" }, methodLink(e.method),
        METHOD_QUALIFIER[e.method]
          ? h("span", { class: "version-chip" }, METHOD_QUALIFIER[e.method])
          : null,
        h("span", { class: "badge lineage-entrant", title: LINEAGE_TIP.entrant },
          LINEAGE_LABEL.entrant))),
    e.row
      ? groups.flatMap((g) => g.metrics.map((m) => metricCell(e.row, m, view.rows)))
      : h("td", { class: "cell-pending", colspan: nMetric,
          title: e.pending.note }, e.pending.note)));

  return h("section", { class: "panel panel-anchor", id: "anchor-block" },
    h("h2", null, meta.label,
      h("span", { class: "h2-note" }, meta.subtitle),
      helpBtn("anchor-eli5", meta.eli5)),
    h("p", { class: "chart-kicker" },
      h("span", { class: "badge badge-warn" }, "anchor — we did not run these"),
      " ", meta.eli5),
    h("p", { class: "addr-readout" }, addressLine(aid)),
    unrankedBanner(aid),
    ni.headline
      ? h("div", { class: "caveats caveats-hard" },
          h("div", { class: "caveats-title" }, "Known non-independence — count, do not read off"),
          h("p", null, h("strong", null, ni.headline), " ", ni.detail,
            helpBtn("anchor-ni", `${ni.headline} ${ni.detail}`)),
          h("ul", { style: "margin:4px 0;padding:0" },
            (ni.duplicate_groups || []).map((g) =>
              h("li", null, g.members.join(" = "), " — ", g.extent))))
      : null,
    (meta.caveats || []).length
      ? h("div", { class: "caveats" },
          h("div", { class: "caveats-title" }, "On the anchor block"),
          h("ul", { style: "margin:4px 0;padding:0" },
            meta.caveats.map((c, i) =>
              h("li", null, c, " ", helpBtn(`anchorcav-${i}`, `${c} ${meta.eli5}`)))))
      : null,
    entries.length
      ? h("div", { class: "table-scroll" },
          h("table", { class: "board" }, h("thead", null, head), h("tbody", null, body)))
      : h("p", { class: "empty-note" }, "No anchor rows stamped."));
}

/* §3.3 — the rows this page used to show. Named, dated, and deliberately not numbered. */
function voidPanel() {
  const v = state.data.void;
  if (!v || !(v.rows || []).length) return null;
  if (!state.showVoid) {
    return h("p", { class: "climb-optin" },
      h("button", {
        class: "climb-link", type: "button",
        onclick: () => { state.showVoid = true; render(); },
      }, `Void rows — ${v.rows.length} retired, not carried forward`));
  }
  const byBoard = new Map();
  for (const r of v.rows) {
    if (!byBoard.has(r.former_board)) byBoard.set(r.former_board, []);
    byBoard.get(r.former_board).push(r);
  }
  return h("section", { class: "panel panel-secondary", id: "void-block" },
    h("h2", null, v.meta.label, helpBtn("void-eli5", v.meta.eli5)),
    h("button", {
      class: "climb-link", type: "button",
      onclick: () => { state.showVoid = false; render(); },
    }, "hide"),
    h("p", { class: "chart-kicker" }, v.meta.eli5),
    ...[...byBoard.entries()].sort().map(([former, rows]) =>
      h("div", { class: "void-group" },
        h("h3", null, `was: ${former}`),
        h("p", { class: "sub" }, rows[0].reason),
        h("p", { class: "void-names" },
          ...rows.flatMap((r, i) => [
            i ? " · " : null,
            h("span", { class: "version-chip", title: `${r.version} · ${r.date}` },
              r.method),
          ])))),
    h("p", { class: "sub" },
      "No number is shown for any of these rows, and that is the point: a void score under a "
      + "new label would read as freshly computed. The files stay in leaderboard/void/, so the "
      + "record is in git rather than on the page."));
}

/* ------------------------------------------------------------- legend + footer --- */
function legendPanel() {
  const reg = registry();
  const bar = reg.candi_self_comparison_bar;
  const pvalCrps = reg.metrics.find((m) => m.arm === "pval" && m.key === "crps");
  const countCrps = reg.metrics.find((m) => m.arm === "count" && m.key === "crps");
  return h("section", { class: "panel" },
    h("h2", null, "How to read a number"),
    h("dl", { class: "legend" },
      h("dt", null, `rank "1${EN_DASH}3"`),
      h("dd", null, "The best and worst achievable rank: two rows whose gap sits under a metric's noise floor are tied, and ties propagate through the category means into the composite. A gap under the floor never decides a rank."),
      h("dt", null, "score ± floor"),
      h("dd", null, `Count-space macro CRPS carries ±${countCrps.floor} (target-clustered). A seed change alone moves pooled CRPS by ${bar.value} — that bar is for CANDI version-over-version arrows, never a cross-method floor. Deltas under the floor grey out with a ~ prefix.`),
      h("dt", null, "medals"),
      h("dd", null, "A row tints gold/silver/bronze only when its rank is unshared — the gap to the next row clears the floor."),
      h("dt", null, "p-value CRPS"),
      h("dd", null,
        "Ordering only — do not read the absolute value as calibrated. ",
        pvalCrps.floor === null ? pvalCrps.floor_note : `floor ±${pvalCrps.floor}.`,
        " ", pvalCrps.calibration_note || ""),
      h("dt", null, "point-to-Gaussian spread"),
      h("dd", null, pvalCrps.calibration_note
        || "Point-only rivals get a constant per-assay spread. Rank on CRPS; do not read the absolute value as calibrated."),
      h("dt", null, "CANDI ▴▾ arrows"),
      h("dd", null, `CANDI's own version-over-version arrows use the stricter seed-alone bar (${bar.arm} ${bar.key} ${bar.value}); it is never a cross-method floor.`),
      h("dt", null, "paired columns"),
      h("dd", null, "Count-space CRPS never renders without its oracle-scaled / scale-error split; p-value CRPS never without PIT KS and coverage. Absent means absent — no number is ever invented."),
      h("dt", null, "results computing"),
      h("dd", null, "A grey row or cell with no numbers: that method is a known entry whose scores have not landed yet. The italic note is the status (still computing, not started, …). It is not a silent omission."),
      h("dt", null, "✓ verified"),
      h("dd", null, "The row's score json resolved on disk when the row was stamped and `check` re-verifies it wherever the artifact is reachable."),
      h("dt", null, "the address"),
      h("dd", null, (state.data.address || {}).rule
        || "Every cell is addressed by method, regime, truth, panel, scope and metric."),
      h("dt", null, "unranked"),
      h("dd", null, "A number with no order beside it. A rank is a claim that one method beat another, and that claim needs a resolution band; until the band is measured the honest output is the number alone. Not an error, and not a missing number."),
      h("dt", null, "the row markers"),
      ...Object.entries(state.data.markers || {}).flatMap(([id, spec]) => [
        h("dd", { class: "legend-sub" }, h("strong", null, spec.label), " — ", spec.eli5),
      ]),
      h("dt", null, "the anchor block"),
      h("dd", null, "The 2019 field sits under the ranked table, never in it. Those rows carry no regime because we never trained them, so they and our rows never share a ranking denominator — and the field has fewer distinct methods than rows, so any \"beats N methods\" claim has to be counted."),
      h("dt", null, "void rows"),
      h("dd", null, "Rows the design retired. They are named and dated at the foot of the page and carry no number, because a void score under a new label would read as freshly computed.")));
}

function renderFooter() {
  const rep = state.data.reproducibility;
  const hashes = boardIds().map((bid) => {
    const meta = metaOf(bid);
    return h("p", null,
      meta.label,
      `: store ${meta.frozen.store_manifest_hash} · regime ${meta.frozen.regime_sha256}.`,
      String(meta.frozen.store_manifest_hash).startsWith("TODO-")
        ? h("span", { class: "badge badge-warn", title: meta.frozen.frozen_note },
            "not frozen — no row may be stamped")
        : null);
  });
  document.getElementById("footer").replaceChildren(
    h("div", { class: "footer-inner" },
      h("h3", null, "Reproducibility"),
      h("p", null, "Score: ", h("code", null, rep.score_command)),
      h("p", null, "Stamp: ", h("code", null, rep.stamp_command)),
      h("p", null, rep.note),
      ...hashes,
      h("p", null,
        "This page is compiled by tools/leaderboard.py build — a pure function of the "
        + "repo tree, rebuilt by CI on every row that lands on main.")));
}

boot();
