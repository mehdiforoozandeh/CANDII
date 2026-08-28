/* CANDI imputation leaderboard — vanilla JS over the compiled leaderboard.json.
 * No framework, no chart library, no external requests. The registry travels inside the
 * payload, so nothing about a metric is hard-coded here (LEADERBOARD_PRD.md §5.1). */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BOARD_ORDER = ["main", "dev", "entrants"];
const HEAD_ORDER = ["count", "pval", "peak"];
const HEAD_LABEL = { count: "Count", pval: "P-value", peak: "Peak" };
/* Covariate diagnostics have arm=null in the registry. They sit under Count
 * because `harness.c_block(..., kind=kinds[0])` predicts NB (mu, n) against
 * count truth (`_predictor`, `_c_contexts`). kinds[0] defaults to "impute". */
const COVARIATE_HEAD = "count";
const RADAR_EDGES = ["pointwise", "distributional", "peaks", "count_arm"];
const EN_DASH = "–";
const LINEAGE_LABEL = {
  candi: "CANDI", rival: "retrained rival",
  baseline: "baseline", entrant: "2019 entrant",
};
/* Table names that collide across boards. Slugs stay; the gloss is the label. */
const METHOD_QUALIFIER = {
  avg: "Dataset-2 · train-only",
  Average: "2019 · train+validation",
};

/* Verified 2026-08-27 from fit_sigma / bench.external and the stamped rival rows.
 * Point-only rivals: one σ per assay, V-pair residuals on eval chr21, reused unchanged.
 * Homoscedastic per assay. PIT KS 0.367–0.377, coverage_95 0.977–0.984. */
const DIST_ONELINER = "Point-only rivals get one Gaussian spread per assay, fitted once on validation-pair residuals (eval chromosome chr21) and reused unchanged. The spread does not adapt bin-by-bin. That device is known-miscalibrated (PIT KS 0.367–0.391, 95% coverage 0.977–0.985 on the stamped rival rows), so CRPS here is ordering-only.";
const DIST_ELI5 = "Most rivals predict one number per bin, not a distribution. To give them CRPS, PIT, and 95% coverage we wrap a Gaussian around that number. The width is one constant per assay: the root-mean-square residual on the validation pairs, chromosome chr21, then frozen. Refitting on a later protocol would leak. Same width at every bin of that assay — the spread does not grow or shrink with the signal. On Avocado, eDICE, and ChromImpute this width is too wide in the same way (PIT KS 0.367–0.391, 95% coverage 0.977–0.985). Rank on CRPS; do not read the absolute as a calibrated score. That is the ordering-only note on this tab. CANDI's Negative Binomial count head and per-bin Gaussian signal head emit a real distribution; they do not use this device. A σ-table on the row means the fitted spread; PIT/coverage without a σ-table means the method emitted its own spread.";
const PEAK_ONELINER = "For methods with no peak score, AUPRC ranks bins by predicted signal against called peaks — a coverage ranking, not a peak classifier. A Bernoulli peak head is a different device.";
const PEAK_ELI5 = "eDICE, Avocado, and ChromImpute do not output a peak probability. The scorer then ranks each bin by the predicted p-value level (or the count mean if that is all they have) and asks whether high predicted signal sits on bins the store called as peaks (MACS2 labels). That is a coverage ranking: does the track light up where peaks were called? It is not a peak classifier. CANDI's peak head, when its rows land, emits a per-bin Bernoulli probability and is labeled separately. AUPRC is always shown with the peak base rate.";

const state = {
  data: null,
  help: { methods: {}, combos: {} },
  view: "default",
  outerEval: null,
  midHead: null,
  innerFamily: null,
  radarEval: "main",
  climbEval: "main",
  showClimb: false,
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
  return h("button", {
    class: "help", type: "button",
    "aria-label": "Explain this", "aria-expanded": String(open),
    onclick: (ev) => {
      ev.stopPropagation();
      state.openHelp = open ? null : id;
      render();
    },
  }, "?", open ? h("span", { class: "help-tip", role: "tooltip" }, text) : null);
}
function comboKey() {
  if (!state.outerEval) return null;
  if (state.midHead === "summary") return `${state.outerEval}/summary`;
  if (!state.midHead || state.midHead === "radar") return null;
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
function writeHash() {
  let next = "";
  if (state.showClimb && !state.outerEval) next = "#over-time";
  else if (state.outerEval) {
    const segs = [state.outerEval];
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
  if (parts[1] === "summary" || parts[1] === "radar") {
    state.midHead = parts[1];
    state.innerFamily = null;
    if (parts[1] === "radar") state.radarEval = parts[0];
    return;
  }
  if (parts[1] && HEAD_ORDER.includes(parts[1])) {
    state.midHead = parts[1];
    state.innerFamily = parts[2] || null;
  }
}
function armClass(m) {
  return m.arm === "count" ? "arm-count" : m.arm === "pval" ? "arm-pval" : "";
}
function spaceTag(m, bid) {
  let s = "diagnostic";
  if (m.arm === "count") s = "counts";
  else if (m.arm === "pval") s = bid === "entrants" ? "2019 signal" : "−log10 p";
  return h("span", { class: "space-tag" }, s);
}

function cellClassBadge(text) {
  if (!text) return null;
  if (text === "zero-shot cell types") {
    return h("span", { class: "badge",
      title: "Stored class: zero-shot cell types. In this vocabulary that means no learned per-cell embedding, not a cell type with no training experiment. Dataset-2 scores training→validation of the same type." },
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
function spreadDevice(row) {
  if (!row) return null;
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
          title: "Gaussian wrapped around a point prediction. One spread per assay, fitted on validation residuals, reused unchanged. Homoscedastic per assay — the spread does not adapt bin-by-bin." },
          "fitted spread device")
      : h("span", { class: "badge badge-device badge-native",
          title: "This method emitted its own per-bin spread. Not the fitted σ-table device." },
          "native distribution");
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
function metricExplainer(m) {
  if (m.arm === "pval" && (m.key === "crps" || m.key === "pit_ks" || m.key === "coverage_95")) {
    return DIST_ELI5;
  }
  if (m.key === "auprc" || m.key === "peak_base_rate") return PEAK_ELI5;
  return null;
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
    state.help = { methods: {}, combos: {} };
  }
  const ids = Object.keys(payload.boards);
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

function boardIds() {
  const ids = Object.keys(state.data.boards);
  return ids.sort((a, b) => {
    const [ia, ib] = [BOARD_ORDER.indexOf(a), BOARD_ORDER.indexOf(b)];
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
}

function viewFor(bid) {
  const board = state.data.boards[bid];
  const want = bid === "main" ? state.view : "default";
  return board.views[want] || board.views.default;
}

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
    nodes.push(legendPanel());
  }
  nodes.push(state.showClimb ? climbPanel() : climbOptIn());
  const overlay = cardOverlay();
  if (overlay) nodes.push(overlay);
  app.replaceChildren(...nodes);
  renderFooter();
}

function readerBadges(meta) {
  return (meta.reader_badges || []).map((b, i) =>
    h("span", { class: "badge badge-warn", title: `Internal code lives in the ?` },
      b.label, helpBtn(`badge-${meta.protocol}-${i}`, b.eli5)));
}

function familyTabCaption(cid, bid) {
  if (cid === "summary") return { title: "Summary", sub: "composite rank" };
  const c = catInfo(cid);
  if (!c) return { title: cid, sub: "" };
  let sub = (c.space_label && c.space_label.toLowerCase() !== c.label.toLowerCase())
    ? c.space_label : "";
  if (bid === "entrants" && cid === "pointwise") sub = "2019 signal";
  return { title: c.label, sub };
}

function familyEli5(cid) {
  if (cid === "summary") {
    return "Composite is the mean of category mean-ranks for categories at least one method on this eval set fully covers. A method missing any of those categories is incomplete, not last: the composite cell is a dash, and it is left out of the headline ranking. It still ranks inside every category it has numbers for. Lower is better.";
  }
  if (cid === "distributional") return DIST_ELI5;
  if (cid === "peaks") return PEAK_ELI5;
  const c = catInfo(cid);
  return c && c.eli5;
}

function headEli5(head, bid) {
  if (head.id === "pval") {
    if (bid === "entrants") {
      return "2019 challenge signal, not store −log10 p. Ranking uses the point-wise composite on this board. The header letters MSE / Pearson / Spearman match Full genome and Chromosome 21; the truth signal does not. Never translate.";
    }
    return "P-value space (−log10 p from the store). Ranking uses the same §5.2 composite as the data-level summary. Peak scores live under the Peak head, even though the composite math still includes the peaks category when it is board-active. Count numbers never appear here. The header letters MSE / Pearson / Spearman match the 2019 tab; the truth signal does not.";
  }
  if (head.id === "count") {
    return "Count space. Ranking is Negative Binomial CRPS, always with its oracle-scaled / scale-error split and the ±0.09 noise floor. Covariate sensitivity sits here because the C-block re-decodes (μ, n) against count truth. P-value numbers never appear here.";
  }
  return "Peak detection. Ranking is AUPRC, always with the peak base rate. For methods that do not emit a peak score, that AUPRC is a coverage ranking against called peaks, not a peak classifier. These scores are stored on the p-value arm in the registry; they are a separate head so they never mix with point-wise or distributional p-value numbers.";
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
    "aria-label": "Data set" },
    ids.map((id) => {
      const m = metaOf(id);
      return h("button", {
        class: "tab outer", type: "button", role: "tab",
        id: `eval-tab-${id}`,
        "aria-selected": String(id === bid),
        "aria-controls": "eval-board",
        title: `Internal code: ${m.protocol} / ${id}`,
        onclick: () => pick({ outerEval: id, radarEval: id }),
      }, coded(m.label, `${m.protocol} / ${id}`),
        h("span", { class: "tab-sub" }, m.subtitle || ""));
    }));

  const midOpts = [
    { id: "summary", label: "Summary", sub: "composite rank", eli5: familyEli5("summary") },
    ...heads.map((hd) => ({
      id: hd.id, label: hd.label,
      sub: (bid === "entrants" && hd.id === "pval") ? "2019 signal" : hd.space,
      eli5: headEli5(hd, bid),
    })),
    { id: "radar", label: "Shape", sub: "per-method radar",
      eli5: "Each edge is a category. Rank 1 sits on the outer ring; last place sits at the centre. Ranks are within this eval set only." },
  ];
  const mid = h("div", { class: "tabs mid", id: "head-tabs", role: "tablist",
    "aria-label": "Output head" },
    midOpts.map((opt) =>
      h("span", { class: "tab-wrap" },
        h("button", {
          class: "tab mid", type: "button", role: "tab",
          id: `head-tab-${opt.id}`,
          "aria-selected": String(dataReady && opt.id === state.midHead),
          "aria-controls": "head-panel",
          "aria-disabled": String(!dataReady),
          disabled: dataReady ? null : "disabled",
          onclick: () => {
            if (!dataReady) return;
            const keep = (opt.id !== "summary" && opt.id !== "radar"
              && state.innerFamily && (state.innerFamily === "summary"
                || (heads.find((hd) => hd.id === opt.id) || { families: [] })
                  .families.includes(state.innerFamily)))
              ? state.innerFamily : null;
            pick({ midHead: opt.id, innerFamily: keep,
                   radarEval: opt.id === "radar" ? state.outerEval : state.radarEval });
          },
        }, opt.label, h("span", { class: "tab-sub" }, opt.sub)),
        helpBtn(`head-${opt.id}`, opt.eli5))));

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
            helpBtn(`fam-${opt.id}`, familyEli5(opt.id)))))
    : h("div", { class: "tabs inner", id: "family-tabs", role: "tablist",
        "aria-label": "Metric family" },
        h("span", { class: "picker-placeholder" },
          dataReady
            ? (state.midHead === "summary" || state.midHead === "radar"
              ? "This pick is complete — a family is not needed."
              : "Pick Count, P-value, or Peak to choose a family — or Summary / Shape above.")
            : "Pick a data set first."));

  return h("div", { class: "nested-wrap picker" },
    h("p", { class: "picker-hint" },
      "Pick a data set, then a head, then a family. Summary is a full view at the first two steps."),
    outer, mid, inner);
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
  return h("section", { class: "panel eval-board", id: "eval-board", role: "tabpanel",
    "aria-labelledby": `eval-tab-${bid}` },
    h("h2", null,
      coded(meta.label, `${meta.protocol} / ${bid}`),
      h("span", { class: "h2-note" }, meta.subtitle),
      helpBtn(`h2-${bid}`, meta.eli5),
      comboHelpBtn(),
      ...readerBadges(meta)),
    (meta.caveats || []).length
      ? h("div", { class: "caveats" },
          h("div", { class: "caveats-title" }, "On this eval set"),
          h("ul", { style: "margin:4px 0;padding:0" },
            meta.caveats.map((c, i) =>
              h("li", null, c, " ", helpBtn(`tabcav-${bid}-${i}`, meta.eli5)))))
      : null,
    strictToggle(bid),
    body);
}

function strictToggle(bid) {
  if (bid !== "main") return null;
  const main = state.data.boards.main;
  if (!main || (main.meta.views || []).length < 2) return null;
  return h("div", { class: "controls" },
    h("span", { class: "chip-group view-toggle" },
      h("span", { class: "chip-label" }, "full-genome chromosomes"),
      ...(main.meta.views || []).map((v) => {
        const labels = main.meta.view_labels || {};
        const label = labels[v] || v;
        const tip = v === "strict"
          ? (main.meta.strict_view && (main.meta.strict_view.eli5 || main.meta.strict_view.note))
          : "Every chromosome in the store. Internal code: default.";
        return h("span", { class: "chip-wrap" },
          h("button", {
            class: "chip", "aria-pressed": String(state.view === v),
            title: `Internal code: ${v}`,
            onclick: () => pick({ view: v }),
          }, label), helpBtn(`view-${v}`, tip));
      })));
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
                         row: null, pending: null, unranked: null });
    }
    const e = byName.get(name);
    if (lineage && !e.lineage) e.lineage = lineage;
    return e;
  };
  for (const row of view.rows) ensure(row.method, row.lineage).row = row;
  for (const u of view.unranked || []) ensure(u.method).unranked = u;
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
      if (cid === "count_arm") return e.lineage === "baseline" || e.lineage === "candi";
      return true;
    }
    if (cid === "summary") return !!(e.row || e.unranked);
    return e.row && rowHasFamily(e.row, cid);
  });
}

function familyKicker(cid, bid) {
  if (cid === "summary") {
    return "Headline ranking on the p-value-space composite (best at top). Methods with partial coverage have no composite rank — incomplete, not last. Count-space ranking lives under the Count head; peak ranking lives under the Peak head.";
  }
  if (cid === "count_arm") {
    return "Count space — Negative Binomial CRPS, ranked separately. Count-space and p-value-space numbers never share an axis.";
  }
  if (cid === "distributional") return DIST_ONELINER;
  if (cid === "peaks") return PEAK_ONELINER;
  if (cid === "pointwise") {
    return bid === "entrants"
      ? "Point-wise scores on 2019 challenge signal. The header letters MSE / Pearson / Spearman match Full genome and Chromosome 21; the truth signal does not. Never translate."
      : "Point-wise scores in store −log10 p. The header letters MSE / Pearson / Spearman match the 2019 tab; the truth signal does not. Never translate.";
  }
  return "";
}

function headKicker(head) {
  if (head.id === "pval") {
    return "P-value head — ranked on the §5.2 composite. Peak metrics are a separate head; count metrics never appear here.";
  }
  if (head.id === "count") {
    return "Count head — ranked on Negative Binomial CRPS, with the oracle-scaled / scale-error split and the ±0.09 noise floor.";
  }
  return PEAK_ONELINER;
}

function summaryBody(bid) {
  const view = viewFor(bid);
  const pending = pendingOf(bid);
  const scored = view.rows.filter((r) => r.rank);
  return h("div", null,
    h("p", { class: "chart-kicker" }, familyKicker("summary", bid),
      helpBtn("composite", familyEli5("summary"))),
    rankBarChart({
      aria: `Ranked methods on ${metaOf(bid).label}, composite`,
      scored,
      pending,
      rankOf: (r) => (r.rank[0] + r.rank[1]) / 2,
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
  const scored = view.rows.filter((r) => familyRankSpread(r, rankCid, view));
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
      note = familyNote(r, rankCid, primary);
    }
    const extra = deviceNote(rankCid, r);
    return extra ? (note ? `${note} · ${extra}` : extra) : note;
  };
  return h("div", { class: "head-summary" },
    h("p", { class: "chart-kicker" }, headKicker(head),
      helpBtn(`head-sum-${head.id}`, head.id === "peak" ? PEAK_ELI5 : headEli5(head, bid))),
    rankBarChart({
      aria: `Ranked methods on ${metaOf(bid).label}, ${head.label} head`,
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
  const scored = view.rows.filter((r) => rowHasFamily(r, cid) && familyRankSpread(r, cid, view));
  const primary = catMetrics(cid, ["ranked"]).filter((m) => !headId || headIdOf(m) === headId)[0];
  const kicker = familyKicker(cid, bid);
  const chart = (!scored.length && !pending.length) ? null : rankBarChart({
      aria: `Ranked methods on ${metaOf(bid).label}, ${familyTabCaption(cid).title}`,
      scored,
      pending,
      rankOf: (r) => {
        const s = familyRankSpread(r, cid, view);
        return s ? (s[0] + s[1]) / 2 : 999;
      },
      labelOf: (r) => r.method,
      noteOf: (r) => {
        const note = familyNote(r, cid, primary);
        const extra = deviceNote(cid, r);
        return extra ? (note ? `${note} · ${extra}` : extra) : note;
      },
      lineageOf: (r) => r.lineage,
    });
  return h("div", null,
    kicker ? h("p", { class: "chart-kicker" }, kicker,
      helpBtn(`kicker-${cid}`, familyEli5(cid))) : null,
    chart,
    familyTable(bid, cid, groups));
}

function familyNote(row, cid, primary) {
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
        helpBtn(`loss-${g.arm}`,
          "The training loss on data the method did not train on. Ranked only among methods that use this same loss family. Count-space NLL and p-value-space NLL never share a table or an axis.")),
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
    return h("p", { class: "empty-note" }, "No loss numbers on this eval set.");
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
          "No covariate-sensitivity numbers on this eval set",
          helpBtn("cov-empty-board", cat.eli5)),
        h("p", { class: "sub" }, measuredNote),
        h("p", null,
          "These numbers live on CANDI-lineage rows scored by the internal bench. ",
          "They are present on another eval set, not this one."),
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
    rankBarChart({
      aria: `CANDI versions on ${metaOf(bid).label}, covariate sensitivity`,
      scored: ordered,
      pending: pendingCandi,
      rankOf: (r) => ordered.indexOf(r) + 1,
      labelOf: (r) => `${r.method} ${r.version || ""}`.trim(),
      methodOf: (r) => r.method,
      noteOf: (r) => familyNote(r, "covariate_diagnostics", rankedDiag),
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
        ? "No scores stamped yet."
        : "No method on this eval set has numbers in this family yet.");
  }

  const head = h("tr", null,
    h("th", { class: "method-col" }, "method"),
    showRank ? h("th", null, cid === "summary" ? "rank" : "family rank") : null,
    showComposite ? h("th", {
      title: "Mean of category sub-scores. Lower is better. A dash is partial coverage, not last place.",
    }, "composite") : null,
    ...groups.flatMap((g) => g.metrics.map((m) =>
      h("th", { class: armClass(m), title: metricTitle(m) },
        h("span", { class: "chip-wrap" },
          m.label, spaceTag(m, bid), helpBtn(`col-${cid}-${metricId(m)}`, metricExplainer(m)))))));

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
      const title = entry.unranked
        ? (entry.unranked.note || "no scores for this chromosome set")
        : "not scored on this eval set";
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
          : h("td", { class: "rank-cell cell-missing" }, EN_DASH));
      }
      if (showComposite) cells.push(compositeCell(entry.row, bid));
      for (const g of groups) {
        for (const m of g.metrics) {
          cells.push(metricCell(entry.row, m, view.rows));
        }
      }
    }
    const tr = h("tr", { class: cls }, cells);
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
      h("span", { class: `badge lineage-${entry.lineage}` },
        LINEAGE_LABEL[entry.lineage] || entry.lineage),
      spreadBadge, peakBadge),
    h("div", { class: "method-line" },
      row ? h("span", { class: "version-chip" }, `${row.version} · ${row.date}`) : null,
      b.position ? h("span", { class: "badge" }, b.position) : null,
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
        coded(metaOf(bid).label, `${metaOf(bid).protocol} / ${bid}`)),
      provDl(entry.row))));
}

/* ------------------------------------------------------------- ranking bars --- */

function rankBarChart({ aria, scored, pending, rankOf, labelOf, noteOf, lineageOf, methodOf }) {
  const n = scored.length;
  const items = scored.slice().sort((a, b) =>
    rankOf(a) - rankOf(b) || labelOf(a).localeCompare(labelOf(b)));
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
        } }, labelOf(row)),
      h("svg:rect", { x: L, y: y + 6, width: w, height: 14, rx: 3, fill: color }),
      h("svg:text", { x: L + w + 8, y: y + 16, "font-size": 11,
        fill: "var(--ink-soft)" },
        `#${rankLabel}` + (noteOf(row) ? ` · ${noteOf(row)}` : "")));
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
  const bits = [`${m.arm ? m.arm + " arm" : "diagnostic"} · `
    + (m.direction ? `${m.direction} is better` : "companion, never ranked")];
  if (m.floor !== null) bits.push(`noise floor ±${m.floor}`);
  if (m.floor_note) bits.push(m.floor_note);
  if (m.calibration_note) bits.push(m.calibration_note);
  if (m.note) bits.push(m.note);
  if (m.eli5) bits.push(m.eli5);
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
  put("strict score json", p.strict_score_json);
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

function radarEdges(view) {
  return RADAR_EDGES.filter((cid) =>
    view.rows.some((r) => categoryRankMid(r, cid, view) !== null));
}

function axisN(view, cid) {
  return view.rows.filter((r) => categoryRankMid(r, cid, view) !== null).length;
}

function radarPanel() {
  const bid = state.outerEval || state.radarEval;
  const ids = boardIds();
  if (!ids.includes(bid)) {
    return h("p", { class: "empty-note" }, "Pick a data set to see per-method shape.");
  }
  const view = viewFor(bid);
  const meta = metaOf(bid);
  const edges = radarEdges(view);
  if (edges.length < 3) {
    return h("div", null,
      h("p", { class: "sub" },
        `${meta.label} has ${edges.length} rankable categor${edges.length === 1 ? "y" : "ies"} — a shape needs three.`));
  }
  const cards = view.rows.map((row) => radarCard(row, view, edges));
  const pending = pendingOf(bid).map((p) =>
    h("div", { class: "radar-card pending-card" },
      h("div", { class: "radar-name" }, methodLink(p.method)),
      h("p", { class: "computing-note" }, p.note || "results computing")));
  return h("div", { id: "radar" },
    h("p", { class: "sub" },
      "Each edge is a category. Rank 1 sits on the outer ring; last place sits at the centre. ",
      "Ranks are computed on ",
      coded(meta.label, `${meta.protocol} / ${bid}`),
      " — not across eval sets."),
    h("div", { class: "radar-grid" }, ...cards, ...pending));
}

function radarCard(row, view, edges) {
  const W = 220, H = 248, cx = 110, cy = 118, R = 78;
  const svg = h("svg:svg", {
    class: "radar-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": `Category ranks for ${row.method}`,
  });
  const nAx = edges.length;
  const pt = (i, t) => {
    const ang = -Math.PI / 2 + i * 2 * Math.PI / nAx;
    return [cx + Math.cos(ang) * R * t, cy + Math.sin(ang) * R * t];
  };
  for (const ring of [0.25, 0.5, 0.75, 1]) {
    const pts = edges.map((_, i) => pt(i, ring).join(",")).join(" ");
    svg.append(h("svg:polygon", {
      points: pts, fill: "none", stroke: "var(--line)", "stroke-width": 1,
    }));
  }
  edges.forEach((cid, i) => {
    const [x, y] = pt(i, 1);
    svg.append(h("svg:line", {
      x1: cx, y1: cy, x2: x, y2: y, stroke: "var(--line)", "stroke-width": 1,
    }));
    const [lx, ly] = pt(i, 1.18);
    const c = catInfo(cid);
    svg.append(h("svg:text", {
      x: lx, y: ly, "text-anchor": "middle", "font-size": 9,
      fill: "var(--ink-soft)",
    }, c ? c.label : cid));
  });
  const verts = [];
  edges.forEach((cid, i) => {
    const rank = categoryRankMid(row, cid, view);
    if (rank === null) return;
    const n = axisN(view, cid) || 1;
    const t = n <= 1 ? 1 : (n - rank + 1) / n;
    verts.push(pt(i, t));
  });
  const colors = { candi: "var(--candi)", rival: "var(--rival)",
                   baseline: "var(--baseline)", entrant: "var(--entrant)" };
  const color = colors[row.lineage] || "var(--ink-soft)";
  if (verts.length >= 3) {
    svg.append(h("svg:polygon", {
      points: verts.map((p) => p.join(",")).join(" "),
      fill: color, "fill-opacity": 0.22, stroke: color, "stroke-width": 1.5,
    }));
  }
  verts.forEach(([x, y]) => {
    svg.append(h("svg:circle", { cx: x, cy: y, r: 2.5, fill: color }));
  });
  return h("div", { class: "radar-card" },
    h("div", { class: "radar-name" }, methodLink(row.method),
      h("span", { class: "version-chip" }, row.version)),
    svg);
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
      title: `Internal code: ${metaOf(id).protocol} / ${id}`,
      onclick: () => { state.climbEval = id; render(); },
    }, metaOf(id).label));

  return h("section", { class: "panel panel-secondary", id: "over-time" },
    h("h2", null, "CANDI progress over time",
      helpBtn("climb",
        "Each CANDI version is a dated point on composite rank. Only CANDI-lineage rows are plotted.")),
    hide,
    h("div", { class: "controls" },
      h("span", { class: "chip-group" },
        h("span", { class: "chip-label" }, "eval set"), ...chips)),
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
    return h("p", null, "No help card recorded for this method.");
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
    const edges = radarEdges(view);
    if (row && edges.length >= 3) {
      bits.push(
        h("div", { class: "card-label" }, "Shape on this eval set"),
        radarCard(row, view, edges));
    }
  }
  return h("div", null, ...bits);
}

function comboTitle(key) {
  const parts = (key || "").split("/");
  const labels = {
    main: "Full genome", dev: "Chromosome 21", entrants: "2019 challenge",
    count: "Count", pval: "P-value", peak: "Peak", summary: "Summary",
    count_arm: "Distributional", pointwise: "Point-wise",
    distributional: "Distributional", loss: "Loss",
    covariate_diagnostics: "Covariate sensitivity", peaks: "Peaks",
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
  const title = card.kind === "method" ? card.id : comboTitle(card.id);
  const body = card.kind === "method"
    ? methodCardBody(card.id, state.help.methods[card.id])
    : comboCardBody(state.help.combos[card.id]);
  return h("div", { class: "card-overlay", onclick: close },
    h("div", { class: "card-sheet", role: "dialog", "aria-label": title,
      onclick: (ev) => ev.stopPropagation() },
      h("div", { class: "card-head" },
        h("h3", null, title),
        h("button", { type: "button", class: "card-close", onclick: close }, "close")),
      body));
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
      h("dd", null, "The row's score json resolved on disk when the row was stamped and `check` re-verifies it wherever the artifact is reachable.")));
}

function renderFooter() {
  const rep = state.data.reproducibility;
  const hashes = boardIds().map((bid) => {
    const meta = metaOf(bid);
    return h("p", null,
      coded(meta.label, `${meta.protocol} / ${bid}`),
      `: store ${meta.frozen.store_manifest_hash} · regime ${meta.frozen.regime_sha256}.`);
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
