/* CANDI imputation leaderboard — vanilla JS over the compiled leaderboard.json.
 * No framework, no chart library, no external requests. The registry travels inside the
 * payload, so nothing about a metric is hard-coded here (LEADERBOARD_PRD.md §5.1). */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BOARD_ORDER = ["main", "dev", "entrants"];
const FAMILY_TABS = [
  "summary", "pointwise", "distributional", "peaks",
  "count_arm", "loss", "covariate_diagnostics",
];
const RADAR_EDGES = ["pointwise", "distributional", "peaks", "count_arm"];
const EN_DASH = "–";
const LINEAGE_LABEL = {
  candi: "CANDI", rival: "retrained rival",
  baseline: "baseline", entrant: "2019 entrant",
};

const state = {
  data: null,
  view: "default",
  outerEval: "main",
  innerFamily: "summary",
  radarEval: "main",
  climbEval: "main",
  openProv: new Set(),
  openHelp: null,
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
function armClass(m) {
  return m.arm === "count" ? "arm-count" : m.arm === "pval" ? "arm-pval" : "";
}
function spaceTag(m) {
  const s = m.arm === "count" ? "counts" : m.arm === "pval" ? "p-value" : "diagnostic";
  return h("span", { class: "space-tag" }, s);
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
  const ids = Object.keys(payload.boards);
  if (!ids.includes(state.outerEval)) state.outerEval = ids[0];
  if (!(state.radarEval in payload.boards)) state.radarEval = ids[0];
  if (!(state.climbEval in payload.boards)) state.climbEval = ids[0];
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

function metricGroups(cid) {
  const all = catMetrics(cid);
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
  app.replaceChildren(
    introPanel(),
    nestedBoard(),
    radarPanel(),
    climbPanel(),
    legendPanel(),
  );
  renderFooter();
}

function introPanel() {
  const anyCaveats = boardIds().flatMap((bid) =>
    (metaOf(bid).caveats || []).map((c) => [bid, c]));
  return h("section", { class: "panel" },
    h("h2", null, "How this page is laid out",
      helpBtn("layout",
        "Outer tabs pick the eval set (what was scored). Inner tabs pick the kind of number. Count-space and p-value-space numbers never share a table or a chart axis. Grey rows with no numbers mean results are still computing.")),
    h("p", { class: "sub" },
      "Each inner tab is one family: a ranking chart (best at top) and a compact table of just that family's metrics. ",
      "CANDI's version-over-version chart sits at the bottom — it is not the headline."),
    h("div", { class: "caveats" },
      h("div", { class: "caveats-title" }, "Read before quoting"),
      h("ul", { style: "margin:4px 0;padding:0" },
        anyCaveats.map(([bid, c], i) =>
          h("li", null, coded(metaOf(bid).label, `${metaOf(bid).protocol} / ${bid}`),
            " — ", c, " ", helpBtn(`cav-${bid}-${i}`, metaOf(bid).eli5))))));
}

function readerBadges(meta) {
  return (meta.reader_badges || []).map((b, i) =>
    h("span", { class: "badge badge-warn", title: `Internal code lives in the ?` },
      b.label, helpBtn(`badge-${meta.protocol}-${i}`, b.eli5)));
}

function familyTabCaption(cid) {
  if (cid === "summary") return { title: "Summary", sub: "composite rank" };
  const c = catInfo(cid);
  if (!c) return { title: cid, sub: "" };
  const sub = (c.space_label && c.space_label.toLowerCase() !== c.label.toLowerCase())
    ? c.space_label : "";
  return { title: c.label, sub };
}

function familyEli5(cid) {
  if (cid === "summary") {
    return "Composite is the mean of category mean-ranks for categories at least one method on this eval set fully covers. A method missing any of those categories is incomplete, not last: the composite cell is a dash, and it is left out of the headline ranking. It still ranks inside every category it has numbers for. Lower is better.";
  }
  const c = catInfo(cid);
  return c && c.eli5;
}

/* ------------------------------------------------------------- nested tabs --- */

function nestedBoard() {
  const ids = boardIds();
  if (!ids.includes(state.outerEval)) state.outerEval = ids[0];
  const bid = state.outerEval;
  const meta = metaOf(bid);
  const families = FAMILY_TABS.filter((cid) =>
    cid === "summary" || cid in registry().categories);
  if (!families.includes(state.innerFamily)) state.innerFamily = "summary";

  const outer = h("div", { class: "tabs outer", id: "eval-tabs", role: "tablist",
    "aria-label": "Eval set" },
    ids.map((id) => {
      const m = metaOf(id);
      return h("button", {
        class: "tab outer", type: "button", role: "tab",
        id: `eval-tab-${id}`,
        "aria-selected": String(id === bid),
        "aria-controls": "eval-board",
        title: `Internal code: ${m.protocol} / ${id}`,
        onclick: () => { state.outerEval = id; render(); },
      }, coded(m.label, `${m.protocol} / ${id}`),
        h("span", { class: "tab-sub" }, m.subtitle || ""));
    }));

  const inner = h("div", { class: "tabs inner", id: "family-tabs", role: "tablist",
    "aria-label": "Metric family" },
    families.map((cid) => {
      const cap = familyTabCaption(cid);
      return h("span", { class: "tab-wrap" },
        h("button", {
          class: "tab inner", type: "button", role: "tab",
          id: `fam-tab-${cid}`,
          "aria-selected": String(cid === state.innerFamily),
          "aria-controls": "family-panel",
          onclick: () => { state.innerFamily = cid; render(); },
        }, cap.title,
          cap.sub ? h("span", { class: "tab-sub" }, cap.sub) : null),
        helpBtn(`fam-${cid}`, familyEli5(cid)));
    }));

  return h("div", { class: "nested-wrap" },
    outer,
    h("section", { class: "panel eval-board", id: "eval-board", role: "tabpanel",
      "aria-labelledby": `eval-tab-${bid}` },
      h("h2", null,
        coded(meta.label, `${meta.protocol} / ${bid}`),
        h("span", { class: "h2-note" }, meta.subtitle),
        helpBtn(`h2-${bid}`, meta.eli5),
        ...readerBadges(meta)),
      (meta.caveats || []).length
        ? h("div", { class: "caveats" },
            h("div", { class: "caveats-title" }, "On this eval set"),
            h("ul", { style: "margin:4px 0;padding:0" },
              meta.caveats.map((c, i) =>
                h("li", null, c, " ", helpBtn(`tabcav-${bid}-${i}`, meta.eli5)))))
        : null,
      strictToggle(bid),
      inner,
      h("div", { id: "family-panel", role: "tabpanel",
        "aria-labelledby": `fam-tab-${state.innerFamily}` },
        familyBody(bid, state.innerFamily))));
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
            onclick: () => { state.view = v; render(); },
          }, label), helpBtn(`view-${v}`, tip));
      })));
}

function familyBody(bid, cid) {
  if (cid === "covariate_diagnostics") return covariateBody(bid);
  if (cid === "loss") return lossBody(bid);
  if (cid === "summary") return summaryBody(bid);
  return familyChartAndTable(bid, cid);
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

function familyKicker(cid) {
  if (cid === "summary") {
    return "Headline ranking on the p-value-space composite (best at top). Methods with partial coverage have no composite rank — incomplete, not last. Count-space ranking lives in its own tab.";
  }
  if (cid === "count_arm") {
    return "Count space — Negative Binomial CRPS, ranked separately. Count-space and p-value-space numbers never share an axis.";
  }
  if (cid === "distributional") {
    return "Distributional scores in p-value space. Rank on CRPS; the absolute value is ordering only. Always shown with PIT KS and 95% coverage.";
  }
  if (cid === "peaks") {
    return "Peak scores in p-value space. AUPRC is always shown with the peak base rate.";
  }
  if (cid === "pointwise") {
    return "Point-wise scores in p-value space. Every method that emits a point track can appear here.";
  }
  return "";
}

function summaryBody(bid) {
  const view = viewFor(bid);
  const pending = pendingOf(bid);
  const scored = view.rows.filter((r) => r.rank);
  return h("div", null,
    h("p", { class: "chart-kicker" }, familyKicker("summary"),
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

function familyChartAndTable(bid, cid) {
  const view = viewFor(bid);
  const pending = cid === "count_arm"
    ? pendingOf(bid).filter((p) => p.lineage === "baseline" || p.lineage === "candi")
    : pendingOf(bid);
  const groups = metricGroups(cid);
  const scored = view.rows.filter((r) => rowHasFamily(r, cid) && familyRankSpread(r, cid, view));
  const primary = catMetrics(cid, ["ranked"])[0];
  const kicker = familyKicker(cid);
  const chart = (!scored.length && !pending.length) ? null : rankBarChart({
      aria: `Ranked methods on ${metaOf(bid).label}, ${familyTabCaption(cid).title}`,
      scored,
      pending,
      rankOf: (r) => {
        const s = familyRankSpread(r, cid, view);
        return s ? (s[0] + s[1]) / 2 : 999;
      },
      labelOf: (r) => r.method,
      noteOf: (r) => familyNote(r, cid, primary),
      lineageOf: (r) => r.lineage,
    });
  return h("div", null,
    kicker ? h("p", { class: "chart-kicker" }, kicker,
      helpBtn(`kicker-${cid}`, catInfo(cid) && catInfo(cid).eli5)) : null,
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
  return s;
}

function lossBody(bid) {
  const view = viewFor(bid);
  const pending = pendingOf(bid);
  const groups = metricGroups("loss");
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

  if (!has) {
    const scorers = cover.scorers || [];
    const lineages = cover.lineages || [];
    if (nGlobal > 0) {
      return h("div", { class: "cov-absent" },
        h("p", { class: "cov-absent-title" },
          "No covariate-sensitivity numbers on this eval set",
          helpBtn("cov-empty-board", cat.eli5)),
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
    h("p", { class: "chart-kicker" }, cat.note,
      helpBtn("cov", cat.eli5)),
    rankBarChart({
      aria: `CANDI versions on ${metaOf(bid).label}, covariate sensitivity`,
      scored: ordered,
      pending: pendingCandi,
      rankOf: (r) => ordered.indexOf(r) + 1,
      labelOf: (r) => `${r.method} ${r.version || ""}`.trim(),
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
  const showComposite = cid === "summary";
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
      h("th", { class: armClass(m), title: metricTitle(m) }, m.label, spaceTag(m)))));

  const bodyRows = entries.flatMap((entry) => {
    const pending = entry.pending && !entry.row;
    const cls = pending
      ? "pending-row"
      : (cid === "summary" ? medalClass(entry.row, view) : null);
    const cells = [methodCell(entry, bid)];
    if (pending) {
      const rest = nCols - 1;
      cells.push(h("td", { class: "rank-cell cell-pending", colspan: rest,
        title: entry.pending.note || "results computing" },
        "results computing"));
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

function methodCell(entry, bid) {
  const row = entry.row;
  const b = (row && row.badges) || {};
  const pendingNote = entry.pending && !row
    ? (metaOf(bid).label)
    : null;
  return h("td", { class: "method-col" },
    h("div", { class: "method-line" },
      h("span", { class: "method-name" }, entry.method),
      h("span", { class: `badge lineage-${entry.lineage}` },
        LINEAGE_LABEL[entry.lineage] || entry.lineage)),
    h("div", { class: "method-line" },
      row ? h("span", { class: "version-chip" }, `${row.version} · ${row.date}`) : null,
      b.position ? h("span", { class: "badge" }, b.position) : null,
      b.cell_types ? h("span", { class: "badge" }, b.cell_types) : null,
      row && row.verified
        ? h("span", { class: "verified", title: "score json resolved when the row was stamped" },
            "✓ verified")
        : row
          ? h("span", { class: "unverified", title: "artifacts not resolved at stamping" },
              "unverified")
          : null,
      pendingNote
        ? h("span", { class: "badge badge-pending" }, "computing · " + pendingNote)
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

function rankBarChart({ aria, scored, pending, rankOf, labelOf, noteOf, lineageOf }) {
  const n = scored.length;
  const items = scored.slice().sort((a, b) =>
    rankOf(a) - rankOf(b) || labelOf(a).localeCompare(labelOf(b)));
  const extra = pending || [];
  if (!items.length && !extra.length) {
    return h("p", { class: "empty-note" }, "No scores stamped yet.");
  }
  const rowH = 26, L = 170, R = 110, T = 6, W = 760;
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
    svg.append(
      h("svg:text", { x: L - 8, y: y + 16, "text-anchor": "end",
        "font-size": 12, fill: "var(--ink)" }, labelOf(row)),
      h("svg:rect", { x: L, y: y + 6, width: w, height: 14, rx: 3, fill: color }),
      h("svg:text", { x: L + w + 8, y: y + 16, "font-size": 11,
        fill: "var(--ink-soft)" },
        `#${rankLabel}` + (noteOf(row) ? ` · ${noteOf(row)}` : "")));
  });
  extra.forEach((p, i) => {
    const y = T + (items.length + i) * rowH;
    svg.append(
      h("svg:text", { x: L - 8, y: y + 16, "text-anchor": "end",
        "font-size": 12, fill: "var(--ink-faint)" }, p.method),
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
  const ids = boardIds();
  if (!ids.includes(state.radarEval)) state.radarEval = ids[0];
  const bid = state.radarEval;
  const view = viewFor(bid);
  const meta = metaOf(bid);
  const edges = radarEdges(view);
  const chips = ids.map((id) =>
    h("button", {
      class: "chip", "aria-pressed": String(id === bid),
      title: `Internal code: ${metaOf(id).protocol} / ${id}`,
      onclick: () => { state.radarEval = id; render(); },
    }, metaOf(id).label));
  if (edges.length < 3) {
    return h("section", { class: "panel" },
      h("h2", null, "Per-method shape",
        helpBtn("radar-need",
          "A radar needs at least three categories with ranks on one eval set. Rank 1 sits on the outer ring; last place sits at the centre. Ranks are within this eval set only.")),
      h("div", { class: "controls" },
        h("span", { class: "chip-group" },
          h("span", { class: "chip-label" }, "ranks computed on"), ...chips)),
      h("p", { class: "sub" },
        `${meta.label} has ${edges.length} rankable categor${edges.length === 1 ? "y" : "ies"} — a shape needs three.`));
  }
  const cards = view.rows.map((row) => radarCard(row, view, edges));
  const pending = pendingOf(bid).map((p) =>
    h("div", { class: "radar-card pending-card" },
      h("div", { class: "radar-name" }, p.method),
      h("p", { class: "computing-note" }, p.note || "results computing")));
  return h("section", { class: "panel", id: "radar" },
    h("h2", null, "Per-method shape",
      helpBtn("radar",
        "Each edge is a category. The value is the method's rank in that category on one eval set (1 = best = outer ring; last place = centre). Count-space and p-value-space are different edges, never a shared axis of raw scores.")),
    h("p", { class: "sub" },
      "Ranks are computed on ",
      coded(meta.label, `${meta.protocol} / ${bid}`),
      " — not across eval sets."),
    h("div", { class: "controls" },
      h("span", { class: "chip-group" },
        h("span", { class: "chip-label" }, "ranks computed on"), ...chips)),
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
    h("div", { class: "radar-name" }, row.method,
      h("span", { class: "version-chip" }, row.version)),
    svg);
}

/* ------------------------------------------------------------- climb --- */

function climbPanel() {
  const ids = boardIds().filter((bid) => {
    const climb = state.data.boards[bid].climb;
    return Object.keys(climb).some((m) => climb[m].some((e) => e.composite));
  });
  if (!ids.length) return null;
  if (!ids.includes(state.climbEval)) state.climbEval = ids[0];
  const bid = state.climbEval;
  const board = state.data.boards[bid];
  const climb = board.climb;
  const methods = Object.keys(climb).filter((m) => climb[m].some((e) => e.composite));
  const W = 900, H = 260, L = 46, R = 150, T = 18, B = 34;
  const entries = methods.flatMap((m) => climb[m]);
  const dates = entries.map((e) => Date.parse(e.date));
  let [d0, d1] = [Math.min(...dates), Math.max(...dates)];
  if (d0 === d1) { d0 -= 864e5 * 7; d1 += 864e5 * 7; }
  const rMax = Math.max(2, Math.ceil(Math.max(...entries
    .filter((e) => e.composite).map((e) => e.composite[1]))));
  const x = (t) => L + (t - d0) / (d1 - d0) * (W - L - R);
  const y = (rank) => T + (rank - 1) / (rMax - 1) * (H - T - B);
  const svg = h("svg:svg", { class: "climb-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": "Composite rank over time" });

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

  const colors = { candi: "var(--candi)", rival: "var(--rival)",
                   baseline: "var(--baseline)", entrant: "var(--entrant)" };
  let labelY = [];
  const placeLabel = (yy) => {
    let out = yy;
    while (labelY.some((used) => Math.abs(used - out) < 13)) out += 13;
    labelY.push(out);
    return out;
  };
  for (const method of methods.sort()) {
    const series = climb[method].filter((e) => e.composite);
    const lineage = series[0].lineage;
    const color = colors[lineage] || "var(--ink-soft)";
    if (lineage === "candi" && series.length > 0) {
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
      svg.append(h("svg:text", { x: W - R + 8, y: placeLabel(last[1] + 4),
        "font-size": 11, "font-weight": 700, fill: color }, method));
    } else {
      const yy = y((series[0].composite[0] + series[0].composite[1]) / 2);
      svg.append(
        h("svg:line", { x1: L, x2: W - R, y1: yy, y2: yy, stroke: color,
          "stroke-width": 1.5, "stroke-dasharray": "2 5" }),
        h("svg:text", { x: W - R + 8, y: placeLabel(yy + 4), "font-size": 11,
          fill: color }, method));
    }
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
    h("h2", null, "CANDI versions over time",
      h("span", { class: "h2-note" }, "secondary — not the headline"),
      helpBtn("climb",
        "Each CANDI version is a dated point. Other methods are flat dotted lines (one score, not a trajectory). The shaded band is the leader's rank interval under noise-floor ties.")),
    h("div", { class: "controls" },
      h("span", { class: "chip-group" },
        h("span", { class: "chip-label" }, "eval set"), ...chips)),
    svg,
    h("div", { class: "legend-row" },
      Object.entries(colors).map(([lin, c]) =>
        h("span", null, h("span", { class: "legend-swatch", style: `background:${c}` }),
          LINEAGE_LABEL[lin] || lin))));
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
      h("dd", null, "A grey row or cell with no numbers: that method is a known entry whose scores have not landed yet. It is not a silent omission."),
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
