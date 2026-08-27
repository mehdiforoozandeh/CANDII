/* CANDI rivals leaderboard — vanilla JS over the compiled leaderboard.json.
 * No framework, no chart library, no external requests. The registry travels inside the
 * payload, so nothing about a metric is hard-coded here (LEADERBOARD_PRD.md §5.1). */
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BOARD_ORDER = ["main", "dev", "entrants"];
const CAT_ORDER = ["pointwise", "distributional", "peaks", "count_arm", "loss"];
const EN_DASH = "–";

const state = {
  data: null,
  board: "main",
  view: "default",
  cats: new Set(CAT_ORDER),
  openProv: new Set(),
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
  if (!(state.board in payload.boards)) state.board = Object.keys(payload.boards)[0];
  renderTabs();
  render();
}

function boardIds() {
  const ids = Object.keys(state.data.boards);
  return ids.sort((a, b) => {
    const [ia, ib] = [BOARD_ORDER.indexOf(a), BOARD_ORDER.indexOf(b)];
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
}

function renderTabs() {
  const nav = document.getElementById("board-tabs");
  nav.replaceChildren(...boardIds().map((bid) => [bid, state.data.boards[bid]]).map(([bid, b]) =>
    h("button", {
      class: "tab", role: "tab", "aria-selected": String(bid === state.board),
      onclick: () => { state.board = bid; state.view = "default"; renderTabs(); render(); },
    }, b.meta.label, h("span", { class: "tab-sub" }, b.meta.subtitle))));
}

/* ------------------------------------------------------------- render --- */

function render() {
  const board = state.data.boards[state.board];
  if (!(state.view in board.views)) state.view = "default";
  const view = board.views[state.view];
  const app = document.getElementById("app");
  app.replaceChildren(
    metaPanel(board),
    ...(view.rows.length === 0 ? [emptyPanel()] : [
      climbPanel(board),
      tablePanel(board, view),
      countPanel(view),
      lineagePanel(view),
    ]),
    legendPanel(),
  );
  renderFooter(board);
}

function metaPanel(board) {
  const meta = board.meta;
  return h("section", { class: "panel" },
    h("div", { class: "meta-row" },
      h("span", { class: "kv" }, h("b", null, meta.protocol), " · ", meta.eval_set.chroms),
      h("span", { class: "kv" }, "pairs: ", meta.eval_set.pairs),
      h("span", { class: "kv" }, "scorer: ", meta.eval_set.scorer),
      h("span", { class: "hash-chip" }, "store ", meta.frozen.store_manifest_hash),
      h("span", { class: "hash-chip" }, "regime ", meta.frozen.regime_sha256)),
    h("div", { class: "caveats" },
      h("div", { class: "caveats-title" }, "Read before quoting"),
      h("ul", { style: "margin:4px 0;padding:0" },
        meta.caveats.map((c) => h("li", null, c)))),
    controls(board));
}

function controls(board) {
  const chips = CAT_ORDER
    .filter((cid) => cid in registry().categories)
    .map((cid) => h("button", {
      class: "chip", "aria-pressed": String(state.cats.has(cid)),
      onclick: () => {
        state.cats.has(cid) ? state.cats.delete(cid) : state.cats.add(cid);
        render();
      },
    }, registry().categories[cid].label));
  const views = (board.meta.views || ["default"]).length > 1
    ? h("span", { class: "chip-group view-toggle" },
        h("span", { class: "chip-label" }, "view"),
        ...board.meta.views.map((v) => h("button", {
          class: "chip", "aria-pressed": String(state.view === v),
          title: v === "strict" ? board.meta.strict_view.note : "All declared eval chromosomes.",
          onclick: () => { state.view = v; render(); },
        }, v === "strict" ? "strict (minus chr19)" : "default")))
    : null;
  return h("div", { class: "controls" },
    h("span", { class: "chip-group" },
      h("span", { class: "chip-label" }, "columns"), ...chips),
    views);
}

function emptyPanel() {
  return h("section", { class: "panel" },
    h("div", { class: "empty-state" },
      h("div", { class: "big" }, "No rows stamped yet"),
      h("p", null, "Rows enter through the gate, never by hand:"),
      h("p", null, h("code", null, "python tools/leaderboard.py add <scores.json> …"))));
}

/* ------------------------------------------------------------- main table --- */

function medalClass(row, rows) {
  if (!row.rank || row.rank[0] !== row.rank[1] || row.rank[0] > 3) return null;
  const k = row.rank[0];
  const shared = rows.some((r) => r !== row && r.rank && r.rank[0] <= k && k <= r.rank[1]);
  return shared ? null : `medal-${k}`; // a medal only when the gap clears the floor
}

function tablePanel(board, view) {
  const cats = CAT_ORDER.filter((cid) =>
    state.cats.has(cid) &&
    catMetrics(cid).some((m) => view.rows.some((r) => rowVal(r, m) !== null)));
  const groupHead = h("tr", { class: "group-head" },
    h("th", { colspan: 3 }),
    cats.map((cid) => {
      const n = catMetrics(cid).length;
      const inComposite = view.categories_in_composite.includes(cid);
      return h("th", { colspan: n },
        registry().categories[cid].label + (inComposite ? "" : " (not in composite)"));
    }));
  const colHead = h("tr", null,
    h("th", null, "rank"),
    h("th", { class: "method-col" }, "method"),
    h("th", { title: "Mean of category sub-scores; a sub-score is the mean rank inside the category. Lower is better; the spread is the best and worst achievable rank given floor ties." }, "composite"),
    cats.flatMap((cid) => catMetrics(cid).map((m) =>
      h("th", { title: metricTitle(m) }, m.label))));

  const bodyRows = view.rows.flatMap((row) => {
    const tr = h("tr", { class: medalClass(row, view.rows) },
      rankCell(row),
      methodCell(row),
      h("td", { class: "num" },
        row.composite
          ? (row.composite[0] === row.composite[1]
              ? row.composite[0].toFixed(2)
              : `${row.composite[0].toFixed(2)}${EN_DASH}${row.composite[1].toFixed(2)}`)
          : "—"),
      cats.flatMap((cid) => catMetrics(cid).map((m) => metricCell(row, m, view.rows))));
    const out = [tr];
    if (state.openProv.has(row.id)) out.push(provRow(row, 3 + cats.reduce((n, c) => n + catMetrics(c).length, 0)));
    return out;
  });

  return h("section", { class: "panel" },
    h("h2", null, `${board.meta.label} board`,
      h("span", { class: "h2-note" }, board.meta.rows_note)),
    view.unranked.length
      ? h("p", { class: "sub" },
          `Not ranked on this view (no ${state.view}-view scores stamped): `
          + view.unranked.map((r) => r.id).join(", "))
      : null,
    h("div", { class: "table-scroll" },
      h("table", { class: "board" },
        h("thead", null, groupHead, colHead),
        h("tbody", null, bodyRows))));
}

function metricTitle(m) {
  const bits = [`${m.arm ? m.arm + " arm" : "diagnostic"} · `
    + (m.direction ? `${m.direction} is better` : "companion, never ranked")];
  if (m.floor !== null) bits.push(`noise floor ±${m.floor}`);
  if (m.floor_note) bits.push(m.floor_note);
  if (m.note) bits.push(m.note);
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
    return h("td", { class: "num cell-missing", title: "absent — never invented" }, "absent");
  }
  const parts = [fmt(v, m)];
  if (m.floor !== null) parts.push(h("span", { class: "floor-suffix" }, ` ±${m.floor}`));
  // delta to the column leader; greyed with "~" when it sits under the metric's floor
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
  return h("td", { class: "num" }, parts);
}

/* CANDI version-over-version arrow on count crps, judged against the seed-alone
 * self-comparison bar (0.1195) — never against the cross-method floor. */
function candiDelta(row) {
  const bar = registry().candi_self_comparison_bar;
  if (row.lineage !== "candi") return null;
  const board = state.data.boards[state.board];
  const versions = (board.climb[row.method] || []);
  const idx = versions.findIndex((e) => e.version === row.version);
  if (idx <= 0) return null;
  const rows = board.views[state.view].rows;
  const here = row.metrics[bar.arm] && row.metrics[bar.arm][bar.key];
  const prevRow = rows.find((r) =>
    r.method === row.method && r.version === versions[idx - 1].version);
  const prev = prevRow && prevRow.metrics[bar.arm] && prevRow.metrics[bar.arm][bar.key];
  if (here === undefined || prev === undefined || here === null || prev === null) return null;
  const d = here - prev;
  const clears = Math.abs(d) >= bar.value;
  const arrow = d < 0 ? "▾" : "▴"; // lower crps is better
  return h("span", {
    class: clears ? (d < 0 ? "verified" : "unverified") : "unverified",
    title: clears
      ? `moved ${d.toFixed(4)} vs ${versions[idx - 1].version} — clears the ${bar.value} seed-alone bar`
      : `moved ${d.toFixed(4)} vs ${versions[idx - 1].version} — under the ${bar.value} seed-alone bar (${bar.note})`,
  }, `${clears ? "" : "~"}${arrow}`);
}

function methodCell(row) {
  const b = row.badges || {};
  return h("td", { class: "method-col" },
    h("div", { class: "method-line" },
      h("span", { class: "method-name" }, row.method),
      h("span", { class: `badge lineage-${row.lineage}` }, row.lineage),
      candiDelta(row)),
    h("div", { class: "method-line" },
      h("span", { class: "version-chip" }, `${row.version} · ${row.date}`),
      b.position ? h("span", { class: "badge" }, b.position) : null,
      b.cell_types ? h("span", { class: "badge" }, b.cell_types) : null,
      row.verified
        ? h("span", { class: "verified", title: "score json resolved when the row was stamped" }, "✓ verified")
        : h("span", { class: "unverified", title: "artifacts not resolved at stamping" }, "unverified"),
      h("button", {
        class: "prov-toggle",
        onclick: () => {
          state.openProv.has(row.id) ? state.openProv.delete(row.id) : state.openProv.add(row.id);
          render();
        },
      }, state.openProv.has(row.id) ? "hide provenance" : "provenance")));
}

function provRow(row, colspan) {
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
  return h("tr", { class: "prov-row" }, h("td", { colspan }, dl));
}

/* ------------------------------------------------------------- sub-boards --- */

function countPanel(view) {
  if (!state.cats.has("count_arm")) return null;
  const sub = view.sub_boards.count_arm;
  if (!sub.rows.length) return null;
  const reg = registry();
  const crps = reg.metrics.find((m) => m.arm === "count" && m.key === "crps");
  const cols = ["crps", "crps_oracle_scaled", "scale_error"];
  const colMs = cols.map((k) => reg.metrics.find((m) => m.arm === "count" && m.key === k));
  const nll = reg.metrics.find((m) => m.arm === "count" && m.key === "nb_nll");
  const asRow = (r) => ({ metrics: { count: r.metrics.count } });
  return h("section", { class: "panel" },
    h("h2", null, "Count arm", h("span", { class: "h2-note" }, "sub-board — not in the composite")),
    h("p", { class: "sub" }, sub.note),
    h("div", { class: "table-scroll" },
      h("table", { class: "board" },
        h("thead", null,
          h("tr", { class: "group-head" },
            h("th", { colspan: 2 }),
            h("th", { colspan: 3 }, "CRPS (NB) — paired split, never separated"),
            h("th", { colspan: 1 }, "loss · per family")),
          h("tr", null,
            h("th", null, "rank"),
            h("th", { class: "method-col" }, "method"),
            colMs.map((m) => h("th", { title: metricTitle(m) }, m.label)),
            h("th", { title: metricTitle(nll) }, nll.label))),
        h("tbody", null, sub.rows.map((r) =>
          h("tr", null,
            h("td", { class: "rank-cell" + (r.rank[0] !== r.rank[1] ? " rank-tied" : "") },
              spreadText(r.rank)),
            h("td", { class: "method-col" },
              h("span", { class: "method-name" }, r.method), " ",
              h("span", { class: "version-chip" }, r.version)),
            colMs.map((m) => metricCell(asRow(r), m, sub.rows.map(asRow))),
            metricCell(asRow(r), nll, [])))))));
}

function lineagePanel(view) {
  const sub = view.sub_boards.candi_lineage;
  if (!sub.rows.length) return null;
  const diags = registry().metrics.filter((m) => metricSlot(m) === "diagnostics");
  return h("section", { class: "panel" },
    h("h2", null, "CANDI lineage",
      h("span", { class: "h2-note" }, "covariate diagnostics — CANDI versions only")),
    h("p", { class: "sub" }, sub.note),
    h("div", { class: "table-scroll" },
      h("table", { class: "board" },
        h("thead", null, h("tr", null,
          h("th", { class: "method-col" }, "version"),
          diags.map((m) => h("th", { title: metricTitle(m) }, m.label)))),
        h("tbody", null, sub.rows.map((r) =>
          h("tr", null,
            h("td", { class: "method-col" },
              h("span", { class: "version-chip" }, `${r.version} · ${r.date}`)),
            diags.map((m) => metricCell({ metrics: r.metrics }, m, []))))))));
}

/* ------------------------------------------------------------- climb chart --- */

function climbPanel(board) {
  const climb = board.climb;
  const methods = Object.keys(climb).filter((m) => climb[m].some((e) => e.composite));
  if (!methods.length) return null;
  const W = 900, H = 260, L = 46, R = 150, T = 18, B = 34;
  const entries = methods.flatMap((m) => climb[m]);
  const dates = entries.map((e) => Date.parse(e.date));
  let [d0, d1] = [Math.min(...dates), Math.max(...dates)];
  if (d0 === d1) { d0 -= 864e5 * 7; d1 += 864e5 * 7; }
  const mids = entries.filter((e) => e.composite)
    .map((e) => (e.composite[0] + e.composite[1]) / 2);
  const rMax = Math.max(2, Math.ceil(Math.max(...entries
    .filter((e) => e.composite).map((e) => e.composite[1]))));
  const x = (t) => L + (t - d0) / (d1 - d0) * (W - L - R);
  const y = (rank) => T + (rank - 1) / (rMax - 1) * (H - T - B); // rank 1 on top
  const svg = h("svg:svg", { class: "climb-svg", viewBox: `0 0 ${W} ${H}`,
    role: "img", "aria-label": "Composite rank over time" });

  // axes + integer rank gridlines
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

  // the leader's composite interval, shaded as the floor band
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
  const placeLabel = (yy) => {   // nudge overlapping right-edge labels apart
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
      // CANDI versions: a labeled line through its dated versions
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
      // a rival/baseline has one score, not a trajectory: a flat dotted line
      const yy = y((series[0].composite[0] + series[0].composite[1]) / 2);
      svg.append(
        h("svg:line", { x1: L, x2: W - R, y1: yy, y2: yy, stroke: color,
          "stroke-width": 1.5, "stroke-dasharray": "2 5" }),
        h("svg:text", { x: W - R + 8, y: placeLabel(yy + 4), "font-size": 11,
          fill: color }, method));
    }
  }
  // x axis dates
  svg.append(
    h("svg:text", { x: L, y: H - 10, "font-size": 11, fill: "var(--ink-faint)" },
      new Date(d0).toISOString().slice(0, 10)),
    h("svg:text", { x: W - R, y: H - 10, "text-anchor": "end", "font-size": 11,
      fill: "var(--ink-faint)" }, new Date(d1).toISOString().slice(0, 10)));

  return h("section", { class: "panel" },
    h("h2", null, "Is CANDI climbing?",
      h("span", { class: "h2-note" }, "composite vs date — shaded band: the leader's rank interval under floor ties")),
    svg,
    h("div", { class: "legend-row" },
      Object.entries(colors).map(([lin, c]) =>
        h("span", null, h("span", { class: "legend-swatch", style: `background:${c}` }), lin))));
}

/* ------------------------------------------------------------- legend + footer --- */

function legendPanel() {
  const reg = registry();
  const bar = reg.candi_self_comparison_bar;
  const pvalCrps = reg.metrics.find((m) => m.arm === "pval" && m.key === "crps");
  return h("section", { class: "panel" },
    h("h2", null, "How to read this board"),
    h("dl", { class: "legend" },
      h("dt", null, `rank "1${EN_DASH}3"`),
      h("dd", null, "The best and worst achievable rank: two rows whose gap sits under a metric's noise floor are tied, and ties propagate through the category means into the composite. A gap under the floor never decides a rank."),
      h("dt", null, "score ± floor"),
      h("dd", null, "The measured noise floor prints inside the cell (count-arm macro CRPS: ±0.09, target-clustered, AGENTS.md §7.2). Deltas under the floor grey out with a ~ prefix."),
      h("dt", null, "medals"),
      h("dd", null, "A row tints gold/silver/bronze only when its rank is unshared — the gap to the next row clears the floor."),
      h("dt", null, "pval CRPS"),
      h("dd", null, pvalCrps.floor === null
        ? "No measured floor yet — " + pvalCrps.floor_note
        : `floor ±${pvalCrps.floor}.`),
      h("dt", null, "CANDI ▴▾ arrows"),
      h("dd", null, `CANDI's own version-over-version arrows use the stricter seed-alone bar (${bar.arm} ${bar.key} ${bar.value}); it is never a cross-method floor.`),
      h("dt", null, "paired columns"),
      h("dd", null, "Count-arm CRPS never renders without its oracle-scaled / scale-error split; pval CRPS never without PIT KS and coverage. Absent means absent — no number is ever invented."),
      h("dt", null, "✓ verified"),
      h("dd", null, "The row's score json resolved on disk when the row was stamped and `check` re-verifies it wherever the artifact is reachable.")));
}

function renderFooter(board) {
  const rep = state.data.reproducibility;
  document.getElementById("footer").replaceChildren(
    h("div", { class: "footer-inner" },
      h("h3", null, "Reproducibility"),
      h("p", null, "Score: ", h("code", null, rep.score_command)),
      h("p", null, "Stamp: ", h("code", null, rep.stamp_command)),
      h("p", null, rep.note),
      h("p", null,
        `Eval set frozen at: store ${board.meta.frozen.store_manifest_hash} · `
        + `regime ${board.meta.frozen.regime_sha256}. `
        + "This page is compiled by tools/leaderboard.py build — a pure function of the "
        + "repo tree, rebuilt by CI on every row that lands on main.")));
}

boot();
