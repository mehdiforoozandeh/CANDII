// The EIC board: reads eic.json (written by tools/eic_board.py) and draws the summary cards, the
// per-layer table and the bump chart. No framework, no external requests.
"use strict";

const LAYER_META = {
  original: {
    no: "Layer 1", title: "Original truth", tab: "Original",
    note: "<b>Raw predictions against the 2019 truth</b> (Additional File 2). The truth was " +
      "processed paired-end while most training data was single-end. The Average baseline " +
      "ranks 3rd. Reproduced exactly: every value and all 25 positions.",
  },
  qn: {
    no: "Layer 2", title: "Quantile normalized", tab: "Normalized",
    note: "<b>The same truth, quantile-normalized per assay</b> in peaks and background " +
      "(Additional File 3); no reprocessing. Reproduced position for position; values within a " +
      "median of 2 × 10<sup>−4</sup>.",
  },
  corrected: {
    no: "Layer 3", title: "Reprocessed + normalized", tab: "Reprocessed + normalized",
    note: "<b>The organizers' corrected board</b> (Additional File 4): 48 truth tracks " +
      "reprocessed single-end, then quantile-normalized. Reproduced exactly: every value and " +
      "all 25 positions.",
  },
};

// Entries drawn as key lines in the bump chart (besides CANDI): the baselines and the 2019 winners.
const KEY_PUBLISHED = new Set([
  "Average", "Avocado", "Hongyang Li and Yuanfang Guan v1", "BrokenNodes_v3", "Lavawizard",
  "Guacamole", "imp",
]);

const state = { data: null, layer: "corrected", variants: false, focus: null };

const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) n.append(k);
  return n;
};
const fmt = (x) => x.toFixed(3);
const slug = (s) => s.replace(/[^a-z0-9]+/gi, "-").toLowerCase();

function candiRows(layer) {
  return layer.candi.filter((c) => c.default || state.variants);
}

// One ranked list per layer: the 25 published entries at their published positions, and each
// CANDI arm placed just above the entry it displaces (it was ranked alone among 26).
function merged(layer) {
  const rows = layer.published.map((p) => ({ ...p, key: p.position }));
  for (const c of candiRows(layer)) rows.push({ ...c, key: c.position - 0.5 + (c.default ? 0 : 0.01) });
  return rows.sort((a, b) => a.key - b.key);
}

function positionIn(layer, name) {
  const hit = layer.published.find((p) => p.name === name) || layer.candi.find((c) => c.name === name);
  return hit ? hit.position : null;
}

function renderCards() {
  const box = document.getElementById("cards");
  box.replaceChildren();
  for (const layer of state.data.layers) {
    const meta = LAYER_META[layer.id];
    const top = layer.candi.find((c) => c.name === "CANDI + tcfloor");
    const raw = layer.candi.find((c) => c.name === "CANDI");
    const avg = layer.published.find((p) => p.name === "Average");
    const card = el("button", {
      class: "card", type: "button", "aria-pressed": String(layer.id === state.layer),
      onclick: () => setLayer(layer.id),
    },
      el("div", { class: "layer-no" }, meta.no),
      el("h3", {}, meta.title),
      el("div", { class: "big" }, el("strong", {}, `#${top.position}`),
        el("span", {}, `of ${top.of} · CANDI + tcfloor`)),
      el("p", { class: "margin" },
        `Score ${fmt(top.score)} vs ${fmt(top.best_other_score)} for ${top.best_other}`),
      el("ul", { class: "rows" },
        el("li", {}, el("span", {}, "CANDI"), el("span", {}, `#${raw.position}`)),
        el("li", {}, el("span", {}, "Average baseline"), el("span", {}, `#${avg.position}`))),
    );
    box.append(card);
  }
}

function renderTabs() {
  const tabs = document.getElementById("tabs");
  tabs.replaceChildren();
  for (const layer of state.data.layers) {
    tabs.append(el("button", {
      role: "tab", type: "button", "aria-selected": String(layer.id === state.layer),
      onclick: () => setLayer(layer.id),
    }, LAYER_META[layer.id].tab));
  }
}

function renderTable() {
  const layers = state.data.layers;
  const i = layers.findIndex((l) => l.id === state.layer);
  const layer = layers[i];
  const prev = i > 0 ? layers[i - 1] : null;
  document.getElementById("layer-note").innerHTML = LAYER_META[layer.id].note;
  const body = document.querySelector("#table tbody");
  body.replaceChildren();
  for (const r of merged(layer)) {
    const tr = el("tr", { class: r.kind, "data-name": r.name });
    tr.addEventListener("mouseenter", () => setFocus(r.name));
    tr.addEventListener("mouseleave", () => setFocus(null));
    const rank = r.kind === "candi"
      ? el("span", { class: "rank" }, String(r.position), el("small", {}, `/${r.of}`))
      : el("span", { class: "rank" }, String(r.position));
    const name = el("div", { class: "name" }, el("strong", {}, r.name));
    if (r.kind === "baseline") name.append(el("span", { class: "tag baseline" }, "baseline"));
    if (r.kind === "candi") name.append(el("span", { class: "tag candi" }, "ours"));
    const cell = el("td", {}, name);
    if (r.kind === "candi") {
      cell.append(el("div", { class: "sub" },
        `Entered alone as a 26th entry · worst bootstrap ${fmt(r.worst_bootstrap)}`));
    }
    let delta = el("span", { class: "delta same" }, "–");
    if (prev) {
      const before = positionIn(prev, r.name);
      if (before != null) {
        const d = before - r.position;
        delta = d === 0 ? el("span", { class: "delta same" }, "0")
          : el("span", { class: `delta ${d > 0 ? "up" : "down"}` }, `${d > 0 ? "▲" : "▼"} ${Math.abs(d)}`);
      }
    }
    const pct = Math.min(100, (r.score / 0.5) * 100);
    tr.append(
      el("td", { class: "num" }, rank),
      cell,
      el("td", {}, el("div", { class: "score" }, el("span", { class: "val" }, fmt(r.score)),
        el("div", { class: "bar" }, el("i", { style: `width:${pct.toFixed(1)}%` })))),
      el("td", { class: "num" }, delta),
    );
    body.append(tr);
  }
  applyFocus();
}

function renderBump() {
  const box = document.getElementById("bump");
  const layers = state.data.layers;
  const W = 900, H = 600, padL = 40, padR = 250, padT = 34, padB = 16;
  const xs = layers.map((_, i) => padL + (i * (W - padL - padR)) / (layers.length - 1));
  // Lines run through each entry's row in the table (its slot), so a CANDI arm ranked 1st of 26
  // sits above the published 1st instead of on top of it. Labels still print the position.
  const slots = layers.map((l) => merged(l));
  const nSlots = Math.max(...slots.map((s) => s.length));
  const y = (slot) => padT + ((slot - 1) * (H - padT - padB)) / (nSlots - 1);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Leaderboard position of every entry on each of the three layers");
  const add = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    svg.append(n);
    return n;
  };
  layers.forEach((l, i) => {
    add("line", { x1: xs[i], x2: xs[i], y1: padT - 8, y2: H - padB, class: "grid" });
    add("text", { x: xs[i], y: 16, "text-anchor": "middle", class: "axis" }, LAYER_META[l.id].title);
  });
  add("text", { x: padL - 16, y: y(1) + 4, "text-anchor": "end", class: "tick" }, "top");
  add("text", { x: padL - 16, y: y(nSlots) + 4, "text-anchor": "end", class: "tick" }, "last");

  const series = new Map();
  slots.forEach((rows, i) => {
    rows.forEach((r, k) => {
      if (!series.has(r.name)) series.set(r.name, { kind: r.kind, pts: [] });
      series.get(r.name).pts.push([i, k + 1, r.position]);
    });
  });
  // grey lines first, key lines on top
  const order = [...series.entries()].sort((a, b) => keyOf(a) - keyOf(b));
  function keyOf([name, s]) { return s.kind === "candi" ? 3 : KEY_PUBLISHED.has(name) ? 2 : 1; }
  const labels = [];
  for (const [name, s] of order) {
    const key = s.kind === "candi" || KEY_PUBLISHED.has(name);
    let cls = `ln ${s.kind}` + (key ? " key" : "");
    if (s.kind === "candi" && name !== "CANDI + tcfloor") cls = "ln candi2 key";
    const d = s.pts.map(([i, pos], k) => `${k ? "L" : "M"}${xs[i]},${y(pos)}`).join(" ");
    const path = add("path", { d, class: cls, "data-name": name });
    path.addEventListener("mouseenter", () => setFocus(name));
    path.addEventListener("mouseleave", () => setFocus(null));
    if (key) {
      for (const [i, pos] of s.pts) {
        const fill = s.kind === "candi" ? "var(--candi)" : s.kind === "baseline" ? "var(--baseline)" : "var(--text-2)";
        add("circle", { cx: xs[i], cy: y(pos), r: 4, fill, class: "dot", "data-name": name });
      }
    }
    const last = s.pts[s.pts.length - 1];
    labels.push({ name, key, y: y(last[1]), pos: last[2] });
  }
  // right-hand labels, nudged apart so none overlap
  labels.sort((a, b) => a.y - b.y || a.name.localeCompare(b.name));
  let prevY = -Infinity;
  for (const lb of labels) {
    const yy = Math.max(lb.y, prevY + 13);
    prevY = yy;
    const t = add("text", { x: xs[xs.length - 1] + 12, y: yy + 4, class: `lbl${lb.key ? " key" : ""}`, "data-name": lb.name },
      `${lb.pos}  ${lb.name}`);
    t.addEventListener("mouseenter", () => setFocus(lb.name));
    t.addEventListener("mouseleave", () => setFocus(null));
  }
  const h = Math.max(H, prevY + 12);
  svg.setAttribute("viewBox", `0 0 ${W} ${h}`);
  box.replaceChildren(svg);
  applyFocus();
}

// The error budget of the selected layer, drawn as Max Libbrecht's sprint 06 report figures.
// Components in eic.json order: shared fine, peak, domain, broad, then specific in the same bands.
const BAND_LABELS = ["≤200 bp", "0.4–3.2 kb", "6.4–51 kb", "≥102 kb"];
const CAL_COLORS = [["offset", "#E69F00"], ["scale", "#56B4E9"], ["nonlinear", "#009E73"]];
const SHARED_COLORS = ["#f2d3e6", "#e2a6c9", "#cf7aab", "#b3568c"];
const SPECIFIC_COLORS = ["#c3dbef", "#84b3da", "#3f86c0", "#0b5a99"];
const EUE_COLORS = {
  sharedDark: "#b3568c", sharedLight: "#ecc6dc", specificDark: "#0b5a99", specificLight: "#bcd6ec",
  added: "#D55E00", calibration: "#E69F00",
};
const BUDGET_MAX = 1.6;  // bars past this are cut, with their length printed
const EUE_RIGHT_MAX = 0.5;

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(parent, tag, attrs = {}, text = null) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  parent.append(n);
  return n;
}

function decompRows(layer) {
  const parts = new Map(layer.decomp.map((d) => [d.name, d]));
  return merged(layer).filter((r) => parts.has(r.name)).map((r) => ({ r, d: parts.get(r.name) }));
}

function rowGroup(svg, r, y, h, width) {
  const g = svgEl(svg, "g", { class: `drow ${r.kind}`, "data-name": r.name });
  svgEl(g, "rect", { class: "rowbg", x: 0, y: y - 2, width, height: h + 4, rx: 4 });
  g.addEventListener("mouseenter", () => setFocus(r.name));
  g.addEventListener("mouseleave", () => setFocus(null));
  return g;
}

// A white zigzag across a cut bar, as in Max's figures.
function zigzag(g, x, y, h) {
  const k = h / 4;
  svgEl(g, "polygon", {
    class: "cut",
    points: `${x - 3},${y} ${x + 2},${y + k} ${x - 3},${y + 2 * k} ${x + 2},${y + 3 * k} ${x - 3},${y + h} ` +
      `${x + 3},${y + h} ${x + 8},${y + 3 * k} ${x + 3},${y + 2 * k} ${x + 8},${y + k} ${x + 3},${y}`,
  });
}

function legend(id, items, rows) {
  const box = document.getElementById(id);
  box.style.gridTemplateRows = `repeat(${rows}, auto)`;
  box.style.gridAutoFlow = "column";
  box.replaceChildren(...items.map((it) => it == null ? el("span", {})
    : el("span", {}, el("i", { style: `background:${it[1]}` }), it[0])));
}

function renderBudget(layer, rows) {
  const svg = document.getElementById("fig-budget");
  svg.replaceChildren();
  const W = 980, L = 262, R = 912, top = 34, rh = 22, bh = 15;
  const x = (v) => L + (Math.min(v, BUDGET_MAX) / BUDGET_MAX) * (R - L);
  svgEl(svg, "text", { class: "ft", x: L, y: 18 },
    `Error budget on the ${LAYER_META[layer.id].title.toLowerCase()} layer (51 blind tracks)`);
  const all = [{ truth: true, name: "(variance of the truth)" }, ...rows];
  all.forEach((row, i) => {
    const y = top + i * rh;
    const segs = row.truth
      ? [[0, "#000"], [0, "#000"], [0, "#000"], ...layer.decomp_truth.map((v, k) =>
        [v, (k < 4 ? SHARED_COLORS : SPECIFIC_COLORS)[k % 4]])]
      : [...CAL_COLORS.map(([k, c]) => [row.d[k], c]),
        ...row.d.bands.map((v, k) => [v, (k < 4 ? SHARED_COLORS : SPECIFIC_COLORS)[k % 4]])];
    const total = row.truth ? layer.decomp_truth.reduce((a, b) => a + b, 0) : row.d.mse;
    const g = row.truth ? svgEl(svg, "g") : rowGroup(svg, row.r, y, bh, W);
    const label = row.truth ? row.name : `${row.r.position} · ${row.r.name}`;
    svgEl(g, "text", { class: `yl ${row.truth ? "truth" : row.r.kind}`, x: L - 8, y: y + bh - 3, "text-anchor": "end" },
      label.length > 38 ? label.slice(0, 37) + "…" : label);
    let acc = 0;
    for (const [v, color] of segs) {
      if (v <= 0 || acc >= BUDGET_MAX) { acc += Math.max(v, 0); continue; }
      svgEl(g, "rect", { x: x(acc), y, width: Math.max(x(acc + v) - x(acc), 0), height: bh, fill: color });
      acc += v;
    }
    const cut = total > BUDGET_MAX;
    if (cut) zigzag(g, R - 14, y, bh);
    svgEl(g, "text", { class: "val", x: x(total) + 6, y: y + bh - 3 }, total.toFixed(2));
    const t = row.truth ? "" : `${row.r.name}: offset ${row.d.offset.toFixed(3)}, scale ${row.d.scale.toFixed(3)}, ` +
      `nonlinear ${row.d.nonlinear.toFixed(3)}, shared ${row.d.shared.toFixed(3)}, specific ` +
      `${row.d.specific.toFixed(3)}; normalized MSE ${row.d.mse.toFixed(3)}` +
      (row.d.n_tracks < 51 ? ` (${row.d.n_tracks} tracks)` : "");
    if (t) svgEl(g, "title", {}, t);
  });
  const yb = top + all.length * rh + 4;
  svgEl(svg, "line", { class: "axis", x1: L, y1: top - 6, x2: L, y2: yb });
  svgEl(svg, "line", { class: "axis", x1: L, y1: yb, x2: R, y2: yb });
  svgEl(svg, "line", { class: "ref", x1: x(1), y1: top - 6, x2: x(1), y2: yb });
  for (let t = 0; t <= BUDGET_MAX + 1e-9; t += 0.2) {
    svgEl(svg, "line", { class: "axis", x1: x(t), y1: yb, x2: x(t), y2: yb + 4 });
    svgEl(svg, "text", { class: "tick", x: x(t), y: yb + 17, "text-anchor": "middle" }, t.toFixed(1));
  }
  svgEl(svg, "text", { class: "xt", x: (L + R) / 2, y: yb + 38, "text-anchor": "middle" },
    "normalized MSE (fraction of the truth's variance; mean over the 51 blind tracks)");
  svg.setAttribute("viewBox", `0 0 ${W} ${yb + 48}`);
  legend("legend-budget", [
    ...CAL_COLORS, null,
    ...BAND_LABELS.map((b, k) => [`shared ${b}`, SHARED_COLORS[k]]),
    ...BAND_LABELS.map((b, k) => [`specific ${b}`, SPECIFIC_COLORS[k]]),
  ], 4);
}

function renderEue(layer, rows) {
  const svg = document.getElementById("fig-eue");
  svg.replaceChildren();
  const W = 980, L = 262, LR = 772, RL = 800, RR = 930, top = 112, rh = 22, bh = 15;
  const x = (v) => L + v * (LR - L);
  const xr = (v) => RL + (Math.min(v, EUE_RIGHT_MAX) / EUE_RIGHT_MAX) * (RR - RL);
  const tv = layer.decomp_truth;
  const edges = [0];
  for (const v of tv) edges.push(edges[edges.length - 1] + v);
  const scale = 1 / edges[8];   // the split sums to 1 up to rounding; draw it to span exactly 0..1
  const ex = (k) => x(edges[k] * scale);
  svgEl(svg, "text", { class: "ft", x: L, y: 18 },
    `Explained and unexplained variance on the ${LAYER_META[layer.id].title.toLowerCase()} layer`);
  // header: each component's band and share of the truth's variance
  // labels anchored at each component's left edge, as in Max's figure; a label too close to the
  // next one moves left so the slanted labels never overlap
  const anchors = tv.map((_, k) => ex(k) + 4);
  for (let k = 6; k >= 0; k--) anchors[k] = Math.min(anchors[k], anchors[k + 1] - 15);
  tv.forEach((v, k) => {
    const cx = anchors[k];
    svgEl(svg, "path", { class: "axis", fill: "none", d: `M${ex(k)},${top - 4} V${top - 9} L${cx},${top - 13}` });
    svgEl(svg, "text", { class: "hdr", x: cx, y: top - 15, transform: `rotate(-55 ${cx} ${top - 15})` },
      `${BAND_LABELS[k % 4]} (${Math.round(v * scale * 100)}%)`);
  });
  rows.forEach(({ r, d }, i) => {
    const y = top + i * rh;
    const g = rowGroup(svg, r, y, bh, W);
    const label = `${r.position} · ${r.name}`;
    svgEl(g, "text", { class: `yl ${r.kind}`, x: L - 8, y: y + bh - 3, "text-anchor": "end" },
      label.length > 38 ? label.slice(0, 37) + "…" : label);
    tv.forEach((_, k) => {
      const e = d.explained[k], u = d.unexplained[k];
      const f = e + u > 0 ? e / (e + u) : 0;
      const x0 = ex(k), x1 = ex(k + 1), xm = x0 + f * (x1 - x0);
      const shared = k < 4;
      svgEl(g, "rect", { x: x0, y, width: xm - x0, height: bh, fill: shared ? EUE_COLORS.sharedDark : EUE_COLORS.specificDark });
      svgEl(g, "rect", { x: xm, y, width: x1 - xm, height: bh, fill: shared ? EUE_COLORS.sharedLight : EUE_COLORS.specificLight });
    });
    const tot = d.added + d.level;
    svgEl(g, "rect", { x: xr(0), y, width: xr(d.added) - xr(0), height: bh, fill: EUE_COLORS.added });
    if (d.added < EUE_RIGHT_MAX)
      svgEl(g, "rect", { x: xr(d.added), y, width: xr(tot) - xr(d.added), height: bh, fill: EUE_COLORS.calibration });
    if (tot > EUE_RIGHT_MAX) {
      zigzag(g, RR - 12, y, bh);
      svgEl(g, "text", { class: "val", x: RR + 5, y: y + bh - 3 }, tot.toFixed(2));
    }
    const ex8 = d.explained.reduce((a, b) => a + b, 0), un8 = d.unexplained.reduce((a, b) => a + b, 0);
    svgEl(g, "title", {}, `${r.name}: explained ${ex8.toFixed(3)}, unexplained ${un8.toFixed(3)}, ` +
      `added ${d.added.toFixed(3)}, calibration ${d.level.toFixed(3)}; normalized MSE ${d.mse.toFixed(3)}`);
  });
  const yb = top + rows.length * rh + 4;
  for (let k = 1; k < 8; k++) if (k !== 4) svgEl(svg, "line", { class: "csep", x1: ex(k), y1: top - 2, x2: ex(k), y2: yb - 4 });
  svgEl(svg, "line", { class: "sep", x1: ex(4), y1: top - 6, x2: ex(4), y2: yb });
  // brackets under the two halves
  const sh = edges[4] * scale;
  [[0, 4, `shared parts (${Math.round(sh * 100)}%)`], [4, 8, `specific parts (${Math.round((1 - sh) * 100)}%)`]]
    .forEach(([a, b, t]) => {
      svgEl(svg, "path", { class: "axis", fill: "none", d: `M${ex(a) + 4},${yb + 2} v4 H${ex(b) - 4} v-4` });
      svgEl(svg, "text", { class: "tick", x: (ex(a) + ex(b)) / 2, y: yb + 20, "text-anchor": "middle" }, t);
    });
  const ya = yb + 30;
  svgEl(svg, "line", { class: "axis", x1: L, y1: top - 6, x2: L, y2: ya });
  svgEl(svg, "line", { class: "axis", x1: L, y1: ya, x2: LR, y2: ya });
  for (let t = 0; t <= 1 + 1e-9; t += 0.2) {
    svgEl(svg, "line", { class: "axis", x1: x(t), y1: ya, x2: x(t), y2: ya + 4 });
    svgEl(svg, "text", { class: "tick", x: x(t), y: ya + 17, "text-anchor": "middle" }, t.toFixed(1));
  }
  svgEl(svg, "text", { class: "xt", x: (L + LR) / 2, y: ya + 38, "text-anchor": "middle" },
    "fraction of the truth's variance");
  svgEl(svg, "line", { class: "axis", x1: RL, y1: ya, x2: RR, y2: ya });
  for (const t of [0, 0.25, 0.5]) {
    svgEl(svg, "line", { class: "axis", x1: xr(t), y1: ya, x2: xr(t), y2: ya + 4 });
    svgEl(svg, "text", { class: "tick", x: xr(t), y: ya + 17, "text-anchor": "middle" }, String(t));
  }
  svgEl(svg, "text", { class: "xt", x: (RL + RR) / 2, y: ya + 38, "text-anchor": "middle" }, "added + calibration");
  svgEl(svg, "text", { class: "xt", x: (RL + RR) / 2, y: ya + 53, "text-anchor": "middle" }, "error");
  svg.setAttribute("viewBox", `0 0 ${W} ${ya + 62}`);
  legend("legend-eue", [
    ["explained (shared)", EUE_COLORS.sharedDark], ["explained (specific)", EUE_COLORS.specificDark],
    ["unexplained (shared)", EUE_COLORS.sharedLight], ["unexplained (specific)", EUE_COLORS.specificLight],
    ["added by the prediction", EUE_COLORS.added], ["calibration error", EUE_COLORS.calibration],
  ], 2);
}

function renderDecomp() {
  const layer = state.data.layers.find((l) => l.id === state.layer);
  document.getElementById("decomp-layer").textContent = LAYER_META[layer.id].title;
  const rows = decompRows(layer);
  renderBudget(layer, rows);
  renderEue(layer, rows);
  applyFocus();
}

function setFocus(name) {
  state.focus = name;
  applyFocus();
}

function applyFocus() {
  const bump = document.getElementById("bump");
  bump.classList.toggle("focus", state.focus != null);
  for (const n of bump.querySelectorAll("[data-name]")) n.classList.toggle("on", n.dataset.name === state.focus);
  for (const tr of document.querySelectorAll("#table tbody tr, .fig .drow")) tr.classList.toggle("hl", tr.dataset.name === state.focus);
}

function setLayer(id) {
  state.layer = id;
  try { localStorage.setItem("eic-layer", id); } catch (e) { /* storage may be blocked */ }
  renderCards();
  renderTabs();
  renderTable();
  renderDecomp();
}

function renderAll() {
  renderCards();
  renderTabs();
  renderTable();
  renderBump();
  renderDecomp();
}

async function main() {
  try {
    const saved = localStorage.getItem("eic-layer");
    if (saved && LAYER_META[saved]) state.layer = saved;
  } catch (e) { /* storage may be blocked */ }
  const res = await fetch("eic.json");
  state.data = await res.json();
  document.getElementById("variants").addEventListener("change", (ev) => {
    state.variants = ev.target.checked;
    renderTable();
    renderBump();
  });
  renderAll();
}

main().catch((err) => {
  document.getElementById("cards").replaceChildren(el("p", { class: "muted" }, `Could not load eic.json: ${err}`));
});
