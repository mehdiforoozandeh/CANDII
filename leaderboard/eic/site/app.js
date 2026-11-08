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

// The error decomposition of the selected layer: one stacked bar per board entry, in board order.
const DECOMP_MAX = 2.0;
function renderDecomp() {
  const layer = state.data.layers.find((l) => l.id === state.layer);
  document.getElementById("decomp-layer").textContent = LAYER_META[layer.id].title;
  const parts = new Map(layer.decomp.map((d) => [d.name, d]));
  const box = document.getElementById("decomp");
  box.replaceChildren();
  const pct = (x) => `${((Math.min(x, DECOMP_MAX) / DECOMP_MAX) * 100).toFixed(2)}%`;
  for (const r of merged(layer)) {
    const d = parts.get(r.name);
    if (!d) continue;
    const clip = d.mse > DECOMP_MAX;
    const bar = el("div", { class: `dbar${clip ? " clip" : ""}`, style: `width:${pct(d.mse)}` },
      ...[["lvl", d.level], ["shr", d.shared], ["spc", d.specific]].map(([k, v]) =>
        el("i", { class: k, style: `width:${((v / d.mse) * 100).toFixed(2)}%` })));
    const track = el("div", { class: "dtrack" }, bar, el("span", { class: "dref", style: `left:${pct(1)}` }));
    const row = el("div", {
      class: `drow ${r.kind}`, "data-name": r.name,
      title: `${r.name}: calibration ${d.level.toFixed(3)} + shared ${d.shared.toFixed(3)} + ` +
        `specific ${d.specific.toFixed(3)} = ${d.mse.toFixed(3)}` + (d.n_tracks < 51 ? ` (${d.n_tracks} tracks)` : ""),
    }, el("span", { class: "dn" }, `${r.position} · ${r.name}`), track,
      el("span", { class: "dv" }, `${d.mse.toFixed(2)}${clip ? "›" : ""}`));
    row.addEventListener("mouseenter", () => setFocus(r.name));
    row.addEventListener("mouseleave", () => setFocus(null));
    box.append(row);
  }
  const ticks = el("div", { class: "ticks" },
    ...[0, 0.5, 1, 1.5, 2].map((t) => el("span", { style: `left:${pct(t)}` }, t === 1 ? "1 = mean" : String(t))));
  box.append(el("div", { class: "daxis" }, el("span", {}), ticks, el("span", {})));
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
  for (const tr of document.querySelectorAll("#table tbody tr, #decomp .drow")) tr.classList.toggle("hl", tr.dataset.name === state.focus);
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
