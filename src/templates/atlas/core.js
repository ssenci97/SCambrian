const D = JSON.parse(document.getElementById("payload").textContent);
const C = D.config, P = D.pages, $ = id => document.getElementById(id);
const FT = Object.keys(D.features || {}), DESC = D.desc ? D.desc.split("\n") : null, FOUND_TERMS = 8, NET_EDGES = 2500;
const FONT_KEYS = [["ui", "Interface"], ["title", "Plot titles"], ["axis", "Axis titles"], ["tick", "Tick labels"], ["legend", "Legend"], ["hover", "Hover boxes"], ["label", "2D cluster labels"], ["net", "Network labels"], ["sankey", "Flux node labels"], ["star", "Bar asterisks"], ["inspect", "Inspector"]];
const SPECIAL = [["unassigned", "Unassigned"], ["other", "Pooled \"other\""], ["minor", "Uncoloured clusters"], ["ghost", "Ghost points"], ["ring", "Find ring"], ["edge", "Network edges"], ["low", "Scale low"], ["high", "Scale high"]];
const STORE = "scambrian-atlas:" + D.title;
const COMMUNITY = "network communities";
function defaults() {
  return {
    font: {ui: C.font_ui, title: C.font_title, axis: C.font_axis, tick: C.font_tick, legend: C.font_legend, hover: C.font_hover, label: C.font_label, net: C.font_net, sankey: C.font_sankey, star: C.font_tick, inspect: C.font_ui}, scale: 1,
    col: {palette: [...C.palette], unassigned: C.color_bg, other: C.color_other, minor: C.color_minor, ghost: C.color_ghost, ring: "#000000", edge: "#9aa1ad", low: C.quant[0], high: C.quant[1]},
    ov: {}, z: {},
    mk: {size: C.marker_size, alpha: C.alpha, ghost: true, ghostAlpha: C.ghost_alpha, ring: 14, labels: true, maxLabels: C.max_labels, maxColored: C.max_colored},
    lay: {stack: "auto", left: 300, right: 320, fluxNodes: C.flux_nodes, maxChips: C.max_chips, noleft: false, noright: false, tab: "insp"},
    enr: {mode: "clusters", metric: "enr", sumMetric: "nmi", rows: 40, cols: 30, min: 3, q: 0.05, order: "diag", clip: 40, values: false},
    net: {terms: C.max_network_terms, edge: C.min_edge_proteins, jac: 0, weight: "jaccard", layout: "fa2", colour: "share", iter: 300, spacing: 1, nodeScale: 1, edgeScale: 1, labels: C.net_labels, clip: 32, isolated: false, seed: 1},
  };
}
function merge(def, got) {
  if (got == null) return def;
  if (typeof got !== "object" || Array.isArray(def)) return got;
  const out = {...def};
  Object.keys(got).forEach(k => { out[k] = def[k] && typeof def[k] === "object" && !Array.isArray(def[k]) && k !== "ov" && k !== "z" ? merge(def[k], got[k]) : got[k]; });
  return out;
}
function load() { try { return merge(defaults(), JSON.parse(localStorage.getItem(STORE) || "null")); } catch (e) { return defaults(); } }
let A = load(), saveT = null;
function save() { clearTimeout(saveT); saveT = setTimeout(() => { try { localStorage.setItem(STORE, JSON.stringify(A)); } catch (e) {} }, 300); }
function aget(path) { return path.split(".").reduce((o, k) => o == null ? o : o[k], A); }
function aset(path, v) { const ks = path.split("."), last = ks.pop(); ks.reduce((o, k) => o[k], A)[last] = v; }
function font(k) { return Math.max(4, Math.round(A.font[k] * A.scale)); }
const S = {page: 0, view: "overview", last: "3d", colour: "sample", fser: D.leiden_key, f: {sample: new Set(), series: {}}, q: {sp: "", cl: "", ft: ""}, ft: {}, ftSel: FT[0] || null, ftCnt: null, lasso: null, find: null, insp: null, cam: {}, rev: 0, proj: {x: 0, y: 1, az: 0, el: 0}, flux: {from: null, to: null}, log: false, pct: false, drag: "lasso", keep: [], ghostIdx: [], gn: 0, cols: [], fit: false, n: 0, netCache: null, ctPage: 0, ctQ: "", enr: {ser: D.leiden_key, ft: null}, enrSum: null};
const VIEWS = {};
function view(id, v) { VIEWS[id] = v; }
function decode(s, T) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); }
function short(k) { return k === "sample" ? "sample" : k.replace(/^.*_([uts]_[ij]_[a-z0-9]+)$/, "$1"); }
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
function clip(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function seq(a, b) { return Array.from({length: b - a}, (_, j) => a + j); }
function pad([a, b]) { const d = (b - a || 1) * 0.02; return [a - d, b + d]; }
function ranges(c) {
  const r = [0, 1, 2].map(() => [Infinity, -Infinity]);
  for (let i = 0; i < c.length; i++) { const a = r[i % 3], v = c[i]; if (v < a[0]) a[0] = v; if (v > a[1]) a[1] = v; }
  return r.map(pad);
}
function page() {
  const p = P[S.page];
  if (!p._c) {
    p._c = Object.fromEntries(p.methods.map(([m]) => [m, decode(p.coords[m], Float32Array)]));
    p._s = Object.fromEntries(Object.entries(p.sets).map(([k, v]) => [k, decode(v.codes, v.wide ? Uint32Array : Uint16Array)]));
    p._r = Object.fromEntries(p.methods.map(([m]) => [m, ranges(p._c[m])]));
    p._ids = p.ids.split("\n");
    p._rows = decode(p.rows, Uint32Array);
    p._b = Object.fromEntries(p.views.bars.clusters.map((c, i) => [c, i]));
    p._h = p._ids.map((id, i) => `<b>${id}</b>` + (DESC && DESC[p._rows[i]] ? `<br>${DESC[p._rows[i]]}` : "") + D.hover_keys.map(k => `<br>${short(k)}: ${esc(p.sets[k].cats[p._s[k][i]][0])}`).join(""));
  }
  return p;
}
function feat(nm) { const f = D.features[nm]; if (!f._off) { f._off = decode(f.off, Uint32Array); f._idx = decode(f.idx, f.wide ? Uint32Array : Uint16Array); } return f; }
function seriesKeys(p) { return ["sample", ...p.keys]; }
function remap(c) {
  const j = C.palette.indexOf(c);
  return j >= 0 ? A.col.palette[j % A.col.palette.length] : c === C.color_other ? A.col.other : c === C.color_minor ? A.col.minor : c === C.color_bg ? A.col.unassigned : c;
}
function colours(p, k, keep) {
  const set = p.sets[k], cols = set.cats.map(c => remap(c[1])), codes = p._s[k], other = i => set.cats[i][1] === C.color_other, n = new Map(), pal = A.col.palette;
  if (!set.dyn) return cols;
  keep.forEach(i => { const c = codes[i]; if (c && !other(c)) n.set(c, (n.get(c) || 0) + 1); });
  cols.forEach((_, i) => { if (i && !other(i)) cols[i] = A.col.minor; });
  [...n].sort((a, b) => b[1] - a[1]).slice(0, A.mk.maxColored).forEach(([c], j) => { cols[c] = pal[j % pal.length]; });
  return cols;
}
function paint(p, k, cols) { const o = A.ov[k]; return o ? cols.map((c, i) => o[p.sets[k].cats[i][0]] || c) : cols; }
function tint(p, k, keep) { return paint(p, k, colours(p, k, keep)); }
function labKey() { return S.colour === "sample" ? S.fser : S.colour; }
function base() { return {paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: {family: C.font_family, size: font("tick"), color: "#1f2330"}, hoverlabel: {font: {size: font("hover"), family: C.font_family}, bgcolor: "#ffffff", bordercolor: "#c9ced6"}}; }
function hook(id, ev, fn) { const el = $(id); el["_" + ev] = fn; if (!el["_h" + ev]) { el["_h" + ev] = true; el.on(ev, e => el["_" + ev](e)); } }
function viewOk(v) { return !VIEWS[v].ok || VIEWS[v].ok(P[S.page]); }
function hiddenMask(p, k, names) { const m = new Uint8Array(p.sets[k].cats.length); p.sets[k].cats.forEach((c, i) => { if (names.has(c[0])) m[i] = 1; }); return m; }
function ftState(nm) { return S.ft[nm] = S.ft[nm] || {on: new Set(), and: false}; }
function ftActive() { return Object.entries(S.ft).filter(([, f]) => f.on.size).map(([nm, f]) => { const F = feat(nm), m = new Uint8Array(F.vocab.length); f.on.forEach(t => m[t] = 1); return {nm, off: F._off, idx: F._idx, m, and: f.and, need: f.on.size}; }); }
function ftOk(a, r) { let h = 0; for (let j = a.off[r], e = a.off[r + 1]; j < e; j++) if (a.m[a.idx[j]]) { if (!a.and) return true; h++; } return a.and && h === a.need; }
function stack(p, keep) {
  const ks = [S.colour, S.colour === "sample" ? S.fser : "sample"].filter(k => p.sets[k] && A.z[k] && Object.keys(A.z[k]).length);
  if (!ks.length) return keep;
  const lv = ks.map(k => Int32Array.from(p.sets[k].cats, c => A.z[k][c[0]] || 0)), cd = ks.map(k => p._s[k]), m = ks.length;
  return keep.sort((a, b) => { for (let j = 0; j < m; j++) { const d = lv[j][cd[j][a]] - lv[j][cd[j][b]]; if (d) return d; } return a - b; });
}
function zmove(k, nm, d) {
  const z = A.z[k] = A.z[k] || {}, v = Object.values(z);
  z[nm] = d > 0 ? Math.max(0, ...v) + 1 : Math.min(0, ...v) - 1;
  save(); draw();
}
function render() {
  const p = page(), n = p._ids.length, keep = [], ghost = [], V = VIEWS[S.view], rows = p._rows;
  const masks = [["sample", S.f.sample], ...Object.entries(S.f.series)].filter(([k, h]) => h.size && p.sets[k]).map(([k, h]) => [p._s[k], hiddenMask(p, k, h)]), nm = masks.length;
  const act = ftActive(), mine = act.find(a => a.nm === S.ftSel), others = act.filter(a => a !== mine), F = S.ftSel ? feat(S.ftSel) : null, cnt = F ? new Uint32Array(F.vocab.length) : null, L = S.lasso;
  for (let i = 0; i < n; i++) {
    let ok = !L || L.has(i);
    for (let j = 0; ok && j < nm; j++) if (masks[j][1][masks[j][0][i]]) ok = false;
    const r = rows[i];
    if (ok && others.length) ok = others.every(a => ftOk(a, r));
    if (ok && F) for (let j = F._off[r], e = F._off[r + 1]; j < e; j++) cnt[F._idx[j]]++;
    if (ok && (!mine || ftOk(mine, r))) keep.push(i); else if (A.mk.ghost) ghost.push(i);
  }
  S.ftCnt = cnt;
  S.keep = stack(p, keep); S.ghostIdx = ghost; S.cols = tint(p, labKey(), keep);
  V.render(p, S.colour, keep);
  if (S.fit) { S.fit = false; (V.plots || []).forEach(id => { if ($(id).data) Plotly.Plots.resize($(id)); }); }
  sidebar(p, keep.length);
  inspect();
  colourTable();
  $("stage").classList.remove("loading");
}
let drawT = null;
function draw() { $("stage").classList.add("loading"); cancelAnimationFrame(drawT); drawT = requestAnimationFrame(() => setTimeout(render, 0)); }
let lazyT = null;
function drawLater() { clearTimeout(lazyT); lazyT = setTimeout(draw, 160); }
