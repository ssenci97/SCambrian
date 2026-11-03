function netFeats() { return FT.filter(nm => D.features[nm].net); }
function rng(seed) { let a = (seed >>> 0) || 1; return () => { a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function seedPos(n, rnd, r0) {
  const x = new Float64Array(n), y = new Float64Array(n), rot = rnd() * 2 * Math.PI;
  for (let i = 0; i < n; i++) { const a = 2.399963 * i + rot, r = r0 * Math.sqrt((i + 0.5) / n) * (0.9 + 0.2 * rnd()); x[i] = r * Math.cos(a); y[i] = r * Math.sin(a); }
  return {x, y};
}
function normalise(x, y) {
  const n = x.length; if (!n) return;
  let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
  let s = 1e-9; for (let i = 0; i < n; i++) { x[i] -= mx; y[i] -= my; s = Math.max(s, Math.abs(x[i]), Math.abs(y[i])); }
  for (let i = 0; i < n; i++) { x[i] /= s; y[i] /= s; }
}
function frLayout(n, E, iter, spacing, rnd) {
  const {x, y} = seedPos(n, rnd, 1), dx = new Float64Array(n), dy = new Float64Array(n), deg = new Float64Array(n), k = spacing * Math.sqrt(1 / Math.max(1, n));
  E.forEach(e => { deg[e.a]++; deg[e.b]++; });
  for (let it = 0; it < iter; it++) {
    const t = 0.1 * (1 - it / iter) + 0.002;
    dx.fill(0); dy.fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const ux = x[i] - x[j], uy = y[i] - y[j], f = k * k / (ux * ux + uy * uy + 1e-6); dx[i] += ux * f; dy[i] += uy * f; dx[j] -= ux * f; dy[j] -= uy * f; }
    E.forEach(e => { const ux = x[e.a] - x[e.b], uy = y[e.a] - y[e.b], f = Math.hypot(ux, uy) / k * (0.1 + e.u) / Math.sqrt(Math.min(deg[e.a], deg[e.b])); dx[e.a] -= ux * f; dy[e.a] -= uy * f; dx[e.b] += ux * f; dy[e.b] += uy * f; });
    for (let i = 0; i < n; i++) { const gx = dx[i] - 0.6 * k * x[i], gy = dy[i] - 0.6 * k * y[i], l = Math.hypot(gx, gy) || 1, s = Math.min(l, t) / l; x[i] += gx * s; y[i] += gy * s; }
  }
  return {x, y};
}
function fa2Layout(n, E, iter, spacing, rnd) {
  const {x, y} = seedPos(n, rnd, Math.sqrt(n) * 2), fx = new Float64Array(n), fy = new Float64Array(n), m = new Float64Array(n).fill(1), kr = 2 * spacing * spacing, kg = 1;
  E.forEach(e => { m[e.a]++; m[e.b]++; });
  for (let it = 0; it < iter; it++) {
    const cap = 10 * (1 - it / iter) + 0.05;
    fx.fill(0); fy.fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const ux = x[i] - x[j], uy = y[i] - y[j], d2 = Math.max(ux * ux + uy * uy, 0.01), f = kr * m[i] * m[j] / d2; fx[i] += ux * f; fy[i] += uy * f; fx[j] -= ux * f; fy[j] -= uy * f; }
    E.forEach(e => { const ux = x[e.a] - x[e.b], uy = y[e.a] - y[e.b], f = e.u; fx[e.a] -= ux * f; fy[e.a] -= uy * f; fx[e.b] += ux * f; fy[e.b] += uy * f; });
    for (let i = 0; i < n; i++) { const d = Math.hypot(x[i], y[i]) || 1, gx = fx[i] - kg * m[i] * x[i] / d, gy = fy[i] - kg * m[i] * y[i] / d, l = Math.hypot(gx, gy) || 1, s = Math.min(l / m[i], cap) / l; x[i] += gx * s; y[i] += gy * s; }
  }
  return {x, y};
}
function louvain(n, E, rnd) {
  const comm = Int32Array.from({length: n}, (_, i) => i);
  let N = n, adj = Array.from({length: n}, () => new Map());
  E.forEach(e => { if (e.a === e.b) return; adj[e.a].set(e.b, (adj[e.a].get(e.b) || 0) + e.u); adj[e.b].set(e.a, (adj[e.b].get(e.a) || 0) + e.u); });
  for (let level = 0; level < 12; level++) {
    const k = new Float64Array(N), c = Int32Array.from({length: N}, (_, i) => i), tot = new Float64Array(N);
    let m2 = 0;
    for (let i = 0; i < N; i++) { adj[i].forEach(w => { k[i] += w; }); tot[i] = k[i]; m2 += k[i]; }
    if (!m2) break;
    let moved = false;
    for (let pass = 0; pass < 20; pass++) {
      let any = false;
      const ord = Array.from({length: N}, (_, i) => i).sort(() => rnd() - 0.5);
      for (const i of ord) {
        const ci = c[i], w = new Map();
        adj[i].forEach((v, j) => { if (j !== i) w.set(c[j], (w.get(c[j]) || 0) + v); });
        tot[ci] -= k[i];
        let best = ci, gain = (w.get(ci) || 0) - tot[ci] * k[i] / m2;
        w.forEach((v, cc) => { const g = v - tot[cc] * k[i] / m2; if (g > gain + 1e-12) { gain = g; best = cc; } });
        tot[best] += k[i];
        if (best !== ci) { c[i] = best; any = moved = true; }
      }
      if (!any) break;
    }
    if (!moved) break;
    const ids = new Map(); c.forEach(v => { if (!ids.has(v)) ids.set(v, ids.size); });
    for (let i = 0; i < n; i++) comm[i] = ids.get(c[comm[i]]);
    const M = ids.size, nadj = Array.from({length: M}, () => new Map());
    for (let i = 0; i < N; i++) adj[i].forEach((v, j) => { const a = ids.get(c[i]), b = ids.get(c[j]); nadj[a].set(b, (nadj[a].get(b) || 0) + v); });
    if (M === N) break;
    adj = nadj; N = M;
  }
  const size = new Map(); comm.forEach(v => size.set(v, (size.get(v) || 0) + 1));
  const rank = new Map([...size].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([v], j) => [v, j]));
  return Int32Array.from(comm, v => rank.get(v));
}
function groupedLayout(n, E, comm, iter, spacing, rnd) {
  const x = new Float64Array(n), y = new Float64Array(n), groups = new Map();
  comm.forEach((c, i) => { if (!groups.has(c)) groups.set(c, []); groups.get(c).push(i); });
  const gs = [...groups.values()], total = n;
  let ang = 0;
  gs.forEach(g => {
    const arc = 2 * Math.PI * g.length / total, mid = ang + arc / 2, R = gs.length > 1 ? 1 : 0, r = gs.length > 1 ? Math.min(0.55, 0.95 * Math.sin(Math.min(arc, Math.PI) / 2)) * Math.sqrt(spacing) : 1;
    ang += arc;
    const loc = new Map(g.map((i, j) => [i, j])), sub = E.filter(e => loc.has(e.a) && loc.has(e.b)).map(e => ({a: loc.get(e.a), b: loc.get(e.b), u: e.u})), L = g.length > 1 ? frLayout(g.length, sub, Math.min(iter, 200), spacing, rnd) : {x: [0], y: [0]};
    if (g.length > 1) normalise(L.x, L.y);
    g.forEach((i, j) => { x[i] = R * Math.cos(mid) + r * L.x[j]; y[i] = R * Math.sin(mid) + r * L.y[j]; });
  });
  return {x, y};
}
function circularLayout(n, comm, cnt) {
  const ord = Array.from({length: n}, (_, i) => i).sort((a, b) => comm[a] - comm[b] || cnt[b] - cnt[a]), slots = n + new Set(comm).size, x = new Float64Array(n), y = new Float64Array(n);
  let s = 0, prev = -1;
  ord.forEach(i => { if (comm[i] !== prev) { s++; prev = comm[i]; } const a = 2 * Math.PI * s / slots; x[i] = Math.cos(a); y[i] = Math.sin(a); s++; });
  return {x, y};
}
function netLabels(x, y, n, size) {
  const rx = Math.max(...x) - Math.min(...x) || 1, ry = Math.max(...y) - Math.min(...y) || 1, on = [], gx = 0.12 * size / 12, gy = 0.04 * size / 12;
  for (let j = 0; j < x.length && on.length < n; j++) if (on.every(i => Math.abs(x[i] - x[j]) / rx > gx || Math.abs(y[i] - y[j]) / ry > gy)) on.push(j);
  return new Set(on);
}
function network(p) {
  const fs = netFeats(), nm = fs.includes(S.net) ? S.net : fs[0], o = A.net;
  S.net = nm;
  $("nFeat").replaceChildren(...fs.map(f => new Option(f, f))); $("nFeat").value = nm;
  const F = feat(nm), off = F._off, idx = F._idx, rows = p._rows, n = S.keep.length, total = rows.length, cnt = new Uint32Array(F.vocab.length), spCodes = p._s.sample, nsp = p.sets.sample.cats.length;
  let msg = !n ? "No point shown." : n === total ? `Every point of the page is shown (${total.toLocaleString()}): untick samples or clusters, lasso or filter by a feature first.` : n > C.max_network_proteins ? `${n.toLocaleString()} proteins shown, the network accepts up to ${C.max_network_proteins.toLocaleString()}: narrow the selection.` : "";
  let annotated = 0;
  if (!msg) S.keep.forEach(i => { const r = rows[i]; if (off[r + 1] > off[r]) annotated++; for (let j = off[r]; j < off[r + 1]; j++) cnt[idx[j]]++; });
  let top = msg ? [] : Array.from(cnt.keys()).filter(t => cnt[t]).sort((a, b) => cnt[b] - cnt[a] || a - b).slice(0, Math.max(2, o.terms));
  if (!msg && !top.length) msg = `No shown protein has a ${nm} term.`;
  $("netMsg").hidden = !msg; $("netMsg").textContent = msg;
  if (msg) { S.netComm = null; Plotly.purge($("net")); return; }
  let T = top.length;
  const loc = new Int32Array(F.vocab.length).fill(-1), pair = new Map(), buf = [], spc = new Uint32Array(T * nsp);
  top.forEach((t, j) => loc[t] = j);
  S.keep.forEach(i => {
    const r = rows[i], s = spCodes[i]; buf.length = 0;
    for (let j = off[r]; j < off[r + 1]; j++) if (loc[idx[j]] >= 0) { buf.push(loc[idx[j]]); spc[loc[idx[j]] * nsp + s]++; }
    for (let a = 0; a < buf.length; a++) for (let b = a + 1; b < buf.length; b++) { const u = Math.min(buf[a], buf[b]), v = Math.max(buf[a], buf[b]), key = u * T + v; pair.set(key, (pair.get(key) || 0) + 1); }
  });
  let E = [...pair].filter(([, w]) => w >= o.edge).map(([key, w]) => { const a = Math.floor(key / T), b = key % T; return {a, b, w, j: w / (cnt[top[a]] + cnt[top[b]] - w)}; }).filter(e => e.j >= o.jac).sort((u, v) => v.w - u.w || u.a - v.a || u.b - v.b).slice(0, NET_EDGES);
  let keepNode = top.map((_, j) => j);
  if (o.isolated) { const has = new Uint8Array(T); E.forEach(e => { has[e.a] = has[e.b] = 1; }); keepNode = keepNode.filter(j => has[j]); }
  if (!keepNode.length) { $("netMsg").hidden = false; $("netMsg").textContent = "No edge passes the thresholds."; S.netComm = null; Plotly.purge($("net")); return; }
  const remapN = new Int32Array(T).fill(-1); keepNode.forEach((j, q) => remapN[j] = q);
  const spOf = keepNode.map(j => spc.subarray(j * nsp, (j + 1) * nsp));
  top = keepNode.map(j => top[j]); T = top.length;
  E = E.filter(e => remapN[e.a] >= 0 && remapN[e.b] >= 0).map(e => ({...e, a: remapN[e.a], b: remapN[e.b]}));
  const wmax = Math.max(1e-9, ...E.map(e => o.weight === "count" ? e.w : e.j));
  E.forEach(e => { e.u = (o.weight === "count" ? e.w : e.j) / wmax; });
  const key = [nm, top.join(","), E.map(e => `${e.a}-${e.b}-${e.w}`).join(","), o.layout, o.weight, o.iter, o.spacing, o.seed].join("|");
  if (!S.netCache || S.netCache.key !== key) {
    const rnd = rng(o.seed), comm = louvain(T, E, rnd), cntT = top.map(t => cnt[t]);
    const L = o.layout === "force" ? frLayout(T, E, o.iter, o.spacing, rnd) : o.layout === "circular" ? circularLayout(T, comm, cntT) : o.layout === "grouped" ? groupedLayout(T, E, comm, o.iter, o.spacing, rnd) : fa2Layout(T, E, o.iter, o.spacing, rnd);
    normalise(L.x, L.y);
    S.netCache = {key, x: L.x, y: L.y, comm};
  }
  const {x, y, comm} = S.netCache, share = top.map(t => cnt[t] / n), name = t => F.vocab[t], csize = new Map();
  comm.forEach(c => csize.set(c, (csize.get(c) || 0) + 1));
  const cname = c => csize.get(c) > 1 ? `C${c + 1}` : "singletons", cnames = [...new Set([...comm].map(cname))].sort((a, b) => (a === "singletons") - (b === "singletons") || +a.slice(1) - +b.slice(1));
  const pal = A.col.palette, cbase = cnames.map((c, j) => c === "singletons" ? A.col.other : pal[j % pal.length]), cov = A.ov[COMMUNITY] || {}, ccol = Object.fromEntries(cnames.map((c, j) => [c, cov[c] || cbase[j]]));
  S.netComm = {names: cnames, base: cbase, counts: cnames.map(c => [...comm].filter(v => cname(v) === c).length)};
  const spCols = paint(p, "sample", p.sets.sample.cats.map(c => remap(c[1]))), major = spOf.map(a => a.indexOf(Math.max(...a)));
  const order = Array.from({length: T}, (_, j) => j), lab = netLabels(Array.from(order, j => x[j]), Array.from(order, j => y[j]), o.labels, font("net"));
  const tiers = [[0, 0.2, 0.6, 0.3], [0.2, 0.5, 1.6, 0.5], [0.5, 1.01, 3.2, 0.75]];
  const edges = tiers.map(([lo, hi, w, op]) => { const b = E.filter(e => e.j >= lo && e.j < hi); return {type: "scatter", mode: "lines", x: b.flatMap(e => [x[e.a], x[e.b], null]), y: b.flatMap(e => [y[e.a], y[e.b], null]), line: {width: w * o.edgeScale, color: A.col.edge}, opacity: o.edgeScale ? op : 0, hoverinfo: "skip", showlegend: false}; });
  const mids = {type: "scatter", mode: "markers", x: E.map(e => (x[e.a] + x[e.b]) / 2), y: E.map(e => (y[e.a] + y[e.b]) / 2), marker: {size: 8, opacity: 0}, customdata: E.map(e => [clip(name(top[e.a]), 60), clip(name(top[e.b]), 60), e.w, e.j]), hovertemplate: "%{customdata[0]}<br>↔ %{customdata[1]}<br>%{customdata[2]:,} shared proteins · Jaccard %{customdata[3]:.2f}<extra></extra>", showlegend: false};
  const smax = Math.max(...share), mode = o.colour, cdata = top.map((t, j) => [name(t), cnt[t], cname(comm[j]), p.sets.sample.cats[major[j]][0]]);
  const marker = {size: top.map(t => (10 + 30 * Math.sqrt(cnt[t] / cnt[top[0]])) * o.nodeScale), line: {width: 1, color: "#ffffff"}, opacity: 1};
  if (mode === "share") Object.assign(marker, {color: share, colorscale: [[0, A.col.low], [1, A.col.high]], cmin: 0, cmax: smax, colorbar: {orientation: "h", x: 0.5, xanchor: "center", y: -0.02, yanchor: "top", len: 0.5, thickness: 12, tickformat: ".0%", tickfont: {size: font("tick")}, title: {text: "share of the shown proteins", side: "top", font: {size: font("axis")}}}});
  else marker.color = top.map((_, j) => mode === "community" ? ccol[cname(comm[j])] : spCols[major[j]]);
  const nodes = {type: "scatter", mode: "markers+text", x: Array.from(x), y: Array.from(y), text: top.map((t, j) => lab.has(j) ? esc(clip(name(t).replace(/\s*\[GO:\d+\]$/, ""), o.clip)) : ""), textposition: "top center", textfont: {size: font("net"), color: "#1f2330"}, customdata: cdata, hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]:,} shown proteins<br>%{customdata[2]} · mostly %{customdata[3]}<br>click keeps its proteins · shift-click adds it<extra></extra>", showlegend: false, marker};
  const hl = {type: "scatter", mode: "lines", x: [], y: [], line: {width: 2.5, color: "#30767A"}, hoverinfo: "skip", showlegend: false};
  const legend = mode === "community" ? cnames.slice(0, 20).map(c => ({type: "scatter", mode: "markers", x: [null], y: [null], name: c, marker: {size: 10, color: ccol[c]}, showlegend: true, hoverinfo: "skip"})) : mode === "sample" ? [...new Set(major)].map(s => ({type: "scatter", mode: "markers", x: [null], y: [null], name: p.sets.sample.cats[s][0], marker: {size: 10, color: spCols[s]}, showlegend: true, hoverinfo: "skip"})) : [];
  const ax = {visible: false, fixedrange: false}, head = `<b>${esc(nm)}</b> · ${T} terms · ${E.length.toLocaleString()} edges · ${csize.size} communities · ${annotated.toLocaleString()}/${n.toLocaleString()} shown proteins annotated`;
  const NI = edges.length + 1;
  Plotly.react($("net"), [...edges, mids, nodes, hl, ...legend], {...base(), margin: {l: 10, r: 44, t: font("axis") + 20, b: mode === "share" ? 70 : 40}, hovermode: "closest", dragmode: "pan", showlegend: legend.length > 0, legend: {orientation: "h", x: 0.5, xanchor: "center", y: 0, yanchor: "top", font: {size: font("legend")}, itemclick: false, itemdoubleclick: false}, xaxis: ax, yaxis: {...ax, scaleanchor: "x"}, annotations: [{text: head, x: 0, xanchor: "left", y: 1, yanchor: "bottom", xref: "paper", yref: "paper", showarrow: false, font: {size: font("axis")}}], modebar: {orientation: "v"}}, {responsive: true, displaylogo: false, scrollZoom: true, toImageButtonOptions: {format: "png", scale: 2, filename: `${p.slug}_network_${nm.replace(/\W+/g, "_")}`}});
  const adj = Array.from({length: T}, () => []); E.forEach(e => { adj[e.a].push([e.b, e.w, e.j]); adj[e.b].push([e.a, e.w, e.j]); });
  hook("net", "plotly_click", e => {
    const pt = e.points && e.points[0]; if (!pt || pt.curveNumber !== NI) return;
    const j = pt.pointIndex, t = top[j];
    S.insp = {type: "term", nm, t, n: cnt[t], share: share[j], comm: cname(comm[j]), nb: adj[j].map(([b, w, jj]) => [top[b], w, jj]).sort((u, v) => v[1] - u[1])};
    ftIsolate(nm, t, e.event && e.event.shiftKey);
  });
  hook("net", "plotly_hover", e => {
    const pt = e.points && e.points[0]; if (!pt || pt.curveNumber !== NI) return;
    const j = pt.pointIndex, nb = new Set([j, ...adj[j].map(a => a[0])]);
    Plotly.restyle($("net"), {"marker.opacity": [top.map((_, q) => nb.has(q) ? 1 : 0.15)]}, [NI]);
    Plotly.restyle($("net"), {x: [adj[j].flatMap(([b]) => [x[j], x[b], null])], y: [adj[j].flatMap(([b]) => [y[j], y[b], null])]}, [NI + 1]);
  });
  hook("net", "plotly_unhover", () => { Plotly.restyle($("net"), {"marker.opacity": 1}, [NI]); Plotly.restyle($("net"), {x: [[]], y: [[]]}, [NI + 1]); });
}
view("network", {label: "Network", title: "Co-annotation network of a UniProt feature over the shown proteins", panel: "netBox", ctl: "ctlNet", plots: ["net"], render: network, ok: () => netFeats().length > 0});
