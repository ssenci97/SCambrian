function stacked() { return A.lay.stack === "stack" || (A.lay.stack === "auto" && $("stage").clientWidth < 760); }
function axis(t, r) { return {range: r, autorange: false, title: {text: t, font: {size: font("axis")}}, backgroundcolor: "#ffffff", showbackground: true, gridcolor: "#e6e8ec", zerolinecolor: "#c9ced6", showline: true, linecolor: "#3a3f4b", tickfont: {size: font("tick")}, showspikes: false}; }
function axis2(t, r, dom, anchor) { return {title: {text: t, font: {size: font("axis")}}, range: r, autorange: false, domain: dom, anchor, constrain: "domain", zeroline: false, showline: true, mirror: true, linecolor: "#3a3f4b", gridcolor: "#e6e8ec", tickfont: {size: font("tick")}}; }
function project(p, m) {
  const pr = S.proj, id = `${pr.x}${pr.y}|${pr.az}|${pr.el}`;
  p._p = p._p || {};
  if (p._p[m] && p._p[m].id === id) return p._p[m];
  const c = p._c[m], i = pr.x, j = pr.y, k = 3 - i - j, mid = p._r[m].map(([a, b]) => (a + b) / 2), N = c.length / 3;
  const th = pr.az * Math.PI / 180, ph = pr.el * Math.PI / 180, ct = Math.cos(th), st = Math.sin(th), cp = Math.cos(ph), sp = Math.sin(ph);
  const x = new Float32Array(N), y = new Float32Array(N), rx = [Infinity, -Infinity], ry = [Infinity, -Infinity];
  for (let n = 0; n < N; n++) {
    const a = c[3 * n + i] - mid[i], b = c[3 * n + j] - mid[j], d = c[3 * n + k] - mid[k];
    x[n] = a * ct + d * st; y[n] = b * cp + (a * st - d * ct) * sp;
    if (x[n] < rx[0]) rx[0] = x[n]; if (x[n] > rx[1]) rx[1] = x[n]; if (y[n] < ry[0]) ry[0] = y[n]; if (y[n] > ry[1]) ry[1] = y[n];
  }
  return (p._p[m] = {id, x, y, rx: pad(rx), ry: pad(ry)});
}
function labels(p, q, keep, s) {
  const lk = labKey(), cats = p.sets[lk].cats, cl = p._s[lk], g = new Map(), med = a => { a.sort((u, v) => u - v); return a[a.length >> 1]; }, skip = new Set([A.col.other, A.col.minor]);
  keep.forEach(i => { const c = cl[i]; if (!c || skip.has(S.cols[c])) return; if (!g.has(c)) g.set(c, [[], []]); const v = g.get(c); v[0].push(q.x[i]); v[1].push(q.y[i]); });
  return [...g].sort((a, b) => b[1][0].length - a[1][0].length).slice(0, A.mk.maxLabels).map(([c, [x, y]]) => ({text: `<b>${esc(cats[c][0])}</b>`, x: med(x), y: med(y), xref: "x" + s, yref: "y" + s, showarrow: false, font: {size: font("label"), color: "#1f2330"}, bgcolor: "rgba(255,255,255,0.7)", borderpad: 1}));
}
function points(p, k, keep) {
  const set = p.sets[k], codes = p._s[k], cols = k === labKey() ? S.cols : tint(p, k, keep), extra = D.hover_keys.includes(k) ? null : k, n = p.methods.length, is3 = S.view === "3d", pr = S.proj, rot = pr.az || pr.el ? "′" : "", st = stacked();
  const text = keep.map(i => extra ? p._h[i] + `<br>${short(extra)}: ${esc(set.cats[codes[i]][0])}` : p._h[i]), color = keep.map(i => cols[codes[i]]);
  const layout = {...base(), uirevision: is3 ? `${S.rev}` : `${S.rev}|${pr.x}${pr.y}|${pr.az}|${pr.el}`, margin: is3 ? {l: 0, r: 0, t: font("title") + 16, b: 0} : {l: 40 + 2 * font("axis"), r: 12, t: font("title") + 16, b: 30 + 2 * font("axis")}, plot_bgcolor: "#ffffff", annotations: []};
  if (!is3) { layout.dragmode = S.drag; layout.selections = []; }
  const traces = p.methods.map(([m, t], j) => {
    const a = j / n + (j ? 0.005 : 0), b = (j + 1) / n - (j < n - 1 ? 0.005 : 0), dom = st ? {x: [0, 1], y: [1 - b, 1 - a - (j < n - 1 ? 0.03 : 0)]} : {x: [a + (is3 ? 0 : 0.02), b - (is3 ? 0 : 0.02)], y: [0, 1]}, s = j ? String(j + 1) : "";
    layout.annotations.push({text: `<b>${t}</b>`, showarrow: false, xref: "paper", yref: "paper", x: (dom.x[0] + dom.x[1]) / 2, y: dom.y[1], yanchor: "bottom", font: {size: font("title")}});
    if (is3) {
      const c = p._c[m], x = new Float32Array(keep.length), y = new Float32Array(keep.length), z = new Float32Array(keep.length), sc = "scene" + s;
      keep.forEach((i, q) => { x[q] = c[3 * i]; y[q] = c[3 * i + 1]; z[q] = c[3 * i + 2]; });
      layout[sc] = {domain: dom, aspectmode: "cube", xaxis: axis(t + "-1", p._r[m][0]), yaxis: axis(t + "-2", p._r[m][1]), zaxis: axis(t + "-3", p._r[m][2])};
      if (S.cam[sc]) layout[sc].camera = S.cam[sc];
      return {type: "scatter3d", mode: "markers", x, y, z, text, hovertemplate: "%{text}<extra></extra>", showlegend: false, scene: sc, marker: {size: A.mk.size, opacity: A.mk.alpha, color, line: {width: 0}}};
    }
    const q = project(p, m);
    layout["xaxis" + s] = axis2(`${t}-${pr.x + 1}${rot}`, q.rx, dom.x, "y" + s);
    layout["yaxis" + s] = {...axis2(`${t}-${pr.y + 1}${rot}`, q.ry, dom.y, "x" + s), scaleanchor: "x" + s};
    if (A.mk.labels) labels(p, q, keep, s).forEach(a => layout.annotations.push(a));
    return {type: "scattergl", mode: "markers", x: Float32Array.from(keep, i => q.x[i]), y: Float32Array.from(keep, i => q.y[i]), xaxis: "x" + s, yaxis: "y" + s, text, hovertemplate: "%{text}<extra></extra>", showlegend: false, selectedpoints: null, marker: {size: A.mk.size * 2, opacity: A.mk.alpha, color, line: {width: 0}}};
  });
  const G = S.ghostIdx, ghosts = !G.length ? [] : p.methods.map(([m], j) => {
    const s = j ? String(j + 1) : "", g = {mode: "markers", hoverinfo: "skip", showlegend: false, marker: {color: A.col.ghost, opacity: A.mk.ghostAlpha, line: {width: 0}}};
    if (is3) { const c = p._c[m]; return {...g, type: "scatter3d", scene: "scene" + s, x: Float32Array.from(G, i => c[3 * i]), y: Float32Array.from(G, i => c[3 * i + 1]), z: Float32Array.from(G, i => c[3 * i + 2]), marker: {...g.marker, size: A.mk.size}}; }
    const q = project(p, m);
    return {...g, type: "scattergl", xaxis: "x" + s, yaxis: "y" + s, x: Float32Array.from(G, i => q.x[i]), y: Float32Array.from(G, i => q.y[i]), selectedpoints: null, marker: {...g.marker, size: A.mk.size * 2}};
  });
  S.gn = ghosts.length; traces.unshift(...ghosts);
  if (S.find !== null) p.methods.forEach(([m], j) => {
    const f = S.find, s = j ? String(j + 1) : "", ring = {mode: "markers", text: [p._h[f]], hovertemplate: "%{text}<extra></extra>", showlegend: false, marker: {size: A.mk.ring, symbol: "circle-open", color: A.col.ring, line: {width: 3}}}, c = p._c[m], q = is3 ? null : project(p, m);
    traces.push(is3 ? {...ring, type: "scatter3d", x: [c[3 * f]], y: [c[3 * f + 1]], z: [c[3 * f + 2]], scene: "scene" + s} : {...ring, type: "scattergl", x: [q.x[f]], y: [q.y[f]], xaxis: "x" + s, yaxis: "y" + s});
  });
  Plotly.react($("plot"), traces, layout, {responsive: true, displaylogo: false, toImageButtonOptions: {format: "png", scale: 2, filename: `${p.slug}_${S.view}_${k.replace(/\W+/g, "_")}`}});
  const el = $("plot");
  if (!el._camHook) {
    el._camHook = true;
    el.on("plotly_relayout", e => Object.keys(e).forEach(k => { const m = k.match(/^(scene\d*)\.camera$/); if (m) S.cam[m[1]] = e[k]; if (k === "dragmode") { S.drag = e[k]; $("lassoTog").checked = S.drag === "lasso"; } }));
    el.on("plotly_selected", e => { if (!e || S.view !== "2d") return; const n = P[S.page].methods.length, sel = new Set(); e.points.forEach(pt => { if (pt.curveNumber >= S.gn && pt.curveNumber < S.gn + n) sel.add(S.keep[pt.pointIndex]); }); if (sel.size) { S.lasso = sel; draw(); } });
    el.on("plotly_deselect", () => { if (S.lasso) { S.lasso = null; draw(); } });
    el.on("plotly_click", e => { const pt = e && e.points && e.points[0], n = P[S.page].methods.length; if (!pt || pt.curveNumber < S.gn || pt.curveNumber >= S.gn + n) return; const i = S.keep[pt.pointNumber ?? pt.pointIndex]; if (i !== undefined) showInspect({type: "protein", i, page: S.page}); });
  }
}
view("3d", {label: "3D", title: "3D scatter", panel: "stage", ctl: "ctlPts", plots: ["plot"], points: true, render: points});
view("2d", {label: "2D", title: "2D projection of the 3D layouts", panel: "stage", ctl: ["ctlPts", "ctl2d"], plots: ["plot"], points: true, render: points});
