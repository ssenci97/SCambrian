function lgamma(z) {
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * z))) - lgamma(1 - z);
  z -= 1; let a = 0.99999999999980993; const t = z + 7.5;
  g.forEach((c, i) => { a += c / (z + i + 1); });
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}
function lchoose(n, k) { return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); }
function hyperTail(k, N, K, n) {
  const hi = Math.min(n, K), lo = Math.max(0, n + K - N);
  if (k <= lo) return 1;
  if (k > hi) return 0;
  let term = Math.exp(lchoose(K, k) + lchoose(N - K, n - k) - lchoose(N, n)), sum = 0;
  for (let x = k; x <= hi; x++) { sum += term; if (term < sum * 1e-12) break; term *= (K - x) * (n - x) / ((x + 1) * (N - K - n + x + 1)); }
  return Math.min(1, sum);
}
function bh(ps) {
  const m = ps.length, ord = ps.map((_, i) => i).sort((a, b) => ps[a] - ps[b]), q = new Float64Array(m);
  let run = 1;
  for (let r = m - 1; r >= 0; r--) { const i = ord[r]; run = Math.min(run, ps[i] * m / (r + 1)); q[i] = run; }
  return q;
}
function enrSingles(p) { return p.keys.filter(k => (D.single || []).includes(k)); }
function enrClusterKeys(p) { return seriesKeys(p).filter(k => !(D.single || []).includes(k)); }
function termReader(p, ft) {
  if (p.sets[ft]) { const cd = p._s[ft], cats = p.sets[ft].cats, bad = cats.map((c, i) => !i || c[1] === C.color_other); return {size: cats.length, name: t => cats[t][0], each: (i, fn) => { const t = cd[i]; if (!bad[t]) fn(t); }, any: i => !bad[cd[i]]}; }
  const F = feat(ft), off = F._off, idx = F._idx, rows = p._rows;
  return {size: F.vocab.length, name: t => F.vocab[t], each: (i, fn) => { const r = rows[i]; for (let j = off[r]; j < off[r + 1]; j++) fn(idx[j]); }, any: i => off[rows[i] + 1] > off[rows[i]]};
}
const SHOWN = "\u0000shown";
function enrClusters(p, k) {
  const cats = p.sets[k].cats, codes = p._s[k], n = new Map();
  S.keep.forEach(i => { const c = codes[i]; if (c && cats[c][1] !== C.color_other) n.set(c, (n.get(c) || 0) + 1); });
  return [...n].sort((a, b) => b[1] - a[1]);
}
function enrTest(p, ft, bg, fg) {
  const R = termReader(p, ft), K = new Map(), k = new Map();
  let N = 0, n = 0;
  bg.forEach(i => { if (!R.any(i)) return; N++; const f = fg(i); if (f) n++; R.each(i, t => { K.set(t, (K.get(t) || 0) + 1); if (f) k.set(t, (k.get(t) || 0) + 1); }); });
  const tests = [...k].filter(([, v]) => v >= A.enr.min).map(([t, v]) => ({t, k: v, n, K: K.get(t), N, fold: (v / n) / (K.get(t) / N), p: hyperTail(v, N, K.get(t), n)}));
  const q = bh(tests.map(x => x.p)); tests.forEach((x, i) => { x.q = q[i]; x.lq = -Math.log10(Math.max(q[i], 1e-300)); });
  return {tests, n, N, R};
}
function wrap(text, w) {
  const lines = [];
  let cur = "";
  text.split(/\s+/).forEach(word => { if (cur && (cur + " " + word).length > w) { lines.push(cur); cur = word; } else cur = cur ? cur + " " + word : word; });
  if (cur) lines.push(cur);
  return lines.slice(0, 4).map((l, j) => j === 3 && lines.length > 4 ? l + " …" : l);
}
function enrStep(d) { const o = [...$("eCl").options].map(x => x.value), j = o.indexOf(S.enr.cl); S.enr.cl = o[Math.max(0, Math.min(o.length - 1, j + d))]; draw(); }
function enrFeatures(p) { return [...enrSingles(p), ...FT]; }
function enrich(p) {
  const sers = enrClusterKeys(p).filter(k => k !== "sample"), fts = enrFeatures(p), o = A.enr;
  if (!sers.includes(S.enr.ser)) S.enr.ser = D.leiden_key;
  if (!fts.includes(S.enr.ft)) S.enr.ft = ["uniprot_family", "GO process", "Family"].find(f => fts.includes(f)) || fts[0];
  const k = S.enr.ser, ft = S.enr.ft, cats = p.sets[k].cats, cl = enrClusters(p, k);
  if (S.enr.cl !== SHOWN && !cl.some(([c]) => cats[c][0] === S.enr.cl)) S.enr.cl = cl.length ? cats[cl[0][0]][0] : SHOWN;
  $("eSer").replaceChildren(...sers.map(x => new Option(short(x), x))); $("eSer").value = k;
  $("eCl").replaceChildren(new Option("Shown proteins vs page", SHOWN), ...cl.map(([c, n]) => new Option(`${cats[c][0]} (${n.toLocaleString()})`, cats[c][0]))); $("eCl").value = S.enr.cl;
  const grp = (label, names) => { const g = document.createElement("optgroup"); g.label = label; g.replaceChildren(...names.map(nm => new Option(nm, nm))); return g; };
  $("eFt").replaceChildren(...[grp("Single-valued", enrSingles(p)), grp("Annotation terms", FT.filter(f => !D.features[f].domain)), grp("Domains", FT.filter(f => D.features[f].domain))].filter(g => g.children.length)); $("eFt").value = ft;
  const shownMode = S.enr.cl === SHOWN, code = shownMode ? -1 : cats.findIndex(c => c[0] === S.enr.cl), codes = p._s[k], keepSet = shownMode ? new Set(S.keep) : null;
  const res = !fts.length ? null : shownMode ? enrTest(p, ft, seq(0, p._ids.length), i => keepSet.has(i)) : enrTest(p, ft, S.keep, i => codes[i] === code);
  const sig = res ? res.tests.filter(x => x.q <= o.q && x.fold > 1).sort((a, b) => a.q - b.q || b.fold - a.fold) : [];
  const fallback = res && !sig.length, rows = (fallback ? res.tests.filter(x => x.fold > 1).sort((a, b) => a.p - b.p) : sig).slice(0, Math.max(1, o.terms));
  const msg = !res ? "No UniProt feature in this report: the Enrichment view needs the feature tables of step 01." : !res.n ? `No protein of ${shownMode ? "the selection" : "cluster " + S.enr.cl} has a ${ft} annotation.` : shownMode && res.n === res.N ? "Every annotated protein of the page is shown: filter or lasso a subset to test it against the page." : !rows.length ? `No ${ft} term is carried by ${o.min}+ proteins and over-represented.` : "";
  $("enrMsg").hidden = !msg; $("enrMsg").textContent = msg;
  if (msg) { Plotly.purge($("enr")); return; }
  const X = x => o.x === "fold" ? x.fold : o.x === "count" ? x.k : x.lq, xt = o.x === "fold" ? "fold enrichment" : o.x === "count" ? "proteins with the term" : "−log10 q";
  const labels = rows.map(x => wrap(res.R.name(x.t).replace(/\s*\[GO:\d+\]$/, ""), o.wrap)), longest = Math.max(...labels.flat().map(l => l.length));
  const qmax = Math.max(...rows.map(x => x.lq), 1e-6), fs = font("tick"), boxW = $("enr").clientWidth || 800;
  const trace = {type: "bar", orientation: "h", y: rows.map((_, j) => j), x: rows.map(X), marker: {color: rows.map(x => x.lq), colorscale: [[0, A.col.enrLow], [1, A.col.enrHigh]], cmin: 0, cmax: qmax, line: {width: 0}, colorbar: {thickness: 12, len: 0.6, tickfont: {size: fs}, title: {text: "−log10 q", side: "right", font: {size: font("axis")}}}},
    text: rows.map(x => `${x.k.toLocaleString()}/${x.n.toLocaleString()}`), textposition: "outside", cliponaxis: false, textfont: {size: fs},
    customdata: rows.map(x => [res.R.name(x.t), x.k, x.n, x.K, x.N, x.fold, x.q]), hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]:,}/%{customdata[2]:,} proteins vs %{customdata[3]:,}/%{customdata[4]:,} in the background<br>fold %{customdata[5]:.2f} · q %{customdata[6]:.2g}<br>click inspects<extra></extra>"};
  const title = `<b>${shownMode ? "Shown proteins vs page" : `Cluster ${esc(S.enr.cl)} of ${esc(short(k))}`}</b> · ${esc(ft)} · ${res.n.toLocaleString()} annotated proteins vs ${res.N.toLocaleString()} · ${fallback ? `no term at q ≤ ${o.q}, most significant shown` : `${sig.length} terms at q ≤ ${o.q}`}`;
  Plotly.react($("enr"), [trace], {...base(), bargap: rows.length < 8 ? 0.55 : 0.25, margin: {l: Math.min(boxW * 0.45, longest * fs * 0.6 + 16), r: 60, t: font("axis") + 24, b: 30 + 2 * font("axis")}, modebar: {orientation: "v"},
    xaxis: {title: {text: xt, font: {size: font("axis")}}, tickfont: {size: fs}, gridcolor: "#e6e8ec", zeroline: false, rangemode: "tozero"},
    yaxis: {tickvals: rows.map((_, j) => j), ticktext: labels.map(l => l.map(esc).join("<br>")), autorange: "reversed", tickfont: {size: fs}, automargin: false},
    annotations: [{text: title, x: 0, xanchor: "left", y: 1, yanchor: "bottom", xref: "paper", yref: "paper", showarrow: false, font: {size: font("axis")}}]},
    {responsive: true, displaylogo: false, toImageButtonOptions: {format: "png", scale: 2, filename: `${p.slug}_enrichment_${shownMode ? "shown" : short(k) + "_" + S.enr.cl}_${ft.replace(/\W+/g, "_")}`}});
  hook("enr", "plotly_click", e => { const pt = e.points && e.points[0]; if (!pt) return; const x = rows[pt.pointIndex]; showInspect({type: "assoc", k, ft, cl: shownMode ? null : S.enr.cl, t: x.t, term: res.R.name(x.t), x}, true); });
  $("ePrev").disabled = $("eCl").selectedIndex <= 0; $("eNext").disabled = $("eCl").selectedIndex >= $("eCl").options.length - 1;
}
view("enrich", {label: "Enrichment", title: "Terms over-represented in a cluster or in the shown proteins", panel: "enrBox", ctl: "ctlEnr", plots: ["enr"], render: enrich, ok: () => FT.length > 0 || enrSingles(P[S.page]).length > 0});
