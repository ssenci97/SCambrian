const ENR_SUMMARY_FEATURES = ["Family", "Keyword", "GO process", "GO function", "GO component", "Pathway", "EC", "Subcellular location"];
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
function contingency(p, k, ft) {
  const codes = p._s[k], cats = p.sets[k].cats, R = termReader(p, ft), TT = R.size, nc = new Map(), nt = new Map(), nct = new Map(), skip = cats.map((c, i) => !i || c[1] === C.color_other);
  let N = 0;
  S.keep.forEach(i => { const c = codes[i]; if (skip[c] || !R.any(i)) return; N++; nc.set(c, (nc.get(c) || 0) + 1); R.each(i, t => { nt.set(t, (nt.get(t) || 0) + 1); const key = c * TT + t; nct.set(key, (nct.get(key) || 0) + 1); }); });
  return {N, nc, nt, nct, TT, R};
}
function enrCompute(p) {
  const o = A.enr, k = S.enr.ser, ft = S.enr.ft, {N, nc, nt, nct, TT, R} = contingency(p, k, ft), cats = p.sets[k].cats;
  if (!N) return {msg: `No shown protein has both a ${short(k)} cluster and a ${ft} annotation.`};
  const tests = [];
  nct.forEach((v, key) => { if (v < o.min) return; const c = Math.floor(key / TT), t = key % TT, a = nc.get(c), b = nt.get(t); tests.push({c, t, v, a, b, frac: v / a, enr: Math.log2((v / a) / (b / N)), p: hyperTail(v, N, b, a)}); });
  if (!tests.length) return {msg: `No cluster × term pair holds ${o.min}+ proteins.`};
  const q = bh(tests.map(x => x.p)); tests.forEach((x, i) => { x.q = q[i]; x.lq = -Math.log10(Math.max(q[i], 1e-300)); x.sig = q[i] <= o.q && x.enr > 0; });
  const rows = [...nc].sort((u, v) => v[1] - u[1]).slice(0, o.rows).map(([c]) => c), rowSet = new Set(rows), score = new Map();
  tests.forEach(x => { if (rowSet.has(x.c) && x.sig) score.set(x.t, Math.max(score.get(x.t) || 0, x.lq)); });
  let cols = [...score].sort((u, v) => v[1] - u[1]).slice(0, o.cols).map(([t]) => t);
  if (!cols.length) return {msg: `No term is enriched in the ${rows.length} largest clusters at q ≤ ${o.q} (min ${o.min} proteins).`};
  const cell = new Map(); tests.forEach(x => { if (rowSet.has(x.c)) cell.set(x.c * TT + x.t, x); });
  const val = x => !x ? null : o.metric === "frac" ? x.frac : !x.sig ? null : o.metric === "lq" ? x.lq : x.enr;
  let rs = rows;
  if (o.order === "name") rs = [...rows].sort((a, b) => cats[a][0].length - cats[b][0].length || cats[a][0].localeCompare(cats[b][0], undefined, {numeric: true}));
  if (o.order === "diag") {
    const best = new Map(rows.map(c => { let bt = -1, bv = -Infinity; cols.forEach(t => { const x = cell.get(c * TT + t); if (x && x.sig && x.lq > bv) { bv = x.lq; bt = t; } }); return [c, [bt, bv]]; }));
    const first = new Map(); rows.forEach(c => { const t = best.get(c)[0]; if (t >= 0 && !first.has(t)) first.set(t, first.size); });
    cols = [...cols].sort((a, b) => (first.has(a) ? first.get(a) : 1e9) - (first.has(b) ? first.get(b) : 1e9));
    const pos = new Map(cols.map((t, j) => [t, j]));
    rs = [...rows].sort((a, b) => { const ta = best.get(a)[0], tb = best.get(b)[0]; return (ta < 0) - (tb < 0) || (pos.get(ta) ?? 1e9) - (pos.get(tb) ?? 1e9) || best.get(b)[1] - best.get(a)[1]; });
  }
  return {N, rows: rs, cols, cell, TT, R, cats, val, tested: tests.length, sig: tests.filter(x => x.sig).length};
}
function nmi(p, k, ft) {
  const {N, nc, nt, nct, TT} = contingency(p, k, ft);
  if (!N) return null;
  let I = 0, hc = 0, ht = 0, purity = 0;
  const best = new Map();
  nct.forEach((v, key) => { const c = Math.floor(key / TT), t = key % TT; I += v / N * Math.log(v * N / (nc.get(c) * nt.get(t))); best.set(c, Math.max(best.get(c) || 0, v)); });
  nc.forEach(v => { hc -= v / N * Math.log(v / N); }); nt.forEach(v => { ht -= v / N * Math.log(v / N); });
  best.forEach(v => { purity += v; });
  return {nmi: hc + ht > 0 ? 2 * I / (hc + ht) : 0, purity: purity / N, n: N, clusters: nc.size, terms: nt.size};
}
function enrSummary(p) {
  const rows = enrClusterKeys(p), singles = enrSingles(p), multis = ENR_SUMMARY_FEATURES.filter(f => FT.includes(f)), metric = A.enr.sumMetric, cols = metric === "nmi" ? singles : [...singles, ...multis];
  if (!cols.length) return {msg: metric === "nmi" ? "NMI needs a single-valued UniProt series (uniprot_family, …): none on this page; switch to purity." : "No UniProt feature on this page."};
  const sig = `${S.page}|${S.keep.length}|${S.keep[0]}|${S.keep[S.keep.length - 1]}|${metric}|${cols.join(",")}`;
  if (!S.enrSum || S.enrSum.sig !== sig) S.enrSum = {sig, z: rows.map(k => cols.map(ft => (metric === "nmi" && !p.sets[ft]) ? null : nmi(p, k, ft)))};
  const ok = cols.map((_, j) => S.enrSum.z.some(r => r[j] && r[j].terms > 1));
  if (!ok.some(Boolean)) return {msg: "Every UniProt feature has a single value over the shown proteins."};
  return {rows, cols: cols.filter((_, j) => ok[j]), z: S.enrSum.z.map(r => r.filter((_, j) => ok[j]))};
}
function heat(el, z, x, y, text, custom, hover, colorbar, zmin, zmax, fname) {
  const lab = Math.max(...y.map(s => s.length)) * font("tick") * 0.62, bot = Math.min(320, Math.max(...x.map(s => s.length)) * font("tick") * 0.62 * 0.75 + 20);
  Plotly.react($(el), [{type: "heatmap", z, x: x.map((_, j) => j), y: y.map((_, j) => j), text, texttemplate: "%{text}", textfont: {size: font("tick")}, customdata: custom, hovertemplate: hover, colorscale: [[0, A.col.low], [1, A.col.high]], zmin, zmax, hoverongaps: false, xgap: 1, ygap: 1, colorbar: {thickness: 12, len: 0.8, tickfont: {size: font("tick")}, title: {text: colorbar, side: "right", font: {size: font("axis")}}}}],
    {...base(), margin: {l: Math.min(360, lab + 16), r: 44, t: font("axis") + 20, b: bot}, modebar: {orientation: "v"}, xaxis: {tickvals: x.map((_, j) => j), ticktext: x.map(esc), tickangle: -45, tickfont: {size: font("tick")}, side: "bottom", showgrid: false, zeroline: false}, yaxis: {tickvals: y.map((_, j) => j), ticktext: y.map(esc), autorange: "reversed", tickfont: {size: font("tick")}, showgrid: false, zeroline: false}, plot_bgcolor: "#f7f8fa"},
    {responsive: true, displaylogo: false, toImageButtonOptions: {format: "png", scale: 2, filename: fname}});
}
function enrich(p) {
  const sers = enrClusterKeys(p), fts = [...enrSingles(p), ...FT], o = A.enr;
  if (!sers.includes(S.enr.ser)) S.enr.ser = D.leiden_key;
  if (!fts.includes(S.enr.ft)) S.enr.ft = fts.includes("uniprot_family") ? "uniprot_family" : fts[0];
  $("eSer").replaceChildren(...sers.map(k => new Option(short(k), k))); $("eSer").value = S.enr.ser;
  const grp = (label, names) => { const g = document.createElement("optgroup"); g.label = label; g.replaceChildren(...names.map(nm => new Option(nm, nm))); return g; };
  $("eFt").replaceChildren(...[grp("Single-valued", enrSingles(p)), grp("Annotation terms", FT.filter(f => !D.features[f].domain)), grp("Domains", FT.filter(f => D.features[f].domain))].filter(g => g.children.length)); $("eFt").value = S.enr.ft;
  const sum = o.mode === "summary";
  document.querySelectorAll(".eClu").forEach(el => el.hidden = sum); document.querySelectorAll(".eSum").forEach(el => el.hidden = !sum);
  const res = !fts.length ? {msg: "No UniProt feature in this report: the Enrichment view needs the feature tables of step 01."} : sum ? enrSummary(p) : enrCompute(p);
  $("enrMsg").hidden = !res.msg; $("enrMsg").textContent = res.msg || "";
  if (res.msg) { Plotly.purge($("enr")); return; }
  if (sum) {
    const m = o.sumMetric, z = res.z.map(r => r.map(v => v ? v[m] : null)), text = z.map(r => r.map(v => v === null ? "" : v.toFixed(2))), custom = res.z.map((r, i) => r.map((v, j) => [res.rows[i], res.cols[j], v ? v.n : 0, v ? v.clusters : 0, v ? v.terms : 0, v ? v.nmi : 0, v ? v.purity : 0]));
    heat("enr", z, res.cols, res.rows.map(short), text, custom, "<b>%{customdata[0]}</b> × <b>%{customdata[1]}</b><br>NMI %{customdata[5]:.3f} · purity %{customdata[6]:.3f}<br>%{customdata[2]:,} shown annotated proteins · %{customdata[3]:,} clusters · %{customdata[4]:,} terms<br>click opens the cluster × term map<extra></extra>", m === "nmi" ? "NMI" : "purity", 0, 1, `${p.slug}_enrichment_summary`);
    Plotly.relayout($("enr"), {annotations: [{text: `<b>Cluster series vs UniProt features</b> · ${m === "nmi" ? "normalised mutual information" : "purity (share of proteins carrying the main term of their cluster)"} · shown proteins`, x: 0, xanchor: "left", y: 1, yanchor: "bottom", xref: "paper", yref: "paper", showarrow: false, font: {size: font("axis")}}]});
    hook("enr", "plotly_click", e => { const pt = e.points && e.points[0]; if (!pt) return; S.enr.ser = res.rows[pt.y]; S.enr.ft = res.cols[pt.x]; A.enr.mode = "clusters"; save(); syncControls(); draw(); });
    return;
  }
  const {rows, cols, cell, TT, R, cats, val} = res, metric = o.metric;
  const z = rows.map(c => cols.map(t => val(cell.get(c * TT + t)))), finite = z.flat().filter(v => v !== null), zmax = finite.length ? Math.max(...finite) : 1;
  const text = rows.map(c => cols.map(t => { const x = cell.get(c * TT + t); if (!x) return ""; if (o.values) return metric === "frac" ? (100 * x.frac).toFixed(0) : metric === "lq" ? x.lq.toFixed(1) : x.sig ? x.enr.toFixed(1) : ""; return x.sig ? (x.q < 0.001 ? "***" : x.q < 0.01 ? "**" : "*") : ""; }));
  const custom = rows.map(c => cols.map(t => { const x = cell.get(c * TT + t); return [cats[c][0], R.name(t), x ? x.v : 0, x ? x.a : 0, x ? x.frac : 0, x ? x.enr : 0, x ? x.q : 1, x ? x.b : 0]; }));
  heat("enr", z, cols.map(t => clip(R.name(t).replace(/\s*\[GO:\d+\]$/, ""), o.clip)), rows.map(c => cats[c][0]), text, custom, "<b>%{customdata[0]}</b> × %{customdata[1]}<br>%{customdata[2]:,} / %{customdata[3]:,} annotated proteins of the cluster (%{customdata[4]:.1%})<br>term on %{customdata[7]:,} shown proteins · log2 enrichment %{customdata[5]:.2f} · q %{customdata[6]:.2g}<br>click inspects<extra></extra>", metric === "frac" ? "share of the cluster" : metric === "lq" ? "−log10 q" : "log2 enrichment", 0, Math.max(zmax, 1e-6), `${p.slug}_enrichment_${short(S.enr.ser)}_${S.enr.ft.replace(/\W+/g, "_")}`);
  Plotly.relayout($("enr"), {annotations: [{text: `<b>${esc(short(S.enr.ser))} × ${esc(S.enr.ft)}</b> · ${rows.length} clusters · ${cols.length} terms · ${res.sig.toLocaleString()}/${res.tested.toLocaleString()} pairs enriched at q ≤ ${o.q} (hypergeometric, BH) · ${res.N.toLocaleString()} shown annotated proteins`, x: 0, xanchor: "left", y: 1, yanchor: "bottom", xref: "paper", yref: "paper", showarrow: false, font: {size: font("axis")}}]});
  hook("enr", "plotly_click", e => {
    const pt = e.points && e.points[0]; if (!pt) return;
    const c = rows[pt.y], t = cols[pt.x], x = cell.get(c * TT + t);
    showInspect({type: "assoc", k: S.enr.ser, ft: S.enr.ft, cl: cats[c][0], t, term: R.name(t), x});
  });
}
view("enrich", {label: "Enrichment", title: "Clusters of any series against UniProt families, GO terms and the other features", panel: "enrBox", ctl: "ctlEnr", plots: ["enr"], render: enrich, ok: () => FT.length > 0 || enrSingles(P[S.page]).length > 0});
