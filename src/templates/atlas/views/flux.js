function fluxKeys(p) {
  const opts = seriesKeys(p), def = p.keys.find(k => /_u_i_/.test(k) && k !== labKey()) || "sample";
  const a = opts.includes(S.flux.from) ? S.flux.from : def, b = opts.includes(S.flux.to) ? S.flux.to : labKey() !== a ? labKey() : D.leiden_key;
  return [a, b];
}
function flux(p) {
  const [fa, fb] = fluxKeys(p), opts = seriesKeys(p), N = Math.max(2, A.lay.fluxNodes);
  ["fFrom", "fTo"].forEach((id, j) => { $(id).replaceChildren(...opts.map(k => new Option(short(k), k))); $(id).value = j ? fb : fa; });
  const A_ = p._s[fa], B = p._s[fb], ca = p.sets[fa].cats, cb = p.sets[fb].cats, ka = tint(p, fa, S.keep), kb = tint(p, fb, S.keep), sp = p._s.sample, spc = p.sets.sample.cats, spk = paint(p, "sample", spc.map(c => remap(c[1])));
  const top = codes => { const n = new Map(); S.keep.forEach(i => { const c = codes[i]; if (c) n.set(c, (n.get(c) || 0) + 1); }); return [...n].sort((u, v) => v[1] - u[1]).slice(0, N).map(([c]) => c); };
  const ta = top(A_), tb = top(B), ia = new Map(ta.map((c, j) => [c, j])), ib = new Map(tb.map((c, j) => [c, j])), oa = ta.length, b0 = oa + 1, ob = b0 + tb.length, links = new Map();
  S.keep.forEach(i => { const a = A_[i], b = B[i]; if (!a || !b) return; const s = ia.has(a) ? ia.get(a) : oa, t = ib.has(b) ? b0 + ib.get(b) : ob, id = s * 65536 + t; let l = links.get(id); if (!l) links.set(id, l = {s, t, n: 0, sp: new Uint32Array(spc.length)}); l.n++; l.sp[sp[i]]++; });
  const L = [...links.values()], msg = fa === fb ? "Pick two different series." : !L.length ? "No shown protein has a cluster in both series." : "";
  $("fluxMsg").hidden = !msg; $("fluxMsg").textContent = msg;
  if (msg) { Plotly.purge($("flux")); return; }
  const label = [...ta.map(c => ca[c][0]), `other ${short(fa)}`, ...tb.map(c => cb[c][0]), `other ${short(fb)}`], color = [...ta.map(c => ka[c]), A.col.other, ...tb.map(c => kb[c]), A.col.other];
  const major = L.map(l => l.sp.indexOf(Math.max(...l.sp)));
  const trace = {type: "sankey", arrangement: "snap", valueformat: ",d", textfont: {size: font("sankey"), family: C.font_family}, node: {label, color, pad: 10, thickness: 18, line: {width: 0.5, color: "#ffffff"}, hovertemplate: "<b>%{label}</b>: %{value:,} proteins<br>click to isolate<extra></extra>"}, link: {source: L.map(l => l.s), target: L.map(l => l.t), value: L.map(l => l.n), color: major.map(m => spk[m] + "88"), customdata: L.map((l, j) => `${spc[major[j]][0]} (${Math.round(100 * l.sp[major[j]] / l.n)}%)`), hovertemplate: "%{source.label} → %{target.label}: %{value:,}<br>mostly %{customdata}<extra></extra>"}};
  const head = (text, x, xanchor) => ({text: `<b>${esc(text)}</b>`, x, xanchor, y: 1, yanchor: "bottom", xref: "paper", yref: "paper", showarrow: false, font: {size: font("title")}});
  Plotly.react($("flux"), [trace], {...base(), margin: {l: 10, r: 44, t: font("title") + 20, b: 10}, annotations: [head(short(fa), 0, "left"), head(short(fb), 1, "right")], modebar: {orientation: "v"}}, {responsive: true, displaylogo: false, toImageButtonOptions: {format: "png", scale: 2, filename: `${p.slug}_flux_${short(fa)}_${short(fb)}`}});
  hook("flux", "plotly_click", e => {
    const pt = e.points && e.points[0]; if (!pt || pt.source !== undefined) return;
    const j = pt.pointNumber ?? pt.index, side = j < oa ? [fa, ta[j]] : j >= b0 && j < ob ? [fb, tb[j - b0]] : null;
    if (side) isolate(p.sets[side[0]].cats[side[1]][0], side[0]);
  });
}
view("flux", {label: "Flux", title: "Proteins flowing between the clusters of two series", panel: "fluxBox", ctl: "ctlFlux", plots: ["flux"], render: flux});
