const INSPECT_FEATURES = ["GO process", "GO function", "Keyword", "Family", "Pathway"];
function showInspect(o, open) {
  S.insp = o;
  if (open) { A.lay.tab = "insp"; save(); paneTog("right", true); }
}
function card(title, sub, body, actions) {
  const el = document.createElement("div"); el.className = "icard";
  el.innerHTML = `<div class="muted">${esc(sub)}</div><h3></h3>${body}<div class="bar"></div>`;
  el.querySelector("h3").textContent = title;
  actions.forEach(([label, fn, tip]) => { const b = document.createElement("button"); b.textContent = label; if (tip) b.title = tip; b.onclick = fn; el.querySelector(".bar").append(b); });
  return el;
}
function table(rows) { return `<table>${rows.map(r => `<tr>${r.map((c, j) => `<td${j ? "" : ' class="l"'}>${c}</td>`).join("")}</tr>`).join("")}</table>`; }
function termsOf(r, max) {
  return FT.filter(nm => !nm.startsWith("GO (")).map(nm => { const F = feat(nm), ts = Array.from(F._idx.subarray(F._off[r], F._off[r + 1]), t => F.vocab[t]); return ts.length ? `<div><b>${esc(nm)}</b>: ${ts.slice(0, max).map(t => `<span class="tag">${esc(t)}</span>`).join("")}${ts.length > max ? ` <span class="muted">+${ts.length - max}</span>` : ""}</div>` : ""; }).join("");
}
function proteinCard(p, i) {
  const r = p._rows[i], id = p._ids[i], desc = DESC && DESC[r] ? DESC[r] : "";
  const rows = seriesKeys(p).map(k => { const c = p._s[k][i], cols = k === labKey() ? S.cols : tint(p, k, S.keep); return [short(k), `<i class="sw" style="background:${cols[c]}"></i>${esc(p.sets[k].cats[c][0])}`]; });
  const acts = [["Copy ID", () => { try { navigator.clipboard.writeText(id); } catch (e) {} }], ["Ring it", () => { $("find").value = id; find(id); }, "Draw a ring around it in 3D and 2D"]];
  if (desc) acts.push(["UniProt ↗", () => open(`https://www.uniprot.org/uniprotkb/${encodeURIComponent(id)}`, "_blank", "noopener")]);
  return card(id, `Protein · ${p.title}`, (desc ? `<div>${desc}</div>` : "") + table(rows) + termsOf(r, FOUND_TERMS), acts);
}
function clusterCard(p, k, nm) {
  const cats = p.sets[k].cats, c = cats.findIndex(x => x[0] === nm);
  if (c < 0) return null;
  const codes = p._s[k], sp = p._s.sample, spc = p.sets.sample.cats, per = new Uint32Array(spc.length), keepSet = new Set(S.keep), rows = p._rows, fts = (INSPECT_FEATURES.filter(f => FT.includes(f)).length ? INSPECT_FEATURES.filter(f => FT.includes(f)) : FT.filter(f => !D.features[f].domain)).slice(0, 3), tc = fts.map(f => new Map());
  let shownN = 0, total = 0;
  for (let i = 0; i < codes.length; i++) {
    if (codes[i] !== c) continue;
    total++; per[sp[i]]++; if (keepSet.has(i)) shownN++;
    fts.forEach((f, j) => { const F = feat(f), r = rows[i]; for (let q = F._off[r]; q < F._off[r + 1]; q++) tc[j].set(F._idx[q], (tc[j].get(F._idx[q]) || 0) + 1); });
  }
  const spCols = paint(p, "sample", spc.map(x => remap(x[1]))), bi = k === D.leiden_key ? p._b[nm] : undefined, b = p.views.bars, tags = [];
  if (bi !== undefined && b.stars && b.stars[bi]) tags.push(`χ² ${b.stars[bi]}`);
  if ((p.selected || []).includes(nm) && k === D.leiden_key) tags.push("selected");
  if ((p.probe || []).includes(nm) && k === D.leiden_key) tags.push("probe cluster");
  const spRows = [...per].map((n, j) => [j, n]).filter(([, n]) => n).sort((u, v) => v[1] - u[1]).map(([j, n]) => [`<i class="sw" style="background:${spCols[j]}"></i>${esc(spc[j][0])}`, `${n.toLocaleString()} · ${(100 * n / total).toFixed(1)}%`]);
  const terms = fts.map((f, j) => { const top = [...tc[j]].sort((u, v) => v[1] - u[1]).slice(0, 6); return top.length ? `<div><b>${esc(f)}</b>: ${top.map(([t, n]) => `<span class="tag" title="${n.toLocaleString()} proteins">${esc(D.features[f].vocab[t])} · ${(100 * n / total).toFixed(0)}%</span>`).join("")}</div>` : ""; }).join("");
  const body = `<div>${total.toLocaleString()} proteins · ${shownN.toLocaleString()} shown</div>${tags.map(t => `<span class="tag">${t}</span>`).join("")}${table(spRows)}${terms}`;
  return card(nm, `Cluster of ${short(k)} · ${p.title}`, body, [["Isolate", () => isolate(nm, k)], ["Hide", () => { hidden(k).add(nm); draw(); }], ["Show", () => { hidden(k).delete(nm); draw(); }], ...(FT.length && k !== "sample" ? [["Enrichment", () => { S.enr.ser = k; S.enr.cl = nm; setView("enrich"); }, "Terms over-represented in this cluster"]] : [])]);
}
function termCard(o) {
  const F = D.features[o.nm], nb = o.nb.slice(0, 10).map(([t, w, j]) => [esc(clip(F.vocab[t], 60)), `${w.toLocaleString()} · J ${j.toFixed(2)}`]);
  return card(F.vocab[o.t], `${o.nm} term · network`, `<div>${o.n.toLocaleString()} shown proteins (${(100 * o.share).toFixed(1)}%) · ${F.n[o.t].toLocaleString()} in the dataset</div>${o.comm ? `<span class="tag">${esc(o.comm)}</span>` : ""}${nb.length ? `<div class="muted">Strongest neighbours (shared proteins · Jaccard)</div>${table(nb)}` : ""}`, [["Keep its proteins", () => ftIsolate(o.nm, o.t)], ["Add to filter", () => ftIsolate(o.nm, o.t, true)]]);
}
function assocCard(o) {
  const x = o.x, single = !FT.includes(o.ft), rows = [["Cluster proteins with the term", `${x.k.toLocaleString()} / ${x.n.toLocaleString()} (${(100 * x.k / x.n).toFixed(1)}%)`], ["Background proteins with the term", `${x.K.toLocaleString()} / ${x.N.toLocaleString()} (${(100 * x.K / x.N).toFixed(1)}%)`], ["Fold enrichment", x.fold.toFixed(2)], ["p (hypergeometric)", x.p.toExponential(2)], ["q (BH)", x.q.toExponential(2)]];
  const keepTerm = () => single ? isolate(o.term, o.ft) : ftIsolate(o.ft, o.t);
  const acts = [["Keep term proteins", keepTerm]];
  if (o.cl !== null) acts.unshift(["Isolate cluster", () => isolate(o.cl, o.k)]);
  return card(o.term, o.cl === null ? `${o.ft} · shown proteins against the page` : `${o.ft} · cluster ${o.cl} of ${short(o.k)}`, table(rows), acts);
}
function inspect() {
  if (document.body.classList.contains("noright") || A.lay.tab !== "insp") return;
  const o = S.insp, p = page();
  let el = null;
  if (o && o.type === "protein" && o.page === S.page) el = proteinCard(p, o.i);
  else if (o && o.type === "cluster" && p.sets[o.k]) el = clusterCard(p, o.k, o.nm);
  else if (o && o.type === "term") el = termCard(o);
  else if (o && o.type === "assoc") el = assocCard(o);
  if (el) $("inspBody").replaceChildren(el); else { $("inspBody").className = "muted"; $("inspBody").textContent = "Click a point, a bar, a flux node, a network node or the ⓘ of a cluster to inspect it."; return; }
  $("inspBody").className = "";
}
