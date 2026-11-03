function hidden(k) { return k === "sample" ? S.f.sample : S.f.series[k] = S.f.series[k] || new Set(); }
function setHidden(k, names) { if (k === "sample") S.f.sample = names; else S.f.series[k] = names; }
function listKey(s) { return s === "sp" ? "sample" : S.fser; }
function order(cats, byName) {
  return cats.map((_, i) => i).filter(i => cats[i][2] > 0).sort((a, b) => (a === 0) - (b === 0) || (byName ? cats[a][0].length - cats[b][0].length || cats[a][0].localeCompare(cats[b][0], undefined, {numeric: true}) : cats[b][2] - cats[a][2]));
}
function shown(s) { const q = S.q[s].trim().toLowerCase(), p = P[S.page], k = listKey(s), cats = p.sets[k].cats; return order(cats, k !== "sample").filter(i => !q || cats[i][0].toLowerCase().includes(q)); }
let timer = null;
function chip(k, nm, all) {
  const h = hidden(k);
  if (timer) { clearTimeout(timer); timer = null; setHidden(k, new Set(all.filter(x => x !== nm))); draw(); return; }
  timer = setTimeout(() => { timer = null; h.has(nm) ? h.delete(nm) : h.add(nm); draw(); }, 240);
}
function checklist(s) {
  const p = P[S.page], k = listKey(s), cats = p.sets[k].cats, b = p.views.bars, leiden = k === D.leiden_key, list = shown(s), all = order(cats, k !== "sample").map(i => cats[i][0]), h = hidden(k), coloured = k === S.colour;
  const cols = k === labKey() ? S.cols : tint(p, k, S.keep), max = A.lay.maxChips;
  const els = list.slice(0, max).map(i => {
    const [nm, , cnt] = cats[i], col = cols[i], zl = (A.z[k] || {})[nm] || 0, el = document.createElement("div"), off = h.has(nm), bi = leiden ? p._b[nm] : undefined, st = bi !== undefined && b.stars ? b.stars[bi] : "";
    el.className = "chip" + (off ? " off" : "");
    el.title = `${nm} (n=${cnt.toLocaleString()})` + (st ? `\nχ² ${st}` : "");
    el.innerHTML = `<input type="checkbox" tabindex="-1"${off ? "" : " checked"}><i class="${coloured ? "" : "plain"}" style="background:${col}"></i><span class="nm"></span><span class="st">${st}</span><span class="z${zl ? " set" : ""}"><b data-i title="Inspect">&#9432;</b><b data-z="1" class="${zl > 0 ? "on" : ""}" title="Draw in front of the others">&#9650;</b><b data-z="-1" class="${zl < 0 ? "on" : ""}" title="Draw behind the others">&#9660;</b></span><span class="n">${cnt.toLocaleString()}</span>`;
    el.querySelector(".nm").textContent = nm;
    el.querySelectorAll("[data-z]").forEach(z => z.onclick = e => { e.stopPropagation(); zmove(k, nm, +z.dataset.z); });
    el.querySelector("[data-i]").onclick = e => { e.stopPropagation(); showInspect({type: "cluster", k, nm}); };
    el.querySelector("i").onclick = e => { e.stopPropagation(); swatch(e, k, nm, col); };
    el.onclick = () => chip(k, nm, all);
    return el;
  });
  if (list.length > max) { const m = document.createElement("div"); m.className = "muted more-n"; m.textContent = `+${(list.length - max).toLocaleString()} more · refine the filter (All/None act on every match)`; els.push(m); }
  return els;
}
function swatch(e, k, nm, col) {
  if (e.shiftKey) { if (A.ov[k]) delete A.ov[k][nm]; save(); draw(); return; }
  const el = $("picker"); el._pick = [k, nm];
  el.style.left = e.clientX + "px"; el.style.top = e.clientY + "px"; el.value = /^#[0-9a-f]{6}$/i.test(col) ? col : "#888888";
  try { el.showPicker(); } catch (err) { el.click(); }
}
function setColour(k, nm, v) { (A.ov[k] = A.ov[k] || {})[nm] = v; save(); }
function pills(p) {
  const out = [], add = (label, body, rm, open) => out.push({label, body, rm, open}), names = arr => arr.slice(0, 3).join(", ") + (arr.length > 3 ? ` +${arr.length - 3}` : "");
  [["sample", S.f.sample], ...Object.entries(S.f.series)].forEach(([k, h]) => {
    if (!h.size || !p.sets[k]) return;
    const all = order(p.sets[k].cats, k !== "sample").map(i => p.sets[k].cats[i][0]), vis = all.filter(x => !h.has(x)), hid = all.filter(x => h.has(x));
    add(k === "sample" ? "Samples" : short(k), vis.length <= hid.length ? (vis.length ? names(vis) : "none") : "not " + names(hid), () => { h.clear(); }, () => { if (k === "sample") $("secSp").open = true; else { S.fser = k; $("secCl").open = true; } });
  });
  Object.entries(S.ft).forEach(([nm, f]) => { if (f.on.size) add(nm, names([...f.on].map(t => D.features[nm].vocab[t])).replace(/, /g, f.and ? " AND " : " OR "), () => f.on.clear(), () => { S.ftSel = nm; $("secFt").open = true; }); });
  if (S.lasso) add("Selection", `${S.lasso.size.toLocaleString()} points`, () => { S.lasso = null; }, () => { $("secSel").open = true; });
  if (S.find !== null) add("Find", p._ids[S.find], () => { S.find = null; $("find").value = ""; }, () => { $("secFind").open = true; });
  const els = out.map(o => {
    const el = document.createElement("span"); el.className = "pill"; el.title = `${o.label}: ${o.body}`;
    el.innerHTML = `<span><b></b> <span class="pb"></span></span><button aria-label="Remove filter">&#10005;</button>`;
    el.querySelector("b").textContent = o.label + ":"; el.querySelector(".pb").textContent = o.body;
    el.querySelector("button").onclick = e => { e.stopPropagation(); o.rm(); draw(); };
    el.onclick = () => { if (!$("left").offsetParent) paneTog("left", true); o.open(); draw(); };
    return el;
  });
  if (!els.length) { const m = document.createElement("span"); m.className = "none"; m.textContent = "No filter · every point of the page is shown"; els.push(m); }
  $("pills").replaceChildren(...els);
  $("clearAll").disabled = !out.length;
}
function sidebar(p, n) {
  S.n = n;
  $("linfo").textContent = `${n.toLocaleString()} / ${p._s.sample.length.toLocaleString()} points shown`;
  $("selInfo").textContent = S.lasso ? `${S.lasso.size.toLocaleString()} points selected by lasso or cluster subset` : "Draw a lasso in the 2D view to keep a subset of points.";
  $("lassoClr").disabled = !S.lasso;
  $("secSp").hidden = order(p.sets.sample.cats, false).filter(i => i > 0).length < 2;
  $("fserSel").replaceChildren(...p.keys.map(k => new Option(k + (S.f.series[k] && S.f.series[k].size ? " ●" : ""), k))); $("fserSel").value = S.fser;
  $("listSp").replaceChildren(...checklist("sp"));
  $("listCl").replaceChildren(...checklist("cl"));
  const pg = P[S.page]; $("selBtn").hidden = !(pg.selected || []).length; $("probeBtn").hidden = !(pg.probe || []).length;
  ftList();
  pills(p);
}
function ftTerms() {
  const nm = S.ftSel, F = feat(nm), st = ftState(nm), q = S.q.ft.trim().toLowerCase(), cnt = S.ftCnt || [];
  return F.vocab.map((_, t) => t).filter(t => (cnt[t] || st.on.has(t)) && (!q || F.vocab[t].toLowerCase().includes(q))).sort((a, b) => st.on.has(b) - st.on.has(a) || cnt[b] - cnt[a] || a - b);
}
function ftList() {
  if (!FT.length) return;
  const nm = S.ftSel, F = feat(nm), st = ftState(nm), cnt = S.ftCnt || [], list = ftTerms(), act = Object.entries(S.ft).filter(([, f]) => f.on.size), max = A.lay.maxChips;
  $("ftSel").value = nm; $("ftAnd").checked = st.and;
  $("ftInfo").textContent = act.length ? "Active: " + act.map(([n, f]) => `${n} (${f.on.size} ${f.and ? "AND" : "OR"})`).join(" · ") : `${list.length.toLocaleString()} terms among the shown proteins`;
  const els = list.slice(0, max).map(t => {
    const el = document.createElement("div"), on = st.on.has(t), c = cnt[t] || 0;
    el.className = "chip" + (st.on.size && !on ? " off" : "");
    el.title = `${F.vocab[t]}\n${c.toLocaleString()} shown proteins · ${F.n[t].toLocaleString()} in the dataset`;
    el.innerHTML = `<input type="checkbox" tabindex="-1"${on ? " checked" : ""}><span class="nm"></span><span class="n">${c.toLocaleString()}</span>`;
    el.querySelector(".nm").textContent = F.vocab[t];
    el.onclick = () => ftChip(t);
    return el;
  });
  if (list.length > max) { const m = document.createElement("div"); m.className = "muted more-n"; m.textContent = `+${(list.length - max).toLocaleString()} more · refine the filter (All acts on every match)`; els.push(m); }
  $("listFt").replaceChildren(...els);
}
function ftChip(t) {
  const st = ftState(S.ftSel);
  if (timer) { clearTimeout(timer); timer = null; st.on = new Set([t]); draw(); return; }
  timer = setTimeout(() => { timer = null; st.on.has(t) ? st.on.delete(t) : st.on.add(t); draw(); }, 240);
}
function ftIsolate(nm, t, add) { S.ftSel = nm; S.q.ft = ""; $("qFt").value = ""; const st = ftState(nm); if (add) st.on.add(t); else st.on = new Set([t]); $("secFt").open = true; toPoints(); draw(); }
function toPoints() { if (!VIEWS[S.view].points) { S.view = S.last; S.fit = true; syncView(); } }
function isolate(name, k = D.leiden_key) {
  const p = page(), cats = p.sets[k].cats;
  if (!cats.some(c => c[0] === name)) return;
  toPoints();
  const all = order(cats, k !== "sample").map(i => cats[i][0]), h = hidden(k), only = h.size === all.length - 1 && !h.has(name);
  if (k !== "sample") { S.colour = k; S.fser = k; S.q.cl = ""; $("qCl").value = ""; }
  S.lasso = null;
  setHidden(k, only ? new Set() : new Set(all.filter(x => x !== name)));
  showInspect({type: "cluster", k, nm: name}, true);
  sync(); draw();
}
function unfilter() { S.f = {sample: new Set(), series: {}}; S.q.sp = S.q.cl = ""; $("qSp").value = ""; $("qCl").value = ""; }
function tick(list) {
  const p = P[S.page], want = new Set(list || []), cats = p.sets[D.leiden_key].cats;
  if (!want.size) return;
  S.colour = D.leiden_key; S.fser = D.leiden_key; S.q.cl = ""; $("qCl").value = "";
  S.f.series[D.leiden_key] = new Set(cats.map(c => c[0]).filter(nm => !want.has(nm)));
}
function startFilters() { tick(P[S.page].selected); }
function find(q) {
  const p = page(); q = q.trim().toLowerCase(); S.find = null;
  if (q) {
    let i = p._ids.findIndex(id => id.toLowerCase() === q);
    if (i < 0) i = p._ids.findIndex(id => id.toLowerCase().includes(q));
    if (i >= 0) { S.find = i; S.f.sample.delete(p.sets.sample.cats[p._s.sample[i]][0]); Object.entries(S.f.series).forEach(([k, h]) => h.delete(p.sets[k].cats[p._s[k][i]][0])); showInspect({type: "protein", i, page: S.page}, true); }
  }
  $("find").classList.toggle("miss", !!q && S.find === null);
  draw();
}
function exportIds() {
  const p = page(), a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([S.keep.map(i => p._ids[i]).join("\n") + "\n"], {type: "text/plain"}));
  a.download = `${p.slug}_${S.colour.replace(/\W+/g, "_")}_${S.keep.length}_ids.txt`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
