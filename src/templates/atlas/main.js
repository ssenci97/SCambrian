function syncView() {
  const V = VIEWS[S.view], own = [V.panel, V.ctl].flat().filter(Boolean);
  new Set(Object.values(VIEWS).flatMap(v => [v.panel, v.ctl].flat()).filter(Boolean)).forEach(id => $(id).hidden = !own.includes(id));
  document.querySelectorAll("[data-view]").forEach(b => { b.classList.toggle("on", b.dataset.view === S.view); b.disabled = !viewOk(b.dataset.view); });
  $("homeBtn").classList.toggle("on", S.view === "overview");
  $("reset").title = S.view === "3d" ? "Reset camera" : "Reset zoom";
  route();
}
function setView(v) { if (!viewOk(v)) v = "3d"; if (v === S.view) { draw(); return; } S.view = v; if (VIEWS[v].points) S.last = v; S.fit = true; syncView(); draw(); }
function syncProj() { const pr = S.proj; $("pX").value = pr.x; $("pY").value = pr.y; $("pAz").value = pr.az; $("pEl").value = pr.el; $("pAzN").value = pr.az; $("pElN").value = pr.el; }
function setAxis(w, v) { const o = w === "x" ? "y" : "x"; if (S.proj[o] === v) S.proj[o] = S.proj[w]; S.proj[w] = v; syncProj(); draw(); }
function sync() {
  const p = P[S.page], keys = seriesKeys(p);
  if (!keys.includes(S.colour)) S.colour = D.leiden_key;
  if (!p.keys.includes(S.fser)) S.fser = D.leiden_key;
  const grp = (label, ks) => { const g = document.createElement("optgroup"); g.label = label; g.replaceChildren(...ks.map(k => new Option(k === "sample" ? "sample" : k, k))); return g; };
  const single = keys.filter(k => (D.single || []).includes(k)), clus = keys.filter(k => k !== "sample" && !single.includes(k));
  $("colourSel").replaceChildren(...[grp("Samples", ["sample"]), grp("Cluster series", clus), grp("UniProt", single)].filter(g => g.children.length)); $("colourSel").value = S.colour;
  $("pageSel").value = S.page; $("prev").disabled = S.page === 0; $("next").disabled = S.page === P.length - 1;
  $("h1").textContent = p.title; $("sub").textContent = `${p.subtitle} · page ${S.page + 1} / ${P.length}`;
  route();
}
function route() { try { history.replaceState(null, "", `#${S.page + 1}/${S.view}`); } catch (e) {} }
function go(i, keepView) {
  if (i < 0 || i >= P.length || i === S.page) return;
  S.page = i; S.find = null; S.lasso = null; S.flux = {from: null, to: null}; S.ft = {}; S.netCache = null; S.enr.cl = null; S.more = {sp: false, cl: false, ft: false}; $("find").value = ""; $("find").classList.remove("miss");
  if (S.insp && S.insp.type !== "cluster") S.insp = null;
  if (!keepView && !viewOk(S.view)) S.view = S.last;
  S.fit = true; unfilter(); syncView(); sync(); draw();
}
function colourBy(k) { S.colour = k; if (k !== "sample") S.fser = k; sync(); draw(); }
document.title = D.title;
$("pageSel").replaceChildren(...P.map((p, i) => new Option(`${i + 1}. ${p.title}`, i)));
$("viewSeg").replaceChildren(...Object.entries(VIEWS).filter(([, v]) => !v.nav).map(([id, v]) => { const b = document.createElement("button"); b.dataset.view = id; b.textContent = v.label; b.title = `${v.title} (V cycles views)`; b.onclick = () => setView(id); return b; }));
["pX", "pY"].forEach(id => $(id).replaceChildren(...[0, 1, 2].map(d => new Option(`dim ${d + 1}`, d))));
$("pX").onchange = e => setAxis("x", +e.target.value);
$("pY").onchange = e => setAxis("y", +e.target.value);
["pAz", "pEl"].forEach(id => { $(id).oninput = e => { S.proj[id === "pAz" ? "az" : "el"] = +e.target.value; syncProj(); }; $(id).onchange = () => draw(); $(id + "N").onchange = e => { const r = $(id), v = Math.max(+r.min, Math.min(+r.max, +e.target.value || 0)); S.proj[id === "pAz" ? "az" : "el"] = v; syncProj(); draw(); }; });
$("pDef").onclick = () => { S.proj = {x: 0, y: 1, az: 0, el: 0}; syncProj(); draw(); };
$("fFrom").onchange = e => { S.flux = {from: e.target.value, to: fluxKeys(page())[1]}; draw(); };
$("fTo").onchange = e => { S.flux = {from: fluxKeys(page())[0], to: e.target.value}; draw(); };
$("fSwap").onclick = () => { const [a, b] = fluxKeys(page()); S.flux = {from: b, to: a}; draw(); };
$("secFt").hidden = !FT.length;
const ftGroup = (label, names) => { const g = document.createElement("optgroup"); g.label = label; g.replaceChildren(...names.map(nm => new Option(nm, nm))); return g; };
$("ftSel").replaceChildren(...[ftGroup("Annotation", FT.filter(nm => !D.features[nm].domain)), ftGroup("Domains", FT.filter(nm => D.features[nm].domain))].filter(g => g.children.length));
$("ftSel").onchange = e => { S.ftSel = e.target.value; S.q.ft = ""; $("qFt").value = ""; draw(); };
$("ftAnd").onchange = e => { ftState(S.ftSel).and = e.target.checked; draw(); };
$("qFt").oninput = e => { S.q.ft = e.target.value; ftList(); };
$("ftNone").onclick = () => { ftState(S.ftSel).on.clear(); draw(); };
$("nFeat").onchange = e => { S.net = e.target.value; draw(); };
$("nRelayout").onclick = () => { A.net.seed = (A.net.seed || 1) + 1; save(); draw(); };
$("eSer").onchange = e => { S.enr.ser = e.target.value; S.enr.cl = null; draw(); };
$("eCl").onchange = e => { S.enr.cl = e.target.value; draw(); };
$("ePrev").onclick = () => enrStep(-1);
$("eNext").onclick = () => enrStep(1);
$("eFt").onchange = e => { S.enr.ft = e.target.value; draw(); };
$("homeBtn").onclick = () => setView("overview");
$("prev").onclick = () => go(S.page - 1);
$("next").onclick = () => go(S.page + 1);
$("pageSel").onchange = e => go(+e.target.value);
$("colourSel").onchange = e => colourBy(e.target.value);
$("fserSel").onchange = e => { S.fser = e.target.value; S.q.cl = ""; $("qCl").value = ""; S.more.cl = false; sidebar(page(), S.n); };
$("picker").onchange = e => { const pk = e.target._pick; if (!pk) return; setColour(pk[0], pk[1], e.target.value); e.target._pick = null; draw(); };
const listOf = w => w === "sample" ? "sp" : "cl";
document.querySelectorAll("[data-all]").forEach(b => b.onclick = () => { const s = listOf(b.dataset.all), k = listKey(s), cats = P[S.page].sets[k].cats, h = hidden(k); shown(s).forEach(i => h.delete(cats[i][0])); draw(); });
document.querySelectorAll("[data-none]").forEach(b => b.onclick = () => { const s = listOf(b.dataset.none), k = listKey(s), cats = P[S.page].sets[k].cats, h = hidden(k); shown(s).forEach(i => h.add(cats[i][0])); draw(); });
$("logBtn").onclick = () => { S.log = !S.log; bars(page()); };
$("pctBtn").onclick = () => { S.pct = !S.pct; bars(page()); };
$("find").onkeydown = e => { if (e.key === "Enter") find(e.target.value); else if (e.key === "Escape") { e.target.value = ""; find(""); e.target.blur(); } };
$("find").onsearch = e => { if (!e.target.value) find(""); };
$("exportIds").onclick = exportIds;
$("lassoTog").checked = S.drag === "lasso";
$("lassoTog").onchange = e => { S.drag = e.target.checked ? "lasso" : "pan"; if (S.view === "2d") Plotly.relayout($("plot"), {dragmode: S.drag}); };
$("selBtn").onclick = () => { delete S.f.series[D.leiden_key]; tick(P[S.page].selected); sync(); draw(); };
$("probeBtn").onclick = () => { delete S.f.series[D.leiden_key]; tick(P[S.page].probe); sync(); draw(); };
$("reset").onclick = () => { S.rev++; S.cam = {}; draw(); };
$("qSp").oninput = e => { S.q.sp = e.target.value; sidebar(page(), S.n); };
$("qCl").oninput = e => { S.q.cl = e.target.value; sidebar(page(), S.n); };
$("leftTog").onclick = () => paneTog("left");
$("inspTog").onclick = () => rightTab("insp");
$("appTog").onclick = () => rightTab("app");
document.querySelectorAll("[data-tab]").forEach(b => b.onclick = () => { A.lay.tab = b.dataset.tab; save(); applyCss(); colourTable(); inspect(); });
$("helpBtn").onclick = () => { $("help").hidden = !$("help").hidden; };
$("rightClose").onclick = () => paneTog("right", false);
$("helpClose").onclick = () => { $("help").hidden = true; };
$("ctSer").onchange = () => { S.ctPage = 0; colourTable(); };
$("ctQ").oninput = e => { S.ctQ = e.target.value; S.ctPage = 0; colourTable(); };
$("ctPrev").onclick = () => { S.ctPage = Math.max(0, S.ctPage - 1); colourTable(); };
$("ctNext").onclick = () => { S.ctPage++; colourTable(); };
$("ctFill").onclick = ctFill;
$("ctReset").onclick = () => { delete A.ov[$("ctSer").value]; save(); draw(); };
$("setExport").onclick = settingsExport;
$("setImport").onclick = () => $("setFile").click();
$("setFile").onchange = e => { const f = e.target.files[0]; if (f) settingsImport(f); e.target.value = ""; };
$("setReset").onclick = settingsReset;
addEventListener("keydown", e => {
  if (e.key === "Escape") { $("help").hidden = true; if (innerWidth < 900) S.openLeft = false; if (S.openRight) paneTog("right", false); else applyCss(); }
  if (e.ctrlKey || e.metaKey || e.altKey || (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) && !["checkbox", "range", "color"].includes(e.target.type))) return;
  if (e.key === "ArrowLeft") go(S.page - 1);
  else if (e.key === "ArrowRight") go(S.page + 1);
  else if (e.key === "c" || e.key === "C") colourBy(S.colour === "sample" ? S.fser : "sample");
  else if (e.key === "l" || e.key === "L") { S.log = !S.log; if (S.view === "bars") bars(page()); }
  else if (e.key === "p" || e.key === "P") { S.pct = !S.pct; if (S.view === "bars") bars(page()); }
  else if (e.key === "v" || e.key === "V") { const vs = Object.keys(VIEWS).filter(v => !VIEWS[v].nav && viewOk(v)); setView(vs[(vs.indexOf(S.view) + 1) % vs.length]); }
  else if (e.key === "/") { e.preventDefault(); if (document.body.classList.contains("noleft")) paneTog("left", true); $("secFind").open = true; $("find").focus(); }
});
let rt = null, wasMobile = innerWidth < 900;
addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { const m = innerWidth < 900; if (m !== wasMobile) { wasMobile = m; applyCss(); } if (VIEWS[S.view].points) draw(); else if (S.view === "bars" || S.view === "enrich") draw(); }, 200); });
S.page = 0;
buildAppearance(); bindControls(); syncControls(); applyCss();
syncView(); syncProj(); sync(); draw();
