const CT_PAGE = 100;
function buildAppearance() {
  const range = (path, label, min, max, step) => `<label class="row">${label} <input type="range" data-a="${path}" min="${min}" max="${max}" step="${step}"><output data-v="${path}"></output></label>`;
  $("fontRows").innerHTML = range("scale", "<b>Scale all</b>", 0.5, 2.5, 0.05) + FONT_KEYS.map(([k, label]) => range(`font.${k}`, label, 6, 48, 1)).join("") + `<div class="bar"><button id="fontReset">Reset fonts</button></div>`;
  $("palRow").innerHTML = A.col.palette.map((_, j) => `<input type="color" data-a="col.palette.${j}" title="Palette colour ${j + 1}" aria-label="Palette colour ${j + 1}">`).join("");
  $("specRows").innerHTML = SPECIAL.map(([k, label]) => `<label class="row">${label} <input type="color" data-a="col.${k}"></label>`).join("");
  $("fontReset").onclick = () => { const d = defaults(); A.font = d.font; A.scale = d.scale; changed("font"); };
  $("palReset").onclick = () => { A.col = defaults().col; changed("col"); };
  numberBoxes();
}
function applyCss() {
  const r = document.documentElement.style, mobile = innerWidth < 900;
  r.setProperty("--fs-ui", font("ui") + "px"); r.setProperty("--fs-inspect", font("inspect") + "px");
  r.setProperty("--lw", A.lay.left + "px"); r.setProperty("--rw", A.lay.right + "px");
  document.body.classList.toggle("noleft", mobile ? !S.openLeft : A.lay.noleft);
  document.body.classList.toggle("noright", !S.openRight);
  ["insp", "app"].forEach(t => { $(t).hidden = A.lay.tab !== t; document.querySelector(`[data-tab="${t}"]`).classList.toggle("on", A.lay.tab === t); });
  $("inspTog").classList.toggle("on", S.openRight && A.lay.tab === "insp");
  $("appTog").classList.toggle("on", S.openRight && A.lay.tab === "app");
  $("leftTog").classList.toggle("on", !document.body.classList.contains("noleft"));
}
let fitT = null;
function refit() { clearTimeout(fitT); fitT = setTimeout(() => (VIEWS[S.view].plots || []).forEach(id => { if ($(id).data) Plotly.Plots.resize($(id)); }), 60); }
function paneTog(side, open) {
  if (side === "right") { S.openRight = open === undefined ? !S.openRight : open; if (S.openRight && innerWidth < 900) S.openLeft = false; applyCss(); colourTable(); inspect(); return; }
  const mobile = innerWidth < 900, cur = mobile ? !!S.openLeft : !A.lay.noleft, next = open === undefined ? !cur : open;
  if (mobile) { S.openLeft = next; if (next) S.openRight = false; } else { A.lay.noleft = !next; save(); }
  applyCss(); refit();
}
function rightTab(t) { if (S.openRight && A.lay.tab === t) { paneTog("right", false); return; } A.lay.tab = t; save(); paneTog("right", true); }
function numberBoxes() {
  document.querySelectorAll("output[data-v]").forEach(o => {
    const r = o.previousElementSibling, n = document.createElement("input");
    n.type = "number"; n.className = "num"; n.dataset.a = o.dataset.v; n.setAttribute("aria-label", o.dataset.v);
    if (r && r.type === "range") { n.min = r.min; n.max = r.max; n.step = "any"; }
    o.replaceWith(n);
  });
}
function syncControls() {
  document.querySelectorAll("[data-a]").forEach(el => { const v = aget(el.dataset.a); if (v === undefined) return; if (el.type === "checkbox") el.checked = !!v; else el.value = v; });

}
function changed(path) {
  save(); syncControls();
  const p = P[S.page], n = p.methods.length, pts = VIEWS[S.view].points && $("plot").data;
  if (path === "mk.size" && pts) { Plotly.restyle($("plot"), {"marker.size": S.view === "2d" ? A.mk.size * 2 : A.mk.size}, seq(0, S.gn + n)); return; }
  if (path === "mk.alpha" && pts) { Plotly.restyle($("plot"), {"marker.opacity": A.mk.alpha}, seq(S.gn, S.gn + n)); return; }
  if (path === "mk.ghostAlpha" && pts) { if (S.gn) Plotly.restyle($("plot"), {"marker.opacity": A.mk.ghostAlpha}, seq(0, S.gn)); return; }
  if (path.startsWith("lay.left") || path.startsWith("lay.right")) { applyCss(); refit(); return; }
  if (path.startsWith("font") || path === "scale") applyCss();
  drawLater();
}
function bindControls() {
  const on = e => {
    const el = e.target.closest("[data-a]");
    if (!el || (e.type === "input" && !["range", "color"].includes(el.type))) return;
    const old = aget(el.dataset.a), v = el.type === "checkbox" ? el.checked : el.type === "range" || el.type === "number" ? +el.value : el.value;
    if (el.type === "number" && (!isFinite(v) || el.value === "")) { el.value = old; return; }
    if (v === old) return;
    aset(el.dataset.a, typeof old === "number" && el.type === "number" ? Math.min(+(el.max || Infinity), Math.max(+(el.min || -Infinity), v)) : v);
    changed(el.dataset.a);
  };
  document.addEventListener("input", on);
  document.addEventListener("change", on);
}
function ctSeries(p) { return [...seriesKeys(p), ...(S.netComm ? [COMMUNITY] : [])]; }
function ctCats(p, k) {
  if (k === COMMUNITY) { const c = S.netComm; return c.names.map((nm, i) => [nm, c.base[i], c.counts[i]]); }
  const cats = p.sets[k].cats, cols = tint(p, k, S.keep);
  return order(cats, k !== "sample").map(i => [cats[i][0], cols[i], cats[i][2]]);
}
function colourTable() {
  if (!S.openRight || A.lay.tab !== "app") return;
  const p = page(), keys = ctSeries(p), cur = $("ctSer").value;
  $("ctSer").replaceChildren(...keys.map(k => new Option(k, k)));
  $("ctSer").value = keys.includes(cur) ? cur : keys.includes(S.colour) ? S.colour : keys[0];
  const k = $("ctSer").value, q = S.ctQ.trim().toLowerCase(), o = A.ov[k] || {}, all = ctCats(p, k).filter(c => !q || c[0].toLowerCase().includes(q)), pages = Math.max(1, Math.ceil(all.length / CT_PAGE));
  S.ctPage = Math.min(S.ctPage, pages - 1);
  const rows = all.slice(S.ctPage * CT_PAGE, (S.ctPage + 1) * CT_PAGE).map(([nm, col, n]) => {
    const el = document.createElement("div"), set = !!o[nm], c = o[nm] || col;
    el.className = "ct" + (set ? " set" : "");
    el.innerHTML = `<input type="color" aria-label="Colour"><span class="nm"></span><span class="n">${(n || 0).toLocaleString()}</span>${set ? `<button title="Back to the automatic colour">&#8634;</button>` : ""}`;
    el.querySelector(".nm").textContent = nm; el.title = nm;
    const inp = el.querySelector("input"); inp.value = /^#[0-9a-f]{6}$/i.test(c) ? c : "#888888";
    inp.onchange = () => { setColour(k, nm, inp.value); draw(); };
    if (set) el.querySelector("button").onclick = () => { delete o[nm]; save(); draw(); };
    return el;
  });
  $("ctRows").replaceChildren(...rows);
  $("ctPage").textContent = `${(S.ctPage + 1)} / ${pages} · ${all.length.toLocaleString()} categories`;
  $("ctPrev").disabled = S.ctPage === 0; $("ctNext").disabled = S.ctPage >= pages - 1;
}
function ctFill() {
  const p = page(), k = $("ctSer").value, pal = A.col.palette, o = A.ov[k] = {};
  ctCats(p, k).filter(([nm]) => nm !== "unassigned" && !nm.startsWith("other ")).forEach(([nm], j) => { o[nm] = pal[j % pal.length]; });
  save(); draw();
}
function settingsExport() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({atlas: D.title, settings: A}, null, 1)], {type: "application/json"}));
  a.download = `${D.title.replace(/\W+/g, "_")}_atlas_settings.json`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function settingsImport(file) {
  file.text().then(t => { const j = JSON.parse(t); A = merge(defaults(), j.settings || j); save(); syncControls(); applyCss(); S.netCache = null; draw(); $("setMsg").textContent = `Imported ${file.name}`; }).catch(e => { $("setMsg").textContent = `Not a settings file: ${e.message}`; });
}
function settingsReset() { A = defaults(); save(); syncControls(); applyCss(); S.netCache = null; draw(); $("setMsg").textContent = "Every setting is back to its default."; }
