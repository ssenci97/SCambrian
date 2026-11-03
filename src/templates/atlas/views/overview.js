function overview() {
  const total = P[0] ? P[0].sets.sample.cats.reduce((s, c) => s + c[2], 0) : 0;
  const head = document.createElement("div");
  head.innerHTML = `<h2></h2><div class="muted"></div>`;
  head.querySelector("h2").textContent = D.title;
  head.querySelector(".muted").textContent = `${P.length} pages · ${total.toLocaleString()} proteins on the joint page · ${D.leiden_key} is the report series` + (FT.length ? ` · UniProt features: ${FT.join(", ")}` : "");
  const cards = document.createElement("div"); cards.className = "cards";
  cards.replaceChildren(...P.map((p, j) => {
    const sp = p.sets.sample.cats, cols = paint(p, "sample", sp.map(c => remap(c[1]))), n = sp.reduce((s, c) => s + c[2], 0), lk = p.sets[D.leiden_key].cats.filter((c, i) => i && c[2] > 0).length;
    const el = document.createElement("div"); el.className = "card" + (j === S.page ? " cur" : ""); el.tabIndex = 0;
    const comp = sp.map((c, i) => c[2] && i ? `<span style="width:${100 * c[2] / n}%;background:${cols[i]}" title="${esc(c[0])}: ${c[2].toLocaleString()}"></span>` : "").join("");
    const list = a => a && a.length ? esc(a.slice(0, 12).join(", ") + (a.length > 12 ? ` +${a.length - 12}` : "")) : "—";
    el.innerHTML = `<h3>${j + 1}. ${esc(p.title)}</h3><div class="comp">${comp}</div><dl class="kv"><dt>Proteins</dt><dd>${n.toLocaleString()}</dd><dt>Samples</dt><dd>${sp.filter((c, i) => i && c[2]).length}</dd><dt>${esc(short(D.leiden_key))}</dt><dd>${lk.toLocaleString()} clusters</dd><dt>Series</dt><dd>${p.keys.length}</dd><dt>Selected</dt><dd>${list(p.selected)}</dd><dt>Probe</dt><dd>${list(p.probe)}</dd></dl>`;
    const openIt = () => { go(j, true); setView(S.last); };
    el.onclick = openIt; el.onkeydown = e => { if (e.key === "Enter") openIt(); };
    return el;
  }));
  $("overview").replaceChildren(head, cards);
}
view("overview", {label: "Overview", nav: true, title: "Every page of the atlas", panel: "ovBox", plots: [], render: overview});
