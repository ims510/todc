/* explorer.js — vanilla, no framework.
   Loads data/<resolution>/words.json, renders a sortable/searchable table,
   and shows per-word detail (scores, feature deltas, plots) on click. */

const COLUMNS = [
  { key: "lemma",        label: "lemma",        num: false },
  { key: "pos",          label: "pos",          num: false },
  { key: "n_bins",       label: "bins",         num: true  },
  { key: "displacement", label: "displacement", num: true  },
  { key: "jitter",       label: "jitter",       num: true  },
  { key: "directness",   label: "directness",   num: true  },
  { key: "path_length",  label: "path",         num: true  },
  { key: "max_dist",     label: "max dist",     num: true  },
  { key: "score",        label: "score",        num: true  },
];

const SLOT_NAMES = { parent: "parent", prev: "previous", next: "following", child: "child", self: "self" };

let state = {
  resolution: "annee",
  words: [],
  sortKey: "displacement",
  sortAsc: false,
  search: "",
  pos: "",
  selected: null,
};

async function loadResolution(res) {
  state.resolution = res;
  try {
    const data = await (await fetch(`data/${res}/words.json`)).json();
    state.words = data.words;
  } catch (e) {
    state.words = [];
    document.getElementById("countNote").textContent =
      "No data yet — run build_site.py to generate it.";
  }
  buildPosFilter();
  render();
}

function buildPosFilter() {
  const sel = document.getElementById("posFilter");
  const current = sel.value;
  const tags = Array.from(new Set(state.words.map(w => w.pos))).sort();
  sel.innerHTML = '<option value="">All categories</option>' +
    tags.map(t => `<option value="${t}">${t}</option>`).join("");
  sel.value = current;
}

function buildHead() {
  const head = document.getElementById("headRow");
  head.innerHTML = COLUMNS.map(c => {
    const sorted = c.key === state.sortKey ? "sorted" + (state.sortAsc ? " asc" : "") : "";
    return `<th class="${sorted}" data-key="${c.key}">${c.label}</th>`;
  }).join("");
  head.querySelectorAll("th").forEach(th => {
    th.onclick = () => {
      const key = th.dataset.key;
      if (state.sortKey === key) state.sortAsc = !state.sortAsc;
      else { state.sortKey = key; state.sortAsc = false; }
      render();
    };
  });
}

function filteredRows() {
  let rows = state.words;
  if (state.pos) rows = rows.filter(w => w.pos === state.pos);
  if (state.search) {
    const q = state.search.toLowerCase();
    rows = rows.filter(w => w.lemma.toLowerCase().includes(q));
  }
  const k = state.sortKey, asc = state.sortAsc;
  rows = rows.slice().sort((a, b) => {
    let va = a[k], vb = b[k];
    if (typeof va === "string") { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return asc ? -1 : 1;
    if (va > vb) return asc ? 1 : -1;
    return 0;
  });
  return rows;
}

function render() {
  buildHead();
  const rows = filteredRows();
  const body = document.getElementById("tableBody");
  body.innerHTML = rows.slice(0, 500).map(w => {
    const sel = (state.selected && state.selected.lemma === w.lemma && state.selected.pos === w.pos) ? "selected" : "";
    const cells = COLUMNS.map(c => {
      if (c.key === "lemma") return `<td class="lemma-cell">${w.lemma}${w.detail ? "" : ' <span title="no detailed plots" style="color:var(--muted)">·</span>'}</td>`;
      if (c.key === "pos")   return `<td><span class="pos-chip">${w.pos}</span></td>`;
      return `<td>${w[c.key]}</td>`;
    }).join("");
    return `<tr class="${sel}" data-lemma="${encodeURIComponent(w.lemma)}" data-pos="${w.pos}">${cells}</tr>`;
  }).join("");

  body.querySelectorAll("tr").forEach(tr => {
    tr.onclick = () => {
      const lemma = decodeURIComponent(tr.dataset.lemma);
      const pos = tr.dataset.pos;
      state.selected = state.words.find(w => w.lemma === lemma && w.pos === pos);
      render();
      showDetail(state.selected);
    };
  });

  const shown = Math.min(rows.length, 500);
  document.getElementById("countNote").textContent =
    `${rows.length} words${rows.length > 500 ? ` (showing first ${shown})` : ""}`;
}

function deltaBar(delta) {
  const w = Math.min(Math.abs(delta) * 600, 90);
  const cls = delta >= 0 ? "up" : "down";
  const sign = delta >= 0 ? "+" : "";
  return `<span class="bar ${cls}" style="width:${w}px"></span> ${sign}${delta.toFixed(3)}`;
}

function showDetail(w) {
  const el = document.getElementById("detail");
  if (!w) { el.innerHTML = '<p class="placeholder">Select a word.</p>'; return; }

  const stats = [
    ["displacement", w.displacement], ["jitter", w.jitter], ["directness", w.directness],
    ["path", w.path_length], ["max dist", w.max_dist], ["score", w.score],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  let detailHtml = "";
  if (w.detail) {
    const fd = w.feature_delta.map(f => `
      <tr class="s-${f.slot}">
        <td class="fd-feature">${f.feature}</td>
        <td>${f.first.toFixed(3)}</td>
        <td>${f.last.toFixed(3)}</td>
        <td>${deltaBar(f.delta)}</td>
      </tr>`).join("");

    const plots = w.plots || {};
    const imgs = [];
    if (plots.pca) imgs.push(`<img src="${plots.pca}" alt="PCA cloud of yearly vectors" loading="lazy" onerror="this.style.display='none'">`);
    if (plots.construction) imgs.push(`<img src="${plots.construction}" alt="construction frequency over time" loading="lazy" onerror="this.style.display='none'">`);
    if (plots.usage) imgs.push(`<img src="${plots.usage}" alt="usage over time" loading="lazy" onerror="this.style.display='none'">`);

    detailHtml = `
      <h3>Features that moved most</h3>
      <table class="fd-table"><thead><tr><th>feature</th><th>first</th><th>last</th><th>change</th></tr></thead>
      <tbody>${fd}</tbody></table>
      ${imgs.join("")}`;
  } else {
    detailHtml = `<div class="nodetail">This word appears in too few bins for detailed plots.
      Its movement scores are above, but a small cloud of points isn't reliable enough to map.</div>`;
  }

  el.innerHTML = `
    <h2>${w.lemma} <span class="pos-chip">${w.pos}</span></h2>
    <p class="count-note">present ${w.n_bins} bins · ${w.first}–${w.last}</p>
    <div class="stat-grid">${stats}</div>
    ${detailHtml}`;
}

/* ---- wire up controls ---- */
document.getElementById("search").addEventListener("input", e => { state.search = e.target.value; render(); });
document.getElementById("posFilter").addEventListener("change", e => { state.pos = e.target.value; render(); });
document.querySelectorAll("#resToggle button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("#resToggle button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.selected = null;
    document.getElementById("detail").innerHTML = '<p class="placeholder">Select a word.</p>';
    loadResolution(btn.dataset.res);
  };
});

loadResolution("annee");
