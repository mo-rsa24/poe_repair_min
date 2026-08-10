"""The inspector's timing tab: drag the window, watch the picture change.

Pre-rendered only, like every other tab. It reads
``window_inspector_manifest.json`` (built by scripts/build_window_manifest.py)
and shows, for one pair and one seed, the image produced when the correction was
injected only inside the chosen window. Dragging the slider moves the window
along the denoising run and swaps the image, the scorer's verdict, and the
marker on the compose-rate curve together.

Nothing here samples or scores. A window with no image on disk keeps the
previous frame and says so, so a half-finished sweep looks half-finished.
"""

from __future__ import annotations

WINDOW_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>correction timing inspector</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --text: #e6e9ef;
    --muted: #8b93a7;
    --accent: #7ab7ff;
    --accent-hot: #f0a458;
    --on:    #2C8F4A;
    --off:   #2a2f3a;
    --border: #2a2f3a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif;
  }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 6px 0; }
  .subtitle { color: var(--muted); font-size: 12px; margin: 0 0 4px 0; max-width: 900px; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 18px; }
  .meta b { color: var(--text); font-weight: 600; }

  .controls {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; margin-bottom: 18px;
  }
  .row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .row + .row { margin-top: 14px; }
  label.k {
    text-transform: uppercase; font-size: 10px; letter-spacing: 0.06em;
    font-weight: 600; color: var(--muted);
  }
  select, button {
    background: #0f1115; color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 5px 10px; font-size: 12px; font-weight: 600;
    font-family: ui-monospace, monospace; cursor: pointer;
  }
  button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  button.hot { border-color: var(--accent-hot); color: var(--accent-hot); }

  input[type=range] {
    -webkit-appearance: none; appearance: none;
    flex: 1; min-width: 260px; height: 4px; border-radius: 2px;
    background: var(--off); outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--accent); cursor: pointer; border: none;
  }
  input[type=range]::-moz-range-thumb {
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--accent); cursor: pointer; border: none;
  }
  .readout {
    font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600;
    color: var(--accent); min-width: 190px;
  }

  /* The 50 denoising steps, with the window drawn on them. */
  .track { display: flex; gap: 1px; margin-top: 4px; }
  .track .step {
    flex: 1; height: 22px; border-radius: 1px; background: var(--off);
    position: relative;
  }
  .track .step.on { background: var(--on); }
  .track .step.fork::after {
    content: ""; position: absolute; inset: 0;
    border-left: 2px dashed var(--text); opacity: 0.75;
  }
  .track-legend {
    display: flex; justify-content: space-between;
    color: var(--muted); font-size: 10px; margin-top: 4px;
    font-family: ui-monospace, monospace;
  }

  .panels {
    display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr);
    gap: 18px; align-items: start;
  }
  @media (max-width: 900px) { .panels { grid-template-columns: minmax(0,1fr); } }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px;
  }
  .card h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); margin: 0 0 10px 0; font-weight: 600;
  }
  .frame { position: relative; }
  .frame img {
    width: 100%; display: block; border-radius: 6px; background: #0a0c10;
    aspect-ratio: 1 / 1; object-fit: cover;
  }
  .frame img.stale { opacity: 0.35; }
  .chip {
    position: absolute; left: 10px; top: 10px;
    padding: 4px 9px; border-radius: 999px; font-size: 11px; font-weight: 700;
    font-family: ui-monospace, monospace; letter-spacing: 0.03em;
    background: rgba(15,17,21,0.86); border: 1px solid var(--border);
  }
  .chip.yes { color: #5fd18a; border-color: #2C8F4A; }
  .chip.no  { color: #f08a8a; border-color: #7a3030; }
  .chip.unscored { color: var(--muted); }
  .caption {
    color: var(--muted); font-size: 11px; margin-top: 8px;
    font-family: ui-monospace, monospace;
  }
  .caption b { color: var(--text); }
  .empty {
    color: var(--muted); font-size: 12px; text-align: center;
    padding: 40px 12px; border: 1px dashed var(--border); border-radius: 6px;
  }
  svg.curve { width: 100%; height: 230px; display: block; }
  .note {
    color: var(--muted); font-size: 11px; margin-top: 10px; line-height: 1.6;
  }
  #toast {
    position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%);
    background: #2a2f3a; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 14px; font-size: 12px;
    opacity: 0; pointer-events: none; transition: opacity 0.18s ease;
  }
  #toast.show { opacity: 1; }
  details.diagnostics {
    margin-top: 24px; padding: 12px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted); font-size: 12px;
  }
  details.diagnostics summary {
    cursor: pointer; user-select: none; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 11px;
  }
  details.diagnostics .body { margin-top: 8px; line-height: 1.7; }
  details.diagnostics code { color: var(--text); }
</style>
</head>
<body>

<h1>When in the run is the correction needed?</h1>
<p class="subtitle">
  Every picture below was made the same way except for one thing: the stretch of
  denoising steps where the correction was allowed to act. Outside that stretch
  nothing is injected. The prompt stays on at every step in every run, so what
  changes across the slider is <em>when</em> the correction acts and nothing else.
</p>
<p class="meta">
  window width <b>{{ width }}</b> steps &middot; stride <b>{{ stride }}</b> &middot;
  <b>{{ n_windows }}</b> positions over <b>{{ num_steps }}</b> steps &middot;
  fork step <b>{{ fork_step }}</b> &middot;
  <b>{{ n_on_disk }}</b> of <b>{{ n_planned }}</b> cells generated,
  <b>{{ n_scored }}</b> scored{% if scorer %} by <b>{{ scorer }}</b>{% endif %}
</p>

{% if n_on_disk == 0 %}
<div class="empty">
  Nothing generated yet. Run
  <code>bash scripts/mechanism_study/run_window_sweep.sh</code>, then
  <code>python scripts/plot_window_curves.py</code>, then
  <code>python scripts/build_window_manifest.py</code>.
</div>
{% else %}

<div class="controls">
  <div class="row">
    <label class="k" for="pair">pair</label>
    <select id="pair"></select>
    <label class="k" for="seed">seed</label>
    <select id="seed"></select>
    <span class="spacer" style="flex:1"></span>
    <button id="pin">pin this one</button>
    <button id="play">play &#9654;</button>
  </div>
  <div class="row">
    <label class="k" for="win">window</label>
    <button id="prev">&#9664;</button>
    <input type="range" id="win" min="0" max="{{ n_windows - 1 }}" value="0" step="1">
    <button id="next">&#9654;</button>
    <span class="readout" id="readout"></span>
  </div>
  <div class="row" style="display:block">
    <div class="track" id="track"></div>
    <div class="track-legend">
      <span>step 0 &middot; noisiest</span>
      <span id="track-note">green = the correction is injected here</span>
      <span>step {{ num_steps - 1 }} &middot; cleanest</span>
    </div>
  </div>
</div>

<div class="panels">
  <div class="card">
    <h2>this window</h2>
    <div class="frame">
      <img id="shot" alt="generated image for the selected window">
      <span class="chip" id="chip"></span>
    </div>
    <div class="caption" id="cap"></div>
  </div>
  <div class="card">
    <h2 id="right-title">compose rate against where the window sits</h2>
    <div id="pinned-wrap" style="display:none">
      <div class="frame">
        <img id="pinned-shot" alt="pinned image">
        <span class="chip" id="pinned-chip"></span>
      </div>
      <div class="caption" id="pinned-cap"></div>
    </div>
    <svg class="curve" id="curve" viewBox="0 0 480 230" preserveAspectRatio="none"></svg>
    <p class="note" id="curve-note"></p>
  </div>
</div>

<details class="diagnostics">
  <summary>what this tab can and cannot tell you</summary>
  <div class="body">
    The curve is a rate over every pair and seed that has been scored, so it is a
    population claim. The picture beside it is one cell, so it illustrates the
    rate and does not prove it: a single cell can go either way at any window.
    Read them together and neither alone.<br><br>
    The dashed line at step <code>{{ fork_step }}</code> is an independent
    estimate of the same moment, measured from cached trajectories rather than
    from these runs (the median step where the broken and working paths pull
    apart, <code>fork_curve.json</code>). Whether the peak lands on it is the
    open question, not a settled result.<br><br>
    Verdicts come from the validated compose scorer counting animal instances;
    two or more counts as composed. The scorer, not the eye, decides.
  </div>
</details>

<div id="toast"></div>

<script>
const M = {{ manifest_json|safe }};
const KEYS = M.window_keys;
const WINDOWS = M.windows;
const NUM_STEPS = M.num_steps;
const FORK = M.fork_step;

const $ = id => document.getElementById(id);
let pinned = null;

function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 1600);
}

function fillSelect(sel, values, current) {
  sel.innerHTML = "";
  for (const v of values) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  }
}

function seedsFor(pair) { return M.seeds_by_pair[pair] || []; }

function cellFor(pair, seed, key) {
  return ((M.cells[pair] || {})[seed] || {})[key] || null;
}

// The 50 steps, with the current window filled in.
function drawTrack(win) {
  const t = $("track");
  if (t.children.length !== NUM_STEPS) {
    t.innerHTML = "";
    for (let i = 0; i < NUM_STEPS; i++) {
      const d = document.createElement("div");
      d.className = "step";
      d.title = "step " + i;
      t.appendChild(d);
    }
  }
  for (let i = 0; i < NUM_STEPS; i++) {
    const el = t.children[i];
    el.className = "step"
      + (i >= win[0] && i < win[1] ? " on" : "")
      + (i === FORK ? " fork" : "");
  }
}

// The compose-rate curve, drawn from the scorer's numbers, with a marker on
// the window currently being viewed.
function drawCurve(centre) {
  const svg = $("curve");
  const c = M.curve || {};
  const xs = c.centres || [], ys = c.compose_rate || [], ns = c.n || [];
  const W = 480, H = 230, L = 42, R = 12, T = 14, B = 30;
  if (!xs.length) {
    svg.innerHTML = '<text x="12" y="30" fill="#8b93a7" font-size="12">'
      + 'nothing scored yet: run scripts/plot_window_curves.py</text>';
    $("curve-note").textContent = "";
    return;
  }
  const px = s => L + (s / (NUM_STEPS - 1)) * (W - L - R);
  const py = r => T + (1 - r) * (H - T - B);
  const parts = [];

  for (let g = 0; g <= 4; g++) {
    const r = g / 4, y = py(r);
    parts.push(`<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="#2a2f3a" stroke-width="1"/>`);
    parts.push(`<text x="${L - 6}" y="${y + 4}" fill="#8b93a7" font-size="10" text-anchor="end">${Math.round(r * 100)}%</text>`);
  }
  for (let s = 0; s < NUM_STEPS; s += 10) {
    parts.push(`<text x="${px(s)}" y="${H - 10}" fill="#8b93a7" font-size="10" text-anchor="middle">${s}</text>`);
  }

  if (c.peak_band && c.peak_band.length === 2) {
    const x0 = px(c.peak_band[0]), x1 = px(c.peak_band[1]);
    parts.push(`<rect x="${x0}" y="${T}" width="${Math.max(x1 - x0, 2)}" height="${H - T - B}" fill="#2C8F4A" opacity="0.16"/>`);
  }
  parts.push(`<line x1="${px(FORK)}" y1="${T}" x2="${px(FORK)}" y2="${H - B}" stroke="#e6e9ef" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>`);

  const pts = xs.map((x, i) => `${px(x)},${py(ys[i])}`).join(" ");
  parts.push(`<polyline points="${pts}" fill="none" stroke="#7ab7ff" stroke-width="2"/>`);
  xs.forEach((x, i) => {
    const se = Math.sqrt(Math.max(ys[i] * (1 - ys[i]), 0) / Math.max(ns[i] || 1, 1));
    parts.push(`<line x1="${px(x)}" y1="${py(Math.max(ys[i] - se, 0))}" x2="${px(x)}" y2="${py(Math.min(ys[i] + se, 1))}" stroke="#7ab7ff" stroke-width="1" opacity="0.6"/>`);
    parts.push(`<circle cx="${px(x)}" cy="${py(ys[i])}" r="3" fill="#7ab7ff"/>`);
  });

  // Where the slider currently is.
  parts.push(`<line x1="${px(centre)}" y1="${T}" x2="${px(centre)}" y2="${H - B}" stroke="#f0a458" stroke-width="2"/>`);
  const i = xs.indexOf(centre);
  if (i >= 0) {
    parts.push(`<circle cx="${px(centre)}" cy="${py(ys[i])}" r="5" fill="#f0a458"/>`);
  }
  svg.innerHTML = parts.join("");

  const note = [];
  if (i >= 0) {
    note.push(`this window: ${Math.round(ys[i] * 100)}% of ${ns[i]} scored cells composed.`);
  } else {
    note.push("this window has no scored cells yet.");
  }
  if (c.peak_centre !== null && c.peak_centre !== undefined) {
    note.push(`Best at centre ${c.peak_centre}; the shaded band is every window within one standard error of it.`);
  }
  $("curve-note").textContent = note.join(" ");
}

function verdict(cell) {
  if (!cell) return ["unscored", "no image"];
  if (!cell.scored) return ["unscored", "not scored yet"];
  return cell.compose
    ? ["yes", `composes · ${cell.n_instances} instances`]
    : ["no", `blended · ${cell.n_instances} instance${cell.n_instances === 1 ? "" : "s"}`];
}

function render() {
  const pair = $("pair").value, seed = $("seed").value;
  const idx = parseInt($("win").value, 10);
  const key = KEYS[idx], win = WINDOWS[idx];
  const centre = (win[0] + win[1]) / 2;

  $("readout").textContent = `steps ${win[0]}–${win[1]} · centre ${centre}`;
  drawTrack(win);
  drawCurve(centre);

  const cell = cellFor(pair, seed, key);
  const img = $("shot"), chip = $("chip");
  if (cell) {
    img.src = "/interaction_window/img?path=" + encodeURIComponent(cell.image);
    img.classList.remove("stale");
  } else {
    img.classList.add("stale");
    toast(`no image yet for ${pair} seed ${seed}, window ${key}`);
  }
  const [cls, text] = verdict(cell);
  chip.className = "chip " + cls;
  chip.textContent = text;
  $("cap").innerHTML = `<b>${pair}</b> · seed ${seed} · correction injected in steps `
    + `<b>${win[0]}–${win[1]}</b>, nothing outside`;
}

function setPair(pair) {
  const seeds = seedsFor(pair);
  const keep = $("seed").value;
  fillSelect($("seed"), seeds, seeds.includes(keep) ? keep : seeds[0]);
}

// Pinning gives the peak-versus-tail comparison the plan asks for: freeze one
// window, then slide to another and read the two side by side.
function pin() {
  const pair = $("pair").value, seed = $("seed").value;
  const idx = parseInt($("win").value, 10);
  const cell = cellFor(pair, seed, KEYS[idx]);
  if (!cell) { toast("nothing to pin: that cell has no image"); return; }
  pinned = { pair, seed, key: KEYS[idx], win: WINDOWS[idx], cell };
  $("pinned-wrap").style.display = "";
  $("pinned-shot").src = "/interaction_window/img?path=" + encodeURIComponent(cell.image);
  const [cls, text] = verdict(cell);
  $("pinned-chip").className = "chip " + cls;
  $("pinned-chip").textContent = text;
  $("pinned-cap").innerHTML = `pinned · <b>${pair}</b> · seed ${seed} · steps `
    + `<b>${pinned.win[0]}–${pinned.win[1]}</b>`;
  $("right-title").textContent = "pinned comparison, and the curve below it";
  $("pin").textContent = "unpin";
  $("pin").classList.add("hot");
}
function unpin() {
  pinned = null;
  $("pinned-wrap").style.display = "none";
  $("right-title").textContent = "compose rate against where the window sits";
  $("pin").textContent = "pin this one";
  $("pin").classList.remove("hot");
}

let timer = null;
function play() {
  if (timer) { stop(); return; }
  $("play").innerHTML = "pause &#10073;&#10073;";
  $("play").classList.add("hot");
  timer = setInterval(() => {
    const el = $("win");
    el.value = (parseInt(el.value, 10) + 1) % KEYS.length;
    render();
  }, 700);
}
function stop() {
  clearInterval(timer); timer = null;
  $("play").innerHTML = "play &#9654;";
  $("play").classList.remove("hot");
}

function step(delta) {
  const el = $("win");
  const v = Math.min(Math.max(parseInt(el.value, 10) + delta, 0), KEYS.length - 1);
  el.value = v; render();
}

fillSelect($("pair"), M.pairs, M.pairs[0]);
setPair($("pair").value);
$("pair").addEventListener("change", () => { setPair($("pair").value); render(); });
$("seed").addEventListener("change", render);
$("win").addEventListener("input", render);
$("prev").addEventListener("click", () => { stop(); step(-1); });
$("next").addEventListener("click", () => { stop(); step(1); });
$("play").addEventListener("click", play);
$("pin").addEventListener("click", () => (pinned ? unpin() : pin()));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  { stop(); step(-1); e.preventDefault(); }
  if (e.key === "ArrowRight") { stop(); step(1);  e.preventDefault(); }
  if (e.key === " ")          { play(); e.preventDefault(); }
});
render();
</script>
{% endif %}
</body>
</html>
"""
