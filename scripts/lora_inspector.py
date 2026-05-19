"""Flask inspector for the LoRA SDXL residual training chain.

Pre-rendered only. Sliders snap to indexed (epoch, lambda) values. When the
selected cell does not exist (some epochs only probed a coarser lambda grid),
a toast appears and the panels keep the previous frame.

Run on the cluster (binds 127.0.0.1 by default, intended for SSH tunnelling):
    python scripts/lora_inspector.py [--port 5050]

Then on your laptop:
    ssh -L 5050:localhost:5050 mscluster106
    open http://localhost:5050
"""
from __future__ import annotations

import argparse
import json
import os.path
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, send_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "outputs/lora/cat_dog/seed_42/results/inspector_manifest.json"
DEFAULT_CW_MANIFEST = REPO_ROOT / "outputs/conditioning_window/cat_dog/seed_42/results/inspector_manifest.json"
DEFAULT_CWL_WP_MANIFEST = REPO_ROOT / "outputs/conditioning_window_lora/cat_dog/seed_42/with_prompt/results/inspector_manifest.json"
DEFAULT_CWL_ALWAYS_MANIFEST = REPO_ROOT / "outputs/conditioning_window_lora/cat_dog/seed_42/always/results/inspector_manifest.json"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LoRA residual inspector</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --text: #e6e9ef;
    --muted: #8b93a7;
    --accent: #7ab7ff;
    --warn: #f0a458;
    --border: #2a2f3a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif;
  }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 16px 0; color: var(--muted); }
  .controls {
    display: grid; gap: 14px; margin-bottom: 20px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .row { display: flex; align-items: center; gap: 16px; }
  .row label { flex: 0 0 80px; color: var(--muted); }
  .row input[type=range] { flex: 1; }
  .row .val {
    flex: 0 0 64px; text-align: right; font-variant-numeric: tabular-nums;
    color: var(--accent); font-weight: 600;
  }
  .row .source { color: var(--muted); font-size: 12px; flex: 0 0 auto; }
  .panels {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
  }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .panel img {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 4px;
    background: #000; display: block;
  }
  .panel .caption {
    margin-top: 8px; color: var(--muted); font-size: 11px;
  }
  #toast {
    position: fixed; top: 24px; right: 24px;
    background: var(--warn); color: #1a1207;
    padding: 10px 14px; border-radius: 6px;
    font-weight: 600; opacity: 0; pointer-events: none;
    transition: opacity 0.18s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<h1>LoRA residual inspector — {{ results_root }}</h1>

<div class="controls">
  <div class="row">
    <label for="epoch">epoch</label>
    <input id="epoch" type="range" min="0" max="{{ epochs|length - 1 }}" step="1" value="0">
    <div class="val" id="epoch-val">0</div>
    <div class="source" id="epoch-src"></div>
  </div>
  <div class="row">
    <label for="lambda">λ</label>
    <input id="lambda" type="range" min="0" max="{{ lambdas|length - 1 }}" step="1" value="{{ lambdas|length - 1 }}">
    <div class="val" id="lambda-val">{{ lambdas[-1] }}</div>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>PoE baseline (λ = 0)</h2>
    <img id="img-poe" alt="PoE baseline">
  </div>
  <div class="panel">
    <h2>PoE + λ · r (current)</h2>
    <img id="img-current" alt="current">
  </div>
</div>

<p style="margin-top:18px;color:var(--muted);font-size:12px;">
  → <a href="/conditioning_window" style="color:var(--accent);">conditioning_window</a>
  (no-LoRA CFG-mask ablation; same seed, same x_T)
  &nbsp;·&nbsp;
  → <a href="/conditioning_window_lora" style="color:var(--accent);">conditioning_window_lora</a>
  (with-LoRA CFG-mask ablation; 3-pane on/off/diff, mode toggle)
</p>

<div id="toast">cell not available</div>

<script>
const MANIFEST = {{ manifest_json|safe }};
const EPOCHS = MANIFEST.epochs;
const LAMBDAS = MANIFEST.lambdas;     // strings like "0.00", "0.50", "1.00"
const CELLS = MANIFEST.cells;          // {"100": {"0.00": "outputs/.../decoded.png", ...}}
const SOURCE = MANIFEST.cell_source_run;

const epochSlider = document.getElementById('epoch');
const epochVal = document.getElementById('epoch-val');
const epochSrc = document.getElementById('epoch-src');
const lambdaSlider = document.getElementById('lambda');
const lambdaVal = document.getElementById('lambda-val');
const imgPoe = document.getElementById('img-poe');
const imgCurrent = document.getElementById('img-current');
const toast = document.getElementById('toast');

let toastTimer = null;
function flashToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
}

function lambdaForIdx(idx) { return LAMBDAS[idx]; }
function epochForIdx(idx) { return EPOCHS[idx]; }

function cellPath(epoch, lam) {
  const e = String(epoch);
  if (!CELLS[e]) return null;
  return CELLS[e][lam] || null;
}

function update() {
  const eIdx = parseInt(epochSlider.value, 10);
  const lIdx = parseInt(lambdaSlider.value, 10);
  const epoch = epochForIdx(eIdx);
  const lam = lambdaForIdx(lIdx);
  epochVal.textContent = epoch;
  lambdaVal.textContent = lam;
  epochSrc.textContent = SOURCE[String(epoch)] || '';

  const poePath = cellPath(epoch, '0.00');
  const curPath = cellPath(epoch, lam);

  if (!curPath) {
    flashToast(`no probe at epoch=${epoch}, λ=${lam}`);
    return;
  }
  if (!poePath) {
    flashToast(`no PoE baseline at epoch=${epoch}`);
    return;
  }
  imgPoe.src = '/img/' + poePath;
  imgCurrent.src = '/img/' + curPath;
}

epochSlider.addEventListener('input', update);
lambdaSlider.addEventListener('input', update);

update();
</script>
</body>
</html>
"""




CW_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>conditioning_window inspector</title>
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
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px 0; color: var(--muted); }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 18px; }

  .controls {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 16px 16px 16px; margin-bottom: 20px;
  }

  /* ---- Dual-handle range slider with side arrows ---- */
  .slider-row {
    display: flex; align-items: center; gap: 10px;
  }
  .arrows {
    display: flex; flex-direction: row; gap: 4px;
    flex: 0 0 auto;
  }
  .arrows.left  { margin-right: 6px; }
  .arrows.right { margin-left: 6px; }
  .arrow-btn {
    width: 28px; height: 28px;
    border-radius: 6px;
    background: var(--off); color: var(--accent);
    border: 1px solid var(--border);
    font-family: ui-monospace, monospace; font-size: 14px; font-weight: 600;
    cursor: pointer; user-select: none;
    display: inline-flex; align-items: center; justify-content: center;
  }
  .arrow-btn:hover:not(:disabled) { background: #232a36; }
  .arrow-btn:disabled {
    opacity: 0.25; cursor: not-allowed;
  }
  .arrow-label {
    font-size: 10px; color: var(--muted);
    margin-bottom: 2px; text-align: center;
    font-family: ui-monospace, monospace; letter-spacing: 0.04em;
  }
  .arrows-stack {
    display: flex; flex-direction: column; align-items: center;
  }

  .dual {
    position: relative; height: 38px; flex: 1 1 auto;
  }
  .dual .track {
    position: absolute; left: 0; right: 0; top: 50%;
    transform: translateY(-50%); height: 4px;
    background: var(--off); border-radius: 2px;
  }
  .dual .highlight {
    position: absolute; top: 50%; transform: translateY(-50%);
    height: 4px; background: var(--on); border-radius: 2px;
    pointer-events: none;
  }
  .dual input[type=range] {
    position: absolute; left: 0; right: 0; top: 0;
    width: 100%; height: 38px;
    background: none; -webkit-appearance: none; appearance: none;
    pointer-events: none;   /* let only the thumb catch events */
    margin: 0;
  }
  .dual input[type=range]::-webkit-slider-runnable-track {
    height: 4px; background: transparent;
  }
  .dual input[type=range]::-moz-range-track {
    height: 4px; background: transparent;
  }
  .dual input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent); border: 2px solid #0f1115;
    cursor: ew-resize; pointer-events: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
    margin-top: -7px;
  }
  .dual input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent); border: 2px solid #0f1115;
    cursor: ew-resize; pointer-events: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
  }
  .dual .endpoints {
    position: absolute; left: 0; right: 0; top: 100%;
    display: flex; justify-content: space-between;
    color: var(--muted); font-size: 10px; font-family: ui-monospace, monospace;
    margin-top: 4px;
  }
  .dual .endpoints span { opacity: 0.5; }

  /* ---- Hover tooltip over the slider track ---- */
  .dual .tooltip {
    position: absolute; bottom: calc(100% + 8px);
    transform: translateX(-50%);
    background: #1f2632; color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 11px;
    font-family: ui-monospace, monospace;
    white-space: nowrap; pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.55);
    opacity: 0; transition: opacity 0.10s ease;
    z-index: 10;
  }
  .dual .tooltip.show { opacity: 1; }
  .dual .tooltip .tt-id { color: var(--accent); font-weight: 600; }
  .dual .tooltip .tt-delta-faster { color: var(--on); }
  .dual .tooltip .tt-delta-slower { color: var(--accent-hot); }
  .dual .tooltip .tt-delta-zero   { color: var(--muted); }
  .dual .tooltip-arrow {
    position: absolute; left: 50%; bottom: -4px;
    transform: translateX(-50%) rotate(45deg);
    width: 8px; height: 8px; background: #1f2632;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }

  /* The time-pill in meta-row */
  .time-pill {
    padding: 1px 8px; border-radius: 8px;
    background: var(--off); color: var(--on);
    font-weight: 700; letter-spacing: 0.02em;
  }

  .meta-row {
    display: flex; gap: 18px; margin-top: 14px;
    font-family: ui-monospace, monospace; font-size: 12px;
    color: var(--muted); flex-wrap: wrap;
  }
  .meta-row b { color: var(--accent); font-weight: 600; }
  .meta-row .mode {
    padding: 1px 8px; border-radius: 8px;
    background: var(--off); color: var(--accent-hot);
    font-weight: 700; letter-spacing: 0.04em;
  }
  .meta-row .schedule-id {
    color: var(--text); font-weight: 600;
  }

  .strip {
    display: grid; grid-template-columns: repeat(50, 1fr);
    gap: 1px; margin-top: 12px; height: 26px;
    border: 1px solid var(--border); border-radius: 4px;
    overflow: hidden;
  }
  .cell { background: var(--off); }
  .cell.on { background: var(--on); }
  .cell.tick { box-shadow: inset 0 -3px 0 0 #555; }

  .legend { color: var(--muted); font-size: 11px; margin-top: 6px; }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                margin-right: 4px; vertical-align: middle; border-radius: 2px; }

  .panels { display: grid; grid-template-columns: 1fr; gap: 16px; max-width: 720px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .panel img {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 4px;
    background: #000; display: block;
  }
  .panel .caption {
    margin-top: 8px; color: var(--muted); font-size: 12px;
    font-family: ui-monospace, monospace; word-break: break-all;
  }
</style>
</head>
<body>
<h1>conditioning_window — {{ prompt }} · seed {{ seed }}</h1>
<div class="meta">
  num_inference_steps = {{ num_steps }} · guidance_scale = {{ guidance }} ·
  {{ schedules|length }} rendered schedule(s)
  · sanity: all_on Δ={{ sanity_on }} ({{ sanity_on_pass }}),
              all_off Δ={{ sanity_off }} ({{ sanity_off_pass }})
  · → <a href="/conditioning_window_lora" style="color:var(--accent);">conditioning_window_lora</a>
</div>

<div class="controls">
  <div class="slider-row">
    <div class="arrows-stack">
      <div class="arrow-label">start</div>
      <div class="arrows left">
        <button class="arrow-btn" id="start-prev" title="previous start endpoint">◀</button>
        <button class="arrow-btn" id="start-next" title="next start endpoint">▶</button>
      </div>
    </div>

    <div class="dual">
      <div class="track"></div>
      <div id="hl" class="highlight"></div>
      <input id="start" type="range" min="0" max="{{ num_steps }}" step="1" value="0">
      <input id="end"   type="range" min="0" max="{{ num_steps }}" step="1" value="{{ num_steps }}">
      <div class="endpoints"><span>0</span><span>{{ num_steps }}</span></div>
      <div id="tt" class="tooltip">
        <span class="tt-id" id="tt-id">—</span>
        &nbsp;·&nbsp;
        <span id="tt-time">—</span>
        &nbsp;·&nbsp;
        <span id="tt-delta">—</span>
        <div class="tooltip-arrow"></div>
      </div>
    </div>

    <div class="arrows-stack">
      <div class="arrow-label">end</div>
      <div class="arrows right">
        <button class="arrow-btn" id="end-prev" title="previous end endpoint">◀</button>
        <button class="arrow-btn" id="end-next" title="next end endpoint">▶</button>
      </div>
    </div>
  </div>

  <div class="meta-row">
    <span><span class="mode" id="mode">all-on</span></span>
    <span>start: <b id="start-val">0</b></span>
    <span>end: <b id="end-val">{{ num_steps }}</b></span>
    <span>num_on: <b id="num-on">{{ num_steps }}</b>/{{ num_steps }}</span>
    <span>render time: <span class="time-pill" id="time-pill">—</span></span>
    <span>schedule: <span class="schedule-id" id="sched-id">—</span></span>
  </div>

  <div id="strip" class="strip" title=""></div>
  <div class="legend">
    <span class="sw" style="background: var(--on);"></span> conditional ON
    &nbsp;&nbsp;
    <span class="sw" style="background: var(--off);"></span> conditional OFF
    &nbsp;&nbsp;
    <span style="color: var(--muted);">
      (drag the two thumbs independently — left handle = window start, right handle = window end.
       Slider snaps to the nearest rendered schedule.)
    </span>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>decoded image</h2>
    <img id="img" alt="decoded">
    <div class="caption" id="caption"></div>
  </div>
</div>

<script>
const MANIFEST  = {{ manifest_json|safe }};
const SCHEDULES = MANIFEST.schedules;
const N         = parseInt("{{ num_steps }}", 10);

// Derive (start, end) for each rendered schedule from its mask string.
// start = index of first '1'; end = index after last '1'. All-off => (0, 0).
function maskBounds(maskStr) {
  const first = maskStr.indexOf('1');
  if (first === -1) return {start: 0, end: 0, num_on: 0};
  let last = maskStr.length - 1;
  while (last >= 0 && maskStr[last] !== '1') last--;
  // num_on may be < (last - first + 1) for punctate masks (sparse '1's),
  // but the snap is by (start, end) endpoints; the rendered mask is the
  // source of truth once snapped.
  let num_on = 0;
  for (const c of maskStr) if (c === '1') num_on++;
  return {start: first, end: last + 1, num_on};
}

const ENTRIES = SCHEDULES.map(s => {
  const b = maskBounds(s.mask);
  return Object.assign({}, s, b);
});

// Sorted unique start / end values across rendered schedules — drives the arrow buttons.
const START_VALUES = Array.from(new Set(ENTRIES.map(e => e.start))).sort((a, b) => a - b);
const END_VALUES   = Array.from(new Set(ENTRIES.map(e => e.end  ))).sort((a, b) => a - b);

// Pre-bucket sanity_all_off as the only entry with (0, 0) bounds.
function nearestSchedule(s, e) {
  let best = null, bestD = Infinity;
  // Special-case the empty window: snap to all-off if it exists.
  if (s === e) {
    for (const ent of ENTRIES) {
      if (ent.start === 0 && ent.end === 0) return ent;   // sanity_all_off
    }
  }
  for (const ent of ENTRIES) {
    // Skip all-off for non-empty windows; otherwise it always wins on suffix-y queries.
    if (ent.end === 0 && s !== e) continue;
    const d = (ent.start - s) * (ent.start - s) + (ent.end - e) * (ent.end - e);
    if (d < bestD) { bestD = d; best = ent; }
  }
  return best;
}

function inferMode(s, e) {
  if (s === 0 && e === N) return 'all-on (full CFG)';
  if (s === e)             return 'all-off (no CFG)';
  if (s === 0)             return 'prefix (early-only)';
  if (e === N)             return 'suffix (late-only)';
  if (e - s <= 3)          return 'pulse (punctate dose)';
  return 'window (mid-trajectory)';
}

function fmtSeconds(elapsed) {
  if (elapsed === null || elapsed === undefined) return '—';
  const t = Number(elapsed);
  if (!isFinite(t) || t < 0) return '—';
  if (t < 60) return t.toFixed(1) + 's';
  const m = Math.floor(t / 60);
  const s = Math.round(t - m * 60);
  return m + 'm ' + String(s).padStart(2, '0') + 's';
}

function fmtDelta(otherT, currentT) {
  if (otherT == null || currentT == null) return {text: '—', cls: 'tt-delta-zero'};
  const d = Number(otherT) - Number(currentT);
  if (Math.abs(d) < 0.05) return {text: '~equal', cls: 'tt-delta-zero'};
  const mag = Math.abs(d).toFixed(1) + 's';
  return d < 0
    ? {text: mag + ' faster', cls: 'tt-delta-faster'}
    : {text: mag + ' slower', cls: 'tt-delta-slower'};
}

function renderStrip(maskStr) {
  const stripEl = document.getElementById('strip');
  stripEl.innerHTML = '';
  for (let i = 0; i < maskStr.length; i++) {
    const c = document.createElement('div');
    c.className = 'cell' + (maskStr[i] === '1' ? ' on' : '')
                         + ((i + 1) % 10 === 0 ? ' tick' : '');
    stripEl.appendChild(c);
  }
  stripEl.title = maskStr;
}

const startEl  = document.getElementById('start');
const endEl    = document.getElementById('end');
const hlEl     = document.getElementById('hl');
const startVal = document.getElementById('start-val');
const endVal   = document.getElementById('end-val');
const numOnEl  = document.getElementById('num-on');
const modeEl   = document.getElementById('mode');
const schedIdEl= document.getElementById('sched-id');
const timePillEl = document.getElementById('time-pill');
const imgEl    = document.getElementById('img');
const capEl    = document.getElementById('caption');

const startPrevBtn = document.getElementById('start-prev');
const startNextBtn = document.getElementById('start-next');
const endPrevBtn   = document.getElementById('end-prev');
const endNextBtn   = document.getElementById('end-next');

const dualEl      = document.querySelector('.dual');
const ttEl        = document.getElementById('tt');
const ttIdEl      = document.getElementById('tt-id');
const ttTimeEl    = document.getElementById('tt-time');
const ttDeltaEl   = document.getElementById('tt-delta');

let currentEntry = null;   // the rendered schedule we're currently snapped to

function update(changed) {
  let s = parseInt(startEl.value, 10);
  let e = parseInt(endEl.value, 10);
  // Enforce s <= e — push the other handle if dragged past.
  if (s > e) {
    if (changed === 'start') { e = s; endEl.value = e; }
    else                     { s = e; startEl.value = s; }
  }

  // Highlight bar between thumbs.
  hlEl.style.left  = (100 * s / N) + '%';
  hlEl.style.width = (100 * (e - s) / N) + '%';

  // Snap to nearest rendered schedule by (start, end) endpoints.
  const ent = nearestSchedule(s, e);
  currentEntry = ent;
  renderStrip(ent.mask);
  imgEl.src = '/img/' + ent.image_path;

  startVal.textContent = s;
  endVal.textContent   = e;
  numOnEl.textContent  = ent.num_on;
  modeEl.textContent   = inferMode(s, e);
  timePillEl.textContent = fmtSeconds(ent.elapsed_s);
  schedIdEl.textContent = ent.id +
      ' (snapped from start=' + s + ', end=' + e + ')';
  let cap = ent.id + '  family=' + ent.family +
            '  num_on=' + ent.num_on + '/' + ent.mask.length;
  if (ent.elapsed_s != null) cap += '  · render=' + fmtSeconds(ent.elapsed_s);
  if (ent.sanity) cap += '  <SANITY>';
  capEl.textContent = cap;

  updateArrowsEnabled();
}

startEl.addEventListener('input', () => update('start'));
endEl.addEventListener('input',   () => update('end'));

// ---- Arrow buttons: jump to next/prev rendered endpoint ----

function nextValueAbove(arr, v) {
  // smallest value in arr strictly greater than v; null if none
  for (const x of arr) if (x > v) return x;
  return null;
}
function nextValueBelow(arr, v) {
  // largest value in arr strictly less than v; null if none
  let r = null;
  for (const x of arr) {
    if (x < v) r = x;
    else break;
  }
  return r;
}

function bumpStart(dir) {
  const cur = parseInt(startEl.value, 10);
  const endV = parseInt(endEl.value, 10);
  let target = dir > 0 ? nextValueAbove(START_VALUES, cur)
                       : nextValueBelow(START_VALUES, cur);
  if (target === null) return;
  // Don't allow start to cross end.
  if (target > endV) target = endV;
  startEl.value = target;
  update('start');
}

function bumpEnd(dir) {
  const cur = parseInt(endEl.value, 10);
  const startV = parseInt(startEl.value, 10);
  let target = dir > 0 ? nextValueAbove(END_VALUES, cur)
                       : nextValueBelow(END_VALUES, cur);
  if (target === null) return;
  if (target < startV) target = startV;
  endEl.value = target;
  update('end');
}

function updateArrowsEnabled() {
  const s = parseInt(startEl.value, 10);
  const e = parseInt(endEl.value, 10);
  startPrevBtn.disabled = (nextValueBelow(START_VALUES, s) === null);
  startNextBtn.disabled = (nextValueAbove(START_VALUES, s) === null) ||
                          (nextValueAbove(START_VALUES, s) > e);
  endPrevBtn.disabled   = (nextValueBelow(END_VALUES,   e) === null) ||
                          (nextValueBelow(END_VALUES,   e) < s);
  endNextBtn.disabled   = (nextValueAbove(END_VALUES,   e) === null);
}

startPrevBtn.addEventListener('click', () => bumpStart(-1));
startNextBtn.addEventListener('click', () => bumpStart(+1));
endPrevBtn  .addEventListener('click', () => bumpEnd  (-1));
endNextBtn  .addEventListener('click', () => bumpEnd  (+1));

// ---- Hover tooltip on slider track ----

dualEl.addEventListener('mousemove', (ev) => {
  const rect = dualEl.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const frac = Math.max(0, Math.min(1, x / rect.width));
  const stepPos = Math.round(frac * N);

  // Hypothetical (start, end): move the closer of the two current thumbs to this position.
  const s = parseInt(startEl.value, 10);
  const e = parseInt(endEl.value, 10);
  let hypS = s, hypE = e;
  if (Math.abs(stepPos - s) <= Math.abs(stepPos - e)) {
    hypS = stepPos;
    if (hypS > hypE) hypE = hypS;
  } else {
    hypE = stepPos;
    if (hypE < hypS) hypS = hypE;
  }

  const ent = nearestSchedule(hypS, hypE);
  ttIdEl.textContent   = ent.id;
  ttTimeEl.textContent = fmtSeconds(ent.elapsed_s);
  const delta = fmtDelta(ent.elapsed_s, currentEntry ? currentEntry.elapsed_s : null);
  ttDeltaEl.textContent = delta.text;
  ttDeltaEl.className = delta.cls;

  ttEl.style.left = (100 * frac) + '%';
  ttEl.classList.add('show');
});
dualEl.addEventListener('mouseleave', () => ttEl.classList.remove('show'));

// Initial render: all-on.
update('init');
</script>
</body>
</html>
"""


CWL_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>conditioning_window_lora inspector</title>
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
    --heat:  #d8584f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif;
  }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px 0; color: var(--muted); }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 18px; }

  .controls {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 16px 16px 16px; margin-bottom: 20px;
  }

  .slider-row { display: flex; align-items: center; gap: 10px; }
  .arrows { display: flex; flex-direction: row; gap: 4px; flex: 0 0 auto; }
  .arrows.left  { margin-right: 6px; }
  .arrows.right { margin-left: 6px; }
  .arrow-btn {
    width: 28px; height: 28px; border-radius: 6px;
    background: var(--off); color: var(--accent);
    border: 1px solid var(--border);
    font-family: ui-monospace, monospace; font-size: 14px; font-weight: 600;
    cursor: pointer; user-select: none;
    display: inline-flex; align-items: center; justify-content: center;
  }
  .arrow-btn:hover:not(:disabled) { background: #232a36; }
  .arrow-btn:disabled { opacity: 0.25; cursor: not-allowed; }
  .arrow-label {
    font-size: 10px; color: var(--muted);
    margin-bottom: 2px; text-align: center;
    font-family: ui-monospace, monospace; letter-spacing: 0.04em;
  }
  .arrows-stack { display: flex; flex-direction: column; align-items: center; }

  .dual { position: relative; height: 38px; flex: 1 1 auto; }
  .dual .track {
    position: absolute; left: 0; right: 0; top: 50%;
    transform: translateY(-50%); height: 4px;
    background: var(--off); border-radius: 2px;
  }
  .dual .highlight {
    position: absolute; top: 50%; transform: translateY(-50%);
    height: 4px; background: var(--on); border-radius: 2px;
    pointer-events: none;
  }
  .dual input[type=range] {
    position: absolute; left: 0; right: 0; top: 0;
    width: 100%; height: 38px;
    background: none; -webkit-appearance: none; appearance: none;
    pointer-events: none; margin: 0;
  }
  .dual input[type=range]::-webkit-slider-runnable-track {
    height: 4px; background: transparent;
  }
  .dual input[type=range]::-moz-range-track {
    height: 4px; background: transparent;
  }
  .dual input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent); border: 2px solid #0f1115;
    cursor: ew-resize; pointer-events: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
    margin-top: -7px;
  }
  .dual input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent); border: 2px solid #0f1115;
    cursor: ew-resize; pointer-events: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
  }
  .dual .endpoints {
    position: absolute; left: 0; right: 0; top: 100%;
    display: flex; justify-content: space-between;
    color: var(--muted); font-size: 10px; font-family: ui-monospace, monospace;
    margin-top: 4px;
  }
  .dual .endpoints span { opacity: 0.5; }

  .meta-row {
    display: flex; gap: 18px; margin-top: 14px;
    font-family: ui-monospace, monospace; font-size: 12px;
    color: var(--muted); flex-wrap: wrap; align-items: center;
  }
  .meta-row b { color: var(--accent); font-weight: 600; }
  .meta-row .mode {
    padding: 1px 8px; border-radius: 8px;
    background: var(--off); color: var(--accent-hot);
    font-weight: 700; letter-spacing: 0.04em;
  }
  .meta-row .schedule-id { color: var(--text); font-weight: 600; }
  .meta-row .time-pill {
    padding: 1px 8px; border-radius: 8px;
    background: var(--off); color: var(--on);
    font-weight: 700; letter-spacing: 0.02em;
  }

  .toggle-group { display: inline-flex; gap: 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .toggle-btn {
    padding: 6px 12px; background: var(--off); color: var(--muted);
    border: none; cursor: pointer; font-family: ui-monospace, monospace;
    font-size: 12px; font-weight: 600;
  }
  .toggle-btn.active { background: var(--accent); color: #0f1115; }
  .toggle-btn:not(.active):hover { background: #232a36; color: var(--text); }
  .toggle-label { color: var(--muted); margin-right: 8px; font-size: 12px; }

  .strip-label { color: var(--muted); font-size: 11px; margin-top: 12px; margin-bottom: 2px;
                 font-family: ui-monospace, monospace; }
  .strip {
    display: grid; grid-template-columns: repeat(50, 1fr);
    gap: 1px; height: 22px;
    border: 1px solid var(--border); border-radius: 4px;
    overflow: hidden;
  }
  .cell { background: var(--off); }
  .cell.on { background: var(--on); }
  .cell.tick { box-shadow: inset 0 -3px 0 0 #555; }
  .cell.heat { background: var(--heat); }

  .legend { color: var(--muted); font-size: 11px; margin-top: 8px; }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                margin-right: 4px; vertical-align: middle; border-radius: 2px; }

  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .panel img, .panel canvas {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 4px;
    background: #000; display: block;
  }
  .panel .caption {
    margin-top: 8px; color: var(--muted); font-size: 11px;
    font-family: ui-monospace, monospace; word-break: break-all;
  }
</style>
</head>
<body>
<h1>conditioning_window_lora — {{ prompt }} · seed {{ seed }}</h1>
<div class="meta">
  num_inference_steps = {{ num_steps }} · guidance_scale = {{ guidance }} ·
  with_prompt cells = {{ n_wp }}, always cells = {{ n_al }}, no_lora schedules = {{ n_off }}
  · sanity (with_prompt, all_on vs run_lora_residual_inject): Δ={{ sanity_delta }} ({{ sanity_pass }})
</div>

<div class="controls">
  <div class="slider-row">
    <div class="arrows-stack">
      <div class="arrow-label">start</div>
      <div class="arrows left">
        <button class="arrow-btn" id="start-prev">◀</button>
        <button class="arrow-btn" id="start-next">▶</button>
      </div>
    </div>
    <div class="dual">
      <div class="track"></div>
      <div id="hl" class="highlight"></div>
      <input id="start" type="range" min="0" max="{{ num_steps }}" step="1" value="0">
      <input id="end"   type="range" min="0" max="{{ num_steps }}" step="1" value="{{ num_steps }}">
      <div class="endpoints"><span>0</span><span>{{ num_steps }}</span></div>
    </div>
    <div class="arrows-stack">
      <div class="arrow-label">end</div>
      <div class="arrows right">
        <button class="arrow-btn" id="end-prev">◀</button>
        <button class="arrow-btn" id="end-next">▶</button>
      </div>
    </div>
  </div>

  <div class="meta-row">
    <span class="toggle-label">LoRA mode:</span>
    <div class="toggle-group">
      <button class="toggle-btn active" data-mode="with_prompt">with_prompt</button>
      <button class="toggle-btn" data-mode="always">always</button>
    </div>
    <span><span class="mode" id="mode-disp">window</span></span>
    <span>start: <b id="start-val">0</b></span>
    <span>end: <b id="end-val">{{ num_steps }}</b></span>
    <span>num_on: <b id="num-on">{{ num_steps }}</b>/{{ num_steps }}</span>
    <span>schedule: <span class="schedule-id" id="sched-id">—</span></span>
    <span>render time: <span class="time-pill" id="time-pill">—</span></span>
  </div>

  <div class="meta-row" style="margin-top:10px;">
    <span class="toggle-label">LoRA epoch (step):</span>
    <input id="epoch-slider" type="range" min="0" max="0" step="1" value="0" style="flex:1;max-width:480px;">
    <span><b id="epoch-val">—</b></span>
    <span class="toggle-label" style="margin-left:18px;">λ:</span>
    <input id="lambda-slider" type="range" min="0" max="0" step="1" value="0" style="flex:1;max-width:240px;">
    <span><b id="lambda-val">—</b></span>
  </div>

  <div class="strip-label">CFG mask (green = prompt-on, grey = off)</div>
  <div id="mask-strip" class="strip"></div>
  <div class="strip-label">LoRA per-step δ-eps norm (heat intensity ∝ ‖Δ̂_t‖, scaled within this schedule)</div>
  <div id="delta-strip" class="strip"></div>

  <div class="legend">
    <span class="sw" style="background: var(--on);"></span> CFG on
    &nbsp;&nbsp;
    <span class="sw" style="background: var(--off);"></span> CFG off
    &nbsp;&nbsp;
    <span class="sw" style="background: var(--heat);"></span> LoRA pushing
    &nbsp;&nbsp;
    <span style="color: var(--muted);">
      Drag the two thumbs to pick a schedule; toggle changes which LoRA mode populates the right pane.
    </span>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>LoRA off (baseline)</h2>
    <img id="img-off" alt="lora off">
    <div class="caption" id="cap-off">—</div>
  </div>
  <div class="panel">
    <h2>LoRA on (<span id="cap-mode">with_prompt</span>)</h2>
    <img id="img-on" alt="lora on">
    <div class="caption" id="cap-on">—</div>
  </div>
</div>

<script>
const MANIFEST_OFF = {{ manifest_off_json|safe }};
const MANIFEST_WP  = {{ manifest_wp_json|safe }};
const MANIFEST_AL  = {{ manifest_al_json|safe }};
const N            = parseInt("{{ num_steps }}", 10);

function maskBounds(maskStr) {
  const first = maskStr.indexOf('1');
  if (first === -1) return {start: 0, end: 0, num_on: 0};
  let last = maskStr.length - 1;
  while (last >= 0 && maskStr[last] !== '1') last--;
  let num_on = 0;
  for (const c of maskStr) if (c === '1') num_on++;
  return {start: first, end: last + 1, num_on};
}

// 3D LoRA manifests: cells[<epoch>][<lambda_tag>] = {schedules: [...], ...}
// Flat baseline: MANIFEST_OFF.schedules = [...]
function indexByIdFlat(manifest) {
  const out = {};
  for (const s of manifest.schedules) out[s.id] = s;
  return out;
}
const BY_ID_OFF = indexByIdFlat(MANIFEST_OFF);

// Returns the schedules array at (mode, epoch, lambdaTag) or [] if missing.
function cellSchedules(mode, epoch, lamTag) {
  const M = (mode === 'always') ? MANIFEST_AL : MANIFEST_WP;
  const cells = (M && M.cells) || {};
  const row = cells[String(epoch)] || {};
  const cell = row[lamTag] || null;
  return cell ? cell.schedules : [];
}
function indexByIdCell(mode, epoch, lamTag) {
  const out = {};
  for (const s of cellSchedules(mode, epoch, lamTag)) out[s.id] = s;
  return out;
}

// Sorted union of epochs / lambdas available per LoRA mode.
function availableEpochs(mode) {
  const M = (mode === 'always') ? MANIFEST_AL : MANIFEST_WP;
  return (M.epochs || []).slice().sort((a, b) => a - b);
}
function availableLambdas(mode, epoch) {
  const M = (mode === 'always') ? MANIFEST_AL : MANIFEST_WP;
  const row = (M.cells || {})[String(epoch)] || {};
  return Object.keys(row).slice().sort();   // lambda tags like "0.00", "1.00"
}
function nearestEpoch(mode, target) {
  const arr = availableEpochs(mode);
  if (!arr.length) return null;
  let best = arr[0], bestD = Math.abs(arr[0] - target);
  for (const v of arr) {
    const d = Math.abs(v - target);
    if (d < bestD) { bestD = d; best = v; }
  }
  return best;
}
function nearestLambda(mode, epoch, targetTag) {
  const arr = availableLambdas(mode, epoch);
  if (!arr.length) return null;
  const t = parseFloat(targetTag);
  let best = arr[0], bestD = Math.abs(parseFloat(arr[0]) - t);
  for (const v of arr) {
    const d = Math.abs(parseFloat(v) - t);
    if (d < bestD) { bestD = d; best = v; }
  }
  return best;
}

// Schedule snapping table: use whatever schedules exist at the current
// (mode, epoch, λ). Rebuilt whenever the cell changes.
let ENTRIES = [];
let START_VALUES = [];
let END_VALUES = [];
function rebuildEntries(mode, epoch, lamTag) {
  ENTRIES = cellSchedules(mode, epoch, lamTag).map(s => {
    const b = maskBounds(s.mask);
    return Object.assign({}, s, b);
  });
  START_VALUES = Array.from(new Set(ENTRIES.map(e => e.start))).sort((a, b) => a - b);
  END_VALUES   = Array.from(new Set(ENTRIES.map(e => e.end  ))).sort((a, b) => a - b);
}

function nearestSchedule(s, e) {
  let best = null, bestD = Infinity;
  if (s === e) {
    for (const ent of ENTRIES) {
      if (ent.start === 0 && ent.end === 0) return ent;
    }
  }
  for (const ent of ENTRIES) {
    if (ent.end === 0 && s !== e) continue;
    const d = (ent.start - s) * (ent.start - s) + (ent.end - e) * (ent.end - e);
    if (d < bestD) { bestD = d; best = ent; }
  }
  return best;
}

function inferMode(s, e) {
  if (s === 0 && e === N) return 'all-on (full CFG)';
  if (s === e)             return 'all-off (no CFG)';
  if (s === 0)             return 'prefix (early-only)';
  if (e === N)             return 'suffix (late-only)';
  if (e - s <= 3)          return 'pulse (punctate dose)';
  return 'window (mid-trajectory)';
}

function fmtSeconds(t) {
  if (t == null) return '—';
  const n = Number(t);
  if (!isFinite(n) || n < 0) return '—';
  return n < 60 ? n.toFixed(1) + 's'
       : Math.floor(n/60) + 'm ' + String(Math.round(n - Math.floor(n/60)*60)).padStart(2,'0') + 's';
}

function renderMaskStrip(maskStr) {
  const el = document.getElementById('mask-strip');
  el.innerHTML = '';
  for (let i = 0; i < maskStr.length; i++) {
    const c = document.createElement('div');
    c.className = 'cell' + (maskStr[i] === '1' ? ' on' : '')
                         + ((i + 1) % 10 === 0 ? ' tick' : '');
    el.appendChild(c);
  }
  el.title = maskStr;
}

function renderDeltaStrip(deltas) {
  const el = document.getElementById('delta-strip');
  el.innerHTML = '';
  const maxD = deltas.reduce((m, v) => v > m ? v : m, 0) || 1.0;
  for (let i = 0; i < deltas.length; i++) {
    const c = document.createElement('div');
    const frac = Math.max(0, Math.min(1, deltas[i] / maxD));
    c.className = 'cell' + ((i + 1) % 10 === 0 ? ' tick' : '');
    // Heat-map: blend off→heat by frac. Render as background-color directly.
    if (frac > 0.0) {
      const r = Math.round(0x2a + (0xd8 - 0x2a) * frac);
      const g = Math.round(0x2f + (0x58 - 0x2f) * frac);
      const b = Math.round(0x3a + (0x4f - 0x3a) * frac);
      c.style.background = `rgb(${r},${g},${b})`;
    }
    el.appendChild(c);
  }
  el.title = 'δ-norms (max in this schedule = ' + maxD.toFixed(3) + ')';
}

let currentEntry = null;
let currentMode = 'with_prompt';
let currentEpoch = null;       // numeric, e.g. 62500
let currentLambdaTag = null;   // string, e.g. "1.00"

const imgOff = document.getElementById('img-off');
const imgOn  = document.getElementById('img-on');

const startEl  = document.getElementById('start');
const endEl    = document.getElementById('end');
const hlEl     = document.getElementById('hl');
const startVal = document.getElementById('start-val');
const endVal   = document.getElementById('end-val');
const numOnEl  = document.getElementById('num-on');
const modeDispEl = document.getElementById('mode-disp');
const schedIdEl  = document.getElementById('sched-id');
const timePillEl = document.getElementById('time-pill');
const capOffEl   = document.getElementById('cap-off');
const capOnEl    = document.getElementById('cap-on');
const capModeEl  = document.getElementById('cap-mode');

function pickOnEntry(id) {
  return indexByIdCell(currentMode, currentEpoch, currentLambdaTag)[id] || null;
}
function pickOffEntry(id) { return BY_ID_OFF[id] || null; }

function update(changed) {
  let s = parseInt(startEl.value, 10);
  let e = parseInt(endEl.value, 10);
  if (s > e) {
    if (changed === 'start') { e = s; endEl.value = e; }
    else                     { s = e; startEl.value = s; }
  }
  hlEl.style.left  = (100 * s / N) + '%';
  hlEl.style.width = (100 * (e - s) / N) + '%';

  const ent = nearestSchedule(s, e);
  currentEntry = ent;
  renderMaskStrip(ent.mask);

  const offEnt = pickOffEntry(ent.id);
  const onEnt  = pickOnEntry(ent.id);
  if (onEnt && onEnt.delta_norm_per_step) {
    renderDeltaStrip(onEnt.delta_norm_per_step);
  } else {
    renderDeltaStrip(new Array(N).fill(0));
  }

  imgOff.src = offEnt ? ('/img/' + offEnt.image_path) : '';
  imgOn .src = onEnt  ? ('/img/' + onEnt.image_path)  : '';

  startVal.textContent = s;
  endVal.textContent   = e;
  numOnEl.textContent  = ent.num_on;
  modeDispEl.textContent = inferMode(s, e);
  schedIdEl.textContent  = ent.id;
  timePillEl.textContent = onEnt ? fmtSeconds(onEnt.elapsed_s) : '—';
  capOffEl.textContent = offEnt ? offEnt.image_path : '(no LoRA-off image for ' + ent.id + ')';
  capOnEl .textContent = onEnt  ? onEnt.image_path  : '(no LoRA-on image for '  + ent.id + ' in ' + currentMode + ')';
  capModeEl.textContent = currentMode;
  updateArrowsEnabled();
}

startEl.addEventListener('input', () => update('start'));
endEl  .addEventListener('input', () => update('end'));

function nextValueAbove(arr, v) { for (const x of arr) if (x > v) return x; return null; }
function nextValueBelow(arr, v) {
  let r = null; for (const x of arr) { if (x < v) r = x; else break; } return r;
}
function bumpStart(dir) {
  const cur = parseInt(startEl.value, 10);
  const endV = parseInt(endEl.value, 10);
  let target = dir > 0 ? nextValueAbove(START_VALUES, cur) : nextValueBelow(START_VALUES, cur);
  if (target === null) return;
  if (target > endV) target = endV;
  startEl.value = target;
  update('start');
}
function bumpEnd(dir) {
  const cur = parseInt(endEl.value, 10);
  const startV = parseInt(startEl.value, 10);
  let target = dir > 0 ? nextValueAbove(END_VALUES, cur) : nextValueBelow(END_VALUES, cur);
  if (target === null) return;
  if (target < startV) target = startV;
  endEl.value = target;
  update('end');
}
function updateArrowsEnabled() {
  const s = parseInt(startEl.value, 10);
  const e = parseInt(endEl.value, 10);
  document.getElementById('start-prev').disabled = (nextValueBelow(START_VALUES, s) === null);
  document.getElementById('start-next').disabled = (nextValueAbove(START_VALUES, s) === null) || (nextValueAbove(START_VALUES, s) > e);
  document.getElementById('end-prev').disabled   = (nextValueBelow(END_VALUES,   e) === null) || (nextValueBelow(END_VALUES,   e) < s);
  document.getElementById('end-next').disabled   = (nextValueAbove(END_VALUES,   e) === null);
}
document.getElementById('start-prev').addEventListener('click', () => bumpStart(-1));
document.getElementById('start-next').addEventListener('click', () => bumpStart(+1));
document.getElementById('end-prev')  .addEventListener('click', () => bumpEnd  (-1));
document.getElementById('end-next')  .addEventListener('click', () => bumpEnd  (+1));

// ---- Epoch / lambda sliders (index-snap into available cells) ----

const epochSlider  = document.getElementById('epoch-slider');
const lambdaSlider = document.getElementById('lambda-slider');
const epochValEl   = document.getElementById('epoch-val');
const lambdaValEl  = document.getElementById('lambda-val');

function refreshEpochSlider() {
  const epochs = availableEpochs(currentMode);
  epochSlider.min = 0;
  epochSlider.max = Math.max(0, epochs.length - 1);
  let idx = epochs.indexOf(currentEpoch);
  if (idx < 0) {
    const near = nearestEpoch(currentMode, currentEpoch ?? 0);
    idx = (near != null) ? epochs.indexOf(near) : 0;
    currentEpoch = epochs[idx] ?? null;
  }
  epochSlider.value = idx;
  epochValEl.textContent = (currentEpoch != null) ? currentEpoch : '—';
  epochSlider.disabled = epochs.length <= 1;
}

function refreshLambdaSlider() {
  const lams = availableLambdas(currentMode, currentEpoch);
  lambdaSlider.min = 0;
  lambdaSlider.max = Math.max(0, lams.length - 1);
  let idx = lams.indexOf(currentLambdaTag);
  if (idx < 0) {
    const near = nearestLambda(currentMode, currentEpoch, currentLambdaTag ?? '1.00');
    idx = (near != null) ? lams.indexOf(near) : 0;
    currentLambdaTag = lams[idx] ?? null;
  }
  lambdaSlider.value = idx;
  lambdaValEl.textContent = (currentLambdaTag != null) ? `λ=${currentLambdaTag}` : '—';
  lambdaSlider.disabled = lams.length <= 1;
}

function onCellChanged() {
  rebuildEntries(currentMode, currentEpoch, currentLambdaTag);
  // Re-clamp the dual-handle slider against the new schedule set.
  if (ENTRIES.length === 0) {
    // No data — clear images, keep UI sane.
    document.getElementById('img-off').removeAttribute('src');
    document.getElementById('img-on') .removeAttribute('src');
    document.getElementById('mask-strip').innerHTML = '';
    document.getElementById('delta-strip').innerHTML = '';
    document.getElementById('sched-id').textContent = '(no data at this cell)';
    return;
  }
  update('cell');
}

epochSlider.addEventListener('input', () => {
  const epochs = availableEpochs(currentMode);
  const idx = Math.min(epochs.length - 1, Math.max(0, parseInt(epochSlider.value, 10)));
  currentEpoch = epochs[idx];
  epochValEl.textContent = currentEpoch;
  refreshLambdaSlider();
  onCellChanged();
});
lambdaSlider.addEventListener('input', () => {
  const lams = availableLambdas(currentMode, currentEpoch);
  const idx = Math.min(lams.length - 1, Math.max(0, parseInt(lambdaSlider.value, 10)));
  currentLambdaTag = lams[idx];
  lambdaValEl.textContent = `λ=${currentLambdaTag}`;
  onCellChanged();
});

document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    refreshEpochSlider();
    refreshLambdaSlider();
    onCellChanged();
  });
});

// Initial state: first available (epoch, λ) in with_prompt.
(function initCell() {
  const epochs = availableEpochs('with_prompt');
  currentEpoch = epochs[epochs.length - 1] ?? null;   // default to last epoch
  const lams = (currentEpoch != null) ? availableLambdas('with_prompt', currentEpoch) : [];
  currentLambdaTag = lams[lams.length - 1] ?? null;   // default to highest λ
  refreshEpochSlider();
  refreshLambdaSlider();
  rebuildEntries(currentMode, currentEpoch, currentLambdaTag);
})();

update('init');
</script>
</body>
</html>
"""


_CW_MISSING_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>conditioning_window manifest missing</title>
<style>body{font:14px -apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e9ef;padding:24px;}
code{background:#161a22;padding:2px 6px;border-radius:4px;color:#7ab7ff;}</style>
</head><body>
<h2>conditioning_window manifest not found</h2>
<p>Expected at:</p>
<p><code>{{ path }}</code></p>
<p>Build it by running:</p>
<p><code>python -m poe_repair.experiments.conditioning_window</code></p>
</body></html>
"""


_CWL_MISSING_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>conditioning_window_lora manifest missing</title>
<style>body{font:14px -apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e9ef;padding:24px;}
code{background:#161a22;padding:2px 6px;border-radius:4px;color:#7ab7ff;}</style>
</head><body>
<h2>conditioning_window_lora manifests not found</h2>
<p>Missing: <code>{{ missing|safe }}</code></p>
<p>Also requires the no-LoRA baseline at <code>{{ off_path }}</code> (rendered by <code>python -m poe_repair.experiments.conditioning_window</code>).</p>
<p>Build the LoRA sweeps with:</p>
<p><code>python -m poe_repair.experiments.conditioning_window_lora</code></p>
</body></html>
"""


def create_app(
    manifest_path: Path,
    outputs_root: Path,
    cw_manifest_path: Path | None = None,
    cwl_wp_manifest_path: Path | None = None,
    cwl_always_manifest_path: Path | None = None,
) -> Flask:
    manifest = json.loads(manifest_path.read_text())
    outputs_root_abs = outputs_root.resolve()

    cw_manifest: dict | None = None
    if cw_manifest_path is not None and Path(cw_manifest_path).is_file():
        cw_manifest = json.loads(Path(cw_manifest_path).read_text())

    cwl_wp: dict | None = None
    if cwl_wp_manifest_path is not None and Path(cwl_wp_manifest_path).is_file():
        cwl_wp = json.loads(Path(cwl_wp_manifest_path).read_text())
    cwl_al: dict | None = None
    if cwl_always_manifest_path is not None and Path(cwl_always_manifest_path).is_file():
        cwl_al = json.loads(Path(cwl_always_manifest_path).read_text())

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            results_root=manifest.get("results_root", ""),
            epochs=manifest["epochs"],
            lambdas=manifest["lambdas"],
            manifest_json=json.dumps(manifest),
        )

    @app.route("/manifest.json")
    def manifest_route():
        return jsonify(manifest)

    @app.route("/conditioning_window")
    def conditioning_window():
        if cw_manifest is None:
            return render_template_string(
                _CW_MISSING_HTML,
                path=str(cw_manifest_path) if cw_manifest_path else "(no path)",
            ), 404
        sanity = cw_manifest.get("sanity") or {}
        on_rec = sanity.get("all_on_vs_run_cfg") or {}
        off_rec = sanity.get("all_off_vs_uncond") or {}
        return render_template_string(
            CW_INDEX_HTML,
            prompt=cw_manifest.get("prompt", ""),
            seed=cw_manifest.get("seed", ""),
            num_steps=cw_manifest.get("num_inference_steps", 50),
            guidance=cw_manifest.get("guidance_scale", ""),
            schedules=cw_manifest["schedules"],
            sanity_on=f"{on_rec.get('max_abs_delta', float('nan')):.2e}"
                      if on_rec else "—",
            sanity_off=f"{off_rec.get('max_abs_delta', float('nan')):.2e}"
                       if off_rec else "—",
            sanity_on_pass="PASS" if on_rec.get("pass") else "FAIL"
                           if on_rec else "—",
            sanity_off_pass="PASS" if off_rec.get("pass") else "FAIL"
                            if off_rec else "—",
            manifest_json=json.dumps(cw_manifest),
        )

    @app.route("/conditioning_window/manifest.json")
    def conditioning_window_manifest():
        if cw_manifest is None:
            abort(404)
        return jsonify(cw_manifest)

    @app.route("/conditioning_window_lora")
    def conditioning_window_lora():
        missing = []
        if cw_manifest is None:
            missing.append(str(cw_manifest_path) if cw_manifest_path else "(no path)")
        if cwl_wp is None:
            missing.append(str(cwl_wp_manifest_path) if cwl_wp_manifest_path else "(no path)")
        if cwl_al is None:
            missing.append(str(cwl_always_manifest_path) if cwl_always_manifest_path else "(no path)")
        if missing:
            return render_template_string(
                _CWL_MISSING_HTML,
                missing="<br>".join(missing),
                off_path=str(cw_manifest_path) if cw_manifest_path else "(no path)",
            ), 404

        # Sanity stats (lifted from the with_prompt manifest).
        sanity = cwl_wp.get("sanity") or {}
        rec = sanity.get("all_on_with_prompt_vs_lora_inject") or {}

        # Per-mode cell counts (cells = epoch × λ pairs that have data).
        def _count_cells(m: dict) -> int:
            return sum(len(row) for row in (m.get("cells") or {}).values())
        return render_template_string(
            CWL_INDEX_HTML,
            prompt=cwl_wp.get("prompt", ""),
            seed=cwl_wp.get("seed", ""),
            num_steps=cwl_wp.get("num_inference_steps", 50),
            guidance=cwl_wp.get("guidance_scale", ""),
            n_wp=_count_cells(cwl_wp),
            n_al=_count_cells(cwl_al),
            n_off=len(cw_manifest.get("schedules", [])),
            sanity_delta=f"{rec.get('max_abs_delta', float('nan')):.2e}" if rec else "—",
            sanity_pass="PASS" if rec.get("pass") else ("FAIL" if rec else "—"),
            manifest_off_json=json.dumps(cw_manifest),
            manifest_wp_json=json.dumps(cwl_wp),
            manifest_al_json=json.dumps(cwl_al),
        )

    @app.route("/conditioning_window_lora/manifest.json")
    def conditioning_window_lora_manifest():
        if cwl_wp is None or cwl_al is None or cw_manifest is None:
            abort(404)
        return jsonify({
            "off": cw_manifest,
            "with_prompt": cwl_wp,
            "always": cwl_al,
        })

    @app.route("/img/<path:rel>")
    def img(rel: str):
        # rel is relative to REPO_ROOT (manifest stores those paths).
        # Reject traversal / absolute paths via *unresolved* normalization —
        # following symlinks here would falsely flag the /datasets-backed
        # symlinked subtrees as escapes.
        norm = os.path.normpath(rel)
        if norm.startswith("..") or os.path.isabs(norm):
            abort(403)
        norm_parts = norm.split(os.sep)
        if not norm_parts or norm_parts[0] != "outputs":
            abort(403)
        target = REPO_ROOT / norm
        if not target.is_file():
            abort(404)
        return send_file(target)

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument(
        "--cw-manifest", default=str(DEFAULT_CW_MANIFEST),
        help="conditioning_window manifest (produced by "
             "`python -m poe_repair.experiments.conditioning_window`). "
             "If missing, the /conditioning_window route renders a hint page.",
    )
    ap.add_argument(
        "--cwl-with-prompt-manifest", default=str(DEFAULT_CWL_WP_MANIFEST),
        help="conditioning_window_lora 'with_prompt' manifest. If missing, "
             "the /conditioning_window_lora route renders a hint page.",
    )
    ap.add_argument(
        "--cwl-always-manifest", default=str(DEFAULT_CWL_ALWAYS_MANIFEST),
        help="conditioning_window_lora 'always' manifest. If missing, "
             "the /conditioning_window_lora route renders a hint page.",
    )
    ap.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}")
        print("run scripts/build_lora_manifest.py first")
        return 1
    cw_path = Path(args.cw_manifest) if args.cw_manifest else None
    if cw_path is not None and not cw_path.is_file():
        print(
            f"[warn] conditioning_window manifest not found: {cw_path}; "
            f"/conditioning_window will render a hint page."
        )
    cwl_wp_path = Path(args.cwl_with_prompt_manifest) if args.cwl_with_prompt_manifest else None
    cwl_al_path = Path(args.cwl_always_manifest) if args.cwl_always_manifest else None
    for label, p in (("with_prompt", cwl_wp_path), ("always", cwl_al_path)):
        if p is not None and not p.is_file():
            print(
                f"[warn] conditioning_window_lora ({label}) manifest not found: {p}; "
                f"/conditioning_window_lora will render a hint page."
            )
    app = create_app(
        manifest_path, Path(args.outputs_root),
        cw_manifest_path=cw_path,
        cwl_wp_manifest_path=cwl_wp_path,
        cwl_always_manifest_path=cwl_al_path,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
