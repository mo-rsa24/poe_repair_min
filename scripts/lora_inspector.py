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
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file
from poe_repair import paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_tab import CROSS_INDEX_HTML  # noqa: E402
from window_tab import WINDOW_INDEX_HTML  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "outputs/lora/a_cat__x__a_dog/seed_42/results/inspector_manifest.json"
DEFAULT_CW_MANIFEST = paths.resolve(paths.CFG_WINDOW_WITHOUT_LORA) / "a_cat__x__a_dog/seed_42/results/inspector_manifest.json"
DEFAULT_CWL_WP_MANIFEST = paths.resolve(paths.CFG_WINDOW_WITH_LORA) / "a_cat__x__a_dog/seed_42/with_prompt/results/inspector_manifest.json"
DEFAULT_CWL_ALWAYS_MANIFEST = paths.resolve(paths.CFG_WINDOW_WITH_LORA) / "a_cat__x__a_dog/seed_42/always/results/inspector_manifest.json"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
# The timing sweeps write to /datasets, not into the repo. The image route is
# rooted at the whole interaction_term tree so one route serves the window
# sweep, the crossed grid and the expert frames.
DEFAULT_IT_ROOT = paths.resolve(paths.INTERACTION_TERM_ROOT)
DEFAULT_IW_ROOT = DEFAULT_IT_ROOT
DEFAULT_IW_MANIFEST = DEFAULT_IT_ROOT / "window/window_inspector_manifest.json"
DEFAULT_CROSS_MANIFEST = DEFAULT_IT_ROOT / "cross/cross_manifest.json"


_TAB_HEADER = r"""
<style>
  .tabs {
    display: flex; align-items: stretch; gap: 0;
    margin: -24px -24px 20px -24px;
    padding: 0 24px; border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .tabs a {
    display: flex; flex-direction: column; gap: 2px;
    padding: 10px 18px; color: var(--muted); text-decoration: none;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .tabs a:hover { color: var(--text); }
  .tabs a.active {
    color: var(--accent); border-bottom-color: var(--accent);
  }
  .tab-title {
    font-size: 13px; font-weight: 600; letter-spacing: 0.02em;
  }
  .tab-role {
    font-size: 10px; font-weight: 500; opacity: 0.7;
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  .tabs .spacer { flex: 1; }
  .pair-picker {
    display: flex; align-items: center; gap: 8px;
    padding: 0 18px; color: var(--muted); font-size: 11px;
    letter-spacing: 0.04em;
  }
  .pair-picker label {
    text-transform: uppercase; font-weight: 600; opacity: 0.7;
  }
  .pair-picker select {
    background: #0f1115; color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 4px 8px; font-size: 12px; font-weight: 600;
    font-family: ui-monospace, monospace; cursor: pointer;
  }
  .pair-picker select:disabled { opacity: 0.4; cursor: not-allowed; }
  .pair-picker .hint { color: var(--muted); opacity: 0.6; font-size: 10px; }
</style>
<nav class="tabs">
  <a href="/conditioning_window{{ query_pair }}" class="{{ 'active' if active == 'cw' else '' }}">
    <span class="tab-title">CFG-mask ablation</span>
    <span class="tab-role">floor &middot; no LoRA</span>
  </a>
  <a href="/{{ query_pair }}" class="{{ 'active' if active == 'residual' else '' }}">
    <span class="tab-title">LoRA residual</span>
    <span class="tab-role">mechanism &middot; training trajectory</span>
  </a>
  <a href="/mds_large{{ query_pair }}" class="{{ 'active' if active == 'mds_large' else '' }}">
    <span class="tab-title">MDS large</span>
    <span class="tab-role">single plot &middot; decoded thumbnails</span>
  </a>
  <a href="/conditioning_window_lora" class="{{ 'active' if active == 'cwl' else '' }}">
    <span class="tab-title">LoRA + CFG-mask</span>
    <span class="tab-role">payoff &middot; rescue test</span>
  </a>
  <a href="/interaction_window" class="{{ 'active' if active == 'iw' else '' }}">
    <span class="tab-title">Correction timing</span>
    <span class="tab-role">when &middot; sliding window</span>
  </a>
  <a href="/interaction_cross" class="{{ 'active' if active == 'ix' else '' }}">
    <span class="tab-title">Prompt &times; correction</span>
    <span class="tab-role">crossed &middot; step by step</span>
  </a>
  <span class="spacer"></span>
  {% if pair_options and pair_options|length > 1 %}
  <div class="pair-picker">
    <label for="pair-select">pair</label>
    <select id="pair-select" {% if active == 'cwl' %}disabled title="LoRA + CFG-mask is only rendered for cat × dog right now."{% endif %}>
      {% for slug, name in pair_options %}
      <option value="{{ slug }}" {% if slug == current_pair %}selected{% endif %}>{{ name }}</option>
      {% endfor %}
    </select>
    {% if active == 'cwl' %}<span class="hint">cat × dog only</span>{% endif %}
  </div>
  <script>
    (function() {
      const sel = document.getElementById('pair-select');
      if (!sel || sel.disabled) return;
      sel.addEventListener('change', () => {
        const u = new URL(window.location);
        u.searchParams.set('pair', sel.value);
        window.location = u.toString();
      });
    })();
  </script>
  {% endif %}
</nav>
"""


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
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 6px 0; color: var(--text); }
  .subtitle { color: var(--muted); font-size: 12px; margin: 0 0 18px 0; }
  details.diagnostics {
    margin-top: 28px; padding: 12px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted); font-size: 12px;
  }
  details.diagnostics summary {
    cursor: pointer; user-select: none; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 11px;
  }
  details.diagnostics .body { margin-top: 8px; line-height: 1.6; word-break: break-all; }
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
  .mds-row {
    margin-top: 16px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .mds-row h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .mds-row .mds-img-wrap {
    display: flex; justify-content: center; align-items: center;
    background: #FBFCFD; border-radius: 4px; min-height: 320px;
    padding: 6px;
  }
  .mds-row img {
    max-width: 100%; max-height: 540px; height: auto; display: block;
    border-radius: 4px;
  }
  .mds-row .caption {
    margin-top: 8px; color: var(--muted); font-size: 11px; text-align: center;
  }
  .mds-row .caption.missing { color: var(--warn); }
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
<h1>How the LoRA residual fits over training &mdash; &lsquo;{{ joint_prompt }}&rsquo;, seed 42</h1>
<p class="subtitle">left: per-arm PoE with no LoRA &nbsp;&middot;&nbsp; middle: PoE + &lambda;&middot;r at the current epoch &nbsp;&middot;&nbsp; right: mono (joint CFG, the diagnostic ceiling)</p>

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
  <div class="panel">
    <h2>Mono (joint CFG, ceiling)</h2>
    {% if mono_path %}
    <img id="img-mono" src="/img/{{ mono_path }}" alt="mono">
    {% else %}
    <img id="img-mono" alt="mono not rendered">
    <p class="caption">no mono.png on disk; render one to populate this pane</p>
    {% endif %}
  </div>
</div>

<div id="toast">cell not available</div>

<details class="diagnostics">
  <summary>diagnostics</summary>
  <div class="body">
    Source: <code>{{ results_root }}</code><br>
    Manifest: <code>outputs/lora/a_cat__x__a_dog/seed_42/results/inspector_manifest.json</code> &middot;
    {{ epochs|length }} epochs &times; {{ lambdas|length }} &lambda; values rendered.
  </div>
</details>

<script>
const MANIFEST = {{ manifest_json|safe }};
const EPOCHS = MANIFEST.epochs;
const LAMBDAS = MANIFEST.lambdas;     // strings like "0.00", "0.50", "1.00"
const CELLS = MANIFEST.cells;          // {"100": {"0.00": "outputs/.../decoded.png", ...}}
const SOURCE = MANIFEST.cell_source_run || {};

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


MDS_LARGE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MDS — single large plot (LoRA inspector)</title>
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
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 6px 0; color: var(--text); }
  .subtitle { color: var(--muted); font-size: 12px; margin: 0 0 18px 0; }
  details.diagnostics {
    margin-top: 28px; padding: 12px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted); font-size: 12px;
  }
  details.diagnostics summary {
    cursor: pointer; user-select: none; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 11px;
  }
  details.diagnostics .body { margin-top: 8px; line-height: 1.6; word-break: break-all; }
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
  .mds-large-wrap {
    background: #FBFCFD; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
    display: flex; justify-content: center; align-items: center;
    min-height: 540px;
  }
  .mds-large-wrap img {
    max-width: 100%; height: auto; display: block; border-radius: 4px;
  }
  .legend {
    display: flex; flex-wrap: wrap; gap: 18px;
    margin: 12px 0 0 0; padding: 10px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted); font-size: 12px;
  }
  .legend .item {
    display: inline-flex; align-items: center; gap: 6px;
  }
  .legend .swatch {
    width: 12px; height: 12px; border-radius: 2px; display: inline-block;
  }
  .legend .swatch.A { background: #C84C5B; }
  .legend .swatch.B { background: #2B6F97; }
  .legend .swatch.AB { background: #3B8D5B; }
  .legend .swatch.LoRA { background: #D9872B; }
  .caption.missing { color: var(--warn); }
  .mode-toggle {
    flex: 1; display: inline-flex; gap: 0;
    background: #0f1115; border: 1px solid var(--border);
    border-radius: 6px; padding: 3px; max-width: 420px;
  }
  .mode-btn {
    flex: 1; padding: 6px 10px; cursor: pointer;
    background: transparent; color: var(--muted);
    border: 0; border-radius: 4px;
    font: inherit; font-size: 12px; font-weight: 600;
    letter-spacing: 0.02em;
    transition: background 0.12s ease, color 0.12s ease;
  }
  .mode-btn:hover { color: var(--text); }
  .mode-btn.active {
    background: var(--accent); color: #0a1320;
  }
  .mode-btn.disabled {
    opacity: 0.45; cursor: not-allowed;
  }
  .mode-hint {
    color: var(--muted); font-size: 11px; flex: 0 0 auto;
    max-width: 240px; line-height: 1.3;
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
<h1>Latent trajectories &mdash; single large MDS plot ({{ pair_display }})</h1>
<p class="subtitle">
  A, B, A&and;B are static; PoE + &lambda;&middot;r (LoRA) moves with the sliders.
  Decoded-image thumbnails sit at each path&rsquo;s terminal point, bordered in the path colour.
  Toggle the embedding to compare raw-latent MDS (pixel appearance) vs semantic MDS
  (DINOv2 over predicted-x&#770;<sub>0</sub>, encodes spatial co-occurrence).
</p>

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
  <div class="row">
    <label for="mode">embedding</label>
    <div class="mode-toggle" id="mode-toggle" role="tablist" aria-label="MDS embedding mode">
      <button id="mode-latent"   class="mode-btn active" data-mode="latent"   role="tab" aria-selected="true">raw latent z<sub>t</sub></button>
      <button id="mode-semantic" class="mode-btn disabled" data-mode="semantic" role="tab" aria-selected="false" disabled title="DINOv2 / semantic mode is disabled">semantic (DINOv2 · x̂<sub>0</sub>)</button>
    </div>
    <div class="mode-hint" id="mode-hint"></div>
  </div>
</div>

<div class="mds-large-wrap">
  <img id="img-mds-large" alt="MDS large panel">
</div>
<div class="legend">
  <div class="item"><span class="swatch A"></span> A &mdash; solo prompt-A CFG (static)</div>
  <div class="item"><span class="swatch B"></span> B &mdash; solo prompt-B CFG (static)</div>
  <div class="item"><span class="swatch AB"></span> A&and;B &mdash; joint-prompt CFG / mono (static)</div>
  <div class="item"><span class="swatch LoRA"></span> LoRA &mdash; PoE + &lambda;&middot;r (moves with sliders)</div>
</div>
<p class="caption" id="mds-large-caption" style="margin-top:8px; color: var(--muted); font-size: 11px;">
  pre-rendered per (epoch, &lambda;). Static endpoints share a global projection so they don&rsquo;t shift as you scrub.
</p>

<div id="toast">cell not available</div>

<details class="diagnostics">
  <summary>diagnostics</summary>
  <div class="body">
    Source: <code>{{ results_root }}</code><br>
    Latent panels: {{ n_large_cells }} / {{ epochs|length * lambdas|length }}
    grid positions ({{ epochs|length }} epochs &times; {{ lambdas|length }} &lambda;).
    Semantic panels: {{ n_large_cells_semantic }} / {{ epochs|length * lambdas|length }}.<br>
    Populate latent cells with <code>scripts/build_lora_inspector_mds.py</code>
    (<code>--stages collect-static,collect-cells,project,render-large,update-manifest</code>).<br>
    Populate semantic cells with <code>scripts/build_lora_inspector_mds_semantic.py</code>
    after the latent build (<code>--stages collect-static,collect-cells,project,align,render-large,update-manifest</code>).
  </div>
</details>

<script>
const MANIFEST = {{ manifest_json|safe }};
const EPOCHS = MANIFEST.epochs;
const LAMBDAS = MANIFEST.lambdas;
const MDS_LARGE = {
  latent:   MANIFEST.mds_cells_large || {},
  semantic: MANIFEST.mds_cells_large_semantic || {},
};
const SOURCE = MANIFEST.cell_source_run || {};

const MODE_INFO = {
  latent: {
    caption: 'MDS over flattened z_t. Distances track pixel-level latent ' +
             'appearance — useful as a path-shape view, but does NOT encode ' +
             'spatial co-occurrence. Don\'t read "closer to mono" semantically here.',
    hint: 'z_t · pixel appearance',
  },
  semantic: {
    caption: 'MDS over DINOv2 features of decoded predicted-x̂₀(t). ' +
             'Distances track semantic content (co-occurrence). Watch the ' +
             'PoE+λ·r path bend toward mono as training progresses.',
    hint: 'DINOv2 · semantic',
  },
};

const epochSlider = document.getElementById('epoch');
const epochVal = document.getElementById('epoch-val');
const epochSrc = document.getElementById('epoch-src');
const lambdaSlider = document.getElementById('lambda');
const lambdaVal = document.getElementById('lambda-val');
const imgMdsLarge = document.getElementById('img-mds-large');
const mdsCaption = document.getElementById('mds-large-caption');
const toast = document.getElementById('toast');
const modeBtnLatent = document.getElementById('mode-latent');
const modeBtnSemantic = document.getElementById('mode-semantic');
const modeHint = document.getElementById('mode-hint');

let currentMode = 'latent';
let toastTimer = null;

function flashToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
}

function lambdaForIdx(idx) { return LAMBDAS[idx]; }
function epochForIdx(idx) { return EPOCHS[idx]; }

function mdsLargePath(epoch, lam, mode) {
  const map = MDS_LARGE[mode] || {};
  const e = String(epoch);
  if (!map[e]) return null;
  return map[e][lam] || null;
}

function semanticAvailable() {
  const m = MDS_LARGE.semantic || {};
  for (const k in m) if (Object.keys(m[k] || {}).length > 0) return true;
  return false;
}

function setMode(mode) {
  if (mode === 'semantic') {
    flashToast('semantic (DINOv2) mode is disabled');
    return;
  }
  currentMode = mode;
  modeBtnLatent.classList.toggle('active', mode === 'latent');
  modeBtnLatent.setAttribute('aria-selected', mode === 'latent');
  modeBtnSemantic.classList.toggle('active', mode === 'semantic');
  modeBtnSemantic.setAttribute('aria-selected', mode === 'semantic');
  modeHint.textContent = MODE_INFO[mode].hint;
  update();
}

function update() {
  const eIdx = parseInt(epochSlider.value, 10);
  const lIdx = parseInt(lambdaSlider.value, 10);
  const epoch = epochForIdx(eIdx);
  const lam = lambdaForIdx(lIdx);
  epochVal.textContent = epoch;
  lambdaVal.textContent = lam;
  epochSrc.textContent = SOURCE[String(epoch)] || '';

  const p = mdsLargePath(epoch, lam, currentMode);
  if (p) {
    imgMdsLarge.src = '/img/' + p;
    imgMdsLarge.style.display = 'block';
    mdsCaption.classList.remove('missing');
    mdsCaption.textContent = MODE_INFO[currentMode].caption;
  } else {
    imgMdsLarge.removeAttribute('src');
    imgMdsLarge.style.display = 'none';
    mdsCaption.classList.add('missing');
    const script = currentMode === 'semantic'
      ? 'scripts/build_lora_inspector_mds_semantic.py'
      : 'scripts/build_lora_inspector_mds.py';
    mdsCaption.textContent =
      `no ${currentMode} panel for epoch=${epoch}, λ=${lam} — re-run ` +
      `${script} with --stages render-large.`;
    flashToast(`no ${currentMode} panel at epoch=${epoch}, λ=${lam}`);
  }
}

epochSlider.addEventListener('input', update);
lambdaSlider.addEventListener('input', update);
modeBtnLatent.addEventListener('click', () => setMode('latent'));
modeBtnSemantic.addEventListener('click', () => setMode('semantic'));

modeBtnSemantic.classList.add('disabled');
modeBtnSemantic.disabled = true;
modeBtnSemantic.title = 'DINOv2 / semantic mode is disabled';
setMode('latent');
</script>
</body>
</html>
"""


def _with_tabs(template: str) -> str:
    """Inject the shared tab nav immediately after <body>."""
    return template.replace("<body>", "<body>\n" + _TAB_HEADER, 1)


_MONO_FALLBACK_CAT_DOG = paths.resolve(paths.HELD_OUT_SEEDS_INDEX) / "trajectory_diagram/seed_42/mono.png"
# Default pair slug — preserves current behaviour for users who don't pass ?pair=.
DEFAULT_PAIR = "a_cat__x__a_dog"


def _pair_has_residual_cells(entry: dict) -> bool:
    """A pair belongs in the LoRA-residual dropdown iff its manifest carries
    a non-empty per-epoch decoded-probe map."""
    m = entry.get("residual") or {}
    cells = m.get("cells") or {}
    return any((v or {}) for v in cells.values()) if isinstance(cells, dict) else False


def _pair_has_mds_large(entry: dict) -> bool:
    """A pair belongs in the MDS-large dropdown iff its manifest carries a
    non-empty mds_cells_large map."""
    m = entry.get("residual") or {}
    cells = m.get("mds_cells_large") or {}
    return any((v or {}) for v in cells.values()) if isinstance(cells, dict) else False


def _pair_options_for(active: str, pairs: dict[str, dict]) -> list[tuple[str, str]]:
    """Per-tab pair dropdown. Each tab filters by what its UI actually needs:
    residual tab → has decoded probes; mds_large → has mds_cells_large;
    cw → has a cw manifest. Default sort: DEFAULT_PAIR first, then alphabetical."""
    if active == "residual":
        ok = _pair_has_residual_cells
    elif active == "mds_large":
        ok = _pair_has_mds_large
    elif active in {"cw", "cwl"}:
        ok = lambda e: bool(e.get("cw"))
    else:
        ok = lambda e: True
    visible = [s for s, e in pairs.items() if ok(e)]
    ordered = [DEFAULT_PAIR] + sorted(s for s in visible if s != DEFAULT_PAIR)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for s in ordered:
        if s in seen or s not in pairs or not ok(pairs[s]):
            continue
        seen.add(s)
        out.append((s, _display_name(s)))
    return out


def _joint_prompt_for(pair_slug: str, outputs_root: Path) -> str:
    """Read joint_prompt from the pair's results config; fall back to the
    display name."""
    cfg_path = outputs_root / "lora" / pair_slug / "seed_42" / "results" / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            jp = (cfg.get("cell") or {}).get("joint_prompt")
            if jp:
                return jp
        except Exception:
            pass
    return _display_name(pair_slug)


def _resolve_mono_path(manifest: dict, pair_slug: str) -> str | None:
    """Per-pair mono lookup. Mono = joint-CFG sample with no LoRA at the same
    seed and sampler config as the residual probes.

    For ``a_cat__x__a_dog``, the existing cross_seed_lora_pooling render is the
    canonical source (preserves pre-existing behaviour). For other pairs,
    the conditioning_window ``prefix_k50`` schedule image is used: it is
    full-prefix (every step conditioned) = mono.
    """
    candidates: list[str] = []
    field = manifest.get("mono_path")
    if field:
        candidates.append(field)
    if pair_slug == "a_cat__x__a_dog":
        candidates.append(_MONO_FALLBACK_CAT_DOG)
    candidates.append(
        f"{paths.CFG_WINDOW_WITHOUT_LORA}/{pair_slug}/seed_42/schedules/prefix_k50/image.png"
    )
    for c in candidates:
        if (REPO_ROOT / c).is_file():
            return c
    return None


def _display_name(slug: str) -> str:
    """Human-readable label for a pair slug.

    ``a_camel__x__a_desert_landscape`` → ``camel × desert landscape``.
    ``a_cat__x__a_dog`` → ``cat × dog``.
    """
    if "__x__" in slug:
        parts = slug.split("__x__")
        cleaned = [p[2:] if p.startswith("a_") else p for p in parts]
        return " × ".join(c.replace("_", " ") for c in cleaned)
    if "_" in slug:
        return slug.replace("_", " × ")
    return slug


def _discover_pairs(outputs_root: Path) -> dict[str, dict]:
    """Scan ``outputs/lora/*/seed_42/results/inspector_manifest.json`` and
    ``outputs/conditioning_window/*/seed_42/results/inspector_manifest.json``,
    returning ``{pair_slug: {"residual": manifest|None, "cw": manifest|None}}``.
    A pair is included iff it has at least one manifest."""
    pairs: dict[str, dict] = {}
    residual_root = outputs_root / "lora"
    if residual_root.is_dir():
        for pair_dir in sorted(residual_root.iterdir()):
            if pair_dir.is_symlink() or not pair_dir.is_dir():
                continue  # skip compat symlinks (e.g. cat_dog -> a_cat__x__a_dog)
            mpath = pair_dir / "seed_42" / "results" / "inspector_manifest.json"
            if mpath.is_file():
                pairs.setdefault(pair_dir.name, {})["residual"] = json.loads(mpath.read_text())
    cw_root = outputs_root / "conditioning_window"
    if cw_root.is_dir():
        for pair_dir in sorted(cw_root.iterdir()):
            if pair_dir.is_symlink() or not pair_dir.is_dir():
                continue  # skip compat symlinks (e.g. cat_dog -> a_cat__x__a_dog)
            mpath = pair_dir / "seed_42" / "results" / "inspector_manifest.json"
            if mpath.is_file():
                pairs.setdefault(pair_dir.name, {})["cw"] = json.loads(mpath.read_text())
    # Ensure every entry has both keys (None when absent).
    for slug, entry in pairs.items():
        entry.setdefault("residual", None)
        entry.setdefault("cw", None)
    return pairs


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
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 6px 0; color: var(--text); }
  .subtitle { color: var(--muted); font-size: 12px; margin: 0 0 4px 0; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 18px; }
  details.diagnostics {
    margin-top: 28px; padding: 12px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted); font-size: 12px;
  }
  details.diagnostics summary {
    cursor: pointer; user-select: none; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 11px;
  }
  details.diagnostics .body { margin-top: 8px; line-height: 1.7; }
  details.diagnostics code { color: var(--text); }

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
<h1>When does conditioning matter? &mdash; CFG-mask ablation on &lsquo;{{ prompt }}&rsquo;, seed {{ seed }}</h1>
<p class="subtitle">
  Each rendered image is a full denoise. The mask below picks which timesteps used the prompt
  (conditional branch) and which ran with the unconditional branch only.
</p>
<div class="meta">
  {{ num_steps }} inference steps &nbsp;&middot;&nbsp; CFG = {{ guidance }} &nbsp;&middot;&nbsp;
  <span title="A schedule is a specific on/off pattern over the {{ num_steps }} steps. Families: prefix_k (ON for the first k), suffix_k (ON for the last k), window_a_b, pulse_t_w, plus all_on/all_off as boundaries.">{{ schedules|length }} mask patterns rendered</span>
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
    <span>conditioned on steps <b id="start-val">0</b>&hairsp;&ndash;&hairsp;<b id="end-val">{{ num_steps }}</b></span>
    <span><b id="num-on">{{ num_steps }}</b>/{{ num_steps }} steps conditioned</span>
    <span>render <span class="time-pill" id="time-pill">&mdash;</span></span>
  </div>

  <div id="strip" class="strip" title=""></div>
  <div class="legend">
    <span class="sw" style="background: var(--on);"></span> conditional ON
    &nbsp;&nbsp;
    <span class="sw" style="background: var(--off);"></span> conditional OFF
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>decoded image</h2>
    <img id="img" alt="decoded">
    <div class="caption" id="caption"></div>
  </div>
</div>

<details class="diagnostics">
  <summary>diagnostics</summary>
  <div class="body">
    <b>Schedule families:</b> <code>prefix_k</code> (conditioned for the first k steps),
    <code>suffix_k</code> (conditioned for the last k), <code>window_a_b</code>,
    <code>pulse_t_w</code>, plus <code>all_on</code> / <code>all_off</code> as boundaries.<br>
    <b>Sanity:</b> <code>all_on</code> &Delta; = {{ sanity_on }} ({{ sanity_on_pass }})
    &middot; <code>all_off</code> &Delta; = {{ sanity_off }} ({{ sanity_off_pass }}).
    These verify the masked sampler is bit-identical to <code>run_cfg</code> at all-on
    and to the unconditional sampler at all-off (max-abs latent &Delta;).
  </div>
</details>

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
  if (s === 0 && e === N) return 'full CFG (every step conditioned)';
  if (s === e)             return 'no CFG (never conditioned)';
  if (s === 0)             return 'early-only';
  if (e === N)             return 'late-only';
  if (e - s <= 3)          return 'brief pulse';
  return 'mid-trajectory';
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

  // All displayed numbers reflect the *snapped* schedule. Ranges are
  // inclusive in the UI so the count and the endpoints agree
  // (e.g. 0–9 = 10 steps).
  const snapStart = ent.mask.indexOf('1');
  const snapEndExcl = ent.mask.lastIndexOf('1') + 1;
  if (ent.num_on === 0) {
    startVal.textContent = '—';
    endVal.textContent   = '—';
  } else {
    startVal.textContent = snapStart;
    endVal.textContent   = snapEndExcl - 1;
  }
  numOnEl.textContent  = ent.num_on;
  modeEl.textContent   = inferMode(snapStart < 0 ? 0 : snapStart, snapEndExcl);
  timePillEl.textContent = fmtSeconds(ent.elapsed_s);
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
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 6px 0; color: var(--text); }
  .subtitle { color: var(--muted); font-size: 12px; margin: 0 0 4px 0; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 18px; }
  details.diagnostics {
    margin-top: 28px; padding: 12px 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; color: var(--muted); font-size: 12px;
  }
  details.diagnostics summary {
    cursor: pointer; user-select: none; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase; font-size: 11px;
  }
  details.diagnostics .body { margin-top: 8px; line-height: 1.7; }
  details.diagnostics code { color: var(--text); }

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

  .panels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 {
    margin: 0 0 8px 0; font-size: 12px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
    display: flex; align-items: center; gap: 8px;
  }
  .panel img, .panel canvas {
    width: 100%; aspect-ratio: 1 / 1; border-radius: 4px;
    background: #000; display: block;
  }
  .panel .caption {
    margin-top: 8px; color: var(--muted); font-size: 11px;
    font-family: ui-monospace, monospace; word-break: break-all;
  }

  /* ---- Info card UX (ⓘ toggles a card; one card open at a time) ---- */
  .info-btn {
    width: 18px; height: 18px; border-radius: 50%;
    border: 1px solid var(--border); background: transparent;
    color: var(--muted); cursor: pointer; font-size: 11px;
    line-height: 16px; padding: 0; font-weight: 700;
    font-family: ui-monospace, monospace;
    transition: color 0.12s ease, border-color 0.12s ease, background 0.12s ease;
  }
  .info-btn:hover { color: var(--text); border-color: var(--text); }
  .info-btn.open  { color: #0f1115; background: var(--accent); border-color: var(--accent); }
  .info-card {
    margin: 0 0 10px 0; padding: 10px 12px;
    background: #0f1115; border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 4px; color: var(--text); font-size: 12px; line-height: 1.55;
    text-transform: none; letter-spacing: 0; font-weight: 400;
    display: none;
  }
  .info-card.open { display: block; }
  .info-card p { margin: 0 0 6px 0; }
  .info-card p:last-child { margin-bottom: 0; }
  .info-card .ask { color: var(--accent-hot); font-style: italic; }

  .strip-header {
    display: flex; align-items: center; gap: 8px;
    color: var(--muted); font-size: 11px; margin-top: 12px; margin-bottom: 2px;
    font-family: ui-monospace, monospace;
  }
  .strip-info-wrap { margin-bottom: 6px; }
</style>
</head>
<body>
<h1>How the LoRA behaves when conditioning is dropped &mdash; &lsquo;{{ prompt }}&rsquo;, seed {{ seed }}</h1>
<div class="meta">
  {{ num_steps }} inference steps &nbsp;&middot;&nbsp; CFG = {{ guidance }} &nbsp;&middot;&nbsp;
  {{ n_off }} prompt schedules rendered
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
    <span><span class="mode" id="mode-disp">window</span></span>
    <span>prompt fires on steps <b id="start-val">0</b>&hairsp;&ndash;&hairsp;<b id="end-val">{{ num_steps }}</b></span>
    <span><b id="num-on">{{ num_steps }}</b>/{{ num_steps }} steps with prompt</span>
    <span>render <span class="time-pill" id="time-pill">&mdash;</span></span>
  </div>

  <div class="meta-row" style="margin-top:10px;">
    <span class="toggle-label">LoRA epoch (training step):</span>
    <input id="epoch-slider" type="range" min="0" max="0" step="1" value="0" style="flex:1;max-width:480px;">
    <span><b id="epoch-val">—</b></span>
    <span class="toggle-label" style="margin-left:18px;">LoRA strength λ:</span>
    <input id="lambda-slider" type="range" min="0" max="0" step="1" value="0" style="flex:1;max-width:240px;">
    <span><b id="lambda-val">—</b></span>
  </div>

  <div class="strip-info-wrap">
    <div class="strip-header">
      <span>Prompt schedule &mdash; green: prompt fires &middot; grey: prompt is off</span>
      <button class="info-btn" data-info="strip" title="What does this strip mean?">i</button>
    </div>
    <div id="info-strip" class="info-card">
      <p>The strip below has one cell per sampling step ({{ num_steps }} total).</p>
      <p><b>Green</b>: the prompt <i>&lsquo;{{ prompt }}&rsquo;</i> fires.<br>
         <b>Grey</b>: no prompt &mdash; unconditional only.</p>
      <p>Drag the slider above to change the pattern. Every image on this page is sampled with this schedule.</p>
    </div>
    <div id="mask-strip" class="strip"></div>
  </div>

  <div class="legend">
    <span class="sw" style="background: var(--on);"></span> prompt fires
    &nbsp;&nbsp;
    <span class="sw" style="background: var(--off);"></span> prompt off
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>
      <span>Baseline (no LoRA)</span>
      <button class="info-btn" data-info="off" title="What is this pane?">i</button>
    </h2>
    <div id="info-off" class="info-card">
      <p>No LoRA anywhere.</p>
      <p><b>Green steps</b>: per-arm PoE with the prompt (raw CFG).<br>
         <b>Grey steps</b>: unconditional only.</p>
      <p>This is the floor &mdash; same image as the CFG-mask ablation tab.</p>
    </div>
    <img id="img-off" alt="baseline">
    <div class="caption" id="cap-off">&mdash;</div>
  </div>
  <div class="panel">
    <h2>
      <span>LoRA fires with the prompt</span>
      <button class="info-btn" data-info="wp" title="What is this pane?">i</button>
    </h2>
    <div id="info-wp" class="info-card">
      <p>LoRA fires only on green steps. When the prompt drops, the LoRA drops with it.</p>
      <p><b>Green steps</b>: PoE + LoRA correction.<br>
         <b>Grey steps</b>: unconditional, no LoRA.</p>
      <p class="ask">Can a short LoRA-pushed green window carry the sample, or does the prior take over once the prompt drops?</p>
    </div>
    <img id="img-wp" alt="LoRA with prompt">
    <div class="caption" id="cap-wp">&mdash;</div>
  </div>
  <div class="panel">
    <h2>
      <span>LoRA fires every step</span>
      <button class="info-btn" data-info="al" title="What is this pane?">i</button>
    </h2>
    <div id="info-al" class="info-card">
      <p>LoRA fires on every step &mdash; green AND grey.</p>
      <p><b>Green steps</b>: PoE + LoRA correction (same as the middle pane).<br>
         <b>Grey steps</b>: unconditional + LoRA correction.</p>
      <p>The LoRA was trained on PoE outputs; on grey steps it is applied to a plain unconditional forward &mdash; off-distribution use.</p>
      <p class="ask">Can the LoRA carry the sample on grey steps, where there is no prompt to lean on?</p>
    </div>
    <img id="img-al" alt="LoRA every step">
    <div class="caption" id="cap-al">&mdash;</div>
  </div>
</div>

<details class="diagnostics">
  <summary>diagnostics</summary>
  <div class="body">
    <b>Sampling grammar.</b> The middle pane corresponds to composition mode <code>with_prompt</code>
    (LoRA gated by the conditioning mask). The right pane corresponds to <code>always</code>
    (LoRA fires on every step, including the prompt-off ones).<br>
    <b>Per-step LoRA push.</b> Each rendered cell records
    <code>delta_norm_per_step</code> = &Vert;&Delta;&#x0302;<sub>t</sub>&Vert; for every step,
    where &Delta;&#x0302;<sub>t</sub> = &epsilon;<sup>PoE</sup><sub>LoRA</sub> &minus; &epsilon;<sup>PoE</sup><sub>frozen</sub>
    is the LoRA&apos;s correction in &epsilon;-space at that timestep. Used to debug whether the
    LoRA is &ldquo;kicking&rdquo; early or driving continuously; not shown by default.<br>
    <b>Sanity.</b> <code>masked(all_on, with_prompt, &lambda;=1)</code> vs
    <code>run_lora_residual_inject(&lambda;=1)</code>: &Delta; = {{ sanity_delta }} ({{ sanity_pass }}).
    Verifies the per-arm-PoE-with-LoRA grammar is bit-identical to the reference sampler at all-on.<br>
    <b>Counts.</b> {{ n_wp }} <code>with_prompt</code> cells, {{ n_al }} <code>always</code> cells
    (each cell = one epoch &times; &lambda; pair).
  </div>
</details>

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
  if (s === 0 && e === N) return 'full CFG (every step conditioned)';
  if (s === e)             return 'no CFG (never conditioned)';
  if (s === 0)             return 'early-only';
  if (e === N)             return 'late-only';
  if (e - s <= 3)          return 'brief pulse';
  return 'mid-trajectory';
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

let currentEntry = null;
let currentEpoch = null;       // numeric, e.g. 62500
let currentLambdaTag = null;   // string, e.g. "1.00"

const imgOff = document.getElementById('img-off');
const imgWp  = document.getElementById('img-wp');
const imgAl  = document.getElementById('img-al');

const startEl  = document.getElementById('start');
const endEl    = document.getElementById('end');
const hlEl     = document.getElementById('hl');
const startVal = document.getElementById('start-val');
const endVal   = document.getElementById('end-val');
const numOnEl  = document.getElementById('num-on');
const modeDispEl = document.getElementById('mode-disp');
const timePillEl = document.getElementById('time-pill');
const capOffEl   = document.getElementById('cap-off');
const capWpEl    = document.getElementById('cap-wp');
const capAlEl    = document.getElementById('cap-al');

function pickEntry(mode, id) {
  return indexByIdCell(mode, currentEpoch, currentLambdaTag)[id] || null;
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
  const wpEnt  = pickEntry('with_prompt', ent.id);
  const alEnt  = pickEntry('always',      ent.id);

  imgOff.src = offEnt ? ('/img/' + offEnt.image_path) : '';
  imgWp .src = wpEnt  ? ('/img/' + wpEnt.image_path)  : '';
  imgAl .src = alEnt  ? ('/img/' + alEnt.image_path)  : '';

  // Display numbers reflect the *snapped* schedule. Inclusive ranges so
  // count and endpoints agree (e.g. 0–9 = 10 steps).
  const snapStart = ent.mask.indexOf('1');
  const snapEndExcl = ent.mask.lastIndexOf('1') + 1;
  if (ent.num_on === 0) {
    startVal.textContent = '—';
    endVal.textContent   = '—';
  } else {
    startVal.textContent = snapStart;
    endVal.textContent   = snapEndExcl - 1;
  }
  numOnEl.textContent  = ent.num_on;
  modeDispEl.textContent = inferMode(snapStart < 0 ? 0 : snapStart, snapEndExcl);
  // Use the with_prompt render time as the headline; both modes render at
  // similar wall-clocks per cell, and reporting one number keeps the UI tidy.
  timePillEl.textContent = wpEnt ? fmtSeconds(wpEnt.elapsed_s)
                                 : (alEnt ? fmtSeconds(alEnt.elapsed_s) : '—');
  capOffEl.textContent = offEnt ? offEnt.image_path : '(no baseline image for ' + ent.id + ')';
  capWpEl .textContent = wpEnt  ? wpEnt.image_path  : '(not rendered for ' + ent.id + ')';
  capAlEl .textContent = alEnt  ? alEnt.image_path  : '(not rendered for ' + ent.id + ')';
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

// Slider grids are driven from the with_prompt manifest. Both manifests
// (with_prompt, always) cover the same (epoch, λ) cells in the sweep, so
// one is canonical for navigation.
const NAV_MODE = 'with_prompt';

function refreshEpochSlider() {
  const epochs = availableEpochs(NAV_MODE);
  epochSlider.min = 0;
  epochSlider.max = Math.max(0, epochs.length - 1);
  let idx = epochs.indexOf(currentEpoch);
  if (idx < 0) {
    const near = nearestEpoch(NAV_MODE, currentEpoch ?? 0);
    idx = (near != null) ? epochs.indexOf(near) : 0;
    currentEpoch = epochs[idx] ?? null;
  }
  epochSlider.value = idx;
  epochValEl.textContent = (currentEpoch != null) ? currentEpoch : '—';
  epochSlider.disabled = epochs.length <= 1;
}

function refreshLambdaSlider() {
  const lams = availableLambdas(NAV_MODE, currentEpoch);
  lambdaSlider.min = 0;
  lambdaSlider.max = Math.max(0, lams.length - 1);
  let idx = lams.indexOf(currentLambdaTag);
  if (idx < 0) {
    const near = nearestLambda(NAV_MODE, currentEpoch, currentLambdaTag ?? '1.00');
    idx = (near != null) ? lams.indexOf(near) : 0;
    currentLambdaTag = lams[idx] ?? null;
  }
  lambdaSlider.value = idx;
  lambdaValEl.textContent = (currentLambdaTag != null) ? `λ=${currentLambdaTag}` : '—';
  lambdaSlider.disabled = lams.length <= 1;
}

function onCellChanged() {
  rebuildEntries(NAV_MODE, currentEpoch, currentLambdaTag);
  if (ENTRIES.length === 0) {
    imgOff.removeAttribute('src');
    imgWp .removeAttribute('src');
    imgAl .removeAttribute('src');
    document.getElementById('mask-strip').innerHTML = '';
    document.getElementById('mode-disp').textContent = '(no data at this cell)';
    return;
  }
  update('cell');
}

epochSlider.addEventListener('input', () => {
  const epochs = availableEpochs(NAV_MODE);
  const idx = Math.min(epochs.length - 1, Math.max(0, parseInt(epochSlider.value, 10)));
  currentEpoch = epochs[idx];
  epochValEl.textContent = currentEpoch;
  refreshLambdaSlider();
  onCellChanged();
});
lambdaSlider.addEventListener('input', () => {
  const lams = availableLambdas(NAV_MODE, currentEpoch);
  const idx = Math.min(lams.length - 1, Math.max(0, parseInt(lambdaSlider.value, 10)));
  currentLambdaTag = lams[idx];
  lambdaValEl.textContent = `λ=${currentLambdaTag}`;
  onCellChanged();
});

// ---- Info cards: ⓘ buttons toggle inline cards; one open at a time ----
document.querySelectorAll('.info-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = 'info-' + btn.dataset.info;
    const card = document.getElementById(target);
    const opening = !card.classList.contains('open');
    // Close all cards and deactivate all buttons.
    document.querySelectorAll('.info-card').forEach(c => c.classList.remove('open'));
    document.querySelectorAll('.info-btn').forEach(b => b.classList.remove('open'));
    if (opening) {
      card.classList.add('open');
      btn.classList.add('open');
    }
  });
});

// Initial state: latest available (epoch, λ).
(function initCell() {
  const epochs = availableEpochs(NAV_MODE);
  currentEpoch = epochs[epochs.length - 1] ?? null;
  const lams = (currentEpoch != null) ? availableLambdas(NAV_MODE, currentEpoch) : [];
  currentLambdaTag = lams[lams.length - 1] ?? null;
  refreshEpochSlider();
  refreshLambdaSlider();
  rebuildEntries(NAV_MODE, currentEpoch, currentLambdaTag);
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
<p><code>python -m poe_repair.experiments.cfg_window_without_lora</code></p>
</body></html>
"""


_CWL_MISSING_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>conditioning_window_lora manifest missing</title>
<style>body{font:14px -apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e9ef;padding:24px;}
code{background:#161a22;padding:2px 6px;border-radius:4px;color:#7ab7ff;}</style>
</head><body>
<h2>conditioning_window_lora manifests not found</h2>
<p>Missing: <code>{{ missing|safe }}</code></p>
<p>Also requires the no-LoRA baseline at <code>{{ off_path }}</code> (rendered by <code>python -m poe_repair.experiments.cfg_window_without_lora</code>).</p>
<p>Build the LoRA sweeps with:</p>
<p><code>python -m poe_repair.experiments.cfg_window_with_lora</code></p>
</body></html>
"""


def create_app(
    manifest_path: Path,
    outputs_root: Path,
    cw_manifest_path: Path | None = None,
    cwl_wp_manifest_path: Path | None = None,
    cwl_always_manifest_path: Path | None = None,
    iw_manifest_path: Path | None = None,
    iw_root: Path = DEFAULT_IW_ROOT,
    ix_manifest_path: Path | None = None,
) -> Flask:
    manifest = json.loads(manifest_path.read_text())
    outputs_root_abs = outputs_root.resolve()

    # Both timing tabs render with an empty manifest too, saying what to run.
    iw_manifest: dict = {}
    if iw_manifest_path is not None and Path(iw_manifest_path).is_file():
        iw_manifest = json.loads(Path(iw_manifest_path).read_text())
    ix_manifest: dict = {}
    if ix_manifest_path is not None and Path(ix_manifest_path).is_file():
        ix_manifest = json.loads(Path(ix_manifest_path).read_text())

    cw_manifest: dict | None = None
    if cw_manifest_path is not None and Path(cw_manifest_path).is_file():
        cw_manifest = json.loads(Path(cw_manifest_path).read_text())

    cwl_wp: dict | None = None
    if cwl_wp_manifest_path is not None and Path(cwl_wp_manifest_path).is_file():
        cwl_wp = json.loads(Path(cwl_wp_manifest_path).read_text())
    cwl_al: dict | None = None
    if cwl_always_manifest_path is not None and Path(cwl_always_manifest_path).is_file():
        cwl_al = json.loads(Path(cwl_always_manifest_path).read_text())

    # Discover all pairs with at least one manifest on disk. Cat_dog is
    # always seeded from the CLI-supplied manifests above so the default
    # behaviour for users who never pass --manifest is identical to before.
    pairs = _discover_pairs(outputs_root_abs)
    pairs.setdefault(DEFAULT_PAIR, {})
    pairs[DEFAULT_PAIR]["residual"] = manifest
    pairs[DEFAULT_PAIR]["cw"] = cw_manifest if cw_manifest is not None \
        else pairs[DEFAULT_PAIR].get("cw")
    def _selected_pair() -> str:
        slug = request.args.get("pair", DEFAULT_PAIR)
        return slug if slug in pairs else DEFAULT_PAIR

    def _query_pair(slug: str) -> str:
        return f"?pair={slug}" if slug != DEFAULT_PAIR else ""

    def _tab_ctx(active: str, current_pair: str) -> dict:
        return {
            "active": active,
            "pair_options": _pair_options_for(active, pairs),
            "current_pair": current_pair,
            "query_pair": _query_pair(current_pair),
        }

    app = Flask(__name__)

    @app.route("/")
    def index():
        pair = _selected_pair()
        entry = pairs[pair]
        m = entry.get("residual")
        if m is None:
            return render_template_string(
                _CW_MISSING_HTML,
                path=f"outputs/lora/{pair}/seed_42/results/inspector_manifest.json",
            ), 404
        return render_template_string(
            _with_tabs(INDEX_HTML),
            **_tab_ctx("residual", pair),
            results_root=m.get("results_root", ""),
            epochs=m["epochs"],
            lambdas=m["lambdas"],
            manifest_json=json.dumps(m),
            mono_path=_resolve_mono_path(m, pair),
            joint_prompt=_joint_prompt_for(pair, outputs_root_abs),
        )

    @app.route("/manifest.json")
    def manifest_route():
        pair = _selected_pair()
        m = pairs.get(pair, {}).get("residual")
        if m is None:
            abort(404)
        return jsonify(m)

    @app.route("/mds_large")
    def mds_large():
        pair = _selected_pair()
        entry = pairs[pair]
        m = entry.get("residual")
        if m is None:
            return render_template_string(
                _CW_MISSING_HTML,
                path=f"outputs/lora/{pair}/seed_42/results/inspector_manifest.json",
            ), 404
        mds_large_cells = m.get("mds_cells_large") or {}
        mds_large_cells_sem = m.get("mds_cells_large_semantic") or {}
        n_large = sum(len(v) for v in mds_large_cells.values())
        n_large_sem = sum(len(v) for v in mds_large_cells_sem.values())
        return render_template_string(
            _with_tabs(MDS_LARGE_HTML),
            **_tab_ctx("mds_large", pair),
            results_root=m.get("results_root", ""),
            epochs=m["epochs"],
            lambdas=m["lambdas"],
            n_large_cells=n_large,
            n_large_cells_semantic=n_large_sem,
            pair_display=_display_name(pair),
            manifest_json=json.dumps(m),
        )

    @app.route("/conditioning_window")
    def conditioning_window():
        pair = _selected_pair()
        cwm = pairs[pair].get("cw")
        if cwm is None:
            return render_template_string(
                _CW_MISSING_HTML,
                path=f"{paths.CFG_WINDOW_WITHOUT_LORA}/{pair}/seed_42/results/inspector_manifest.json",
            ), 404
        sanity = cwm.get("sanity") or {}
        on_rec = sanity.get("all_on_vs_run_cfg") or {}
        off_rec = sanity.get("all_off_vs_uncond") or {}
        return render_template_string(
            _with_tabs(CW_INDEX_HTML),
            **_tab_ctx("cw", pair),
            prompt=cwm.get("prompt", ""),
            seed=cwm.get("seed", ""),
            num_steps=cwm.get("num_inference_steps", 50),
            guidance=cwm.get("guidance_scale", ""),
            schedules=cwm["schedules"],
            sanity_on=f"{on_rec.get('max_abs_delta', float('nan')):.2e}"
                      if on_rec else "—",
            sanity_off=f"{off_rec.get('max_abs_delta', float('nan')):.2e}"
                       if off_rec else "—",
            sanity_on_pass="PASS" if on_rec.get("pass") else "FAIL"
                           if on_rec else "—",
            sanity_off_pass="PASS" if off_rec.get("pass") else "FAIL"
                            if off_rec else "—",
            manifest_json=json.dumps(cwm),
        )

    @app.route("/conditioning_window/manifest.json")
    def conditioning_window_manifest():
        pair = _selected_pair()
        cwm = pairs.get(pair, {}).get("cw")
        if cwm is None:
            abort(404)
        return jsonify(cwm)

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
            _with_tabs(CWL_INDEX_HTML),
            **_tab_ctx("cwl", DEFAULT_PAIR),
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

    @app.route("/interaction_window")
    def interaction_window():
        m = iw_manifest or {}
        curve = m.get("curve") or {}
        return render_template_string(
            _with_tabs(WINDOW_INDEX_HTML),
            **_tab_ctx("iw", _selected_pair()),
            width=m.get("width", "—"),
            stride=m.get("stride", "—"),
            num_steps=m.get("num_steps", 50),
            fork_step=m.get("fork_step", "—"),
            n_windows=len(m.get("windows") or []),
            n_on_disk=m.get("n_cells_on_disk", 0),
            n_planned=m.get("n_cells_planned", 0),
            n_scored=m.get("n_cells_scored", 0),
            scorer=curve.get("scorer"),
            manifest_json=json.dumps(m),
        )

    @app.route("/interaction_cross")
    def interaction_cross():
        m = ix_manifest or {}
        return render_template_string(
            _with_tabs(CROSS_INDEX_HTML),
            **_tab_ctx("ix", _selected_pair()),
            num_steps=m.get("num_steps", 50),
            fork_step=m.get("fork_step", "—"),
            width=m.get("width", "—"),
            n_on_disk=m.get("n_cells_on_disk", 0),
            n_planned=m.get("n_cells_planned", 0),
            n_frames=m.get("n_cells_with_frames", 0),
            manifest_json=json.dumps(m),
        )

    @app.route("/interaction_cross/manifest.json")
    def interaction_cross_manifest():
        if not ix_manifest:
            abort(404)
        return jsonify(ix_manifest)

    @app.route("/interaction_window/manifest.json")
    def interaction_window_manifest():
        if not iw_manifest:
            abort(404)
        return jsonify(iw_manifest)

    @app.route("/interaction_window/img")
    def interaction_window_img():
        # The sweep writes to /datasets, which is outside the repo, so the
        # repo-relative /img route cannot reach it. Serve by absolute path
        # instead, and only from inside the sweep's own output root: anything
        # that resolves elsewhere is refused rather than clamped.
        raw = request.args.get("path", "")
        if not raw:
            abort(400)
        target = Path(raw).resolve()
        if not target.is_relative_to(iw_root.resolve()):
            abort(403)
        if not target.is_file():
            abort(404)
        return send_file(target)

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
             "`python -m poe_repair.experiments.cfg_window_without_lora`). "
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
    ap.add_argument(
        "--window-manifest", default=str(DEFAULT_IW_MANIFEST),
        help="correction-timing manifest (produced by "
             "scripts/build_window_manifest.py). If missing, "
             "/interaction_window renders the commands that make it.",
    )
    ap.add_argument("--window-root", default=str(DEFAULT_IW_ROOT),
                    help="only images under this directory are served to the "
                         "timing tabs")
    ap.add_argument(
        "--cross-manifest", default=str(DEFAULT_CROSS_MANIFEST),
        help="crossed-grid manifest (scripts/build_cross_manifest.py). If "
             "missing, /interaction_cross renders the commands that make it.",
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
    iw_path = Path(args.window_manifest) if args.window_manifest else None
    if iw_path is not None and not iw_path.is_file():
        print(
            f"[warn] correction-timing manifest not found: {iw_path}; "
            f"/interaction_window will render a hint page."
        )
    app = create_app(
        manifest_path, Path(args.outputs_root),
        cw_manifest_path=cw_path,
        cwl_wp_manifest_path=cwl_wp_path,
        cwl_always_manifest_path=cwl_al_path,
        iw_manifest_path=iw_path,
        iw_root=Path(args.window_root),
        ix_manifest_path=(Path(args.cross_manifest)
                          if args.cross_manifest else None),
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
