"""Flask inspector for the m5 LoRA SDXL residual training chain.

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
DEFAULT_MANIFEST = REPO_ROOT / "outputs/lora/a_cat__x__a_dog/seed_42/inspector_manifest.json"
DEFAULT_CFG_SCHEDULE_MANIFEST = REPO_ROOT / "outputs/cfg_schedule_ablation_no_lora/seed_42/inspector_manifest.json"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>m5 residual inspector</title>
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
<h1>m5 residual inspector — {{ pair_slug }} · seed {{ seed }}</h1>

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
    <h2>Mono (oracle reference)</h2>
    {% if mono_path %}
    <img id="img-mono" src="/img/{{ mono_path }}" alt="mono oracle">
    <div class="caption">epoch-independent · DDIM 50 steps · same seed/prompt</div>
    {% else %}
    <div class="caption">no mono.png in manifest</div>
    {% endif %}
  </div>
</div>

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


CFG_SCHEDULE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CFG-schedule inspector</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --text: #e6e9ef;
    --muted: #8b93a7;
    --accent: #7ab7ff;
    --on: #7ab7ff;
    --off: #2a2f3a;
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
  h1 .prompt { color: var(--text); }
  .controls {
    display: grid; gap: 14px; margin-bottom: 20px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }
  .row { display: flex; align-items: center; gap: 16px; }
  .row label { flex: 0 0 100px; color: var(--muted); }
  .row input[type=range] { flex: 1; }
  .row .val {
    flex: 0 0 80px; text-align: right; font-variant-numeric: tabular-nums;
    color: var(--accent); font-weight: 600;
  }
  .row .meta { color: var(--muted); font-size: 12px; flex: 0 0 auto; }
  .layout {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 520px); gap: 20px;
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
    font-variant-numeric: tabular-nums;
  }
  .mask-strip {
    display: grid; grid-template-columns: repeat(50, 1fr);
    gap: 2px; padding: 8px 0;
    cursor: help;
  }
  .mask-strip .cell {
    aspect-ratio: 1 / 1;
    background: var(--off);
    border-radius: 2px;
  }
  .mask-strip .cell.on { background: var(--on); }
  .mask-strip .cell.boundary { box-shadow: inset 0 -2px 0 0 var(--border); }
  .mask-legend {
    display: flex; gap: 12px; font-size: 11px; color: var(--muted);
    margin-top: 8px; align-items: center;
  }
  .mask-legend .sw { display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; vertical-align: middle; margin-right: 4px; }
  .mask-legend .sw.on { background: var(--on); }
  .mask-legend .sw.off { background: var(--off); }
  .mask-string {
    font: 11px/1.3 ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--muted); word-break: break-all;
    margin-top: 8px; padding: 6px 8px;
    background: #0b0d12; border: 1px solid var(--border); border-radius: 4px;
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
<h1>CFG-schedule inspector · seed {{ seed }} · prompt
  &ldquo;<span class="prompt">{{ prompt }}</span>&rdquo;</h1>

<div class="controls">
  <div class="row">
    <label for="sched">schedule</label>
    <input id="sched" type="range" min="0" max="{{ num_schedules - 1 }}" step="1" value="{{ num_schedules - 1 }}">
    <div class="val" id="sched-val"></div>
    <div class="meta" id="sched-meta"></div>
  </div>
</div>

<div class="layout">
  <div class="panel">
    <h2>image</h2>
    <img id="img" alt="cfg-schedule sample">
    <div class="caption" id="img-caption"></div>
  </div>
  <div class="panel">
    <h2>per-step CFG mask (step 0 = noisy → step 49 = clean)</h2>
    <div id="mask-strip" class="mask-strip" title=""></div>
    <div class="mask-legend">
      <span><span class="sw on"></span>conditional ON (ε_uncond + w·(ε_c − ε_uncond))</span>
      <span><span class="sw off"></span>conditional OFF (ε = ε_uncond)</span>
    </div>
    <div class="mask-string" id="mask-string"></div>
  </div>
</div>

<div id="toast">no schedule</div>

<script>
const MANIFEST = {{ manifest_json|safe }};
const SCHEDULES = MANIFEST.schedules;     // ordered list
const NUM_STEPS = MANIFEST.num_inference_steps || 50;

const schedSlider = document.getElementById('sched');
const schedVal = document.getElementById('sched-val');
const schedMeta = document.getElementById('sched-meta');
const img = document.getElementById('img');
const imgCaption = document.getElementById('img-caption');
const maskStrip = document.getElementById('mask-strip');
const maskString = document.getElementById('mask-string');
const toast = document.getElementById('toast');

let toastTimer = null;
function flashToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
}

function buildStrip() {
  // Build NUM_STEPS empty cells once; we'll just toggle .on on update.
  maskStrip.innerHTML = '';
  for (let i = 0; i < NUM_STEPS; i++) {
    const c = document.createElement('div');
    c.className = 'cell';
    if ((i + 1) % 10 === 0 && i < NUM_STEPS - 1) c.classList.add('boundary');
    maskStrip.appendChild(c);
  }
}

function update() {
  const idx = parseInt(schedSlider.value, 10);
  const entry = SCHEDULES[idx];
  if (!entry) { flashToast('no schedule at index ' + idx); return; }

  schedVal.textContent = entry.schedule_id;
  schedMeta.textContent = 'k = ' + entry.k + ' · on for ' + entry.num_on +
    '/' + NUM_STEPS + ' steps' + (entry.sanity ? ' · sanity' : '');

  img.src = '/img/' + entry.image_path;
  imgCaption.textContent = entry.image_path;

  // Mask strip
  const mask = entry.mask || '';
  const cells = maskStrip.children;
  for (let i = 0; i < cells.length; i++) {
    if (mask.charAt(i) === '1') cells[i].classList.add('on');
    else cells[i].classList.remove('on');
  }
  maskStrip.title = mask;
  maskString.textContent = mask;
}

buildStrip();
schedSlider.addEventListener('input', update);
update();
</script>
</body>
</html>
"""


_CFG_MISSING_HTML = r"""
<!doctype html>
<html><head><meta charset="utf-8"><title>cfg-schedule manifest missing</title>
<style>body{font:14px -apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e9ef;padding:24px;}
code{background:#161a22;padding:2px 6px;border-radius:4px;color:#7ab7ff;}</style>
</head><body>
<h2>CFG-schedule manifest not found</h2>
<p>Expected at:</p>
<p><code>{{ path }}</code></p>
<p>Build it by running:</p>
<p><code>python scripts/cfg_schedule_no_lora_seed42.py</code></p>
</body></html>
"""


def create_app(
    manifest_path: Path,
    outputs_root: Path,
    cfg_schedule_manifest_path: Path | None = None,
) -> Flask:
    manifest = json.loads(manifest_path.read_text())
    outputs_root_abs = outputs_root.resolve()

    cfg_schedule_manifest: dict | None = None
    if cfg_schedule_manifest_path is not None and cfg_schedule_manifest_path.is_file():
        cfg_schedule_manifest = json.loads(cfg_schedule_manifest_path.read_text())

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(
            INDEX_HTML,
            pair_slug=manifest["pair_slug"],
            seed=manifest["seed"],
            epochs=manifest["epochs"],
            lambdas=manifest["lambdas"],
            mono_path=manifest.get("mono_path"),
            manifest_json=json.dumps(manifest),
        )

    @app.route("/manifest.json")
    def manifest_route():
        return jsonify(manifest)

    @app.route("/cfg_schedule")
    def cfg_schedule():
        if cfg_schedule_manifest is None:
            return render_template_string(
                _CFG_MISSING_HTML,
                path=str(cfg_schedule_manifest_path)
                if cfg_schedule_manifest_path else "(no path configured)",
            ), 404
        return render_template_string(
            CFG_SCHEDULE_HTML,
            prompt=cfg_schedule_manifest.get("prompt", ""),
            seed=cfg_schedule_manifest.get("seed", ""),
            num_schedules=len(cfg_schedule_manifest.get("schedules") or []),
            manifest_json=json.dumps(cfg_schedule_manifest),
        )

    @app.route("/cfg_schedule/manifest.json")
    def cfg_schedule_manifest_route():
        if cfg_schedule_manifest is None:
            abort(404)
        return jsonify(cfg_schedule_manifest)

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
        "--cfg-schedule-manifest", default=str(DEFAULT_CFG_SCHEDULE_MANIFEST),
        help="Manifest produced by scripts/cfg_schedule_no_lora_seed42.py. "
        "If missing, /cfg_schedule renders a friendly hint page.",
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
    cfg_sched_path = Path(args.cfg_schedule_manifest) if args.cfg_schedule_manifest else None
    app = create_app(
        manifest_path,
        Path(args.outputs_root),
        cfg_schedule_manifest_path=cfg_sched_path,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
