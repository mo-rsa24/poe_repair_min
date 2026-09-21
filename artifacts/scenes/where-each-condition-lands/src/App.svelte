<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';
  import TimeSlider from '../vendor/ui/components/TimeSlider.svelte';
  import FigureLegend from '../vendor/ui/components/FigureLegend.svelte';

  // ---- data -----------------------------------------------------------------
  let data: any = null;
  let steps: number[] = [];
  let seeds: number[] = [];
  let conds: string[] = [];

  // ---- state ----------------------------------------------------------------
  let t = 0;                 // 0..1 over the sampler's 50 steps
  let isPlaying = false;
  let visible: Record<string, boolean> = {};
  let focusSeed: number | null = null;   // null = all seeds
  let hovered: { cond: string; seed: number } | null = null;
  let showTrails = false;   // trails read well for one seed and clutter for eight; follow the seed choice unless toggled
  $: showTrails = focusSeed !== null;

  const W = 760, H = 620, M = { top: 20, right: 20, bottom: 52, left: 62 };

  // ---- helpers ----------------------------------------------------------------
  function stepOf(tt: number) { return tt * 50; }

  // position of (cond, seed) at continuous step s, linear between saved frames
  function posAt(cond: string, seed: number, s: number): [number, number] {
    const tr = data.tracks[cond][String(seed)];
    if (s <= steps[0]) return tr[0];
    if (s >= steps[steps.length - 1]) return tr[tr.length - 1];
    let i = 0;
    while (steps[i + 1] < s) i++;
    const a = tr[i], b = tr[i + 1];
    const u = (s - steps[i]) / (steps[i + 1] - steps[i]);
    return [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])];
  }
  function nearestFrame(s: number) {
    let best = steps[0];
    for (const k of steps) if (Math.abs(k - s) < Math.abs(best - s)) best = k;
    return best;
  }
  function framePath(cond: string, seed: number, k: number) {
    return `frames/${cond}/seed_${seed}/step_${String(k).padStart(3, '0')}.png`;
  }

  // ---- scales -----------------------------------------------------------------
  let x = d3.scaleLinear(), y = d3.scaleLinear();
  $: if (data) {
    const all: number[][] = [];
    for (const c of conds) for (const s of seeds) for (const p of data.tracks[c][String(s)]) all.push(p);
    const xe = d3.extent(all, (p) => p[0]) as [number, number];
    const ye = d3.extent(all, (p) => p[1]) as [number, number];
    const pad = 0.06;
    x = d3.scaleLinear().domain([xe[0] - pad, xe[1] + pad]).range([M.left, W - M.right]);
    y = d3.scaleLinear().domain([ye[0] - pad, ye[1] + pad]).range([H - M.bottom, M.top]);
  }

  // ---- derived geometry -------------------------------------------------------
  $: s = stepOf(t);
  $: k = data ? nearestFrame(s) : 0;
  $: hulls = data ? ['solo_a', 'solo_b', 'joint'].map((c) => {
      const pts = seeds.map((sd) => data.tracks[c][String(sd)][steps.length - 1]);
      const hull = d3.polygonHull(pts.map((p) => [x(p[0]), y(p[1])] as [number, number]));
      const cx = d3.mean(pts, (p) => x(p[0]))!, cy = d3.mean(pts, (p) => y(p[1]))!;
      return { cond: c, hull, cx, cy, pts };
    }) : [];
  $: shownSeeds = focusSeed === null ? seeds : [focusSeed];
  $: markers = data ? conds.filter((c) => visible[c]).flatMap((c) => shownSeeds.map((sd) => {
      const p = posAt(c, sd, s);
      return { cond: c, seed: sd, px: x(p[0]), py: y(p[1]) };
    })) : [];
  $: trails = data && showTrails ? conds.filter((c) => visible[c]).flatMap((c) => shownSeeds.map((sd) => {
      const pts: [number, number][] = [];
      for (const kk of steps) { if (kk > s) break; const p = data.tracks[c][String(sd)][steps.indexOf(kk)]; pts.push([x(p[0]), y(p[1])]); }
      const p = posAt(c, sd, s); pts.push([x(p[0]), y(p[1])]);
      return { cond: c, seed: sd, d: d3.line()(pts) ?? '' };
    })) : [];

  // ---- playback ---------------------------------------------------------------
  let raf = 0, last = 0;
  const SECONDS_PER_LOOP = 8;
  function tick(now: number) {
    if (!isPlaying) return;
    if (last) t = (t + (now - last) / 1000 / SECONDS_PER_LOOP) % 1;
    last = now;
    raf = requestAnimationFrame(tick);
  }
  $: if (isPlaying) { last = 0; raf = requestAnimationFrame(tick); } else { cancelAnimationFrame(raf); }

  onMount(async () => {
    data = await (await fetch('data.json')).json();
    steps = data.steps; seeds = data.seeds; conds = Object.keys(data.conditions);
    for (const c of conds) visible[c] = c !== 'lora_1.0';
    focusSeed = null;
  });

  $: legendItems = data ? conds.map((c) => ({
      type: data.conditions[c].kind === 'reference' ? 'circle' : 'square',
      color: data.conditions[c].color, label: data.conditions[c].label })) : [];
  $: thumbCond = hovered?.cond ?? 'poe';
  $: thumbSeeds = focusSeed !== null ? [focusSeed] : (hovered ? [hovered.seed] : []);
</script>

<main>
  <h1>Where each condition lands, step by step</h1>
  <p class="lede">
    Cat × dog, held-out seeds 9 to 16. Each point is one sampling run's running estimate of its finished image,
    embedded with DINOv2 and placed on two axes fixed from the finished renders: left to right is cat to dog,
    up is toward where the joint prompt "a cat and a dog" ends up. Press play or drag the slider to watch the
    50 denoising steps. The correction is a rank-32 LoRA (step 30050) that never trained on this pair.
  </p>

  {#if data}
    <div class="row">
      <svg viewBox="0 0 {W} {H}" width={W} height={H} role="img" aria-label="cloud axes plane">
        <!-- axes -->
        <line x1={x(0)} y1={M.top} x2={x(0)} y2={H - M.bottom} class="axis-line" />
        <line x1={M.left} y1={y(0)} x2={W - M.right} y2={y(0)} class="axis-line" />
        <text x={W / 2} y={H - 14} class="axis-label" text-anchor="middle">which animal: cat (left) to dog (right)</text>
        <text transform="translate(16,{H / 2}) rotate(-90)" class="axis-label" text-anchor="middle">both-ness: toward the joint-prompt cloud</text>
        <text x={M.left + 4} y={y(0) - 6} class="tick">solo midpoint</text>
        <line x1={x(0) - 5} y1={y(data.axes.joint_centroid_both_ness)} x2={x(0) + 5} y2={y(data.axes.joint_centroid_both_ness)} class="axis-line" />
        <text x={x(0) - 8} y={y(data.axes.joint_centroid_both_ness) + 4} class="tick" text-anchor="end">joint centroid</text>

        <!-- reference hulls: endpoints of the three single-prompt conditions -->
        {#each hulls as h}
          {#if h.hull}
            <path d={'M' + h.hull.map((p) => p.join(',')).join('L') + 'Z'} fill={data.conditions[h.cond].color}
                  fill-opacity="0.10" stroke={data.conditions[h.cond].color} stroke-opacity="0.5" />
          {/if}
          <text x={h.cx} y={h.cy} class="hull-label" fill={data.conditions[h.cond].color} text-anchor="middle">{data.conditions[h.cond].label}</text>
        {/each}

        <!-- trails -->
        {#each trails as tr}
          <path d={tr.d} fill="none" stroke={data.conditions[tr.cond].color} stroke-width="1.4"
                stroke-opacity={data.conditions[tr.cond].kind === 'reference' ? 0.25 : 0.6} />
        {/each}

        <!-- moving points -->
        {#each markers as m}
          {#if data.conditions[m.cond].kind === 'reference'}
            <circle cx={m.px} cy={m.py} r="5" fill={data.conditions[m.cond].color} fill-opacity="0.85"
                    role="button" tabindex="-1"
                    onmouseenter={() => (hovered = { cond: m.cond, seed: m.seed })} onmouseleave={() => (hovered = null)} />
          {:else}
            <rect x={m.px - 6} y={m.py - 6} width="12" height="12" fill={data.conditions[m.cond].color}
                  transform={m.cond === 'poe' ? '' : `rotate(45 ${m.px} ${m.py})`} role="button" tabindex="-1"
                  onmouseenter={() => (hovered = { cond: m.cond, seed: m.seed })} onmouseleave={() => (hovered = null)} />
          {/if}
          {#if focusSeed === null}
            <text x={m.px + 8} y={m.py - 7} class="seed">s{m.seed}</text>
          {/if}
        {/each}
      </svg>

      <aside>
        <div class="step">step {Math.round(s)} of 50 <span class="muted">(nearest saved frame: {k})</span></div>

        {#if thumbSeeds.length}
          <div class="thumbs">
            {#each conds.filter((c) => visible[c]) as c}
              {#each thumbSeeds as sd}
                <figure class:hot={hovered && hovered.cond === c && hovered.seed === sd}>
                  <img src={framePath(c, sd, k)} alt="{data.conditions[c].label}, seed {sd}, step {k}" width="150" height="150"
                       style="border-color: {data.conditions[c].color}" />
                  <figcaption style="color: {data.conditions[c].color}">{data.conditions[c].label}<br /><span class="muted">seed {sd}</span></figcaption>
                </figure>
              {/each}
            {/each}
          </div>
        {:else}
          <p class="muted">Hover a point to see what it is at this step, or pick one seed below to see all six conditions side by side.</p>
        {/if}

        <div class="controls">
          <label>seed
            <select bind:value={focusSeed}>
              <option value={null}>all eight</option>
              {#each seeds as sd}<option value={sd}>{sd}</option>{/each}
            </select>
          </label>
          <label><input type="checkbox" bind:checked={showTrails} /> trails</label>
          {#each conds as c}
            <label style="color: {data.conditions[c].color}"><input type="checkbox" bind:checked={visible[c]} /> {data.conditions[c].label}</label>
          {/each}
        </div>
      </aside>
    </div>

    <TimeSlider value={t} {isPlaying} min={0} max={1} step={0.002} timeLabel="denoising" minLabel="step 0" maxLabel="step 50"
                onTogglePlay={() => (isPlaying = !isPlaying)} onInput={(v) => { t = v; }} maxWidth="760px" />
    <div class="legend-row"><FigureLegend items={legendItems} fontSize={14} /></div>

    <details>
      <summary>What this can and cannot say</summary>
      <ul>
        <li>The plane is fixed from the finished renders of the three single-prompt conditions, so the joint cloud sitting on the vertical axis is by construction. The content is where PoE and the corrected runs sit relative to it, and when they get there.</li>
        <li>Each frame is the Tweedie estimate at that step (what the model currently thinks the finished image is), decoded through the VAE. Early frames are blurry by nature.</li>
        <li>Positions between saved frames are straight-line interpolations; the saved steps are {steps.join(', ')}.</li>
        <li>DINOv2 CLS is a global embedding; the validated compose scorer counts instances instead. The thumbnails are the ground truth, the plane is a two-number summary.</li>
        <li>Eight seeds show where things land; they do not support a claim about the shape of the distributions.</li>
        <li>Sampler: {data.sampler.steps} DDIM steps, guidance {data.sampler.guidance}, {data.sampler.size} px, one pinned initial latent per seed shared by all six conditions. Correction checkpoint: <code>{data.correction.checkpoint}</code>.</li>
      </ul>
    </details>
  {:else}
    <p>loading…</p>
  {/if}
</main>

<style>
  main { max-width: 1180px; margin: 0 auto; padding: 16px 20px 40px; font-family: Helvetica, Arial, sans-serif; color: #222; }
  h1 { font-size: 1.5em; margin: 0 0 6px; }
  .lede { font-size: 0.98em; line-height: 1.45; max-width: 900px; margin: 0 0 12px; }
  .row { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
  svg { background: white; border: 1px solid #e5e5e5; flex: 0 0 auto; max-width: 100%; height: auto; }
  aside { flex: 1 1 300px; min-width: 280px; }
  .axis-line { stroke: #cccccc; stroke-width: 1; }
  .axis-label { font-size: 13px; fill: #444; }
  .tick { font-size: 11px; fill: #888; }
  .hull-label { font-size: 13px; font-weight: bold; paint-order: stroke; stroke: white; stroke-width: 4px; }
  .seed { font-size: 10px; fill: #555; }
  .step { font-size: 1.05em; margin-bottom: 8px; }
  .muted { color: #888; font-size: 0.9em; }
  .thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
  figure { margin: 0; }
  figure img { display: block; width: 150px; height: 150px; border: 3px solid; border-radius: 4px; }
  figure.hot img { box-shadow: 0 0 0 3px #ffd54f; }
  figcaption { font-size: 12px; line-height: 1.2; margin-top: 3px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px 16px; margin-top: 14px; font-size: 0.92em; }
  .legend-row { margin-top: 26px; }
  details { margin-top: 18px; font-size: 0.92em; max-width: 900px; }
  details li { margin-bottom: 4px; }
  code { font-size: 0.85em; word-break: break-all; }
</style>
