import { fmtParams, M, nodeParams, SYNTH, TOTAL_PARAMS, trace } from "./data";
import { useScene } from "./store";

/** L0: the Rombach three-lane view, VAE lit, neighbours dimmed. */
export function VOverview() {
  const navigate = useScene((s) => s.navigate);
  const setHover = useScene((s) => s.setHover);
  const encP = nodeParams("encoder");
  const decP = nodeParams("decoder");
  const W = 1000;
  const H = 520;
  const enc = "var(--enc)", lat = "var(--lat)", dec = "var(--dec)", cond = "var(--cond)",
    faint = "var(--faint)", soft = "var(--soft)", ink = "var(--ink)";

  return (
    <div className="viz">
      <h2>the VAE in the v4 pipeline</h2>
      <div className="stats">
        <span className="stat"><b>{trace.summary.model}</b></span>
        <span className="stat">loaded fp16, forced fp32 <b>runtime.py:115</b></span>
        <span className="stat">params <b>{fmtParams(TOTAL_PARAMS)}</b> (encoder {fmtParams(encP)}, decoder {fmtParams(decP)})</span>
        <span className="stat"><span className="tag real">real</span> shape-traced, batch 1, 3×1024×1024, CPU</span>
      </div>
      <p className="note">
        Pixel space on the left, latent space in the middle, conditioning on the right.
        The VAE is the left lane: <b>encoder</b> squeezes 3×1024² to a 4×128² latent (48× fewer numbers),
        the diffusion loop and PoE composition happen entirely in that latent box, and the <b>decoder</b> is
        the only way anything returns to pixels. Click a lit box to descend; dimmed boxes are neighbours outside this map.
      </p>
      <div className="figure">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" style={{ width: "100%", height: "auto", minWidth: 760 }}>
          <defs>
            <marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={faint} />
            </marker>
          </defs>
          {/* lanes: pixel 20..320, latent 340..760, conditioning 780..980 */}
          <rect x={20} y={40} width={300} height={460} rx={14} fill="none" stroke={enc} strokeDasharray="6 4" opacity={0.5} />
          <text x={170} y={66} textAnchor="middle" fontSize={12.5} fill={enc} fontWeight={700}>pixel space</text>
          <rect x={340} y={40} width={420} height={460} rx={14} fill="none" stroke={lat} strokeDasharray="6 4" opacity={0.5} />
          <text x={550} y={66} textAnchor="middle" fontSize={12.5} fill={lat} fontWeight={700}>latent space</text>
          <rect x={780} y={40} width={200} height={460} rx={14} fill="none" stroke={cond} strokeDasharray="6 4" opacity={0.38} />
          <text x={880} y={66} textAnchor="middle" fontSize={12.5} fill={cond} opacity={0.75} fontWeight={700}>conditioning</text>

          {/* row A · encode, centerline y=170 */}
          <clipPath id="xclip"><rect x={48} y={118} width={94} height={64} rx={8} /></clipPath>
          <rect x={40} y={110} width={110} height={118} rx={11} fill="var(--surface)" stroke={ink} />
          <image href={import.meta.env.BASE_URL + "sample.jpg"} x={48} y={118} width={94} height={64}
            preserveAspectRatio="xMidYMid slice" clipPath="url(#xclip)" />
          <rect x={48} y={118} width={94} height={64} rx={8} fill="none" stroke="var(--line2)" />
          <text x={95} y={200} textAnchor="middle" fontSize={11} className="m" fill={ink} fontWeight={700}>x · 3×1024²</text>
          <text x={95} y={216} textAnchor="middle" fontSize={8.5} fill={soft}>real sample: "a cat"</text>
          <line x1={150} y1={170} x2={176} y2={170} stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />

          <g data-tok="self.encoder" onMouseEnter={() => setHover("self.encoder")} onMouseLeave={() => setHover(null)}
            onClick={() => navigate("encoder")} style={{ cursor: "pointer" }}>
            <rect x={180} y={133} width={120} height={74} rx={10} fill={`color-mix(in srgb, ${enc} 22%, var(--surface))`} stroke={enc} strokeWidth={2} />
            <text x={240} y={160} textAnchor="middle" fontSize={13} fill={ink} fontWeight={700}>encoder E</text>
            <text x={240} y={177} textAnchor="middle" fontSize={10} className="m" fill={soft}>{fmtParams(encP)} · {((100 * encP) / TOTAL_PARAMS).toFixed(0)}%</text>
            <text x={240} y={193} textAnchor="middle" fontSize={9.5} fill={soft}>click to descend</text>
          </g>
          <line x1={300} y1={170} x2={320} y2={170} stroke={faint} strokeWidth={2} />
          <g data-tok="quant_conv" onClick={() => navigate("quant_conv")} style={{ cursor: "pointer" }}
            onMouseEnter={() => setHover("quant_conv")} onMouseLeave={() => setHover(null)}>
            <rect x={320} y={156} width={60} height={28} rx={8} fill="var(--surface)" stroke={enc} />
            <text x={350} y={174} textAnchor="middle" fontSize={8.5} className="m" fill={soft}>quant 8→8</text>
          </g>
          <line x1={380} y1={170} x2={396} y2={170} stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />

          <g data-tok="latent_dist" onClick={() => navigate(SYNTH.gaussian)} style={{ cursor: "pointer" }}
            onMouseEnter={() => setHover("latent_dist")} onMouseLeave={() => setHover(null)}>
            <rect x={400} y={133} width={118} height={74} rx={10} fill={`color-mix(in srgb, ${lat} 22%, var(--surface))`} stroke={lat} strokeWidth={2} />
            <text x={459} y={158} textAnchor="middle" fontSize={12} fill={ink} fontWeight={700}>z ~ N(μ, σ²)</text>
            <text x={459} y={175} textAnchor="middle" fontSize={10} className="m" fill={soft}>4×128×128 · ×0.13025</text>
            <text x={459} y={192} textAnchor="middle" fontSize={9.5} fill={soft}>click to descend</text>
          </g>
          <line x1={518} y1={170} x2={566} y2={170} stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />
          <text x={542} y={160} textAnchor="middle" fontSize={9.5} fill={soft}>diffusion</text>
          <rect x={570} y={137} width={100} height={66} rx={10} fill="var(--surface)" stroke={lat} opacity={0.85} />
          <text x={620} y={165} textAnchor="middle" fontSize={12} fill={ink}>z_T</text>
          <text x={620} y={183} textAnchor="middle" fontSize={10} fill={soft}>noised</text>

          {/* row B · the denoise loop unrolled, right to left */}
          <g opacity={0.55}>
            {[0, 1, 2].map((i) => {
              const cx = 360 + i * 124; // cells: 360, 484, 608
              return (
                <g key={"u" + i}>
                  <rect x={cx} y={250} width={88} height={52} rx={9} fill="var(--surface)" stroke={faint} strokeDasharray="5 4" />
                  <text x={cx + 44} y={272} textAnchor="middle" fontSize={10.5} fill={soft} fontWeight={700}>UNet ε_θ</text>
                  <text x={cx + 44} y={288} textAnchor="middle" fontSize={8.5} fill={soft}>same weights</text>
                </g>
              );
            })}
            <line x1={484} y1={276} x2={452} y2={276} stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />
            <text x={590} y={281} textAnchor="middle" fontSize={14} fill={soft}>…</text>
            <text x={550} y={472} textAnchor="middle" fontSize={9.5} fill={soft}>×50 steps (DDIM) · 2.57B params · PoE composes here · outside this map</text>
            <text x={550} y={487} textAnchor="middle" fontSize={9} fill={soft}>(capture_attention.py and metrics.py live on this side)</text>
          </g>
          <path d="M 620 203 Q 652 222 652 246" fill="none" stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />
          <text x={664} y={230} textAnchor="start" fontSize={9.5} className="m" fill={soft}>z_T</text>

          {/* decode: z0 drops out of the leftmost cell, through post, into the decoder */}
          <path d="M 404 302 L 404 372 Q 404 400 378 400" fill="none" stroke={lat} strokeWidth={2} markerEnd="url(#ah)" />
          <text x={414} y={348} textAnchor="start" fontSize={9.5} className="m" fill={lat}>z₀ / 0.13025</text>
          <text x={414} y={362} textAnchor="start" fontSize={9.5} className="m" fill={lat}>→ decode_latents</text>

          {/* row C · decode, centerline y=400 */}
          <g data-tok="post_quant_conv" onClick={() => navigate("post_quant_conv")} style={{ cursor: "pointer" }}
            onMouseEnter={() => setHover("post_quant_conv")} onMouseLeave={() => setHover(null)}>
            <rect x={316} y={386} width={60} height={28} rx={8} fill="var(--surface)" stroke={dec} />
            <text x={346} y={404} textAnchor="middle" fontSize={8.5} className="m" fill={soft}>post 4→4</text>
          </g>
          <line x1={316} y1={400} x2={304} y2={400} stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />
          <g data-tok="self.decoder" onMouseEnter={() => setHover("self.decoder")} onMouseLeave={() => setHover(null)}
            onClick={() => navigate("decoder")} style={{ cursor: "pointer" }}>
            <rect x={180} y={363} width={120} height={74} rx={10} fill={`color-mix(in srgb, ${dec} 22%, var(--surface))`} stroke={dec} strokeWidth={2} />
            <text x={240} y={390} textAnchor="middle" fontSize={13} fill={ink} fontWeight={700}>decoder D</text>
            <text x={240} y={407} textAnchor="middle" fontSize={10} className="m" fill={soft}>{fmtParams(decP)} · {((100 * decP) / TOTAL_PARAMS).toFixed(0)}%</text>
            <text x={240} y={423} textAnchor="middle" fontSize={9.5} fill={soft}>click to descend</text>
          </g>
          <line x1={180} y1={400} x2={156} y2={400} stroke={faint} strokeWidth={2} markerEnd="url(#ah)" />
          <rect x={40} y={363} width={110} height={74} rx={11} fill="var(--surface)" stroke={ink} />
          <text x={95} y={396} textAnchor="middle" fontSize={12.5} fill={ink}>x̃</text>
          <text x={95} y={414} textAnchor="middle" fontSize={10} className="m" fill={soft}>3×1024×1024</text>

          {/* conditioning lane, dimmed */}
          <g opacity={0.45}>
            <rect x={800} y={137} width={160} height={66} rx={10} fill="var(--surface)" stroke={cond} strokeDasharray="5 4" />
            <text x={880} y={164} textAnchor="middle" fontSize={11} fill={soft}>CLIP ×2 → 77×2048</text>
            <text x={880} y={182} textAnchor="middle" fontSize={9.5} fill={soft}>prompts A, B, joint</text>
            <path d="M 800 196 Q 736 232 702 262" fill="none" stroke={cond} strokeWidth={2} markerEnd="url(#ah)" />
            <text x={790} y={234} textAnchor="middle" fontSize={9.5} fill={soft}>cross-attention</text>
          </g>
        </svg>
      </div>
      <div className="legend">
        <span><span className="sw" style={{ background: "var(--enc)" }} /> encode path</span>
        <span><span className="sw" style={{ background: "var(--lat)" }} /> latent</span>
        <span><span className="sw" style={{ background: "var(--dec)" }} /> decode path</span>
        <span><span className="sw" style={{ background: "var(--cond)" }} /> conditioning (dimmed: not in this map)</span>
      </div>
      <p className="note">
        Where your code touches this lane: <b>runtime.py:114-116</b> loads it (and force-upcasts to fp32 because the
        original SDXL VAE overflows in fp16; that is what <span className="mono">force_upcast: true</span> in the config encodes),
        <b> decode_latents</b> (runtime.py:214) is the single exit every sampled image takes,
        <b> sdipc_utils.py:133</b> divides by scaling_factor 0.13025 before decoding, and
        <b> export_vae_activations.py</b> fills this app's real-activation slot. The code pane on the right has all four.
      </p>
      <div className="kids">
        {(["encoder", "decoder", "quant_conv", "post_quant_conv"] as const).map((k) => (
          <button className="kid" key={k} onClick={() => navigate(k)}>
            <div className="k-name">{k}</div>
            <div className="k-meta">{M[k].cls} · {fmtParams(M[k].params)}</div>
          </button>
        ))}
        <button className="kid" onClick={() => navigate(SYNTH.gaussian)}>
          <div className="k-name">gaussian head + ×0.13025</div>
          <div className="k-meta">DiagonalGaussianDistribution</div>
        </button>
        <button className="kid" onClick={() => navigate(SYNTH.tree)}>
          <div className="k-name">the whole tree</div>
          <div className="k-meta">all 242 modules, one page, clickable</div>
        </button>
      </div>
    </div>
  );
}
