import { Eq } from "./Eq";
import { fmtShape, M } from "./data";
import { useScene } from "./store";

/** The Gaussian head: conv_out's 8 channels split into μ and logvar, the
 *  reparameterised sample, then ×0.13025 before diffusion ever sees z. */
export function VGaussian() {
  const setHover = useScene((s) => s.setHover);
  const co = M["encoder.conv_out"];
  const qc = M["quant_conv"];
  const soft = "var(--soft)", ink = "var(--ink)", lat = "var(--lat)", enc = "var(--enc)", cond = "var(--cond)", dec2 = "var(--dec)";
  return (
    <div className="viz">
      <h2>gaussian head + ×0.13025 <span className="tag real">real shapes</span></h2>
      <div className="stats">
        <span className="stat">encoder.conv_out → <b>{fmtShape(co.shapes?.out)}</b></span>
        <span className="stat">quant_conv 8→8 → split <b>4 + 4</b></span>
        <span className="stat">sampled z <b>{fmtShape(qc.shapes ? [1, 4, 128, 128] : null)}</b></span>
      </div>
      <p className="note">
        The encoder does not output a latent; it outputs a <b>distribution</b>. The 8 channels leaving
        quant_conv are chunked into μ (4) and logvar (4). Sampling uses the reparameterisation below,
        and only after multiplying by <b>scaling_factor 0.13025</b> does the diffusion loop see z.
        That constant normalises latent variance to ~1 so the noise schedule's SNR means what it says.
      </p>
      <div className="figure">
        {/* the operator circuit: boxes are tensors, circles are ops, every arrow carries its term */}
        <svg viewBox="0 0 860 300" role="img" style={{ width: "100%", height: "auto", minWidth: 760 }}>
          <defs>
            <marker id="ga" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L7,4 L0,8 Z" fill="var(--faint)" />
            </marker>
          </defs>
          {/* input tensor */}
          <g data-tok="chunk" onMouseEnter={() => setHover("chunk")} onMouseLeave={() => setHover(null)}>
            <rect x={16} y={120} width={104} height={46} rx={10} fill={`color-mix(in srgb, ${lat} 16%, var(--surface))`} stroke={lat} strokeWidth={1.6} />
            <text x={68} y={140} textAnchor="middle" fontSize={12} className="m" fill={ink} fontWeight={700}>8×128²</text>
            <text x={68} y={156} textAnchor="middle" fontSize={9} className="m" fill={soft}>quant_conv out</text>
          </g>
          {/* chunk into mu / logvar */}
          <path d="M 120 132 Q 150 110 176 78" fill="none" stroke="var(--faint)" strokeWidth={1.6} markerEnd="url(#ga)" />
          <path d="M 120 154 Q 150 176 176 208" fill="none" stroke="var(--faint)" strokeWidth={1.6} markerEnd="url(#ga)" />
          <text x={146} y={94} textAnchor="middle" fontSize={9.5} className="m" fill={soft}>chunk(2)</text>
          <g data-tok="self.mean" onMouseEnter={() => setHover("self.mean")} onMouseLeave={() => setHover(null)}>
            <rect x={180} y={52} width={96} height={40} rx={9} fill={`color-mix(in srgb, ${enc} 10%, var(--surface))`} stroke={enc} strokeWidth={1.6} />
            <text x={228} y={77} textAnchor="middle" fontSize={12} className="m" fill={enc} fontWeight={700}>μ 4×128²</text>
          </g>
          <g data-tok="self.logvar" onMouseEnter={() => setHover("self.logvar")} onMouseLeave={() => setHover(null)}>
            <rect x={180} y={192} width={96} height={40} rx={9} fill={`color-mix(in srgb, ${cond} 10%, var(--surface))`} stroke={cond} strokeWidth={1.6} />
            <text x={228} y={217} textAnchor="middle" fontSize={11.5} className="m" fill={cond} fontWeight={700}>logvar 4×128²</text>
          </g>
          {/* exp(1/2 ·) op */}
          <g data-tok="self.std" onMouseEnter={() => setHover("self.std")} onMouseLeave={() => setHover(null)}>
            <line x1={276} y1={212} x2={330} y2={212} stroke="var(--faint)" strokeWidth={1.6} markerEnd="url(#ga)" />
            <circle cx={352} cy={212} r={21} fill="var(--surface)" stroke={cond} strokeWidth={1.8} />
            <text x={352} y={209} textAnchor="middle" fontSize={9.5} className="m" fill={cond} fontWeight={600}>exp</text>
            <text x={352} y={221} textAnchor="middle" fontSize={9.5} className="m" fill={cond} fontWeight={600}>½·</text>
            <line x1={373} y1={212} x2={430} y2={212} stroke={cond} strokeWidth={1.6} markerEnd="url(#ga)" />
            <text x={402} y={203} textAnchor="middle" fontSize={10.5} className="m" fill={cond} fontWeight={700}>σ</text>
          </g>
          {/* noise input */}
          <g data-tok="randn_tensor" onMouseEnter={() => setHover("randn_tensor")} onMouseLeave={() => setHover(null)}>
            <rect x={404} y={252} width={96} height={38} rx={9} fill={`color-mix(in srgb, ${dec2} 10%, var(--surface))`} stroke={dec2} strokeWidth={1.6} />
            <text x={452} y={272} textAnchor="middle" fontSize={11.5} className="m" fill={dec2} fontWeight={700}>ε ~ N(0, I)</text>
            <text x={452} y={285} textAnchor="middle" fontSize={8.5} className="m" fill={soft}>fresh each sample</text>
            <line x1={452} y1={252} x2={452} y2={234} stroke={dec2} strokeWidth={1.6} markerEnd="url(#ga)" />
          </g>
          {/* multiply op */}
          <circle cx={452} cy={212} r={17} fill="var(--surface)" stroke={ink} strokeWidth={1.6} />
          <text x={452} y={218} textAnchor="middle" fontSize={14} fill={ink} fontWeight={700}>⊗</text>
          <line x1={469} y1={212} x2={540} y2={212} stroke="var(--faint)" strokeWidth={1.6} markerEnd="url(#ga)" />
          <text x={506} y={203} textAnchor="middle" fontSize={10.5} className="m" fill={soft}>σ ⊙ ε</text>
          {/* mu rides across to the add */}
          <path d="M 276 72 Q 430 72 552 196" fill="none" stroke={enc} strokeWidth={1.6} markerEnd="url(#ga)" />
          <text x={430} y={62} textAnchor="middle" fontSize={10.5} className="m" fill={enc} fontWeight={700}>μ</text>
          {/* add op */}
          <circle cx={562} cy={212} r={17} fill="var(--surface)" stroke={ink} strokeWidth={1.6} />
          <text x={562} y={218} textAnchor="middle" fontSize={14} fill={ink} fontWeight={700}>⊕</text>
          <line x1={579} y1={212} x2={632} y2={212} stroke="var(--faint)" strokeWidth={1.6} markerEnd="url(#ga)" />
          {/* z out */}
          <rect x={636} y={190} width={92} height={44} rx={10} fill={`color-mix(in srgb, ${lat} 22%, var(--surface))`} stroke={lat} strokeWidth={2} />
          <text x={682} y={217} textAnchor="middle" fontSize={12.5} className="m" fill={ink} fontWeight={700}>z 4×128²</text>
          {/* scale chip to diffusion */}
          <line x1={728} y1={212} x2={766} y2={212} stroke="var(--faint)" strokeWidth={1.6} markerEnd="url(#ga)" />
          <g data-tok="scaling_factor" onMouseEnter={() => setHover("scaling_factor")} onMouseLeave={() => setHover(null)}>
            <rect x={768} y={192} width={78} height={40} rx={9} fill="var(--panel)" stroke="var(--line2)" />
            <text x={807} y={209} textAnchor="middle" fontSize={10} className="m" fill={ink} fontWeight={600}>× 0.13025</text>
            <text x={807} y={224} textAnchor="middle" fontSize={8.5} className="m" fill={soft}>→ diffusion</text>
          </g>
          <text x={430} y={24} textAnchor="middle" fontSize={11} fill={soft}>boxes are tensors, circles are ops · every arrow carries its term of z = μ + σ⊙ε</text>
        </svg>
      </div>
      <Eq block tex={String.raw`z=\mu+\sigma\odot\varepsilon,\qquad \sigma=e^{\tfrac12\log\sigma^2},\qquad \varepsilon\sim\mathcal N(0,I)`} />
      <p className="note">
        Hover μ, logvar, or the sample line to light the matching lines of
        DiagonalGaussianDistribution in the code pane. In your pipeline the sample is drawn once per
        image at encode time; at generation time nothing is sampled here, the diffusion loop hands
        z₀ straight to post_quant_conv and the decoder.
      </p>
      <p className="drip">
        want the ELBO this head implements derivable and touchable? <code>/drip --math vae-elbo</code>, then <code>/polish</code>, then <code>/math-scene</code>.
      </p>
    </div>
  );
}
