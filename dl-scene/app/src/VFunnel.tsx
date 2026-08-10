import { childrenOf, DEC_STAGES, ENC_STAGES, fmtParams, fmtShape, M, TOTAL_PARAMS } from "./data";
import { useScene } from "./store";

function faceSize(H: number): number {
  const lo = Math.log(96), hi = Math.log(1100);
  return 34 + 96 * (Math.log(H) - lo) / (hi - lo);
}
function depthSize(C: number): number {
  const lo = Math.log(3), hi = Math.log(560);
  return 7 + 42 * (Math.log(C) - lo) / (hi - lo);
}

function Volume({ x, yBase, f, d, id }: { x: number; yBase: number; f: number; d: number; id: string }) {
  const y = yBase - f;
  return (
    <g>
      <ellipse cx={x + (f + d) / 2} cy={yBase + 7} rx={(f + d) / 2 + 4} ry={6} fill="rgba(60,50,30,0.14)" />
      <path d={`M${x},${y} L${x + d},${y - d} L${x + f + d},${y - d} L${x + f},${y} Z`} fill={`url(#${id}-top)`} strokeWidth={0} />
      <path d={`M${x + f},${y} L${x + f + d},${y - d} L${x + f + d},${y - d + f} L${x + f},${y + f} Z`} fill={`url(#${id}-side)`} strokeWidth={0} />
      <rect x={x} y={y} width={f} height={f} fill={`url(#${id}-face)`} rx={1.5} />
      <path d={`M${x},${y} L${x + d},${y - d}`} stroke="rgba(255,255,255,0.35)" strokeWidth={0.8} />
      <rect x={x} y={y} width={f} height={Math.min(6, f * 0.18)} fill="rgba(255,255,255,0.18)" rx={1.5} />
    </g>
  );
}

function VolumeDefs({ id, col }: { id: string; col: string }) {
  return (
    <defs>
      <linearGradient id={id + "-face"} x1="0" y1="0" x2="0.9" y2="1">
        <stop offset="0" stopColor={`color-mix(in srgb, ${col} 82%, #ffffff)`} />
        <stop offset="0.55" stopColor={col} />
        <stop offset="1" stopColor={`color-mix(in srgb, ${col} 72%, #000000)`} />
      </linearGradient>
      <linearGradient id={id + "-top"} x1="0" y1="1" x2="0.6" y2="0">
        <stop offset="0" stopColor={`color-mix(in srgb, ${col} 55%, #ffffff)`} />
        <stop offset="1" stopColor={`color-mix(in srgb, ${col} 78%, #ffffff)`} />
      </linearGradient>
      <linearGradient id={id + "-side"} x1="0" y1="0" x2="1" y2="0.4">
        <stop offset="0" stopColor={`color-mix(in srgb, ${col} 62%, #000000)`} />
        <stop offset="1" stopColor={`color-mix(in srgb, ${col} 40%, #000000)`} />
      </linearGradient>
    </defs>
  );
}

/** L1: the proportioned volume funnel. Face ∝ spatial (log), depth ∝ channels (log). */
export function VFunnel({ which }: { which: "encoder" | "decoder" }) {
  const navigate = useScene((s) => s.navigate);
  const setHover = useScene((s) => s.setHover);
  const stages = which === "encoder" ? ENC_STAGES : DEC_STAGES;
  const col = which === "encoder" ? "var(--enc)" : "var(--dec)";
  const info = M[which];
  const yBase = 262;
  let x = 48;
  const items = stages.map((p) => {
    const m = M[p];
    const out = m.shapes?.out as number[] | null;
    const C = out?.[1] ?? 0, H = out?.[2] ?? 0;
    const f = faceSize(H), d = depthSize(C);
    // L2 chips: this stage's own execution chain, one level down
    let subs: string[] = [];
    if (m.cls === "DownEncoderBlock2D" || m.cls === "UpDecoderBlock2D") {
      subs = [...childrenOf(p + ".resnets"), ...childrenOf(p + (which === "encoder" ? ".downsamplers" : ".upsamplers"))];
    } else if (m.cls === "UNetMidBlock2D") {
      subs = [p + ".resnets.0", p + ".attentions.0", p + ".resnets.1"];
    }
    subs = subs.filter((s) => M[s]);
    const it = { p, m, C, H, f, d, x, subs };
    x += Math.max(f + d, 96) + 62;
    return it;
  });
  const W = x + 36;
  const maxStageP = Math.max(...items.map((i) => i.m.params));

  return (
    <div className="viz">
      <h2>{which} <span className="tag real">real</span></h2>
      <div className="stats">
        <span className="stat">{info.cls} · <b>{fmtParams(info.params)}</b> ({((100 * info.params) / TOTAL_PARAMS).toFixed(0)}% of the VAE)</span>
        <span className="stat">in <b>{fmtShape(info.shapes?.in)}</b> → out <b>{fmtShape(info.shapes?.out)}</b></span>
      </div>
      <p className="note">
        {which === "encoder"
          ? <>Four stages of halving: 1024 → 512 → 256 → 128, while channels grow 3 → 128 → 256 → 512. The shrinking face and thickening slab ARE the compression. The bar under each volume is its parameter share; capacity concentrates where channels are widest, not where the image is biggest.</>
          : <>The encoder mirrored: channels 512 → 256 → 128 fall away while 128 → 256 → 512 → 1024 doubles back up, via nearest-neighbour upsample + conv at each step. The decoder is heavier than the encoder (extra resnet per block: 3 vs 2).</>}
        {" "}Click any volume to descend into that block.
      </p>
      <div className="figure">
        <svg viewBox={`0 0 ${W} 420`} width={W} height={420} role="img" style={{ minWidth: W }}>
          <VolumeDefs id={which} col={col} />
          {items.map(({ p, m, C, H, f, d, x, subs }) => {
            const short = p.split(".").slice(1).join(".");
            const cw = Math.max(f + d, 96);
            const barW = Math.max(3, cw * (m.params / maxStageP));
            return (
              <g key={p}>
                <g data-tok={short.split(".")[0]} style={{ cursor: "pointer" }}
                  onClick={() => navigate(p)}
                  onMouseEnter={() => setHover(short.split(".")[0])} onMouseLeave={() => setHover(null)}>
                  <Volume x={x + (cw - f - d) / 2} yBase={yBase} f={f} d={d} id={which} />
                  <text x={x + cw / 2} y={yBase + 26} textAnchor="middle" fontSize={11.5} className="m" fill="var(--ink)" fontWeight={600}>{short}</text>
                  <text x={x + cw / 2} y={yBase + 42} textAnchor="middle" fontSize={10} className="m" fill="var(--soft)">{C}×{H}×{H}</text>
                  <rect x={x} y={yBase + 52} width={cw} height={6} rx={3} fill="var(--line)" />
                  <rect x={x} y={yBase + 52} width={barW} height={6} rx={3} fill={col} />
                  <text x={x + cw / 2} y={yBase + 72} textAnchor="middle" fontSize={9.5} className="m" fill="var(--faint)">{fmtParams(m.params)}</text>
                </g>
                {subs.map((s, i) => {
                  const sub = M[s];
                  const label = s.slice(p.length + 1).replace("resnets.", "res ").replace("downsamplers.0", "↓ sample").replace("upsamplers.0", "↑ sample").replace("attentions.0", "attn");
                  const scol = sub.cls === "Attention" ? "var(--cond)" : sub.cls.includes("sample") ? col : "var(--faint)";
                  const stage = short.split(".")[0];
                  const hoverTok = stage === "mid_block" ? "mid_block" : which === "encoder" ? "down_block" : "up_block";
                  return (
                    <g key={s} className="pill" data-tok={hoverTok} style={{ cursor: "pointer" }}
                      onClick={() => navigate(s)}
                      onMouseEnter={() => setHover(hoverTok)} onMouseLeave={() => setHover(null)}>
                      <rect x={x + 4} y={yBase + 84 + i * 24} width={cw - 8} height={19} rx={9.5}
                        fill="var(--surface)" stroke={scol} strokeWidth={1} opacity={0.9} />
                      <text x={x + cw / 2} y={yBase + 84 + i * 24 + 13.5} textAnchor="middle" fontSize={9} className="m" fill="var(--soft)">{label}</text>
                    </g>
                  );
                })}
              </g>
            );
          })}
          <text x={48} y={34} fontSize={11.5} fill="var(--soft)">face ∝ spatial size (log) · slab depth ∝ channels (log) · bar = parameter share of stage (max = {fmtParams(maxStageP)}) · pills = the level below, clickable</text>
        </svg>
      </div>
      <p className="note">
        {which === "encoder"
          ? <>The tail matters: <b>conv_out</b> emits 8 channels, not 4. They are μ (4) and logvar (4); the Gaussian head on the map splits them.</>
          : <>Note <b>up_blocks.0</b> and <b>up_blocks.1</b> both stay at 512 channels; SDXL reverses the encoder's channel list, so the decoder spends most of its 49.5M parameters at 512 wide.</>}
      </p>
    </div>
  );
}
