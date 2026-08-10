import { fmtParams, fmtShape, M, paramShare } from "./data";
import { useScene } from "./store";

/** Conv2d: the kernel stepper with the receptive-field cone. Real k/s/p,
 *  display grid downsampled, the size formula computed with the node's real
 *  numbers and checked. On encoder.conv_in the input grid sits on the real
 *  sample image (the "a cat" anchor render). */
export function VConv({ path }: { path: string }) {
  const setHover = useScene((s) => s.setHover);
  const step = useScene((s) => s.step);
  const setStep = useScene((s) => s.setStep);
  const m = M[path];
  const a = m.args as { k: number[]; s: number[]; p: number[]; cin: number; cout: number };
  const k = a.k[0], s = a.s[0], p = a.p[0];
  const Hin = (m.shapes?.in as number[])?.[2] ?? 0;
  const Hout = (m.shapes?.out as number[])?.[2] ?? 0;
  const prePadded = path.includes("downsamplers"); // Downsample2D pads (0,1,0,1) before its conv
  const formulaOut = Math.floor((Hin + 2 * p - k) / s) + 1;
  const ok = formulaOut === Hout;
  const isImageInput = path === "encoder.conv_in";

  // display grid: real sizes shown as a small grid
  const G = Math.min(8, Hin);
  const Gout = Math.max(1, Math.floor((G + 2 * p - k) / s) + 1);
  const pos = step % (Gout * Gout);
  const wr = Math.floor(pos / Gout), wc = pos % Gout;
  const cell = 26, ox = 24 + p * cell, oy = 20 + p * cell;
  const ox2 = ox + G * cell + 104;
  const oc = Math.max(14, (cell * G) / Gout * 0.9);
  const isOneByOne = k === 1;

  // receptive-field cone: lit input window edge → lit output cell edge
  const wx1 = ox + (wc * s - p) * cell, wy1 = oy + (wr * s - p) * cell;
  const wx2 = wx1 + k * cell, wy2 = wy1 + k * cell;
  const cx1 = ox2 + wc * oc, cy1 = oy + wr * oc;
  const midX = ox + G * cell + 52;

  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <div className="stats">
        <span className="stat">Conv2d <b>k={k} s={s} p={p}</b></span>
        <span className="stat">{a.cin}→{a.cout} ch · <b>{fmtParams(m.params)}</b> ({paramShare(m.params)})</span>
        <span className="stat">in <b>{fmtShape(m.shapes?.in)}</b> → out <b>{fmtShape(m.shapes?.out)}</b></span>
      </div>
      <p className="note">
        {isOneByOne
          ? <>A 1×1 conv touches no neighbours: it is a per-pixel linear map across channels ({a.cin}→{a.cout}). The window below is a single cell; what matters is the channel mix.</>
          : s === 2
            ? <>Stride 2 is the downsampling: the window jumps 2 cells per step, so the output grid has half the positions per axis.{prePadded && <> The input arrives pre-padded (0,1,0,1) by Downsample2D, which is why H is {Hin}, an odd number, and p=0 here.</>}</>
            : <>Stride 1 with p=1 keeps the grid size: each output cell sees a {k}×{k} neighbourhood of the input.</>}
        {" "}The cone ties the lit window to the output cell it produces; drag the slider to walk it.
        {isImageInput && <> The input here is the <b>real sample</b>: your "a cat" anchor render, 3×1024×1024.</>}
      </p>
      <div className="figure">
        <svg viewBox={`0 0 ${ox2 + Gout * oc + 48} ${Math.max(G, 6) * cell + 100}`} width={ox2 + Gout * oc + 48} height={Math.max(G, 6) * cell + 100} role="img">
          {/* receptive-field cone, under everything */}
          {!isOneByOne && (
            <polygon
              points={`${wx2},${Math.max(wy1, oy - p * cell)} ${cx1},${cy1} ${cx1},${cy1 + oc} ${wx2},${Math.min(wy2, oy + (G + p) * cell)}`}
              fill="color-mix(in srgb, var(--accent) 10%, transparent)"
              stroke="color-mix(in srgb, var(--accent) 45%, transparent)"
              strokeWidth={1}
            />
          )}
          {p > 0 && (
            <>
              <rect x={ox - p * cell} y={oy - p * cell} width={(G + 2 * p) * cell - 2} height={(G + 2 * p) * cell - 2} rx={3}
                fill="none" stroke="var(--faint)" strokeWidth={1.2} strokeDasharray="4 3" />
              <text x={ox + (G * cell) / 2} y={oy - p * cell - 6} textAnchor="middle" fontSize={9} fill="var(--faint)">zero-pad ring p={p}</text>
            </>
          )}
          {isImageInput && (
            <image href={import.meta.env.BASE_URL + "sample.jpg"} x={ox} y={oy} width={G * cell - 2} height={G * cell - 2}
              preserveAspectRatio="xMidYMid slice" opacity={0.92} />
          )}
          {Array.from({ length: G * G }).map((_, i) => {
            const r = Math.floor(i / G), c = i % G;
            const inWin = r >= wr * s - p && r < wr * s - p + k && c >= wc * s - p && c < wc * s - p + k;
            return (
              <rect key={i} x={ox + c * cell} y={oy + r * cell} width={cell - 2} height={cell - 2} rx={2}
                fill={inWin ? (isImageInput ? "color-mix(in srgb, var(--accent) 26%, transparent)" : "color-mix(in srgb, var(--enc) 42%, var(--surface))") : isImageInput ? "transparent" : "var(--panel)"}
                stroke={inWin ? "var(--accent)" : isImageInput ? "rgba(255,255,255,0.35)" : "var(--line)"} strokeWidth={inWin ? 2 : 1} />
            );
          })}
          <text x={ox + (G * cell) / 2} y={oy + G * cell + 18} textAnchor="middle" fontSize={10.5} className="m" fill="var(--ink)" fontWeight={700}>{Hin}×{Hin}</text>
          <text x={ox + (G * cell) / 2} y={oy + G * cell + 32} textAnchor="middle" fontSize={8.5} fill="var(--faint)">
            {isImageInput ? `real sample, shown as ${G}×${G}` : `shown as ${G}×${G}`}
          </text>
          <g>
            <circle cx={midX} cy={oy + (G * cell) / 2 - 20} r={15} fill="var(--surface)" stroke="var(--cond)" strokeWidth={1.5} />
            <text x={midX} y={oy + (G * cell) / 2 - 14} textAnchor="middle" fontSize={14} fill="var(--cond)">✳</text>
            <text x={midX} y={oy + (G * cell) / 2 + 6} textAnchor="middle" fontSize={9.5} className="m" fill="var(--cond)">k={k} s={s}</text>
            {p > 0 && <text x={midX} y={oy + (G * cell) / 2 + 19} textAnchor="middle" fontSize={9.5} className="m" fill="var(--cond)">p={p}</text>}
          </g>
          {Array.from({ length: Gout * Gout }).map((_, i) => {
            const r = Math.floor(i / Gout), c = i % Gout;
            const lit = r === wr && c === wc;
            return (
              <rect key={"o" + i} x={ox2 + c * oc} y={oy + r * oc} width={oc - 2} height={oc - 2} rx={2}
                fill={lit ? "var(--dec)" : "var(--panel)"} stroke={lit ? "var(--dec)" : "var(--line)"} strokeWidth={lit ? 2 : 1} />
            );
          })}
          <text x={ox2 + (Gout * oc) / 2} y={oy + G * cell + 18} textAnchor="middle" fontSize={10.5} className="m" fill="var(--dec)" fontWeight={700}>{Hout}×{Hout}</text>
          <text x={ox2 + (Gout * oc) / 2} y={oy + G * cell + 32} textAnchor="middle" fontSize={8.5} fill="var(--faint)">shown as {Gout}×{Gout}</text>
          <text x={ox - p * cell} y={oy + G * cell + 56} fontSize={9.5} className="m" fill="var(--faint)">× {a.cin} input channels summed per output cell · {a.cout} filters → {a.cout} output channels</text>
        </svg>
      </div>
      <div className="ctl" onMouseEnter={() => setHover("F.conv2d")} onMouseLeave={() => setHover(null)}>
        <label>window</label>
        <input type="range" min={0} max={Gout * Gout - 1} value={pos} onChange={(e) => setStep(Number(e.target.value))} />
        <span className="readout">output cell ({wr},{wc})</span>
      </div>
      <div className="formula">
        out = ⌊(H + 2p − k)/s⌋ + 1 = ⌊({Hin} + {2 * p} − {k})/{s}⌋ + 1 = {formulaOut}{" "}
        {ok ? <span className="ok">✓ matches traced {Hout}</span> : <span className="no">✗ traced {Hout}</span>}
      </div>
      <p className="note mono" style={{ fontSize: "0.78rem" }}>
        params = k·k·cin·cout + cout = {k}·{k}·{a.cin}·{a.cout} + {a.cout} = {(k * k * a.cin * a.cout + a.cout).toLocaleString()} {k * k * a.cin * a.cout + a.cout === m.params ? "✓" : "✗"}
      </p>
    </div>
  );
}
