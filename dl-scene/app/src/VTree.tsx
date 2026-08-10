import { fmtParams, fmtShape, M, TOTAL_PARAMS } from "./data";
import { useScene } from "./store";

const SEC_COLOR: Record<string, string> = {
  encoder: "var(--enc)",
  decoder: "var(--dec)",
  quant_conv: "var(--lat)",
  post_quant_conv: "var(--lat)",
};

function rowColor(path: string): string {
  return SEC_COLOR[path.split(".")[0]] ?? "var(--accent)";
}

/** The deep view: every traced module, in registration order, indented by
 *  depth, with its class, out-shape, and parameter share. All 242 rows. */
export function VTree() {
  const navigate = useScene((s) => s.navigate);
  const names = Object.keys(M);
  const maxLeafP = Math.max(...names.filter((n) => M[n].n_children === 0).map((n) => M[n].params));

  const sections: { title: string; blurb: string; rows: string[] }[] = [
    { title: "encoder", blurb: "3×1024² down to 8×128² (μ and logvar stacked)", rows: names.filter((n) => n === "encoder" || n.startsWith("encoder.")) },
    { title: "latent seam", blurb: "the two 1×1 convs either side of the Gaussian head", rows: ["quant_conv", "post_quant_conv"] },
    { title: "decoder", blurb: "4×128² back up to 3×1024²", rows: names.filter((n) => n === "decoder" || n.startsWith("decoder.")) },
  ];

  return (
    <div className="viz">
      <h2>the whole tree <span className="tag real">real</span></h2>
      <div className="stats">
        <span className="stat"><b>{names.length}</b> modules</span>
        <span className="stat"><b>{fmtParams(TOTAL_PARAMS)}</b> parameters</span>
        <span className="stat">bar = leaf parameter count (log scale, max {fmtParams(maxLeafP)})</span>
      </div>
      <p className="note">
        Every module the trace saw, in registration order, indented by depth: the same tree
        <span className="mono"> named_modules()</span> walks. Containers (ModuleList) carry no shapes of
        their own; leaves show their traced out-shape. Click any row to open its view.
      </p>
      {sections.map((sec) => (
        <div key={sec.title}>
          <div className="tree-sec">{sec.title} <span style={{ fontFamily: "var(--sans)", fontSize: "0.78rem", color: "var(--faint)", fontWeight: 400 }}>{sec.blurb}</span></div>
          <div className="tree">
            {sec.rows.map((n) => {
              const m = M[n];
              const depth = n.split(".").length - 1;
              const isLeaf = m.n_children === 0;
              const frac = isLeaf && m.params > 0 ? Math.max(0.04, Math.log1p(m.params) / Math.log1p(maxLeafP)) : 0;
              return (
                <button key={n} className="tree-row" style={{ paddingLeft: 0.5 + depth * 1.35 + "rem" }} onClick={() => navigate(n)}>
                  <span className="t-name" style={{ color: isLeaf ? "var(--ink)" : rowColor(n), opacity: m.cls === "ModuleList" ? 0.55 : 1 }}>
                    {n.split(".").pop()}
                  </span>
                  <span className="t-cls">{m.cls}</span>
                  {isLeaf && m.params > 0 && (
                    <span className="t-bar"><i style={{ width: frac * 100 + "%", background: rowColor(n) }} /></span>
                  )}
                  <span className="t-shape">
                    {m.shapes ? "→ " + fmtShape(m.shapes.out) : ""}{m.params > 0 ? " · " + fmtParams(m.params) : ""}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
