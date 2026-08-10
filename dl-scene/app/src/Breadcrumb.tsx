import { childrenOf, M, nodeTitle, SYNTH } from "./data";
import { useScene } from "./store";

/** The persistent map strip: one row per level of the current path,
 *  siblings dimmed but clickable, the active path lit. */
export function Breadcrumb() {
  const path = useScene((s) => s.path);
  const navigate = useScene((s) => s.navigate);

  const rows: { label: string; nodes: { key: string; label: string; state: "on" | "path" | "off" }[] }[] = [];

  // L0 row: the map itself plus synthetic siblings
  const l1 = [...childrenOf(""), SYNTH.gaussian, SYNTH.tree];
  rows.push({
    label: "L0 model",
    nodes: [{ key: SYNTH.root, label: "map", state: path === SYNTH.root ? "on" : "path" }],
  });
  const seg = path === SYNTH.root ? [] : path === SYNTH.gaussian ? [SYNTH.gaussian] : path.split(".");
  const depth = seg.length;
  for (let lvl = 0; lvl < Math.max(depth, 1); lvl++) {
    const parent = lvl === 0 ? "" : seg.slice(0, lvl).join(".");
    if (lvl > 0 && (parent === SYNTH.gaussian || !M[parent])) break;
    const sibs = lvl === 0 ? l1 : childrenOf(parent);
    if (sibs.length === 0) break;
    const activePrefix = seg.slice(0, lvl + 1).join(".");
    rows.push({
      label: "L" + (lvl + 1),
      nodes: sibs.map((s) => {
        const label = lvl === 0 ? nodeTitle(s).split(" ")[0] || s : s.slice(parent.length ? parent.length + 1 : 0);
        const st: "on" | "path" | "off" =
          s === path ? "on" : s === activePrefix || path.startsWith(s + ".") ? "path" : "off";
        return { key: s, label: s === SYNTH.gaussian ? "gaussian+scale" : s === SYNTH.tree ? "whole tree" : label, state: st };
      }),
    });
  }

  return (
    <div className="strip">
      {rows.map((r) => (
        <div className="crumbrow" key={r.label}>
          <span className="lvl">{r.label}</span>
          {r.nodes.map((n) => (
            <button
              key={n.key}
              className={"nd" + (n.state === "on" ? " on" : n.state === "path" ? " path" : "")}
              onClick={() => navigate(n.key)}
              title={n.key || "map"}
            >
              {n.label}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}
