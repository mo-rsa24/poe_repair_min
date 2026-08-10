import raw from "./trace.json";

export interface NodeInfo {
  cls: string;
  args: Record<string, unknown>;
  params: number;
  shapes: { in: number[] | null; out: number[] | null } | null;
  n_children: number;
}
export interface Snippet {
  file: string;
  start: number;
  label?: string;
  lines: string[];
}
export interface Trace {
  summary: {
    model: string;
    config: Record<string, unknown>;
    input: number[];
    latent_sampled: number[];
    decoded: number[];
    total_params: number;
    n_modules: number;
    provenance: string;
  };
  modules: Record<string, NodeInfo>;
  class_src: Record<string, Snippet | null>;
  call_sites: Snippet[];
  versions: Record<string, string>;
}

export const trace = raw as unknown as Trace;
export const M = trace.modules;
export const TOTAL_PARAMS = trace.summary.total_params;

/** Direct children of a dotted path, in registration order. */
export function childrenOf(path: string): string[] {
  const names = Object.keys(M);
  const prefix = path === "" ? "" : path + ".";
  return names.filter((n) => {
    if (path !== "" && !n.startsWith(prefix)) return false;
    const rest = path === "" ? n : n.slice(prefix.length);
    return rest.length > 0 && !rest.includes(".");
  });
}

export function parentOf(path: string): string {
  const i = path.lastIndexOf(".");
  return i === -1 ? "" : path.slice(0, i);
}

export function fmtShape(s: number[] | null | undefined): string {
  if (!s || !Array.isArray(s) || typeof s[0] !== "number") return "?";
  return s.join("×");
}

export function fmtParams(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

export function paramShare(n: number): string {
  return ((100 * n) / TOTAL_PARAMS).toFixed(n / TOTAL_PARAMS > 0.001 ? 1 : 2) + "%";
}

/** Snippet for a node's class, with special cases for the model root. */
export function snippetFor(cls: string): Snippet | null {
  return (trace.class_src[cls] ?? null) as Snippet | null;
}

/** Encoder / decoder stage lists for the funnels (real module paths). */
export const ENC_STAGES = [
  "encoder.conv_in",
  "encoder.down_blocks.0",
  "encoder.down_blocks.1",
  "encoder.down_blocks.2",
  "encoder.down_blocks.3",
  "encoder.mid_block",
  "encoder.conv_out",
];
export const DEC_STAGES = [
  "decoder.conv_in",
  "decoder.mid_block",
  "decoder.up_blocks.0",
  "decoder.up_blocks.1",
  "decoder.up_blocks.2",
  "decoder.up_blocks.3",
  "decoder.conv_out",
];

/** Synthetic (non-module) routes: the pipeline view and the Gaussian head. */
export const SYNTH = {
  root: "",
  gaussian: "~gaussian",
  tree: "~tree",
};

export function nodeTitle(path: string): string {
  if (path === SYNTH.root) return "the VAE in the v4 pipeline";
  if (path === SYNTH.gaussian) return "DiagonalGaussianDistribution + ×0.13025";
  if (path === SYNTH.tree) return "the whole tree";
  return path;
}

/** Subtree parameter total (containers report recursively already). */
export function nodeParams(path: string): number {
  return M[path]?.params ?? 0;
}
