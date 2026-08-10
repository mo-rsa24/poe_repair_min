import { useMemo, useState, type ReactNode } from "react";
import { M, SYNTH, snippetFor, trace, type Snippet } from "./data";
import { useScene } from "./store";

/* Quiet python tokenizer: keywords, strings, comments, numbers, self. */
const PY_KW =
  /\b(def|class|return|if|elif|else|for|while|with|as|import|from|in|not|and|or|is|None|True|False|lambda|yield|raise|try|except|pass|assert|global|del)\b/;
const TOKEN_RE = new RegExp(
  ["(#.*$)", "(\"\"\"[\\s\\S]*?\"\"\"|'[^']*'|\"[^\"]*\")", PY_KW.source, "\\b(self)\\b", "\\b(\\d[\\d_.e]*)\\b"].join("|"),
  "gm"
);

/* Route badges: mark branch points in the shown forward code as taken /
   not taken / build-time, with the reason computed from the traced run
   (batch 1, no_grad, 1024², defaults) rather than hand-typed per line. */
interface Badge {
  kind: "taken" | "skip" | "build";
  why: string;
}
function nearestResBlock(path: string): string | null {
  const seg = path.split(".");
  for (let i = seg.length; i > 0; i--) {
    const p = seg.slice(0, i).join(".");
    if (M[p]?.cls === "ResnetBlock2D") return p;
  }
  return null;
}
function branchBadge(line: string, path: string): Badge | null {
  const t = line.trim();
  if (!(t.startsWith("if ") || t.startsWith("elif "))) return null;
  if (t.includes("torch.is_grad_enabled()") || t.includes("gradient_checkpointing"))
    return { kind: "skip", why: "no_grad inference, checkpointing off" };
  if (t.includes("self.use_tiling"))
    return { kind: "skip", why: "use_tiling=False (1024 ≤ tile_sample_min_size)" };
  if (t.includes("self.use_slicing")) return { kind: "skip", why: "use_slicing=False, batch 1" };
  if (t.includes("not return_dict")) return { kind: "skip", why: "return_dict=True" };
  if (t.includes("self.quant_conv is not None")) return { kind: "taken", why: "use_quant_conv=True" };
  if (t.includes("self.post_quant_conv is not None")) return { kind: "taken", why: "use_post_quant_conv=True" };
  if (t.includes("self.use_conv_transpose")) return { kind: "build", why: "False: nearest+conv route built" };
  if (t.includes("self.use_conv")) return { kind: "taken", why: "use_conv=True" };
  if (t.includes("self.padding == 0")) return { kind: "taken", why: "padding=0: manual (0,1,0,1) pad" };
  if (t.includes("latent_embeds is None")) return { kind: "taken", why: "latent_embeds=None in the VAE" };
  if (t.includes("temb is not None") || t.includes("temb_channels"))
    return { kind: "skip", why: "temb=None in the VAE" };
  if (t.includes("time_embedding_norm")) return { kind: "build", why: "norm_type=group here" };
  if (t.includes("self.deterministic")) return { kind: "build", why: "deterministic=False" };
  if (t.includes("self.upsample") || t.includes("self.downsample"))
    return { kind: "skip", why: "up/down=False in these resnets" };
  if (t.includes("self.conv_shortcut is not None")) {
    const rb = nearestResBlock(path);
    if (rb) {
      const has = Object.keys(M).some((n) => n === rb + ".conv_shortcut");
      return has
        ? { kind: "taken", why: "channels change: 1×1 shortcut" }
        : { kind: "skip", why: "channels equal: identity skip" };
    }
    return { kind: "build", why: "per block: only where channels change" };
  }
  return null;
}

function tokenize(line: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  TOKEN_RE.lastIndex = 0;
  let key = 0;
  while ((m = TOKEN_RE.exec(line)) !== null) {
    if (m.index > last) out.push(line.slice(last, m.index));
    const [text] = m;
    const cls = m[1] ? "tk-com" : m[2] ? "tk-str" : m[4] === "self" ? "tk-self" : m[5] ? "tk-num" : "tk-kw";
    out.push(
      <span key={key++} className={cls}>
        {text}
      </span>
    );
    last = m.index + text.length;
  }
  if (last < line.length) out.push(line.slice(last));
  return out;
}

interface TabbedSnippet extends Snippet {
  tab: string;
  provenance: "real" | "repo";
}

/** Which snippets a node's pane offers, first tab active by default. */
function snippetsFor(path: string): TabbedSnippet[] {
  const out: TabbedSnippet[] = [];
  const push = (s: Snippet | null, tab: string, provenance: "real" | "repo") => {
    if (s) out.push({ ...s, tab, provenance });
  };
  if (path === SYNTH.root) {
    for (const cs of trace.call_sites) push(cs, cs.file.split("/").pop() ?? cs.file, "repo");
    push(snippetFor("AutoencoderKL.encode"), "encode()", "real");
    push(snippetFor("AutoencoderKL.decode"), "decode()", "real");
    return out;
  }
  if (path === SYNTH.gaussian) {
    push(snippetFor("DiagonalGaussianDistribution"), "DiagonalGaussianDistribution", "real");
    const scale = trace.call_sites.find((c) => c.file.includes("sdipc_utils"));
    push(scale ?? null, "scaling call site", "repo");
    return out;
  }
  const info = M[path];
  if (!info) return out;
  if (path === "encoder") push(snippetFor("Encoder"), "Encoder.forward", "real");
  else if (path === "decoder") push(snippetFor("Decoder"), "Decoder.forward", "real");
  else push(snippetFor(info.cls), info.cls, "real");
  if (info.cls === "Attention") push(snippetFor("AttnProcessor2_0"), "AttnProcessor2_0", "real");
  // parent context tab so a leaf still shows the chain that calls it
  const seg = path.split(".");
  for (let i = seg.length - 1; i > 0; i--) {
    const p = seg.slice(0, i).join(".");
    const pc = M[p]?.cls;
    if (pc && pc !== "ModuleList" && snippetFor(pc) && pc !== info.cls) {
      push(snippetFor(pc), pc + " (caller)", "real");
      break;
    }
  }
  return out;
}

export function CodePane() {
  const path = useScene((s) => s.path);
  const hoverToken = useScene((s) => s.hoverToken);
  const snips = useMemo(() => snippetsFor(path), [path]);
  const [tab, setTab] = useState(0);
  const active = snips[Math.min(tab, Math.max(snips.length - 1, 0))];
  if (!active) {
    return (
      <div className="codepane">
        <div className="codebar">
          <span className="f">no code available for this node</span>
        </div>
      </div>
    );
  }
  const short = active.file.includes("site-packages")
    ? active.file.replace(/^.*site-packages\//, "")
    : active.file;
  return (
    <div className="codepane">
      {snips.length > 1 && (
        <div className="codeswitch">
          {snips.map((s, i) => (
            <button key={s.tab + i} aria-pressed={i === tab} onClick={() => setTab(i)}>
              {s.tab}
            </button>
          ))}
        </div>
      )}
      <div className="codebar">
        <span className="f">
          {short}:{active.start}
        </span>
        <span className={"tag " + (active.provenance === "repo" ? "repo" : "real")}>
          {active.provenance === "repo" ? "your repo" : "real (diffusers " + trace.versions.diffusers + ")"}
        </span>
      </div>
      <pre className="code">
        {active.lines.map((l, i) => {
          const hl = !!hoverToken && l.includes(hoverToken);
          const badge = branchBadge(l, path);
          return (
            <span key={i} className={"ln" + (hl ? " hl" : "")}>
              <span className="no">{active.start + i}</span>
              {l ? tokenize(l) : " "}
              {badge && <span className={"branch b-" + badge.kind}>{badge.kind === "taken" ? "✓ taken" : badge.kind === "skip" ? "✗ not taken" : "◆ build-time"} · {badge.why}</span>}
            </span>
          );
        })}
      </pre>
    </div>
  );
}
