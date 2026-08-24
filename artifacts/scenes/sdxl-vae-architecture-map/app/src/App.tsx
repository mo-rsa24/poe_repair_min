import { useEffect } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Breadcrumb } from "./Breadcrumb";
import { CodePane } from "./CodePane";
import { M, SYNTH, trace } from "./data";
import { useScene } from "./store";
import { VOverview } from "./VOverview";
import { VFunnel } from "./VFunnel";
import { VGaussian } from "./VGaussian";
import { VBlock } from "./VBlock";
import { VResBlock } from "./VResBlock";
import { VConv } from "./VConv";
import { VAttention } from "./VAttention";
import { VDropout, VGroupNorm, VLinear, VSiLU } from "./VLeaf";
import { VTree } from "./VTree";

function View({ path }: { path: string }) {
  if (path === SYNTH.root) return <VOverview />;
  if (path === SYNTH.gaussian) return <VGaussian />;
  if (path === SYNTH.tree) return <VTree />;
  const m = M[path];
  if (!m) return <VOverview />;
  if (path === "encoder") return <VFunnel which="encoder" />;
  if (path === "decoder") return <VFunnel which="decoder" />;
  switch (m.cls) {
    case "ResnetBlock2D": return <VResBlock path={path} />;
    case "Attention": return <VAttention path={path} />;
    case "Conv2d": return <VConv path={path} />;
    case "GroupNorm": return <VGroupNorm path={path} />;
    case "SiLU": return <VSiLU path={path} />;
    case "Dropout": return <VDropout path={path} />;
    case "Linear": return <VLinear path={path} />;
    default: return <VBlock path={path} />; // blocks, samplers, ModuleLists
  }
}

export default function App() {
  const path = useScene((s) => s.path);
  const setActivations = useScene((s) => s.setActivations);
  const activations = useScene((s) => s.activations);

  // the real-data slot: drop vae_act.json (from export_vae_activations.py) into app/public/
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + "vae_act.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => j && setActivations(j))
      .catch(() => {});
  }, [setActivations]);

  return (
    <div className="shell">
      <div className="header">
        <div className="eyebrow">poe_repair_min · dl-scene · shape-traced SDXL VAE</div>
        <h1>The v4 pipeline's VAE, level by level</h1>
        <p className="sub">
          {trace.summary.model} · {trace.summary.n_modules} modules · all shapes <span className="tag real">real</span> (traced, batch 1, 3×1024×1024) ·
          diffusers {trace.versions.diffusers} · {activations ? "real activations loaded" : "activation slot empty (run export_vae_activations.py, drop vae_act.json in app/public/)"}
        </p>
      </div>
      <Breadcrumb />
      <div className="stage">
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.div
            key={path || "~root"}
            initial={{ opacity: 0, y: 10, filter: "blur(3px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -8, filter: "blur(3px)" }}
            transition={{ duration: 0.24, ease: [0.25, 0.1, 0.25, 1] }}
            style={{ minWidth: 0 }}
          >
            <View path={path} />
          </motion.div>
        </AnimatePresence>
        <CodePane />
      </div>
      <p className="foot">
        hover answers "what is this" (and lights the code line) · click answers "show me" (descends one level) ·
        traced on CPU from the locally cached config, no weights loaded, GPUs untouched
      </p>
    </div>
  );
}
