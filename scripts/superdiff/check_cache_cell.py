"""Task 1.2 check: the target formed from the saved step files must match the norms the
builder computed on the fly, to fp16 tolerance."""
import json, sys, torch
from pathlib import Path
cell = Path(sys.argv[1]); gs, kappa = 7.5, 0.5
meta = json.load(open(cell / "meta.json")); files = sorted((cell / "residuals").glob("step_*.pt"))
print("step files:", len(files), "| meta steps:", meta["num_inference_steps"], "| timesteps[:3]:", meta["timesteps"][:3], "... last:", meta["timesteps"][-1])
errs = []
for i in (0, 50, 100, 199):
    r = torch.load(files[i], map_location="cpu")
    ea, eb, ej, eu = (r[k].float() for k in ("eps_a_raw", "eps_b_raw", "eps_j_raw", "eps_uncond"))
    eps_m = eu + gs * ((eb - eu) + kappa * (ea - eb)); eps_j = eu + gs * (ej - eu)
    n_cache = (eps_j - eps_m).norm().item(); n_live = meta["r_t_sd_norms"][i]
    errs.append(abs(n_cache - n_live) / n_live)
    print(f"step {i:3d}: ||r_t^SD|| from cache {n_cache:8.3f}  live {n_live:8.3f}  rel err {errs[-1]:.2e}  timestep {r['timestep']}  sigma_t {r['sigma_t']:.3f}")
print("CHECK", "PASS" if max(errs) < 2e-2 else "FAIL", f"(max rel err {max(errs):.2e}, fp16 tolerance 2e-2)")
