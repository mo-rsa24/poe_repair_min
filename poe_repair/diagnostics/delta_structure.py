"""Structure-or-noise diagnostics for the oracle mono per-step correction Δ_t.

Pure linear algebra and matplotlib — no diffusion model dependencies.

Five prongs, each consuming a ``tensors.pt`` produced by the
``delta_structure`` experiment orchestrator and emitting a dict for
``results.json`` plus one matplotlib axis for the composite figure.

Conventions:
    delta:    Tensor[S, T, C, H, W]   # ε̃_mono - ε̃_poe at PoE state, guided
    eps_poe:  Tensor[S, T, C, H, W]
    eps_mono: Tensor[S, T, C, H, W]

Tensors are loaded as float32; storage on disk is float16.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _flatten(delta: torch.Tensor) -> torch.Tensor:
    """(S, T, C, H, W) → (S, T, D)."""
    s, t = delta.shape[:2]
    return delta.reshape(s, t, -1)


def _per_seed_svd(delta_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-seed economy SVD.

    delta_flat: (S, T, D)
    Returns:
        S_vals: (S, T) singular values, descending
        Vh: (S, T, D) right singular vectors (rows are top-T basis of seed)
    """
    S_, T_, _ = delta_flat.shape
    s_list, vh_list = [], []
    for s in range(S_):
        u, sv, vh = torch.linalg.svd(delta_flat[s].float(), full_matrices=False)
        s_list.append(sv)
        vh_list.append(vh)
    return torch.stack(s_list, dim=0), torch.stack(vh_list, dim=0)


# ---------------------------------------------------------------------------
# Prong A — per-seed time compressibility
# ---------------------------------------------------------------------------


def prong_a_time_compressibility(
    delta: torch.Tensor,
    ax: plt.Axes | None = None,
) -> dict[str, Any]:
    """Per-seed singular-value spectrum → effective rank + k90/k95/k99.

    Mathematical content:
        For each seed s, SVD of Δ_s ∈ R^(T×D). Effective rank is the
        participation ratio (Σ σ_i^2)^2 / Σ σ_i^4. ``k_p`` is the smallest
        k such that the top-k captures fraction p of total energy
        (Σ σ_i^2).
    """
    delta_flat = _flatten(delta)
    S_vals, _ = _per_seed_svd(delta_flat)
    energies = S_vals ** 2

    total = energies.sum(dim=1, keepdim=True)
    cum_frac = torch.cumsum(energies, dim=1) / torch.clamp(total, min=1e-30)

    def first_k_at(p: float) -> list[int]:
        ks = []
        for s in range(cum_frac.shape[0]):
            idx = (cum_frac[s] >= p).nonzero(as_tuple=True)[0]
            ks.append(int(idx[0].item()) + 1 if len(idx) > 0 else int(cum_frac.shape[1]))
        return ks

    eff_rank = ((energies.sum(dim=1) ** 2) / torch.clamp((energies ** 2).sum(dim=1), min=1e-30)).tolist()
    k90 = first_k_at(0.90)
    k95 = first_k_at(0.95)
    k99 = first_k_at(0.99)

    if all(k <= 5 for k in k90):
        verdict = "compressible"
    elif any(k > 20 for k in k90):
        verdict = "not_compressible"
    else:
        verdict = "marginal"

    if ax is not None:
        x = np.arange(1, cum_frac.shape[1] + 1)
        for s in range(cum_frac.shape[0]):
            ax.plot(x, cum_frac[s].numpy(), label=f"seed idx {s}", alpha=0.8)
        ax.axhline(0.90, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("k (top-k singular vectors)")
        ax.set_ylabel("cumulative energy fraction")
        ax.set_title(f"A. time compressibility — {verdict}")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(1, cum_frac.shape[1])
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    return {
        "effective_rank_per_seed": eff_rank,
        "k90_per_seed": k90,
        "k95_per_seed": k95,
        "k99_per_seed": k99,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Prong B — cross-seed alignment with isotropic null
# ---------------------------------------------------------------------------


def _principal_angles_deg(Va: torch.Tensor, Vb: torch.Tensor) -> torch.Tensor:
    """Principal angles between row-spaces of Va and Vb (both k × D, orthonormal rows).

    Returns angles in degrees, ascending, shape (k,).
    """
    # Cosines of principal angles are singular values of Va @ Vb.T  (k × k).
    m = Va @ Vb.T
    sv = torch.linalg.svdvals(m).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(sv))
    return torch.sort(angles).values


def prong_b_cross_seed_alignment(
    delta: torch.Tensor,
    k: int | None = None,
    n_null: int = 100,
    ax_matrix: plt.Axes | None = None,
    ax_null: plt.Axes | None = None,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Cross-seed principal-angle test with isotropic-Gaussian null.

    Mathematical content:
        For each seed: top-k right singular vectors V_s ∈ R^(k×D) (orthonormal rows).
        Principal angles between V_s and V_{s'} are arccos of the singular
        values of V_s V_{s'}^T. We report the per-pair mean principal angle
        (in degrees) and compare the across-pair average to a null formed
        by replacing each Δ_s with an isotropic Gaussian whose per-row
        norms match Δ_s's row norms.
    """
    delta_flat = _flatten(delta).float()
    S_, T_, D_ = delta_flat.shape

    # Pick k from k90 of each seed (capped).
    if k is None:
        S_vals, _ = _per_seed_svd(delta_flat)
        energies = S_vals ** 2
        cum_frac = torch.cumsum(energies, dim=1) / torch.clamp(
            energies.sum(dim=1, keepdim=True), min=1e-30
        )
        k_per = []
        for s in range(S_):
            idx = (cum_frac[s] >= 0.90).nonzero(as_tuple=True)[0]
            k_per.append(int(idx[0].item()) + 1 if len(idx) > 0 else T_)
        k = min(max(k_per), 10)

    # Observed top-k right singular vectors per seed.
    Vh_topk = []
    for s in range(S_):
        _, _, vh = torch.linalg.svd(delta_flat[s], full_matrices=False)
        Vh_topk.append(vh[:k])
    Vh_topk = torch.stack(Vh_topk, dim=0)  # (S, k, D)

    # Pairwise principal-angle matrix (mean angle per pair).
    angle_matrix = torch.zeros(S_, S_)
    pair_angles = []
    for i in range(S_):
        for j in range(S_):
            if i == j:
                angle_matrix[i, j] = 0.0
                continue
            ang = _principal_angles_deg(Vh_topk[i], Vh_topk[j])
            angle_matrix[i, j] = ang.mean()
            if i < j:
                pair_angles.append(float(ang.mean().item()))
    mean_observed = float(np.mean(pair_angles)) if pair_angles else float("nan")

    # Null: isotropic Gaussian with matched per-row norms.
    row_norms = delta_flat.norm(dim=2)  # (S, T)
    gen = torch.Generator(device="cpu").manual_seed(int(rng_seed))
    null_means = []
    for _ in range(int(n_null)):
        Vh_null = []
        for s in range(S_):
            g = torch.randn(T_, D_, generator=gen)
            g = g / torch.clamp(g.norm(dim=1, keepdim=True), min=1e-30) * row_norms[s].unsqueeze(1)
            _, _, vh = torch.linalg.svd(g, full_matrices=False)
            Vh_null.append(vh[:k])
        Vh_null = torch.stack(Vh_null, dim=0)
        ang_list = []
        for i in range(S_):
            for j in range(i + 1, S_):
                ang_list.append(float(_principal_angles_deg(Vh_null[i], Vh_null[j]).mean().item()))
        null_means.append(float(np.mean(ang_list)))

    null_means = np.asarray(null_means, dtype=np.float64)
    p5 = float(np.percentile(null_means, 5.0))
    p50 = float(np.percentile(null_means, 50.0))

    if mean_observed < p5:
        verdict = "below_null"
    elif mean_observed > p50:
        verdict = "indistinguishable_from_null"
    else:
        verdict = "marginal"

    if ax_matrix is not None:
        im = ax_matrix.imshow(angle_matrix.numpy(), cmap="viridis", vmin=0, vmax=90)
        ax_matrix.set_xticks(range(S_))
        ax_matrix.set_yticks(range(S_))
        ax_matrix.set_xlabel("seed index")
        ax_matrix.set_ylabel("seed index")
        ax_matrix.set_title(
            f"B. principal angle (deg), k={k} — {verdict}"
        )
        plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)
    if ax_null is not None:
        ax_null.hist(null_means, bins=20, color="gray", alpha=0.7, label="null")
        ax_null.axvline(mean_observed, color="red", linewidth=2, label=f"obs={mean_observed:.1f}°")
        ax_null.axvline(p5, color="black", linestyle="--", linewidth=1, label=f"null p5={p5:.1f}°")
        ax_null.set_xlabel("mean principal angle (deg)")
        ax_null.set_ylabel("null permutations")
        ax_null.set_title("B. observed vs null")
        ax_null.legend(fontsize=7)
        ax_null.grid(True, alpha=0.3)

    return {
        "k_used": int(k),
        "principal_angles_deg_matrix": angle_matrix.tolist(),
        "mean_angle_observed_deg": mean_observed,
        "null_n_permutations": int(n_null),
        "null_mean_angle_p5_deg": p5,
        "null_mean_angle_p50_deg": p50,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Prong C — energy vs time
# ---------------------------------------------------------------------------


def prong_c_energy_vs_time(
    delta: torch.Tensor,
    eps_poe: torch.Tensor | None = None,
    ax: plt.Axes | None = None,
) -> dict[str, Any]:
    """||Δ_t|| vs t and a ratio to ||ε̃_poe_t||.

    Mathematical content:
        For each (s, t): scalar n_st = ||Δ(s, t)||_2 over the (C, H, W) axes.
        Plot mean ± std over seeds. Report peak step and fraction of total
        energy in steps 10–29 (mid-SNR band).
    """
    delta_norm = delta.flatten(2).float().norm(dim=2)  # (S, T)
    mean_t = delta_norm.mean(dim=0)
    std_t = delta_norm.std(dim=0)

    energy_t = (delta_norm ** 2).mean(dim=0)  # mean energy across seeds, per t
    total = float(energy_t.sum().item())
    t_count = int(delta.shape[1])
    mid_end = min(30, t_count)
    mid_frac = float(energy_t[10:mid_end].sum().item() / max(total, 1e-30))
    peak_step = int(torch.argmax(mean_t).item())

    if mid_frac >= 0.60:
        verdict = "mid_snr_concentrated"
    elif mid_frac <= 0.30:
        verdict = "early_or_late_dominant"
    else:
        verdict = "flat"

    out: dict[str, Any] = {
        "delta_norm_per_step_mean": mean_t.tolist(),
        "delta_norm_per_step_std": std_t.tolist(),
        "peak_step_index": peak_step,
        "energy_fraction_in_steps_10_29": mid_frac,
        "verdict": verdict,
    }
    if eps_poe is not None:
        poe_norm = eps_poe.flatten(2).float().norm(dim=2).mean(dim=0)
        out["eps_poe_norm_per_step_mean"] = poe_norm.tolist()

    if ax is not None:
        x = np.arange(t_count)
        ax.plot(x, mean_t.numpy(), color="C0", label="||Δ_t|| (mean across seeds)")
        ax.fill_between(
            x, (mean_t - std_t).numpy(), (mean_t + std_t).numpy(),
            color="C0", alpha=0.2,
        )
        ax.axvline(peak_step, color="C3", linewidth=1, linestyle="--",
                   label=f"peak step {peak_step}")
        ax.axvspan(10, mid_end - 1, color="C2", alpha=0.1, label="mid band [10,29]")
        ax.set_xlabel("denoising step index")
        ax.set_ylabel("||Δ_t||_2")
        ax.set_title(f"C. energy vs time — {verdict} (mid-band frac={mid_frac:.2f})")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    return out


# ---------------------------------------------------------------------------
# Prong D — spatial locus
# ---------------------------------------------------------------------------


def _concentration_ratio(energy_2d: torch.Tensor, top_frac: float = 0.10) -> float:
    """Top-q% energy / total energy. Uniform → top_frac, concentrated → 1.0."""
    flat = energy_2d.flatten()
    n = flat.numel()
    k = max(1, int(round(n * top_frac)))
    top_sum = float(flat.topk(k).values.sum().item())
    total = float(flat.sum().item())
    return top_sum / max(total, 1e-30)


def prong_d_spatial_locus(
    delta: torch.Tensor,
    ax: plt.Axes | None = None,
) -> dict[str, Any]:
    """Per-pixel energy map split by timestep bucket.

    Buckets: early=[0,9], mid=[10,29], late=[30,T).
    Energy per pixel: E(h,w) = mean over (seed, t in bucket, c) of Δ(s,t,c,h,w)^2.
    Concentration ratio: top-10%-by-magnitude energy / total energy.
    """
    S_, T_, C_, H_, W_ = delta.shape
    early = (0, min(10, T_))
    mid = (min(10, T_), min(30, T_))
    late = (min(30, T_), T_)

    def bucket_energy(rng: tuple[int, int]) -> torch.Tensor:
        a, b = rng
        if a >= b:
            return torch.zeros(H_, W_)
        sl = delta[:, a:b].float()  # (S, t', C, H, W)
        return (sl ** 2).mean(dim=(0, 1, 2))  # (H, W)

    energy_early = bucket_energy(early)
    energy_mid = bucket_energy(mid)
    energy_late = bucket_energy(late)

    cr_early = _concentration_ratio(energy_early)
    cr_mid = _concentration_ratio(energy_mid)
    cr_late = _concentration_ratio(energy_late)

    bucket_crs = {"early": cr_early, "mid": cr_mid, "late": cr_late}
    localized = [name for name, cr in bucket_crs.items() if cr > 0.5]
    if localized:
        verdict = "localized_in_" + "_".join(localized)
    elif all(cr < 0.15 for cr in bucket_crs.values()):
        verdict = "diffuse"
    else:
        verdict = "intermediate"

    if ax is not None:
        # Compose a 1x3 strip side-by-side: early | mid | late.
        strip = torch.cat([
            energy_early / max(float(energy_early.max()), 1e-30),
            energy_mid / max(float(energy_mid.max()), 1e-30),
            energy_late / max(float(energy_late.max()), 1e-30),
        ], dim=1).numpy()
        ax.imshow(strip, cmap="magma", aspect="equal")
        ax.set_xticks([W_ // 2, W_ + W_ // 2, 2 * W_ + W_ // 2])
        ax.set_xticklabels([
            f"early\nCR={cr_early:.2f}",
            f"mid\nCR={cr_mid:.2f}",
            f"late\nCR={cr_late:.2f}",
        ])
        ax.set_yticks([])
        ax.set_title(f"D. spatial locus — {verdict}")

    return {
        "early_steps_energy_concentration_ratio": cr_early,
        "mid_steps_energy_concentration_ratio": cr_mid,
        "late_steps_energy_concentration_ratio": cr_late,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Prong E — seed-shared vs seed-specific energy decomposition
# ---------------------------------------------------------------------------


def prong_e_noise_floor(
    eps_poe: torch.Tensor,
    eps_mono: torch.Tensor,
    ax: plt.Axes | None = None,
) -> dict[str, Any]:
    """Decompose Δ into a seed-shared signal and a seed-specific residual.

    The plan's original Prong E (matched vs mismatched-seed Δ) is confounded
    by state divergence: at step t, two seeds occupy different latents x_t,
    so ||ε̃_mono(i) - ε̃_poe(j)|| measures both (a) a structured correction
    and (b) the seed-driven state gap. (b) dominates.

    This version asks the same question — *is Δ signal or noise?* — by
    splitting Δ at each step into the cross-seed mean (the seed-shared
    part) and per-seed residual:

        Δ̄(t)   = mean_s Δ(s, t)
        r(s,t) = Δ(s, t) - Δ̄(t)
        σ²(t)  = (1/(N-1)) Σ_s ||r(s,t)||²            (unbiased seed variance)

    Naïve SNR := ||Δ̄(t)||² / σ²(t) is *biased* by 1/(N-1) because under
    pure-noise null (true signal = 0), E[||Δ̄||²] = σ² / N. We therefore
    report the bias-corrected estimator:

        signal_hat(t)   = ||Δ̄(t)||² - σ²(t) / N      (unbiased; can be < 0)
        SNR_corr(t)     = signal_hat(t) / σ²(t)
        aggregate_SNR   = Σ_t signal_hat(t) / Σ_t σ²(t)

    Under pure noise, E[SNR_corr] = 0 for any N. Positive values indicate
    a true shared signal beyond finite-sample noise.

    Verdict (uses bias-corrected SNR; an effect-size threshold of 5%
    distinguishes "essentially zero" from "small but real"):
        SNR_corr ≥ 0.5  → "signal_dominated"
        SNR_corr < 0.05 → "noise_dominated"
        otherwise       → "mixed"
    """
    delta = (eps_mono - eps_poe).float()  # (S, T, C, H, W)
    S_, T_ = delta.shape[:2]
    if S_ < 2:
        raise ValueError("prong_e_noise_floor requires N≥2 seeds")

    delta_mean = delta.mean(dim=0)                # (T, C, H, W)
    delta_resid = delta - delta_mean.unsqueeze(0) # (S, T, C, H, W)

    shared_energy_per_t = delta_mean.flatten(1).pow(2).sum(dim=1)         # (T,)
    # σ²(t) — unbiased estimator of seed-spread variance (ddof=1).
    seed_var_per_t = (
        delta_resid.flatten(2).pow(2).sum(dim=2).sum(dim=0) / float(S_ - 1)
    )
    # Bias correction: under pure-noise null, E[||Δ̄||²] = σ²/N.
    signal_hat_per_t = shared_energy_per_t - seed_var_per_t / float(S_)
    snr_corr_per_t = signal_hat_per_t / seed_var_per_t.clamp(min=1e-30)

    total_signal_hat = float(signal_hat_per_t.sum().item())
    total_seed_var = float(seed_var_per_t.sum().item())
    total_shared = float(shared_energy_per_t.sum().item())
    aggregate_snr_corr = total_signal_hat / max(total_seed_var, 1e-30)
    aggregate_snr_naive = total_shared / max(total_seed_var, 1e-30)

    if aggregate_snr_corr >= 0.5:
        verdict = "signal_dominated"
    elif aggregate_snr_corr < 0.05:
        verdict = "noise_dominated"
    else:
        verdict = "mixed"

    if ax is not None:
        x = np.arange(T_)
        ax.plot(x, shared_energy_per_t.numpy(), color="C0",
                label="||Δ̄_t||² (raw shared)")
        ax.plot(x, (seed_var_per_t / float(S_)).numpy(), color="C2", linestyle=":",
                label=f"σ²/N (null shared @ N={S_})")
        ax.plot(x, seed_var_per_t.numpy(), color="C3",
                label="σ² (seed-spread)")
        ax.set_yscale("log")
        ax.set_xlabel("denoising step")
        ax.set_ylabel("energy")
        ax.set_title(
            f"E. shared vs seed-specific — {verdict}\n"
            f"bias-corrected SNR = {aggregate_snr_corr:+.3f} (naïve {aggregate_snr_naive:.3f})"
        )
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3, which="both")

    return {
        "shared_energy_per_step": shared_energy_per_t.tolist(),
        "seed_variance_per_step": seed_var_per_t.tolist(),
        "signal_hat_per_step": signal_hat_per_t.tolist(),
        "snr_corrected_per_step": snr_corr_per_t.tolist(),
        "shared_total_energy": total_shared,
        "seed_variance_total": total_seed_var,
        "signal_hat_total": total_signal_hat,
        "aggregate_snr_naive": aggregate_snr_naive,
        "aggregate_snr_corrected": aggregate_snr_corr,
        "n_seeds": int(S_),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def synthesize_landing(results: dict[str, Any]) -> str:
    """Map per-prong verdicts → final_landing label.

    Decision rules (see plan §6 Phase 4, updated for Prong E):
        E=noise_dominated → landing_6 (overrides all)
        A=compressible AND B=below_null AND E=signal_dominated → landing_1
        A=compressible AND B=indistinguishable_from_null → landing_2
        A=compressible AND B=below_null (but E=mixed) → landing_2 (weak landing_1)
        A=not_compressible AND D=localized_* → landing_4
        D=diffuse → add landing_3
        C=mid_snr_concentrated → always append landing_5

    A=marginal is treated as A=compressible for the purpose of choosing
    between landings 1/2 — we don't gate the cross-seed verdict on a
    crisp rank cutoff.
    """
    a = results["prong_a_time_compressibility"]["verdict"]
    b = results["prong_b_cross_seed_alignment"]["verdict"]
    c = results["prong_c_energy_vs_time"]["verdict"]
    d = results["prong_d_spatial_locus"]["verdict"]
    e = results["prong_e_noise_floor"]["verdict"]

    if e == "noise_dominated":
        return "landing_6"

    a_compress = a in ("compressible", "marginal")

    parts: list[str] = []
    if a_compress and b == "below_null" and e == "signal_dominated":
        parts.append("landing_1")
    elif a_compress and b == "below_null" and e == "mixed":
        parts.append("landing_2")  # weak shared signal; still structured-but-personal
    elif a_compress and b == "indistinguishable_from_null":
        parts.append("landing_2")
    elif a == "not_compressible" and d.startswith("localized_"):
        parts.append("landing_4")
    elif d == "diffuse" and a_compress:
        parts.append("landing_3")

    if c == "mid_snr_concentrated":
        parts.append("landing_5")
    if not parts:
        parts.append("indeterminate")
    return "_plus_".join(parts)
