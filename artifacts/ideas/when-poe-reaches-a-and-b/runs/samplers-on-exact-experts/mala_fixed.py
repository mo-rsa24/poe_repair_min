"""AnnealedMALASampler copied from anneal_samplers.py (lines 5-44) with two bugs fixed.

Fix 1 (anneal_samplers.py lines 34-35): the proposal is x + ss*grad + N(0, 2*ss), so its
log-density divides the squared distance by 2 * variance = 4*ss. The original divides by
2*ss**2.

Fix 2 (anneal_samplers.py line 39): the original draws one torch.rand(1) and accepts or rejects
the whole batch against energies summed over the batch. Here each sample has its own
log-density, proposal ratio and uniform draw, and moves or stays on its own.

Also added: `last_accept`, the per-step batch acceptance share, so the caller can tune the step.
The gradient_function contract is unchanged: it returns (log density per sample, score).
anneal_samplers.py itself is not edited.
"""
import torch


class AnnealedMALASampler:
  """Implements annealed MALA (per-sample Metropolis-Hastings)."""

  def __init__(self,
               num_steps,
               num_samples_per_step,
               step_sizes,
               gradient_function,
               ):
    assert len(step_sizes) == num_steps, "Must have as many stepsizes as intermediate distributions."
    self._step_sizes = step_sizes
    self._num_steps = num_steps
    self._num_samples_per_step = num_samples_per_step
    self._gradient_function = gradient_function
    self.last_accept = []

  def sample_step(self, x, t, ts, model_args):
    self.last_accept = []
    e_old, grad = self._gradient_function(x, ts, **model_args)
    dims = tuple(range(1, x.dim()))
    for i in range(self._num_samples_per_step):
      ss = self._step_sizes[t]
      std = (2 * ss) ** .5
      noise = torch.randn_like(grad) * std

      x_proposal = x + grad * ss + noise

      e_new, grad_new = self._gradient_function(x_proposal, ts, **model_args)

      # FIX 1: proposal variance is 2*ss, so divide by 2 * (2*ss) = 4*ss.
      log_xhat_given_x = -1.0 * ((x_proposal - x - ss * grad) ** 2).sum(dims) / (4 * ss)
      log_x_given_xhat = -1.0 * ((x - x_proposal - ss * grad_new) ** 2).sum(dims) / (4 * ss)
      log_alpha = e_new - e_old + log_x_given_xhat - log_xhat_given_x

      # FIX 2: one uniform per sample, accept per sample.
      u = torch.rand(log_alpha.shape, dtype=log_alpha.dtype, device=log_alpha.device)
      accept = torch.log(u) < log_alpha.detach()
      mask = accept.view(-1, *([1] * (x.dim() - 1)))
      x = torch.where(mask, x_proposal, x)
      e_old = torch.where(accept, e_new, e_old)
      grad = torch.where(mask, grad_new, grad)
      self.last_accept.append(accept.double().mean().item())

    return x
