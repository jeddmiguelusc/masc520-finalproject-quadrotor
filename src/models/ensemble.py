"""Deep ensemble: K independently seeded copies of one config, combined as a Gaussian mixture.

    mean = (1/K) sum mu_k
    var  = (1/K) sum (sigma_k^2 + mu_k^2) - mean^2        (members with NLL heads)
    var  = (1/K) sum (mu_k - mean)^2                       (members without: disagreement only)

The variance splits into aleatoric (mean of sigma_k^2) and epistemic (spread of mu_k); H4
looks at how the total grows with speed.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def combine(means: list[torch.Tensor], variances: list[torch.Tensor | None]) -> dict:
    mu = torch.stack(means)                                  # (K, B, 6)
    mean = mu.mean(0)
    epistemic = mu.var(0, unbiased=False)
    if all(v is not None for v in variances):
        aleatoric = torch.stack(variances).mean(0)
    else:
        aleatoric = torch.zeros_like(mean)
    return {"mean": mean, "var": aleatoric + epistemic, "aleatoric": aleatoric, "epistemic": epistemic}


@torch.no_grad()
def predict(models, x: torch.Tensor) -> dict:
    outs = [m(x) for m in models]
    return combine([o[0] for o in outs], [o[1] for o in outs])
