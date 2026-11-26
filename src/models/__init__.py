"""Model registry. model.type in a config picks one of:

    constant     P0a  predicts the training-set mean (out_mean)                no prior
    physics      P0b  omega^2 + linear drag, least squares                     prior only
    mlp          M1 (prior: false) / M2 (prior: true)   current sample only
    tcn          M3   causal TCN over the window, prior: true
    transformer  M4   small transformer over the window, prior: true

Optional keys (CP4 sweep; defaults reproduce the Wed models exactly):
    dropout      MLP / TCN dropout probability (default 0)
    input_noise  std of Gaussian noise added to the standardised inputs while training (default 0)
    history      "all" (default) or "actuator": windowed models see past motor speeds, dmot and vbat,
                 but vel / ang_vel / g_body only at the current sample (no finite-differencing path
                 to the targets; see B2 in src/eval/baselines.py)
"""
from __future__ import annotations

from src.data.features import N_FEATURES, N_TARGETS


def build_model(mcfg: dict):
    # torch imports here, so torch-free submodules (src.models.physics_np) import without torch
    from src.models.core import ForceModel
    from src.models.mlp import MLPBackbone
    from src.models.temporal import TCNBackbone
    from src.models.transformer import TransformerBackbone
    t = mcfg["type"]
    nll = bool(mcfg.get("nll", False))
    n_out = N_TARGETS * (2 if nll else 1)
    window = int(mcfg.get("window", 1))
    if t == "constant":
        return ForceModel(None, N_FEATURES, N_TARGETS, 1, prior=False)
    if t == "physics":
        return ForceModel(None, N_FEATURES, N_TARGETS, 1, prior=True)
    if t == "mlp":
        bb = MLPBackbone(N_FEATURES, n_out, tuple(mcfg.get("hidden", (128, 128, 64))), float(mcfg.get("dropout", 0.0)))
        window = 1
    elif t == "tcn":
        bb = TCNBackbone(N_FEATURES, n_out, mcfg.get("channels", 40), mcfg.get("kernel", 3),
                         tuple(mcfg.get("dilations", (1, 2, 4, 8))), mcfg.get("head_hidden", 64),
                         float(mcfg.get("dropout", 0.0)))
    elif t == "transformer":
        bb = TransformerBackbone(N_FEATURES, n_out, mcfg.get("d_model", 40), mcfg.get("n_heads", 4),
                                 mcfg.get("n_layers", 2), mcfg.get("d_ff", 80), window)
    else:
        raise ValueError(f"unknown model type {t!r}")
    return ForceModel(bb, N_FEATURES, N_TARGETS, window, prior=bool(mcfg.get("prior", False)), nll=nll,
                      input_noise=float(mcfg.get("input_noise", 0.0)), history=mcfg.get("history", "all"))
