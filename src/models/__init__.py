"""Model registry. model.type in a config picks one of:

    constant     P0a  predicts the training-set mean (out_mean)                no prior
    physics      P0b  omega^2 + linear drag, least squares                     prior only
    mlp          M1 (prior: false) / M2 (prior: true)   current sample only
    tcn          M3   causal TCN over the window, prior: true
    transformer  M4   small transformer over the window, prior: true
"""
from __future__ import annotations

from src.data.features import N_FEATURES, N_TARGETS
from src.models.core import ForceModel
from src.models.mlp import MLPBackbone
from src.models.temporal import TCNBackbone
from src.models.transformer import TransformerBackbone


def build_model(mcfg: dict) -> ForceModel:
    t = mcfg["type"]
    nll = bool(mcfg.get("nll", False))
    n_out = N_TARGETS * (2 if nll else 1)
    window = int(mcfg.get("window", 1))
    if t == "constant":
        return ForceModel(None, N_FEATURES, N_TARGETS, 1, prior=False)
    if t == "physics":
        return ForceModel(None, N_FEATURES, N_TARGETS, 1, prior=True)
    if t == "mlp":
        bb = MLPBackbone(N_FEATURES, n_out, tuple(mcfg.get("hidden", (128, 128, 64))))
        window = 1
    elif t == "tcn":
        bb = TCNBackbone(N_FEATURES, n_out, mcfg.get("channels", 40), mcfg.get("kernel", 3),
                         tuple(mcfg.get("dilations", (1, 2, 4, 8))), mcfg.get("head_hidden", 64))
    elif t == "transformer":
        bb = TransformerBackbone(N_FEATURES, n_out, mcfg.get("d_model", 40), mcfg.get("n_heads", 4),
                                 mcfg.get("n_layers", 2), mcfg.get("d_ff", 80), window)
    else:
        raise ValueError(f"unknown model type {t!r}")
    return ForceModel(bb, N_FEATURES, N_TARGETS, window, prior=bool(mcfg.get("prior", False)), nll=nll)
