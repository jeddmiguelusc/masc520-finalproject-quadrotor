"""Model inputs and targets, one row per 400 Hz sample.

Inputs (18, never acc or ang_acc - those define the targets):
    vel (3, body) · ang_vel (3, body) · gravity in body (3) · motor speeds (4) · dmot (4) · vbat (1)
Targets (6, body frame):
    F = m * acc                      (acc is specific force - CP1 test 2)
    M = J * ang_acc + ang_vel x (J * ang_vel)   (rigid-body Euler equation)

Targets are defined in src/data/targets.py (checked by tests/test_targets.py). Split-aware
loading with contact trimming is src/data/dataset.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "vel_x", "vel_y", "vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "g_b_x", "g_b_y", "g_b_z",
    "mot1", "mot2", "mot3", "mot4",
    "dmot1", "dmot2", "dmot3", "dmot4",
    "vbat",
]
TARGETS = ["F_x", "F_y", "F_z", "M_x", "M_y", "M_z"]
N_FEATURES, N_TARGETS = len(FEATURES), len(TARGETS)
IDX = {name: i for i, name in enumerate(FEATURES)}


def build_features(df: pd.DataFrame) -> np.ndarray:
    from src.data import frames as F
    g_b = F.gravity_in_body(F.rot_body_to_world(df))
    cols = [df[c].to_numpy() for c in FEATURES[:6]] + [g_b[:, 0], g_b[:, 1], g_b[:, 2]] \
        + [df[c].to_numpy() for c in FEATURES[9:]]
    return np.stack(cols, axis=1).astype(np.float32)


def build_targets(df: pd.DataFrame) -> np.ndarray:
    """Body-frame force and torque; the definition lives in src/data/targets.py."""
    from src.data.targets import build_targets as _bt
    return _bt(df)


def segment_arrays(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(segment_id, X (T,18), Y (T,6)), ...] in the frame's segment order."""
    out = []
    for sid, g in df.groupby("segment_id", observed=True, sort=False):
        g = g.sort_values("t")
        out.append((str(sid), build_features(g), build_targets(g)))
    return out
