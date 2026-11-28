"""Physically consistent fake NeuroBEM segments (raw-column DataFrame) for rollout-script tests.

Level attitude with a slow yaw, smooth body velocity; acc = specific force (dv/dt + w x v - g_b),
pos = integral of R v, so the recorded-force (oracle) rollout reproduces the trajectory. Motor
speeds give hover thrust plus the vertical acceleration. Never used for results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.frames import G, MASS
from src.data.load import COLUMNS

DT = 0.0025
KT = 1.5e-6


def segment(flight_id: str, seg: int, n: int = 1200, seed: int = 0, gap_at: int | None = None,
            speed: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    if gap_at is not None:
        t[gap_at:] += 0.05                                       # 50 ms Vicon dropout
    f = rng.uniform(0.3, 0.8, 3); ph = rng.uniform(0, 2 * np.pi, 3)
    vel_w = speed * np.stack([np.sin(2 * np.pi * f[i] * t + ph[i]) for i in range(3)], 1) * np.array([1, 1, 0.3])
    acc_w = speed * np.stack([2 * np.pi * f[i] * np.cos(2 * np.pi * f[i] * t + ph[i]) for i in range(3)], 1) * np.array([1, 1, 0.3])
    yaw_rate = 0.2
    yaw = yaw_rate * t
    c, s = np.cos(yaw), np.sin(yaw)
    # body = world rotated by yaw about z; R_bw = Rz(yaw)
    to_body = lambda a: np.stack([c * a[:, 0] + s * a[:, 1], -s * a[:, 0] + c * a[:, 1], a[:, 2]], 1)  # noqa: E731
    vel_b = to_body(vel_w)
    spec_b = to_body(acc_w + np.array([0, 0, G]))                 # specific force, body
    pos = np.cumsum(vel_w, 0) * DT
    q = np.stack([np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)], 1)
    thrust = MASS * spec_b[:, 2]
    mot = np.sqrt(np.maximum(thrust, 1.0) / (4 * KT))[:, None] + rng.normal(0, 2, (n, 4))
    dmot = np.gradient(mot, DT, axis=0)
    d = {"t": t, "ang_acc_x": 0.0, "ang_acc_y": 0.0, "ang_acc_z": 0.0,
         "ang_vel_x": 0.0, "ang_vel_y": 0.0, "ang_vel_z": yaw_rate,
         "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
         "acc_x": spec_b[:, 0], "acc_y": spec_b[:, 1], "acc_z": spec_b[:, 2],
         "vel_x": vel_b[:, 0], "vel_y": vel_b[:, 1], "vel_z": vel_b[:, 2],
         "pos_x": pos[:, 0], "pos_y": pos[:, 1], "pos_z": pos[:, 2] + 1.0,
         "mot1": mot[:, 0], "mot2": mot[:, 1], "mot3": mot[:, 2], "mot4": mot[:, 3],
         "dmot1": dmot[:, 0], "dmot2": dmot[:, 1], "dmot3": dmot[:, 2], "dmot4": dmot[:, 3], "vbat": 160.0}
    df = pd.DataFrame({k: (np.full(n, v) if np.isscalar(v) else v) for k, v in d.items()})[COLUMNS]
    df["flight_id"] = flight_id
    df["segment_id"] = f"{flight_id}_seg_{seg}"
    return df


def flights(n_flights: int = 3, segs: int = 2, n: int = 1200, fast: bool = True) -> pd.DataFrame:
    out = []
    for i in range(n_flights):
        fid = f"2021-01-0{i + 1}-12-00-00"
        for s in range(segs):
            out.append(segment(fid, s + 1, n, seed=10 * i + s, gap_at=(n // 2 if (i == 0 and s == 0) else None),
                               speed=(7.0 if (fast and i == n_flights - 1) else 3.0)))
    return pd.concat(out, ignore_index=True)
