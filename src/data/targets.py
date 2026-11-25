"""Regression targets: body-frame force and torque, one row per 400 Hz sample.

    F = m * acc                                   (acc_* is specific force in body FLU - CP1 test 2)
    M = J * ang_acc + ang_vel x (J * ang_vel)     (Euler's rigid-body equation, body frame)

Semantics (tests/test_targets.py checks them against the data):
  - acc_*      : specific force (thrust + aero) / m, body frame. No gravity term to remove.
  - ang_acc_*  : time derivative of the body rates ang_vel_* in the body frame (matches
                 d(ang_vel)/dt at zero lag, slightly low-pass filtered), so Euler's equation
                 applies directly with no frame transport term.
  - m = 0.772 kg, J = diag(0.0025, 0.0021, 0.0043) kg m^2 (data/README.md).

Ground contact: when the drone sits on the ground the IMU still reads ~g but the rotors make
little thrust, so m*acc_z is the ground reaction, not aerodynamics. `contact_mask` flags those
samples (rule from scripts/ground_contact_report.py / results/ground_contact.json) and
`trim_contact` cuts them out, splitting a segment into pieces so no window bridges the cut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import frames as F

MASS = F.MASS
INERTIA = F.INERTIA
TARGETS = ["F_x", "F_y", "F_z", "M_x", "M_y", "M_z"]

# contact rule: mean motor speed < 700 rad/s AND speed < 0.3 m/s (results/ground_contact.json)
CONTACT_MOT_MEAN_MAX = 700.0
CONTACT_SPEED_MAX = 0.3
MOTORS = ["mot1", "mot2", "mot3", "mot4"]


def body_force(df: pd.DataFrame) -> np.ndarray:
    """(N,3) total non-gravitational force on the body, body frame, N."""
    return MASS * F.specific_force_body(df)


def body_torque(df: pd.DataFrame, inertia: np.ndarray = INERTIA) -> np.ndarray:
    """(N,3) total torque about the CoM, body frame, N m."""
    w = F.vec(df, "ang_vel")
    dw = F.vec(df, "ang_acc")
    return dw @ inertia.T + np.cross(w, w @ inertia.T)


def build_targets(df: pd.DataFrame) -> np.ndarray:
    return np.concatenate([body_force(df), body_torque(df)], axis=1).astype(np.float32)


def contact_mask(df: pd.DataFrame) -> np.ndarray:
    """True where the drone is (very likely) resting on the ground."""
    mot_mean = df[MOTORS].to_numpy().mean(1)
    speed = np.linalg.norm(F.vec(df, "vel"), axis=1)
    return (mot_mean < CONTACT_MOT_MEAN_MAX) & (speed < CONTACT_SPEED_MAX)


def trim_contact(seg: pd.DataFrame, min_len: int = 1) -> list[pd.DataFrame]:
    """Split ONE segment (sorted by t) into contact-free pieces.

    Contact runs at either end are cut off; a run in the middle splits the segment in two, so a
    window can never span the removed samples. Pieces shorter than min_len are dropped.
    """
    keep = ~contact_mask(seg)
    d = np.diff(np.r_[0, keep.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [seg.iloc[a:b] for a, b in zip(starts, ends) if b - a >= min_len]
