"""Frame conventions for the NeuroBEM data, and the checks that establish them.

World frame: z up. Body frame: FLU (x forward, y left, z along thrust). The quaternion
convention and the meaning of acc_* are NOT documented; tests/test_frames.py decides them and
writes the evidence to results/frame_tests.json. The constants below record the outcome and
are guarded by that test (it fails if the data disagrees with them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81                       # m/s^2
G_WORLD = np.array([0.0, 0.0, -G])
MASS = 0.772                   # kg (data/README.md)
INERTIA = np.diag([0.0025, 0.0021, 0.0043])

# Set from the CP1 test results (see results/frame_tests.json). None = not yet decided.
QUAT_CONVENTION: str | None = "xyzw_hamilton_body_to_world"
ACC_IS_SPECIFIC_FORCE: bool | None = True

EDGE = 5                       # samples dropped at each segment end when finite-differencing


# ---------------------------------------------------------------- quaternions
def _rot_hamilton(x, y, z, w):
    """Rotation matrices (N,3,3) of unit Hamilton quaternions; active rotation R(q)."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((len(x), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _q(df):
    return df["qx"].to_numpy(), df["qy"].to_numpy(), df["qz"].to_numpy(), df["qw"].to_numpy()


# Each candidate returns R_bw (body -> world): p_world = R_bw @ p_body.
CANDIDATES = {
    # columns read as (x, y, z, w), Hamilton, q rotates body into world (Eigen/ROS default)
    "xyzw_hamilton_body_to_world": lambda df: _rot_hamilton(*_q(df)),
    # same columns, q rotates world into body (also = JPL xyzw body->world)
    "xyzw_hamilton_world_to_body": lambda df: _rot_hamilton(*_q(df)).transpose(0, 2, 1),
    # header mislabelled: columns are really (w, x, y, z)
    "wxyz_hamilton_body_to_world": lambda df: (lambda a, b, c, d: _rot_hamilton(b, c, d, a))(*_q(df)),
    "wxyz_hamilton_world_to_body": lambda df: (lambda a, b, c, d: _rot_hamilton(b, c, d, a))(*_q(df)).transpose(0, 2, 1),
}


def rot_body_to_world(df: pd.DataFrame, convention: str | None = None) -> np.ndarray:
    conv = convention or QUAT_CONVENTION
    if conv is None:
        raise RuntimeError("QUAT_CONVENTION not set - run tests/test_frames.py first")
    return CANDIDATES[conv](df)


def gravity_in_body(R_bw: np.ndarray) -> np.ndarray:
    """World gravity expressed in body frame: R_bw^T g_w, shape (N,3)."""
    return np.einsum("nji,j->ni", R_bw, G_WORLD)


def to_world(R_bw, v_b):
    return np.einsum("nij,nj->ni", R_bw, v_b)


def to_body(R_bw, v_w):
    return np.einsum("nji,nj->ni", R_bw, v_w)


def vec(df, prefix):
    return df[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy()


# ---------------------------------------------------------------- per-segment derivatives
def segment_gradient(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """d(values)/dt computed separately inside each segment (central differences)."""
    out = np.full_like(values, np.nan, dtype=float)
    t = df["t"].to_numpy()
    for idx in df.groupby("segment_id", observed=True).indices.values():
        if len(idx) >= 3:
            out[idx] = np.gradient(values[idx], t[idx], axis=0)
    return out


def interior_mask(df: pd.DataFrame, edge: int = EDGE) -> np.ndarray:
    """False for the first/last `edge` samples of every segment."""
    m = np.ones(len(df), bool)
    for idx in df.groupby("segment_id", observed=True).indices.values():
        m[idx[:edge]] = False
        m[idx[-edge:]] = False
    return m


def hover_mask(df: pd.DataFrame, v_max=0.3, w_max=0.3, dv_max=0.3, mot_min=800.0) -> np.ndarray:
    """Near-hover samples: low speed, low body rates, low rate of change of body velocity,
    and every motor above mot_min rad/s (drops on-ground samples at segment starts, where
    motors idle at ~300 rad/s and the ground carries the weight). Uses no acc, so it is
    independent of the acc hypothesis."""
    v = vec(df, "vel"); w = vec(df, "ang_vel")
    dv = segment_gradient(df, v)
    mot_ok = (df[["mot1", "mot2", "mot3", "mot4"]].to_numpy() > mot_min).all(1)
    m = (np.linalg.norm(v, axis=1) < v_max) & (np.linalg.norm(w, axis=1) < w_max) \
        & (np.linalg.norm(dv, axis=1) < dv_max) & interior_mask(df) & mot_ok
    return m & np.isfinite(dv).all(1)


def _rms(a):
    return float(np.sqrt(np.mean(np.sum(a * a, axis=1))))


# ---------------------------------------------------------------- the three checks
def quaternion_gravity_check(df: pd.DataFrame) -> dict:
    """Test 1: rotate world gravity into body with each candidate during near-hover samples.

    At hover the kinematic acceleration is ~0, so the specific force the IMU would see is
    f_b = -g_b. The candidates only differ in the sign/arrangement of the tilt terms, so we
    compare the horizontal (body x, y) parts: rms |acc_xy + g_b_xy| for each candidate, versus
    the 'level' baseline rms |acc_xy| (what you'd get assuming zero tilt).
    """
    m = hover_mask(df)
    sub = df.loc[m]
    acc = vec(sub, "acc")
    out = {"n_hover": int(m.sum()), "n_hover_segments": int(sub["segment_id"].nunique()),
           "baseline_level_rms_xy": float(np.sqrt(np.mean(np.sum(acc[:, :2] ** 2, 1)))),
           "candidates": {}}
    for name, fn in CANDIDATES.items():
        gb = gravity_in_body(fn(sub))
        tilt = np.degrees(np.arccos(np.clip(-gb[:, 2] / G, -1, 1)))
        res_xy = acc[:, :2] + gb[:, :2]
        cosang = np.sum(acc * -gb, 1) / (np.linalg.norm(acc, axis=1) * G)
        out["candidates"][name] = {
            "rms_xy_residual": float(np.sqrt(np.mean(np.sum(res_xy ** 2, 1)))),
            "rms_3d_residual": _rms(acc + gb),
            "median_angle_deg_acc_vs_minus_g": float(np.degrees(np.median(np.arccos(np.clip(cosang, -1, 1))))),
            "median_tilt_deg": float(np.median(tilt)),
        }
    out["median_acc_z_hover"] = float(np.median(acc[:, 2]))
    return out


def velocity_kinematics_check(df: pd.DataFrame) -> dict:
    """Cross-check independent of acc: d(pos)/dt should equal R_bw @ vel_b for the right
    convention. Also tests the hypothesis that vel_* is actually world-frame."""
    m = interior_mask(df)
    dp = segment_gradient(df, vec(df, "pos"))[m]
    v = vec(df, "vel")
    out = {"n": int(m.sum()), "rms_dpos_dt": _rms(dp),
           "vel_is_world_rms": _rms(dp - v[m]), "candidates": {}}
    for name, fn in CANDIDATES.items():
        out["candidates"][name] = {"rms_residual": _rms(dp - to_world(fn(df)[m], v[m]))}
    return out


def acc_type_check(df: pd.DataFrame, convention: str) -> dict:
    """Test 2: is acc_* kinematic (dv/dt in body axes) or specific force (dv/dt - g, body axes)?
    Reference: a_w = d(R_bw vel_b)/dt from velocity + attitude only."""
    R = CANDIDATES[convention](df)
    v_w = to_world(R, vec(df, "vel"))
    a_w = segment_gradient(df, v_w)
    m = interior_mask(df) & np.isfinite(a_w).all(1)
    R, a_w, acc = R[m], a_w[m], vec(df, "acc")[m]
    kin = to_body(R, a_w)
    spf = to_body(R, a_w - G_WORLD)
    def per_axis(r):
        return [float(np.sqrt(np.mean(r[:, i] ** 2))) for i in range(3)]
    # least-squares scale on z (should be ~1 under the right hypothesis)
    scale_spf_z = float(np.dot(acc[:, 2], spf[:, 2]) / np.dot(spf[:, 2], spf[:, 2]))
    return {"n": int(m.sum()), "convention": convention,
            "kinematic_rms": _rms(acc - kin), "kinematic_rms_per_axis": per_axis(acc - kin),
            "specific_force_rms": _rms(acc - spf), "specific_force_rms_per_axis": per_axis(acc - spf),
            "specific_force_z_scale": scale_spf_z,
            "median_acc_z": float(np.median(acc[:, 2]))}


def specific_force_body(df: pd.DataFrame) -> np.ndarray:
    """Specific force (thrust + aero)/m in body frame, whatever acc_* turned out to be."""
    if ACC_IS_SPECIFIC_FORCE is None:
        raise RuntimeError("ACC_IS_SPECIFIC_FORCE not set - run tests/test_frames.py first")
    acc = vec(df, "acc")
    return acc if ACC_IS_SPECIFIC_FORCE else acc - gravity_in_body(rot_body_to_world(df))


def kinematic_acc_world(df: pd.DataFrame) -> np.ndarray:
    R = rot_body_to_world(df)
    return to_world(R, specific_force_body(df)) + G_WORLD
