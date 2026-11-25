"""EDA computations and plots for notebooks/01_eda.ipynb (the notebook holds no logic).

Needs the frame decisions in src/data/frames.py (set by tests/test_frames.py).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import frames as F
from src.data.load import REPO

RESULTS = REPO / "results"
FIGS = RESULTS / "figures"
DT = 0.0025                                     # 400 Hz
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3de"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Adds speed, |kinematic acc| (world), |specific force|, sum of omega^2."""
    out = df.copy()
    out["speed"] = np.linalg.norm(F.vec(df, "vel"), axis=1)          # frame-invariant
    out["acc_kin_norm"] = np.linalg.norm(F.kinematic_acc_world(df), axis=1)
    out["spf_norm"] = np.linalg.norm(F.specific_force_body(df), axis=1)
    out["sum_omega2"] = (df[["mot1", "mot2", "mot3", "mot4"]].to_numpy() ** 2).sum(1)
    return out


def flight_summary(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """One row per flight. Durations sum the segments (gaps between segments excluded)."""
    seg = df.groupby("segment_id", observed=True).agg(
        flight_id=("flight_id", "first"), n=("t", "size"),
        max_speed=("speed", "max"), max_acc_kin=("acc_kin_norm", "max"),
        max_spf=("spf_norm", "max"), vbat_min=("vbat", "min"), vbat_max=("vbat", "max"))
    seg["duration_s"] = seg["n"] * DT
    seg = seg.reset_index()
    seg["flight_id"] = seg["flight_id"].astype(str)
    fl = seg.groupby("flight_id").agg(
        n_segments=("segment_id", "size"), n_samples=("n", "sum"), duration_s=("duration_s", "sum"),
        max_speed=("max_speed", "max"), max_acc_kin=("max_acc_kin", "max"),
        max_specific_force=("max_spf", "max"), vbat_min=("vbat_min", "min"), vbat_max=("vbat_max", "max"))
    meta = idx.groupby("flight_id").agg(description=("description", "first"),
                                        trajectory_type=("trajectory_type", "first"),
                                        commanded_vel=("commanded_vel", "first"),
                                        has_server_testset_segment=("in_server_testset", "any"))
    s = meta.join(fl, how="inner").reset_index()
    cols = ["flight_id", "trajectory_type", "description", "commanded_vel", "n_segments", "n_samples",
            "duration_s", "max_speed", "max_acc_kin", "max_specific_force", "vbat_min", "vbat_max",
            "has_server_testset_segment"]
    return s[cols].sort_values("flight_id").reset_index(drop=True)


def pick_example_flights(summary: pd.DataFrame, min_duration=10.0) -> dict:
    """slow = lowest max speed, aggressive = highest, medium = closest to the median.
    Only flights with >= min_duration s and no server-testset segment."""
    c = summary[(summary.duration_s >= min_duration) & ~summary.has_server_testset_segment]
    med = c.max_speed.median()
    return {"slow": c.loc[c.max_speed.idxmin(), "flight_id"],
            "medium": c.loc[(c.max_speed - med).abs().idxmin(), "flight_id"],
            "aggressive": c.loc[c.max_speed.idxmax(), "flight_id"]}


def plot_example_flights(df: pd.DataFrame, summary: pd.DataFrame, picks: dict, path=None):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row, (label, fid) in enumerate(picks.items()):
        d = df[df.flight_id.astype(str) == fid]
        info = summary.set_index("flight_id").loc[fid]
        ax_xy, ax_v, ax_m = axes[row]
        for _, g in d.groupby("segment_id", observed=True):
            ax_xy.plot(g.pos_x, g.pos_y, color=BLUE, lw=1.2)
            ax_v.plot(g.t, g.speed, color=BLUE, lw=1.2, label="speed [m/s]")
            ax_v.plot(g.t, g.acc_kin_norm, color=ORANGE, lw=1.2, label="|a| kinematic [m/s²]")
            for k, col in zip(range(1, 5), [BLUE, ORANGE, AQUA, YELLOW]):
                ax_m.plot(g.t, g[f"mot{k}"], color=col, lw=0.8, label=f"motor {k}")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_title(f"{label}: {fid}\n{info.description}", fontsize=9, color=INK, loc="left")
        ax_xy.set_xlabel("x [m]", fontsize=8, color=MUTED); ax_xy.set_ylabel("y [m]", fontsize=8, color=MUTED)
        ax_v.set_title(f"max speed {info.max_speed:.2f} m/s, max |a| {info.max_acc_kin:.1f} m/s²",
                       fontsize=9, color=INK, loc="left")
        ax_v.set_xlabel("t [s] (per segment)", fontsize=8, color=MUTED)
        ax_m.set_title("motor speeds [rad/s]", fontsize=9, color=INK, loc="left")
        ax_m.set_xlabel("t [s] (per segment)", fontsize=8, color=MUTED)
        for ax in (ax_xy, ax_v, ax_m):
            _style(ax)
        if row == 0:
            h, l = ax_v.get_legend_handles_labels(); ax_v.legend(h[:2], l[:2], fontsize=7, frameon=False)
            h, l = ax_m.get_legend_handles_labels(); ax_m.legend(h[:4], l[:4], fontsize=7, frameon=False, ncol=2)
    if path:
        fig.savefig(path, dpi=150)
    return fig


def hover_thrust_fit(df: pd.DataFrame, idx: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit thrust T = m * f_z (body z specific force) against sum(omega^2) at near-hover.
    Server-testset segments are excluded. Exploratory only: the k used by any prior must be
    refit on the CP2 training split."""
    keep = ~df.segment_id.astype(str).isin(idx.loc[idx.in_server_testset, "segment_id"])
    d = df[keep]
    m = F.hover_mask(d)
    h = d.loc[m]
    T = F.MASS * F.specific_force_body(h)[:, 2]
    x = h["sum_omega2"].to_numpy()
    k0 = float(np.dot(x, T) / np.dot(x, x))                       # through origin
    A = np.c_[x, np.ones_like(x)]
    (k1, b1), *_ = np.linalg.lstsq(A, T, rcond=None)
    def r2(pred):
        return float(1 - np.sum((T - pred) ** 2) / np.sum((T - T.mean()) ** 2))
    res = {"n_hover_samples": int(m.sum()), "n_segments": int(h.segment_id.nunique()),
           "mask": "speed<0.3 m/s, |omega|<0.3 rad/s, |dv/dt|<0.3 m/s^2, all motors>800 rad/s, 5-sample segment edges dropped",
           "k_through_origin_N_per_rad2s2": k0, "rmse_through_origin_N": float(np.sqrt(np.mean((T - k0 * x) ** 2))),
           "r2_through_origin": r2(k0 * x),
           "k_with_intercept": float(k1), "intercept_N": float(b1), "r2_with_intercept": r2(k1 * x + b1),
           "thrust_N_range": [float(T.min()), float(T.max())],
           "sum_omega2_range": [float(x.min()), float(x.max())],
           "mass_kg": F.MASS, "excluded": "server testset.txt segments"}
    # Hover thrust is ~m*g by construction, so R^2 vs the mean is not informative here; the
    # spread in sum(omega^2) mostly reflects flight-to-flight changes in k. Record that.
    kk = pd.DataFrame({"flight_id": h["flight_id"].astype(str).to_numpy(), "k": T / x,
                       "vbat": h["vbat"].to_numpy()})
    kk["day"] = kk.flight_id.str[:10]
    per_flight = kk.groupby("flight_id").k.median()
    res["k_across_flights"] = {"n_flights": int(len(per_flight)), "mean": float(per_flight.mean()),
                               "std": float(per_flight.std()), "min": float(per_flight.min()),
                               "max": float(per_flight.max())}
    res["k_within_flight_rel_std_median"] = float((kk.groupby("flight_id").k.std() / kk.k.median()).median())
    res["k_median_per_day"] = {d: float(v) for d, v in kk.groupby("day").k.median().items()}
    res["corr_k_vbat"] = float(np.corrcoef(kk.k, kk.vbat)[0, 1])
    res["thrust_median_N"] = float(np.median(T))
    res["note"] = ("R^2 is vs the mean of a near-constant hover thrust (~m*g); use rmse and k spread instead. "
                   "Exploratory: refit k on the CP2 training split before using it in a prior.")
    pts = pd.DataFrame({"sum_omega2": x, "thrust_N": T, "vbat": h["vbat"].to_numpy()})
    return res, pts


def plot_hover_fit(pts: pd.DataFrame, res: dict, path=None):
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    sub = pts.sample(min(len(pts), 20000), random_state=0)
    ax.scatter(sub.sum_omega2, sub.thrust_N, s=3, color=BLUE, alpha=0.25, linewidths=0, label="near-hover samples")
    xs = np.linspace(pts.sum_omega2.min(), pts.sum_omega2.max(), 50)
    ax.plot(xs, res["k_through_origin_N_per_rad2s2"] * xs, color=ORANGE, lw=2,
            label=f"T = k·Σω², k = {res['k_through_origin_N_per_rad2s2']:.3e}  (RMSE = {res['rmse_through_origin_N']:.3f} N)")
    ax.set_xlabel("Σ ω² [rad²/s²]", fontsize=9, color=MUTED); ax.set_ylabel("thrust m·f_z [N]", fontsize=9, color=MUTED)
    ax.set_title(f"Hover thrust vs Σω²  (n = {res['n_hover_samples']:,})", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    if path:
        fig.savefig(path, dpi=150)
    return fig


def save_json(obj, name):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2))
