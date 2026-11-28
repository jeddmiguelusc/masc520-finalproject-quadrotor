"""Open-loop rollouts of a learned (or physics) force/torque model, differentiable end to end (CP4).

A rollout starts from the TRUE state at sample k0 and is driven by the RECORDED motor speeds,
dmot and vbat. At every step the force model sees the usual 18-feature, 20-sample window, in
which the state channels (vel, ang_vel, g_body) come from the simulation and the actuator
channels come from the data. The 19 samples before k0 are the recorded history.

    from src.sim.rollout import Episodes, ModelForce, rollout
    ep  = Episodes.from_piece(piece_df, starts=[19, 419], n_steps=40)      # 0.1 s at 400 Hz
    out = rollout(ModelForce(model), ep, n_steps=40, integrator="rk4", mode="full")
    out["pos_err"]  # (B, n_steps+1) |p_sim - p_true| [m]

State (all body FLU except position; quaternion Hamilton (x, y, z, w), body -> world, world z up,
the conventions fixed by tests/test_frames.py):
    p_dot = R(q) v_b
    v_dot = F/m + g_b(q) - w x v_b                 (F = total non-gravitational force, body)
    q_dot = 1/2 q (x) (w, 0)
    w_dot = J^-1 (M - w x J w)

Modes
    "translational"  attitude q and body rates w are taken from the data at every step (and
                     interpolated at RK4 stage times); only p and v are integrated; M is unused.
    "full"           p, v, q, w all integrated (6-DOF).

Integrators (dt = data step, 2.5 ms)
    "euler"  semi-implicit (symplectic) Euler: forces at step k; w and v updated first; then q with
             the new w (exponential map) and p with R(q_{k+1}) v_{k+1}. First order.
    "rk4"    classic RK4. The model is re-evaluated at each stage: the last window row is the stage
             state with linearly interpolated actuator inputs; the 19 earlier rows are the committed
             samples (the window is not re-sampled at half steps). q renormalised after each step.

Differentiability: only out-of-place torch ops; gradients flow to the model parameters, the force
model's inputs and the initial state. The quaternion exponential uses a series near zero rate, so
gradients stay finite at w = 0. Works on any device / dtype the model supports (MPS: float32).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.data.features import N_FEATURES
from src.data.frames import G, INERTIA, MASS

DT = 0.0025
WINDOW = 20
V, W, GB, ACT = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, N_FEATURES)


# ---------------------------------------------------------------- quaternion helpers (xyzw, Hamilton)
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_exp(r: torch.Tensor) -> torch.Tensor:
    """Unit quaternion exp((r, 0)) for a rotation-vector half-angle r (..., 3) -> (..., 4) xyzw."""
    th2 = (r * r).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-30)
    small = th < 1e-4
    sinc = torch.where(small, 1 - th2 / 6 + th2 * th2 / 120, torch.sin(th) / torch.where(small, 1.0, th))
    cos = torch.where(small, 1 - th2 / 2 + th2 * th2 / 24, torch.cos(th))
    return torch.cat([r * sinc, cos], -1)


def quat_normalise(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True)


def rot(q: torch.Tensor) -> torch.Tensor:
    """R(q) (..., 3, 3), body -> world (same formula as frames._rot_hamilton)."""
    x, y, z, w = quat_normalise(q).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def gravity_body(q: torch.Tensor) -> torch.Tensor:
    """R(q)^T g_w with g_w = (0, 0, -G): third row of R times -G."""
    R = rot(q)
    return -G * R[..., 2, :]


def quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Rotation angle between two unit quaternions [rad] (sign-insensitive)."""
    d = (quat_normalise(q1) * quat_normalise(q2)).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(d)


# ---------------------------------------------------------------- recorded data
@dataclass
class Episodes:
    """B rollout start points with their recorded data, all (B, T, ...), T = H + n_steps + 1.

    Index H (= window - 1) is the start sample k0; indices 0..H-1 are the recorded history.
    feats are the 18 model inputs; Y the 6 targets (only used by RecordedForce)."""
    feats: torch.Tensor          # (B, T, 18)
    q: torch.Tensor              # (B, T, 4) xyzw body -> world
    pos: torch.Tensor            # (B, T, 3) world
    Y: torch.Tensor | None = None
    H: int = WINDOW - 1
    dt: float = DT

    @classmethod
    def from_piece(cls, piece, starts, n_steps: int, window: int = WINDOW, dtype=torch.float32,
                   device="cpu") -> "Episodes":
        """piece: one contact-trimmed DataFrame (src.data.dataset.frame_pieces); starts: sample
        indices k0 >= window-1 inside the piece with k0 + n_steps < len(piece)."""
        from src.data.features import build_features
        from src.data.targets import build_targets
        t = piece["t"].to_numpy()
        dts = np.diff(t)
        if not np.allclose(np.median(dts), DT, rtol=1e-3):
            raise ValueError(f"piece sample time {np.median(dts):.5f} s, expected {DT}")
        X = build_features(piece)
        Y = build_targets(piece)
        q = piece[["qx", "qy", "qz", "qw"]].to_numpy()
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        p = piece[["pos_x", "pos_y", "pos_z"]].to_numpy()
        H = window - 1
        idx = []
        for k0 in starts:
            if k0 < H or k0 + n_steps >= len(piece):
                raise ValueError(f"start {k0} needs {H} samples before and {n_steps} after (piece {len(piece)})")
            if np.max(np.abs(dts[k0 - H:k0 + n_steps] - DT)) > 0.25 * DT:
                raise ValueError(f"time gap inside the episode at start {k0}")
            idx.append(np.arange(k0 - H, k0 + n_steps + 1))
        idx = np.stack(idx)
        T = lambda a: torch.as_tensor(np.asarray(a)[idx], dtype=dtype, device=device)  # noqa: E731
        return cls(T(X), T(q), T(p), T(Y), H)

    def to(self, device=None, dtype=None) -> "Episodes":
        f = lambda a: None if a is None else a.to(device=device, dtype=dtype)  # noqa: E731
        return Episodes(f(self.feats), f(self.q), f(self.pos), f(self.Y), self.H, self.dt)


# ---------------------------------------------------------------- force / torque sources
class ModelForce:
    """Wraps a ForceModel (or any module window (B, W, 18) -> (mean (B, 6), var)). Gradients pass."""

    def __init__(self, model, window: int = WINDOW):
        self.model, self.window = model, window

    def __call__(self, win: torch.Tensor, k: int, tau: float) -> torch.Tensor:
        out = self.model(win[:, -self.window:, :])
        return out[0] if isinstance(out, tuple) else out


class RecordedForce:
    """Oracle: the recorded targets F = m*acc, M = J*ang_acc + w x Jw (interpolated at stage times).
    Rolling this out checks the integrator and the frame conventions, not a model."""

    def __init__(self, ep: Episodes):
        self.Y = ep.Y

    def __call__(self, win, k: int, tau: float) -> torch.Tensor:
        return self.Y[:, k] if tau == 0 else (1 - tau) * self.Y[:, k] + tau * self.Y[:, k + 1]


# ---------------------------------------------------------------- dynamics
def _inertia(ref: torch.Tensor):
    J = torch.tensor(np.diag(INERTIA).copy(), dtype=ref.dtype, device=ref.device)
    return J, 1.0 / J


def _derivs(p, v, q, w, FM, translational):
    J, Jinv = _inertia(v)
    F, M = FM[:, :3], FM[:, 3:]
    vdot = F / MASS + gravity_body(q) - torch.cross(w, v, dim=-1)
    pdot = (rot(q) @ v.unsqueeze(-1)).squeeze(-1)
    if translational:
        return pdot, vdot, None, None
    qdot = 0.5 * quat_mul(q, torch.cat([w, torch.zeros_like(w[:, :1])], -1))
    wdot = Jinv * (M - torch.cross(w, J * w, dim=-1))
    return pdot, vdot, qdot, wdot


def _row(v, w, q, act):
    return torch.cat([v, w, gravity_body(q), act], -1)


def rollout(force, ep: Episodes, n_steps: int | None = None, integrator: str = "rk4",
            mode: str = "full", v0=None, w0=None, detach_every: int | None = None) -> dict:
    """Simulate n_steps steps from sample ep.H. Returns simulated and recorded trajectories
    (B, n_steps+1, ...) plus per-step errors. v0 / w0 override the initial body velocity / rates
    (e.g. a tensor with requires_grad for sensitivity checks).

    detach_every = k (truncated BPTT): after every k steps the simulated state (and the simulated
    rows of the model's input history) is detached, so gradients flow back at most k steps through
    the dynamics. The forward values are unchanged. None (default) = full backpropagation."""
    if detach_every is not None and int(detach_every) < 1:
        raise ValueError("detach_every must be >= 1 or None")
    if integrator not in ("rk4", "euler"):
        raise ValueError(integrator)
    if mode not in ("translational", "full"):
        raise ValueError(mode)
    trans = mode == "translational"
    H, dt = ep.H, ep.dt
    n_steps = n_steps if n_steps is not None else ep.feats.shape[1] - H - 1
    if H + n_steps >= ep.feats.shape[1]:
        raise ValueError("episode too short for n_steps")
    X = ep.feats
    act = X[:, :, ACT]
    p = ep.pos[:, H]
    v = X[:, H, V] if v0 is None else v0
    q = ep.q[:, H]
    w = X[:, H, W] if w0 is None else w0
    hist = [X[:, j] for j in range(H)]                       # recorded history rows, k0-H .. k0-1
    hist.append(_row(v, w, q, act[:, H]))                     # current row uses the simulated state
    P, Vs, Q, Ws = [p], [v], [q], [w]

    def att(k, tau):                                         # attitude / rates (translational: data)
        if tau == 0:
            return ep.q[:, k], X[:, k, W]
        qa, qb = ep.q[:, k], ep.q[:, k + 1]
        qb = torch.where(((qa * qb).sum(-1, keepdim=True) < 0), -qb, qb)
        return quat_normalise((1 - tau) * qa + tau * qb), (1 - tau) * X[:, k, W] + tau * X[:, k + 1, W]

    def window_with(last):
        return torch.stack(hist[-(WINDOW - 1):] + [last], 1)

    for k in range(H, H + n_steps):
        if integrator == "euler":
            FM = force(torch.stack(hist[-WINDOW:], 1), k, 0.0)
            if trans:
                _, vdot, _, _ = _derivs(p, v, q, w, FM, True)
                v = v + dt * vdot
                q, w = ep.q[:, k + 1], X[:, k + 1, W]
            else:
                _, vdot, _, wdot = _derivs(p, v, q, w, FM, False)
                w = w + dt * wdot
                v = v + dt * vdot
                q = quat_normalise(quat_mul(q, quat_exp(0.5 * dt * w)))
            p = p + dt * (rot(q) @ v.unsqueeze(-1)).squeeze(-1)
        else:
            def stage(ps, vs, qs, ws, tau):
                if trans:
                    qs, ws = att(k, tau)
                a = act[:, k] if tau == 0 else (1 - tau) * act[:, k] + tau * act[:, k + 1]
                win = torch.stack(hist[-WINDOW:], 1) if tau == 0 else window_with(_row(vs, ws, qs, a))
                return _derivs(ps, vs, qs, ws, force(win, k, tau), trans)
            k1 = stage(p, v, q, w, 0.0)
            if trans:
                k2 = stage(p + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], None, None, 0.5)
                k3 = stage(p + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], None, None, 0.5)
                k4 = stage(p + dt * k3[0], v + dt * k3[1], None, None, 1.0)
                q, w = ep.q[:, k + 1], X[:, k + 1, W]
            else:
                k2 = stage(p + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], q + 0.5 * dt * k1[2], w + 0.5 * dt * k1[3], 0.5)
                k3 = stage(p + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], q + 0.5 * dt * k2[2], w + 0.5 * dt * k2[3], 0.5)
                k4 = stage(p + dt * k3[0], v + dt * k3[1], q + dt * k3[2], w + dt * k3[3], 1.0)
                q = quat_normalise(q + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))
                w = w + dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            p = p + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if detach_every is not None and (k - H + 1) % int(detach_every) == 0:
            p, v, q, w = p.detach(), v.detach(), q.detach(), w.detach()
            hist = [h.detach() for h in hist]
        hist.append(_row(v, w, q, act[:, k + 1]))
        P.append(p); Vs.append(v); Q.append(q); Ws.append(w)

    sl = slice(H, H + n_steps + 1)
    out = {"pos": torch.stack(P, 1), "vel": torch.stack(Vs, 1), "q": torch.stack(Q, 1), "w": torch.stack(Ws, 1),
           "pos_true": ep.pos[:, sl], "vel_true": X[:, sl, V], "q_true": ep.q[:, sl], "w_true": X[:, sl, W],
           "t": torch.arange(n_steps + 1, dtype=X.dtype, device=X.device) * dt,
           "integrator": integrator, "mode": mode}
    # velocity error compared in the world frame (body velocities live in different frames once q drifts)
    vw = (rot(out["q"]) @ out["vel"].unsqueeze(-1)).squeeze(-1)
    vw_true = (rot(out["q_true"]) @ out["vel_true"].unsqueeze(-1)).squeeze(-1)
    out["pos_err"] = (out["pos"] - out["pos_true"]).norm(dim=-1)
    out["vel_err"] = (vw - vw_true).norm(dim=-1)
    out["att_err"] = quat_angle(out["q"], out["q_true"])
    out["w_err"] = (out["w"] - out["w_true"]).norm(dim=-1)
    return out


def physics_model_from_coefs(coefs) -> torch.nn.Module:
    """P0b as a ForceModel from the coefficient lists in results/val_baselines.json (P0b_coefs)."""
    from src.models import build_model
    m = build_model({"type": "physics"})
    for k, c in enumerate(coefs):
        getattr(m.prior, f"coef{k}").copy_(torch.as_tensor(c, dtype=torch.float32))
    m.prior.fitted.fill_(True)
    return m.eval()
