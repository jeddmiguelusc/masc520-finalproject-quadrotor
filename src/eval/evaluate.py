"""Shared scoring for validation (and, at CP7 only, test) predictions.

Every model is scored on the same samples: the ends of 20-sample windows inside contact-trimmed
pieces (src/data/dataset.WindowSet). `score` returns, for the whole set and for |v| > 5 m/s:
point metrics with flight-bootstrap 95% CIs, the per-flight SSE (so paired comparisons can be
made later, also across scripts), and a per-speed-bin table.
"""
from __future__ import annotations

import numpy as np

from src.data.dataset import WindowSet
from src.eval import metrics as M

SUBSETS = {"all": (0.0, np.inf), "gt5": (5.0, np.inf)}


def scored_rows(pieces, window: int = 20):
    """Inputs, targets, flight id and speed at every scored sample of a list of pieces."""
    ws = WindowSet(pieces, window)
    e = ws.ends
    flights = np.array([M.flight_of_piece(ws.piece_ids[k]) for k in ws.owner])
    X, Y = ws.X[e], ws.Y[e]
    return {"ws": ws, "X": X, "Y": Y, "flights": flights, "speed": np.linalg.norm(X[:, 0:3], axis=1)}


def sse_to_json(s: M.FlightSSE) -> dict:
    return {"flights": s.flights.tolist(), "sse": s.sse.tolist(), "n": s.n.tolist()}


def sse_from_json(d: dict) -> M.FlightSSE:
    return M.FlightSSE(np.array(d["flights"]), np.array(d["sse"]), np.array(d["n"]))


def score(rows: dict, P: np.ndarray, b: int = 2000) -> dict:
    out = {}
    for name, (lo, hi) in SUBSETS.items():
        m = (rows["speed"] > lo) & (rows["speed"] <= hi) if lo > 0 else np.ones(len(P), bool)
        if not m.any():
            out[name] = {"n": 0}
            continue
        s = M.flight_sse(rows["Y"][m], P[m], rows["flights"][m])
        out[name] = {"n": int(m.sum()), "n_flights": int(len(s.flights)),
                     "ci95": M.bootstrap_ci(s, b=b), "flight_sse": sse_to_json(s)}
    out["by_speed"] = M.by_speed(rows["Y"], P, rows["speed"])
    return out


def paired(a: dict, b: dict, subset: str = "all", nb: int = 2000) -> dict:
    """Paired flight-bootstrap CI of metric(a) - metric(b) from two `score` outputs."""
    return M.paired_diff_ci(sse_from_json(a[subset]["flight_sse"]), sse_from_json(b[subset]["flight_sse"]), b=nb)


def short(sc: dict, subset: str = "all") -> str:
    c = sc[subset]["ci95"]
    return (f"F {c['F'][0]:.3f} [{c['F'][1]:.3f},{c['F'][2]:.3f}] N  Fxy {c['Fxy'][0]:.3f} Fz {c['Fz'][0]:.3f}  "
            f"M {c['M'][0]:.4f} [{c['M'][1]:.4f},{c['M'][2]:.4f}] N·m  ({c['F_pct_hover'][0]:.1f}% hover)")
