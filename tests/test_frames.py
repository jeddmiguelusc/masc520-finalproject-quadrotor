"""CP1 frame tests. Run:  pytest tests/test_frames.py -v -s

Decides (1) the quaternion convention and (2) whether acc_* is kinematic or specific force,
and writes every number to results/frame_tests.json. Uses a fixed sample: every 4th segment
that is NOT in the server's testset.txt (no test data touched).

Test 1 has two parts:
  1a  gravity rotated into body vs acc at near-hover (the test as designed). Hover is nearly
      level, so this separates element ORDER (xyzw vs wxyz) but is too weak to separate
      DIRECTION (body->world vs world->body); that is recorded, not hidden.
  1b  d(pos)/dt vs R_bw @ vel_b over all sampled data - independent of acc, decides DIRECTION.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.data import frames as F
from src.data.load import RAW_DIR, REPO, load_all, read_server_testset, segment_index

RESULTS = REPO / "results" / "frame_tests.json"
SAMPLE_EVERY = 4
REPORT: dict = {}

pytestmark = pytest.mark.skipif(not (RAW_DIR / "processed_data").exists(),
                                reason="data not downloaded (scripts/download_data.sh)")


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


@pytest.fixture(scope="module")
def idx():
    return segment_index()


@pytest.fixture(scope="module")
def sample(idx):
    s = idx[~idx.in_server_testset].iloc[::SAMPLE_EVERY]
    REPORT["sample"] = {"rule": f"every {SAMPLE_EVERY}th non-testset segment (sorted by flight, seg)",
                        "n_segments": int(len(s)), "segment_ids": s.segment_id.tolist()}
    return load_all(segments=s)


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    REPORT["git_hash"] = _git_hash()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(REPORT, indent=2))
    print(f"\nwrote {RESULTS.relative_to(REPO)}")


# ------------------------------------------------------------------ 0. mapping sanity
def test_segment_to_flight_mapping(idx):
    from src.data.load import parse_flights
    flights = parse_flights()
    test_ids = read_server_testset()
    REPORT["mapping"] = {
        "n_segment_csvs": int(len(idx)), "n_flights_in_flights_txt": int(len(flights)),
        "n_flights_with_segments": int(idx.flight_id.nunique()),
        "n_server_testset_segments": int(len(test_ids)),
        "server_testset_ids_missing": sorted(test_ids - set(idx.segment_id)),
        "trajectory_type_counts": flights.trajectory_type.value_counts().to_dict(),
    }
    assert idx.segment_id.is_unique
    assert idx.flight_id.isin(flights.flight_id).all()
    assert not REPORT["mapping"]["server_testset_ids_missing"]
    assert (flights.trajectory_type != "other").all(), flights[flights.trajectory_type == "other"]


def test_quaternions_unit_norm(sample):
    q = sample[["qx", "qy", "qz", "qw"]].to_numpy()
    e = np.abs(np.linalg.norm(q, axis=1) - 1)
    REPORT["quat_norm_abs_err"] = {"median": float(np.median(e)), "p99": float(np.percentile(e, 99)),
                                   "max": float(e.max()), "frac_over_1e-3": float((e > 1e-3).mean())}
    print("\n[0] | |q| - 1 |:", REPORT["quat_norm_abs_err"])
    # Not exactly unit (filtered data, max ~1%); frames._rot_hamilton renormalises.
    assert np.percentile(e, 99) < 1e-3 and e.max() < 0.05


# ------------------------------------------------------------------ 1a. gravity at hover
def test_1a_quaternion_order_from_hover_gravity(sample):
    r = F.quaternion_gravity_check(sample)
    REPORT["test1a_hover_gravity"] = r
    c = r["candidates"]
    print("\n[1a] near-hover samples:", r["n_hover"], "in", r["n_hover_segments"], "segments")
    for k, v in c.items():
        print(f"     {k:30s} rms_xy={v['rms_xy_residual']:.3f}  median tilt={v['median_tilt_deg']:.2f} deg")
    print(f"     level-attitude baseline rms_xy={r['baseline_level_rms_xy']:.3f}")
    assert r["n_hover"] >= 1000
    xyzw = min(c["xyzw_hamilton_body_to_world"]["rms_xy_residual"], c["xyzw_hamilton_world_to_body"]["rms_xy_residual"])
    wxyz = min(c["wxyz_hamilton_body_to_world"]["rms_xy_residual"], c["wxyz_hamilton_world_to_body"]["rms_xy_residual"])
    assert xyzw < 0.5 and wxyz > 2 * xyzw + 0.5, "columns are not (x, y, z, w)"
    a, b = (c["xyzw_hamilton_body_to_world"]["rms_xy_residual"], c["xyzw_hamilton_world_to_body"]["rms_xy_residual"])
    REPORT["test1a_direction_resolvable"] = bool(max(a, b) > 3 * min(a, b))


# ------------------------------------------------------------------ 1b. velocity kinematics
def test_1b_quaternion_direction_from_velocity(sample):
    r = F.velocity_kinematics_check(sample)
    REPORT["test1b_velocity_kinematics"] = r
    ranked = sorted(r["candidates"].items(), key=lambda kv: kv[1]["rms_residual"])
    print("\n[1b] rms |dpos/dt - R vel| [m/s]:")
    for k, v in ranked:
        print(f"     {k:30s} {v['rms_residual']:.4f}")
    print(f"     vel already world-frame      {r['vel_is_world_rms']:.4f}")
    best, second = ranked[0], ranked[1]
    REPORT["quat_convention_winner"] = best[0]
    assert best[1]["rms_residual"] < 0.1
    assert second[1]["rms_residual"] > 10 * best[1]["rms_residual"]
    assert r["vel_is_world_rms"] > 10 * best[1]["rms_residual"], "vel_* looks world-frame"
    assert best[0] == F.QUAT_CONVENTION, f"data says {best[0]}, frames.py says {F.QUAT_CONVENTION}"


# ------------------------------------------------------------------ 2. acc type
def test_2_acc_is_kinematic_or_specific_force(sample):
    r = F.acc_type_check(sample, F.QUAT_CONVENTION)
    hover = F.quaternion_gravity_check(sample)["median_acc_z_hover"]
    r["median_acc_z_hover"] = hover
    REPORT["test2_acc_type"] = r
    print(f"\n[2] rms |acc - R^T dv_w/dt|       (kinematic)      = {r['kinematic_rms']:.3f} m/s^2")
    print(f"    rms |acc - R^T (dv_w/dt - g)| (specific force) = {r['specific_force_rms']:.3f} m/s^2")
    print(f"    z scale (spec. force) = {r['specific_force_z_scale']:.4f}; median acc_z at hover = {hover:.3f}")
    is_sf = r["specific_force_rms"] < r["kinematic_rms"]
    REPORT["acc_is_specific_force"] = bool(is_sf)
    assert min(r["specific_force_rms"], r["kinematic_rms"]) < 1.0
    assert max(r["specific_force_rms"], r["kinematic_rms"]) > 5 * min(r["specific_force_rms"], r["kinematic_rms"])
    if is_sf:
        assert abs(r["specific_force_z_scale"] - 1) < 0.02
        assert abs(hover - F.G) < 0.2
    assert is_sf == F.ACC_IS_SPECIFIC_FORCE
