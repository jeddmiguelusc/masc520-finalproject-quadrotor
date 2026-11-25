"""Find on-ground / idle samples and measure what they would do to the targets.

    python scripts/ground_contact_report.py      -> results/ground_contact.json

Two candidate rules are compared:
  motor rule   : any motor < 500 rad/s
  contact rule : mean motor < 700 rad/s AND speed < 0.3 m/s  (drone not flying, ground carries it)
Evidence of ground contact = thrust residual  m*acc_z - k*sum(w^2)  (k from the CP1 hover fit):
in flight it is ~0; on the ground acc_z still reads ~g but the rotors make little thrust.
Uses all segments for counts; target statistics use non-server-testset segments only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import frames as F  # noqa: E402
from src.data.load import REPO, load_all, segment_index  # noqa: E402

K = json.loads((REPO / "results" / "hover_thrust_fit.json").read_text())["k_through_origin_N_per_rad2s2"]


def runs(mask):
    """(start, end_exclusive) of consecutive True runs."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    idx = segment_index()
    df = load_all().reset_index(drop=True)
    test_ids = set(idx.loc[idx.in_server_testset, "segment_id"])
    seg = df["segment_id"].astype(str).to_numpy()
    is_test = np.isin(seg, list(test_ids))
    mot = df[["mot1", "mot2", "mot3", "mot4"]].to_numpy()
    speed = np.linalg.norm(F.vec(df, "vel"), axis=1)
    Fz = F.MASS * F.specific_force_body(df)[:, 2]
    resid = Fz - K * (mot ** 2).sum(1)

    motor_rule = mot.min(1) < 500
    contact_rule = (mot.mean(1) < 700) & (speed < 0.3)
    out = {"k_used": K, "n_samples": int(len(df)), "rules": {}}
    for name, m in [("motor_rule_min_lt_500", motor_rule), ("contact_rule", contact_rule)]:
        tr = m & ~is_test
        out["rules"][name] = {
            "n": int(m.sum()), "pct_of_all": float(100 * m.mean()), "n_segments": int(len(set(seg[m]))),
            "n_in_server_testset_segments": int((m & is_test).sum()),
            "n_with_speed_gt_1": int((m & (speed > 1)).sum()),
            "speed_pctl_10_50_90": np.percentile(speed[m], [10, 50, 90]).round(3).tolist(),
            "thrust_resid_median_N_nontest": float(np.median(resid[tr])),
        }
    out["thrust_resid_median_N_all_other_nontest"] = float(np.median(resid[~contact_rule & ~is_test]))

    # where the contact runs sit inside their segments
    run_rows = []
    for sid, g in df.groupby("segment_id", observed=True, sort=False):
        ii = g.index.to_numpy()
        m = contact_rule[ii]
        for a, b in runs(m):
            run_rows.append({"segment_id": str(sid), "start": int(a), "length": int(b - a),
                             "seg_len": int(len(ii)), "at_start": bool(a == 0), "at_end": bool(b == len(ii)),
                             "median_resid_N": float(np.median(resid[ii[a:b]])),
                             "pos_z_minus_seg_min_m": float(np.median(g["pos_z"].to_numpy()[a:b]) - g["pos_z"].min())})
    out["contact_runs"] = run_rows
    out["contact_runs_summary"] = {
        "n_runs": len(run_rows), "at_segment_start": sum(r["at_start"] for r in run_rows),
        "at_segment_end": sum(r["at_end"] for r in run_rows),
        "mid_segment": sum(not r["at_start"] and not r["at_end"] for r in run_rows),
        "runs_ge_20_samples": sum(r["length"] >= 20 for r in run_rows)}

    # how much these samples weigh in a squared-error metric (non-test data): if a model predicted
    # flight thrust there, the contact samples alone would add this much to the F_z sum of squares
    tr_c = contact_rule & ~is_test
    n_tr = int((~is_test).sum())
    out["error_budget_nontest"] = {
        "contact_sq_resid_sum": float(np.sum(resid[tr_c] ** 2)),
        "n_nontest": n_tr,
        "rmse_added_alone_N": float(np.sqrt(np.sum(resid[tr_c] ** 2) / n_tr)),
        "example_rmse_0p335_becomes_N": float(np.sqrt(0.335 ** 2 + np.sum(resid[tr_c] ** 2) / n_tr)),
    }
    path = REPO / "results" / "ground_contact.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "contact_runs"}, indent=2))
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
