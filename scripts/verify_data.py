"""Check that data/raw/ holds a complete NeuroBEM processed_data download.

Run from the repo root:  python scripts/verify_data.py
Exit code 0 = all checks passed. Paste the full output into the CP1 chat.

Checks against facts in the dataset Readme: 29 columns per CSV, 400 Hz sampling,
about 1 h 15 min of flight in total, and Flights.txt present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
N_COLS = 29
DT = 1 / 400  # s
EXPECTED_TOTAL_MIN = 75  # "about 1h15min" per the Readme


def main() -> int:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= cond

    for f in ("Flights.txt", "Readme.md", "testset.txt"):
        check((RAW / f).is_file(), f"{f} present")

    csvs = sorted(RAW.rglob("*.csv"))
    check(len(csvs) > 0, f"found {len(csvs)} CSV files under data/raw/")
    if not csvs:
        return 1
    print(f"     first: {csvs[0].relative_to(RAW)}")
    print(f"     last:  {csvs[-1].relative_to(RAW)}")

    head = pd.read_csv(csvs[0], nrows=3)
    print(f"     header of first file: {list(head.columns)}")

    bad_cols, total_s, total_rows, dts, nan_files = [], 0.0, 0, [], 0
    for p in csvs:
        df = pd.read_csv(p)
        if df.shape[1] != N_COLS:
            bad_cols.append((p.name, df.shape[1]))
            continue
        t = df.iloc[:, 0].to_numpy(dtype=float)
        total_rows += len(df)
        if len(t) > 1:
            total_s += t[-1] - t[0]
            dts.append(np.median(np.diff(t)))
        nan_files += int(df.isna().any().any())

    check(not bad_cols, f"every CSV has {N_COLS} columns"
          + (f" (bad: {bad_cols[:5]})" if bad_cols else ""))
    med_dt = float(np.median(dts)) if dts else float("nan")
    check(abs(med_dt - DT) < 1e-4, f"median sample interval {med_dt*1e3:.3f} ms (expect 2.5 ms = 400 Hz)")
    total_min = total_s / 60
    # Informational only: segments drop non-flying parts and Vicon dropouts, so this can
    # sit below the Readme's total. Far below (< 50%) suggests a partial unpack.
    check(total_min >= 0.5 * EXPECTED_TOTAL_MIN,
          f"total segment duration {total_min:.1f} min (Readme: ~{EXPECTED_TOTAL_MIN} min of flights)")
    print(f"     total rows: {total_rows:,}")
    print(f"INFO {nan_files} file(s) contain NaNs" if nan_files else "PASS no NaNs")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
