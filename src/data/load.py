"""Load NeuroBEM processed segment CSVs and map each segment to its flight.

Files are named  merged_<flight_id>_seg_<n>.csv  where <flight_id> is the timestamp used in
Flights.txt (e.g. 2021-02-03-13-43-38). segment_id is "<flight_id>_seg_<n>", the same format
as the server's testset.txt.

Usage:
    from src.data.load import segment_index, load_all
    idx = segment_index()            # one row per CSV
    df  = load_all()                 # all samples, with flight_id / segment_id columns
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw"
CACHE_DIR = REPO / "data" / "cache"          # git-ignored ("cache/" in .gitignore)

# Exact header of every processed CSV (checked on load) -> short canonical names.
RAW_COLUMNS = [
    "t",
    "ang acc x", "ang acc y", "ang acc z",
    "ang vel x", "ang vel y", "ang vel z",
    "quat x", "quat y", "quat z", "quat w",
    "acc x", "acc y", "acc z",
    "vel x", "vel y", "vel z",
    "pos x", "pos y", "pos z",
    "mot 1", "mot 2", "mot 3", "mot 4",
    "dmot 1", "dmot 2", "dmot 3", "dmot 4",
    "vbat",
]
COLUMNS = [
    "t",
    "ang_acc_x", "ang_acc_y", "ang_acc_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "qx", "qy", "qz", "qw",
    "acc_x", "acc_y", "acc_z",
    "vel_x", "vel_y", "vel_z",
    "pos_x", "pos_y", "pos_z",
    "mot1", "mot2", "mot3", "mot4",
    "dmot1", "dmot2", "dmot3", "dmot4",
    "vbat",
]
RENAME = dict(zip(RAW_COLUMNS, COLUMNS))

SEG_RE = re.compile(r"^merged_(?P<flight>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_seg_(?P<seg>\d+)\.csv$")
FLIGHT_RE = re.compile(r'dataset\s*=\s*"(?P<flight>[\d-]+)"\s*;\s*%\s*(?P<desc>.*)$')

# Order matters: first match wins ("3d circle" before "circle", "wobbly circle" before "circle").
TRAJ_TYPES = [
    ("wobbly circle", "wobbly_circle"),
    ("3d circle", "3d_circle"),
    ("circle", "circle"),          # includes "battery test NN, short circle" -> handled below
    ("linear oscillation", "linear_oscillation"),
    ("vertical oscillation", "vertical_oscillation"),
    ("lemniscate", "lemniscate"),
    ("random points", "random_points"),
    ("satellite", "satellite"),
    ("cpc", "cpc"),
    ("ellipse", "ellipse"),
]


def trajectory_type(description: str) -> str:
    """Map a Flights.txt comment to a coarse trajectory type."""
    d = description.lower().strip()
    if d.startswith("battery test"):
        return "battery_test_circle"
    for key, name in TRAJ_TYPES:
        if d.startswith(key):
            return name
    return "other"


def commanded_speed(description: str) -> float:
    """Commanded 'vel = X' from the Flights.txt comment, NaN if absent."""
    m = re.search(r"vel\s*=\s*([\d.]+)", description)
    return float(m.group(1)) if m else np.nan


def parse_flights(path: Path | str = RAW_DIR / "Flights.txt") -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text().splitlines():
        m = FLIGHT_RE.search(line.strip())
        if m:
            desc = m.group("desc").strip()
            rows.append(dict(flight_id=m.group("flight"), description=desc,
                             trajectory_type=trajectory_type(desc),
                             commanded_vel=commanded_speed(desc)))
    fl = pd.DataFrame(rows)
    if fl.flight_id.duplicated().any():
        raise ValueError(f"duplicate flights in Flights.txt: {fl.flight_id[fl.flight_id.duplicated()].tolist()}")
    return fl


def read_server_testset(path: Path | str = RAW_DIR / "testset.txt") -> set[str]:
    p = Path(path)
    return {s.strip() for s in p.read_text().split() if s.strip()} if p.exists() else set()


def segment_index(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """One row per processed CSV: path, flight_id, segment_id, seg_num, flight metadata.

    Raises if any CSV name doesn't parse or its flight is missing from Flights.txt.
    """
    raw_dir = Path(raw_dir)
    files = sorted((raw_dir / "processed_data").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {raw_dir / 'processed_data'} - run scripts/download_data.sh")
    rows, bad = [], []
    for f in files:
        m = SEG_RE.match(f.name)
        if not m:
            bad.append(f.name)
            continue
        rows.append(dict(path=str(f), flight_id=m.group("flight"), seg_num=int(m.group("seg")),
                         segment_id=f"{m.group('flight')}_seg_{m.group('seg')}"))
    if bad:
        raise ValueError(f"unparseable segment file names: {bad[:5]} ...")
    idx = pd.DataFrame(rows)
    flights = parse_flights(raw_dir / "Flights.txt")
    missing = sorted(set(idx.flight_id) - set(flights.flight_id))
    if missing:
        raise ValueError(f"segments whose flight is not in Flights.txt: {missing}")
    idx = idx.merge(flights, on="flight_id", how="left", validate="many_to_one")
    idx["in_server_testset"] = idx.segment_id.isin(read_server_testset(raw_dir / "testset.txt"))
    return idx.sort_values(["flight_id", "seg_num"]).reset_index(drop=True)


def load_segment(path: Path | str, flight_id: str | None = None, segment_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, dtype=np.float64, engine="c")
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"{path.name}: unexpected header {list(df.columns)}")
    df = df.rename(columns=RENAME)
    if flight_id is None or segment_id is None:
        m = SEG_RE.match(path.name)
        flight_id = m.group("flight")
        segment_id = f"{m.group('flight')}_seg_{m.group('seg')}"
    df.insert(0, "segment_id", segment_id)
    df.insert(0, "flight_id", flight_id)
    return df


def load_all(raw_dir: Path | str = RAW_DIR, segments: pd.DataFrame | None = None,
             use_cache: bool = True) -> pd.DataFrame:
    """Load segments (default: all) into one frame. Full load is cached as a pickle."""
    full = segments is None
    cache = CACHE_DIR / f"processed_all_pandas{pd.__version__}.pkl"   # pickles are not portable across pandas versions
    if full and use_cache and cache.exists():
        return pd.read_pickle(cache)
    if segments is None:
        segments = segment_index(raw_dir)
    parts = [load_segment(r.path, r.flight_id, r.segment_id) for r in segments.itertuples()]
    df = pd.concat(parts, ignore_index=True)
    df["flight_id"] = df["flight_id"].astype("category")
    df["segment_id"] = df["segment_id"].astype("category")
    if full and use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df
