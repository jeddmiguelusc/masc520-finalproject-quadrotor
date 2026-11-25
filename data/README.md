# Data: NeuroBEM quadrotor flight dataset

This project uses the **processed** flight data released with NeuroBEM (RSS 2021) by the
Robotics and Perception Group (RPG), University of Zurich.

## Source

- Download server: <https://download.ifi.uzh.ch/rpg/NeuroBEM/>
- Project page: <https://rpg.ifi.uzh.ch/NeuroBEM.html>
- Paper: <https://arxiv.org/abs/2106.08015>

What the server's Readme says about the data: about 1 h 15 min of agile flights of a custom
quadrotor in a large Vicon motion-capture hangar, logged at 400 Hz. Inertial frame z-up; body
frame front-left-up (FLU). Mass 0.772 kg, inertia diag(0.0025, 0.0021, 0.0043) kg·m².
Each flight is split into one or more segment CSVs (non-flying parts and Vicon dropouts removed);
`Flights.txt` lists all flights. Each processed CSV has 29 columns:

| Cols | Content |
| --- | --- |
| 1 | t |
| 2–4 | angular acceleration, body x y z |
| 5–7 | angular velocity, body x y z |
| 8–11 | quaternion qx qy qz qw |
| 12–14 | acceleration, body x y z |
| 15–17 | velocity, body x y z |
| 18–20 | position x y z |
| 21–24 | motor speeds (back-right, front-right, back-left, front-left) |
| 25–28 | motor speed derivatives |
| 29 | battery voltage |

Quaternion convention and whether acceleration is kinematic or specific force are **not** stated;
both are settled by tests in CP1 (`tests/test_frames.py`).

## Citation

Bauersfeld, L., Kaufmann, E., Foehn, P., Sun, S., and Scaramuzza, D. "NeuroBEM: Hybrid
Aerodynamic Quadrotor Model." *Robotics: Science and Systems (RSS) XVII*, 2021.
arXiv:2106.08015.

```bibtex
@inproceedings{bauersfeld2021neurobem,
  title     = {{NeuroBEM}: Hybrid Aerodynamic Quadrotor Model},
  author    = {Bauersfeld, Leonard and Kaufmann, Elia and Foehn, Philipp and Sun, Sihao and Scaramuzza, Davide},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2021},
  doi       = {10.15607/RSS.2021.XVII.042}
}
```

## Why the data is not in this repo

1. **No license.** The download server and its Readme state no license or redistribution terms,
   so copying the files into a public repo isn't clearly permitted. Linking to the authors'
   server and scripting the download is.
2. **Size.** `processed_data.zip` alone is ~627 MB, above GitHub's file limits.
3. **Provenance.** Everyone downloads the same files from the original source, so results are
   traceable to the authors' release.

`data/raw/` is git-ignored. Only derived, small artifacts (`data/splits.json`) are committed.

## Download

From the repo root (macOS/Linux terminal, or Git Bash / WSL on Windows), with the venv active:

```bash
bash scripts/download_data.sh      # ~627 MB; resumes if interrupted; safe to re-run
python scripts/verify_data.py      # must end with "ALL CHECKS PASSED"
```

The script fetches only `processed_data.zip` plus the small metadata files `Flights.txt`,
`Readme.md` and `testset.txt`. It checks the size against the server, CRC-tests the zip,
unpacks into `data/raw/`, and deletes the zip (`KEEP_ZIP=1` keeps it). It does **not** fetch
`raw_data.zip`, `predictions.tar.xz`, `pdf.zip` or `code.tar.xz`.

Note: the server's `testset.txt` lists the authors' test flights; this project's own splits are
defined in `data/splits.json` (CP2).
