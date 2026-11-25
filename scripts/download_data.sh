#!/usr/bin/env bash
# Download the NeuroBEM *processed* data (plus the small metadata files) into data/raw/.
# Source: https://download.ifi.uzh.ch/rpg/NeuroBEM/  (see data/README.md)
#
# Fetches:  processed_data.zip (~627 MB), Flights.txt, Readme.md, testset.txt
# Skips:    raw_data.zip, predictions.tar.xz, pdf.zip, code.tar.xz
#
# Usage:    bash scripts/download_data.sh            # download + verify + unpack
#           KEEP_ZIP=1 bash scripts/download_data.sh # keep the zip after unpacking
# Re-running is safe: finished steps are skipped and an interrupted download resumes.
# Needs:    bash, curl, Python 3 (macOS/Linux terminal, or Git Bash / WSL on Windows).
set -euo pipefail

BASE_URL="${NEUROBEM_URL:-https://download.ifi.uzh.ch/rpg/NeuroBEM}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
ZIP="$RAW/processed_data.zip"
DONE_MARK="$RAW/.processed_data_unpacked"

mkdir -p "$RAW"

# Pick a Python interpreter (Windows/Git Bash often only has "python").
if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys' >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "ERROR: Python 3 not found on PATH (activate your venv first)." >&2; exit 1; fi
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found." >&2; exit 1; }

fetch() {  # fetch <remote-name> <local-path>   (resumes a partial file; retried)
  local rc=0
  curl -fL --retry 5 --retry-delay 5 -C - -o "$2" "$BASE_URL/$1" || rc=$?
  if [[ $rc -eq 33 ]]; then  # server refused byte ranges -> start this file over
    echo "    server can't resume; restarting $1 from scratch"
    rm -f "$2"; rc=0
    curl -fL --retry 5 --retry-delay 5 -o "$2" "$BASE_URL/$1" || rc=$?
  fi
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: download of $1 failed (curl exit $rc). Re-run the script to resume." >&2
    exit 1
  fi
}

echo "==> Metadata files"
for f in Flights.txt Readme.md testset.txt; do
  if [[ -s "$RAW/$f" ]]; then echo "    $f already present"; else fetch "$f" "$RAW/$f"; fi
done

if [[ -f "$DONE_MARK" ]]; then
  echo "==> processed_data already unpacked ($DONE_MARK exists). Nothing to do."
  echo "    Run: $PY scripts/verify_data.py"
  exit 0
fi

echo "==> processed_data.zip"
REMOTE_BYTES="$(curl -sIL "$BASE_URL/processed_data.zip" | tr -d '\r' \
                | awk 'tolower($1)=="content-length:" {n=$2} END {print n+0}')"
LOCAL_BYTES=0; [[ -f "$ZIP" ]] && LOCAL_BYTES="$(wc -c < "$ZIP" | tr -d ' ')"

if [[ "$REMOTE_BYTES" -gt 0 && "$LOCAL_BYTES" -eq "$REMOTE_BYTES" ]]; then
  echo "    already fully downloaded ($LOCAL_BYTES bytes)"
else
  fetch processed_data.zip "$ZIP"
  LOCAL_BYTES="$(wc -c < "$ZIP" | tr -d ' ')"
fi

if [[ "$REMOTE_BYTES" -gt 0 && "$LOCAL_BYTES" -ne "$REMOTE_BYTES" ]]; then
  echo "ERROR: size mismatch: local $LOCAL_BYTES vs server $REMOTE_BYTES bytes. Re-run to resume." >&2
  exit 1
fi
echo "    size OK: $LOCAL_BYTES bytes (server: $REMOTE_BYTES)"

echo "==> Testing zip integrity (CRC of every member; takes a minute)"
"$PY" - "$ZIP" <<'EOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    bad = z.testzip()
    if bad:
        sys.exit(f"ERROR: corrupt member in zip: {bad}  -> delete the zip and re-run")
    n = sum(1 for i in z.infolist() if i.filename.lower().endswith(".csv"))
    print(f"    zip OK: {len(z.infolist())} entries, {n} CSV files")
EOF

echo "==> Unpacking into data/raw/"
"$PY" -m zipfile -e "$ZIP" "$RAW"
touch "$DONE_MARK"

if [[ "${KEEP_ZIP:-0}" != "1" ]]; then rm -f "$ZIP"; echo "    removed zip (set KEEP_ZIP=1 to keep it)"; fi

echo "==> Done. Now run:  $PY scripts/verify_data.py"
