#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
set -uo pipefail

usage() {
    echo "usage: $0 [attempts]  (default 5; env REPRO_REQUESTS, REPRO_LOG_DIR, REPRO_RESET=1 to tt-smi -glx_reset_auto first)"
}
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

ATTEMPTS="${1:-5}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"
[[ -z "${VIRTUAL_ENV:-}" && -f python_env/bin/activate ]] && source python_env/bin/activate
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO}" PYTHONPATH="${PYTHONPATH:-$REPO}"
export REPRO_REQUESTS="${REPRO_REQUESTS:-100}"
LOG_DIR="${REPRO_LOG_DIR:-$REPO/generated/ring_mla_collapse_repro/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"

echo "host=$(hostname) attempts=$ATTEMPTS requests=$REPRO_REQUESTS logs=$LOG_DIR"
if [[ "${REPRO_RESET:-0}" == "1" ]]; then
    tt-smi -glx_reset_auto || { echo "reset failed"; exit 2; }
fi

for i in $(seq 1 "$ATTEMPTS"); do
    log="$LOG_DIR/attempt_$i.log"
    echo "=== attempt $i/$ATTEMPTS $(date -u +%H:%M:%S) -> $log"
    pytest -svq models/demos/deepseek_v3_d_p/tests/test_ring_mla_collapse_repro.py > "$log" 2>&1
    rc=$?
    probes=$(grep -ac ' ring_out med=' "$log")
    if grep -aq 'stem probe saw exploded rows' "$log"; then
        echo "=== HIT on attempt $i after $probes ring_out probes"
        grep -a '\[repro\] HIT' "$log" | sed 's/.*\[repro\]/  /'
        grep -a '\[stem_probe\] layer' "$log" | grep -a 'total=[1-9]' | sed -E 's/.*\] layer/  layer/; s/ band_max=.*valid=/ valid=/; s/ nrows=.*//'
        grep -a '\[repro\] device ids' "$log" | sed 's/.*\[repro\]/  /'
        exit 1
    fi
    if [[ $rc -ne 0 ]]; then
        echo "=== attempt $i errored (rc=$rc, no probe hit); tail of $log:"
        tail -20 "$log"
        exit 2
    fi
    echo "=== attempt $i clean ($probes ring_out probes)"
done
echo "=== no hit in $ATTEMPTS attempts"
