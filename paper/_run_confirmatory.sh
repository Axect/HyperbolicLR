#!/usr/bin/env bash
# Run one deterministic lane of the frozen v13 confirmatory campaign.
# Usage: _run_confirmatory.sh <phase> <setting> <pilot|full> <horizons> <device> <lane/count>
set -euo pipefail

root="${HLR_ROOT:-$HOME/HyperbolicLR/paper}"
cd "$root"

phase="$1"
setting="$2"
replicates="$3"
horizons="$4"
device="$5"
shard="$6"
lane_index="${shard%/*}"
lane_count="${shard#*/}"

case "$setting" in
    S1|S2|S4)
        data_manifest="protocol/operator_v13.json"
        data_root="data_confirmatory_v13"
        ;;
    S3|S5)
        data_manifest="protocol/cifar10_v13.json"
        data_root="data"
        ;;
    *)
        echo "unknown setting: $setting" >&2
        exit 2
        ;;
esac

exec .venv/bin/python confirmatory_orchestrator.py run \
    --protocol protocol/confirmatory_v13.json \
    --data-manifest "$data_manifest" \
    --data-root "$data_root" \
    --output-root results_confirmatory \
    --phase "$phase" \
    --setting "$setting" \
    --replicates "$replicates" \
    --horizons "$horizons" \
    --device "$device" \
    --lane-index "$lane_index" \
    --lane-count "$lane_count"
