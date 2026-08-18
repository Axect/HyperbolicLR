#!/usr/bin/env bash
# pueue entry point for the E1 kill-rate grid.
# Usage: _run_e1.sh <task> <device> <shard>     e.g. _run_e1.sh cifar10 cuda:0 0/2
# The device is an argument, not a property of the pueue group, so the group and the
# device must be kept consistent by the caller (darkmatter convention: HLR_0 -> cuda:1,
# HLR_1 -> cuda:0).
set -eu
cd "$HOME/HyperbolicLR/paper"
task="$1"; device="$2"; shard="$3"
exec .venv/bin/python phase2_orchestrate_e1.py \
    --task "$task" --device "$device" --shard "$shard"
