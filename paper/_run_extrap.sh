#!/usr/bin/env bash
# pueue entry point for the budget-extrapolation campaign (operator learning).
# Usage: _run_extrap.sh <task> <model> <device> <shard> [schedulers]
#   e.g. _run_extrap.sh integral DeepONet cuda:0 1/6
#        _run_extrap.sh integral DeepONet cuda:0 0/2 POW,WSQD,D2Z
# Sharding is by scheduler, so one scheduler's HPO and its deployments stay on one GPU and
# two shards never write the same Optuna study.
set -eu
cd "$HOME/HyperbolicLR/paper"
task="$1"; model="$2"; device="$3"; shard="$4"
scheds="${5:-N,E,P,C,L,H,EH,WC,WSD,WEH,SF}"
exec .venv/bin/python phase2_orchestrate_extrap.py \
    --task "$task" --model "$model" --device "$device" --shard "$shard" \
    --schedulers "$scheds" \
    --tune-budgets 10,50 \
    --deploy-budgets 50,100,200 \
    --seeds 89,231,928 \
    --n-trials 25
