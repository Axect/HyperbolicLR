"""
Re-run selected campaign cells with sharpness measurement enabled, so that the exposure above the
measured stability boundary can be paired with the damage that cell suffered.

The campaign runs (results_extrap/) carry no lambda_max column: measurement was deliberately off
so that the campaign could not be accused of perturbing its own trajectories. Measurement is
trajectory-neutral (verified: identical val_loss with the flag on and off), so re-running the same
configuration with the flag on reproduces the same curve and adds the sharpness columns.

Configurations are read back out of the campaign parquets themselves (lr0, infimum_lr,
sched_extra, batch_size), so this script cannot drift from what was actually run.

Output goes to a separate directory because phase2_runner does not encode the flag in the
filename, so writing into the campaign directory would just skip every file.

Usage:
  python 22_remeasure_sharpness.py --device cuda:0 --shard 0/3 [--budget 200] [--tune-budget 10]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import polars as pl

DEFAULT_SCHEDULERS = ["C", "L", "WSD", "P", "N", "EH", "POW", "E", "H"]


def cells(src, task_model, budget, tune_budget, schedulers, seeds):
    out = []
    for sched in schedulers:
        for seed in seeds:
            pat = os.path.join(src, f"{task_model}_{sched}_lr*_N{budget}_s{seed}_B{tune_budget}.parquet")
            hits = sorted(glob.glob(pat))
            if not hits:
                print(f"  no campaign run for {sched} N={budget} seed={seed} B={tune_budget}",
                      flush=True)
                continue
            df = pl.read_parquet(hits[0])
            extra = df["sched_extra"][0] if "sched_extra" in df.columns else "{}"
            out.append({
                "sched": sched,
                "seed": seed,
                "lr": float(df["lr0"][0]),
                "inf": float(df["infimum_lr"][0]),
                "extra": json.loads(extra or "{}"),
                "bs": int(df["batch_size"][0]) if "batch_size" in df.columns else 100,
                "src": os.path.basename(hits[0]),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results_extrap/integral-DeepONet")
    ap.add_argument("--task", default="integral")
    ap.add_argument("--model", default="DeepONet", help="model to RUN")
    ap.add_argument("--src-model", default=None,
                    help="model whose tuned configurations are reused (default: --model). "
                         "Set --model DeepONetLN --src-model DeepONet to run the normalization "
                         "ablation on exactly the configurations that collapsed without it.")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--tune-budget", type=int, default=10)
    ap.add_argument("--schedulers", default=",".join(DEFAULT_SCHEDULERS))
    ap.add_argument("--seeds", default="89,231,928")
    ap.add_argument("--every", type=int, default=5, help="measure sharpness every K epochs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=None,
                    help="default results_sharp/{task}-{model}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_model = args.src_model or args.model
    out = args.out or f"results_sharp/{args.task}-{args.model}"
    scheds = [s for s in args.schedulers.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    jobs = cells(args.src, f"{args.task}-{src_model}", args.budget, args.tune_budget,
                 scheds, seeds)
    i, n = (int(x) for x in args.shard.split("/"))
    mine = jobs[i::n]
    print(f"{len(jobs)} cells found, shard {i}/{n} takes {len(mine)}, device={args.device}, "
          f"out={out}", flush=True)
    if args.dry_run:
        for j in mine:
            print("   ", j["sched"], j["seed"], f"lr={j['lr']:.4g}", j["extra"])
        return

    t0 = time.time()
    for k, j in enumerate(mine, 1):
        cmd = [sys.executable, "phase2_runner.py",
               "--task", args.task, "--model", args.model,
               "--scheduler", j["sched"], "--lr", repr(j["lr"]),
               "--infimum-lr", repr(j["inf"]),
               "--epochs", str(args.budget), "--seed", str(j["seed"]),
               "--batch-size", str(j["bs"]),
               "--tune-budget", str(args.tune_budget),
               "--device", args.device, "--out", out,
               "--measure-sharpness", str(args.every), "--sharpness-precond"]
        for key, val in j["extra"].items():
            if key == "upper_bound":
                cmd += ["--upper-bound", str(val)]
            elif key == "cooldown_frac":
                cmd += ["--cooldown-frac", str(val)]
            else:
                cmd += ["--sched-extra", f"{key}={val}"]
        print(f"[{time.strftime('%H:%M:%S')}] ({k}/{len(mine)}, {(time.time()-t0)/60:.1f}m) "
              + " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}), continuing", flush=True)
    print(f"shard {i}/{n} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
