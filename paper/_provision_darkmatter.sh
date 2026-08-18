#!/usr/bin/env bash
# Provision darkmatter for the E1 kill-rate grid: data, venv, sanity check.
# Idempotent. Run under nohup; progress goes to ~/HyperbolicLR/provision.log.
set -u
cd "$HOME/HyperbolicLR/paper" || exit 1
echo "=== provision start $(date -Is) ==="

cd data || exit 1
get() {  # url out md5
    if [ -f "$3.ok" ]; then echo "already verified: $2"; return 0; fi
    echo "downloading $2"
    curl -sL --retry 3 --retry-delay 5 -o "$2" "$1" || { echo "DOWNLOAD FAILED $2"; return 1; }
    have=$(md5sum "$2" | cut -d' ' -f1)
    if [ "$have" != "$3" ]; then echo "MD5 MISMATCH $2: got $have want $3"; return 1; fi
    touch "$3.ok"; echo "verified $2"
}
get "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" \
    cifar-10-python.tar.gz c58f30108f718f92721af3b95e74349a
get "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
    cifar-100-python.tar.gz eb9058c3a382ffc7106e4002c42a8d85
# Extract only what verified, so a half-downloaded tar cannot leave a corrupt data dir.
[ -f c58f30108f718f92721af3b95e74349a.ok ] && [ ! -d cifar-10-batches-py ] && tar xzf cifar-10-python.tar.gz
[ -f eb9058c3a382ffc7106e4002c42a8d85.ok ] && [ ! -d cifar-100-python ] && tar xzf cifar-100-python.tar.gz
du -sh cifar-10-batches-py cifar-100-python 2>/dev/null
cd ..

echo "=== venv ==="
export UV_CACHE_DIR="$HOME/.cache/uv"
if [ ! -x .venv/bin/python ]; then
    uv venv --clear --python 3.12 .venv || exit 1
fi
uv pip install --python .venv/bin/python -r requirements.txt >/tmp/hlr_pip.log 2>&1 \
    || { echo "PIP FAILED, tail:"; tail -20 /tmp/hlr_pip.log; exit 1; }
.venv/bin/python -c "
import torch, polars, optuna, torchvision, wandb
print('torch', torch.__version__, 'cuda_devices', torch.cuda.device_count())
import schedulefree; print('schedulefree ok')
"

echo "=== reuse partB parquets for the E1 grid (same protocol) ==="
mkdir -p results_e1/cifar10
n=0
for f in results_phase2/partB/*.parquet; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ -e "results_e1/cifar10/$b" ] || { cp "$f" "results_e1/cifar10/$b"; n=$((n+1)); }
done
echo "copied $n partB parquets into results_e1/cifar10 (total $(ls results_e1/cifar10 | wc -l))"

echo "=== sanity: 2-epoch probe on each GPU ==="
for dev in cuda:0 cuda:1; do
    /usr/bin/time -f "  probe $dev wall=%es" .venv/bin/python phase2_runner.py \
        --task cifar10 --scheduler C --lr 0.001 --infimum-lr 1e-6 --epochs 2 \
        --seed 999 --subset-ratio 0.2 --device "$dev" --out /tmp/hlr_probe 2>&1 | tail -3
done

echo "=== timing probe: one full 100-epoch cifar10 config ==="
/usr/bin/time -f "  cifar10 100ep wall=%es" .venv/bin/python phase2_runner.py \
    --task cifar10 --scheduler C --lr 0.001 --infimum-lr 1e-5 --epochs 100 \
    --seed 999 --subset-ratio 0.2 --device cuda:0 --out /tmp/hlr_probe 2>&1 | tail -3
echo "=== timing probe: one full 100-epoch osc config ==="
/usr/bin/time -f "  osc 100ep wall=%es" .venv/bin/python phase2_runner.py \
    --task osc --scheduler C --lr 0.001 --infimum-lr 1e-5 --epochs 100 \
    --seed 999 --subset-ratio 0.2 --device cuda:1 --out /tmp/hlr_probe 2>&1 | tail -3

echo "=== provision done $(date -Is) ==="
