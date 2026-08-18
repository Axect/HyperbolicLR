#!/usr/bin/env bash
# Validate the code, environment, and frozen data before a v13 production lane starts.
set -euo pipefail

root="${HLR_ROOT:-$HOME/HyperbolicLR/paper}"
cd "$root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.13 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python confirmatory_protocol.py --validate protocol/confirmatory_v13.json
.venv/bin/python confirmatory_data.py validate \
    --protocol protocol/confirmatory_v13.json \
    --manifest protocol/operator_v13.json \
    --data-root data_confirmatory_v13
.venv/bin/python confirmatory_data.py validate \
    --protocol protocol/confirmatory_v13.json \
    --manifest protocol/cifar10_v13.json \
    --data-root data
.venv/bin/python -c "import torch; assert torch.cuda.device_count() >= 2; print('cuda_devices', torch.cuda.device_count())"

git diff --quiet
git diff --cached --quiet
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tracked working tree is not clean" >&2
    exit 1
fi

echo "confirmatory preflight passed commit=$(git rev-parse HEAD)"
