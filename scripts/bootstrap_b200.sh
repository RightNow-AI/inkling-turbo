#!/usr/bin/env bash
# First-contact bootstrap for a fresh Lambda B200 box (runs ON the instance).
# Installs the pinned fork, applies toolchain fixes if needed, runs the FA4
# rel-attention parity suite (sheared path on sm_100), captures evidence.
set -euo pipefail
exec 2>&1

PIN=850295881041754184717804104fcaadd2b2129e

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== toolchain ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"

cd ~
if [ ! -d vllm ]; then
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git vllm
fi
cd vllm
git checkout -q "$PIN"
echo "vllm @ $(git rev-parse --short HEAD)"

uv venv --python 3.12 >/dev/null 2>&1 || true
source .venv/bin/activate
echo "=== installing (precompiled) ==="
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto 2>&1 | tail -2

python - <<'EOF'
import torch
print("torch:", torch.__version__, "| capability:", torch.cuda.get_device_capability(0))
EOF

echo "=== tml_fa4 import check ==="
if ! python -c "import vllm.third_party.tml_fa4" 2>/dev/null; then
  echo "applying ThrMma/TiledMma cutlass-4.6.0 fix"
  sed -i 's/cute\.core\.ThrMma/cute.ThrMma/g; s/cute\.core\.TiledMma/cute.TiledMma/g' \
    vllm/third_party/tml_fa4/*.py
  python -c "import vllm.third_party.tml_fa4; print('tml_fa4 import OK after fix')"
else
  echo "tml_fa4 import OK (no fix needed)"
fi

echo "=== parity: FA4 rel attention (sheared path expected on sm_100) ==="
python ~/parity_fa4_rel.py || true

echo "=== BOOTSTRAP COMPLETE ==="
