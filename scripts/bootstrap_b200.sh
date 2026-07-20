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
# Triton JIT needs Python headers when the venv uses the system interpreter
sudo apt-get install -y python3.12-dev python3-dev >/dev/null 2>&1 || true

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

echo "=== cutlass-4.6.0 API drift fixes (idempotent) ==="
# ThrMma/TiledMma moved out of cute.core; make_fragment renamed to
# make_rmem_tensor. Evidence: journal/local-tier-bringup.md + H100 session 1.
sed -i 's/cute\.core\.ThrMma/cute.ThrMma/g; s/cute\.core\.TiledMma/cute.TiledMma/g; s/cute\.make_fragment(/cute.make_rmem_tensor(/g' \
  vllm/third_party/tml_fa4/*.py vllm/vllm_flash_attn/cute/*.py
# tml-fa4 keys old-vs-new nvvm API off CUDA 12.9, but the binding signature
# tracks nvidia-cutlass-dsl (pinned 4.6.0 = new API). Wrong branch on cu129
# torch -> fmax()/atomicrmw TypeError. Evidence: H100 session 2.
python - <<'PYEOF'
import glob
old = "if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:"
new = "if False:  # nvvm API tracks nvidia-cutlass-dsl (pinned 4.6.0 = new API)"
n = 0
for p in glob.glob("vllm/third_party/tml_fa4/*.py"):
    s = open(p).read()
    if old in s:
        n += s.count(old)
        open(p, "w").write(s.replace(old, new))
print(f"nvvm-branch sites patched: {n}")
PYEOF
if [ -d ~/tml_fa4_modified ]; then
  cp ~/tml_fa4_modified/*.py vllm/third_party/tml_fa4/
  echo "inkling-turbo modified kernels deployed"
fi
python -c "import vllm.third_party.tml_fa4; print('tml_fa4 import OK')"

echo "=== parity: FA4 rel attention (sheared path expected on sm_100) ==="
python ~/parity_fa4_rel.py || true

echo "=== microbench: day-0 attention + gate at real shapes ==="
python ~/microbench_attn_day0.py || true

echo "=== microbench: honest score_mod baseline (sm_90 production path) ==="
python ~/microbench_attn_scoremod.py || true

echo "=== BOOTSTRAP COMPLETE ==="
