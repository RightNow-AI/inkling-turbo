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
# ENVIRONMENT TIME-CAPSULE (2026-07-23, drift #5): upstream regenerated the
# wheel bucket, deleting cu12x wheels for the pinned sha and defaulting to
# CUDA-13 builds; --torch-backend=auto then picks cu12x torch and the import
# dies on libcudart.so.13. Pin EVERYTHING that proved green on H100:
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/850295881041754184717804104fcaadd2b2129e/vllm-0.23.1rc1.dev1237%2Bg850295881-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cu130 2>&1 | tail -2
uv pip install "torch==2.11.0" torchvision --torch-backend=cu130 2>&1 | tail -1
# CUDA-13 userspace on Lambda driver 570: NVIDIA forward-compat shim
if [ ! -d /usr/local/cuda-13.0/compat ]; then
  curl -sO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.173.02-1ubuntu1_amd64.deb
  sudo dpkg -i cuda-compat-13-0_580.173.02-1ubuntu1_amd64.deb >/dev/null
fi
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

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
  # Deploy to the RESOLVED package path (precompiled install may import
  # tml_fa4 from site-packages, not the source tree).
  TML_PKG=$(python -c "import vllm.third_party.tml_fa4 as m, os; print(os.path.dirname(m.__file__))")
  echo "resolved tml_fa4 package dir: $TML_PKG"
  cp ~/tml_fa4_modified/*.py "$TML_PKG/"
  echo "inkling-turbo modified kernels deployed to $TML_PKG"
fi
python -c "import vllm.third_party.tml_fa4.flash_fwd_sm90 as m; src=open(m.__file__).read(); print('DEPLOY_CHECK file:', m.__file__); print('DEPLOY_CHECK tiled-copy bias present:', 'bias_thr_copy_C' in src)"
python -c "import vllm.third_party.tml_fa4; print('tml_fa4 import OK')"

echo "=== parity: FA4 rel attention (sheared path expected on sm_100) ==="
python ~/parity_fa4_rel.py || true

echo "=== U2 sm_90 A/B: generic reference routing (proven, slow) ==="
U2_SM90_GENERIC=1 python ~/parity_fa4_rel.py || true
echo "=== U2 probe ladder (native path; coordinate probes removed) ==="
U2_DEBUG_SENTINEL=1 python ~/parity_fa4_rel.py --debug || true
U2_DEBUG_ZEROBIAS=1 python ~/parity_fa4_rel.py --debug || true

echo "=== U3: FP8 paged-KV patch + parity ==="
VLLM_ROOT=$(python -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
if [ -f ~/u3_fp8_kv.py ]; then
  python ~/u3_fp8_kv.py "$VLLM_ROOT" || true
  python ~/parity_kv_fp8.py || true
fi

echo "=== microbench: day-0 attention + gate at real shapes ==="
python ~/microbench_attn_day0.py || true

echo "=== microbench: honest score_mod baseline (sm_90 production path) ==="
python ~/microbench_attn_scoremod.py || true

echo "=== BOOTSTRAP COMPLETE ==="
