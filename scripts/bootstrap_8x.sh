#!/usr/bin/env bash
# Bootstrap a fresh 8x H100 Lambda box for the Inkling validation gates
# (runs ON the instance). Mirrors scripts/bootstrap_b200.sh: pinned fork,
# cutlass drift fixes, deploy-to-resolved-path, plus a backup dir so the
# gate scripts can flip stock <-> ours, and the model download.
#
# After this completes: run scripts/gate_logit_parity.py then
# scripts/gate_e2e_bench.sh on the box.
set -euo pipefail
exec 2>&1

PIN=850295881041754184717804104fcaadd2b2129e

# --- session cost (Lambda 8x H100 SXM5) ------------------------------------
RATE_PER_HOUR=31.92
echo "=== COST: 8x H100 = \$${RATE_PER_HOUR}/hr on Lambda ==="
echo "estimated bootstrap wall clock: 0.5-1.5h (install ~15min + 592GB"
echo "NVFP4 model download) => ~\$16-\$48 for bootstrap alone."
echo "Full gate session (parity + e2e bench) is multi-hour; watch the"
echo "Set a spend cap before running, and kill the box when idle."

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== disk (NVFP4 checkpoint is ~592GB, journal/phase0.md) ==="
df -h "$HOME" | tail -1

echo "=== toolchain ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
# Triton JIT needs Python headers when the venv uses the system interpreter
sudo apt-get install -y python3.12-dev python3-dev >/dev/null 2>&1 || true

cd ~
if [ ! -d vllm ]; then
  # origin URL as used in scripts/bootstrap_b200.sh (fork base lives on
  # vllm-project/vllm; the pin is the day-0 Inkling series merge base)
  git clone --filter=blob:none https://github.com/vllm-project/vllm.git vllm
fi
cd vllm
git checkout -q "$PIN"
echo "vllm @ $(git rev-parse --short HEAD)"

uv venv --python 3.12 >/dev/null 2>&1 || true
source .venv/bin/activate
echo "=== installing (precompiled) ==="
# VLLM_USE_PRECOMPILED verified in the pinned tree: setup.py lines 50-53;
# install command per the pinned tree's AGENTS.md "Installing dependencies"
# ENVIRONMENT TIME-CAPSULE (2026-07-23, drift #5) - mirrors bootstrap_b200.sh:
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/850295881041754184717804104fcaadd2b2129e/vllm-0.23.1rc1.dev1237%2Bg850295881-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=cu130 2>&1 | tail -2
uv pip install "torch==2.11.0" torchvision --torch-backend=cu130 2>&1 | tail -1
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

echo "=== stock kernel backup (BEFORE any deploy; gates restore from it) ==="
# Deploy to the RESOLVED package path (precompiled install may import
# tml_fa4 from site-packages, not the source tree).
TML_PKG=$(python -c "import vllm.third_party.tml_fa4 as m, os; print(os.path.dirname(m.__file__))")
echo "resolved tml_fa4 package dir: $TML_PKG"
BACKUP_DIR="$HOME/tml_fa4_backup"
if [ ! -d "$BACKUP_DIR" ]; then
  # taken AFTER the drift fixes: "stock" = drift-fixed day-0 kernels, the
  # same baseline both builds need to even compile on this toolchain
  mkdir -p "$BACKUP_DIR"
  cp "$TML_PKG"/*.py "$BACKUP_DIR/"
  echo "stock backup created: $BACKUP_DIR ($(ls "$BACKUP_DIR" | wc -l) files)"
else
  echo "stock backup already exists: $BACKUP_DIR (kept; delete to refresh)"
fi

if [ -d ~/tml_fa4_modified ]; then
  echo "=== deploy modified kernels (import smoke test), then restore stock ==="
  cp ~/tml_fa4_modified/*.py "$TML_PKG/"
  echo "inkling-turbo modified kernels deployed to $TML_PKG"
  python -c "import importlib, vllm.third_party.tml_fa4 as m; importlib.reload(m); print('tml_fa4 import OK (modified)')"
  # leave the box in STOCK state; the gate scripts flip builds themselves
  cp "$BACKUP_DIR"/*.py "$TML_PKG/"
  echo "restored stock kernels; box is in STOCK state"
else
  echo "NOTE: ~/tml_fa4_modified not present; upload it before running gates"
fi
python -c "import vllm.third_party.tml_fa4; print('tml_fa4 import OK')"

echo "=== model download -> ~/models/inkling ==="
# CLI name `hf download` verified in the pinned tree docs:
# docs/models/supported_models.md line 230 (`hf download <repo>`); the `hf`
# CLI ships with huggingface_hub, a vllm dependency.
# HF_HUB_ENABLE_HF_TRANSFER is the standard huggingface_hub accelerated
# download switch and requires the hf_transfer package.
# TODO(owner): confirm exact repo name before spend. journal/phase0.md
# records the NVFP4 checkpoint at thinkingmachines/Inkling-NVFP4 (VERIFIED
# via HF hub API 2026-07-21: repo exists, 45 files; base repo Inkling is
# the 1.9TB BF16 original, wrong for serving gates);
# on Hopper (no FP4 MMA) it serves via the W4A16 dequant path.
MODEL_REPO="${MODEL_REPO:-thinkingmachines/Inkling-NVFP4}"
# On AWS DLAMI boxes the instance-store NVMe RAID is at /opt/dlami/nvme;
# override MODEL_DIR (and HF cache) to land the 592GB download there.
MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling}"
export HF_HOME="${HF_HOME:-$(dirname "$MODEL_DIR")/hf_cache}"
mkdir -p "$MODEL_DIR"
uv pip install hf_transfer >/dev/null 2>&1 || true
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f "$MODEL_DIR/.download_complete" ]; then
  echo "model already downloaded (marker present): $MODEL_DIR"
else
  # hf download resumes partial downloads, so re-running is safe
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  touch "$MODEL_DIR/.download_complete"
  echo "model downloaded: $MODEL_DIR"
fi
du -sh "$MODEL_DIR" || true

echo "=== BOOTSTRAP COMPLETE ==="
echo "next: python ~/inkling-scripts/gate_logit_parity.py  (or scripts/ path)"
echo "then: bash ~/inkling-scripts/gate_e2e_bench.sh"
