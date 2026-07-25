#!/usr/bin/env python3
"""Modal deployment: end-to-end Inkling serving benchmark on 8x H100.

This is the last-resort path for the one missing result in the project.
Lambda and AWS have had zero 8-GPU capacity for days. The previous attempt
died when a watchdog killed the box mid-matrix and every result was lost,
because results were only collected at the end. This script fixes that:
every single run JSON is copied to a Modal Volume and committed the
instant that run finishes. A terminated container costs at most one
configuration.

COST DESIGN (hard cap $200 total)
  The model download and the benchmark are SEPARATE functions sharing one
  Volume. Downloading 552GB on the 8x H100 container would burn about $34
  per hour of pure waste, so the download runs CPU-only at about $0.63/hr
  and the GPU container mounts the already-populated Volume READ ONLY and
  never downloads anything.
  A spend ledger on the results Volume accumulates the estimated spend of
  every container across invocations, so a resume cannot silently blow
  through the cap.

WHAT IT DOES
  step=download  CPU-only container, pulls thinkingmachines/Inkling-NVFP4
                 (~552GB) into a Modal Volume. Idempotent: a rerun with the
                 completion marker present is a no-op.
  step=bench     8x H100 container. Deploy kernels, serve, benchmark,
                 persist. No downloading.
  step=all       download then bench.
  step=validate  ONE H100. Parity gates + microbenches only. NEEDS NO MODEL:
                 every harness builds random tensors and calls the kernels
                 directly, so this answers the open kernel questions at
                 ~$4.6/hr instead of ~$34.4/hr and without waiting for the
                 552GB download. Mounts only the results Volume.

LAUNCH
  modal run --detach scripts/modal_e2e_bench.py --step all
  # trim the matrix if close to the cap (env or flags, flags win):
  RUNS=3 CONCURRENCIES="1 8 32" modal run --detach scripts/modal_e2e_bench.py --step bench
  # resume an interrupted matrix (existing run JSONs are skipped):
  modal run --detach scripts/modal_e2e_bench.py --step bench
  # cheap single-GPU kernel validation (no model needed):
  modal run --detach scripts/modal_e2e_bench.py --step validate
  # pull results down:
  modal volume get inkling-bench-results /bench ./bench_results
  modal volume get inkling-bench-results /validate ./validate_results

MODAL API PROVENANCE (verified 2026-07-24 against modal.com/docs, and
against the installed modal 1.5.2 by constructing the object graph)
  gpu="H100:8"                     modal.com/docs/guide/gpu  ("append :n";
                                   H100 supports up to 8 per container)
  modal.Volume.from_name(name, create_if_missing=True)
                                   modal.com/docs/guide/volumes and
                                   modal.com/docs/reference/modal.Volume
  vol.commit()                     modal.com/docs/reference/modal.Volume
  vol.with_mount_options(read_only=True)
                                   modal.Volume.with_mount_options docstring
                                   (modal 1.5.2)
  volumes={"/path": vol}           modal.com/docs/guide/volumes
  Image.from_registry(tag, add_python=...)
  Image.apt_install / pip_install / run_commands(..., env=) / env
  Image.add_local_dir(local, remote, *, copy=False, ignore=[])
  Image.add_local_file(local, remote, *, copy=False)
                                   modal.com/docs/reference/modal.Image
  timeout= on @app.function, 1s to 24h, default 300s
                                   modal.com/docs/guide/timeouts
  cpu= (physical cores), memory= (MiB)
                                   modal.com/docs/guide/resources
  Host driver 580.95.05 / CUDA driver API 13.0, so a CUDA 13.0 runtime
  image needs no forward-compat shim
                                   modal.com/docs/guide/cuda
  @app.local_entrypoint() + `modal run --detach`
                                   modal.com/docs/guide/apps
  Pricing: H100 $0.001097/sec, CPU $0.0000131/core/sec, memory
  $0.00000222/GiB/sec               modal.com/pricing, read 2026-07-24

ENVIRONMENT RECIPE
  Mirrors scripts/bootstrap_8x.sh and scripts/bootstrap_b200.sh exactly.
  Pinned fork, pinned precompiled wheel, torch 2.11.0 cu130, scipy, and
  the CuTe DSL drift fixes. Every value there is the outcome of a measured
  failure. Nothing here alters them.

BENCH FLAG PROVENANCE
  Every flag below is verified against the pinned tree at vllm/, same
  file:line evidence as scripts/gate_e2e_bench.sh. See BENCH_FLAG_EVIDENCE.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Pins. Do not change without new measurements.
# --------------------------------------------------------------------------

APP_NAME = "inkling-e2e-bench"

VLLM_PIN = "850295881041754184717804104fcaadd2b2129e"
VLLM_WHEEL = (
    "https://wheels.vllm.ai/850295881041754184717804104fcaadd2b2129e/"
    "vllm-0.23.1rc1.dev1237%2Bg850295881-cp38-abi3-manylinux_2_28_x86_64.whl"
)
# CUDA 13.0 runtime. Modal hosts run driver 580.95.05 with CUDA driver API
# 13.0, so the cuda-compat shim that bootstrap_8x.sh needs on Lambda's
# driver 570 is not needed here and is deliberately NOT on the library path.
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
PYTHON_VERSION = "3.12"

MODEL_REPO = "thinkingmachines/Inkling-NVFP4"

# Container paths.
VLLM_SRC = "/opt/vllm"
INK_DIR = "/opt/inkling"
OURS_TML_DIR = f"{INK_DIR}/tml_fa4_modified"
ROUTE_PATCH = f"{INK_DIR}/u2_serving_route.py"
STOCK_TML_DIR = f"{INK_DIR}/tml_fa4_stock"
STOCK_ROUTE = f"{INK_DIR}/fa4_rel_attention.stock.py"
ROUTE_REL = "vllm/models/inkling/nvidia/ops/fa4_rel_attention.py"
# The u3 and u2 patches edit files in the vLLM tree, not just the tml_fa4
# package. Restoring only tml_fa4 leaves them patched, and u2 rewrites the
# exact text u3 anchors on, so a reused warm container fails setup with
# "anchor not found in qkvr_prep.py".
#
# The span has to be the whole nvidia/ subtree, not nvidia/ops/: u2 targets
# ops/qkvr_prep.py and ops/fa4_rel_attention.py but ALSO nvidia/attention.py,
# one level up. Backing up ops/ alone leaves attention.py patched forever.
INK_NVIDIA_REL = "vllm/models/inkling/nvidia"
STOCK_OPS_DIR = f"{INK_DIR}/inkling_nvidia_stock"

MODEL_MOUNT = "/models"
MODEL_DIR = f"{MODEL_MOUNT}/inkling"
MODEL_MARKER = f"{MODEL_DIR}/.download_complete"
RESULTS_MOUNT = "/results"
# PIN THE GPU KIND, and assert at runtime that we got what we asked for.
#
# Requesting "H100:8" is not a guarantee. A run on 2026-07-25 asked for H100:8
# and received 8x NVIDIA H200 with 143771 MiB each, and the difference is not
# cosmetic: the KV pool went from 4379 tokens to 188160 and vLLM's own reported
# max concurrency from 1.43x to 61.25x. A stock-versus-ours comparison split
# across those two would be comparing memory budgets, not kernels.
#
# H200 is the pinned choice for end-to-end serving. Same architecture, sm_90, so
# the kernel under test is identical to the one the microbenchmarks measured on
# H100. What changes is headroom: 8x143GB against a 592GB checkpoint leaves room
# for a real batch sweep, where 8x80GB leaves room for 1.43 concurrent
# maximum-length requests and no sweep at all. Serving a 975B model at a
# concurrency of one is not a deployment anyone would run.
BENCH_GPU_KIND = "H200"

# Namespaced by GPU kind. Resume skips runs whose JSON already exists, so a
# shared path plus a hardware substitution silently produces a mixed-hardware
# comparison. The path is the guard: results from different silicon cannot land
# in the same tree. bench/ (unsuffixed) holds the 2026-07-25 partial H100 run.
RESULTS_ROOT = f"{RESULTS_MOUNT}/bench_{BENCH_GPU_KIND.lower()}"
LOGS_ROOT = f"{RESULTS_MOUNT}/logs_{BENCH_GPU_KIND.lower()}"
LEDGER_PATH = f"{RESULTS_MOUNT}/spend_ledger.json"

# Validate step (single GPU, no model). Image-mounted sources.
HARNESS_SRC_DIR = f"{INK_DIR}/harness"
RELPROJ_SRC = f"{INK_DIR}/relproj_score_mod.py"
U3_PATCH = f"{INK_DIR}/u3_fp8_kv.py"
SHEAR_PATCH = f"{INK_DIR}/u2_shear_fusion.py"
# Writable staging area. The harnesses write their JSON next to themselves
# (Path(__file__).with_suffix(".json")), so they cannot run from an image
# mount; they are copied here first.
WORK_ROOT = "/tmp/inkling_validate"
# Rebound per run by _set_validate_root, so two patch sets cannot overwrite
# each other's manifest.json and summary.json. Those two filenames are not
# tagged, unlike the microbench artifacts, so without this the second run
# silently replaces the first run's verdict.
VALIDATE_ROOT = f"{RESULTS_MOUNT}/validate"
VALIDATE_LOGS = f"{VALIDATE_ROOT}/logs"


def _set_validate_root(tag: str) -> None:
    """Point every validate write at a per-tag subdirectory."""
    global VALIDATE_ROOT, VALIDATE_LOGS
    VALIDATE_ROOT = f"{RESULTS_MOUNT}/validate/{tag}"
    VALIDATE_LOGS = f"{VALIDATE_ROOT}/logs"

# --------------------------------------------------------------------------
# Serving recipe. These exact values are the outcome of seven distinct
# failures on 8x H100. 0.95 dies on the warmup allocation, 0.90 leaves no
# room for KV at all. Do not change them.
# --------------------------------------------------------------------------

N_GPU = 8
TP_SIZE = 8
MAX_MODEL_LEN = 3072
GPU_MEM_UTIL = 0.94
ENFORCE_EAGER = True
SEED = 0
SERVED_NAME = "inkling"
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
ALLOC_CONF = "expandable_segments:True"

# --------------------------------------------------------------------------
# Matrix defaults. All four knobs are env-configurable so the matrix can be
# trimmed near the cap without editing this file. Flags beat env, env beats
# these defaults.
#   RUNS=3  CONCURRENCIES="1 8 32"  MIXES="prefill:2048:128,decode:512:1024"
#   BUILDS="stock,ours"
# --------------------------------------------------------------------------

BUILDS = ("stock", "ours")
DEFAULT_RUNS = 5
DEFAULT_CONCURRENCIES = "1 8 32"
# name:input_len:output_len. Both fit under max_model_len 3072.
DEFAULT_MIXES = "prefill:2048:128,decode:512:1024"
DEFAULT_BUILDS = "stock,ours"
MIN_PROMPTS = 16
PROMPTS_PER_SLOT = 4  # num_prompts = max(conc*4, 16), same as gate_e2e_bench.sh

# --------------------------------------------------------------------------
# Cost. Rates read from modal.com/pricing on 2026-07-24. Verify before
# spend, prices move. Every dollar figure the script prints is derived from
# these constants, none of it is hardcoded output.
# --------------------------------------------------------------------------

USD_PER_H100_SEC = 0.001097  # $3.9492 per GPU-hour
# H200 is billed above H100. Rate read from modal.com/pricing on 2026-07-25;
# VERIFY before spend, prices move. Erring high is the safe direction here,
# because the ledger uses this to decide whether a container may start.
USD_PER_H200_SEC = 0.001261  # $4.5396 per GPU-hour
USD_PER_A100_40_SEC = 0.000583  # $2.0988 per GPU-hour
USD_PER_A100_80_SEC = 0.000772  # $2.7792 per GPU-hour
GPU_SEC_RATE = {
    "H100": USD_PER_H100_SEC,
    "H200": USD_PER_H200_SEC,
    "A100-40GB": USD_PER_A100_40_SEC,
    "A100-80GB": USD_PER_A100_80_SEC,
}
USD_PER_CPU_CORE_SEC = 0.0000131
USD_PER_GIB_MEM_SEC = 0.00000222

BENCH_CPU = 16.0
BENCH_MEMORY_MIB = 262144  # 256 GiB, soft request so there is no OOM kill
DL_CPU = 8.0
DL_MEMORY_MIB = 32768

# HARD CAP: $200 total on Modal.
BUDGET_USD = 200.0
# Hard function timeout, sized so a single container cannot exceed the cap
# even if everything hangs. The bench container bills
#   8 x $3.9492 + 16 x $0.04716 + 256 x $0.007992 = ~$34.4/hr,
# so the cap alone allows ~5.8h. 4h is chosen for margin: it leaves room
# for the download container, for a resume invocation, and for the rate
# being higher than estimated if Modal applies a per-GPU resource floor.
#
# 3.5h x ~$34.4/hr = ~$120 worst case for one bench container, and the
# effective deadline is 3.33h after SHUTDOWN_MARGIN_S, so ~$115. Was 4.0, which
# authorised $138 for a container whose expected work is about $70. The real
# protection is not this number, it is the per-run timeout and the per-build
# time allowance, both of which stop a single hang from eating the container.
# 1.75h at the pinned H200 rate bounds one container to about $62, against
# $77.67 remaining in the ledger after the stock run. Was 2.75h ($101), sized
# when $145 remained.
#
# The stock run cost $59.22 for a 28 minute model load plus 10 runs, and it did
# NOT get to the ours build: the per-build allowance yielded correctly, then the
# pre-build guard refused ours because it wanted run_reserve_s of 51 minutes plus
# server_wait_s of 45 on top, and only 78 remained. The 51 came from
# decode/conc1, where one run took about 32 minutes, because 16 prompts at
# concurrency 1 is 16384 strictly sequential decode steps.
#
# So the matched A/B is bought by dropping conc1 rather than by buying more time:
# BUILDS=ours CONCURRENCIES=8 is 6 runs against the 6 stock runs already on the
# Volume at prefill/8 and decode/8.
# 1.1h bounds one container to about $36 at the H200 rate, against $55 remaining
# after the A100 validate run. Expected work is 28 minutes of model load plus five
# 3.4 minute runs, about $29. The matrix deliberately still lists decode/conc1,
# whose runs cost about 32 minutes each; the deadline and the forward-looking
# reserve stop it from starting rather than a hand-maintained exclusion list.
BENCH_TIMEOUT_HOURS = 1.1
BENCH_TIMEOUT_S = int(BENCH_TIMEOUT_HOURS * 3600)
DOWNLOAD_TIMEOUT_S = 6 * 3600  # CPU only at ~$0.63/hr, so at most ~$4
# Leave room to stop the server, copy results and commit before Modal kills
# the container.
SHUTDOWN_MARGIN_S = 600
# Below this much remaining budget a bench container cannot even load the
# model and finish one run, so it refuses to start rather than burn it.
MIN_USEFUL_USD = 30.0

SERVER_WAIT_S = 5400  # model load from a Volume across 8 ranks is slow
SERVER_STOP_WAIT_S = 240
GPU_RELEASE_SLEEP_S = 30

# --------------------------------------------------------------------------
# Validate step cost. One H100 plus modest CPU/memory. Same ledger, same cap.
#   1 x $3.9492 + 8 x $0.04716 + 32 x $0.007992 = ~$4.58/hr
# The harnesses allocate at most ~10 GB of DEVICE memory (32 sequences x 64K
# KV in bf16) and almost no host memory, so 32 GiB is generous; memory is the
# second-largest term in the rate, which is why it is not 256 GiB here.
# --------------------------------------------------------------------------

# The validate step runs parity gates and microbenchmarks, which build random
# tensors and need no checkpoint, so it can target whichever architecture needs
# evidence. sm_80 is the one with no correctness result at all for the code in
# the tree, so it is worth a couple of dollars.
#
# A100-40GB rate read from modal.com/pricing on 2026-07-25; VERIFY before spend.
VALIDATE_GPU_KIND = os.environ.get("VALIDATE_GPU", "H100").strip()
VALIDATE_N_GPU = 1
VALIDATE_CPU = 8.0
VALIDATE_MEMORY_MIB = 32768
# CuTe DSL JIT compiles dominate the wall clock, not the kernels. 2h is a
# ceiling, not an expectation; every artifact is committed the instant it
# exists, so a timeout costs at most one harness.
VALIDATE_TIMEOUT_HOURS = 2.0
VALIDATE_TIMEOUT_S = int(VALIDATE_TIMEOUT_HOURS * 3600)
VALIDATE_SHUTDOWN_MARGIN_S = 180
# Below this a container cannot finish the parity gates plus one microbench.
MIN_USEFUL_VALIDATE_USD = 10.0
# Goes into every artifact filename so these can never be confused with the
# session-25 H100 numbers or with an 8x run.
VALIDATE_TAG = "modal_h100x1"

BENCH_FLAG_EVIDENCE = """\
Verified against the pinned fork @850295881 (repo path vllm/):
  benchmarks/benchmark_serving.py is a deprecated shim that exits 1; the
  real entrypoint is `vllm bench serve`
  (vllm/entrypoints/cli/benchmark/serve.py, class name = "serve").
  vllm bench serve, vllm/benchmarks/serve.py:
    --backend            1488
    --base-url           1495
    --max-concurrency    1519
    --model              1533
    --save-result        1625
    --result-dir         1649
    --result-filename    1656
    --ignore-eos         1665
    --percentile-metrics 1681
    --metric-percentiles 1691
    --served-model-name  1768
    result path = result_dir/result_filename (compute_result_filename,
    vllm/benchmarks/serve.py:1468-1475)
  dataset flags, vllm/benchmarks/datasets/datasets.py:
    --seed 1607, --num-prompts 1609, --dataset-name 1615,
    --random-input-len 1931, --random-output-len 1937
  vllm serve, vllm/engine/arg_utils.py:
    --enforce-eager 838, --gpu-memory-utilization 1163
  GET /health readiness:
    vllm/entrypoints/serve/instrumentator/health.py:22
"""

# --------------------------------------------------------------------------
# Image
# --------------------------------------------------------------------------

# CuTe DSL drift fixes, identical substitutions to bootstrap_8x.sh:
# ThrMma/TiledMma moved out of cute.core, make_fragment was renamed to
# make_rmem_tensor. tml-fa4 also keys the old-vs-new nvvm API off CUDA 12.9
# while the binding signature actually tracks nvidia-cutlass-dsl 4.6.0, so
# that branch is forced off.
_DRIFT_SED = (
    r"s/cute\.core\.ThrMma/cute.ThrMma/g; "
    r"s/cute\.core\.TiledMma/cute.TiledMma/g; "
    r"s/cute\.make_fragment(/cute.make_rmem_tensor(/g"
)
_NVVM_SED = (
    r"s/if CUDA_VERSION\.major == 12 and CUDA_VERSION\.minor == 9:"
    r"/if False:  # nvvm API tracks nvidia-cutlass-dsl (pinned 4.6.0 = new API)/g"
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_OURS_TML = _REPO_ROOT / "kernels" / "tml_fa4_modified"
_LOCAL_ROUTE_PATCH = _REPO_ROOT / "kernels" / "patches" / "u2_serving_route.py"
_LOCAL_HARNESS = _REPO_ROOT / "harness"
_LOCAL_RELPROJ = _REPO_ROOT / "kernels" / "relproj_score_mod.py"
_LOCAL_U3_PATCH = _REPO_ROOT / "kernels" / "patches" / "u3_fp8_kv.py"
_LOCAL_SHEAR_PATCH = _REPO_ROOT / "kernels" / "patches" / "u2_shear_fusion.py"

bench_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "procps",
        # Triton JIT needs Python headers when it compiles against the
        # system interpreter. Same reason as bootstrap_8x.sh.
        "python3-dev",
        "python3.12-dev",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
    .run_commands(
        # Pinned fork. blob:none keeps the clone small; the checkout still
        # resolves because all refs are fetched.
        f"git clone --filter=blob:none https://github.com/vllm-project/vllm.git {VLLM_SRC}",
        f"git -C {VLLM_SRC} checkout -q {VLLM_PIN}",
        f"git config --global --add safe.directory {VLLM_SRC}",
        f"git -C {VLLM_SRC} rev-parse HEAD",
    )
    .run_commands(
        # Precompiled install against the exact wheel that proved green.
        # Upstream regenerated the wheel bucket and --torch-backend=auto now
        # picks cu12x torch, which dies on libcudart.so.13. Everything is
        # pinned.
        # --system --break-system-packages: the target interpreter is the
        # image python that Modal itself uses, not a venv. Installing there
        # keeps the modal client and vllm in one interpreter.
        f"cd {VLLM_SRC} && uv pip install --system --break-system-packages "
        '--python "$(command -v python3)" -e . --torch-backend=cu130',
        env={
            "VLLM_USE_PRECOMPILED": "1",
            "VLLM_PRECOMPILED_WHEEL_LOCATION": VLLM_WHEEL,
        },
    )
    .run_commands(
        "uv pip install --system --break-system-packages "
        '--python "$(command -v python3)" '
        '"torch==2.11.0" torchvision --torch-backend=cu130',
        # scipy is imported by the TP workers. Its absence killed a whole
        # run once. hf_transfer and huggingface_hub drive the download step.
        "uv pip install --system --break-system-packages "
        '--python "$(command -v python3)" '
        "scipy hf_transfer huggingface_hub",
    )
    .run_commands(
        # Torch cu13 wheels ship their CUDA libs under site-packages/nvidia.
        # Register them with ldconfig instead of relying on LD_LIBRARY_PATH
        # inheritance across the TP worker fork.
        'SP=$(python3 -c "import site; print(site.getsitepackages()[0])"); '
        'for d in "$SP"/nvidia/*/lib "$SP"/nvidia/*/lib64; do '
        '[ -d "$d" ] && echo "$d"; done | sort -u '
        "> /etc/ld.so.conf.d/zz-nvidia-torch.conf; "
        "cat /etc/ld.so.conf.d/zz-nvidia-torch.conf; ldconfig",
        "python3 -c \"import torch, scipy; print('torch', torch.__version__, "
        "'| scipy', scipy.__version__)\"",
    )
    .run_commands(
        # Drift fixes run AFTER the install: the precompiled wheel is what
        # populates vllm/third_party/tml_fa4 and vllm/vllm_flash_attn/cute
        # in the source tree. A non-matching glob makes sed exit non-zero
        # and fails the build, which is exactly what we want. Failing at
        # build time is free, failing at $34/hr is not.
        f"ls -1 {VLLM_SRC}/vllm/third_party/tml_fa4/*.py "
        f"{VLLM_SRC}/vllm/vllm_flash_attn/cute/*.py",
        f"cd {VLLM_SRC} && sed -i '{_DRIFT_SED}' "
        "vllm/third_party/tml_fa4/*.py vllm/vllm_flash_attn/cute/*.py",
        f"cd {VLLM_SRC} && sed -i '{_NVVM_SED}' vllm/third_party/tml_fa4/*.py",
        # Negative assertions: the pre-drift spellings must be gone. If they
        # were never present these still pass, so this cannot fail falsely.
        f"cd {VLLM_SRC} && ! grep -rn "
        r"'cute\.core\.ThrMma\|cute\.core\.TiledMma\|cute\.make_fragment(' "
        "vllm/third_party/tml_fa4 vllm/vllm_flash_attn/cute",
        f"cd {VLLM_SRC} && ! grep -rn "
        "'CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9' "
        "vllm/third_party/tml_fa4",
        # Informational: how many nvvm branch sites were forced off.
        f"cd {VLLM_SRC} && grep -rc 'nvvm API tracks nvidia-cutlass-dsl' "
        "vllm/third_party/tml_fa4 || true",
        # The route patch anchor must exist, or the ours build is a no-op.
        f"grep -q 'capability.major in (10, 11)' {VLLM_SRC}/{ROUTE_REL}",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # hf_transfer is deprecated in current huggingface_hub; Xet is the
            # fast path now. Without it the 552GB pull runs about 4x slower.
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": f"{MODEL_MOUNT}/hf_home",
            "PYTORCH_CUDA_ALLOC_CONF": ALLOC_CONF,
            "VLLM_USE_PRECOMPILED": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
    # copy=False so editing our kernels does not rebuild the whole image.
    .add_local_dir(
        _LOCAL_OURS_TML.as_posix(),
        OURS_TML_DIR,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_file(_LOCAL_ROUTE_PATCH.as_posix(), ROUTE_PATCH)
    # Validate-step inputs. All copy=False mounts, so they add no build time
    # and cannot invalidate the expensive layers above. The bench step never
    # looks at them.
    # *.json is excluded deliberately: the repo carries stale local
    # microbench_attn_*.json artifacts, and a crashed run must never be able
    # to publish one of those as if it were a fresh measurement.
    .add_local_dir(
        _LOCAL_HARNESS.as_posix(),
        HARNESS_SRC_DIR,
        ignore=["__pycache__", "*.pyc", "*.json"],
    )
    # harness/parity_fa4_rel.py and harness/microbench_attn_scoremod.py import
    # kernels.relproj_score_mod from the repo root.
    .add_local_file(_LOCAL_RELPROJ.as_posix(), RELPROJ_SRC)
    .add_local_file(_LOCAL_U3_PATCH.as_posix(), U3_PATCH)
    .add_local_file(_LOCAL_SHEAR_PATCH.as_posix(), SHEAR_PATCH)
)

app = modal.App(APP_NAME)

# Two Volumes. The model Volume is write-once read-many: the CPU-only
# download function mounts it writable, the GPU function mounts it READ
# ONLY so a bug can never re-download 552GB at 8x H100 rates. The results
# Volume is the whole point, it is committed after every individual run.
model_vol = modal.Volume.from_name("inkling-model", create_if_missing=True)
results_vol = modal.Volume.from_name("inkling-bench-results", create_if_missing=True)


# --------------------------------------------------------------------------
# Cost helpers
# --------------------------------------------------------------------------


def _usd_per_hour(
    n_gpu: float, cpu: float, mem_mib: float, kind: str = "H100"
) -> float:
    per_sec = (
        n_gpu * GPU_SEC_RATE[kind]
        + cpu * USD_PER_CPU_CORE_SEC
        + (mem_mib / 1024.0) * USD_PER_GIB_MEM_SEC
    )
    return per_sec * 3600.0


def _bench_usd_per_hour() -> float:
    return _usd_per_hour(N_GPU, BENCH_CPU, BENCH_MEMORY_MIB, BENCH_GPU_KIND)


def _download_usd_per_hour() -> float:
    return _usd_per_hour(0.0, DL_CPU, DL_MEMORY_MIB)


def _validate_usd_per_hour() -> float:
    return _usd_per_hour(VALIDATE_N_GPU, VALIDATE_CPU, VALIDATE_MEMORY_MIB,
                        VALIDATE_GPU_KIND)


def _print_validate_cost_banner(where: str) -> None:
    rate = _validate_usd_per_hour()
    print(f"=== COST ESTIMATE, validate ({where}) ===")
    print(
        f"{VALIDATE_GPU_KIND} ${GPU_SEC_RATE[VALIDATE_GPU_KIND] * 3600:.4f}/GPU-hr x {VALIDATE_N_GPU} GPU "
        f"+ {VALIDATE_CPU:g} cores + {VALIDATE_MEMORY_MIB // 1024} GiB "
        f"= ${rate:.2f}/hr"
    )
    print(
        f"hard timeout {VALIDATE_TIMEOUT_HOURS:g}h => worst case "
        f"${rate * VALIDATE_TIMEOUT_HOURS:.2f} for one container"
    )
    print(
        "expected 30-60 min, dominated by CuTe DSL JIT compiles rather than "
        f"by the kernels: about ${rate * 0.5:.2f} to ${rate * 1.0:.2f}. "
        "This is an ESTIMATE, not a measurement."
    )
    print(f"HARD CAP ${BUDGET_USD:.0f} total, enforced by the same spend ledger")
    print(
        "rates read from modal.com/pricing on 2026-07-24. VERIFY before "
        "spend, prices move. Modal may also apply a per-GPU resource floor "
        "above the requested cpu/memory, which would raise the real rate."
    )
    print(
        "no model Volume is mounted: every harness builds random tensors and "
        "calls the kernels directly, so the 552GB checkpoint is not needed."
    )


def _print_cost_banner(where: str, runs: int, n_configs: int, n_builds: int) -> None:
    gpu_hr = N_GPU * GPU_SEC_RATE[BENCH_GPU_KIND] * 3600.0
    rate = _bench_usd_per_hour()
    total_runs = n_builds * n_configs * runs
    print(f"=== COST ESTIMATE ({where}) ===")
    print(
        f"{BENCH_GPU_KIND} ${GPU_SEC_RATE[BENCH_GPU_KIND] * 3600:.4f}/GPU-hr x {N_GPU} GPUs "
        f"= ${gpu_hr:.2f}/hr"
    )
    print(
        f"+ {BENCH_CPU:g} cores + {BENCH_MEMORY_MIB // 1024} GiB "
        f"= ${rate:.2f}/hr for the bench container"
    )
    print(f"download container, CPU only: ${_download_usd_per_hour():.2f}/hr")
    print(
        f"hard timeout {BENCH_TIMEOUT_HOURS:g}h => projected worst case for "
        f"one bench container = {BENCH_TIMEOUT_HOURS:g} GPU-container-hours "
        f"x ${rate:.2f} = ${rate * BENCH_TIMEOUT_HOURS:.0f}"
    )
    print(
        f"matrix: {n_builds} builds x {n_configs} configs x {runs} runs "
        f"= {total_runs} runs"
    )
    print(f"HARD CAP ${BUDGET_USD:.0f} total, enforced by the spend ledger")
    print(
        "rates read from modal.com/pricing on 2026-07-24. VERIFY before "
        "spend, prices move. Modal may also apply a per-GPU resource floor "
        "above the requested cpu/memory, which would raise the real rate."
    )


# --------------------------------------------------------------------------
# Matrix parsing, all env-configurable
# --------------------------------------------------------------------------


def _parse_concurrencies(spec: str) -> list[int]:
    out = []
    for tok in spec.replace(",", " ").split():
        v = int(tok)
        if v < 1:
            raise ValueError(f"concurrency must be >= 1, got {v}")
        out.append(v)
    if not out:
        raise ValueError("no concurrencies given")
    return out


def _parse_mixes(spec: str) -> list[tuple[str, int, int]]:
    """"name:in:out,name:in:out". Rejects anything over max_model_len."""
    out = []
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad mix {tok!r}, want name:input_len:output_len")
        name, ilen, olen = parts[0], int(parts[1]), int(parts[2])
        if ilen + olen > MAX_MODEL_LEN:
            raise ValueError(
                f"mix {name} needs {ilen + olen} tokens but max_model_len "
                f"is {MAX_MODEL_LEN}; the serve recipe is fixed, shrink the mix"
            )
        out.append((name, ilen, olen))
    if not out:
        raise ValueError("no mixes given")
    return out


def _num_prompts(conc: int) -> int:
    # Same rule as scripts/gate_e2e_bench.sh: enough requests to saturate
    # the concurrency level without exploding session cost.
    return max(conc * PROMPTS_PER_SLOT, MIN_PROMPTS)


def _ordered_configs(
    mixes: list[tuple[str, int, int]], concs: list[int]
) -> list[tuple[str, int, int, int, int]]:
    """Cheapest configuration first.

    Cost proxy is the number of sequential decode steps per run,
    (num_prompts / concurrency) * output_len. At concurrency 1 the
    16-prompt floor makes a run 4x longer than the same mix at
    concurrency 8. Running cheap first means a budget or timeout stop
    still leaves matched stock/ours pairs for the configs that completed,
    instead of one build finished and the other empty.
    """
    items = []
    for name, ilen, olen in mixes:
        for c in concs:
            npr = _num_prompts(c)
            steps = (npr / c) * olen
            items.append((steps, name, c, ilen, olen, npr))
    items.sort(key=lambda x: (x[0], x[2], x[1]))
    return [(name, c, ilen, olen, npr) for _, name, c, ilen, olen, npr in items]


# --------------------------------------------------------------------------
# Spend ledger. Persisted on the results Volume so it survives container
# death and accumulates across resume invocations.
# --------------------------------------------------------------------------


def _run_id() -> str:
    return os.environ.get("MODAL_TASK_ID") or f"local-{int(time.time())}"


def _ledger_read() -> dict:
    # reload() first, or a container reads the mount snapshot it booted with and
    # a whole-file write then drops any entry another live container added. The
    # $200 cap is enforced from this file, so a dropped entry is money that
    # stops counting.
    try:
        results_vol.reload()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: ledger volume reload failed ({exc}); reading local view")
    p = Path(LEDGER_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: ledger unreadable ({exc}); treating prior spend as 0")
    return {"budget_usd": BUDGET_USD, "total_usd_est": 0.0, "entries": []}


def _ledger_total(ledger: dict) -> float:
    return round(sum(float(e.get("usd_est") or 0.0) for e in ledger["entries"]), 2)


def _ledger_upsert(fn: str, hours: float, usd: float, note: str) -> float:
    """Record this container's spend so far. Returns the new total."""
    ledger = _ledger_read()
    rid = _run_id()
    entry = {
        "run_id": rid,
        "fn": fn,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hours": round(hours, 4),
        "usd_est": round(usd, 2),
        "note": note,
    }
    for i, e in enumerate(ledger["entries"]):
        if e.get("run_id") == rid and e.get("fn") == fn:
            ledger["entries"][i] = entry
            break
    else:
        ledger["entries"].append(entry)
    ledger["budget_usd"] = BUDGET_USD
    ledger["total_usd_est"] = _ledger_total(ledger)
    _write_committed(json.dumps(ledger, indent=2), Path(LEDGER_PATH), quiet=True)
    return ledger["total_usd_est"]


def _prior_spend(fn: str) -> float:
    """Estimated spend of every OTHER container already on the ledger."""
    ledger = _ledger_read()
    rid = _run_id()
    return round(
        sum(
            float(e.get("usd_est") or 0.0)
            for e in ledger["entries"]
            if not (e.get("run_id") == rid and e.get("fn") == fn)
        ),
        2,
    )


# --------------------------------------------------------------------------
# Container-side helpers
# --------------------------------------------------------------------------


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _py() -> str:
    return shutil.which("python3") or sys.executable


def _vllm_bin() -> str:
    exe = shutil.which("vllm")
    if not exe:
        raise RuntimeError("vllm CLI not on PATH; image build is broken")
    return exe


def _resolve_paths() -> tuple[str, str]:
    """Return (vllm_root, tml_fa4_package_dir).

    Resolved by import, not assumed. A precompiled install can import
    tml_fa4 from site-packages rather than the source tree, and deploying
    to the wrong directory is a silent no-op. Same rule as
    scripts/bootstrap_8x.sh.
    """
    probe = (
        "import os, vllm, vllm.third_party.tml_fa4 as m;"
        "print(os.path.dirname(os.path.dirname(vllm.__file__)));"
        "print(os.path.dirname(m.__file__))"
    )
    out = subprocess.run(
        [_py(), "-c", probe], check=True, capture_output=True, text=True
    ).stdout.split()
    return out[0], out[1]


def _sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _purge_pycache(*dirs: str) -> None:
    for d in dirs:
        for pc in Path(d).rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)


def _make_stock_backup(vllm_root: str, tml_pkg: str) -> None:
    """Pristine backup, taken BEFORE any deploy.

    The benchmark compares both builds, so stock has to be restorable byte
    for byte. "Stock" here means the drift-fixed day-0 kernels, which is
    the same baseline both builds need to compile on this toolchain.
    """
    stock = Path(STOCK_TML_DIR)
    if stock.exists():
        print(f"stock backup already present: {stock}")
    else:
        stock.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(Path(tml_pkg).glob("*.py")):
            shutil.copy2(p, stock / p.name)
            n += 1
        shutil.copy2(Path(vllm_root) / ROUTE_REL, STOCK_ROUTE)
        print(f"stock backup created: {stock} ({n} files) + {STOCK_ROUTE}")
    # Outside the branch above on purpose. These are two independent backups,
    # and a container warm from a build that predates the nvidia backup would
    # otherwise skip it and then fail the restore.
    _make_stock_ops_backup(vllm_root)


def _make_stock_ops_backup(vllm_root: str) -> None:
    """Pristine copy of vllm/models/inkling/nvidia, taken BEFORE any patch runs.

    u3_fp8_kv.py and u2_shear_fusion.py both rewrite files in here. u2 rewrites
    the signature u3 anchors on, so re-patching an already-patched tree aborts.
    Restoring tml_fa4 alone is not enough; these files have to come back too.
    Recursive, because the patch targets straddle two directory levels.
    """
    src = Path(vllm_root) / INK_NVIDIA_REL
    dst = Path(STOCK_OPS_DIR)
    if dst.exists():
        print(f"stock nvidia backup already present: {dst}")
        return
    if not src.is_dir():
        raise RuntimeError(f"inkling nvidia dir not found: {src}")
    # Refuse to snapshot an already-patched tree as "stock". If this is ever
    # reached on a warm container whose backup was lost, the honest outcome is
    # a loud failure, not a backup that quietly bakes our patches into the
    # baseline and makes every later restore a no-op.
    qkvr = src / "ops/qkvr_prep.py"
    if qkvr.exists():
        text = qkvr.read_text(errors="replace")
        dirty = [m for m in ("RelShearSpec", "quantize_kv") if m in text]
        if dirty:
            raise RuntimeError(
                f"refusing to back up a patched tree as stock: {qkvr} already "
                f"contains {dirty}. Restart on a cold container."
            )
    n = 0
    for p in sorted(src.rglob("*.py")):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    print(f"stock nvidia backup created: {dst} ({n} files)")


def _restore_stock_ops(vllm_root: str) -> int:
    """Put vllm/models/inkling/nvidia back exactly as the image shipped it."""
    src = Path(STOCK_OPS_DIR)
    dst = Path(vllm_root) / INK_NVIDIA_REL
    if not src.is_dir():
        raise RuntimeError(f"no stock nvidia backup at {src}")
    n = 0
    for p in sorted(src.rglob("*.py")):
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    print(f"RESTORE_STOCK_OPS {n} files -> {dst}")
    return n


def _deploy_build(build: str, vllm_root: str, tml_pkg: str) -> None:
    """Flip the whole build, kernels AND routing.

    On sm_90 the day-0 router sends Hopper down the score_mod gather, so
    without the route patch the ours build would never reach our kernels
    and the A/B would measure nothing. Both halves flip together.
    """
    if build not in BUILDS:
        raise ValueError(f"unknown build {build}")
    route_dst = Path(vllm_root) / ROUTE_REL

    # Always start from a clean stock state, then overlay.
    for p in sorted(Path(STOCK_TML_DIR).glob("*.py")):
        shutil.copy2(p, Path(tml_pkg) / p.name)
    shutil.copy2(STOCK_ROUTE, route_dst)

    if build == "ours":
        ours = sorted(Path(OURS_TML_DIR).glob("*.py"))
        if not ours:
            raise RuntimeError(f"no kernels in {OURS_TML_DIR}")
        for p in ours:
            shutil.copy2(p, Path(tml_pkg) / p.name)
        _run([_py(), ROUTE_PATCH, vllm_root])

    _purge_pycache(tml_pkg, str(route_dst.parent))
    _verify_build(build, tml_pkg, route_dst)


def _verify_build(build: str, tml_pkg: str, route_dst: Path) -> None:
    """Content-hash check. A deploy that silently did nothing is a defect."""
    expect_dir = Path(STOCK_TML_DIR if build == "stock" else OURS_TML_DIR)
    checked = []
    for p in sorted(expect_dir.glob("*.py")):
        got = Path(tml_pkg) / p.name
        if not got.exists():
            raise RuntimeError(f"deploy check: {got} missing")
        if _sha8(got) != _sha8(p):
            raise RuntimeError(f"deploy check: {got.name} does not match {p}")
        checked.append(f"{p.name}:{_sha8(p)}")
    routed = "Inkling-turbo: sm_90" in route_dst.read_text()
    want_routed = build == "ours"
    if routed != want_routed:
        raise RuntimeError(
            f"deploy check: route patch applied={routed}, expected {want_routed}"
        )
    print(f"DEPLOY_CHECK build={build} pkg={tml_pkg}")
    print(f"DEPLOY_CHECK files={' '.join(checked)}")
    print(f"DEPLOY_CHECK sheared_route_patched={routed}")


def _server_env() -> dict:
    env = os.environ.copy()
    # Measured allocator recipe. Without expandable_segments the warmup
    # allocation fragments and the server dies.
    env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    return env


def _tail(path: Path, n: int) -> None:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    for line in lines[-n:]:
        print("  | " + line)


def _wait_healthy(proc: subprocess.Popen, log_path: Path, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"FATAL: server exited rc={proc.returncode}; last 60 lines:")
            _tail(log_path, 60)
            raise RuntimeError("server died during startup")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    print("server healthy")
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(10)
    _tail(log_path, 60)
    raise RuntimeError(f"server not healthy within {timeout_s}s")


def _kv_pool_facts(log_path: Path) -> dict:
    """Pull the KV pool size and max concurrency out of the server's own log.

    This decides whether the A/B is valid. Both builds do independent memory
    profiling at util 0.94, and per session 28 the pool lands within about 10%
    of the bare minimum, so if one server ends up with fewer KV blocks than the
    other then throughput differs for a reason that has nothing to do with the
    kernels. vLLM logs both numbers at startup
    (vllm/v1/core/kv_cache_utils.py); nothing was reading them.
    """
    facts: dict = {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return facts
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
    if m:
        facts["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", text)
    if m:
        facts["max_model_len_tokens"] = int(m.group(1).replace(",", ""))
        facts["max_concurrency_x"] = float(m.group(2))
    return facts


def _assert_kv_pools_match(per_build: dict) -> None:
    """Refuse to publish a comparison between two different KV budgets."""
    sizes = {b: f.get("kv_cache_tokens") for b, f in per_build.items()}
    known = {b: v for b, v in sizes.items() if v}
    if len(known) < 2:
        print(f"KV_POOL could not be read for every build: {sizes}")
        return
    lo, hi = min(known.values()), max(known.values())
    drift = (hi - lo) / hi
    print(f"KV_POOL per build: {known}, drift {drift:.2%}")
    if drift > 0.02:
        raise RuntimeError(
            f"KV pool differs by {drift:.1%} between builds ({known}). "
            "Throughput would differ for a memory reason rather than a kernel "
            "reason, so this comparison is not publishable. Re-run cold."
        )


CORRECTNESS_PROMPTS = [
    "List the first five prime numbers.",
    "Write one sentence about the ocean.",
    "What is 17 times 23?",
    "Name three primary colors.",
]


def _greedy_probe(build: str) -> dict:
    """Greedy completions from the running server, for a build-vs-build check.

    The benchmark runs with --ignore-eos, which means numerically garbage
    output is indistinguishable from good output in every metric it collects.
    Nothing else in this function confirms that the ours server actually took
    our kernel path and produced sane tokens. Four prompts at temperature 0
    costs seconds and turns a throughput number into a throughput number you
    can publish.
    """
    out = {"build": build, "completions": []}
    for prompt in CORRECTNESS_PROMPTS:
        body = json.dumps({
            "model": SERVED_NAME,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0.0,
            "seed": SEED,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            out["completions"].append(data["choices"][0]["text"])
        except Exception as exc:  # noqa: BLE001
            out["completions"].append(f"<ERROR {type(exc).__name__}: {exc}>")
    return out


def _start_server(build: str, log_path: Path) -> subprocess.Popen:
    cmd = [
        _vllm_bin(),
        "serve",
        MODEL_DIR,
        "--served-model-name",
        SERVED_NAME,
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEM_UTIL),
        "--seed",
        str(SEED),
        "--port",
        str(PORT),
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    print(f"starting server ({build}): {' '.join(cmd)}", flush=True)
    fh = log_path.open("w", buffering=1)
    # Own session so the whole TP worker tree can be signalled at once.
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=_server_env(),
        start_new_session=True,
    )


def _stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=SERVER_STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
    # Let the TP8 workers and NCCL actually release device memory.
    time.sleep(GPU_RELEASE_SLEEP_S)


# --------------------------------------------------------------------------
# Persistence. Commit after EVERY run, never batched.
# --------------------------------------------------------------------------


_SM_SUFFIX = re.compile(r"_sm\d+(?=\.[^.]+$)")


def _artifact_candidates(work: Path, local_name: str) -> list:
    """Every on-disk file that could be `local_name` for some architecture.

    The parity harnesses name their output after the silicon they ran on,
    `parity_rel_chunked_decode_sm{cc}.json`, so the step table cannot hardcode
    one capability. It did: it asked for `_sm90.json` on every arch, so the
    A100 session marked a 7/7 pass as FAIL with a missing artifact while `rc`
    was 0 and the JSON sat right there as `_sm80.json`. Match the family.
    """
    exact = work / local_name
    if exact.exists():
        return [exact]
    if not _SM_SUFFIX.search(local_name):
        return []
    pattern = _SM_SUFFIX.sub("_sm*", local_name)
    return sorted(work.glob(pattern))


def _resolve_artifact(work: Path, local_name: str):
    """The one candidate to publish, or None. Ambiguity is a failure, not a
    coin flip: two capability-suffixed files in one workdir means a stale one
    survived deletion, and publishing either would be a guess."""
    found = _artifact_candidates(work, local_name)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        print(f"ARTIFACT AMBIGUOUS for {local_name}: "
              f"{[p.name for p in found]}")
    return None


def _write_committed(text: str, remote: Path, quiet: bool = False) -> None:
    remote.parent.mkdir(parents=True, exist_ok=True)
    tmp = remote.with_suffix(remote.suffix + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, remote)
    results_vol.commit()
    if not quiet:
        print(f"PERSISTED {remote}", flush=True)


def _persist(local: Path, remote: Path, note: str) -> None:
    """Copy to the Volume and commit NOW.

    This is the whole reason this script exists. The previous attempt
    batched persistence to the end of the matrix and lost everything when
    the box was terminated. Never batch this. Writing through a .partial
    then renaming means a kill mid-copy cannot leave a truncated JSON.
    """
    remote.parent.mkdir(parents=True, exist_ok=True)
    tmp = remote.with_suffix(remote.suffix + ".partial")
    shutil.copy2(local, tmp)
    os.replace(tmp, remote)
    results_vol.commit()
    print(f"PERSISTED {note} -> {remote}", flush=True)


def _persist_log_tail(log_path: Path, remote: Path, max_bytes: int = 2_000_000) -> None:
    try:
        data = log_path.read_bytes()
    except OSError:
        return
    _write_committed(data[-max_bytes:].decode("utf-8", errors="replace"), remote)


# --------------------------------------------------------------------------
# Modal functions
# --------------------------------------------------------------------------


@app.function(
    image=bench_image,
    # NO gpu= here. This is the single biggest cost lever: pulling 552GB on
    # the 8x H100 container would cost about $34 per hour of download.
    volumes={MODEL_MOUNT: model_vol, RESULTS_MOUNT: results_vol},
    timeout=DOWNLOAD_TIMEOUT_S,
    cpu=DL_CPU,
    memory=DL_MEMORY_MIB,
)
def download_model(force: int = 0) -> str:
    """Pull the NVFP4 checkpoint into the model Volume. CPU only, idempotent."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    rate = _download_usd_per_hour()
    print(f"=== download {MODEL_REPO} -> {MODEL_DIR} ===")
    print(f"CPU-only container at ${rate:.2f}/hr")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    if Path(MODEL_MARKER).exists() and not force:
        print("marker present, download already complete; no-op")
        return MODEL_DIR

    # hf_transfer is enabled via HF_HUB_ENABLE_HF_TRANSFER in the image env.
    # snapshot_download with local_dir writes files directly, resumes a
    # partial download, and does not duplicate into a hub blob cache.
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, max_workers=16)
    Path(MODEL_MARKER).write_text(
        json.dumps({"repo": MODEL_REPO, "utc": time.time()}), encoding="utf-8"
    )
    model_vol.commit()

    total = sum(f.stat().st_size for f in Path(MODEL_DIR).rglob("*") if f.is_file())
    hours = (time.time() - t0) / 3600.0
    usd = rate * hours
    print(f"downloaded {total / 1e9:.1f} GB in {hours * 60:.1f} min, ~${usd:.2f}")
    ledger_total = _ledger_upsert("download_model", hours, usd, f"{total / 1e9:.1f} GB")
    print(f"LEDGER total spend estimate: ${ledger_total:.2f} of ${BUDGET_USD:.0f}")
    return MODEL_DIR


@app.function(
    image=bench_image,
    gpu=f"{BENCH_GPU_KIND}:{N_GPU}",
    volumes={
        # Read only. The GPU container must never download anything.
        MODEL_MOUNT: model_vol.with_mount_options(read_only=True),
        RESULTS_MOUNT: results_vol,
    },
    timeout=BENCH_TIMEOUT_S,
    cpu=BENCH_CPU,
    memory=BENCH_MEMORY_MIB,
)
def run_bench(
    runs: int = DEFAULT_RUNS,
    builds: str = DEFAULT_BUILDS,
    concurrencies: str = DEFAULT_CONCURRENCIES,
    mixes: str = DEFAULT_MIXES,
    budget_usd: float = BUDGET_USD,
    server_wait_s: int = SERVER_WAIT_S,
) -> dict:
    """Deploy kernels, serve, benchmark, persist. No downloading."""
    t0 = time.time()
    rate_hr = _bench_usd_per_hour()

    mix_list = _parse_mixes(mixes)
    conc_list = _parse_concurrencies(concurrencies)
    configs = _ordered_configs(mix_list, conc_list)
    wanted = [b.strip() for b in builds.split(",") if b.strip()]
    for b in wanted:
        if b not in BUILDS:
            raise ValueError(f"unknown build {b!r}")

    _print_cost_banner("container", runs, len(configs), len(wanted))

    # Budget gate. The ledger accumulates across invocations, so a resume
    # cannot quietly blow past the cap.
    prior = _prior_spend("run_bench")
    remaining_usd = budget_usd - prior
    print(f"LEDGER prior spend estimate: ${prior:.2f}, remaining ${remaining_usd:.2f}")
    if remaining_usd < MIN_USEFUL_USD:
        msg = (
            f"ABORT: only ${remaining_usd:.2f} left under the ${budget_usd:.0f} "
            f"cap, less than the ${MIN_USEFUL_USD:.0f} needed to load the "
            "model and finish one run. Nothing was run."
        )
        print(msg)
        return {"aborted_before_start": True, "reason": msg, "prior_usd": prior}

    # Two independent deadlines, earliest wins.
    hard_deadline = t0 + BENCH_TIMEOUT_S - SHUTDOWN_MARGIN_S
    budget_deadline = t0 + (remaining_usd / rate_hr) * 3600.0
    deadline = min(hard_deadline, budget_deadline)
    print(
        f"effective deadline: {(deadline - t0) / 3600:.2f}h from now "
        f"(hard timeout {(hard_deadline - t0) / 3600:.2f}h, "
        f"budget {(budget_deadline - t0) / 3600:.2f}h)"
    )

    print("=== host ===")
    smi_out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    print(smi_out.stdout.strip())
    # Assert the silicon, before the model load, before any money goes on a
    # comparison that cannot be published. Asking for a GPU kind does not
    # guarantee it: a run on 2026-07-25 requested H100:8 and got 8x H200, which
    # changed the KV pool by 43x. Fail here rather than discover it in the
    # numbers.
    gpu_names = [
        ln.split(",")[0].strip()
        for ln in smi_out.stdout.splitlines() if ln.strip()
    ]
    wrong = [n for n in gpu_names if BENCH_GPU_KIND.upper() not in n.upper()]
    if wrong or len(gpu_names) != N_GPU:
        msg = (
            f"GPU MISMATCH. Requested {BENCH_GPU_KIND}:{N_GPU}, got "
            f"{len(gpu_names)} device(s): {gpu_names}. Refusing to spend on a "
            "comparison whose hardware is not what was asked for."
        )
        print(msg)
        _ledger_upsert(
            "run_bench", (time.time() - t0) / 3600.0,
            rate_hr * (time.time() - t0) / 3600.0, "aborted: gpu mismatch",
        )
        raise RuntimeError(msg)
    print(f"GPU_CHECK ok: {len(gpu_names)}x {BENCH_GPU_KIND}")
    # /dev/shm size matters: the vLLM v1 message queue lives there.
    subprocess.run(["df", "-h", "/dev/shm"], check=False)
    subprocess.run(["df", "-h", MODEL_MOUNT], check=False)

    # Read-only mount, already at the latest committed state at container
    # start. Nothing here downloads.
    if not Path(MODEL_MARKER).exists():
        raise RuntimeError(
            f"model not present at {MODEL_DIR}; run --step download first "
            "(CPU only, cheap)"
        )

    vllm_root, tml_pkg = _resolve_paths()
    print(f"vllm root: {vllm_root}")
    print(f"resolved tml_fa4 package dir: {tml_pkg}")
    _make_stock_backup(vllm_root, tml_pkg)

    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(LOGS_ROOT).mkdir(parents=True, exist_ok=True)

    manifest = {
        "app": APP_NAME,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vllm_pin": VLLM_PIN,
        "vllm_wheel": VLLM_WHEEL,
        "cuda_image": CUDA_IMAGE,
        "model_repo": MODEL_REPO,
        "gpu": f"H100:{N_GPU}",
        "serve": {
            "tensor_parallel_size": TP_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "enforce_eager": ENFORCE_EAGER,
            "seed": SEED,
            "PYTORCH_CUDA_ALLOC_CONF": ALLOC_CONF,
        },
        "matrix": {
            "builds": wanted,
            "mixes": [
                {"name": n, "input_len": i, "output_len": o} for n, i, o in mix_list
            ],
            "concurrencies": conc_list,
            "order_cheapest_first": [
                {"mix": n, "concurrency": c, "num_prompts": p}
                for n, c, _i, _o, p in configs
            ],
            "runs": runs,
            "min_prompts": MIN_PROMPTS,
        },
        "cost": {
            "usd_per_hour": round(rate_hr, 4),
            "budget_usd": budget_usd,
            "prior_spend_usd_est": prior,
            "hard_timeout_hours": BENCH_TIMEOUT_HOURS,
            "rates_quoted": "modal.com/pricing 2026-07-24",
        },
        "flag_evidence": BENCH_FLAG_EVIDENCE,
    }
    _write_committed(json.dumps(manifest, indent=2), Path(RESULTS_ROOT) / "manifest.json")

    completed, skipped, failed, aborted = 0, 0, 0, False
    kv_facts: dict = {}
    probes: dict = {}
    # Carry forward validity evidence from an earlier container. Resume skips
    # runs that already exist, so a build finished in a previous container is
    # never re-measured here, and without this its KV pool would drop out of the
    # comparison entirely. A cross-container stock-vs-ours comparison is only
    # defensible if the pools are shown equal, so load them rather than assume.
    _prior_validity = Path(RESULTS_ROOT) / "validity.json"
    if _prior_validity.exists():
        try:
            _pv = json.loads(_prior_validity.read_text(encoding="utf-8"))
            kv_facts.update(_pv.get("kv_pool") or {})
            probes.update(_pv.get("greedy_probe") or {})
            print(
                f"carried forward validity evidence for builds "
                f"{sorted(kv_facts)} from a previous container"
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: prior validity.json unreadable ({exc})")
    # aborted  = the matrix is incomplete, so say so in the summary.
    # hard_stop = out of budget or time for real, so stop the whole matrix.
    # Yielding at the end of one build's allowance sets the first, not the
    # second, so the next build still gets its turn.
    hard_stop = False
    scratch = Path("/tmp/bench_run")
    scratch.mkdir(parents=True, exist_ok=True)
    # Conservative reserve until a run has actually been timed.
    # Reserve for the FIRST run, before any duration has been observed. 1800 was
    # a guess made when no run had ever been timed. Measured on 8x H200 in
    # session 30: a conc-8 run of either mix takes about 3.4 minutes, and the
    # expensive outlier is decode at conc 1 at about 32 minutes, which the
    # forward-looking update below catches after the first sample.
    #
    # 1800 on top of a 2400 server wait demanded 70 minutes of headroom, which
    # refused a container that had 56 and would have completed five runs in 45.
    # That abort cost nothing, which is the guard working, but it also produced
    # nothing. 600 leaves room for one slow first run without vetoing the
    # container outright.
    run_reserve_s = 600.0

    def burn(note: str) -> None:
        hours = (time.time() - t0) / 3600.0
        usd = rate_hr * hours
        total = _ledger_upsert("run_bench", hours, usd, note)
        print(
            f"BURN elapsed {hours:.2f}h | this container ~${usd:.2f} | "
            f"ledger total ~${total:.2f} of ${budget_usd:.0f} | "
            f"{(deadline - time.time()) / 60:.0f} min left",
            flush=True,
        )

    burn("start")

    def _book_and_summarise(note: str) -> dict:
        """Always book the spend and write the verdict, including on a crash.

        Without this, a _wait_healthy failure after a 90 minute model load, or
        a Modal hard kill, left the last burn() at ~$0 while real money had
        been spent. _prior_spend then read the understated total and the next
        container computed its budget from money that was already gone, so the
        $200 cap silently stopped being a cap.
        """
        elapsed_h = (time.time() - t0) / 3600.0
        total = _ledger_upsert(
            "run_bench", elapsed_h, rate_hr * elapsed_h, note
        )
        summary = {
            "completed_runs": completed,
            "skipped_existing": skipped,
            "failed_runs": failed,
            "aborted_early": aborted,
            "outcome": note,
            "elapsed_hours": round(elapsed_h, 3),
            "spend_usd_est": round(rate_hr * elapsed_h, 2),
            "ledger_total_usd_est": total,
            "budget_usd": budget_usd,
            "results_root": RESULTS_ROOT,
        }
        _write_committed(
            json.dumps(summary, indent=2), Path(RESULTS_ROOT) / "summary.json"
        )
        return summary

    # Book the spend even if we never reach the normal tail. atexit covers an
    # unhandled exception and a normal-ish exit; the SIGTERM handler covers
    # Modal's container kill, which is what a hung run leads to. Neither can
    # help against SIGKILL, so the run timeout above is the primary defence and
    # this is the backstop.
    _booked = {"done": False}

    def _emergency_book(note: str) -> None:
        if _booked["done"]:
            return
        _booked["done"] = True
        try:
            print(f"BOOKING SPEND from the backstop path: {note}")
            print(json.dumps(_book_and_summarise(note), indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"backstop booking failed: {exc!r}")

    atexit.register(lambda: _emergency_book("exited without a final write"))

    def _on_sigterm(signum, _frame):
        _emergency_book(f"killed by signal {signum}")
        raise SystemExit(143)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError) as exc:
        print(f"could not install SIGTERM handler: {exc!r}")

    for build_idx, build in enumerate(wanted):
        pending = []
        for mix, conc, ilen, olen, npr in configs:
            for n in range(1, runs + 1):
                out = Path(RESULTS_ROOT) / build / mix / str(conc) / f"run{n}.json"
                if out.exists():
                    skipped += 1
                else:
                    pending.append((mix, conc, ilen, olen, npr, n, out))
        if not pending:
            print(f"=== BUILD {build}: all {runs}-run results present, skipping ===")
            continue
        if time.time() + run_reserve_s + server_wait_s > deadline:
            print(f"=== STOP before build {build}: not enough budget/time left ===")
            aborted = True
            hard_stop = True
            break

        # Build is the OUTER loop, so a global deadline lets the first build eat
        # the container and the second never start. That is not a partial
        # result, it is no A/B at all, and it already happened once on Lambda in
        # session 28: stock finished, ours never ran. Each remaining build gets
        # an equal share of the time left, minus its own server startup.
        builds_left = max(1, len(wanted) - build_idx)
        share = (deadline - time.time()) / builds_left
        build_deadline = min(deadline, time.time() + share)
        if builds_left > 1:
            print(
                f"=== BUILD {build}: allowance "
                f"{(build_deadline - time.time()) / 60:.0f} min of the "
                f"{(deadline - time.time()) / 60:.0f} min left, so the "
                f"{builds_left - 1} build(s) after it still get a turn ==="
            )

        print(f"=== BUILD {build}: {len(pending)} runs pending ===")
        _deploy_build(build, vllm_root, tml_pkg)

        log_path = Path(f"/tmp/serve_{build}.log")
        proc = None
        try:
            proc = _start_server(build, log_path)
            _wait_healthy(proc, log_path, server_wait_s)
            _persist_log_tail(log_path, Path(LOGS_ROOT) / f"serve_{build}_startup.log")
            burn(f"{build} server ready")

            # Validity evidence, collected before any timing. Both cost seconds
            # and both are the difference between a number and a publishable
            # number.
            kv_facts[build] = _kv_pool_facts(log_path)
            print(f"KV_POOL {build}: {kv_facts[build]}")
            probes[build] = _greedy_probe(build)
            for p, c in zip(CORRECTNESS_PROMPTS, probes[build]["completions"]):
                print(f"PROBE {build}: {p!r} -> {c[:80]!r}")
            _write_committed(
                json.dumps({"kv_pool": kv_facts, "greedy_probe": probes},
                           indent=2),
                Path(RESULTS_ROOT) / "validity.json",
            )

            for mix, conc, ilen, olen, npr, n, out in pending:
                # build_deadline, not deadline: yielding here leaves matched
                # partial curves for both builds instead of one complete build
                # and one empty one.
                if time.time() + run_reserve_s > build_deadline:
                    out_of_budget = time.time() + run_reserve_s > deadline
                    print(
                        f"=== STOP mid-build {build}: "
                        f"{(build_deadline - time.time()) / 60:.0f} min left of "
                        f"this build's allowance, reserve "
                        f"{run_reserve_s / 60:.0f} min. "
                        + (
                            "Out of budget entirely."
                            if out_of_budget
                            else "Yielding so the next build gets its turn."
                        )
                        + " ==="
                    )
                    aborted = True
                    # Only a genuine budget/time exhaustion should end the
                    # matrix. Hitting this build's own share must fall through
                    # to the next build, or the allowance would just be a
                    # slower way of producing one complete build and one empty.
                    hard_stop = out_of_budget
                    break

                tag = f"{build}/{mix}/conc{conc}/run{n}"
                print(
                    f"=== RUN {tag}: in={ilen} out={olen} np={npr} ===",
                    flush=True,
                )

                # Write to scratch first so a kill mid-write can never leave
                # a truncated JSON on the Volume.
                for stale in scratch.glob("*.json"):
                    stale.unlink()
                local_json = scratch / f"run{n}.json"
                cmd = [
                    _vllm_bin(),
                    "bench",
                    "serve",
                    "--backend",
                    "openai",
                    "--base-url",
                    BASE_URL,
                    "--model",
                    MODEL_DIR,
                    "--served-model-name",
                    SERVED_NAME,
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(ilen),
                    "--random-output-len",
                    str(olen),
                    "--num-prompts",
                    str(npr),
                    "--seed",
                    str(SEED),
                    "--ignore-eos",
                    "--max-concurrency",
                    str(conc),
                    "--percentile-metrics",
                    "ttft,tpot,itl,e2el",
                    "--metric-percentiles",
                    "99",
                    "--save-result",
                    "--result-dir",
                    str(scratch),
                    "--result-filename",
                    local_json.name,
                ]
                r0 = time.time()
                # The deadline is otherwise only checked BETWEEN runs, so one
                # stalled request hangs the client until Modal hard-kills the
                # container, losing the whole remaining matrix. That is a live
                # risk here: --ignore-eos forces every request to full length
                # while the KV pool holds far fewer sequences than the offered
                # concurrency, so preemption thrash is expected.
                run_timeout_s = max(120.0, deadline - time.time())
                try:
                    rc = subprocess.run(
                        cmd,
                        env=_server_env(),
                        check=False,
                        timeout=run_timeout_s,
                    ).returncode
                except subprocess.TimeoutExpired:
                    rc = -9
                    print(
                        f"RUN TIMEOUT after {run_timeout_s:.0f}s: {tag}. "
                        "Killed rather than allowed to burn to the container cap."
                    )
                dur = time.time() - r0

                if rc == 0 and local_json.exists():
                    _persist(local_json, out, tag)
                    completed += 1
                    # Reserve tracks the slowest observed run, with headroom.
                    run_reserve_s = max(run_reserve_s * 0.5, dur * 1.6, 300.0)
                else:
                    failed += 1
                    print(f"RUN FAILED rc={rc}: {tag} (left absent = null in summary)")
                    _persist_log_tail(
                        log_path,
                        Path(LOGS_ROOT) / f"failed_{build}_{mix}_{conc}_run{n}.log",
                    )

                _write_committed(
                    json.dumps(
                        {
                            "completed": completed,
                            "skipped": skipped,
                            "failed": failed,
                            "last": tag,
                            "last_rc": rc,
                            "last_seconds": round(dur, 1),
                            "elapsed_hours": round((time.time() - t0) / 3600, 3),
                            "spend_usd_est": round(
                                rate_hr * (time.time() - t0) / 3600, 2
                            ),
                        },
                        indent=2,
                    ),
                    Path(RESULTS_ROOT) / "progress.json",
                    quiet=True,
                )
                burn(f"after {tag}")
        finally:
            _persist_log_tail(log_path, Path(LOGS_ROOT) / f"serve_{build}.log")
            _stop_server(proc)

        if hard_stop:
            break

    # Validity verdicts, once both builds have been seen. These do not gate the
    # run (the numbers are already on the Volume) but they decide what may be
    # published from it, so they are recorded explicitly rather than inferred.
    validity: dict = {"kv_pool": kv_facts, "greedy_probe": probes}
    try:
        _assert_kv_pools_match(kv_facts)
        validity["kv_pool_match"] = True
    except RuntimeError as exc:
        validity["kv_pool_match"] = False
        validity["kv_pool_error"] = str(exc)
        print(f"VALIDITY FAILURE: {exc}")

    if len(probes) == 2:
        a, b = sorted(probes)
        same = probes[a]["completions"] == probes[b]["completions"]
        validity["greedy_match"] = same
        validity["greedy_builds"] = [a, b]
        print(f"GREEDY_PROBE {a} vs {b}: "
              + ("identical" if same else "DIFFER, see validity.json"))
        if not same:
            validity["greedy_note"] = (
                "The two builds produced different greedy text on at least one "
                "prompt. That is not automatically a defect: the platform is "
                "not batch-deterministic at TP8 across 66 bf16 layers, and the "
                "32-prompt gate in gate_logit_parity_8xh100.json measured the "
                "same-build control disagreeing with itself. Read the actual "
                "completions before drawing a conclusion."
            )
    _write_committed(
        json.dumps(validity, indent=2), Path(RESULTS_ROOT) / "validity.json"
    )

    _booked["done"] = True
    summary = _book_and_summarise("final")
    summary["validity"] = validity
    _write_committed(
        json.dumps(summary, indent=2), Path(RESULTS_ROOT) / "summary.json"
    )
    print("=== BENCH DONE ===")
    print(json.dumps(summary, indent=2))
    if aborted:
        print(
            "Matrix incomplete. Re-run `modal run --detach "
            "scripts/modal_e2e_bench.py --step bench` to resume; existing "
            "run JSONs are skipped and the ledger keeps enforcing the cap. "
            "If the remaining budget is tight, trim with RUNS=3."
        )
    return summary


# --------------------------------------------------------------------------
# Validate step: parity gates + microbenches on ONE H100. No model.
# --------------------------------------------------------------------------

_ENV_PROBE_SRC = """
import json

import torch

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    version = None


def _v(name):
    if version is None:
        return None
    try:
        return version(name)
    except Exception:  # noqa: BLE001
        return None


props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "sm_count": props.multi_processor_count,
    "total_memory_gb": round(props.total_memory / 1e9, 1),
    "device_count": torch.cuda.device_count(),
    "vllm": _v("vllm"),
    "nvidia_cutlass_dsl": _v("nvidia-cutlass-dsl"),
}, indent=2))
"""


# The one thing the existing harnesses cannot express. Written into the
# staging directory at run time; it is NOT a repo file, and it adds no new
# kernel interface: section A uses the `bias=` argument that
# kernels/patches/u2_shear_fusion.py adds to flash_attn_varlen_func, section B
# uses the `num_splits=` argument that has always been there
# (kernels/tml_fa4_modified/interface.py:1543).
EXTRA_MICROBENCH_SRC = r'''#!/usr/bin/env python3
"""Generated by scripts/modal_e2e_bench.py --step validate. Not a repo file.

WHY THIS EXISTS. INKLING_TURBO_FUSED_SHEAR=1 does NOT change what
microbench_attn_day0.py measures. That harness calls tml-fa4's
flash_attn_varlen_func directly with rel_bias=<natural layout>, so the
ShearingBias pre-kernel still runs inside the timed region. The env var is
read only by vllm/models/inkling/nvidia/ops/fa4_rel_attention.py::
use_fused_shear (kernels/patches/u2_shear_fusion.py, FA4_HELPER_NEW), which
only the vLLM serving path calls, and which additionally requires
_use_sheared_bias() to be true. kernels/patches/u2_shear_fusion_notes.md step
4 states the same thing about parity_fa4_rel.py. What actually activates the
fusion at the kernel boundary is passing the ALREADY-SHEARED buffer as
`bias=` with rel_bias=None (IFACE_NORMALIZE / IFACE_BLOCK edits).

A. presheared_*  times the attention call with the shear already done. The
   sheared buffer is built OUT OF BAND by the stock ShearingBias kernel, via
   harness/parity_shear_fusion.py::run_shearing_bias, so no layout arithmetic
   is invented here. Each case is parity-checked against the rel_bias= path
   before its timing is reported; a case that fails parity reports a null
   timing. The matching __rel_bias_natural case is timed in the SAME process
   on the SAME tensors, so the difference between the two is the cost the
   fusion removes, measured rather than subtracted from another session.

   These numbers are the SAVING only. The cost is section B.

B. writer_*  the other half of the same equation, and the number
   u2_shear_fusion_notes.md calls "the measurement that does not exist yet".
   Times (natural writer + ShearingBias) against (fused sheared writer) on the
   same inputs in the same process. The fusion is a net win on a shape only if

       writer_*.writer_delta_us_per_iter  <  presheared_*.saved_us_per_iter

   Read the two sections together or not at all. Either one alone overstates.

C. splitkv_*  decode with num_splits > 1 on sm_90. Never run on Hopper before.
   Every split count is compared against num_splits=1 output.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

import microbench_attn_day0 as mb
import parity_shear_fusion as psf

RESULTS: dict = {}
D = 128
DEV = "cuda"
OUT = Path(__file__).with_suffix(".json")


def save() -> None:
    """Write after every case: a crash must not cost the earlier ones."""
    OUT.write_text(json.dumps(RESULTS, indent=2))


def record(name: str, timed_key: str, extra: dict) -> None:
    rec = dict(mb.RESULTS.get(timed_key, {}))
    rec.update(extra)
    RESULTS[name] = rec


def presheared_case(name, T_q, T_k, Hq, Hkv, ext, is_local):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    torch.manual_seed(11)
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=DEV) / D**0.25
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV) / D**0.25
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV)
    rel = torch.randn(T_q, Hq, ext, dtype=torch.bfloat16, device=DEV) * 0.3
    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=DEV)
    seqused_k = torch.tensor([T_k], dtype=torch.int32, device=DEV)

    # Same convention as harness/parity_shear_fusion.py::case_rel_proj.
    window_left = ext - 1 if is_local else None
    window_right = 0 if is_local else None

    # run_shearing_bias reads psf.HQ only to shape its output buffer; the
    # harness pins it at 8 for its own cases and the microbench uses 64.
    psf.HQ = Hq
    sheared = psf.run_shearing_bias(
        rel,
        ext=ext,
        is_local=is_local,
        window_left=window_left,
        window_right=window_right,
        cu_seqlens_q=cu_q,
        seq_lens=seqused_k,
        max_seqlen_q=T_q,
        max_seqlen_k=T_k,
    )

    kw = dict(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=T_q, max_seqlen_k=T_k,
        softmax_scale=1.0 / D,
        causal=True,
        window_size=(window_left, 0) if is_local else (None, None),
        # The bias path forces this off on sm_90 anyway (journal/upstream/04);
        # pinned so the two timed calls cannot differ in packing.
        pack_gqa=False,
    )

    def run(**bias_kw):
        out = flash_attn_varlen_func(**kw, **bias_kw)
        return out[0] if isinstance(out, tuple) else out

    # This kernel is not run-to-run deterministic, so the reference is run
    # twice to measure its own noise and the fused-vs-reference difference has
    # to sit inside it. Same discipline as parity_shear_fusion.py.
    ref = run(rel_bias=rel).float()
    ref2 = run(rel_bias=rel).float()
    got = run(bias=sheared).float()
    noise = (ref - ref2).abs()
    noise_mx, noise_mean = float(noise.max()), float(noise.mean())
    diff = (ref - got).abs()
    mx, mean = float(diff.max()), float(diff.mean())
    tol_mx, tol_mean = max(4.0 * noise_mx, 1e-2), max(4.0 * noise_mean, 1e-4)
    parity_ok = bool(mx <= tol_mx and mean <= tol_mean)
    parity = {
        "parity_ok": parity_ok,
        "max_abs_diff": mx,
        "mean_abs_diff": mean,
        "ref_vs_ref_noise_max": noise_mx,
        "ref_vs_ref_noise_mean": noise_mean,
        "tol_max": tol_mx,
        "tol_mean": tol_mean,
    }
    print(f"[{name}] presheared parity {'OK' if parity_ok else 'FAIL'}: "
          f"max={mx:.4e} tol={tol_mx:.4e}")

    nat = name + "__rel_bias_natural"
    mb.profile_case(nat, lambda: run(rel_bias=rel))
    record(nat, nat, {"path": "rel_bias natural, ShearingBias inside the timing"})

    if parity_ok:
        mb.profile_case(name, lambda: run(bias=sheared))
        record(name, name, {
            "path": "bias= pre-sheared, ShearingBias NOT in the timing",
            "parity_vs_rel_bias": parity,
        })
        nat_total = RESULTS[nat].get("total_us_per_iter")
        cur_total = RESULTS[name].get("total_us_per_iter")
        if nat_total and cur_total:
            RESULTS[name]["saved_us_per_iter_vs_natural"] = round(
                nat_total - cur_total, 1)
    else:
        RESULTS[name] = {
            "total_us_per_iter": None,
            "path": "bias= pre-sheared",
            "parity_vs_rel_bias": parity,
            "note": "parity failed, timing withheld on purpose",
        }


def set_geometry(Hq, Hkv):
    """Point parity_shear_fusion's input builder at production head counts.

    make_inputs() and run_shearing_bias() read these as module globals, and
    WIDTH is DERIVED from them at import time, so setting HQ alone silently
    builds a qkvr tensor of the wrong width. All five move together.
    """
    psf.HQ, psf.HKV = Hq, Hkv
    psf.QW, psf.KVW = Hq * psf.D, Hkv * psf.D
    psf.R_OFFSET = psf.QW + 2 * psf.KVW
    psf.WIDTH = psf.R_OFFSET + Hq * psf.D_REL


def writer_case(name, seq_lens_q, ctx_lens, ext, Hq, Hkv, is_local):
    """The COST side of the fusion, which nothing has measured until now.

    presheared_* above measures what attention SAVES when the bias arrives
    pre-sheared. It does not measure what producing it costs. The fused writer
    emits rel_extent + 256 columns into a (T + 128, H, ext + 256) buffer
    instead of rel_extent columns into (T, H, ext), so it is strictly more
    work than the natural writer. The fusion is a net win only if

        (natural writer + ShearingBias) - (fused writer)

    is positive, and no number on either side of that existed before this run.
    Both paths are timed here in ONE process on THE SAME inputs, so the delta
    is measured rather than subtracted across sessions.

    The fused output is checked against the stock ShearingBias result before
    any timing is reported. A writer that is fast because it is wrong is not a
    result, so a parity failure withholds the numbers instead of printing them.
    """
    from vllm.models.inkling.nvidia.ops.qkvr_prep import (
        RelShearSpec,
        qkvr_rel_proj,
    )

    set_geometry(Hq, Hkv)
    inp = psf.make_inputs(seq_lens_q, ctx_lens, ext, seed=17)
    tokens = inp["tokens"]
    dev = inp["qkvr"].device
    common = dict(
        num_q_heads=Hq, num_kv_heads=Hkv, head_dim=psf.D, d_rel=psf.D_REL
    )
    window_left = ext - 1 if is_local else None
    window_right = 0 if is_local else None

    rel_nat = torch.empty(tokens, Hq, ext, dtype=torch.bfloat16, device=dev)
    sheared_out = torch.empty(
        (tokens + psf.SHEAR_ROW_PAD, Hq, ext + psf.SHEAR_PAD),
        dtype=torch.bfloat16, device=dev,
    )
    spec = RelShearSpec(
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        seq_idx=inp["seq_idx"],
        num_tokens=tokens,
        window_left=window_left,
        window_right=0,
    )

    def natural_plus_shear():
        qkvr_rel_proj(inp["qkvr"], inp["rel_proj"], rel_nat, None, **common)
        psf.run_shearing_bias(
            rel_nat,
            ext=ext,
            is_local=is_local,
            window_left=window_left,
            window_right=window_right,
            cu_seqlens_q=inp["cu_seqlens_q"],
            seq_lens=inp["seq_lens"],
            max_seqlen_q=max(seq_lens_q),
            max_seqlen_k=int(inp["seq_lens"].max()),
        )

    def fused():
        qkvr_rel_proj(
            inp["qkvr"], inp["rel_proj"], sheared_out, None,
            shear=spec, **common,
        )

    # Correctness first. ShearingBias does no arithmetic, so this is exact.
    qkvr_rel_proj(inp["qkvr"], inp["rel_proj"], rel_nat, None, **common)
    torch.cuda.synchronize()
    ref = psf.run_shearing_bias(
        rel_nat,
        ext=ext,
        is_local=is_local,
        window_left=window_left,
        window_right=window_right,
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        max_seqlen_q=max(seq_lens_q),
        max_seqlen_k=int(inp["seq_lens"].max()),
    )
    sheared_out.fill_(float("nan"))
    fused()
    torch.cuda.synchronize()
    # ShearingBias does no arithmetic, so the whole buffer must match bit for
    # bit. torch.equal stays on the GPU and is false on any NaN, which is
    # exactly the unwritten-column case. psf.compare() is the diagnostic path
    # only: it moves both buffers to CPU as float32, which at these production
    # shapes is several GB, so it runs on a slice and only after a failure.
    exact = bool(torch.equal(ref[:tokens], sheared_out[:tokens]))
    if exact:
        errs = []
    else:
        probe = min(tokens, 2048)
        errs = psf.compare(ref, sheared_out, probe, ext, name) or [
            f"{name}: buffers differ, but the first {probe} tokens match. "
            "The defect is further in; rerun parity_shear_fusion.py."
        ]
    parity_ok = not errs
    print(f"[{name}] fused writer parity {'OK' if parity_ok else 'FAIL'}")
    for line in errs[:4]:
        print(f"    {line}")

    nat = name + "__natural_writer_plus_shearingbias"
    mb.profile_case(nat, natural_plus_shear, iters=20, warmup=5)
    record(nat, nat, {
        "path": "qkvr_rel_proj natural layout, then the ShearingBias kernel",
        "shape": {"tokens": tokens, "Hq": Hq, "Hkv": Hkv, "ext": ext,
                  "is_local": is_local},
    })

    if not parity_ok:
        RESULTS[name] = {
            "total_us_per_iter": None,
            "path": "qkvr_rel_proj writing the sheared buffer directly",
            "parity_ok": False,
            "parity_errors": errs[:8],
            "note": "parity failed, timing withheld on purpose",
        }
        return

    mb.profile_case(name, fused, iters=20, warmup=5)
    record(name, name, {
        "path": "qkvr_rel_proj writing the sheared buffer directly",
        "parity_ok": True,
        "shape": {"tokens": tokens, "Hq": Hq, "Hkv": Hkv, "ext": ext,
                  "is_local": is_local},
    })
    a = RESULTS[nat].get("total_us_per_iter")
    b = RESULTS[name].get("total_us_per_iter")
    if a and b:
        RESULTS[name]["writer_delta_us_per_iter"] = round(b - a, 1)
        RESULTS[name]["writer_delta_note"] = (
            "positive means the fused writer costs MORE than the natural "
            "writer plus ShearingBias combined, which makes the fusion a net "
            "loss on this shape. Attention consumes an identical buffer in "
            "both paths, so this delta IS the whole effect of the fusion, not "
            "half of it."
        )
        # run_shearing_bias returns a NaN-initialised buffer so the parity gate
        # can catch unwritten columns. Production does not: interface.py:725
        # and :735 allocate the bias with torch.empty. That torch.full shows up
        # as a vectorized_elementwise kernel inside the natural path and is
        # harness scaffolding, so charging it to the natural path flatters the
        # fusion. Record the corrected figure here rather than making whoever
        # reads the JSON redo the subtraction.
        fill = 0.0
        for kname, kus in (RESULTS[nat].get("kernels_us") or {}).items():
            if "vectorized_elementwise" in kname:
                fill += float(kus)
        if fill:
            RESULTS[name]["harness_nan_prefill_us"] = round(fill, 1)
            RESULTS[name]["natural_total_excl_harness_prefill_us"] = round(
                a - fill, 1
            )
            RESULTS[name]["writer_delta_excl_harness_prefill_us"] = round(
                b - (a - fill), 1
            )
            RESULTS[name]["harness_nan_prefill_note"] = (
                "the natural path is timed with run_shearing_bias, which "
                "torch.full(NaN)s its output every iteration for the parity "
                "gate. Production uses torch.empty (interface.py:725,735), so "
                "writer_delta_excl_harness_prefill_us is the production-"
                "representative number and writer_delta_us_per_iter is a "
                "conservative bound that favours the fusion."
            )


def splitkv_case(name, B, L, Hq, Hkv, ext, splits):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    # Same construction as microbench_attn_day0.batched_decode_case: B true
    # sequences of one query token each, every one with its own KV of length
    # L, which is what makes the grid heads x batch.
    torch.manual_seed(23)
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=DEV) / D**0.25
    k = torch.randn(B * L, Hkv, D, dtype=torch.bfloat16, device=DEV) / D**0.25
    v = torch.randn(B * L, Hkv, D, dtype=torch.bfloat16, device=DEV)
    rel = torch.randn(B, Hq, ext, dtype=torch.bfloat16, device=DEV) * 0.3
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=DEV)
    cu_k = torch.arange(B + 1, dtype=torch.int32, device=DEV) * L

    def run(num_splits):
        out = flash_attn_varlen_func(
            q=q, k=k, v=v, rel_bias=rel,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=1.0 / D, causal=True, window_size=(None, None),
            num_splits=num_splits, pack_gqa=False)
        return out[0] if isinstance(out, tuple) else out

    base = run(1).float()
    for num_splits in splits:
        case = f"{name}_splits{num_splits}"
        try:
            got = run(num_splits).float()
            d = (got - base).abs()
            vs = {
                "max_abs_diff": float(d.max()),
                "mean_abs_diff": float(d.mean()),
            }
            mb.profile_case(case, lambda ns=num_splits: run(ns))
            record(case, case, {"num_splits": num_splits,
                                "vs_num_splits1": vs})
            print(f"[{case}] vs num_splits=1: max={vs['max_abs_diff']:.4e} "
                  f"mean={vs['mean_abs_diff']:.4e}")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            RESULTS[case] = {"num_splits": num_splits,
                             "error": traceback.format_exc(limit=4)}
        save()


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")

    # Shapes identical to microbench_attn_day0.py so the JSONs line up:
    # global 64q/8kv ext1024, SWA 64q/16kv ext512 win511.
    presheared = [
        ("presheared_prefill_global_8k",
         dict(T_q=8192, T_k=8192, Hq=64, Hkv=8, ext=1024, is_local=False)),
        ("presheared_prefill_swa_8k",
         dict(T_q=8192, T_k=8192, Hq=64, Hkv=16, ext=512, is_local=True)),
        ("presheared_decode_b1_global_kv64k",
         dict(T_q=1, T_k=65536, Hq=64, Hkv=8, ext=1024, is_local=False)),
        ("presheared_decode_b32_global_kv64k",
         dict(T_q=32, T_k=65536, Hq=64, Hkv=8, ext=1024, is_local=False)),
    ]
    for name, kwargs in presheared:
        try:
            presheared_case(name, **kwargs)
        except Exception:  # noqa: BLE001
            print(f"[{name}] FAILED:")
            traceback.print_exc()
            RESULTS[name] = {"error": traceback.format_exc(limit=4)}
        save()

    # AFTER presheared_*, because these mutate parity_shear_fusion's geometry
    # globals and presheared_case sets psf.HQ for itself.
    #
    # The prefill shapes are the ones that matter: session 25 measured
    # ShearingBias at 827.2 us of the 3308.8 us global prefill and 460.9 us of
    # the 1223.0 us sliding-window prefill. Those are the costs the fusion
    # claims to remove; writer_case measures what it charges to remove them.
    writers = [
        ("writer_prefill_global_8k",
         dict(seq_lens_q=[8192], ctx_lens=[0], ext=1024, Hq=64, Hkv=8,
              is_local=False)),
        ("writer_prefill_swa_8k",
         dict(seq_lens_q=[8192], ctx_lens=[0], ext=512, Hq=64, Hkv=16,
              is_local=True)),
        ("writer_decode_b32_global_kv64k",
         dict(seq_lens_q=[1] * 32, ctx_lens=[65535] * 32, ext=1024, Hq=64,
              Hkv=8, is_local=False)),
    ]
    for name, kwargs in writers:
        try:
            writer_case(name, **kwargs)
        except Exception:  # noqa: BLE001
            print(f"[{name}] FAILED:")
            traceback.print_exc()
            RESULTS[name] = {"error": traceback.format_exc(limit=4)}
        save()

    splitkv = [
        ("splitkv_decode_1seq_global_kv64k", 1, 65536, [1, 4, 8, 16]),
        ("splitkv_decode_32seqs_global_kv64k", 32, 65536, [1, 2, 4]),
    ]
    for name, B, L, splits in splitkv:
        try:
            splitkv_case(name, B, L, 64, 8, 1024, splits)
        except Exception:  # noqa: BLE001
            print(f"[{name}] FAILED:")
            traceback.print_exc()
            RESULTS[name] = {"error": traceback.format_exc(limit=4)}
        save()

    save()
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
'''


def _stage_harness() -> str:
    """Copy the harnesses somewhere writable and drop the generated driver in.

    The harnesses write their JSON next to themselves and
    parity_kv_fp8.py imports parity_qkvr_prep from its own directory, so they
    must run from a real directory with the repo's relative layout:
        <work>/harness/*.py
        <work>/kernels/relproj_score_mod.py   (parity_fa4_rel backend 3)
    """
    work = Path(WORK_ROOT)
    harness = work / "harness"
    kernels = work / "kernels"
    harness.mkdir(parents=True, exist_ok=True)
    kernels.mkdir(parents=True, exist_ok=True)

    staged = []
    for p in sorted(Path(HARNESS_SRC_DIR).glob("*.py")):
        shutil.copy2(p, harness / p.name)
        staged.append(p.name)
    if not staged:
        raise RuntimeError(f"no harness files at {HARNESS_SRC_DIR}")
    shutil.copy2(RELPROJ_SRC, kernels / "relproj_score_mod.py")
    (harness / "extra_microbench.py").write_text(
        EXTRA_MICROBENCH_SRC, encoding="utf-8"
    )
    _purge_pycache(str(work))
    print(f"staged {len(staged)} harness files -> {harness}")
    print("staged: " + " ".join(staged))
    return str(harness)


def _deploy_validate_build(
    vllm_root: str, tml_pkg: str, patches: str = "u3+shear"
) -> dict:
    """Deploy our kernels plus the selected patch set.

    `patches` selects what goes on top of our kernels:

      "u3+shear"  u3_fp8_kv then u2_shear_fusion. Everything the shear gates
                  need. THE ORDER IS LOAD-BEARING, and the reason is the
                  reverse of the obvious one: u3 anchors on STOCK text that
                  u2_shear_fusion rewrites (the `conv_block_size: /
                  log_scaling:` signature tail, the `_run_fused_small(` call
                  tail), so u3 has to land while that stock form still exists.
                  u2 has no anchor on anything u3 introduces and applies
                  cleanly on its own. Both apply() calls assert every anchor,
                  so the wrong order aborts loudly instead of half-patching.
      "route"     u2_serving_route only. This is EXACTLY what the 8x e2e
                  bench deploys for its "ours" build (see _deploy_build), so
                  it is the configuration whose health decides whether the
                  expensive run is worth starting.
      "none"      our kernels, no patches.

    Session 26 is why this switch exists. Running "u3+shear" broke every
    attention call on sm_90 with an unbound `n_block` in flash_fwd_sm90.py,
    and with only one configuration ever tested there was no way to tell
    whether the defect was in a patch or in the kernel itself. That
    distinction decides whether the published sm_90 speedups are still
    reproducible, so it needs to be answerable for the price of one small
    container rather than one 8-GPU container.
    """
    if patches not in ("u3+shear", "route", "none"):
        raise ValueError(f"unknown patch set {patches!r}")
    ours = sorted(Path(OURS_TML_DIR).glob("*.py"))
    if not ours:
        raise RuntimeError(f"no kernels in {OURS_TML_DIR}")

    # Restore stock first so a reused warm container starts from the same place
    # every time, then overlay ours.
    #
    # Both halves matter. u3 and u2 are each individually idempotent, but they
    # are NOT idempotent as a chain: u2's QKVR_FUSED_SIG edit rewrites the
    # signature line u3 anchors on, destroying both u3's `old` anchor and its
    # already-applied check, so re-running the chain on a patched tree aborts
    # with "anchor not found in qkvr_prep.py". Restoring tml_fa4 alone left
    # that trap armed on any reused warm container.
    for p in sorted(Path(STOCK_TML_DIR).glob("*.py")):
        shutil.copy2(p, Path(tml_pkg) / p.name)
    _restore_stock_ops(vllm_root)
    for p in ours:
        shutil.copy2(p, Path(tml_pkg) / p.name)

    checked = []
    for p in ours:
        got = Path(tml_pkg) / p.name
        if _sha8(got) != _sha8(p):
            raise RuntimeError(f"deploy check: {got.name} does not match {p}")
        checked.append(f"{p.name}:{_sha8(p)}")
    print(f"DEPLOY_CHECK pkg={tml_pkg}")
    print("DEPLOY_CHECK files=" + " ".join(checked))

    # u2_shear_fusion.py edits ROOT/vllm/third_party/tml_fa4/interface.py by
    # path. If the interpreter imports tml_fa4 from somewhere else, that edit
    # would land on a file nobody loads, which is a silent no-op. Mirror our
    # kernels into the path the patch targets, patch, then copy back.
    src_tml = Path(vllm_root) / "vllm/third_party/tml_fa4"
    same_tree = src_tml.resolve() == Path(tml_pkg).resolve()
    print(
        f"PATCH_TARGET source-tree={src_tml} imported={tml_pkg} "
        f"same={same_tree}"
    )
    if not same_tree:
        src_tml.mkdir(parents=True, exist_ok=True)
        for p in ours:
            shutil.copy2(p, src_tml / p.name)

    qkvr_py = Path(vllm_root) / "vllm/models/inkling/nvidia/ops/qkvr_prep.py"
    route_py = Path(vllm_root) / ROUTE_REL
    if patches == "u3+shear":
        _run([_py(), U3_PATCH, vllm_root])
        _run([_py(), SHEAR_PATCH, vllm_root])
        marks = {
            "u3_quantize_kv": (qkvr_py, "quantize_kv"),
            "u2_shear_relshearspec": (qkvr_py, "RelShearSpec"),
            "u2_shear_use_fused_shear": (route_py, "def use_fused_shear"),
            "u2_shear_presheared_iface": (
                Path(tml_pkg) / "interface.py",
                "rel_bias_presheared",
            ),
        }
    elif patches == "route":
        # Exactly what _deploy_build("ours") does for the 8x e2e bench. On
        # sm_90 the day-0 router sends Hopper down the score_mod gather, so
        # without this patch our kernels are never reached and the run
        # measures nothing.
        _run([_py(), ROUTE_PATCH, vllm_root])
        marks = {"u2_serving_route": (route_py, "Inkling-turbo: sm_90")}
    else:
        marks = {}

    if not same_tree:
        for p in ours:
            shutil.copy2(src_tml / p.name, Path(tml_pkg) / p.name)

    # Markers, so "the patch ran" is never assumed. Each string is introduced
    # by the patch it is checked for. Also assert the patches NOT selected left
    # no trace, so a stale warm container cannot quietly widen the config.
    absent = {
        "u3+shear": {},
        "route": {"u3_quantize_kv": (qkvr_py, "quantize_kv"),
                  "u2_shear_relshearspec": (qkvr_py, "RelShearSpec")},
        "none": {"u3_quantize_kv": (qkvr_py, "quantize_kv"),
                 "u2_shear_relshearspec": (qkvr_py, "RelShearSpec"),
                 "u2_serving_route": (route_py, "Inkling-turbo: sm_90")},
    }[patches]
    for key, (path, needle) in absent.items():
        if path.exists() and needle in path.read_text(errors="replace"):
            raise RuntimeError(
                f"patch set {patches!r} must NOT contain {key}, but {needle!r} "
                f"is present in {path}. The stock restore did not take."
            )
        print(f"PATCH_ABSENT {key}=confirmed-absent")

    status = {}
    for key, (path, needle) in marks.items():
        present = path.exists() and needle in path.read_text(errors="replace")
        status[key] = present
        print(f"PATCH_CHECK {key}={present} ({path})")
    missing = [k for k, v in status.items() if not v]
    if missing:
        raise RuntimeError(f"patch verification failed, missing: {missing}")

    _purge_pycache(
        tml_pkg,
        str(Path(vllm_root) / "vllm/models/inkling"),
        str(src_tml) if not same_tree else tml_pkg,
    )
    return {
        "tml_pkg": tml_pkg,
        "tml_source_tree_same": same_tree,
        "deployed": checked,
        "patch_set": patches,
        "patch_order": {
            "u3+shear": ["u3_fp8_kv.py", "u2_shear_fusion.py"],
            "route": ["u2_serving_route.py"],
            "none": [],
        }[patches],
        "patch_markers": status,
        "patches_confirmed_absent": sorted(absent),
    }


def _parse_harness_output(text: str) -> dict:
    """Advisory line counts. rc and the JSON artifact are the real verdict."""
    counts = {
        "lines_ok": len(re.findall(r"(?m)\bOK\s*$", text)),
        "lines_fail": len(re.findall(r"FAIL", text)),
        "lines_skip": len(re.findall(r"SKIP", text)),
    }
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+cases bit-exact", text)
    if m:
        counts["cases_bit_exact"] = f"{m.group(1)}/{m.group(2)}"
    return counts


def _microbench_totals(path: Path) -> dict:
    """Case -> total us/iter, or None where the case errored."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"_unreadable": str(exc)}
    out: dict = {}
    for name, rec in data.items():
        if not isinstance(rec, dict) or "error" in rec:
            out[name] = None
            continue
        v = rec.get("total_us_per_iter")
        out[name] = round(v, 1) if isinstance(v, (int, float)) else None
    return out


def _run_harness(
    name: str,
    argv: list[str],
    env_extra: dict,
    artifacts: list[tuple[str, str]],
    expect: str,
    workdir: str,
    deadline: float,
) -> dict:
    """Run one harness. NEVER raises: a partial result set beats nothing."""
    rec = {
        "step": name,
        "cmd": " ".join(argv),
        "env": env_extra,
        "expect": expect,
        "verdict": "NOT_RUN",
    }
    if time.time() > deadline:
        rec["reason"] = "skipped: out of time or budget"
        print(f"=== SKIP {name}: out of time/budget ===")
        return rec

    work = Path(workdir)
    # Stale artifacts must never be republished under a new name. This is what
    # keeps the fusion-off and fusion-on JSONs from being the same file. The
    # glob covers every capability suffix, not just the requested one, so a
    # leftover sm80 file cannot be picked up by an sm90 run or vice versa.
    for local_name, _remote_name in artifacts:
        for stale in _artifact_candidates(work, local_name):
            stale.unlink()

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in env_extra.items()})
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_local = work / f"{name}.log"
    env_note = " ".join(f"{k}={v}" for k, v in env_extra.items()) or "(none)"
    # The deadline is otherwise only checked BETWEEN steps, so one hung CuTe
    # compile would burn to the 2h container cap. Give each harness whatever is
    # left and no more, so the worst case is the budget, not the timeout.
    budget_s = max(60.0, deadline - time.time())
    print(f"\n=== STEP {name} === env: {env_note}")
    print(f"    cwd={workdir} cmd={' '.join(argv)}")
    print(f"    timeout={budget_s / 60.0:.1f} min (remaining budget)")
    t0 = time.time()
    try:
        with log_local.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(
                argv,
                cwd=workdir,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=budget_s,
            ).returncode
    except subprocess.TimeoutExpired:
        rc = -9
        rec["timed_out_after_s"] = round(budget_s, 1)
        with log_local.open("a", encoding="utf-8") as fh:
            fh.write(f"\nKILLED: exceeded {budget_s:.0f}s remaining budget\n")
        print(f"TIMEOUT: {name} killed after {budget_s:.0f}s")
    except (OSError, ValueError) as exc:
        rc = -1
        log_local.write_text(f"launch failed: {exc!r}\n", encoding="utf-8")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["rc"] = rc
    rec["verdict"] = "PASS" if rc == 0 else "FAIL"

    text = ""
    try:
        text = log_local.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    print(text)
    rec.update(_parse_harness_output(text))
    _write_committed(text, Path(VALIDATE_LOGS) / f"{name}.log")

    saved = []
    for local_name, remote_name in artifacts:
        local = _resolve_artifact(work, local_name)
        if local is None:
            rec["verdict"] = "FAIL"
            rec.setdefault("missing_artifacts", []).append(local_name)
            print(f"ARTIFACT MISSING: {work / local_name}")
            continue
        remote = Path(VALIDATE_ROOT) / remote_name
        _persist(local, remote, f"{name}:{remote_name}")
        saved.append(remote_name)
        if remote_name.endswith(".json"):
            rec.setdefault("totals_us_per_iter", {})[remote_name] = (
                _microbench_totals(local)
            )
    if saved:
        rec["artifacts"] = saved
    print(f"=== STEP {name}: {rec['verdict']} rc={rc} in {rec['seconds']}s ===")
    return rec


@app.function(
    image=bench_image,
    gpu=f"{VALIDATE_GPU_KIND}:{VALIDATE_N_GPU}",
    # Results Volume only. The model Volume is deliberately NOT mounted: no
    # part of this step reads the checkpoint, and leaving it unmounted makes
    # that structurally impossible instead of a promise.
    volumes={RESULTS_MOUNT: results_vol},
    timeout=VALIDATE_TIMEOUT_S,
    cpu=VALIDATE_CPU,
    memory=VALIDATE_MEMORY_MIB,
)
def run_validate(
    budget_usd: float = BUDGET_USD,
    tag: str = VALIDATE_TAG,
    patches: str = "u3+shear",
    gpu_kind: str = "H100",
) -> dict:
    """Parity gates + microbenches on one H100. Needs no model."""
    t0 = time.time()
    rate_hr = _validate_usd_per_hour()
    _print_validate_cost_banner("container")

    prior = _prior_spend("run_validate")
    remaining_usd = budget_usd - prior
    print(f"LEDGER prior spend estimate: ${prior:.2f}, remaining ${remaining_usd:.2f}")
    if remaining_usd < MIN_USEFUL_VALIDATE_USD:
        msg = (
            f"ABORT: only ${remaining_usd:.2f} left under the ${budget_usd:.0f} "
            f"cap, less than the ${MIN_USEFUL_VALIDATE_USD:.0f} needed to run "
            "the gates. Nothing was run."
        )
        print(msg)
        return {"aborted_before_start": True, "reason": msg, "prior_usd": prior}

    hard_deadline = t0 + VALIDATE_TIMEOUT_S - VALIDATE_SHUTDOWN_MARGIN_S
    budget_deadline = t0 + (remaining_usd / rate_hr) * 3600.0
    deadline = min(hard_deadline, budget_deadline)
    print(
        f"effective deadline: {(deadline - t0) / 3600:.2f}h from now "
        f"(hard timeout {(hard_deadline - t0) / 3600:.2f}h, "
        f"budget {(budget_deadline - t0) / 3600:.2f}h)"
    )

    _set_validate_root(tag)
    Path(VALIDATE_ROOT).mkdir(parents=True, exist_ok=True)
    Path(VALIDATE_LOGS).mkdir(parents=True, exist_ok=True)
    print(f"results root for this run: {VALIDATE_ROOT}")

    records: list[dict] = []

    def burn(note: str) -> float:
        hours = (time.time() - t0) / 3600.0
        usd = rate_hr * hours
        total = _ledger_upsert("run_validate", hours, usd, note)
        print(
            f"BURN elapsed {hours:.2f}h | this container ~${usd:.2f} | "
            f"ledger total ~${total:.2f} of ${budget_usd:.0f} | "
            f"{(deadline - time.time()) / 60:.0f} min left",
            flush=True,
        )
        return total

    def flush(note: str) -> None:
        _write_committed(
            json.dumps(
                {
                    "steps": records,
                    "elapsed_hours": round((time.time() - t0) / 3600, 3),
                    "spend_usd_est": round(rate_hr * (time.time() - t0) / 3600, 2),
                    "last": note,
                },
                indent=2,
            ),
            Path(VALIDATE_ROOT) / "progress.json",
            quiet=True,
        )

    burn("start")

    # ---------------------------------------------------------------- 1. env
    print("\n=== STEP env_proof ===")
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(smi.stdout.strip() or smi.stderr.strip())
    probe = subprocess.run(
        [_py(), "-c", _ENV_PROBE_SRC], capture_output=True, text=True, check=False
    )
    print(probe.stdout.strip() or probe.stderr.strip())
    env_info: dict = {}
    if probe.returncode == 0:
        try:
            env_info = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            env_info = {"parse_error": str(exc), "raw": probe.stdout[-2000:]}
    else:
        env_info = {"probe_rc": probe.returncode, "stderr": probe.stderr[-2000:]}

    cap = tuple(env_info.get("capability") or ())
    # Expect the capability that matches the GPU we asked for, not sm_90
    # unconditionally. This step needs no checkpoint, so it is the cheap way to
    # get correctness evidence on an architecture that has none, and sm_80 is
    # exactly that case.
    EXPECT_CAP = {
        "H100": (9, 0),
        "H200": (9, 0),
        "A100-40GB": (8, 0),
        "A100-80GB": (8, 0),
    }
    want = EXPECT_CAP.get(gpu_kind)
    arch_ok = want is not None and cap == want
    env_record = {
        "step": "env_proof",
        "verdict": "PASS" if arch_ok else "FAIL",
        "expect": f"{gpu_kind}, capability {want}",
        "requested_gpu": gpu_kind,
        "capability": list(cap),
        "arch_matches_request": arch_ok,
        "nvidia_smi": smi.stdout.strip(),
        **{k: v for k, v in env_info.items() if k != "capability"},
    }
    records.append(env_record)
    _write_committed(
        json.dumps(env_record, indent=2), Path(VALIDATE_ROOT) / f"env_{tag}.json"
    )
    flush("env_proof")

    if not arch_ok:
        msg = (
            f"ABORT: device capability {cap or 'unknown'} is not sm_90. Every "
            "number this step exists to produce is arch-specific, and "
            "publishing them under the requested heading would be wrong. Nothing "
            "else was run."
        )
        print(msg)
        summary = {
            "aborted": True,
            "reason": msg,
            "steps": records,
            "elapsed_hours": round((time.time() - t0) / 3600, 3),
        }
        _write_committed(
            json.dumps(summary, indent=2), Path(VALIDATE_ROOT) / "summary.json"
        )
        _ledger_upsert(
            "run_validate",
            (time.time() - t0) / 3600.0,
            rate_hr * (time.time() - t0) / 3600.0,
            "aborted: wrong arch",
        )
        return summary

    # -------------------------------------------------------------- 2. setup
    deploy_info: dict = {}
    workdir = ""
    try:
        vllm_root, tml_pkg = _resolve_paths()
        print(f"vllm root: {vllm_root}")
        print(f"resolved tml_fa4 package dir: {tml_pkg}")
        _make_stock_backup(vllm_root, tml_pkg)
        deploy_info = _deploy_validate_build(vllm_root, tml_pkg, patches)
        workdir = _stage_harness()
        setup_rec = {"step": "setup", "verdict": "PASS", **deploy_info}
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb

        setup_rec = {
            "step": "setup",
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": _tb.format_exc(limit=6),
        }
        print("SETUP FAILED:")
        print(setup_rec["traceback"])
    records.append(setup_rec)
    flush("setup")
    burn("setup")

    manifest = {
        "app": APP_NAME,
        "step": "validate",
        "tag": tag,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": f"H100:{VALIDATE_N_GPU}",
        "vllm_pin": VLLM_PIN,
        "vllm_wheel": VLLM_WHEEL,
        "cuda_image": CUDA_IMAGE,
        "model": "NONE. Every harness builds random tensors.",
        "env": env_info,
        "deploy": deploy_info,
        "cost": {
            "usd_per_hour": round(rate_hr, 4),
            "budget_usd": budget_usd,
            "prior_spend_usd_est": prior,
            "hard_timeout_hours": VALIDATE_TIMEOUT_HOURS,
            "rates_quoted": "modal.com/pricing 2026-07-24",
        },
        "caveats": {
            "fused_shear_env_var": (
                "INKLING_TURBO_FUSED_SHEAR does NOT change what "
                "microbench_attn_day0.py measures. That harness calls tml-fa4 "
                "directly with rel_bias=<natural layout>, so ShearingBias "
                "still runs inside the timed region. The env var is read only "
                "by fa4_rel_attention.use_fused_shear, on the vLLM serving "
                "path, and it additionally requires _use_sheared_bias(), "
                "which is false on sm_90 unless u2_serving_route.py is "
                "applied (it is NOT applied here). Expect the fusion_on and "
                "fusion_off JSONs to agree within noise. The measurement that "
                "actually isolates the removed pre-kernel is the presheared_* "
                "section of the generated extra_microbench.py."
            ),
            "qkvr_writer_cost": (
                "MEASURED by the writer_* section of extra_microbench.py. The "
                "fused writer emits rel_extent + 256 columns instead of "
                "rel_extent, so it is strictly more work than the natural "
                "writer and the fusion only pays if that extra cost is "
                "smaller than the ShearingBias launch it removes. The verdict "
                "on any shape is writer_*.writer_delta_us_per_iter against "
                "presheared_*.saved_us_per_iter_vs_natural. Quoting either "
                "section on its own overstates the fusion."
            ),
            "artifact_names": (
                "microbench_attn_day0_*.json holds OUR numbers, not day-0 "
                "ones. The harness times whatever tml_fa4 resolves to, which "
                "here is our build. The actual day-0 baseline is "
                "microbench_attn_scoremod_*.json. The file name is historical "
                "and this repo has already published one wrong headline by "
                "confusing our own code for a baseline."
            ),
            "parity_fa4_rel_verdict": (
                "parity_fa4_rel's PASS/FAIL is rc-based, and the harness "
                "counts every non-tuple result including SKIP as a failure "
                "across all 9 case x backend combinations. A red verdict there "
                "may be the abandoned relproj_v1 prototype rather than the "
                "real tml_fa4_rel_bias gate. Read logs/parity_fa4_rel.log "
                "before concluding anything from it."
            ),
            "split_kv": (
                "sm_90 split-KV has never run on Hopper. Any splitkv_* result "
                "here is its first execution."
            ),
        },
    }
    _write_committed(
        json.dumps(manifest, indent=2), Path(VALIDATE_ROOT) / "manifest.json"
    )

    # -------------------------------------------------------------- 3. steps
    if setup_rec["verdict"] == "PASS":
        py = _py()
        steps = [
            (
                "parity_fa4_rel",
                [py, "parity_fa4_rel.py"],
                {},
                [],
                "3/3 green for tml_fa4_rel_bias on the sm_90 native path "
                "(the harness also reports score_mod and our abandoned "
                "relproj prototype; it exits non-zero if any of the 9 "
                "case x backend checks fails)",
            ),
            (
                # THE gate for the shear-shift fix. Every case here has
                # seqlen_k > seqlen_q, which is the shape family the old
                # 128*(m_block+1) formula got wrong and which no existing gate
                # covered. A green run here is what restores the decode claim.
                "parity_rel_chunked_decode",
                [py, "parity_rel_chunked_decode.py"],
                {},
                [
                    (
                        "parity_rel_chunked_decode_sm90.json",
                        f"parity_rel_chunked_decode_{tag}.json",
                    )
                ],
                "7/7. control_full_prefill must pass either way; the chunked_* "
                "and decode_* cases FAIL before the n_block_max fix and pass "
                "after it. Before the fix, decode got bias on 1 of 512 KV "
                "blocks and it was the oldest one.",
            ),
            (
                "parity_shear_fusion",
                [py, "parity_shear_fusion.py"],
                {"INKLING_TURBO_FUSED_SHEAR": "1"},
                [
                    (
                        "parity_shear_fusion_sm90.json",
                        f"parity_shear_fusion_{tag}.json",
                    )
                ],
                "16/16 now expected. It scored 14/16 on Hopper in session 26 "
                "because both attention_consumes_* cases hit the unbound "
                "n_block, which is fixed.",
            ),
            (
                "parity_kv_fp8",
                [py, "parity_kv_fp8.py"],
                {},
                [],
                "2/2 OK",
            ),
            (
                # This run patches qkvr_prep.py twice. The default, unsheared
                # path has to come out untouched, and nothing else here checks
                # that: every other gate exercises the sheared path.
                "parity_qkvr_prep",
                [py, "parity_qkvr_prep.py"],
                {},
                [],
                "5/5. Regression gate: the DEFAULT qkvr_prep path must behave "
                "identically after u3 and u2 have both rewritten this file.",
            ),
            (
                "microbench_attn_scoremod",
                [py, "microbench_attn_scoremod.py"],
                {},
                [
                    (
                        "microbench_attn_scoremod.json",
                        f"microbench_attn_scoremod_{tag}.json",
                    )
                ],
                "day-0 baseline. score_mod is the ONLY day-0 path; relproj "
                "and relprojT in this JSON are OUR abandoned prototypes.",
            ),
            (
                # Named "ours", not "day0", on purpose. microbench_attn_day0.py
                # times whatever tml_fa4 resolves to, which is our build. The
                # day-0 baseline is the scoremod file above. This repo has
                # already published one wrong headline by mistaking our own
                # code for a baseline; the file name will not invite a second.
                "microbench_ours_attn_shearfusion_OFF",
                [py, "microbench_attn_day0.py"],
                {"INKLING_TURBO_FUSED_SHEAR": "0"},
                [
                    (
                        "microbench_attn_day0.json",
                        f"microbench_ours_attn_shearfusion_OFF_{tag}.json",
                    )
                ],
                "reproduces session 25 (decode b1 64K 852.6, b32 64K 854.8, "
                "b32 8K 124.1, prefill global 3308.8, prefill SWA 1223.0 "
                "us/iter on one H100 SXM5)",
            ),
            (
                "microbench_ours_attn_shearfusion_ON",
                [py, "microbench_attn_day0.py"],
                {"INKLING_TURBO_FUSED_SHEAR": "1"},
                [
                    (
                        "microbench_attn_day0.json",
                        f"microbench_ours_attn_shearfusion_ON_{tag}.json",
                    )
                ],
                "run exactly as asked. READ manifest.json caveats first: this "
                "harness passes rel_bias=<natural> straight to tml-fa4, so "
                "the env var cannot reach it and this SHOULD equal the OFF "
                "run within noise. If it does not, something else moved.",
            ),
            (
                "microbench_presheared_and_splitkv",
                [py, "extra_microbench.py"],
                {},
                [
                    (
                        "extra_microbench.json",
                        f"microbench_presheared_splitkv_{tag}.json",
                    )
                ],
                "presheared_*: attention timed with the ShearingBias "
                "pre-kernel already done, via the bias= argument, "
                "parity-checked before any timing is reported. splitkv_*: "
                "num_splits > 1 on sm_90, compared against num_splits=1.",
            ),
        ]

        # The shear gates read RelShearSpec out of a patched qkvr_prep, and
        # parity_kv_fp8 exercises u3's quantize_kv. Under any other patch set
        # they would fail on an ImportError that says nothing, so they are
        # dropped rather than run and reported red.
        NEEDS_SHEAR = {
            "parity_shear_fusion",
            "parity_kv_fp8",
            "microbench_presheared_and_splitkv",
            "microbench_ours_attn_shearfusion_ON",
        }
        if patches != "u3+shear":
            dropped = [s[0] for s in steps if s[0] in NEEDS_SHEAR]
            steps = [s for s in steps if s[0] not in NEEDS_SHEAR]
            print(
                f"patch set {patches!r}: dropping "
                f"{len(dropped)} shear/u3-dependent steps: {dropped}"
            )

        # Architecture-specific extras. These gates are arch-agnostic Python but
        # they exercise code paths that only exist per arch, and on an A100 they
        # are the entire point of the run. Expectations updated after session 31
        # (journal/remote/validate_a100x1_s31), which is what they are measured
        # against now.
        EXTRA_BY_ARCH = {
            "A100-40GB": [
                ("parity_rel_varlen_batch", "parity_rel_varlen_batch.py",
                 "12/12 expected. Session 31 scored 11/12 on sm_80, failing "
                 "single_m_tail_chunked at mean 1.9666e-03, which is the "
                 "pack_gqa shear-granularity defect. The guard for it is "
                 "verified 12/12 on sm_120 and this is its first execution on "
                 "sm_80. An 11/12 here means the guard does not cover Ampere."),
                ("parity_rel_bias_coverage", "parity_rel_bias_coverage.py",
                 "6/6, unchanged from session 31. Its probe, not its oracle, "
                 "is the discriminating check at decode depth."),
                ("tune_sm80", "tune_sm80.py",
                 "the tile sweep, now with interleaved repeat rounds and a "
                 "disjoint-interval rule. It will print NO WINNER on any shape "
                 "where run-to-run spread exceeds the gap between configs, "
                 "which session 31's two-run comparison says is most of them. "
                 "That verdict is the result; a named winner is the surprise. "
                 "Its parity geometry now comes from CASES, so this is also "
                 "the first Ampere run of a bias gate at Hq=64 over Hkv=8."),
            ],
        }
        for name, script, expect in EXTRA_BY_ARCH.get(gpu_kind, []):
            if any(s[0] == name for s in steps):
                continue
            cc = "sm80" if gpu_kind.startswith("A100") else "sm90"
            stem = script.replace(".py", "")
            steps.append((
                name, [py, script], {},
                [(f"{stem}_{cc}.json", f"{stem}_{tag}.json")],
                expect,
            ))
            print(f"{gpu_kind}: added step {name}")

        for name, argv, env_extra, artifacts, expect in steps:
            rec = _run_harness(
                name, argv, env_extra, artifacts, expect, workdir, deadline
            )
            records.append(rec)
            flush(name)
            burn(f"after {name}")
    else:
        print("setup failed, no harness was run")

    # ------------------------------------------------------------ 4. summary
    elapsed_h = (time.time() - t0) / 3600.0
    total = _ledger_upsert("run_validate", elapsed_h, rate_hr * elapsed_h, "final")
    ran = [r for r in records if r.get("verdict") != "NOT_RUN"]
    summary = {
        "tag": tag,
        "steps": records,
        "passed": [r["step"] for r in ran if r.get("verdict") == "PASS"],
        "failed": [r["step"] for r in ran if r.get("verdict") == "FAIL"],
        "not_run": [r["step"] for r in records if r.get("verdict") == "NOT_RUN"],
        "elapsed_hours": round(elapsed_h, 3),
        "spend_usd_est": round(rate_hr * elapsed_h, 2),
        "ledger_total_usd_est": total,
        "budget_usd": budget_usd,
        "results_root": VALIDATE_ROOT,
        "caveats": manifest["caveats"],
    }
    _write_committed(
        json.dumps(summary, indent=2), Path(VALIDATE_ROOT) / "summary.json"
    )

    print("\n=== VALIDATE SUMMARY ===")
    print(f"{'step':46s} {'verdict':8s} {'rc':>4s} {'secs':>8s}  detail")
    for r in records:
        detail = []
        if "cases_bit_exact" in r:
            detail.append(r["cases_bit_exact"] + " bit-exact")
        if "lines_ok" in r:
            detail.append(
                f"ok={r['lines_ok']} fail={r['lines_fail']} skip={r['lines_skip']}"
            )
        if r.get("missing_artifacts"):
            detail.append("MISSING " + ",".join(r["missing_artifacts"]))
        if r.get("reason"):
            detail.append(r["reason"])
        if r.get("error"):
            detail.append(r["error"])
        print(
            f"{r['step']:46s} {str(r.get('verdict')):8s} "
            f"{str(r.get('rc', '-')):>4s} {str(r.get('seconds', '-')):>8s}  "
            + "; ".join(detail)
        )
    print("\nmeasured totals (us/iter), per artifact:")
    for r in records:
        for artifact, totals in (r.get("totals_us_per_iter") or {}).items():
            print(f"  {artifact}")
            for case, value in totals.items():
                print(f"    {case:44s} {value}")
    print("\ncaveats:")
    for key, text in summary["caveats"].items():
        print(f"  [{key}] {text}")
    print(f"\nspend this container ~${summary['spend_usd_est']:.2f}, "
          f"ledger total ~${total:.2f} of ${budget_usd:.0f}")
    print(f"artifacts: {VALIDATE_ROOT} (committed after every step)")
    return summary


# --------------------------------------------------------------------------
# Local entrypoint. Defaults come from env vars so the matrix can be
# trimmed without editing this file; explicit flags override env.
# --------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    step: str = "all",
    runs: int = int(os.environ.get("RUNS", DEFAULT_RUNS)),
    builds: str = os.environ.get("BUILDS", DEFAULT_BUILDS),
    concurrencies: str = os.environ.get("CONCURRENCIES", DEFAULT_CONCURRENCIES),
    mixes: str = os.environ.get("MIXES", DEFAULT_MIXES),
    budget_usd: float = float(os.environ.get("BUDGET_USD", BUDGET_USD)),
    patches: str = os.environ.get("VALIDATE_PATCHES", "u3+shear"),
    server_wait_s: int = int(os.environ.get("SERVER_WAIT_S", SERVER_WAIT_S)),
    force_download: int = 0,
) -> None:
    """modal run --detach scripts/modal_e2e_bench.py --step all"""
    if step not in ("all", "download", "bench", "validate"):
        raise SystemExit(
            f"unknown step {step!r}; use all|download|bench|validate"
        )
    if patches not in ("u3+shear", "route", "none"):
        raise SystemExit(
            f"unknown --patches {patches!r}; use u3+shear|route|none"
        )

    # validate is independent of the matrix and of the model, so it returns
    # before any of the bench-matrix parsing. It is deliberately NOT part of
    # step=all: it is a separate, cheap decision.
    if step == "validate":
        _print_validate_cost_banner("local")
        print(f"budget cap in effect: ${budget_usd:.2f} (ledger-enforced)")
        print(
            "artifacts are committed to the inkling-bench-results Volume "
            "after EVERY step; a killed container costs at most one harness."
        )
        print()
        # Not "1x H100": this step targets any arch now, and a banner that names
        # the wrong silicon is the same trap as the b200_first_contact logs that
        # are all H100 sessions.
        print(f">>> step: validate ({VALIDATE_N_GPU}x {VALIDATE_GPU_KIND}, "
              f"no model), patches={patches!r}")
        if patches == "route":
            print(
                "    'route' is EXACTLY what the 8x e2e bench deploys for its "
                "'ours' build, so this run is the cheap health check that says "
                "whether the expensive one is worth starting."
            )
        # The tag names the hardware, because these artifacts get copied into
        # journal/remote/ and read months later. A file called
        # modal_h100x1_route holding sm_80 numbers is the same class of trap as
        # the b200_first_contact logs that are all H100 sessions.
        base = f"modal_{VALIDATE_GPU_KIND.lower().replace('-', '')}x{VALIDATE_N_GPU}"
        tag = base if patches == "u3+shear" else (
            f"{base}_{patches.replace('+', '')}"
        )
        # Lets a deliberately-broken control run land beside the real one
        # instead of overwriting it. Used to prove a new gate actually fails
        # against the defect it was written for, which is the difference
        # between having a gate and believing you have one.
        suffix = os.environ.get("VALIDATE_TAG_SUFFIX", "").strip()
        if suffix:
            tag = f"{tag}_{re.sub(r'[^A-Za-z0-9_.-]', '', suffix)}"
            print(f"    tag suffix in effect: results land under {tag}")
        out = run_validate.remote(
            budget_usd=budget_usd, tag=tag, patches=patches,
            gpu_kind=VALIDATE_GPU_KIND,
        )
        print(json.dumps(out, indent=2))
        print()
        print("fetch results:")
        print(
            "  modal volume get inkling-bench-results /validate "
            "./validate_results"
        )
        return

    mix_list = _parse_mixes(mixes)
    conc_list = _parse_concurrencies(concurrencies)
    configs = _ordered_configs(mix_list, conc_list)
    build_list = [b.strip() for b in builds.split(",") if b.strip()]

    _print_cost_banner("local", runs, len(configs), len(build_list))
    rate_hr = _bench_usd_per_hour()
    print(
        f"mixes: {[(n, i, o) for n, i, o in mix_list]} | "
        f"concurrencies: {conc_list} | builds: {build_list}"
    )
    print(
        "cheapest-first order: "
        + ", ".join(f"{n}/c{c}(np{p})" for n, c, _i, _o, p in configs)
    )
    print(
        "measured-shape estimate: 4-8h of GPU wall clock for the full "
        f"matrix including two model loads, i.e. ${rate_hr * 4:.0f} to "
        f"${rate_hr * 8:.0f}. The 4h hard timeout caps ONE container at "
        f"${rate_hr * BENCH_TIMEOUT_HOURS:.0f}, so the full matrix may need "
        "a resume. This is an ESTIMATE, not a measurement."
    )
    print(f"budget cap in effect: ${budget_usd:.2f} (ledger-enforced)")
    print(
        "results are committed to the inkling-bench-results Volume after "
        "EVERY run; a killed container costs at most one configuration."
    )
    print()

    if step in ("all", "download"):
        print(">>> step: download (CPU only)")
        print(download_model.remote(force=force_download))

    if step in ("all", "bench"):
        print(f">>> step: bench ({N_GPU}x {BENCH_GPU_KIND})")
        out = run_bench.remote(
            runs=runs,
            builds=builds,
            concurrencies=concurrencies,
            mixes=mixes,
            budget_usd=budget_usd,
            server_wait_s=server_wait_s,
        )
        print(json.dumps(out, indent=2))
        print()
        print("fetch results:")
        print("  modal volume get inkling-bench-results /bench ./bench_results")
        print("then summarize with scripts/gate_summarize.py --root ./bench_results")
