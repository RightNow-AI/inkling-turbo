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

LAUNCH
  modal run --detach scripts/modal_e2e_bench.py --step all
  # trim the matrix if close to the cap (env or flags, flags win):
  RUNS=3 CONCURRENCIES="1 8 32" modal run --detach scripts/modal_e2e_bench.py --step bench
  # resume an interrupted matrix (existing run JSONs are skipped):
  modal run --detach scripts/modal_e2e_bench.py --step bench
  # pull results down:
  modal volume get inkling-bench-results /bench ./bench_results

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

import hashlib
import json
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
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

MODEL_MOUNT = "/models"
MODEL_DIR = f"{MODEL_MOUNT}/inkling"
MODEL_MARKER = f"{MODEL_DIR}/.download_complete"
RESULTS_MOUNT = "/results"
RESULTS_ROOT = f"{RESULTS_MOUNT}/bench"
LOGS_ROOT = f"{RESULTS_MOUNT}/logs"
LEDGER_PATH = f"{RESULTS_MOUNT}/spend_ledger.json"

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
# 4h x ~$34.4/hr = ~$138 worst case for one bench container.
BENCH_TIMEOUT_HOURS = 4.0
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


def _usd_per_hour(n_gpu: float, cpu: float, mem_mib: float) -> float:
    per_sec = (
        n_gpu * USD_PER_H100_SEC
        + cpu * USD_PER_CPU_CORE_SEC
        + (mem_mib / 1024.0) * USD_PER_GIB_MEM_SEC
    )
    return per_sec * 3600.0


def _bench_usd_per_hour() -> float:
    return _usd_per_hour(N_GPU, BENCH_CPU, BENCH_MEMORY_MIB)


def _download_usd_per_hour() -> float:
    return _usd_per_hour(0.0, DL_CPU, DL_MEMORY_MIB)


def _print_cost_banner(where: str, runs: int, n_configs: int, n_builds: int) -> None:
    gpu_hr = N_GPU * USD_PER_H100_SEC * 3600.0
    rate = _bench_usd_per_hour()
    total_runs = n_builds * n_configs * runs
    print(f"=== COST ESTIMATE ({where}) ===")
    print(
        f"H100 ${USD_PER_H100_SEC * 3600:.4f}/GPU-hr x {N_GPU} GPUs "
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
        return
    stock.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(tml_pkg).glob("*.py")):
        shutil.copy2(p, stock / p.name)
        n += 1
    shutil.copy2(Path(vllm_root) / ROUTE_REL, STOCK_ROUTE)
    print(f"stock backup created: {stock} ({n} files) + {STOCK_ROUTE}")


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
    gpu=f"H100:{N_GPU}",
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
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
    )
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
    scratch = Path("/tmp/bench_run")
    scratch.mkdir(parents=True, exist_ok=True)
    # Conservative reserve until a run has actually been timed.
    run_reserve_s = 1800.0

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

    for build in wanted:
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
            break

        print(f"=== BUILD {build}: {len(pending)} runs pending ===")
        _deploy_build(build, vllm_root, tml_pkg)

        log_path = Path(f"/tmp/serve_{build}.log")
        proc = None
        try:
            proc = _start_server(build, log_path)
            _wait_healthy(proc, log_path, server_wait_s)
            _persist_log_tail(log_path, Path(LOGS_ROOT) / f"serve_{build}_startup.log")
            burn(f"{build} server ready")

            for mix, conc, ilen, olen, npr, n, out in pending:
                if time.time() + run_reserve_s > deadline:
                    print(
                        f"=== STOP mid-build {build}: "
                        f"{(deadline - time.time()) / 60:.0f} min left, "
                        f"reserve {run_reserve_s / 60:.0f} min ==="
                    )
                    aborted = True
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
                rc = subprocess.run(cmd, env=_server_env(), check=False).returncode
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

        if aborted:
            break

    elapsed_h = (time.time() - t0) / 3600.0
    total = _ledger_upsert("run_bench", elapsed_h, rate_hr * elapsed_h, "final")
    summary = {
        "completed_runs": completed,
        "skipped_existing": skipped,
        "failed_runs": failed,
        "aborted_early": aborted,
        "elapsed_hours": round(elapsed_h, 3),
        "spend_usd_est": round(rate_hr * elapsed_h, 2),
        "ledger_total_usd_est": total,
        "budget_usd": budget_usd,
        "results_root": RESULTS_ROOT,
    }
    _write_committed(json.dumps(summary, indent=2), Path(RESULTS_ROOT) / "summary.json")
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
    force_download: int = 0,
) -> None:
    """modal run --detach scripts/modal_e2e_bench.py --step all"""
    if step not in ("all", "download", "bench"):
        raise SystemExit(f"unknown step {step!r}; use all|download|bench")

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
        print(">>> step: bench (8x H100)")
        out = run_bench.remote(
            runs=runs,
            builds=builds,
            concurrencies=concurrencies,
            mixes=mixes,
            budget_usd=budget_usd,
        )
        print(json.dumps(out, indent=2))
        print()
        print("fetch results:")
        print("  modal volume get inkling-bench-results /bench ./bench_results")
        print("then summarize with scripts/gate_summarize.py --root ./bench_results")
