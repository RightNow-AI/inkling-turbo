#!/usr/bin/env python3
"""Gate: logit parity, stock day-0 build vs inkling-turbo tml_fa4 kernels.

Runs ON the 8x H100 Lambda box (after scripts/bootstrap_8x.sh). Serves
thinkingmachines/Inkling (NVFP4 checkpoint -> W4A16 dequant path on Hopper,
TP=8) twice with `vllm serve`:

  1. STOCK : vllm.third_party.tml_fa4 restored from ~/tml_fa4_backup
  2. OURS  : ~/tml_fa4_modified/*.py copied into the resolved package dir

and compares per-token logprobs through the OpenAI-compatible completions
API using echo=True + logprobs on 32 fixed prompts (bundled below, varied
lengths, one >600 tokens). Also runs a batched-vs-batch-1 consistency check
on 4 prompts per build (gate: batched output == batch-1 output).

API mechanism, verified against the pinned fork @850295881
($REPO/vllm):
  - `echo: bool | None = False` and `logprobs: int | None = None` are
    CompletionRequest fields: vllm/entrypoints/openai/completion/protocol.py
    lines 59 and 62.
  - With echo=True and prompt_logprobs unset, prompt_logprobs defaults to
    the `logprobs` value: protocol.py lines 327-329.
  - echo=True with max_tokens=0 is the supported "echo without generation"
    path: protocol.py line 331; the non-streaming response then returns
    prompt-only token logprobs: completion/serving.py lines 534-537.
  - Response shape choices[0].logprobs.{tokens, token_logprobs}:
    completion/serving.py lines 733-737. token_logprobs[0] is None (no
    logprob for the first prompt token), and values are clamped to
    >= -9999.0 (serving.py line 703).
  - GET /health readiness endpoint:
    vllm/entrypoints/serve/instrumentator/health.py line 22.
  - `vllm serve` flags used below, verified in vllm/engine/arg_utils.py:
    --tensor-parallel-size (line 1014), --max-model-len (line 829),
    --gpu-memory-utilization (line 1163), --seed (line 816),
    --served-model-name (line 859); positional model_tag and --port:
    vllm/entrypoints/openai/cli_args.py lines 346-351 and 229.

Output: prints one "GATE <name>: PASS|FAIL" line per check and writes JSON
results to ~/gate_logit_parity.json (measured-or-null: a failed build leaves
its fields null with an actionable last_error, never fabricated numbers).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
VENV_BIN = HOME / "vllm" / ".venv" / "bin"
VENV_PY = VENV_BIN / "python"
VLLM_BIN = VENV_BIN / "vllm"
MODEL_DIR = HOME / "models" / "inkling"
MODIFIED_DIR = HOME / "tml_fa4_modified"
BACKUP_DIR = HOME / "tml_fa4_backup"
ROUTE_PATCH = HOME / "u2_serving_route.py"
ROUTE_BACKUP = HOME / "fa4_rel_attention.stock.py"
RESULT_PATH = HOME / "gate_logit_parity.json"
LOG_DIR = HOME / "gate_logs"

SERVED_NAME = "inkling"
PORT = int(os.environ.get("GATE_PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"
TP = 8
# 16384 covers every bundled prompt (longest is ~1k tokens) while keeping the
# KV budget small: the NVFP4 checkpoint is ~592GB against 8x80GB = 640GB HBM,
# so KV headroom on this box is thin (journal/phase0.md, remote tier notes).
MAX_MODEL_LEN = int(os.environ.get("GATE_MAX_MODEL_LEN", "16384"))
# 975B weights loading over 8 ranks can take a long time from local NVMe.
SERVER_WAIT_S = int(os.environ.get("GATE_SERVER_WAIT_S", "5400"))
REQUEST_TIMEOUT_S = 900

# ---------------------------------------------------------------------------
# TOLERANCE DERIVATION (do not tighten or loosen without re-deriving):
#
# The two builds differ ONLY in the tml_fa4 attention kernel files. Weights,
# the W4A16 dequant path (FP4 E2M1 weights, FP8 E4M3 group-16 block scales
# dequantized to BF16; journal/day0-implementation.md "NVFP4 layout"), the
# scheduler, and sampling are identical, so W4A16 itself contributes ZERO
# cross-build difference. All cross-build divergence comes from attention
# floating-point reduction reorder (split-KV policy, tile traversal, bias
# applied in registers vs a materialized bf16 bias tensor).
#
# 1) Per-op bound, measured: the house parity harness gates the modified
#    kernel at 2e-2 max abs on bf16 attention outputs
#    (harness/parity_fa4_rel.py, --tol default 2e-2) and the kernel measured
#    7.8e-3 max abs on H100 and 1.6e-2 on sm_120 vs the oracle, the same
#    error class as the production score_mod route
#    (journal/u2-hopper-design.md, "PARITY 3/3 GREEN ON H100").
# 2) Propagation: 66 layers (journal/phase0.md). Each block RMS-normalizes
#    its input, so a per-layer attention perturbation of ~1e-2 relative does
#    not compound multiplicatively; independent per-layer rounding
#    differences accumulate like a random walk, sqrt(66) ~= 8.1x, giving a
#    final-hidden relative perturbation on the order of a few e-2.
# 3) Logit map: log-softmax is 2-Lipschitz wrt logits in the infinity norm
#    (|d logp_i| <= 2 * ||d logits||_inf). With final RMSNorm + unembed and
#    mup_width_multiplier 24 (journal/phase0.md) typical logit magnitudes
#    are O(10), so a few e-2 relative hidden-state error maps to roughly
#    <= 0.1 abs on individual logits and <= ~0.2 abs on tail logprobs.
# 4) W4A16's indirect role: 4-bit expert weights lower the margin between
#    top-6 router candidates, so an attention-level perturbation can flip
#    expert routing on a near-tie token and produce an isolated outlier.
#    Mean and p99.9 are therefore reported alongside max to make outliers
#    visible instead of hiding them inside a looser max.
# 5) Calibration cross-check, built in: the batched-vs-batch-1 check on the
#    STOCK build measures the identical error class (reduction reorder only,
#    same kernels) end to end; the cross-build gate must sit above that
#    measured floor. A genuinely broken kernel (canonical defect: rel_bias
#    silently dropped, journal/upstream/01-rel-bias-silently-ignored-...)
#    produces O(1..10) logprob errors on >600-token prompts, more than an
#    order of magnitude beyond this gate.
#
# GATE: per-token abs logprob diff, over all 32 prompts:
TOL_MAX = 0.25   # max over every compared token position
TOL_MEAN = 0.02  # mean over every compared token position
# ---------------------------------------------------------------------------

# --- 32 fixed prompts: varied lengths, deterministic, ASCII only. ----------
_PARA = (
    "The scheduler assigns each incoming sequence to a block table, tracks "
    "block reuse across preemptions, and reclaims blocks when a sequence "
    "finishes or is swapped out. Continuous batching admits new sequences "
    "at every step, so the effective batch composition changes between "
    "decode iterations, and paged attention must follow the block table "
    "rather than assume contiguous keys and values in device memory. "
)

# GPT-style byte-BPE tokenizers do not merge across whitespace, so token
# count >= whitespace-delimited word count. 16 sections x ~55 words > 850
# words, which lower-bounds the prompt at well over 600 tokens.
_LONG_PROMPT = "".join(f"Section {i}. {_PARA}" for i in range(16))
assert len(_LONG_PROMPT.split()) > 700

PROMPTS = [
    # short (1-10 tokens)
    "The capital of France is",
    "2 + 2 =",
    "def fibonacci(n):",
    "Once upon a time,",
    "The chemical symbol for gold is",
    "Roses are red,",
    "SELECT name FROM users WHERE",
    "E = mc",
    # medium-short (10-40 tokens)
    "Explain the difference between a process and a thread in one sentence.",
    "Translate to French: The weather is beautiful today and we should walk.",
    "List three prime numbers greater than one hundred and justify briefly.",
    "Write a haiku about graphics processors humming in a cold datacenter.",
    "The mitochondria is the powerhouse of the cell because it produces",
    "In Rust, the borrow checker enforces that a mutable reference is",
    "Summarize the plot of Hamlet in exactly two short sentences, please.",
    "A tensor of shape (64, 128, 512) contracted over its last axis with",
    # medium (40-150 tokens)
    "You are given an array of integers and a target sum. Describe an "
    "algorithm that finds whether any two distinct elements add up to the "
    "target, state its time and space complexity, and mention one common "
    "pitfall in its implementation.",
    "The relative attention mechanism replaces rotary embeddings with a "
    "learned projection of pairwise distance terms that are mixed into the "
    "attention logits before softmax, which changes how the kernel must "
    "stage its operands compared to a standard flash attention pipeline.",
    "Dear hiring committee, I am writing to express my interest in the "
    "systems engineering position. My background covers CUDA kernel "
    "development, distributed inference serving, and performance profiling "
    "under production constraints. In my previous role I",
    "Consider a bank that processes wire transfers in batches every hour. "
    "Each transfer has an amount, a source account, and a destination "
    "account. Design a reconciliation procedure that detects duplicated "
    "or dropped transfers and explain the invariants it relies on.",
    "El aprendizaje profundo ha transformado el procesamiento del lenguaje "
    "natural en la ultima decada. Los modelos de gran escala requieren "
    "infraestructura especializada para servir peticiones con baja "
    "latencia, y por eso los nucleos de computo deben",
    "In thermodynamics, entropy is often described informally as disorder, "
    "but a more precise statement involves the number of microstates "
    "consistent with a macrostate. Using that framing, explain why heat "
    "flows from hot to cold objects spontaneously.",
    "A sliding window attention layer with window size 512 only attends to "
    "the previous 512 positions, while a global layer attends to the full "
    "prefix. When these two layer types alternate in one decoder stack, "
    "the KV cache manager must",
    "Here is a bug report: the service returns HTTP 500 for roughly one "
    "request in ten thousand, only under high concurrency, and only when "
    "the response would exceed one megabyte. Propose three plausible root "
    "causes and one experiment to discriminate among them.",
    # longer structured prompts (150+ tokens)
    "Step 1: read the configuration file and validate every field against "
    "the schema. Step 2: open a connection pool with at most sixteen "
    "connections and exponential backoff. Step 3: for each incoming job, "
    "check the deduplication cache before enqueueing. Step 4: emit a "
    "structured log line with the job identifier, queue depth, and wall "
    "clock latency. Step 5: on shutdown, drain the queue, flush metrics, "
    "and release the pool. Now rewrite these steps as a numbered checklist "
    "for an operations runbook, keeping every technical detail intact and "
    "adding a rollback note after each step.",
    "The following is a transcript of a code review. Reviewer: this "
    "function allocates inside the hot loop; can we hoist the buffer? "
    "Author: the buffer size depends on the batch, but we can reserve the "
    "maximum. Reviewer: also the mutex is held across the network call, "
    "which will serialize all workers. Author: agreed, I will move the "
    "send outside the critical section. Reviewer: finally, the retry "
    "policy retries on every error class including permanent failures. "
    "Author: I will restrict it to transient classes. Continue the "
    "transcript with two more findings about error handling and testing.",
    _PARA + "Given that description, explain how a prefix cache interacts "
    "with paged blocks when two requests share a long common prefix, and "
    "what invalidation rule keeps the shared blocks correct when one "
    "request is preempted midway through decode.",
    "Chapter 1. " + _PARA + "Chapter 2. " + _PARA + "Now write Chapter 3 "
    "in the same style, focusing on how speculative decoding changes the "
    "accounting of accepted and rejected draft tokens.",
    # the >600-token prompt (see word-count assertion above)
    _LONG_PROMPT + "Question: summarize the recurring theme of these "
    "sections in one paragraph.",
    # tails to reach 32, mixed content
    "0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181",
    "The quick brown fox jumps over the lazy dog. " * 8,
    "Q: What is the tallest mountain on Earth?\nA: Mount Everest.\nQ: What "
    "is the deepest point in the ocean?\nA:",
]
assert len(PROMPTS) == 32, f"expected 32 prompts, got {len(PROMPTS)}"

BATCH_PROMPTS = [PROMPTS[8], PROMPTS[16], PROMPTS[20], PROMPTS[24]]
BATCH_MAX_TOKENS = 32


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    print(f"[{stamp()}] {msg}", flush=True)


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def resolve_tml_pkg() -> Path:
    """Resolve the imported tml_fa4 package dir with the venv interpreter.

    Mirrors scripts/bootstrap_b200.sh: a precompiled install may import
    tml_fa4 from site-packages, not the source tree, so the deploy target
    must be the RESOLVED path.
    """
    r = run([str(VENV_PY), "-c",
             "import vllm.third_party.tml_fa4 as m, os;"
             "print(os.path.dirname(m.__file__))"])
    if r.returncode != 0:
        raise RuntimeError(f"cannot resolve tml_fa4 package dir: {r.stderr.strip()}")
    return Path(r.stdout.strip())


def ensure_backup(pkg: Path) -> None:
    """Create ~/tml_fa4_backup from the CURRENT package, once.

    Refuses to create a backup if the package already contains the modified
    kernels (a backup taken then would capture non-stock files and poison
    every later restore).
    """
    if BACKUP_DIR.is_dir() and any(BACKUP_DIR.glob("*.py")):
        return
    for f in sorted(MODIFIED_DIR.glob("*.py")):
        cur = pkg / f.name
        if cur.exists() and cur.read_bytes() == f.read_bytes():
            raise RuntimeError(
                f"no backup dir and {cur} is byte-identical to the modified "
                f"kernel {f}; the package is not stock. Re-run "
                f"scripts/bootstrap_8x.sh to rebuild a clean backup.")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(pkg.glob("*.py")):
        shutil.copy2(f, BACKUP_DIR / f.name)
        n += 1
    log(f"backup created: {n} files -> {BACKUP_DIR}")


def resolve_route_file() -> Path:
    """Resolve the serving router module (fa4_rel_attention) like the pkg."""
    r = run([str(VENV_PY), "-c",
             "import vllm.models.inkling.nvidia.ops.fa4_rel_attention as m;"
             "print(m.__file__)"])
    if r.returncode != 0:
        raise RuntimeError(
            f"cannot resolve fa4_rel_attention: {r.stderr.strip()}")
    return Path(r.stdout.strip())


def ensure_route_backup(route_file: Path) -> None:
    """Snapshot the STOCK serving router once; refuse if already patched."""
    if ROUTE_BACKUP.exists():
        return
    if "Inkling-turbo: sm_90" in route_file.read_text():
        raise RuntimeError(
            f"{route_file} is already patched and no stock backup exists; "
            f"re-run scripts/bootstrap_8x.sh for a clean tree.")
    shutil.copy2(route_file, ROUTE_BACKUP)
    log(f"route backup created: {ROUTE_BACKUP}")


def deploy(build: str, pkg: Path) -> list[str]:
    """Put the package AND the serving router into the requested state."""
    src = BACKUP_DIR if build == "stock" else MODIFIED_DIR
    files = sorted(src.glob("*.py"))
    if not files:
        raise RuntimeError(f"no .py files in {src}; cannot deploy '{build}'")
    for f in files:
        shutil.copy2(f, pkg / f.name)
    route_file = resolve_route_file()
    ensure_route_backup(route_file)
    if build == "stock":
        shutil.copy2(ROUTE_BACKUP, route_file)
        log("serving router restored to stock (score_mod on sm_90)")
    else:
        vllm_root = route_file.parents[5]
        r = run([str(VENV_PY), str(ROUTE_PATCH), str(vllm_root)])
        if r.returncode != 0:
            raise RuntimeError(f"route patch failed: {r.stderr.strip()}")
        log(f"serving router patched: {r.stdout.strip()}")
    log(f"deployed {len(files)} files from {src} -> {pkg} ({build})")
    return [f.name for f in files]


def http_json(path: str, payload: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(
        BASE + path,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode() if payload is not None else None,
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
        return r.status, (json.loads(body) if body else None)


def start_server(build: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(LOG_DIR / f"serve_{build}.log", "w", encoding="utf-8")
    cmd = [
        str(VLLM_BIN), "serve", str(MODEL_DIR),
        "--served-model-name", SERVED_NAME,
        "--tensor-parallel-size", str(TP),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.95",
        "--seed", "0",
        "--port", str(PORT),
    ]
    log(f"starting server ({build}): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    deadline = time.monotonic() + SERVER_WAIT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = (LOG_DIR / f"serve_{build}.log").read_text(
                encoding="utf-8", errors="replace").splitlines()[-30:]
            raise RuntimeError(
                f"server ({build}) died rc={proc.returncode}:\n" + "\n".join(tail))
        try:
            status, _ = http_json("/health")
            if status == 200:
                log(f"server ({build}) healthy")
                return proc
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(10)
    proc.kill()
    raise TimeoutError(f"server ({build}) not healthy within {SERVER_WAIT_S}s")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    log("stopping server")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=60)
    # let TP8 workers and NCCL release GPU memory before the next launch
    time.sleep(30)


def completion_logprobs(prompt: str, max_tokens: int) -> dict:
    """One /v1/completions call; returns {tokens, token_logprobs}."""
    payload = {
        "model": SERVED_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,   # 0 = echo-only prompt logprobs
        "echo": True,               # protocol.py line 59
        "logprobs": 1,              # protocol.py line 62 -> prompt_logprobs
        "temperature": 0.0,
    }
    status, body = http_json("/v1/completions", payload,
                             timeout=REQUEST_TIMEOUT_S)
    if status != 200:
        raise RuntimeError(f"completions HTTP {status}: {body}")
    lp = body["choices"][0]["logprobs"]
    return {"tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"]}


def collect_parity(build: str) -> list[dict]:
    out = []
    for i, p in enumerate(PROMPTS):
        r = completion_logprobs(p, max_tokens=0)
        out.append(r)
        log(f"  [{build}] prompt {i + 1}/32: {len(r['tokens'])} tokens")
    return out


def collect_batch(build: str) -> dict:
    """4 prompts concurrently, then one at a time. Greedy, 32 new tokens."""
    log(f"  [{build}] batched: {len(BATCH_PROMPTS)} concurrent requests")
    with ThreadPoolExecutor(max_workers=len(BATCH_PROMPTS)) as ex:
        batched = list(ex.map(
            lambda p: completion_logprobs(p, BATCH_MAX_TOKENS), BATCH_PROMPTS))
    log(f"  [{build}] batch-1: same prompts sequentially")
    single = [completion_logprobs(p, BATCH_MAX_TOKENS) for p in BATCH_PROMPTS]
    return {"batched": batched, "single": single}


def diff_stats(a: dict, b: dict) -> dict:
    """Compare two {tokens, token_logprobs} records position-wise.

    Skips positions where either logprob is None (first prompt token).
    tokens_match is exact string-list equality; a mismatch in the echoed
    prompt region is impossible with one tokenizer, so a mismatch means the
    greedy continuations diverged.
    """
    tokens_match = a["tokens"] == b["tokens"]
    n = min(len(a["token_logprobs"]), len(b["token_logprobs"]))
    diffs = [abs(x - y)
             for x, y in zip(a["token_logprobs"][:n], b["token_logprobs"][:n])
             if x is not None and y is not None]
    if not diffs:
        return {"tokens_match": tokens_match, "n": 0,
                "max": None, "mean": None}
    return {"tokens_match": tokens_match, "n": len(diffs),
            "max": max(diffs), "mean": sum(diffs) / len(diffs)}


def aggregate(per_prompt: list[dict]) -> dict:
    maxes = [s["max"] for s in per_prompt if s["max"] is not None]
    all_n = sum(s["n"] for s in per_prompt)
    wmean = (sum(s["mean"] * s["n"] for s in per_prompt if s["mean"] is not None)
             / all_n) if all_n else None
    return {"max": max(maxes) if maxes else None, "mean": wmean,
            "tokens_compared": all_n,
            "tokens_match_all": all(s["tokens_match"] for s in per_prompt)}


def main() -> int:
    for path, why in [
        (VLLM_BIN, "venv missing; run scripts/bootstrap_8x.sh"),
        (MODEL_DIR, "model missing; run scripts/bootstrap_8x.sh"),
        (MODIFIED_DIR, "put the modified kernels in ~/tml_fa4_modified"),
        (ROUTE_PATCH, "scp kernels/patches/u2_serving_route.py to ~"),
    ]:
        if not Path(path).exists():
            log(f"FATAL: {path} not found ({why})")
            return 2

    results: dict = {
        "started": stamp(),
        "config": {"model_dir": str(MODEL_DIR), "served_name": SERVED_NAME,
                   "tp": TP, "max_model_len": MAX_MODEL_LEN, "port": PORT,
                   "tol_max": TOL_MAX, "tol_mean": TOL_MEAN,
                   "num_prompts": len(PROMPTS),
                   "batch_prompts": len(BATCH_PROMPTS),
                   "batch_max_tokens": BATCH_MAX_TOKENS},
        "builds": {}, "parity": None, "batch_consistency": {},
        "last_error": None,
    }

    pkg = resolve_tml_pkg()
    ensure_backup(pkg)
    parity_data: dict[str, list[dict] | None] = {"stock": None, "ours": None}
    batch_data: dict[str, dict | None] = {"stock": None, "ours": None}

    try:
        for build in ("stock", "ours"):
            files = deploy(build, pkg)
            results["builds"][build] = {"deployed_files": files,
                                        "serve_log": str(LOG_DIR / f"serve_{build}.log")}
            proc = None
            try:
                proc = start_server(build)
                parity_data[build] = collect_parity(build)
                batch_data[build] = collect_batch(build)
            finally:
                stop_server(proc)
    except Exception as exc:  # noqa: BLE001
        results["last_error"] = f"{type(exc).__name__}: {exc}"
        log(f"ERROR: {results['last_error']}")
    finally:
        # always leave the box in stock state
        try:
            deploy("stock", pkg)
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: could not restore stock kernels: {exc}")

    all_pass = True

    # cross-build parity on the 32 echoed prompts
    if parity_data["stock"] and parity_data["ours"]:
        per_prompt = [diff_stats(a, b) for a, b in
                      zip(parity_data["stock"], parity_data["ours"])]
        agg = aggregate(per_prompt)
        ok = (agg["max"] is not None and agg["max"] <= TOL_MAX
              and agg["mean"] <= TOL_MEAN and agg["tokens_match_all"])
        results["parity"] = {"per_prompt": per_prompt, **agg, "pass": ok}
        all_pass &= ok
        print(f"GATE logit_parity: {'PASS' if ok else 'FAIL'} "
              f"(max={agg['max']}, mean={agg['mean']}, "
              f"n={agg['tokens_compared']}, tol_max={TOL_MAX}, "
              f"tol_mean={TOL_MEAN})", flush=True)
    else:
        results["parity"] = None
        all_pass = False
        print("GATE logit_parity: FAIL (missing build data, see last_error)",
              flush=True)

    # batched-vs-batch-1 per build; the stock number is also the measured
    # noise floor for the cross-build tolerance (see derivation above)
    for build in ("stock", "ours"):
        data = batch_data[build]
        if data:
            per = [diff_stats(a, b)
                   for a, b in zip(data["batched"], data["single"])]
            agg = aggregate(per)
            ok = (agg["max"] is not None and agg["max"] <= TOL_MAX
                  and agg["mean"] <= TOL_MEAN and agg["tokens_match_all"])
            results["batch_consistency"][build] = {"per_prompt": per, **agg,
                                                   "pass": ok}
            all_pass &= ok
            print(f"GATE batch_consistency_{build}: "
                  f"{'PASS' if ok else 'FAIL'} (max={agg['max']}, "
                  f"mean={agg['mean']}, tokens_match="
                  f"{agg['tokens_match_all']})", flush=True)
        else:
            results["batch_consistency"][build] = None
            all_pass = False
            print(f"GATE batch_consistency_{build}: FAIL (no data)", flush=True)

    results["finished"] = stamp()
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"results written: {RESULT_PATH}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
