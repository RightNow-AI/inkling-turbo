#!/usr/bin/env bash
# Gate: e2e serving benchmark, stock day-0 build vs Inkling-turbo kernels.
# Runs ON the 8x H100 Lambda box (after scripts/bootstrap_8x.sh).
#
# Benchmark entrypoint, verified against the pinned fork @850295881
# ($REPO/vllm):
#   - benchmarks/benchmark_serving.py is a DEPRECATED shim that exits 1 and
#     points at `vllm bench serve` (benchmarks/benchmark_serving.py lines
#     5-17). The real entrypoint is `vllm bench serve`
#     (vllm/entrypoints/cli/benchmark/serve.py -> vllm/benchmarks/serve.py).
#
# VERIFIED FLAGS (file:line in the pinned tree):
#   vllm bench serve:
#     --backend openai (default; endpoint default /v1/completions)
#                              vllm/benchmarks/serve.py:1487-1508
#     --base-url               vllm/benchmarks/serve.py:1494
#     --model                  vllm/benchmarks/serve.py:1532 (tokenizer source)
#     --served-model-name      vllm/benchmarks/serve.py:1767 (API model name)
#     --max-concurrency        vllm/benchmarks/serve.py:1518
#     --ignore-eos             vllm/benchmarks/serve.py:1664
#     --percentile-metrics     vllm/benchmarks/serve.py:1680 (ttft,tpot,itl,e2el)
#     --metric-percentiles     vllm/benchmarks/serve.py:1690 (default "99")
#     --save-result            vllm/benchmarks/serve.py:1624
#     --result-dir             vllm/benchmarks/serve.py:1648
#     --result-filename        vllm/benchmarks/serve.py:1655
#       (result path = result_dir/result_filename:
#        compute_result_filename, vllm/benchmarks/serve.py:1468-1473)
#     --dataset-name random    vllm/benchmarks/datasets/datasets.py:1614-1635
#     --num-prompts            vllm/benchmarks/datasets/datasets.py:1608-1613
#     --seed                   vllm/benchmarks/datasets/datasets.py:1607
#     --random-input-len       vllm/benchmarks/datasets/datasets.py:1930-1935
#     --random-output-len      vllm/benchmarks/datasets/datasets.py:1936-1941
#   vllm serve:
#     positional model_tag     vllm/entrypoints/openai/cli_args.py:346-351
#     --port                   vllm/entrypoints/openai/cli_args.py:229
#     --served-model-name      vllm/engine/arg_utils.py:859
#     --tensor-parallel-size   vllm/engine/arg_utils.py:1014
#     --max-model-len          vllm/engine/arg_utils.py:829
#     --gpu-memory-utilization vllm/engine/arg_utils.py:1163
#     --seed                   vllm/engine/arg_utils.py:816
#   GET /health readiness: vllm/entrypoints/serve/instrumentator/health.py:22
#
# Matrix (METHODOLOGY.md: same checkpoint, same GPUs, same SLO, median of
# 5 + best): 2 builds x 2 mixes x concurrency {1,8,32} x 5 runs = 60 runs.
# Results: ~/bench_results/<build>/<mix>/<concurrency>/run<N>.json
# Idempotent: existing run JSONs are skipped, so an interrupted session
# resumes where it stopped.
#
# COST WARNING: at 8x H100 = $31.92/hr the full default matrix is a
# multi-hour session on expensive hardware. Set a spend cap first.
# Trim with RUNS/CONCURRENCIES env vars if budget is tight.
set -euo pipefail
exec 2>&1

VENV_BIN="$HOME/vllm/.venv/bin"
VLLM="$VENV_BIN/vllm"
PY="$VENV_BIN/python"
MODEL_DIR="$HOME/models/inkling"
SERVED_NAME="inkling"
PORT="${GATE_PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/bench_results}"
LOG_DIR="$HOME/bench_logs"
BACKUP_DIR="$HOME/tml_fa4_backup"
MODIFIED_DIR="$HOME/tml_fa4_modified"
RUNS="${RUNS:-5}"
CONCURRENCIES="${CONCURRENCIES:-1 8 32}"
MAX_MODEL_LEN="${GATE_MAX_MODEL_LEN:-16384}"
SERVER_WAIT_S="${GATE_SERVER_WAIT_S:-5400}"
SEED=0
PID_FILE="$HOME/gate_serve.pid"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

[ -x "$VLLM" ] || { echo "FATAL: $VLLM missing; run scripts/bootstrap_8x.sh"; exit 2; }
[ -d "$MODEL_DIR" ] || { echo "FATAL: $MODEL_DIR missing; run scripts/bootstrap_8x.sh"; exit 2; }
[ -d "$MODIFIED_DIR" ] || { echo "FATAL: $MODIFIED_DIR missing"; exit 2; }
[ -d "$BACKUP_DIR" ] || { echo "FATAL: $BACKUP_DIR missing; bootstrap_8x.sh creates it BEFORE any deploy"; exit 2; }

# Deploy to the RESOLVED package path (precompiled install may import
# tml_fa4 from site-packages, not the source tree) -- same rule as
# scripts/bootstrap_b200.sh.
TML_PKG=$("$PY" -c "import vllm.third_party.tml_fa4 as m, os; print(os.path.dirname(m.__file__))")
echo "resolved tml_fa4 package dir: $TML_PKG"

deploy_build() {
  local build="$1" src
  if [ "$build" = "stock" ]; then src="$BACKUP_DIR"; else src="$MODIFIED_DIR"; fi
  cp "$src"/*.py "$TML_PKG/"
  echo "deployed $build kernels from $src"
}

start_server() {
  local build="$1"
  local log="$LOG_DIR/serve_${build}.log"
  echo "starting server ($build), log: $log"
  nohup "$VLLM" serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.95 \
    --seed "$SEED" \
    --port "$PORT" >"$log" 2>&1 &
  echo $! >"$PID_FILE"
  local deadline=$(( $(date +%s) + SERVER_WAIT_S ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "FATAL: server ($build) died; last 30 log lines:"
      tail -n 30 "$log"
      return 1
    fi
    if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
      echo "server ($build) healthy"
      return 0
    fi
    sleep 10
  done
  echo "FATAL: server ($build) not healthy within ${SERVER_WAIT_S}s"
  return 1
}

stop_server() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "stopping server pid $pid"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 36); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 5
      done
      kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 30  # let TP8 workers and NCCL release GPU memory
}

cleanup() {
  stop_server
  deploy_build stock || true  # always leave the box in stock state
}
trap cleanup EXIT

num_prompts_for() {
  # enough requests to saturate the concurrency level and give stable
  # medians, without exploding session cost on a 975B W4A16 model
  local conc="$1" np
  np=$(( conc * 4 ))
  [ "$np" -lt 16 ] && np=16
  echo "$np"
}

run_matrix() {
  local build="$1"
  for mix in prefill decode; do
    local ilen olen
    if [ "$mix" = "prefill" ]; then ilen=8192; olen=128; else ilen=512; olen=1024; fi
    for conc in $CONCURRENCIES; do
      local np outdir
      np=$(num_prompts_for "$conc")
      outdir="$RESULTS_ROOT/$build/$mix/$conc"
      mkdir -p "$outdir"
      for n in $(seq 1 "$RUNS"); do
        local outfile="$outdir/run${n}.json"
        if [ -f "$outfile" ]; then
          echo "SKIP (exists): $outfile"
          continue
        fi
        echo "=== RUN $build/$mix/conc${conc}/run${n}: in=$ilen out=$olen np=$np ==="
        "$VLLM" bench serve \
          --backend openai \
          --base-url "$BASE_URL" \
          --model "$MODEL_DIR" \
          --served-model-name "$SERVED_NAME" \
          --dataset-name random \
          --random-input-len "$ilen" \
          --random-output-len "$olen" \
          --num-prompts "$np" \
          --seed "$SEED" \
          --ignore-eos \
          --max-concurrency "$conc" \
          --percentile-metrics ttft,tpot,itl,e2el \
          --metric-percentiles 99 \
          --save-result \
          --result-dir "$outdir" \
          --result-filename "run${n}.json" \
          || { echo "RUN FAILED: $build/$mix/$conc/run${n} (left absent = null in summary)"; rm -f "$outfile"; }
      done
    done
  done
}

for BUILD in stock ours; do
  # skip a build entirely if every result file already exists (idempotent)
  missing=0
  for mix in prefill decode; do
    for conc in $CONCURRENCIES; do
      for n in $(seq 1 "$RUNS"); do
        [ -f "$RESULTS_ROOT/$BUILD/$mix/$conc/run${n}.json" ] || missing=1
      done
    done
  done
  if [ "$missing" -eq 0 ]; then
    echo "=== BUILD $BUILD: all results present, skipping ==="
    continue
  fi
  echo "=== BUILD $BUILD ==="
  deploy_build "$BUILD"
  start_server "$BUILD"
  run_matrix "$BUILD"
  stop_server
done

echo "=== BENCH COMPLETE: results under $RESULTS_ROOT ==="
echo "next: python scripts/gate_summarize.py"
