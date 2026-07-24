#!/bin/bash
# Fully autonomous ON-BOX gate run: keeps going after the machine that
# launched it disconnects. Runs on a rented node, so it is burning money for
# its whole duration; the watchdog below is what bounds that.
# Launch with:  nohup bash ~/run_gates_full.sh > ~/gates_run.log 2>&1 &
#
# Stages: bootstrap (install + model download) -> per-op parity ->
# 32-prompt logit gate (stock vs ours) -> e2e serving bench -> summary.
# Every stage appends to ~/gates_run.log; a DONE or FAILED marker file is
# written at the end. The companion watchdog (gates_watchdog.sh) terminates
# this instance via the Lambda API at the hard cap no matter what.
set -uo pipefail

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$HOME/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"

log "=== STAGE 1: bootstrap (install + 592GB model download) ==="
if ! bash ~/bootstrap_8x.sh >> ~/stage_bootstrap.log 2>&1; then
  log "BOOTSTRAP FAILED (see stage_bootstrap.log tail below)"
  tail -30 ~/stage_bootstrap.log
  touch ~/GATES_FAILED
  exit 1
fi
log "bootstrap OK"

cd ~/vllm && source .venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$HOME/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

log "=== STAGE 2: per-op parity (8x box, GPU 0) ==="
timeout 1200 python ~/parity_fa4_rel.py 2>&1 | tee ~/stage_parity.log | grep tml_fa4
if grep -q "FAIL" ~/stage_parity.log; then
  log "PARITY RED - continuing to logit gate anyway (it compares builds, not oracle)"
fi

log "=== STAGE 3: 32-prompt logit gate (stock vs ours) ==="
if MODEL_DIR="$HOME/models/inkling" timeout 14400 python ~/gate_logit_parity.py >> ~/stage_logit.log 2>&1; then
  log "logit gate script rc=0"
else
  log "logit gate rc!=0 (see stage_logit.log tail)"
  tail -20 ~/stage_logit.log
fi
tail -5 ~/stage_logit.log

log "=== STAGE 4: e2e serving bench (stock vs ours) ==="
if MODEL_DIR="$HOME/models/inkling" timeout 32400 bash ~/gate_e2e_bench.sh >> ~/stage_e2e.log 2>&1; then
  log "e2e bench rc=0"
else
  log "e2e bench rc!=0 (see stage_e2e.log tail)"
  tail -20 ~/stage_e2e.log
fi

log "=== STAGE 5: summarize ==="
python ~/gate_summarize.py >> ~/stage_summary.log 2>&1 || true
[ -f ~/gate_summary.md ] && log "summary written" && cat ~/gate_summary.md

touch ~/GATES_DONE
log "=== ALL STAGES COMPLETE - artifacts: gate_logit_parity.json, bench_results/, gate_summary.md ==="
log "Box stays up for artifact retrieval until the watchdog hard cap."
