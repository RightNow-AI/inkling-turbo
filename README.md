# Inkling-turbo

Faster serving kernels for [TML Inkling](https://huggingface.co/thinkingmachines/Inkling) (975B total / 41B active MoE, 256 experts, 66 layers, relative attention, 1M context) on top of the vLLM day-0 support. Target: B200/B300-class multi-batch datacenter inference, upstream-quality diffs.

**Status: Phase 0 — baseline study. No performance claims exist yet.** Every number this repo will ever publish is measured, with the profile artifact next to it (see `LEDGER.md`, `journal/`). Fast-and-wrong is treated as broken.

## Layout

- `vllm/` — fork base (nested clone, pinned in `journal/phase0.md`); unit branches live here
- `journal/` — phase notes, ncu profiles, methodology, citations
- `kernels/` — kernel sources before upstreaming into the fork
- `harness/` — parity harness: per-op reference checks + 32-prompt logit gate + batched==batch-1 consistency
- `scripts/` — remote (Lambda) orchestration, benchmark sweeps; scriptable, kill-on-exit
- `LEDGER.md` — measured-or-null results ledger
- `BLOCKERS.md` — open decisions, never guessed through

## Method (gates every unit must pass)

1. Parity: 32/32 fixed-prompt logit parity vs HF reference (or documented tolerance) AND batched output == batch-1 output.
2. Kernel: >=90% of the binding roofline (HBM BW where memory-bound, FP4 TC throughput where compute-bound) in ncu, profile saved.
3. E2E: `vllm bench serve` vs stock day-0 build — same checkpoint, quant, GPUs, SLO; throughput-latency curves across batch sweep; median of 5 + best; prefill-heavy and decode-heavy mixes.
