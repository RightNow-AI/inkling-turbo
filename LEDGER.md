# LEDGER, Inkling-turbo

Every number here is measured-or-null. `null` = not yet measured. No estimates in this file, ever.
Every non-null number names its evidence, and its evidence class. The strong class is an artifact under `journal/remote/` that a reader can re-parse: JSON, a raw log, a profiler export. The weak class is a number transcribed into a journal session with no machine-readable record behind it. Both are allowed; mixing them without saying which is which is not, because it lets a reader assume every number here is checkable. Weak-class numbers are marked **journal-only** in the cell that carries them, so the marker is the authority and no count is repeated here to drift out of date. `grep -n "journal-only" LEDGER.md` enumerates them. Everything else cites an artifact. See [journal/remote/README.md](journal/remote/README.md) for what each artifact is and [docs/METHODOLOGY.md](docs/METHODOLOGY.md#measured-or-null-ledger) for the rule.
Baseline reference (external claim, NOT ours): vLLM day-0 blog reports 380 tok/s/user (MTP8) / 140 tok/s/user (no MTP) on 4x GB200 TP8, 8K in / 1K out.

## E2E serving (remote, vs stock day-0 build, same checkpoint/quant/GPUs/SLO)

| Metric | Stock day-0 | Inkling-turbo | Delta | Evidence |
|---|---|---|---|---|
| Decode tok/s/user @ bs1 | null | null | null | null |
| Throughput tok/s @ bs8 | null | null | null | null |
| Throughput tok/s @ bs32 | null | null | null | null |
| Throughput tok/s @ bs128 | null | null | null | null |
| TTFT p50 prefill-heavy | null | null | null | null |
| TPOT p50 decode-heavy | null | null | null | null |

Why every cell above is `null`: the sweep was launched on 8x H100 and the box
self-terminated mid-run because of a watchdog race in our own orchestration. The
stock side had completed roughly 26 to 30 of 30 runs and none of it was
retrieved. Nothing partial is reported from it. Written up in
[journal/u2-hopper-design.md](journal/u2-hopper-design.md#session-28-postscript-e2e-curves-lost-to-a-watchdog-race-orchestrator-error).

## Kernel gates (ncu, >=90% of binding roofline)

| Unit | Kernel | Binding roofline | % achieved | Parity 32/32 | batched==bs1 | Profile |
|---|---|---|---|---|---|---|
| U1 | fused NVFP4 MoE grouped GEMM | null | null | null | null | null |
| U2 | rel-attn prefill | ncu ran, 90% gate NOT met (SM SOL 45.6%, MemSOL 55.9%, occupancy 14.0% at prefill 8k; decode b1 is parallelism-bound at 64 CTAs on 132 SMs, DRAM 7.2%). **journal-only**: these four percentages were read off Nsight Compute and transcribed into journal session 24. The `.ncu-rep` files are gitignored, so no artifact under `journal/remote/` carries them | below gate, ceiling documented in journal session 24 ncu | per-op 3/3 on sm_90+sm_80+sm_120; FULL-MODEL 32/32 greedy-token match vs stock on 8x H100 (2026-07-24); logprob-tolerance gate fail-as-specified with control (noise floor) also failing - see journal session 28 | batched==bs1 FAILS ON STOCK ITSELF (platform nondeterminism, documented) | session 24: decode b1 kv64k 905.6us vs day-0 score_mod 2375 (2.6x); prefill 8k 3362us vs day-0 score_mod 5368 (1.6x). Also measured on the same box: relprojT 4163 / relproj 6209 at decode and 8483 / 13049 at prefill, but those are OUR abandoned V1/V1.5 prototypes, NOT day-0 paths, and no speedup is claimed against them. journal/remote/microbench_attn_day0_native_sm90_session24.json + microbench_attn_scoremod_session24.json |
| U2 | rel-attn, session-25 reproduction (different box, torch 2.11/cu130) | see row above | see row above | per-op 3/3 native sm_90, max 1.56e-2 (**journal-only**: the 1.56e-2 is transcribed from journal session 25, `parity_fa4_rel.py` prints it and writes no artifact) | not re-run | vs the only day-0 path (score_mod): decode b1 kv64k 852.6 vs 2326.6 (2.7x); decode b32 854.8 vs 2391.2 (2.8x); prefill_global_8k 3308.8 vs 4799.4 (1.45x); **prefill_swa_8k 1223.0 vs 956.5 = 1.28x SLOWER, we lose this case**. Reference points, not baselines: plain no-bias 736.0 (decode b1); our abandoned relprojT 5154.7 / relproj 7194.5. True 32-seq decode kv64k 13821.4 (431.9/seq). journal/remote/microbench_attn_day0_session25_h100.json + microbench_attn_scoremod_session25_h100.json |
| U2 | rel-attn on sm_80 (A100) | null | null | per-op 3/3 (2026-07-23) | null | no speedup ratio exists: every day-0 path raises NotImplementedError on SM8x, so there is no baseline. Support claim only. Tile sweep (parity-gated, journal/remote/tune_sm80_a100.json): tile_n=32 b1 decode 5350.1us vs untuned tile_n=64 5953.7 = 10.1% faster; 32-seq 60801.4 vs 74356.6 = 18.2% faster. Post-deploy re-validation on A100, 32-seq 60977.5 vs 75013.4 = 18.7% faster, journal session 27 (**journal-only**, no separate JSON; the tile sweep above is strong class, this re-run is not) |
| U2 | rel-attn shear fusion (`u2_shear_fusion.py`: qkvr_prep writes the sheared layout, ShearingBias and its two scheduler kernels removed) | null | null | sm_90 (H100, session 26): **14/16**. All 14 writer cases bit-exact vs stock ShearingBias. Both attention cases FAILED: `NameError: cannot access local variable 'n_block'` in `flash_fwd_sm90.py`, so the pre-sheared `bias=` path cannot execute on Hopper at all. sm_120 (RTX 5090): 16/16, one run, **journal-only**, the only record is commit `7375849`; that gate wrote no file. NOT run on sm_100. Artifact: journal/remote/validate_s26_h100x1/ | null | **MEASURED, and it is a net LOSS on prefill.** Fused writer vs natural writer + ShearingBias, same inputs, one process, per-kernel: global prefill 8K 2336.1 vs 1312.1 = **+1019.4us**; SWA prefill 8K 1251.6 vs 685.9 = **+561.1us**; decode b32 kv64k 10.3 vs 10.8 = -4.7us, a small win. Attention consumes an identical buffer either way, so this delta is the whole effect. This **refutes** the earlier projection that the fusion would turn the SWA prefill loss into a win. Ships off by default (`INKLING_TURBO_FUSED_SHEAR=1` to enable) and should stay off for prefill. journal/remote/validate_s26_h100x1/microbench_presheared_splitkv_modal_h100x1.json |
| U2 | rel-attn split-KV decode | null | null | null | null | null |
| U3 | quantized paged KV | null | null | per-op 2/2 on sm_120 + sm_90/H100 + sm_80/A100 (2026-07-20/23); 32/32 pending | null | null |
| U4 | router+dispatch fusion | null | null | null | null | null |
| U5 | QKVR fused GEMM | null | null | null | null | null |
| U6 | graphs+overlap | n/a (timeline) | null | null | null | null |
| U7 | batch-aware dispatch | null | null | null | null | null |

## Corrections

Numbers in this file have been wrong twice. Both are recorded rather than
quietly edited, because a ledger that only ever agreed with itself would not be
worth reading.

| date | what was wrong | how it was found | where it is written up |
|---|---|---|---|
| 2026-07-24 | The headline divided our kernel by `relproj`/`relprojT`, which are **our own** abandoned prototypes, not day-0 paths. It also omitted the measured `score_mod` prefill of 4799.4 us and quoted 3.2x instead of the true 1.45x, and never disclosed that we are 1.28x slower on sliding-window prefill. | An agent cross-checking citations against the raw artifacts | README, "What we do not claim" |
| 2026-07-25 | The projection that removing the `ShearingBias` pre-kernel would turn the sliding-window prefill loss into a win. It was labelled arithmetic, and it was wrong: the fused writer costs more than the kernel it removes. | Measured it on an H100 (session 26) instead of leaving it projected | README, "Removing the pre-kernel" |

## Reproducibility incidents

| date | what happened | effect on published numbers | write-up |
|---|---|---|---|
| 2026-07-25 | `3b78fc6` moved the `n_block` and `page_idx` bindings inside a non-`const_expr` `if` body, which the CuTe DSL traces as a separate scope. Every sm_90 attention call then failed at trace time with "cannot access local variable 'n_block'". The commit's own message said "unvalidated" and it was the tip of public `main` for several hours. | **None of the numbers were wrong.** They were measured on a file byte-identical to `f585227` and remain reproducible from it. They were **not** reproducible from `3b78fc6`, which cannot compile a single sm_90 attention call. Fixed by hoisting both bindings back to function scope. | [journal/regression-sm90-n-block.md](journal/regression-sm90-n-block.md) |
| 2026-07-25 | Two single-element tuples in `kernels/tml_fa4_modified/interface.py` (`t.shape == (batch_size)`, `.shape == (1)`) were still damaged by the em-dash cleanup regex, at sites the repair in `8c91a86` missed. A `torch.Size == int` comparison is always False, so both asserts fire for any caller passing `scheduler_metadata`. | No published number depends on that path. Fixed, and a tree-wide sweep for the pattern now returns zero. | same file |

## last_error

null
