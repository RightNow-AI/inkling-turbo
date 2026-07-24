# Session 26: one H100 SXM5, Modal, 2026-07-25

`modal run --detach scripts/modal_e2e_bench.py --step validate`
5.3 minutes of GPU, about $0.41. Ledger total $0.69 of the $200 cap.

Device: NVIDIA H100 80GB HBM3, capability (9, 0). torch 2.11 / cu130.
No model checkpoint: every harness builds random tensors and calls the kernels
directly.

This run applied **both** `u3_fp8_kv.py` and `u2_shear_fusion.py`, in that
order. That combination had never run on a GPU before.

## Read this before quoting any file in here

**`microbench_ours_attn_shearfusion_OFF/ON_*.json` contain no attention
numbers.** Every one of the seven attention cases is `null`. Only the two
`gate_select_*` cases, which do not touch attention, produced timings. The step
is recorded as PASS because the harness catches each case separately and the
process still exited 0. The verdict line is not the result; the JSON is.

The cause is a single defect, and it hit everything that launches attention:

```
DSLUserCodeError: NameError in `__call__`: cannot access local variable
'n_block' where it is not associated with a value
  --> vllm/third_party/tml_fa4/flash_fwd_sm90.py:893
```

Also note: `microbench_ours_*` holds OUR timings, not day-0 ones. The harness
times whatever `tml_fa4` resolves to, which here is our build. The only day-0
attention baseline is `microbench_attn_scoremod_*.json`.

## What passed

| gate | result |
|---|---|
| `parity_kv_fp8` | PASS. u3 FP8 KV works on sm_90. |
| `parity_qkvr_prep` | PASS, 5/5. The default `qkvr_prep` path is unchanged after both patches rewrote that file. |
| `parity_shear_fusion` writer cases | 14 of 14 bit-exact. First time the fused sheared writer has been validated on sm_90; previously sm_120 only. |
| `microbench_attn_scoremod` | PASS. The day-0 baseline still runs. |

## What failed

| gate | result |
|---|---|
| `parity_fa4_rel` | `tml_fa4_rel_bias` failed all 3 cases with the `n_block` error. `score_mod` and `relproj_v1` passed all 3, so the machine and harness are fine and the defect is in our rel-bias path. |
| `parity_shear_fusion` attention cases | 2 of 2 failed, same error. These are the only two of the 16 that launch attention. |
| `presheared_*` | all 4 failed, same error. The pre-sheared `bias=` path cannot execute on sm_90. |
| `splitkv_*` | both failed. sm_90 split-KV remains unvalidated. |

## The one new measurement: what the shear fusion costs

This is the number `u2_shear_fusion_notes.md` called "the measurement that does
not exist yet". Both paths timed in one process on the same inputs, per-kernel
via `torch.profiler`. Every case parity-checked bit-exact against the stock
ShearingBias output before its timing was recorded.

Attention consumes an **identical** buffer in both paths, so this delta is the
entire effect of the fusion, not half of it.

| shape | natural writer | ShearingBias | fused writer | net |
|---|---|---|---|---|
| prefill global 8K | 439.7 | 872.4 | 2336.1 | **+1019.4, loss** |
| prefill SWA 8K | 217.8 | 468.1 | 1251.6 | **+561.1, loss** |
| decode b32, 64K KV | 3.4 | 7.4 | 10.3 | **-4.7, win** |

All values us/iter. The fused writer is 5.3x slower on global prefill and 5.7x
slower on SWA prefill than the natural writer it replaces. It emits
`rel_extent + 256` columns into a `(T + 128, H, ext + 256)` buffer instead of
`rel_extent` columns into `(T, H, ext)`; on these shapes that costs far more
than the ShearingBias launch it removes.

The `net` column excludes a `torch.full(NaN)` kernel that
`run_shearing_bias` runs every iteration so the parity gate can detect
unwritten columns. Production allocates with `torch.empty`
(`kernels/tml_fa4_modified/interface.py:725,735`), so that fill is harness
scaffolding and charging it to the natural path would flatter the fusion. The
raw totals including it are 1730.1 / 936.7 / 23.7, giving deltas of +606.0 /
+314.9 / -13.3, which are conservative bounds in the fusion's favour and still
losses on both prefill shapes.

Cross-check on the shared kernel: ShearingBias measures 468.1 us here against
460.9 us in session 25 for SWA prefill (1.6% apart) and 872.4 against 827.2 for
global prefill (5.5% apart), on a different machine and a different container.

### What this refutes

The repo previously projected that removing the ShearingBias pre-kernel would
take SWA prefill from 1223.0 to about 762.0 us/iter and turn our one measured
loss into a win. That projection was labelled as arithmetic rather than
measurement, and it is now **measured to be wrong**: the fusion does not
subtract 460.9 us, it adds 561.1 us. Default off was the right call for a
reason nobody had established until this run.

The fusion is a small win on decode. Nothing here supports enabling it on
prefill.
