# Session 30 (local, `sm_120`): the pack_gqa fix, and real artifacts this time

2026-07-25, RTX 5090 Laptop GPU, capability 12.0, WSL2, torch 2.11.0+cu130.
Local hardware, so **$0.00**.

Two jobs. Record the `pack_gqa` shear-granularity fix
([../../regression-pack-gqa-shear-granularity.md](../../regression-pack-gqa-shear-granularity.md))
on the architecture it was root-caused on, and replace
[local_sm120_s29/](../local_sm120_s29/), whose three JSON files are **zero
bytes** because a `wsl.exe -- cat > file` redirection wrote nothing while
appearing to succeed. That gap is named in three places in this repository as
the worst shape a weak-class number can take, since the directory looks like an
artifact set from outside. These files are 4 KB to 27 KB and parse.

| gate | result | note |
|---|---|---|
| `parity_rel_varlen_batch` | **12/12** | was 11/12. `single_m_tail_chunked` max 0.1035 to 4.8828e-04, mean 1.9674e-03 to 4.1567e-05 |
| `parity_rel_chunked_decode` | **7/7** | all seven means and maxes **bit-identical** to session 29, so the fix moved nothing it should not have |
| `parity_rel_bias_coverage` | **6/6** | probes unchanged, the 64K case still touches tiles 504..511 |
| `parity_shear_fusion` | **0/16** | **not a kernel result.** See below |

`parity_fa4_rel` (3/3 on `tml_fa4_rel_bias`, 3/3 on `score_mod`) and
`parity_qkvr_prep` (5/5) also ran green in this session and write no JSON, so
they are journal-only here as they were in session 29.

## The 0/16 is an import failure, not a kernel failure

Every one of the sixteen cases records the same thing:

```
EXCEPTION ImportError: cannot import name 'RelShearSpec' from
'vllm.models.inkling.nvidia.ops.qkvr_prep'
```

`RelShearSpec` is absent from the deployed local vLLM tree, so the harness cannot
construct a single case and fails before touching a GPU. It is an environment
defect in this laptop checkout, in a module the `pack_gqa` guard does not touch,
and `parity_shear_fusion` was not in session 29's gate list either. It is
reported here rather than dropped, because a 0/16 that goes unexplained is
indistinguishable from a regression. **It is not evidence that fused shear works,
and it is not evidence that it does not.**

## What this session does and does not establish

- **Establishes** that the `pack_gqa` guard takes multi-sequence varlen batching
  from 11/12 to 12/12 on `sm_120`, that it leaves every previously passing case
  bit-identical across all 18 recorded cases, and that with the guard removed the
  failure returns. The gate has been seen failing on the defect it catches.
- **Does not establish anything on `sm_80`.** Ampere runs the same
  `flash_fwd.py` and the same heuristic, and session 31 observed the defect there
  at mean 1.9666e-03, but the fix has not executed on that architecture.
- **No perf claim.** The same 4096-token prefill shape timed 808 and 6293 us/iter
  minutes apart on this laptop. Nothing timed here is usable.
