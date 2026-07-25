# Inkling has no attention path on SM8x: the only non-Blackwell route is `score_mod`, which the cute backend hard-blocks on SM8x

**Target tracker:** vllm-project/vllm (issues enabled, verified 2026-07-25)
**Severity:** medium. Support gap, not a wrong-output bug. The model cannot
serve at all on A100-class hardware, and the failure arrives at the first
attention call rather than at load.

> **Review status, 2026-07-25: FILEABLE, with two additions made below.**
> Tracker correct and issues enabled. Duplicate check run and empty for this
> defect. All four vLLM-side line citations re-verified against our build base
> (`850295881`): `fa4_rel_attention.py:19-22` and `:133-143`,
> `attention.py:312`, `moe.py:73`. All four are correct as written.
>
> **And they hold at vLLM `main`, which is the important part for filing.**
> `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py` is **byte-identical**
> between our build base and vLLM `main` as of 2026-07-25, so `_use_sheared_bias`
> is still at `:20` and the router branch still at `:133`. A maintainer reading
> `main` will find exactly what this report quotes. The reproducer's import,
> `from vllm.models.inkling.nvidia.ops.fa4_rel_attention import
> inkling_fa4_rel_attention`, also resolves on `main`; that module path was
> confirmed present. The reproducer depends only on vLLM itself and on nothing
> from our repository, which is what it needs to be for this tracker.
>
> **The report's own open item is now CLOSED.** It said of the score_mod SM8x
> raise in the vllm-flash-attention copy: "Our notes record it at
> `interface.py:722` in that copy. Confirm the current line number when filing."
> Confirmed. At the pinned commit `caaa4eb5`, the file is
> `flash_attn/cute/interface.py` and the raise is at **`:722`**, inside the
> `elif score_mod is not None:` block that opens at `:720` with the arch test at
> `:721`. Note also a **second** instance at `:1683`, in the backward path, which
> this report does not need but which a maintainer will see. Note the upstream
> path is `flash_attn/cute/`, not `vllm_flash_attn/cute/`; the latter is the
> vendored name inside a built vLLM and does not exist in the upstream repo.
>
> **Context the filer should include, because it makes the report land better.**
> The router this report quotes was written deliberately, and recently. vLLM PR
> #48858, "[Model] Add Hopper FA4 relative attention for Inkling", merged
> 2026-07-16, is what introduced `_use_sheared_bias()` and the `score_mod` else
> branch. Its own description says it "adds architecture-routing and numerical
> correctness coverage for the standard score-mod path", and its test file covers
> routing for SM90, SM100, SM110 and SM120. **SM8x is not in that list.** So this
> is not a case of nobody having thought about arch routing. It is a case of the
> routing having been designed for four families and sm_80 falling through the
> else branch into a backend that rejects it. Framing it that way is accurate,
> is easy for a maintainer to confirm, and does not imply carelessness.
>
> **Duplicate check, commands run and their real output:**
>
> ```bash
> gh repo view vllm-project/vllm --json hasIssuesEnabled   # {"hasIssuesEnabled":true}
> gh search issues --repo vllm-project/vllm "Inkling Ampere"            # empty
> gh search issues --repo vllm-project/vllm "Inkling unsupported GPU"   # empty
> gh search issues --repo vllm-project/vllm "score_mod not supported"   # empty
> gh search issues --repo vllm-project/vllm "score_mod SM8x"            # empty
> gh search issues --repo vllm-project/vllm "inkling A100"              # empty
> gh search issues --repo vllm-project/vllm "Inkling"                   # 4 hits, none is this
> gh search prs    --repo vllm-project/vllm "Inkling" --limit 25        # 25 hits, none is this
> ```
>
> Nothing covers the SM8x gap. Two adjacent open items were found and neither is
> a duplicate, recorded so the filer is not blindsided:
>
> - **vLLM issue #49049**, open, "[Bug] Inkling on sm_121a (GB10): unclamped
>   q-row in the rel-bias score-mod gather causes deterministic illegal address".
>   That is a memory-safety defect in the score_mod **gather**, on sm_121a, with
>   coredump evidence. Different architecture, different mechanism, and it is
>   about the path working incorrectly rather than being unavailable. Not a
>   duplicate. Worth reading before filing, because it is the closest existing
>   report on this exact code and it shows the maintainers engage with
>   hardware-evidence reports here.
> - **vLLM PRs #48841 and #48954**, both open, add ROCm Inkling support. Adjacent
>   support-matrix work, not SM8x, not a duplicate.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/vllm | fork base `850295881` | our build base |
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` |
| vllm-project/flash-attention | `caaa4eb59845388a20b1f435ecaafb4bd9517ad8` | `cmake/external_projects/vllm_flash_attn.cmake:42` |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` |

Verified on 8x A100-SXM4-40GB, compute capability `(8, 0)`, torch 2.11.0+cu130.

## Summary

Two pieces of routing logic do not agree, and nothing checks between them.

**The model router sends everything that is not Blackwell to `score_mod`.**
`vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`:

```python
# :19-22
@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)
```

```python
# :133-143
if _use_sheared_bias():
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func
    bias_kwargs: dict[str, Any] = {"rel_bias": rel_logits}
else:
    from vllm.vllm_flash_attn.cute import flash_attn_varlen_func
    bias_kwargs = {
        "score_mod": _get_score_mod(rel_extent),
        "aux_tensors": [rel_logits],
    }
```

This is the only attention path in the model. It is called from
`vllm/models/inkling/nvidia/attention.py:312`, inside `_attention`, which is
the sole attention entry point for the layer.

**The cute backend refuses `score_mod` on SM8x.** In tml-fa4 @ `13374f0c`,
`flash_attn/cute/interface.py:605-607`:

```python
elif score_mod is not None:
    if arch // 10 == 8:
        raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")
```

The same block is present in the vllm-flash-attention copy that the `else`
branch above actually imports. Our notes record it at `interface.py:722` in
that copy. Confirm the current line number when filing.

So on capability `(8, x)`, the router picks `score_mod` and the kernel library
raises. There is no third path. There is no arch guard for the attention path
anywhere in `vllm/models/inkling/`. The only capability check in that tree
outside the router is `moe.py:73`, which selects a fast GEMM and falls back
cleanly, so it does not stop the load.

## Reproduction

```bash
uv venv --python 3.12 && source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
.venv/bin/python repro_sm8x_inkling_attention.py
```

```python
# repro_sm8x_inkling_attention.py, run on an A100 or any capability (8, x) GPU
import torch
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    inkling_fa4_rel_attention,
)

print("capability:", torch.cuda.get_device_capability())

NT, HQ, HKV, D, EXT = 128, 8, 2, 128, 1024
BS, NB = 16, 64
dev = "cuda"
q = torch.randn(NT, HQ, D, dtype=torch.bfloat16, device=dev)
kc = torch.randn(NB, BS, HKV, D, dtype=torch.bfloat16, device=dev)
vc = torch.randn(NB, BS, HKV, D, dtype=torch.bfloat16, device=dev)
rel = torch.randn(NT, HQ, EXT, dtype=torch.bfloat16, device=dev)
bt = torch.arange(NT // BS, dtype=torch.int32, device=dev).view(1, -1)
cs = torch.tensor([NT], dtype=torch.int32, device=dev)
cu = torch.tensor([0, NT], dtype=torch.int32, device=dev)

inkling_fa4_rel_attention(
    q, kc, vc,
    block_table=bt, cache_seqlens=cs, cu_seqlens_q=cu,
    max_seqlen_q=NT, softmax_scale=1 / D, causal=True,
    window_size=(-1, -1), rel_extent=EXT, rel_logits=rel, num_splits=1,
)
```

Equivalently, start a server with the Inkling checkpoint on an A100 node and
send one request.

### Expected

Either attention output, or a clear unsupported-hardware error at model load
that names the arch and the reason.

### Actual

```
NotImplementedError: Custom user-provided score_mod is not supported on SM8x architectures.
```

raised at the first attention call, after the model has loaded and the KV cache
has been allocated.

## Evidence

Session 26 in `journal/u2-hopper-design.md` at
<https://github.com/RightNow-AI/inkling-turbo>, on an 8x A100-SXM4-40GB node,
capability `(8, 0)`, torch 2.11.0+cu130.

- Day-0 production path: `NotImplementedError` on every parity case.
- Our generic sheared-bias kernel on the same cases: 3 of 3 green against the
  float32 reference, max abs diff 7.8e-3, 7.8e-3, 1.56e-2, tolerance 2e-2. All
  three of those cases are `seqlen_q == seqlen_k`, so they establish the
  full-prefill family and nothing else. Decode and chunked prefill on sm_80 have
  no correctness result on any hardware.
- FP8 paged KV parity on the same box: 2 of 2 green.

Because no day-0 path executes here, no speedup comparison is possible on this
arch. The following are absolute per-op timings of the working kernel, with no
baseline to compare against:

| Case | us |
|---|---|
| prefill_global_8k | 10749.9 |
| prefill_swa_8k | 10297.2 |
| decode_b1_kv64k | 5510.3 |
| decode 32 seqs, kv64k | 75013.4, that is 2344 per sequence |
| gate_select, T=1 / T=4096 | 7.4 / 47.1 |

The two decode rows carry a caveat added 2026-07-25: they were timed on a kernel
whose sheared-bias reader used the `seqlen_q == seqlen_k` specialisation, so they
are not the cost of correct decode attention on Ampere. They are kept because
they are what the box measured. The two prefill rows are unaffected.
`journal/regression-ampere-tile-sweep.md`.

## Root cause

The router at `fa4_rel_attention.py:19-22` treats "not Blackwell" as "use
`score_mod`". The kernel library treats `score_mod` as "sm_90 and up". Neither
side knows about the other's assumption, and the model has no minimum
capability declared, so the mismatch surfaces at runtime rather than at load.

## Suggested fix

Two options. They are not mutually exclusive.

1. **Fail fast and honestly.** Declare a minimum compute capability for the
   Inkling attention path and raise at model init with a message that names the
   arch. A user on an A100 node should learn this in the first second, not
   after the weights are resident.

2. **Give SM8x a real path.** Route SM8x to a sheared-bias kernel instead of
   `score_mod`. The bias tensor and the `ShearingBias` pre-kernel already exist
   and are arch-independent in tml-fa4's interface. What is missing is bias
   consumption in the SM80-family kernel, which is upstream finding 01 in this
   series. Our implementation is at
   `kernels/tml_fa4_modified/flash_fwd.py` and
   `kernels/patches/u2_v0_generic_bias.py`, with the routing change in
   `kernels/patches/u2_serving_route.py`, at
   <https://github.com/RightNow-AI/inkling-turbo>.

If option 2 is taken, note that the SM80-family tile default is untuned.
`flash_attn/cute/interface.py:520` in tml-fa4 reads
`fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune`. Our A100 sweep
data is in `journal/remote/tune_sm80_a100.json`. Two of its findings stand and
one is withdrawn, and the distinction is `seqlen_q` against `seqlen_k`:

- **Stands.** Sliding-window prefill 8K prefers `tile_n=64`, 9175.2 us against
  10565.6 us at 32, and global prefill 8K prefers 32, 10712.7 against 11124.1.
  Both cases are 8192 query rows against 8192 keys. **Read those two margins as
  single samples, because that is what they are.** Repeated five times on a
  verified A100, the sliding-window preference for 64 holds but at 1.3 percent
  rather than 15, and global prefill becomes undecidable, because `tile_n=32`
  swings 22 percent across rounds on that shape.
- **Stands.** `tile_n=128` collapses on sm_80 shared-memory pressure, 362806.1 us
  against 10712.7 us on global prefill, a factor of 34. Do not select it.
- **WITHDRAWN, and the re-measurement it was pending has now happened and
  reversed it.** Our decode-shaped percentages, ~~10.1 percent~~ at batch 1 with
  64K KV and ~~18.2 percent~~ on the 32-sequence case, and a ~~18.7 percent~~
  post-deploy re-run. Our own harness timed those at `T_q = 1` against
  `T_k = 65536` while its parity gate ran `seqlen_q == seqlen_k` at
  `Hq == Hkv`, and our generic kernel carried a `seqlen_q == seqlen_k`
  specialisation in its sheared-bias reader plus a `pack_gqa` mismatch that only
  bites at `Hq != Hkv`, so the tile size was selected under a reader addressing
  the bias outside its own tile domain at the timed shapes and geometry. Both
  were our defects, not upstream's.

  Re-measured 2026-07-25 on a verified A100 with five interleaved rounds per
  cell: on batch-1 decode at 64K, **`tile_n=64` is 9.7 percent faster than
  `tile_n=32`**, the opposite of what we published, with the two sample intervals
  spanning 0.03 and 0.07 percent of their medians. On the 32-sequence case
  `tile_n=32` does win, by 3.3 percent rather than 18.2. **Nothing here asks
  upstream to change the `FwdConfig(128, 64)` default**: it wins two of the three
  shapes we can decide and we cannot decide the third.
  `journal/regression-ampere-tile-sweep.md`,
  `journal/remote/validate_a100x1_s32_packgqa/`.

So the actionable part of this note for upstream is the first two bullets and the
`# SM80, should tune` comment itself. Do not carry our decode percentages into a
patch description.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check was run
against this tracker before filing. A human submitter reviewed the report and
will review and defend every line of any follow-up PR.
