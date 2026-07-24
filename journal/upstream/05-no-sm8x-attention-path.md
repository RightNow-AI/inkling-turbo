# Inkling has no attention path on SM8x: the only non-Blackwell route is `score_mod`, which the cute backend hard-blocks on SM8x

**Target tracker:** vllm-project/vllm
**Severity:** medium. Support gap, not a wrong-output bug. The model cannot
serve at all on A100-class hardware, and the failure arrives at the first
attention call rather than at load.

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
  float32 reference, max abs diff 7.8e-3, 7.8e-3, 1.56e-2, tolerance 2e-2.
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
`fwd_cfg = FwdConfig(128, 64, True, True)  # SM80, should tune`. Our
parity-gated sweep on A100 measured `tile_n=32` as faster for decode-shaped
calls, 5350.1 us against 5953.7 us at batch 1 with 64K KV, and 60801.4 us
against 74356.6 us for the 32-sequence case, while SWA prefill prefers 64,
9175.2 us against 10565.6 us. `tile_n=128` collapses by roughly 30x on sm_80
shared memory pressure. Selecting 32 for `max_seqlen_q <= 32` and 64 otherwise
measured 18.7 percent faster on the 32-sequence decode case after deployment.
Sweep data is in `journal/remote/tune_sm80_a100.json`.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check was run
against this tracker before filing. A human submitter reviewed the report and
will review and defend every line of any follow-up PR.
