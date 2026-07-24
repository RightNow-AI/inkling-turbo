# `rel_bias` is accepted and silently dropped on every non-Blackwell arch, plain attention is returned as the biased result

**Target tracker:** vllm-project/tml-fa4
**Severity:** highest of this series. Silent numerical wrongness, no error, no warning.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` in the vLLM tree |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` in the vLLM tree |

Affected code paths: `arch // 10 in (8, 9, 12)`. The `arch // 10 in (10, 11)`
Blackwell path is correct and is not affected.

## Summary

`flash_attn_varlen_func(..., rel_bias=...)` on SM8x, SM90 and SM120 does all of
the following and then throws the result away:

1. allocates the padded sheared bias tensor (`interface.py:682-699`),
2. compiles and launches the `ShearingBias` pre-kernel (`interface.py:816-858`),
3. constructs the forward kernel with **no** bias argument,
4. returns plain, bias-free attention as if it were the requested result.

The bias tensor is even returned by the internal entry point
(`interface.py:1434`) and then dropped by the public wrapper. Callers get
plausible-looking output that is numerically wrong.

## Root cause

Bias is threaded to exactly one kernel family. In `flash_attn/cute/interface.py`:

- `arch // 10 in (10, 11)` passes `has_bias=bias is not None` to the kernel
  constructor (`interface.py:1230`), passes `bias_tensor` to `cute.compile`
  (`interface.py:1312`) and `bias` at exec time (`interface.py:1390`).
- `arch // 10 == 9` builds `FlashAttentionForwardSm90` at
  `interface.py:1137-1158` with no bias-related argument.
- `arch // 10 == 8` builds `FlashAttentionForwardSm80` at
  `interface.py:1115-1134` with no bias-related argument.
- `arch // 10 == 12` builds `FlashAttentionForwardSm120` at
  `interface.py:1244-1260` with no bias-related argument.
- The shared `else` compile and exec branches (`interface.py:1320-1341` and
  `interface.py:1397-1417`) contain no bias argument at all.

The kernels themselves have no bias code to call. `grep -ci bias` returns
**zero** in `flash_fwd_sm90.py`, `flash_fwd.py` and `flash_fwd_sm120.py`,
against 236 in `flash_fwd_sm100.py`:

```bash
cd flash_attn/cute
grep -ci bias flash_fwd_sm90.py flash_fwd.py flash_fwd_sm120.py flash_fwd_sm100.py
# flash_fwd_sm90.py:0
# flash_fwd.py:0
# flash_fwd_sm120.py:0
# flash_fwd_sm100.py:236
```

`-c` counts matching **lines**. The equivalent count of raw occurrences,
`grep -oi bias flash_fwd_sm100.py | wc -l`, is 346. The three zeros are zero
under either flag, which is the claim being made here.

There is no guard anywhere that rejects `rel_bias` on these arches.

## Reproduction

Tested on 1x H100 SXM5 (sm_90), torch 2.11.0+cu129, nvidia-cutlass-dsl 4.6.0.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "nvidia-cutlass-dsl[cu13]==4.6.0" torch==2.11.0
# tml-fa4 checked out at 13374f0c, installed editable
.venv/bin/python repro_rel_bias_dropped.py
```

```python
# repro_rel_bias_dropped.py
import torch
from flash_attn.cute import flash_attn_varlen_func  # tml-fa4 @ 13374f0c

T, HQ, HKV, D, EXT = 128, 8, 1, 128, 1024
dev = "cuda"
q = torch.randn(T, HQ, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device=dev)
rb = torch.randn(T, HQ, EXT, dtype=torch.bfloat16, device=dev)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)

common = dict(
    q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=T, max_seqlen_k=T, softmax_scale=1 / D, causal=True,
)
out_bias = flash_attn_varlen_func(rel_bias=rb, **common)
out_plain = flash_attn_varlen_func(**common)
print("max |biased - plain| =", (out_bias - out_plain).abs().max().item())
```

### Expected

`out_bias` differs from `out_plain` by roughly the magnitude of the bias term.
Alternatively, the call raises `NotImplementedError` on an arch that cannot
consume the bias.

### Actual

`max |biased - plain|` is approximately 0, on an input where `rel_bias` is
random normal. The bias never reached the kernel.

### Measured, on the same box

Our float32 reference implementation of the documented Inkling relative
attention semantics is the oracle. Numbers from
`journal/remote/h100-session1.md`, sessions 3 and 4:

| Path | max abs error vs fp32 reference | mean abs error |
|---|---|---|
| `rel_bias=` on sm_90 | 0.90 to 1.63 across 3 parity cases | 0.02 to 0.06 |
| `score_mod=` on sm_90, same inputs | 7.8e-3 | n/a |

Timing on the same shapes confirms no bias work is performed:

| Case | `rel_bias=` path | plain attention, no bias |
|---|---|---|
| decode b1, 64K KV | 739.6 us | 742.7 us |
| decode b32, 64K KV | 747.1 us | 743.2 us |

For comparison, a correct native sm_90 bias implementation on the same box
costs 905.6 us at decode b1 64K KV, about 21 percent over plain attention
(`journal/u2-hopper-design.md`, session 24). A path that applies the bias
cannot be free.

### Note on SM8x and SM120

The missing-bias kernel is selected on those arches too, but with default tile
sizes the assertion `assert tile_n == 128` at `interface.py:673` fires first,
so those users get a loud failure rather than a silent one. Forcing
`tile_mn=(128, 128)` reaches the same silent path. Our measured
silent-wrong-output evidence is sm_90 only. The static defect, a kernel with
no bias code receiving a bias request, is common to all three.

## Suggested fix

Minimum, and enough to close the correctness hole: fail loudly. Raise
`NotImplementedError` in `_flash_attn_fwd` when `rel_bias is not None` and
`arch // 10 not in (10, 11)`, before the bias tensor is allocated at
`interface.py:682`. This also avoids the wasted `ShearingBias` launch.

Better: implement bias consumption on these arches. We have working
implementations and are happy to upstream them.

- Native sm_90 wgmma path, parity 3/3 on H100:
  `kernels/tml_fa4_modified/flash_fwd_sm90.py` at
  <https://github.com/RightNow-AI/inkling-turbo>. That file is the shipped
  implementation. `kernels/patches/u2_sm90_bias_port.py` in the same repo is a
  superseded attempt, kept for the journal, and is not the thing to read.
- Generic SM80-family path, used for sm_80 and sm_120, parity 3/3:
  `kernels/tml_fa4_modified/flash_fwd.py` and
  `kernels/patches/u2_v0_generic_bias.py`
- Interface wiring for both:
  `kernels/tml_fa4_modified/interface.py`

Design notes and the failure history behind those kernels are in
`journal/u2-hopper-design.md` in the same repository.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check was run
against this tracker before filing. A human submitter reviewed the report and
will review and defend every line of any follow-up PR.
