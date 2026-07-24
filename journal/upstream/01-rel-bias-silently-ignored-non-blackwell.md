# [tml-fa4] `rel_bias` silently ignored on every non-Blackwell arch, wrong attention output returned without error

**Repo:** vllm-project/tml-fa4 @ `13374f0c` (vLLM fork-base `850295881` pin)
**Severity: highest of this series, silent numerical correctness trap.**

## Summary

`flash_attn_varlen_func(..., rel_bias=...)` accepts the bias tensor on sm_80
sm_90, and sm_120, allocates the padded sheared tensor, launches the
`ShearingBias` pre-kernel, then constructs the forward kernel WITHOUT any
bias argument and returns plain (bias-free) attention as if it were the
requested result. No error, no warning.

## Evidence

- `FlashAttentionForwardSm90` constructor call (interface.py ~1137-1160)
  receives no bias-related argument; compare the Sm100 branch which passes
  `has_bias=bias is not None` (~980). `flash_fwd_sm90.py` contains zero
  bias-handling code (grep "bias" -> only the softmax import line).
- Same for the SM80-family path (`flash_fwd.py`), used by sm_80/sm_120.
- Measured on 1x H100 SXM5 (torch 2.11.0+cu129, nvidia-cutlass-dsl 4.6.0):
  output of the rel_bias call is bit-close to a plain-attention run
  (timings identical within noise: 747us vs 743us @ 64K KV decode) and
  diverges from a bias-applying fp32 reference by max ~0.9-1.6.

## Expected

Either implement bias consumption on these arches or raise
`NotImplementedError` at the interface when `rel_bias is not None` and
`arch // 10 not in (10, 11)`. Silent wrong output is the worst outcome:
downstream users (e.g. anyone calling tml-fa4 directly for Inkling-style
relative attention on Hopper) get plausible-looking garbage.

## Repro

```python
import torch
from flash_attn.cute import flash_attn_varlen_func  # tml-fa4 @13374f0c
T = 128
q = torch.randn(T, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(T, 1, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(T, 1, 128, dtype=torch.bfloat16, device="cuda")
rb = torch.randn(T, 8, 1024, dtype=torch.bfloat16, device="cuda")
cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
out_bias = flash_attn_varlen_func(q=q, k=k, v=v, rel_bias=rb
    cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T
    softmax_scale=1/128, causal=True)
out_plain = flash_attn_varlen_func(q=q, k=k, v=v
    cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T
    softmax_scale=1/128, causal=True)
print((out_bias - out_plain).abs().max())  # ~0 on sm_90: bias was dropped
```

Found during Inkling-turbo kernel work (open-source vLLM Inkling serving
kernels); a working sm_90/sm_120 bias implementation is available and can be
upstreamed on request.
