import torch, traceback
from vllm.third_party.tml_fa4 import flash_attn_varlen_func

T = 1536
q = torch.randn(T, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(T, 2, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(T, 2, 128, dtype=torch.bfloat16, device="cuda")
rb = torch.randn(T, 8, 512, dtype=torch.bfloat16, device="cuda")
cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
try:
    flash_attn_varlen_func(
        q=q, k=k, v=v, rel_bias=rb, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T, softmax_scale=1 / 128, causal=True,
        window_size=(511, 0))
    print("SWA RAN OK")
except Exception:
    traceback.print_exc()
