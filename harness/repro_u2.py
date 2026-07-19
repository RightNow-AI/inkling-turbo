import torch, traceback
from vllm.third_party.tml_fa4 import flash_attn_func, flash_attn_varlen_func

T = 128
# Non-varlen batch form: (b, s, h, d)
q = torch.randn(1, T, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(1, T, 1, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(1, T, 1, 128, dtype=torch.bfloat16, device="cuda")
rb = torch.randn(1, T, 8, 1024, dtype=torch.bfloat16, device="cuda")
try:
    out = flash_attn_func(q, k, v, rel_bias=rb, softmax_scale=1 / 128,
                          causal=True)
    if isinstance(out, tuple):
        out = out[0]
    print("BATCH FORM RAN OK", out.shape)

    # reference check
    ref_s = torch.einsum("bihd,bjhd->bhij", q.float(), k.float()) / 128
    ii = torch.arange(T).view(-1, 1)
    jj = torch.arange(T).view(1, -1)
    dist = ii - jj
    valid = (dist >= 0) & (dist < 1024)
    d_idx = dist.clamp(0, 1023).to(q.device)
    bias = rb[0].float().permute(1, 0, 2).gather(
        2, d_idx.unsqueeze(0).expand(8, T, T)) * valid.to(q.device)
    ref_s = ref_s + bias.unsqueeze(0)
    ref_s.masked_fill_(~((dist >= 0).to(q.device)), float("-inf"))
    ref = torch.einsum("bhij,bjhd->bihd", torch.softmax(ref_s, -1), v.float())
    diff = (out.float() - ref).abs()
    print(f"max_diff={diff.max().item():.4e} mean={diff.mean().item():.4e}")
except Exception:
    traceback.print_exc()
