"""Minimal repro: does the generic kernel fault on multi-sequence varlen?

Isolates three variables against the failing case (T_q = [256, 137, 64], no ctx):
  1 sequence vs 3 sequences
  with rel_bias vs without rel_bias
so the output says which combination faults rather than that something did.

Each probe runs in its own subprocess, because cudaErrorIllegalAddress poisons
the CUDA context and every later probe in the same process would fault as a
cascade regardless of its own inputs.
"""
import os
import subprocess
import sys

PROBE = r'''
import sys, torch
from vllm.third_party.tml_fa4 import flash_attn_varlen_func
D = 128
dev = "cuda"
lens = [int(x) for x in sys.argv[1].split(",")]
use_bias = sys.argv[2] == "bias"
Hq, Hkv, ext = 8, 1, 512
total = sum(lens)
torch.manual_seed(7)
q = torch.randn(total, Hq, D, dtype=torch.bfloat16, device=dev) / (D ** 0.25)
k = torch.randn(total, Hkv, D, dtype=torch.bfloat16, device=dev) / (D ** 0.25)
v = torch.randn(total, Hkv, D, dtype=torch.bfloat16, device=dev)
cu = [0]
for L in lens:
    cu.append(cu[-1] + L)
cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
kw = dict(q=q, k=k, v=v, cu_seqlens_q=cu_t, cu_seqlens_k=cu_t,
          max_seqlen_q=max(lens), max_seqlen_k=max(lens),
          softmax_scale=1.0 / D, causal=True)
if use_bias:
    r = torch.randn(total, Hq, 16, dtype=torch.bfloat16, device=dev) * 0.4
    proj = torch.randn(16, ext, dtype=torch.bfloat16, device=dev) * 0.3
    kw["rel_bias"] = (r.float() @ proj.float()).to(torch.bfloat16)
out = flash_attn_varlen_func(**kw)
if isinstance(out, tuple):
    out = out[0]
torch.cuda.synchronize()
print("OK  finite=%s" % bool(torch.isfinite(out.float()).all()))
'''


def probe(lens, mode):
    r = subprocess.run(
        ["./.venv/bin/python", "-c", PROBE, lens, mode],
        capture_output=True, text=True, cwd="/home/jaber/inkling-turbo/vllm",
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1",
             "PYTHONPATH": "/home/jaber/inkling-turbo/gates"},
        timeout=900,
    )
    tail = (r.stdout + r.stderr).strip().splitlines()
    msg = "OK" if r.returncode == 0 else "FAULT"
    detail = ""
    for ln in tail:
        if "IllegalAddress" in ln or "illegal memory" in ln:
            detail = "cudaErrorIllegalAddress"
            break
        if "Error" in ln or "assert" in ln.lower():
            detail = ln.strip()[:80]
    if r.returncode == 0 and tail:
        detail = tail[-1]
    return msg, detail


print(f"{'sequences':>26s} {'bias':>6s}  result")
print("-" * 62)
for lens, label in [("512", "1 seq, 512"),
                    ("256", "1 seq, 256"),
                    ("137", "1 seq, 137 (not 128-mult)"),
                    ("256,137,64", "3 seqs, 256/137/64"),
                    ("256,256", "2 seqs, 256/256"),
                    ("128,128", "2 seqs, 128/128")]:
    for mode in ("nobias", "bias"):
        m, d = probe(lens, mode)
        print(f"{label:>26s} {mode:>6s}  {m:<6s} {d[:46]}")
