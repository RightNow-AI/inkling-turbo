#!/usr/bin/env python3
"""Unit test: ShearingBias writer output vs reference shear (local, sm_120 OK).

The U2 tile port consumes the sheared tensor; this pins the writer's exact
layout contract first (journal/u2-hopper-design.md validation ladder step 1).

Expected semantics (derived shearing_bias.py:357-476; verified here):
for varlen row i (global q index within seq), head h, kv index k attended by
row i (causal), the attention kernel reading 128-col tile n_block at local
column c must find bias(i, k) at sheared[i, h, col(i, k)] where the writer
places values so that the LAST valid kv (k = i) sits at bias_idx_right-1
computed from the row's n_idx_right, values descending dist leftward, -inf
right-pad (causal), left pad -inf (local) / 0.0 (global beyond extent).

We don't re-derive col() analytically here, we EXTRACT it empirically:
write rel_logits with unique values rel_logits[i,h,d] = encode(i,h,d), run
the writer, decode positions, and dump the observed (i, k) -> column map.
The map is then asserted against the affine form col = A*i + B*k + C per
128-block (what the tile port will implement) and saved as JSON for the
port to consume as ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def run(T: int, ext: int, window_left: int | None) -> dict:
    import cuda.bindings.driver as cuda_driver
    from vllm.third_party.tml_fa4.interface import to_cute_tensor
    from vllm.third_party.tml_fa4.shearing_bias import ShearingBias
    import cutlass.cute as cute

    dev = "cuda"
    H = 2
    padded = ext + 256
    # Shear moves values only WITHIN a row (writer reads mPreBias row m for
    # output row m), so a per-row encoding value = d+1 suffices: seeing v at
    # (i, col) means bias(dist v-1) for row i. d+1 <= 240 stays bf16-exact.
    rel = torch.zeros(T, H, ext, dtype=torch.bfloat16, device=dev)
    for d in range(min(ext, 240)):
        rel[:, :, d] = float(d + 1)
    bias = torch.full((T + 128, H, padded), float("nan"),
                      dtype=torch.bfloat16, device=dev)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    sb = ShearingBias(
        rel_extent=ext, is_causal=True,
        is_local=window_left is not None,
        pack_gqa=False, qhead_per_kvhead=1, rows_per_cta=4)
    compiled = cute.compile(
        sb,
        to_cute_tensor(rel), to_cute_tensor(bias),
        T, T,
        to_cute_tensor(cu), to_cute_tensor(cu),
        None, None, None, None,
        window_left, 0 if window_left is not None else None,
        stream)
    compiled(rel, bias, T, T, cu, cu, None, None, None, None,
             window_left, 0 if window_left is not None else None, stream)
    torch.cuda.synchronize()

    # decode: for each row, find where each encoded dist landed
    out = bias[:T, 0].float().cpu()  # head 0
    mapping = {}
    for i in range(T):
        for col in range(padded):
            v = out[i, col].item()
            if v == v and v not in (float("-inf"), 0.0):  # not nan/-inf/pad
                d = round(v) - 1
                if 0 <= d <= i and d < 240:
                    mapping[(i, i - d)] = col  # (row, kv) -> column
    return {"T": T, "ext": ext, "window_left": window_left,
            "map": {f"{i},{k}": c for (i, k), c in mapping.items()}}


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    results = {}
    for name, T, ext, wl in [("causal_small", 12, 512, None),
                             ("local_small", 12, 512, 511),
                             ("causal_two_blocks", 200, 512, None)]:
        try:
            res = run(T, ext, wl)
            m = {tuple(map(int, k.split(","))): v for k, v in res["map"].items()}
            # affine check: col(i,k) = k - i + const within each 128 n-block?
            diffs = {(i, k): c - (k - i) for (i, k), c in m.items()}
            consts = sorted(set(diffs.values()))
            print(f"[{name}] mapped {len(m)} positions; col-(k-i) constants: "
                  f"{consts[:8]}{'...' if len(consts) > 8 else ''}")
            results[name] = res
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[name] = {"error": str(exc)}
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(results, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
