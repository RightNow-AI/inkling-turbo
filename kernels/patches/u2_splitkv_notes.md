# U2 sm_90 split-KV decode: implementation notes

Status: **written, NOT validated on silicon.** No GPU was available to the
author. Every claim below about behaviour on H100 is a design argument, not a
measurement. Nothing here may be quoted as a result until the validation
sequence at the bottom has been run and the JSON artifact lands in
`journal/remote/`.

Files touched:
- `kernels/tml_fa4_modified/flash_fwd_sm90.py`
- `kernels/tml_fa4_modified/interface.py`

## Why

`journal/u2-hopper-design.md`, SESSION 24 ncu section, on H100 SXM5:

```
decode_b1_kv64k : DRAM 7.2%, MemSOL 20.3%, SM 31.3%, occupancy 14.1%, 242 GB/s
                  grid = heads x batch = 64 CTAs on 132 SMs
decode_b32      : profiles identically (32 q rows over ONE kv sequence -> same grid)
```

The kernel is parallelism-starved, not bandwidth-starved. Half the GPU is
idle. Splitting the KV range across CTAs is the only lever that changes the
grid. This is extending a lead, not repairing a loss: our decode is already
2.73x (b1 64K) and 2.80x (b32 64K) faster than the day-0 score_mod path
(`journal/remote/microbench_attn_day0_session25_h100.json` vs
`journal/remote/microbench_attn_scoremod_session25_h100.json`).

## The correctness question, answered

**Does the sheared bias apply compose across splits? YES. Verified by reading
the code, stated explicitly as required.**

The bias apply in `apply_rel_bias_sm90` is driven by exactly two quantities,
both computed in `mma()` per work tile:

```python
bias_tile_shift = padded_bias // self.tile_n - (128 * (m_block + 1)) // self.tile_n
bias_num_tiles  = padded_bias // self.tile_n
tile_idx        = n_block + bias_tile_shift
```

`padded_bias` is `mBias_cur.shape[1]`, a property of the bias tensor.
`self.tile_n` is a constexpr. `m_block` comes from the work tile. `n_block` is
the KV block index.

Neither expression references `split_idx`, `num_splits`, `n_block_min`,
`n_block_max`, or `seqlen_k`. So for a fixed `m_block`, **every split computes
the identical `bias_tile_shift`**, and for a fixed `(m_block, n_block)` pair
the same gmem bias tile is selected regardless of which split executes it.

Two supporting facts that this rests on:

1. `BlockInfo.get_n_block_min_max` returns split ranges in **absolute** KV
   block coordinates (`n_block_min = base_min + split_idx * per_split`). The
   loader already uses those same values to index `gK`/`gV`/the page table, so
   if they were split-relative the KV loads themselves would be wrong. They
   are absolute.
2. `gBias_tiles = cute.local_tile(mBias_cur, (tile_m, tile_n), (m_block, None))`
   leaves the n-tile mode free and is indexed by `tile_idx`. It is a function
   of `m_block` only.

Mathematically: the bias is an additive per-(row, key) term applied to the
logits before softmax. Its contribution to a logit does not depend on which
other keys are present in the tile. Partitioning the key range therefore
partitions the softmax denominator exactly, which is precisely what the
combine kernel reconstructs from the per-split `(m, l)` state. The
decomposition is exact, not approximate.

Also composes: `softmax_scale` folding. With bias, `compute_softmax_scale_log2`
strips the scale out of the softmax and `apply_rel_bias_sm90` applies
`acc_S * softmax_scale + bias`. That is per-tile arithmetic, identical in every
split, and the emitted `lse` is in natural log units in every split.

## What was built

Minimal port of the sm_100 mechanism. No new combine kernel; the existing
`FlashAttentionForwardCombine` is reused unchanged.

1. **`FlashAttentionForwardSm90(is_split_kv=...)`**, a new ctor flag.

2. **Layout trick that avoids forking the epilogue.** The partial tensors carry
   a *leading* split mode. `__call__` transposes it to the *trailing* mode:

   ```
   dense  O   (splits, b, s, h, dv)   -> (s, dv, h, b, splits)
   varlen O   (splits, total_q, h, dv)-> (total_q, dv, h, splits)
   dense  LSE (splits, b, h, s)       -> (s, h, b, splits)
   varlen LSE (splits, h, total_q)    -> (total_q, h, splits)
   ```

   Dropping the last mode (`mO[..., split_idx]`) then reproduces the non-split
   rank *and* layout exactly, so `mma()` hands `epilogue()` tensors it cannot
   distinguish from the non-split case and `epilogue()` is reused verbatim.
   `pack_gqa_layout` is rank-agnostic, so packing composes with this.

3. **Scheduler.** `TileSchedulerArguments` now carries the real `num_splits` and
   `is_split_kv`. All three schedulers sm_90 can select (`SingleTileScheduler`,
   `SingleTileLPTScheduler`, `SingleTileVarlenScheduler`) already implement
   split unpacking upstream; nothing new was written. The `_` in
   `m_block, head_idx, batch_idx, _ = work_tile.tile_idx` is now bound.

4. **`BlockInfo(is_split_kv=..., num_splits=...)`.** `num_splits` is recomputed
   device-side from `mO.shape`, deliberately, so the non-split kernel signature
   does not change.

5. **Empty splits.** `num_splits` need not divide the block count, so a split
   can own an empty range. Producer and consumer both compute
   `has_work = n_block_min < n_block_max` from the same inputs and both skip:
   the producer issues no K/V loads, the consumer consumes none, so the
   pipelines stay in lockstep. The consumer instead does
   `softmax.reset(); acc_O.fill(0.0)`, which makes `finalize()` emit
   `lse = -inf` and `O = 0`. The combine kernel already handles that
   (`flash_fwd_combine.py:688,695-697` explicitly guard `-inf` and the
   all-`-inf` case). The Q load and the Q pipeline handshake stay *outside* the
   guard on both sides, so phases never drift.

6. **Negative trip counts.** With splits, a split whose whole range sits left of
   the causal boundary would give `n_block_max - n_block_min_causal_local_mask
   < 0`. Both boundaries are now clamped with `cutlass.min(..., n_block_max)`
   under `const_expr(self.is_split_kv)`. The clamp only moves the
   masked/unmasked boundary; the algebra
   `1 + (nbm - B1) + (B1 - B2) + (B2 - n_block_min) = n_block_max - n_block_min`
   shows the total block count, and therefore K/V pipeline consumption, is
   unchanged.

7. **`use_tma_O = False` under split.** The O TMA descriptor is built host-side
   from the whole tensor, but a partial is addressed with a device-side split
   index. The epilogue's predicated gmem store path is used instead.

8. **interface.py**: the `assert not is_split_kv, "SplitKV not supported on SM 9.0"`
   is gone; `is_split_kv=is_split_kv` is passed through; the sm_90 execution
   call now passes `out_partial`/`lse_partial` (the *compile* call already did).

## Deviation from upstream you must know about: bf16 partials

Upstream (sm_100, FA3 C++) stores `out_partial` in **fp32**. On sm_90 this port
stores it in the **kernel's element type** (bf16 for Inkling).

Reason: the sm_90 epilogue stages O through a `self.dtype`-typed smem tile that
*aliases sQ*. An fp32 partial needs 2x the bytes, so it would need either a
second 64 KB O buffer or a forked register-to-gmem epilogue. Either is a real
change to proven code, and I have no GPU to prove it on. The combine kernel is
dtype-generic (`can_implement` accepts fp16/bf16/fp32 for `dtype_partial`, and
`_flash_attn_fwd_combine` derives it from `out_partial.dtype`), so bf16
partials are legal end to end.

Numerical cost, stated honestly and **unmeasured**: each partial O is already
row-normalized, so all splits carry comparable magnitudes and the combine is a
convex combination. The added error is roughly one extra bf16 rounding
(~2^-9 relative) on top of the bf16 rounding the non-split path already pays,
i.e. expect on the order of 2x the output error of the non-split kernel, not an
order of magnitude. **This is an estimate from the format, not a measurement.**
If the parity gate is tight, this is the first thing to suspect, and the fix is
the fp32-partial epilogue, not the split logic.

## Deliberately blocked on sm_90 + split

Asserted in `interface.py` rather than silently allowed:
- `learnable_sink`: an empty split resets `row_max` to `-inf`, and the sink
  term would then evaluate `exp2(sink*log2e - (-inf))` = `+inf`.
- `return_logits_max`: the sm_90 kernel never writes a row-max tensor, so
  `logits_max_partial` would be uninitialized garbage fed to the combine.
- dynamic per-batch splits: `disable_scheduler_metadata` is forced on, because
  the sm_90 kernel ignores `num_splits_dynamic_ptr`; leaving it live would let
  the forward kernel and the combine kernel disagree on how many splits exist.

Already blocked upstream and left alone: block sparsity + split
(`NotImplementedError`), and the `U2_SM90_GENERIC=1` reference route
(`assert not is_split_kv, "SplitKV not supported on generic path"`).

Untested combination, use only after checking: `pack_gqa=True` + split. The
layout work is rank-agnostic and should compose, but the bias path already
forces `pack_gqa=False` on sm_90 (session 24 root cause), so the first
validation will not exercise it.

## Opt-in, by design

`num_splits` still defaults to 1 in `_flash_attn_fwd`/`flash_attn_func`. Nothing
splits unless the caller asks (`num_splits=N`, or `num_splits=-1` to use
`num_splits_heuristic`). The default path is byte-identical to HEAD. Flipping
the default is a separate decision that needs measured curves behind it.
