# `pack_gqa` redefines what a score-tile row means, and nothing in the kernel API surfaces that to row-indexed features

**Target tracker:** vllm-project/tml-fa4
**Severity:** medium. Not a defect in shipped behavior. It is a contract hazard
that costs a new contributor a very long time, and the existing handling is
implicit in three separate places.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` in the vLLM tree |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` in the vLLM tree |

Evidence gathered on 1x H100 SXM5 (sm_90), torch 2.11.0+cu129, and RTX 5090
Laptop (sm_120), torch 2.11.0+cu130.

## Summary

`pack_gqa` folds `qhead_per_kvhead` q-heads into the seqlen mode. From the
`pack_gqa_layout` docstring at `flash_attn/cute/pack_gqa.py:15-30`:

```
(seqlen_q, headdim, nheads, batch, ...)
    -> ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...)
```

The consequence for the MMA accumulator is not stated anywhere. A 128-row
score tile stops being 128 sequence positions. With `qhead_per_kvhead = 8` it
becomes 16 sequence positions times 8 q-heads. Any feature that indexes a
per-row tensor by tile row is silently wrong from that point on, unless it also
packs.

`pack_gqa` is not something a caller opts into. `_pack_gqa_heuristic` at
`interface.py:209-231` returns `qhead_per_kvhead > 1` for forward-only use, and
`requires_grad` is hard-set to `False` at `interface.py:432`. So packing is on
by default for every GQA model, decided at `interface.py:481-488`.

## Where the interaction is already handled, implicitly

The relative-bias feature is row-indexed. Its 128-row shear contract and its
per-head slice both assume rows are sequence rows. Three separate places make
it work under packing, and none of them is documented as a rule:

1. `flash_fwd_sm100.py:868` packs the bias tensor alongside Q, O and LSE:
   `mBias = pack_gqa_layout(mBias, self.qhead_per_kvhead, nheads_kv, head_idx=2)`
2. `shearing_bias.py:110-115` packs both `mPreBias` and `mBias` in the shear
   writer.
3. The shear writer then scales every row index by
   `qhead_per_kvhead_packgqa`, at `shearing_bias.py:180`, `:298-303`, `:348`,
   `:399` and `:505`.

`_group_tile_bias` at `interface.py:246-248` looks like part of this handling
but is not. Its `128 * qhead_per_kvhead_packgqa` expression is commented out
and it returns a constant 128.

A new row-indexed feature on sm_90 or on the SM80-family kernel gets none of
this. There is no assertion, no flag, and no doc note. The kernel constructors
take `pack_gqa` and say nothing about what it does to rows.

## How this presented, and the cost

We spent 17 debugging sessions on a native sm_90 sheared-bias kernel. Every
bias addressing scheme we tried, linear indexing, `reshape_acc_to_mn`
coordinates, `make_tiled_copy_C`, and `partition_C`, was correct or nearly
correct on unpacked geometry and unfixable on packed geometry.

The diagnostic that ended it was printing the strides of the partitioned
source fragment. The wgmma submode that steps 8 tile rows had stride 81920.
At the anchor shape, 64 q-heads and `rel_extent` 1024, the padded bias extent
is `1024 + 256 = 1280` (`interface.py:670`), so:

```
64 heads * 1280 = 81920 = exactly one sequence row of the bias tensor
```

A step of 8 tile rows was landing one sequence row away. That is the packing,
visible in a stride. Full account in `journal/u2-hopper-design.md`, sessions
23 and 24, at <https://github.com/RightNow-AI/inkling-turbo>.

## Reproduction

There is no crash to reproduce. The observable is the parity flip.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "nvidia-cutlass-dsl[cu13]==4.6.0" torch==2.11.0
# tml-fa4 checked out at 13374f0c with a row-indexed bias kernel on sm_90
.venv/bin/python harness/parity_fa4_rel.py --tol 2e-2
```

The harness is `harness/parity_fa4_rel.py` at
<https://github.com/RightNow-AI/inkling-turbo>. It checks Inkling relative
attention against a float32 PyTorch reference on three cases.

### Expected

A row-indexed bias kernel that is correct on one arch stays correct when the
same math is ported to another arch with the same tile shape.

### Actual

The kernel is correct on sm_120, where the generic path never packs, and wrong
on sm_90, where packing is on by default for the same call. Nothing in the API
signals the difference.

With `pack_gqa` forced off for `arch // 10 == 9` and `rel_bias is not None`,
parity on H100 goes 3 of 3 green, max abs diff 1.56e-2, 7.8e-3, 7.8e-3 on the
three cases, tolerance 2e-2 for bfloat16.

## Root cause

`pack_gqa` is an internal layout optimization that changes the meaning of a
row in the MMA accumulator. It is auto-enabled by a heuristic, is not part of
the documented public contract, and there is no mechanism that forces a
row-indexed feature to declare whether it is pack-aware. The bias feature
survives only because the sm_100 path and the shear writer each remembered to
pack, independently.

## Suggested fix

Any of these would have saved the debugging cost. They are ordered cheapest
first.

1. **Docs.** State the row contract in the `pack_gqa_layout` docstring at
   `pack_gqa.py:15` and in the kernel constructor docstrings. One sentence:
   under `pack_gqa`, a score-tile row is `(q_head, seq_position)`, not a
   sequence position, so any per-row tensor must be packed with the same
   `pack_gqa_layout` call.
2. **Assertion.** Have the kernel constructors reject the combination of
   `pack_gqa=True` and a row-indexed feature that has not been packed, rather
   than producing wrong numbers. In `flash_fwd_sm90.py` and `flash_fwd.py`
   this is a constructor-time check.
3. **Reference implementation.** Our working sm_90 sheared-bias kernel, which
   uses `thr_mma_qk.partition_C(gBias_tile)` to partition the sheared tile with
   the same partitioner that produced `acc_S`, so element `i` of the
   accumulator pairs with element `i` of the bias by construction. It is at
   `kernels/tml_fa4_modified/flash_fwd_sm90.py` and
   `kernels/patches/u2_sm90_bias_port.py` at
   <https://github.com/RightNow-AI/inkling-turbo>.

Our interface currently takes the conservative route: `pack_gqa = False` for
`arch // 10 == 9` with `rel_bias`, in
`kernels/tml_fa4_modified/interface.py`, with a comment recording why.
Packed-bias addressing on sm_90 is a performance item we have not done, and it
is exactly the item the sm_100 path already solves.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check was run
against this tracker before filing. A human submitter reviewed the report and
will review and defend every line of any follow-up PR.
