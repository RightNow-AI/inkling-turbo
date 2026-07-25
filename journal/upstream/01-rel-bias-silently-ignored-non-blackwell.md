# `rel_bias` is accepted, the shear pre-kernel runs, and the bias never reaches the forward kernel on pre-Blackwell arches

> **SCOPE CORRECTED 2026-07-26, after two independent adversarial verification passes. Read this before quoting the title.**
>
> The earlier title said "silently dropped on every non-Blackwell arch". That is wrong, and a maintainer would correct it in one reply. There are two `assert` statements inside the `rel_bias` block itself, at `interface.py:672-673`:
>
> ```python
> assert tile_m == 128
> assert tile_n == 128
> ```
>
> They run on the already-resolved tile config, so the behaviour splits by architecture and head dimension:
>
> | arch | default tile | behaviour with `rel_bias` |
> |---|---|---|
> | `sm_80` | 128x64 unconditionally (`interface.py:520`) | **raises `AssertionError`.** Never silent. The public API does not expose `tile_mn`, so it cannot be reached at all. |
> | `sm_120`, head_dim > 64 | 128x64 (`:518`) | **raises `AssertionError`.** Inkling's head_dim is 128, so this is Inkling's case. |
> | `sm_120`, head_dim <= 64 | 128x128 (`:516`) | **silent**, bias dropped |
> | `sm_90`, head_dim 97-128 | 128x128 (`:148`) | **silent**, bias dropped. head_dim 128 is the common case, so this is the real one. |
> | `sm_90`, head_dim <= 96 | 192x128 (`:136`) | raises `AssertionError` |
> | `sm_100` / `sm_11x` | n/a | **correct**, bias is applied |
>
> So the silent-wrong-answer path is **`sm_90` at head_dim 97-128**, plus `sm_120` at head_dim <= 64. Elsewhere the failure is loud. Note also that the correct path is `arch // 10 in (10, 11)`, which is **`sm_100` and `sm_11x`**, not sm_100 alone.
>
> **The arch breakdown above is read from source. The silent drop itself was MEASURED on sm_90**, in session 1 on a real H100 at head_dim 128, which is precisely the case the table calls silent. Two independent signatures, both in `journal/remote/h100-session1.md`, **journal-only** class:
>
> | | |
> |---|---|
> | `rel_bias=` max abs error vs a float32 reference | **0.90 to 1.63** |
> | `score_mod=` on the same inputs | 7.8e-3 |
> | `rel_bias=` cost, batch 1 at 64K KV | **739.6 us** |
> | bias-free attention, same shape | **742.7 us** |
>
> The first says the returned tensor is not a biased attention output. The second says no bias work happened inside the timed region, since the biased call costs what the bias-free call costs. Those are exactly the two fingerprints of a dropped bias, and together they are stronger than either alone.
>
> What is **not** reproducible today: stock tml-fa4 no longer imports under nvidia-cutlass-dsl 4.6.0 (`utils.py:279`, `cute.core.ThrMma` was removed), so the 2026-07-26 verification passes could not re-execute it, and no Hopper was available to them. Quote the session-1 numbers as journal-only and offer to re-run on request.

**Target tracker:** vllm-project/tml-fa4 (issues enabled, verified 2026-07-25)
**Severity:** highest of this series. Silent numerical wrongness, no error, no warning.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` in the vLLM tree |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` in the vLLM tree |

**Also verified live at tml-fa4 `main`.** Re-checked 2026-07-25 against
`b206834606ed5b5f21f8eed6b0683f528ea9cf7d`, which is current `main` and is also
the tml-fa4 pin on vLLM `main`. Every line number below is valid at **both**
commits: `flash_attn/cute/interface.py`, `flash_fwd_sm90.py` and
`flash_fwd_sm120.py` are byte-identical between the two, because the only change
between them, PR #3, did not touch those files. The `grep -ci bias` counts below
are also identical at both. So this report is not describing a stale pin.

Affected code paths: the bias staging block runs for `arch // 10 in (8, 9, 12)`, but see the scope correction above for which of those are **silent** and which raise. The `arch // 10 in (10, 11)` path, covering sm_100 and sm_11x, is correct and is not affected.

## Scope, stated up front so this is not read as a live vLLM serving bug

This is a defect in tml-fa4's own public API contract, reachable by any direct
caller. It is **not**, today, reachable through vLLM's Inkling serving path on
sm_90.

vLLM PR #48858, "[Model] Add Hopper FA4 relative attention for Inkling", merged
2026-07-16, deliberately routes everything that is not Blackwell to `score_mod`
plus `aux_tensors`, and keeps the tml-fa4 sheared-bias path for
capability major 10 and 11 only. So vLLM does not pass `rel_bias` on sm_90.
Our own notes recorded that from the first day we measured it
(`journal/remote/h100-session1.md:56-57`, "NOT a vLLM production path (vLLM uses
score_mod on Hopper), but it accepts the input and returns wrong attention
silently").

The report stands on its own terms regardless: `flash_attn_varlen_func` is a
public entry point, it documents and accepts `rel_bias`, it does measurable work
to honour it, and then returns a result that does not contain it, on three of the
four architecture branches it dispatches (`arch // 10 == 8`, `== 9`, `in (10, 11)`,
`== 12`; all but the Blackwell branch). A caller has no way to find that out
except by having an oracle. We think that is worth a guard even though vLLM's own
router currently steps around it.

## Summary

`flash_attn_varlen_func(..., rel_bias=...)` on SM8x, SM90 and SM120 does all of
the following and then throws the result away:

1. allocates the padded sheared bias tensor (`interface.py:682-699`),
2. compiles and launches the `ShearingBias` pre-kernel (`interface.py:816-858`),
3. constructs the forward kernel with **no** bias argument,
4. returns plain, bias-free attention as if it were the requested result.

The bias tensor is even returned by the internal entry point
(`interface.py:1434`, `return out, lse, logits_max, bias, cu_total_m_blocks_bias,
blocks_to_batch_idx`) and then discarded by both public wrappers, which unpack it
into a throwaway: `out, lse, logits_max, *_ = _flash_attn_fwd(...)` at
`interface.py:1471` and `:1543`. Callers get plausible-looking output that is
numerically wrong.

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

**Self-contained.** The script below needs nothing from our repository: two
imports, `torch` and `flash_attn.cute`, and no helper files, no reference
implementation, no harness. Copy it into a file and run it.

**Preconditions, so it reaches the defect instead of an assert.** Needs an
sm_90 device and `head_dim = 128`, which is what the script uses, so that the
default config is `(tile_m, tile_n) = (128, 128)` and the two asserts at
`interface.py:672-673` pass. `rel_extent` must be a multiple of 128; the script
uses 1024. On sm_80, and on sm_120 above head_dim 64, add
`tile_mn=(128, 128)` to force past the tile assert, and see the
per-architecture table above for why.

**One caveat on item 2 of the mechanism, if you run this against the pinned
tree on a cu129 torch build.** You may hit an unrelated `fmax()` TypeError from
`flash_attn/cute/utils.py:352` before reaching this defect. That is a separate
issue, reported separately, and it is not part of this one. A cu130 build does
not take that branch.

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

### What we actually measured, and what we are inferring

Keeping these apart, because they are not the same strength of claim.

**Measured on 1x H100 SXM5 (sm_90).** A float32 reference implementation of the
documented Inkling relative attention semantics is the oracle. From
`journal/remote/h100-session1.md`, sessions 3 and 4:

| Path | max abs error vs fp32 reference | mean abs error |
|---|---|---|
| `rel_bias=` on sm_90 | 0.90 to 1.63 across 3 parity cases | 0.02 to 0.06 |
| `score_mod=` on sm_90, same inputs | 7.8e-3 | n/a |

The mean being far below the max means scattered wrong positions rather than a
uniform offset, which is what a missing additive bias term looks like.

**Measured on the same box.** Per-op timing of the `rel_bias=` path is
indistinguishable from bias-free attention:

| Case | `rel_bias=` path | plain attention, no bias |
|---|---|---|
| decode b1, 64K KV | 739.6 us | 742.7 us |
| decode b32, 64K KV | 747.1 us | 743.2 us |

For comparison, a correct native sm_90 bias implementation on the same box
costs 905.6 us at decode b1 64K KV, about 21 percent over plain attention
(`journal/u2-hopper-design.md`, session 24). A path that applies the bias
cannot be free. The `rel_bias=` path is free, to within noise.

**Inferred, not separately measured.** That `max |out_bias - out_plain|` is
approximately 0 is what the reproducer above will print, and it follows directly
from the static path: the kernel is constructed with no bias argument and
contains no bias code, so the bias cannot affect the output. We did not run that
exact two-call subtraction and print it. Our evidence is the oracle comparison
and the timing, both above. A maintainer running the reproducer gets the
subtraction as the one-line demonstration, and it is the cheapest way to see it,
which is why it is written that way. It is labelled here so nobody cites it as
one of our measurements.

### Observed versus inferred, per architecture

The static defect, a kernel with no bias code receiving a bias request, is
common to all three non-Blackwell families and is established by reading the
pinned tree. Whether a given user hits it **silently** depends on the default
tile size for their arch, because `assert tile_n == 128` at `interface.py:673`
guards the bias path and fires before anything else when it does not hold.

The bias path requires **both** `assert tile_m == 128` (`interface.py:672`) and
`assert tile_n == 128` (`:673`). Whichever fails, fails loudly. So the silent
case is exactly the case where the arch's default config is `(128, 128)`.

At `head_dim = 128`, which is Inkling's head dim and the shape we tested:

| Arch | Default `(tile_m, tile_n)` at head_dim 128 | Silent with default tiles? | Our evidence |
|---|---|---|---|
| sm_90 | `(128, 128)`, `interface.py:148` | **Yes** | **Observed.** Wrong output against an fp32 oracle on H100, numbers above. |
| sm_120 | `(128, 64)`, `interface.py:518` | No, `:673` fires | Inferred from the tree. Loud failure. |
| sm_80 | `(128, 64)`, `interface.py:520` | No, `:673` fires | Inferred from the tree. Loud failure. |

The head-dim dependence matters and is worth stating, because it means the loud
failure is not a reliable backstop:

- **sm_120 at `head_dim <= 64`** takes `interface.py:516`, which is
  `FwdConfig(128, 128, True, True)`. Both asserts pass and the silent path is
  reached with **no user action at all**.
- **sm_80** is `FwdConfig(128, 64, ...)` for every head_dim, so it is the one
  family that always fails loudly on defaults. Forcing `tile_mn=(128, 128)`
  reaches the silent path there too.
- **sm_90 below head_dim 96** actually fails loudly for the *other* reason:
  `_tile_size_fwd_sm90` returns `tile_m = 192` there (`interface.py:136`, `:144`,
  `:146`), so `:672` fires, not `:673`.

Two corrections to the earlier draft of this report, recorded rather than quietly
dropped. It claimed that on SM8x and SM120 "with default tile sizes the
assertion `assert tile_n == 128` fires first, so those users get a loud failure
rather than a silent one." That is right for sm_80, right for sm_120 at
head_dim > 64, and **wrong** for sm_120 at head_dim <= 64. It also named only the
`tile_n` assert, when on sm_90 at small head_dim it is the `tile_m` assert that
guards.

Our measured silent-wrong-output evidence is **sm_90 at head_dim 128 only**.
Everything said about sm_80 and sm_120 is read off the pinned source and is
labelled as such. We have no non-Blackwell hardware result for the silent path
other than H100.

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

## Duplicate-work check, run 2026-07-25

Commands actually executed, with their real results. `gh search issues` and
`gh search prs` do **not** accept `--state all`; omitting `--state` searches all
states, which is what these do.

```bash
gh repo view vllm-project/tml-fa4 --json hasIssuesEnabled
#   {"hasIssuesEnabled":true,"name":"tml-fa4"}   -> an issue is the right route

gh issue list --repo vllm-project/tml-fa4 --state all --limit 200
#   EMPTY. The tracker has zero issues, open or closed.

gh pr list --repo vllm-project/tml-fa4 --state all --limit 200
#   3 PRs total, none about rel_bias on non-Blackwell:
#     #3 Migrate deprecated CuTe DSL APIs for cutlass-dsl 4.6   MERGED 2026-07-17
#     #2 Fix forward argument handling on pre-Blackwell GPUs    MERGED 2026-07-16
#     #1 Add Blackwell plain FP8 attention support              CLOSED

gh search issues --repo vllm-project/tml-fa4 "rel_bias"      # empty
gh search issues --repo vllm-project/tml-fa4 "sheared bias"  # empty
gh search issues --repo vllm-project/tml-fa4 "ShearingBias"  # empty
gh search issues --repo vllm-project/tml-fa4 "silently"      # empty
gh search prs    --repo vllm-project/tml-fa4 "rel_bias"      # empty
gh search issues --repo vllm-project/vllm "rel_bias silently"       # empty
gh search issues --repo vllm-project/vllm "rel_bias non-Blackwell"  # empty
gh search issues --repo vllm-project/vllm "ShearingBias"            # empty
gh search issues --repo vllm-project/vllm "rel_bias NotImplementedError"  # empty
gh search prs    --repo vllm-project/flash-attention "rel_bias"     # empty
gh search prs    --repo vllm-project/flash-attention "sheared"      # empty
```

**Nothing covers this defect.** Two adjacent things were found and neither is a
duplicate, recorded so the filer is not surprised by them:

- **tml-fa4 PR #2**, merged, is the change that *created* the non-Blackwell
  `else` branches this report cites at `interface.py:1320-1341` and
  `:1397-1417`. Its diff adds those branches and passes no bias argument in
  either. It is upstream's most recent touch of this exact code and it did not
  add bias threading. That is context supporting the report, not a fix of it.
- **vLLM PR #48858**, merged 2026-07-16, routes non-Blackwell Inkling to
  `score_mod`, which is why this is not a live vLLM serving bug. See the scope
  section at the top.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check above was run
against this tracker on 2026-07-25, immediately before this report was marked
filing-ready. A human submitter reviewed the report and will review and defend
every line of any follow-up PR.
