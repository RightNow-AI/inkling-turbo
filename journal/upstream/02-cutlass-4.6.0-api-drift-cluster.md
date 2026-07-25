# PARTLY SUPERSEDED. Four incompatibilities with nvidia-cutlass-dsl 4.6.0 stop the FA4 path at import or first call on every arch

> ## STATUS, 2026-07-25: items 1 and 2 are FIXED UPSTREAM. File items 3 and 4 only.
>
> The duplicate check for this report had never been run against tml-fa4's
> **pull requests**, only its issues. Run on 2026-07-25, it found the fix.
>
> **tml-fa4 PR #3, "Migrate deprecated CuTe DSL APIs for cutlass-dsl 4.6", by
> mgoin, MERGED 2026-07-17** as commit `b206834606`.
> <https://github.com/vllm-project/tml-fa4/pull/3>
>
> It is our items 1 and 2, and the match is exact, not approximate. Its six
> touched files and their edit counts reconcile line for line with the two
> tables in this report:
>
> | File | PR #3 edits | This report: item 1 (`ThrMma`) | item 2 (`make_fragment`) | Sum |
> |---|---|---|---|---|
> | `flash_fwd_sm100.py` | 18 | 11 | 7 | 18 |
> | `utils.py` | 10 | 2 | 8 | 10 |
> | `softmax.py` | 5 | 0 | 5 | 5 |
> | `flash_fwd.py` | 2 | 0 | 2 | 2 |
> | `block_sparse_utils.py` | 1 | 1 | 0 | 1 |
> | `pack_gqa.py` | 1 | 0 | 1 | 1 |
>
> Verified fixed at `main`: `grep -rn "cute.core.ThrMma\|cute.core.TiledMma\|cute.make_fragment("`
> over `flash_attn/cute` at `b206834606` returns nothing.
>
> **And vLLM has already taken the bump.** vLLM PR #48988, "[Bugfix] Bump
> tml-fa4 for cutlass-dsl 4.6 API compatibility", MERGED 2026-07-18. vLLM `main`
> now pins `b206834606ed5b5f21f8eed6b0683f528ea9cf7d` at
> `cmake/external_projects/tml_fa4.cmake:17`. Our build base pins `13374f0c`,
> which predates it.
>
> ### WITHDRAWN, not edited away
>
> The severity line and the framing paragraph of this report said "Nothing in
> the package runs against the DSL version vLLM pins" and "the combination that
> ships is the combination that does not work." **Both are withdrawn as of
> 2026-07-25.** They were true of our build base and are false of vLLM `main`.
> Filing this report as written would assert a broken state that upstream fixed
> eight days before, and it would be a fair criticism of us. Items 1 and 2 must
> not be filed in any form.
>
> ### What is still live, re-verified at `main` (`b206834606`)
>
> | Item | Status | Evidence at `main` |
> |---|---|---|
> | 1, `cute.core.ThrMma` / `TiledMma` | **FIXED by PR #3.** Do not file. | grep returns nothing |
> | 2, `cute.make_fragment` | **FIXED by PR #3.** Do not file. | grep returns nothing |
> | 3, nvvm branch keyed to CUDA version | **STILL LIVE. Fileable.** | `utils.py:352` and `:496` still test `CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9` |
> | 4, raw Python `int` window args | **STILL LIVE. Fileable.** | PR #3 did not touch `interface.py`; `flash_fwd.py` still has no `Int32` normalization in `__call__`, only `Optional[Int32]` annotations at `:629` and `:753` |
>
> Item 3's line numbers shifted by PR #3's edits to the same file: the `fmax`
> definition is at `utils.py:346` at our pin and `:349` at `main`, while the
> branch itself is at `:352` in both. Item 4's anchors are unchanged.
>
> ### Strong prior art for item 3, which the filer should lead with
>
> The sibling fork already made exactly this fix. **vllm-project/flash-attention
> PR #157, "Fix compatibility with nvidia-cutlass-dsl 4.6.0", MERGED
> 2026-07-13**, includes the line "fmax: drop the CUDA 12.9 branch, use the new
> NVVM API unconditionally (mirrors Dao-AILab/flash-attention#2648)". So item 3
> is not a matter of opinion about how to key the branch. The same organisation
> resolved it the same way in the neighbouring repository, and tml-fa4 simply
> did not receive that half of the migration. That makes item 3 a short,
> uncontroversial filing.
>
> **Rewrite before filing.** Retitle to name two items, not four. Renumber them
> 1 and 2, or keep the numbering and mark the first two struck, but do not
> present a four-item report. The "Each fix exposes the next one" argument for
> bundling no longer applies, since the first two exposures are already merged.

**Target tracker:** vllm-project/tml-fa4
**Severity, as originally written and now withdrawn:** ~~high. Nothing in the
package runs against the DSL version vLLM pins.~~ Revised: medium, for the two
surviving items. Item 3 breaks every sm_90 attention call on a cu129 stack.
Item 4 breaks local and SWA mode on the SM80-family kernel.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` in the vLLM tree |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` in the vLLM tree |

tml-fa4's own `pyproject.toml:24` declares `nvidia-cutlass-dsl>=4.4.2` with no
upper bound.

~~4.6.0 satisfies that constraint and the package cannot run against it. vLLM
vendors this exact commit while pinning exactly that DSL version, so the
combination that ships is the combination that does not work.~~
**Withdrawn 2026-07-25, see the status block at the top.** True of our build base
`13374f0c`, false of tml-fa4 `main`, where PR #3 fixed the API drift, and false
of vLLM `main`, which bumped the pin in its PR #48988.

**The dependency floor is worth a sentence to the filer, though, because PR #3
left it inconsistent in the opposite direction.** `pyproject.toml:24` still reads
`nvidia-cutlass-dsl>=4.4.2` at `main` (`b206834606`), unchanged. But `main` now
calls `cute.make_rmem_tensor` and `cute.ThrMma`, which are 4.6.0 APIs. So a
resolver honouring the declared floor may install 4.4.2 and the package will
fail on the new names. The sibling fork handled this: flash-attention PR #157
bumped its floor to `nvidia-cutlass-dsl>=4.6.0` in the same change that did the
renames. tml-fa4's PR #3 did the renames without the floor bump. This is a
one-line addition and it fits naturally alongside surviving items 3 and 4.

Verified on 1x H100 SXM5 (sm_90) and RTX 5090 Laptop (sm_120), torch
2.11.0+cu129 and 2.11.0+cu130, python 3.12.

All four fixes are mechanical. We are happy to open the PR.

## 1. `cute.core.ThrMma` and `cute.core.TiledMma` no longer exist

4.6.0 exposes both on `cutlass.cute` directly. `ThrMma` is defined in
`cutlass/cute/atom.py` in the installed 4.6.0 package.

14 textual references remain on the old path in tml-fa4 @ `13374f0c`:

| File | References |
|---|---|
| `flash_attn/cute/flash_fwd_sm100.py` | 11, of which `:4251` is a docstring |
| `flash_attn/cute/utils.py` | 2, at `:279` and `:288` |
| `flash_attn/cute/block_sparse_utils.py` | 1, at `:736` |

None of these files use `from __future__ import annotations`, so the 13 live
references are function parameter annotations evaluated at module import.
The failure is at import, not at call.

**Fix:** `cute.core.ThrMma` becomes `cute.ThrMma`, same for `TiledMma`.

## 2. `cute.make_fragment` was renamed `cute.make_rmem_tensor`

Same positional signature `(layout_or_shape, dtype)`.

23 call sites in tml-fa4 @ `13374f0c`:

| File | Call sites |
|---|---|
| `flash_attn/cute/utils.py` | 8 |
| `flash_attn/cute/flash_fwd_sm100.py` | 7 |
| `flash_attn/cute/softmax.py` | 5 |
| `flash_attn/cute/flash_fwd.py` | 2 |
| `flash_attn/cute/pack_gqa.py` | 1 |

Note the sm_100 count. The flagship Blackwell sheared-bias kernel is in that
list, so the day-0 Blackwell path also cannot run against the pinned DSL.

**Fix:** mechanical rename. We verified parity-identical results after the
rename on sm_120 and sm_90.

## 3. The nvvm API branch is keyed to the CUDA version instead of the DSL version

`flash_attn/cute/utils.py`:

- `fmax`, defined at `:346`, branches at `:352` on
  `CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9`.
- `atomic_add_i32`, defined at `:492`, branches at `:496` on the same test.

The comment on both branches says "NVVM call based on nvvm version", but the
binding signature tracks the nvidia-cutlass-dsl version, not the CUDA version.
4.6.0 always uses the new form with two positional arguments. On a cu129 torch
build, which is Lambda's default H100 stack, the old branch is taken against
the new binding and every sm_90 attention call dies with:

```
TypeError: fmax() takes 2 positional arguments but 3 ... were given
```

A cu130 stack never takes the branch, which is why this only appears on some
machines.

**Fix:** key the branch off the DSL version, or delete the old branch given
the `>=4.4.2` floor already excludes nothing relevant here.

## 4. Raw Python `int` window arguments are unmarshalable

The generic-family compile and exec call sites pass `window_size_left` and
`window_size_right` straight through as Python `int`. The DSL requires
`Int32 | None`:

```
[Internal Error] ... expects argument #13 (window_size_left) to be one of
(Int32, NoneType), but got <class 'int'>
```

Only local and SWA mode trips this. Global mode passes `None` and marshals
fine.

The dedicated kernels already normalize this inside their own `__call__`:

- `flash_attn/cute/flash_fwd_sm90.py:351-352`
- `flash_attn/cute/flash_fwd_sm100.py:1264-1265`

`flash_attn/cute/flash_fwd.py`, the SM80-family kernel used by sm_80 and
sm_120, has no equivalent normalization.

**Fix:** either add the same two lines to the generic `__call__`, or wrap with
`Int32(...)` at the interface call sites. We took the interface-side route,
which also covers the exec path.

## Reproduction

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "nvidia-cutlass-dsl[cu13]==4.6.0" torch==2.11.0
# tml-fa4 checked out at 13374f0c, installed editable

# Item 1 and 2: import time
.venv/bin/python -c "from flash_attn.cute import flash_attn_varlen_func"

# Items 3 and 4: first call, on a cu129 torch build for item 3
.venv/bin/python repro_first_call.py
```

```python
# repro_first_call.py
import torch
from flash_attn.cute import flash_attn_varlen_func

T, HQ, HKV, D = 1536, 8, 2, 128
dev = "cuda"
q = torch.randn(T, HQ, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device=dev)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)

flash_attn_varlen_func(
    q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=T, max_seqlen_k=T, softmax_scale=1 / D,
    causal=True, window_size=(511, 0),  # local mode, needed for item 4
)
```

### Expected

Import succeeds. The call returns attention output.

### Actual

- Import raises `AttributeError` on `cute.core.ThrMma`.
- After fixing that, import raises `AttributeError` on `cute.make_fragment`.
- After fixing that, on a cu129 stack the call raises the `fmax()` `TypeError`
  above.
- After fixing that, the local-mode call raises the `window_size_left`
  marshaling error above.

Each fix exposes the next one. That ordering is why all four are reported
together rather than as separate issues.

## Suggested fix

All four fixes, applied idempotently, are in
`scripts/apply_local_sm120_fixes.sh` at
<https://github.com/RightNow-AI/inkling-turbo>. The resulting sources are
`kernels/tml_fa4_modified/` in the same repository. The interface-side `Int32`
wrapping for item 4 is in `kernels/tml_fa4_modified/interface.py`.

Recommended upstream shape:

1. Items 1 and 2 as one mechanical rename commit.
2. Item 3 as a one-line branch-condition change.
3. Item 4 as a two-line normalization in `flash_fwd.py.__call__`, matching what
   `flash_fwd_sm90.py:351-352` already does.
4. Add a CI job that imports and runs one varlen call, global and local mode,
   against the pinned DSL version. All four of these are caught by that job.

## Open item for the filer, CLOSED 2026-07-25

~~Our notes record that tml-fa4 main at `09d2482` does not fix item 1. We did not
re-verify the other three against that commit. Re-check all four against
current main before filing and adjust the text.~~

**Done.** The re-check was run, and the note above was reading the wrong commit.
`09d2482` is the merge of PR #2, not the head of main. The relevant commit is
`b206834606`, the merge of PR #3, which is one commit later and is current main.
Result of checking all four there:

- Items 1 and 2: **fixed**, by PR #3. Withdrawn from this report.
- Items 3 and 4: **still live**. Fileable.

For the record, our pin `13374f0c` is dated 2026-07-16T02:08:50Z and is PR #2's
branch commit; main's merge of it is `09d2482ed3` at 02:13:31Z. So our pin sits
between PR #2 and PR #3, which is exactly why it carries items 1 and 2.

## Duplicate-work check, run 2026-07-25

```bash
gh pr list --repo vllm-project/tml-fa4 --state all --limit 200
#   #3 Migrate deprecated CuTe DSL APIs for cutlass-dsl 4.6  MERGED 2026-07-17  <-- items 1 and 2
#   #2 Fix forward argument handling on pre-Blackwell GPUs   MERGED 2026-07-16
#   #1 Add Blackwell plain FP8 attention support             CLOSED
gh pr view 3 --repo vllm-project/tml-fa4
gh issue list --repo vllm-project/tml-fa4 --state all --limit 200   # EMPTY, zero issues
gh search prs --repo vllm-project/tml-fa4 "cutlass-dsl 4.6"  # finds #3
gh search issues --repo vllm-project/tml-fa4 "make_fragment" # empty
gh search issues --repo vllm-project/tml-fa4 "ThrMma"        # empty
gh search prs --repo vllm-project/tml-fa4 "fmax"             # empty  -> item 3 not covered
gh search prs --repo vllm-project/tml-fa4 "window_size"      # empty  -> item 4 not covered
```

Note for whoever maintains these checklists: the keyword queries this repository
had written down for this report were `make_rmem_tensor` and `make_fragment` and
`ThrMma`, run through `gh search issues`. None of them would have found PR #3,
because it is a PR and because its title and body use neither
`make_rmem_tensor` nor `ThrMma` in a form those queries match. `gh pr list
--state all` found it immediately. Listings beat keywords.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check above was run
on 2026-07-25 and it changed this report substantially: two of its four items
were withdrawn as already fixed upstream. A human submitter reviewed the report
and will review and defend every line of any follow-up PR.
