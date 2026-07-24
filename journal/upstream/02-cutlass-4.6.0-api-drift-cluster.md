# Four incompatibilities with nvidia-cutlass-dsl 4.6.0 stop the FA4 path at import or first call on every arch

**Target tracker:** vllm-project/tml-fa4
**Severity:** high. Nothing in the package runs against the DSL version vLLM pins.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` in the vLLM tree |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` in the vLLM tree |

tml-fa4's own `pyproject.toml:24` declares `nvidia-cutlass-dsl>=4.4.2` with no
upper bound. 4.6.0 satisfies that constraint and the package cannot run against
it. vLLM vendors this exact commit while pinning exactly that DSL version, so
the combination that ships is the combination that does not work.

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

## Open item for the filer

Our notes record that tml-fa4 main at `09d2482` does not fix item 1. We did not
re-verify the other three against that commit. Re-check all four against
current main before filing and adjust the text.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check was run
against this tracker before filing. A human submitter reviewed the report and
will review and defend every line of any follow-up PR.
