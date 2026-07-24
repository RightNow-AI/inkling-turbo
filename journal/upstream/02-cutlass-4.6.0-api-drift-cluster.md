# [tml-fa4] Four incompatibilities with nvidia-cutlass-dsl 4.6.0 (vLLM's own pin) break every import/first-call of the FA4 path

**Repo:** vllm-project/tml-fa4 @ `13374f0c` (latest main `09d2482` unaffected
by fixes). vLLM pins `nvidia-cutlass-dsl[cu13]==4.6.0` (requirements/cuda.txt)
while vendoring tml-fa4 at this commit, the combination cannot run on any
arch we tested (sm_90 H100, sm_120 RTX 5090). Fixes for all four are one-line
mechanical changes; happy to PR.

## 1. `cute.core.ThrMma` / `cute.core.TiledMma` removed

4.6.0 moved both to `cutlass.cute` (defined in `cutlass/cute/atom.py`).
14 references across `utils.py`, `flash_fwd*.py` fail at import.
Fix: `cute.core.ThrMma` -> `cute.ThrMma`, same for TiledMma.

## 2. `cute.make_fragment` renamed `cute.make_rmem_tensor`

23 call sites (`utils.py` 8, `softmax.py` 5, `flash_fwd_sm100.py` 7
`flash_fwd.py` 2, `pack_gqa.py` 1). Same positional signature
`(layout_or_shape, dtype)`. Note the sm_100 count: the flagship
Blackwell sheared-bias path cannot run against the pinned DSL either.
Fix: mechanical rename (verified: parity-identical results after).

## 3. nvvm API branch keyed to CUDA version instead of DSL version

`utils.py:352` (`fmax`) and `utils.py:496` (`atomic_add_i32`) select the
old "explicit result type" nvvm binding when `CUDA_VERSION == 12.9`. The
binding signature actually tracks the nvidia-cutlass-dsl version: 4.6.0
always uses the new form (2 positional args). On cu129 torch (Lambda's
default H100 stack) the old branch fires against the new binding:
`TypeError: fmax() takes 2 positional arguments but 3 ... were given`
on every sm_90 attention call.
Fix: key the branch off the DSL version (or drop the old branch when
requiring >=4.6.0).

## 4. Raw-int window args unmarshalable

The generic-family compile/exec call sites pass `window_size_left/right`
as Python ints; the DSL requires `Int32 | None`:
`[Internal Error] ... expects argument #13 (window_size_left) to be one of
(Int32, NoneType), but got <class 'int'>`. Only local/SWA mode trips it
(global passes None). Fix: wrap with `Int32(...)` at the call sites
(the sm_90 path already does this at interface.py:351-352).

## Environment

torch 2.11.0+cu129 and 2.11.0+cu130, nvidia-cutlass-dsl 4.6.0
python 3.12, H100 SXM5 (sm_90) and RTX 5090 (sm_120). All four verified
independently; sequence of discovery + minimal repros in the linked logs.

Found during Inkling-turbo kernel work (open-source vLLM Inkling serving
kernels). A patch script applying all four fixes idempotently is available.
