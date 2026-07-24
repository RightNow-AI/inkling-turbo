# Generic SM80-family forward path is broken on sm_120: three defects that no H100 or B200 CI job can reach

**Target tracker:** vllm-project/flash-attention for defects 1 and 2.
Defect 3 is in vllm-project/tml-fa4 and is marked as such below.

**Severity:** medium. Every RTX 50-series (sm_120) user hits these on the first
call. Nothing on sm_90 or sm_100 is affected.

## Affected versions

| Component | Pin | Where the pin lives |
|---|---|---|
| vllm-project/flash-attention | `caaa4eb59845388a20b1f435ecaafb4bd9517ad8` | `cmake/external_projects/vllm_flash_attn.cmake:42` in the vLLM tree |
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `cmake/external_projects/tml_fa4.cmake:17` in the vLLM tree |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `requirements/cuda.txt:28` in the vLLM tree |

Verified on RTX 5090 Laptop (sm_120), WSL2 Ubuntu, torch 2.11.0+cu130,
nvidia-cutlass-dsl 4.6.0.

## Why CI never sees this

sm_90 and sm_100 have dedicated kernel files. The SM80-family kernel in
`flash_fwd.py` is only reached on sm_80 and sm_120. CI on H100 and B200 never
compiles it, so these defects sat latent. Stock vLLM Inkling attention is
unusable on sm_120 until all three are fixed.

The three failures are sequential. Each one has to be fixed before the next is
visible, which is why they are reported together.

## 1. `mDynamicCausal` is read in the kernel but never threaded through

**Repo: vllm-project/flash-attention.** File
`vllm_flash_attn/cute/flash_fwd.py`.

The `@cute.kernel` body reads `mDynamicCausal` (our notes place it near line
840 in the wheel copy, confirm against current source). The tensor is only a
parameter of the launcher. It is absent from the `@cute.kernel` signature and
absent from the `.launch()` argument list.

Result on any generic-path call:

```
NameError: ... name 'mDynamicCausal' is not defined
```

This is a DSL staging error, raised during tracing, so it fires before any
kernel runs.

**Fix:** add `mDynamicCausal: Optional[cute.Tensor] = None` to the
`@cute.kernel` signature and pass it at the `.launch()` call site. Two lines.
The exact anchors we patch against are in
`scripts/apply_local_sm120_fixes.sh` at
<https://github.com/RightNow-AI/inkling-turbo>.

## 2. `FlashAttentionForwardSm120` never sets `is_split_kv`

**Repo: vllm-project/flash-attention.** File
`vllm_flash_attn/cute/flash_fwd_sm120.py`.

The sm_120 shim subclasses the SM80-family class and sets `self.arch =
Arch.sm_80`, but never sets `self.is_split_kv`. The shared epilogue in
`flash_fwd.py` reads it. `AttributeError` at first call.

**Fix:** default it to `False` in the shim constructor.

Note for the reviewer: the tml-fa4 copy of this file does not have this defect.
Its sm_120 shim carries no constructor at all, and its interface asserts
split-KV off for sm_120. The defect is specific to the vllm-flash-attention
copy.

## 3. Generic path declares `pack_gqa` without packing, and turns on a TMA output path it never configures

**Repo: vllm-project/tml-fa4 @ `13374f0c`.** Both halves verified by reading
the pinned tree.

### 3a. `pack_gqa=True` reaches the generic kernel, which never packs anything

`_pack_gqa_heuristic` at `interface.py:209-231` returns `qhead_per_kvhead > 1`
for forward-only use, and `requires_grad` is hard-set to `False` at
`interface.py:432`. So `pack_gqa` is `True` by default for any GQA model. It is
passed to the generic constructors at `interface.py:1125` (sm_80) and
`interface.py:1251` (sm_120).

`flash_fwd.py` then consumes `self.pack_gqa` in the mask, block info and
epilogue helper selection, at `:363`, `:421`, `:693`, `:787`, `:999` and
`:1221`. It never calls `pack_gqa_layout` on `mQ`, `mO` or `mLSE`. A grep for
`pack_gqa_layout` in `flash_fwd.py` returns zero matches, against
`flash_fwd_sm90.py:253-258` and `flash_fwd_sm100.py:860-870` which both do the
folding.

The tensors stay unpacked while the surrounding logic assumes packed geometry.
The result is a rank mismatch in the epilogue:

```
'cute.slice' ... weakly congruent
```

### 3b. `use_tma_O` is enabled on sm_120 and `tma_atom_O` is always `None`

`flash_fwd_sm120.py:17` sets the class attribute `arch = 80`, with the comment
"Keep arch = 80 to use CpAsync code paths (no TMA for output)".

`FlashAttentionForwardBase.__init__` overwrites it. `flash_fwd.py:110`:

```python
self.arch = BaseDSL._get_dsl().get_arch_enum()
```

The instance attribute shadows the class attribute with the real runtime arch,
so the shim's stated intent is silently defeated. `flash_fwd.py:653` then does:

```python
self.use_tma_O = self.arch >= Arch.sm_90
```

which is `True` on sm_120. The epilogue takes the TMA branch at
`flash_fwd.py:391-400`, and the single `self.epilogue(...)` call site at
`flash_fwd.py:1062-1076` passes `None` for `tma_atom_O` (`:1070`). The
SM80-family kernel never builds that atom. `flash_fwd.py:385` also flips
`ragged` on, changing the `mO` offsetting for varlen, which produces 2D-view
slice errors on the ragged branch.

**Fix for 3a:** force `pack_gqa=False` for the SM80-family constructors until
packing is implemented for that kernel, or implement `pack_gqa_layout` folding
in `flash_fwd.py.__call__` the way sm_90 does.

**Fix for 3b:** set `use_tma_O = False` unconditionally in the SM80-family
`__call__`. That kernel has no TMA output path to select. Alternatively, stop
`__init__` from clobbering an arch a subclass deliberately pinned.

## Reproduction

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "nvidia-cutlass-dsl[cu13]==4.6.0" torch==2.11.0
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
.venv/bin/python repro_sm120_generic.py
```

```python
# repro_sm120_generic.py, run on an sm_120 GPU
import torch
from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

T, HQ, HKV, D = 1536, 8, 2, 128
dev = "cuda"
q = torch.randn(T, HQ, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device=dev)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)

out = flash_attn_varlen_func(
    q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
    max_seqlen_q=T, max_seqlen_k=T, softmax_scale=1 / D, causal=True,
)
print(out.shape)
```

Swap the import to `from flash_attn.cute import flash_attn_varlen_func` against
tml-fa4 @ `13374f0c` to reach defect 3.

### Expected

The call returns attention output of shape `(1536, 8, 128)`.

### Actual

`NameError` on `mDynamicCausal`. After fixing that, `AttributeError` on
`is_split_kv`. Against the tml-fa4 copy, the `cute.slice` congruence error and
the TMA-O failures.

### After the fixes

Parity against a float32 reference implementation of Inkling relative
attention, `harness/parity_fa4_rel.py`, 3 of 3 green on RTX 5090:

| Case | max abs diff | mean abs diff |
|---|---|---|
| global_short, T=128, 8q/1kv, hd128, ext1024 | 1.5625e-02 | 2.02e-04 |
| global_beyond_extent, T=1536, ext1024 | 7.8125e-03 | 6.81e-05 |
| swa_512, T=1536, 8q/2kv, window 511 | 7.8125e-03 | 7.99e-05 |

Tolerance 2e-2 for bfloat16 accumulation.

## Suggested fix

All three fixes, applied idempotently, are in
`scripts/apply_local_sm120_fixes.sh` at
<https://github.com/RightNow-AI/inkling-turbo>. The tml-fa4 side of defect 3 is
also carried in `kernels/tml_fa4_modified/flash_fwd.py` and
`kernels/tml_fa4_modified/interface.py`, where the `use_tma_O` line and the
`pack_gqa=False` constructor arguments both carry inline comments pointing at
this issue.

We also suggest a CI job that runs one varlen forward on an sm_120 device, or a
build-only staging test that traces the SM80-family kernel. Defects 1 and 2 are
staging errors and would be caught without any GPU of that class.

## Disclosure

This report was prepared with AI assistance. Per the vLLM contribution policy
in `AGENTS.md`, this is stated up front. The duplicate-work check was run
against both trackers before filing. A human submitter reviewed the report and
will review and defend every line of any follow-up PR.
