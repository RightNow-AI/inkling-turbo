# FILING.md, the order to file in and what to say

Written 2026-07-25, after the duplicate check was run properly for the first
time. Supersedes the filing order that used to live in `00-INDEX.md`.

You file these under your own name. Nothing below overstates, and where we report
a defect we report it as a measurement with a reproducer, never as a claim about
a contributor. If a paragraph below feels like it is asking you to defend
something you cannot defend, do not file that one.

## What actually gets filed

Six defects, four new issues, one comment on somebody else's PR. Down from ten
defects and five issues, because the duplicate check found four of the ten
already fixed or already reported upstream.

| Order | Item | Tracker | Type | Source document |
|---|---|---|---|---|
| 1 | `rel_bias` accepted and silently dropped on non-Blackwell | vllm-project/tml-fa4 | **Issue** | `01-rel-bias-silently-ignored-non-blackwell.md` |
| 2 | Hardware confirmation of `mDynamicCausal` on sm_120 | vllm-project/flash-attention | **Comment on PR #156** | `03-...md` status block, `journal/regression-sm120-varlen-illegal-address.md:84-102` |
| 3 | Two surviving cutlass-dsl 4.6.0 items, nvvm branch keyed to CUDA version, and unmarshalable window ints | vllm-project/tml-fa4 | **Issue**, needs a rewrite to two items first | `02-cutlass-4.6.0-api-drift-cluster.md` |
| 4 | Inkling has no attention path on SM8x | vllm-project/vllm | **Issue** | `05-no-sm8x-attention-path.md` |
| 5 | `pack_gqa` row semantics, plus generic-path `pack_gqa` and `use_tma_O` | vllm-project/tml-fa4 | **Issue**, optionally merged with 04 | `04-pack-gqa-row-semantics.md` and `03-...md` defect 3 |

### Retired, do not file

| Item | Why |
|---|---|
| Report 03 defect 1, `mDynamicCausal` NameError | Duplicate of flash-attention PR #156, open since 2026-06-30 |
| Report 03 defect 2, `is_split_kv` never set | Duplicate of the same PR, whose fix site is better than ours |
| Report 02 item 1, `cute.core.ThrMma` | Fixed by tml-fa4 PR #3, merged 2026-07-17 |
| Report 02 item 2, `cute.make_fragment` | Fixed by the same PR |

### Tracker routing, verified 2026-07-25

| Tracker | Issues enabled | So |
|---|---|---|
| vllm-project/tml-fa4 | yes | issues are fine |
| vllm-project/vllm | yes | issues are fine |
| vllm-project/flash-attention | **no** | **PR or PR comment only.** Nothing can be filed as an issue here. |

## 1. `rel_bias` silently dropped, to vllm-project/tml-fa4, as an issue

File this first. It is the only item in the series where shipped code returns
wrong numbers with no error, and it is the one whose evidence is strongest.

> `flash_attn_varlen_func` accepts a `rel_bias` argument on every architecture,
> and on sm_80, sm_90 and sm_120 it allocates the padded bias tensor, compiles and
> launches the `ShearingBias` pre-kernel, then constructs the forward kernel with
> no bias argument and returns plain attention. The three non-Blackwell forward
> kernels contain no bias code at all: `grep -ci bias` returns 0 in
> `flash_fwd_sm90.py`, `flash_fwd.py` and `flash_fwd_sm120.py`, against 236 in
> `flash_fwd_sm100.py`. On an H100 we measured the `rel_bias=` path at 0.90 to
> 1.63 max absolute error against a float32 reference, while `score_mod=` on the
> same inputs gives 7.8e-3, and the `rel_bias=` path costs the same as bias-free
> attention to within noise, 739.6 us against 742.7 us at batch 1 with 64K KV.
> A minimal fix is to raise `NotImplementedError` before the bias tensor is
> allocated. We also have working bias implementations for sm_90 and for the
> SM80-family kernel and are happy to upstream them. Verified at the pinned
> commit `13374f0c` and at current `main` `b206834606`; the relevant files are
> byte-identical between the two.

Say explicitly, early, that this is not a live vLLM serving bug, because vLLM PR
#48858 routes non-Blackwell Inkling to `score_mod`. Volunteering that is what
makes the rest of the report credible, and a maintainer will find it in thirty
seconds anyway. The report's scope section is written to be pasted.

## 2. Hardware confirmation on flash-attention PR #156, as a comment

Not a report. A comment on somebody else's open PR, adding evidence.

> Confirming this PR's second defect on two further architectures, both times as
> the control arm of an unrelated investigation rather than a search for this bug.
>
> On an A100-SXM4-40GB (sm_80), driver 580.95.05, torch 2.11.0+cu130, CPython
> 3.12, every no-bias call through the generic SM80-family path fails at
> `flash_attn/cute/flash_fwd.py:840`:
>
> ```
> psc = mDynamicCausal[batch_size] if const_expr(mDynamicCausal is not None) else None
> NameError: name 'mDynamicCausal' is not defined
> ```
>
> Reproduced on `decode_b1_plain_kv64k` and `decode_b32_plain_kv64k`, batch 1 and
> batch 32 against 65536 keys, head dim 128. Same shapes with a `rel_bias`
> argument take a different path and do not hit it.
>
> Also on an RTX 5090 Laptop (sm_120), WSL2, torch 2.11.0+cu130, where the message
> reads `NameError: cannot access local variable 'mDynamicCausal'` instead. Newer
> CPython words the same error differently, which is worth knowing for anyone
> searching for it. Both defects are still present at `main` `ed4b7342`, which is
> also the tml-fa4 pin currently used by vLLM.

Three things to be careful about here:

- **Do not restate the diagnosis as if it were ours.** tgmerritt found and
  diagnosed both defects on 2026-06-30. We ran into the second one 25 days later
  by accident. The comment adds two architectures and a Python version, nothing
  more.
- **The sm_120 session's JSON artifacts are zero bytes.** Do not quote any number
  from that session. The error string is a categorical observation and is safe.
  Anything numeric from session 29 is not currently backed, see the artifact gap
  note in `00-INDEX.md`.
- **The A100 evidence is backed and is the one to lead with.** Added 2026-07-25
  from session 31, `journal/remote/validate_a100x1_s31/`, which has a committed
  non-zero artifact, an asserted compute capability and a full environment
  record. The full traceback is in `run.log` in that directory.

## 3. The two surviving cutlass-dsl 4.6.0 items, to vllm-project/tml-fa4

**Rewrite report 02 before filing it.** As written it claims four defects and two
of them were fixed upstream eight days before. Retitle it for two items, renumber
or strike the first two visibly, and drop the "nothing in the package runs"
framing, which is withdrawn.

> Two smaller cutlass-dsl 4.6.0 compatibility items appear to have been missed by
> PR #3. First, `flash_attn/cute/utils.py` branches on the CUDA version rather
> than the DSL version in two places, `fmax` at `:352` and `atomic_add_i32` at
> `:496`, both testing `CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9`. The
> binding signature tracks the nvidia-cutlass-dsl version, so on a cu129 torch
> build, which is a common H100 stack, the old branch is taken against the new
> binding and every sm_90 attention call fails with
> `TypeError: fmax() takes 2 positional arguments but 3 were given`. A cu130 build
> never takes the branch, which is why it only shows up on some machines. The
> sibling fork already resolved this the same way in
> vllm-project/flash-attention PR #157, "drop the CUDA 12.9 branch, use the new
> NVVM API unconditionally". Second, the SM80-family compile and exec paths pass
> `window_size_left` and `window_size_right` as raw Python `int`, which the DSL
> rejects with `expects argument #13 (window_size_left) to be one of (Int32,
> NoneType), but got <class 'int'>`. Only local and SWA mode trips this, since
> global mode passes `None`. `flash_fwd_sm90.py:351-352` and
> `flash_fwd_sm100.py:1264-1265` already normalize this in their own `__call__`;
> `flash_fwd.py` has no equivalent. Related, `pyproject.toml:24` still declares
> `nvidia-cutlass-dsl>=4.4.2` while `main` now calls 4.6.0-only APIs, so the
> declared floor no longer matches what the code needs. Both items verified live
> at `main` `b206834606`.

Leading with "PR #3 appears to have missed these two" is accurate, is generous,
and is the correct read: PR #3 was a rename migration and these are the two
non-rename parts of the same migration. Do not frame it as PR #3 being wrong.

## 4. No attention path on SM8x, to vllm-project/vllm, as an issue

> On capability (8, x), `vllm/models/inkling/nvidia/ops/fa4_rel_attention.py`
> routes to `score_mod` at `:133-143`, because `_use_sheared_bias()` at `:19-22`
> is true only for capability major 10 and 11. The cute backend then refuses:
> `flash_attn/cute/interface.py:722` in the pinned vllm-flash-attention raises
> `NotImplementedError("Custom user-provided score_mod is not supported on SM8x
> architectures.")`. There is no third path, and no minimum capability is declared
> for the model, so on an 8x A100-SXM4-40GB node the failure arrives at the first
> attention call, after the weights are resident and the KV cache is allocated,
> rather than at load. Two options, not mutually exclusive: declare a minimum
> capability and fail at init with a message that names the arch, or give SM8x a
> sheared-bias route, which needs bias consumption in the SM80-family kernel. We
> have an implementation of the latter with 3 of 3 parity on full-prefill shapes on
> A100; decode and chunked prefill on sm_80 have no correctness result from us on
> any hardware. Also note `interface.py:520` in tml-fa4 still reads
> `FwdConfig(128, 64, True, True)  # SM80, should tune`.

Three cautions specific to this one:

- **Do not carry our withdrawn A100 decode percentages into it.** They were timed
  against a kernel with a `seqlen_q == seqlen_k` specialisation in its bias reader.
  The report already marks them withdrawn. The prefill rows and the `tile_n=128`
  collapse stand.
- **Our sm_80 support claim is narrower than it reads.** There is an open,
  unresolved fault in our own generic kernel on multi-sequence varlen batches,
  `journal/regression-sm120-varlen-illegal-address.md`, expected to affect A100
  too and unconfirmed there. If you offer our kernel as option 2, say that the
  parity result is single-sequence.
- **Mention PR #48858 as context.** It added the routing for SM90, SM100, SM110
  and SM120 and did not cover SM8x. That framing is accurate and reads as a gap
  rather than an oversight.

## 5. `pack_gqa` row semantics and the generic-path defects, to vllm-project/tml-fa4

Lowest urgency, file last. Two ways to do it, your call:

- **As one issue**, report 04 plus report 03's defect 3, on the theme "the
  SM80-family kernel takes `pack_gqa` and does not pack, and the row contract is
  undocumented". They are the same subsystem and the same root cause shape.
- **As two**, keeping 04 as a docs-and-assertion request and defect 3 as a
  concrete bug. Defect 3 is the more actionable half.

> `pack_gqa` folds `qhead_per_kvhead` q-heads into the seqlen mode, so a 128-row
> score tile stops being 128 sequence positions. Nothing in the kernel API states
> that, and it is on by default for any GQA forward call via
> `_pack_gqa_heuristic` at `interface.py:209-231` with `requires_grad` hard-set
> false at `:432`. Any row-indexed feature is silently wrong unless it also packs.
> Three places already handle it for the bias feature and none is documented as a
> rule: `flash_fwd_sm100.py:868`, `shearing_bias.py:110-115`, and the row-index
> scaling at `shearing_bias.py:180` and elsewhere. Separately, the SM80-family
> kernel receives `pack_gqa=True` from `interface.py:1125` and `:1251` but never
> calls `pack_gqa_layout` on `mQ`, `mO` or `mLSE`; a grep for `pack_gqa_layout` in
> `flash_fwd.py` returns zero matches against `flash_fwd_sm90.py:255-258` and
> `flash_fwd_sm100.py:861-864`, which both fold. And `flash_fwd.py:110` overwrites
> the `arch = 80` that `flash_fwd_sm120.py:17` deliberately pins, so
> `flash_fwd.py:653` sets `use_tma_O = True` on sm_120 while the only
> `self.epilogue(...)` call site passes `None` for `tma_atom_O`. A one-sentence
> note in the `pack_gqa_layout` docstring would have saved us a great deal of
> time. Verified at the pin and at `main` `b206834606`, where `interface.py` is
> byte-identical.

Report 04 is honest that it has no crash to reproduce and that its observable
needs a row-indexed bias kernel on sm_90, which only exists in our tree. Keep
that honesty. File it as a documentation and assertion request, which is what it
is, and let defect 3 carry the reproducible part.

## Things to do in every filing

- State the AI-assistance disclosure. Every source document already carries one.
  The vLLM policy requires it and the trackers we are filing to are vLLM's.
- Give the pin **and** the `main` commit you verified against. For items 1, 4 and
  5 we verified both and the files are identical, so say so. It is the cheapest
  way to show the report is not about a stale tree.
- One issue per defect cluster. Do not bundle across trackers.
- Link the reproducer inline in the issue body rather than linking to our repo for
  it. Items 1 and 4 have reproducers that need nothing from us.
- Offer the implementation, do not lead with it. The defect is the contribution;
  the kernel is an offer.

## Things not to do

- Do not file anything to vllm-project/flash-attention as an issue. It is not
  possible; issues are disabled.
- Do not file report 03 defects 1 or 2 in any form. PR #156 has them.
- Do not file report 02 items 1 or 2 in any form. PR #3 fixed them.
- Do not quote any number from `journal/remote/local_sm120_s29/`. Those artifacts
  are zero bytes.
- Do not describe any of these as vLLM shipping wrong output to users unless the
  specific claim survives the routing in PR #48858. For report 01 it does not,
  and the report says so.
