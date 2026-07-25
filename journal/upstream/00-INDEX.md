# Upstream findings index (Inkling-turbo)

10 defects across the vLLM day-0 Inkling stack, found while building
open-source replacement kernels. Every technical claim traces to a journal entry
or to a file and line in the pinned trees.

**Status as of 2026-07-25: the duplicate check was finally run properly, and it
changed the picture.** Four of the ten defects are already fixed or already
reported upstream. What remains is 6 fileable defects, which package into 4
issues plus 1 comment on an existing PR. See [FILING.md](FILING.md) for the
order, the trackers, and the one-paragraph framing for each.

Before 2026-07-25 this file said "FINALIZED, NOT FILED", which was wrong in a
specific way worth naming: the checks it told the filer to run **could not
execute**, so "finalized" rested on a step that had never happened. Fixed below.

## The five documents, current status

| # | File | Target tracker | Status 2026-07-25 |
|---|---|---|---|
| 01 | [01-rel-bias-silently-ignored-non-blackwell.md](01-rel-bias-silently-ignored-non-blackwell.md) | vllm-project/tml-fa4 | **FILE FIRST.** Dup check empty. Verified live at pin and at `main`. Scope narrowed: not a live vLLM serving bug, see the report. |
| 02 | [02-cutlass-4.6.0-api-drift-cluster.md](02-cutlass-4.6.0-api-drift-cluster.md) | vllm-project/tml-fa4 | **PARTLY SUPERSEDED.** Items 1 and 2 fixed by merged tml-fa4 PR #3; withdrawn. Items 3 and 4 still live and fileable. Needs a rewrite to 2 items before filing. |
| 03 | [03-vllm-flash-attn-generic-path-bugs.md](03-vllm-flash-attn-generic-path-bugs.md) | was vllm-project/flash-attention | **RETIRED AS A DUPLICATE.** Defects 1 and 2 duplicate flash-attention PR #156. That tracker has issues disabled. Defect 3 survives, targets tml-fa4, still fileable. |
| 04 | [04-pack-gqa-row-semantics.md](04-pack-gqa-row-semantics.md) | vllm-project/tml-fa4 | **FILEABLE AS WRITTEN.** Dup check empty. All line numbers re-verified at pin and `main`. |
| 05 | [05-no-sm8x-attention-path.md](05-no-sm8x-attention-path.md) | vllm-project/vllm | **FILEABLE.** Dup check empty. Its own open item now closed. Citations verified identical at vLLM `main`. |

### What the check found, in one place

| Our defect | Already handled upstream by | State |
|---|---|---|
| 02 item 1, `cute.core.ThrMma` | tml-fa4 PR #3 | MERGED 2026-07-17 |
| 02 item 2, `cute.make_fragment` | tml-fa4 PR #3 | MERGED 2026-07-17 |
| 03 defects 1 and 2, `mDynamicCausal` and `is_split_kv` | flash-attention PR #156 | OPEN since 2026-06-30, still unmerged |

vLLM also already bumped its tml-fa4 pin past 02 items 1 and 2, in its own PR
#48988, MERGED 2026-07-18. Our build base predates that bump, which is why we saw
those two at all.

## Verified pins used across all five

| Component | Pin | Source of truth |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `vllm/cmake/external_projects/tml_fa4.cmake:17` |
| vllm-project/flash-attention | `caaa4eb59845388a20b1f435ecaafb4bd9517ad8` | `vllm/cmake/external_projects/vllm_flash_attn.cmake:42` |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `vllm/requirements/cuda.txt:28` |

No other pin appears in any of the five files. Do not add one without a source.

### Upstream has moved past both of our pins. Know this before filing

Checked 2026-07-25. Our pins are our build base's, and both are now behind:

| Component | Our pin | Upstream `main` / vLLM `main` pins | Consequence |
|---|---|---|---|
| tml-fa4 | `13374f0c` | `b206834606` | Report 02 items 1 and 2 are fixed there. Reports 01, 04 and 03's defect 3 are **unaffected**: `interface.py`, `flash_fwd_sm90.py` and `flash_fwd_sm120.py` are byte-identical between the two commits. |
| flash-attention | `caaa4eb5` | `ed4b7342` | Report 03's defects 1 and 2 are still live at `ed4b7342`, which is also current `main`. PR #156 is still the pending fix. |

Where our pin is `13374f0c`: that commit is PR #2's branch commit, dated
2026-07-16T02:08:50Z. Main's merge of PR #2 is `09d2482ed3`, five minutes later.
PR #3 is `b206834606`, the next commit. So our pin sits between PR #2 and PR #3.

Say "verified at our pin `X` and at `main` `Y`" in each filing where we have
checked both. For 01, 04 and 05 we have. That single sentence removes the
easiest possible rebuttal, which is that we are reporting a stale tree.

## Filing order

Moved to [FILING.md](FILING.md), which supersedes the order that used to live
here. That order listed report 03 as the fourth filing; 03 is now retired.

## Duplicate check, MANDATORY, re-run immediately before filing

The vLLM contribution policy in `vllm/AGENTS.md` requires a duplicate-work
check before proposing work. An empty result set is the thing being recorded, so
record it.

### Read this first: the old commands in this file did not run

Every `gh search` command previously listed here carried `--state all`. **That
flag does not exist on `gh search`.** It errors out:

```
$ gh search issues --repo vllm-project/tml-fa4 --state all "rel_bias"
invalid argument "all" for "--state" flag: valid values are {open|closed}
```

So none of the 24 keyword searches in this section had ever executed
successfully. `--state all` is valid on `gh issue list` and `gh pr list`, and
invalid on `gh search issues` and `gh search prs`. Omit `--state` on the search
commands; they cover all states by default.

Three further lessons, learned by the checks that finally ran on 2026-07-25:

1. **`gh search issues` does not return PRs.** Both duplicates we eventually
   found are PRs. On a tracker with issues disabled, only a PR query can find
   anything at all.
2. **Listings beat keywords.** `gh pr list --state all` found tml-fa4 PRs #2 and
   #3 in one call. The keyword queries written down for report 02 would have
   missed PR #3 even had they run, because they searched for identifiers that do
   not appear in its title or body in a matching form.
3. **Check whether the tracker accepts issues before naming it as the target.**
   Report 03 named a tracker that has had issues disabled the whole time.

### Step 1, tracker capability. Run this before anything else

```bash
gh repo view vllm-project/tml-fa4        --json hasIssuesEnabled
gh repo view vllm-project/flash-attention --json hasIssuesEnabled
gh repo view vllm-project/vllm            --json hasIssuesEnabled
```

Real output, 2026-07-25:

| Tracker | `hasIssuesEnabled` | Route |
|---|---|---|
| vllm-project/tml-fa4 | `true` | issue |
| vllm-project/vllm | `true` | issue |
| vllm-project/flash-attention | **`false`** | **PR only** |

Confirming the third independently:

```bash
$ gh issue list --repo vllm-project/flash-attention --state all --limit 200
the 'vllm-project/flash-attention' repository has disabled issues
```

### Step 2, full listings. These are what actually catch duplicates

```bash
gh issue list --repo vllm-project/tml-fa4 --state all --limit 200
gh pr   list --repo vllm-project/tml-fa4 --state all --limit 200
gh pr   list --repo vllm-project/flash-attention --state all --limit 200
```

Real output, 2026-07-25:

- **tml-fa4 issues:** empty. The tracker has **zero** issues, open or closed.
- **tml-fa4 PRs:** exactly 3.
  `#3 Migrate deprecated CuTe DSL APIs for cutlass-dsl 4.6` MERGED 2026-07-17,
  `#2 Fix forward argument handling on pre-Blackwell GPUs` MERGED 2026-07-16,
  `#1 Add Blackwell plain FP8 attention support` CLOSED.
  **#3 is report 02 items 1 and 2. #2 created the code report 01 describes.**
- **flash-attention PRs:** 173 listed. `#156` is report 03 defects 1 and 2, open
  since 2026-06-30. `#157 Fix compatibility with nvidia-cutlass-dsl 4.6.0`
  MERGED 2026-07-13 is the sibling-fork precedent for report 02 item 3.

The 2026-07-21 sweep recorded "tml-fa4: Zero issues. Open PRs: none." That was
literally true and materially misleading: PRs #2 and #3 had already **merged**
by then, and a query restricted to open PRs cannot see a merged one. Use
`--state all`.

### Step 3, per-report keyword searches, corrected syntax

```bash
# 01, rel_bias never reaches the kernel pre-Blackwell   ALL EMPTY 2026-07-25
gh search issues --repo vllm-project/tml-fa4 "rel_bias"
gh search issues --repo vllm-project/tml-fa4 "sheared bias"
gh search issues --repo vllm-project/tml-fa4 "ShearingBias"
gh search issues --repo vllm-project/tml-fa4 "silently"
gh search prs    --repo vllm-project/tml-fa4 "rel_bias"
gh search issues --repo vllm-project/vllm "rel_bias silently"
gh search issues --repo vllm-project/vllm "rel_bias non-Blackwell"
gh search issues --repo vllm-project/vllm "ShearingBias"
gh search prs    --repo vllm-project/flash-attention "rel_bias"
gh search prs    --repo vllm-project/flash-attention "sheared"

# 02 items 3 and 4, surviving DSL drift
gh search prs    --repo vllm-project/tml-fa4 "cutlass-dsl 4.6"   # FINDS #3, items 1+2, dup
gh search prs    --repo vllm-project/tml-fa4 "cutlass"           # FINDS #3 and #1
gh search prs    --repo vllm-project/tml-fa4 "fmax"              # empty, item 3 not covered
gh search prs    --repo vllm-project/tml-fa4 "window_size"       # empty, item 4 not covered
gh search issues --repo vllm-project/tml-fa4 "make_fragment"     # empty
gh search issues --repo vllm-project/tml-fa4 "ThrMma"            # empty

# 03 defect 3 only, targets tml-fa4                     ALL EMPTY 2026-07-25
gh search issues --repo vllm-project/tml-fa4 "use_tma_O"
gh search issues --repo vllm-project/tml-fa4 "pack_gqa"
# 03 defects 1 and 2 are flash-attention PR #156. Confirm it is still open:
gh pr view 156 --repo vllm-project/flash-attention

# 04, pack_gqa row semantics                            ALL EMPTY 2026-07-25
gh search issues --repo vllm-project/tml-fa4 "pack_gqa"
gh search prs    --repo vllm-project/tml-fa4 "pack_gqa"

# 05, no SM8x attention path                            NO DUPLICATE 2026-07-25
gh search issues --repo vllm-project/vllm "Inkling Ampere"
gh search issues --repo vllm-project/vllm "Inkling unsupported GPU"
gh search issues --repo vllm-project/vllm "score_mod not supported"
gh search issues --repo vllm-project/vllm "score_mod SM8x"
gh search issues --repo vllm-project/vllm "inkling A100"
gh search issues --repo vllm-project/vllm "Inkling"            # 4 hits, none is this
gh search prs    --repo vllm-project/vllm "Inkling" --limit 25 # 25 hits, none is this
```

### Adjacent work found, none of it a duplicate. Read before filing

| Item | What it is | Why it is not a duplicate |
|---|---|---|
| vLLM issue #49049, open | Unclamped q-row in the rel-bias **score_mod gather** causes a deterministic illegal address on sm_121a, with coredump evidence | Different arch, different mechanism, and it is about the score_mod path being wrong rather than the rel_bias path being ignored or sm_80 having no path |
| vLLM PR #48858, merged 2026-07-16 | Introduced `_use_sheared_bias()` and the non-Blackwell `score_mod` route | It is the **cause** of report 01's narrowed scope and the context for report 05, not a fix for either |
| vLLM PR #48988, merged 2026-07-18 | Bumped the vLLM tml-fa4 pin for cutlass-dsl 4.6 | Closes report 02 items 1 and 2 on the vLLM side |
| vLLM PRs #48841, #48954, open | ROCm Inkling support | Adjacent support-matrix work, not SM8x |

If any command returns a match, do not file. Read the match, decide whether our
report is materially different, and if it is, say how in the new report. If it
is not, add our evidence as a comment on the existing item instead. That last
case is live right now: it is what we do with flash-attention PR #156.

## Prior duplicate-check record, 2026-07-21, superseded

Queries run via `gh search issues` and `gh issue list`, account jaberjaber23:

- vllm-project/vllm: "inkling rel_bias", "tml-fa4", "inkling attention",
  "rel_bias ignored", "cutlass 4.6 make_fragment", "sheared bias". All empty.
- vllm-project/tml-fa4: full issue list, state all. Zero issues. Open PRs: none.
- vllm-project/flash-attention: **NOT CHECKED.**

Superseded by the 2026-07-25 sweep above, which found three duplicates the
2026-07-21 sweep structurally could not have found: two merged PRs on a tracker
where only open PRs were listed, and one PR on a tracker that was not queried at
all. Kept here because the failure mode is the useful part.

### Known gap, CLOSED 2026-07-25

~~Report 03 targets vllm-project/flash-attention for two of its three defects, and
no duplicate check has ever been run there.~~

Closed. The check was run. It found that the tracker has issues disabled, and
that PR #156 already covers report 03 defects 1 and 2. Report 03 is retired as a
duplicate; its defect 3 survives against tml-fa4. Full record in
[03-vllm-flash-attn-generic-path-bugs.md](03-vllm-flash-attn-generic-path-bugs.md).

## Evidence chain

- `journal/local-tier-bringup.md`, sm_120 bring-up and the first three defects.
- `journal/remote/h100-session1.md`, sessions 1 to 4, the DSL drift cluster and
  the sm_90 rel_bias measurements.
- `journal/u2-hopper-design.md`, THE KEY INSIGHT, sessions 23 to 27, the
  pack_gqa root cause and the sm_80 support gap.
- `journal/remote/tune_sm80_a100.json`, the sm_80 tile sweep. Its **decode**
  percentages are withdrawn, because the harness parity-gated on
  `seqlen_q == seqlen_k` and timed decode shapes: see
  `journal/regression-ampere-tile-sweep.md`. The prefill rows and the
  `tile_n=128` collapse stand, and they are the only parts report 05 now cites.

- `journal/regression-sm120-varlen-illegal-address.md`, 2026-07-25, the sm_120
  hardware confirmation of flash-attention PR #156's second bug, at lines 84-102.
  This is the only new upstream-facing evidence from that session.

Fix artifacts: `scripts/apply_local_sm120_fixes.sh`, `scripts/bootstrap_b200.sh`
(drift section), `kernels/patches/`, `kernels/tml_fa4_modified/`.

### Artifact gap found 2026-07-25, do not cite these three

The three JSON artifacts under `journal/remote/local_sm120_s29/` are **zero
bytes**:

```
journal/remote/local_sm120_s29/parity_rel_bias_coverage_sm120.json    0 bytes
journal/remote/local_sm120_s29/parity_rel_chunked_decode_sm120.json   0 bytes
journal/remote/local_sm120_s29/parity_rel_varlen_batch_sm120.json     0 bytes
```

So the numeric sm_120 figures narrated in
`journal/regression-sm120-varlen-illegal-address.md`, including the "signal 0.5x
of tolerance" blindness figure and the 7/7, 3/3, 6/6 and 5/5 gate tallies, have
**no backing artifact in the tree**. The narrative record stands as a session
note. The numbers do not meet this repository's own rule that every number traces
to an artifact.

Consequence, applied: **no sm_120 numeric figure has been carried into any
upstream report.** The one thing taken from that session is the
`mDynamicCausal` NameError, which is a categorical observation of an error string
rather than a measurement, and it is used only as a hardware-confirmation comment
on an existing PR. Either re-run session 29 with artifact capture, or mark those
figures withdrawn.

## Filing rules

- Do not file from an agent. A human submits, under their own name, and must be
  able to defend every claim.
- Every file already carries the AI-assistance disclosure the vLLM policy
  requires. Do not remove it.
- Do not bundle. Each file is one issue.
- Check `hasIssuesEnabled` on the target tracker before writing a word for it.
- After filing, add the issue URL to the table above.
