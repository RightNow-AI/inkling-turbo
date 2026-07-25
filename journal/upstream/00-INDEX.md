# Upstream findings index (Inkling-turbo)

10 defects across the vLLM day-0 Inkling stack, found while building
open-source replacement kernels. They are packaged as 5 filing-ready issues.
Every technical claim traces to a journal entry or to a file and line in the
pinned trees.

Status: FINALIZED, NOT FILED. A human files these under their own name after
re-running the duplicate check below.

## The five issues

| # | File | Target tracker | Defects | Severity |
|---|---|---|---|---|
| 01 | [01-rel-bias-silently-ignored-non-blackwell.md](01-rel-bias-silently-ignored-non-blackwell.md) | vllm-project/tml-fa4 | 1 | HIGH, silent wrong output |
| 02 | [02-cutlass-4.6.0-api-drift-cluster.md](02-cutlass-4.6.0-api-drift-cluster.md) | vllm-project/tml-fa4 | 4 | HIGH, nothing runs against the pinned DSL |
| 03 | [03-vllm-flash-attn-generic-path-bugs.md](03-vllm-flash-attn-generic-path-bugs.md) | vllm-project/flash-attention, one item in vllm-project/tml-fa4 | 3 | MED, sm_120 users broken |
| 04 | [04-pack-gqa-row-semantics.md](04-pack-gqa-row-semantics.md) | vllm-project/tml-fa4 | 1 | MED, contract hazard |
| 05 | [05-no-sm8x-attention-path.md](05-no-sm8x-attention-path.md) | vllm-project/vllm | 1 | MED, no attention path on SM8x |

## Verified pins used across all five

| Component | Pin | Source of truth |
|---|---|---|
| vllm-project/tml-fa4 | `13374f0c855acc1add1bf30444bd67aebbc24a8e` | `vllm/cmake/external_projects/tml_fa4.cmake:17` |
| vllm-project/flash-attention | `caaa4eb59845388a20b1f435ecaafb4bd9517ad8` | `vllm/cmake/external_projects/vllm_flash_attn.cmake:42` |
| vllm-project/vllm | fork base `850295881` | our build base |
| nvidia-cutlass-dsl | `4.6.0` | `vllm/requirements/cuda.txt:28` |

No other pin appears in any of the five files. Do not add one without a source.

## Recommended filing order

1. **01, rel_bias silently ignored.** Highest severity in the series. It is the
   only defect that returns wrong numbers with no error, so it goes first
   regardless of anything else.
2. **02, CuTe DSL 4.6.0 drift.** File immediately after 01. It is the
   prerequisite for reproducing 01 and 04. A maintainer who tries 01's
   reproducer against the pinned DSL hits 02's import error before reaching the
   bug. Filing 02 second means the answer is already in the tracker.
3. **04, pack_gqa row semantics.** Same repo, same bias subsystem, reads as the
   follow-up to 01. It also explains why a naive fix for 01 on sm_90 does not
   work, which is context a maintainer will want while 01 is still open.
4. **03, generic sm_120 path.** Different repo and different arch, independent
   of 01, 02 and 04. Two of its three defects belong to
   vllm-project/flash-attention. The third belongs to vllm-project/tml-fa4 and
   is labeled in the file. Decide at filing time whether to cross-post that
   item as a separate tml-fa4 issue or to link it from the main one.
5. **05, no SM8x attention path.** Files last. Its second fix option depends on
   01 being fixed, so it should be able to link to the already-open 01.

## Duplicate check, MANDATORY, re-run immediately before filing

The vLLM contribution policy in `vllm/AGENTS.md` requires a duplicate-work
check before proposing work. Run every command below and paste the results into
the filing notes. An empty result set is the thing being recorded, so record it.

### Full listings, both trackers, both states

```bash
gh issue list --repo vllm-project/tml-fa4 --state all --limit 200
gh pr   list --repo vllm-project/tml-fa4 --state all --limit 200
gh issue list --repo vllm-project/flash-attention --state all --limit 200
gh pr   list --repo vllm-project/flash-attention --state all --limit 200
```

### Per-issue keyword searches

```bash
# 01, rel_bias silently ignored
gh search issues --repo vllm-project/tml-fa4 --state all "rel_bias"
gh search issues --repo vllm-project/tml-fa4 --state all "sheared bias"
gh search issues --repo vllm-project/vllm --state all "rel_bias ignored"
gh search issues --repo vllm-project/vllm --state all "ShearingBias"
gh search prs    --repo vllm-project/tml-fa4 --state all "rel_bias"

# 02, CuTe DSL 4.6.0 drift
gh search issues --repo vllm-project/tml-fa4 --state all "make_fragment"
gh search issues --repo vllm-project/tml-fa4 --state all "ThrMma"
gh search issues --repo vllm-project/tml-fa4 --state all "cutlass-dsl 4.6"
gh search issues --repo vllm-project/vllm --state all "nvidia-cutlass-dsl 4.6.0 tml-fa4"
gh search prs    --repo vllm-project/tml-fa4 --state all "make_rmem_tensor"

# 03, generic sm_120 path
gh search issues --repo vllm-project/flash-attention --state all "mDynamicCausal"
gh search issues --repo vllm-project/flash-attention --state all "sm120 cute"
gh search issues --repo vllm-project/flash-attention --state all "is_split_kv"
gh search issues --repo vllm-project/tml-fa4 --state all "use_tma_O"
gh search issues --repo vllm-project/vllm --state all "sm_120 flash attention cute"
gh search prs    --repo vllm-project/flash-attention --state all "sm120"

# 04, pack_gqa row semantics
gh search issues --repo vllm-project/tml-fa4 --state all "pack_gqa"
gh search issues --repo vllm-project/tml-fa4 --state all "pack_gqa bias"
gh search prs    --repo vllm-project/tml-fa4 --state all "pack_gqa"

# 05, no SM8x attention path
gh search issues --repo vllm-project/vllm --state all "inkling attention"
gh search issues --repo vllm-project/vllm --state all "score_mod SM8x"
gh search issues --repo vllm-project/vllm --state all "inkling A100"
gh search prs    --repo vllm-project/vllm --state open --search "inkling attention sm80"
```

### Extra checks the vLLM policy names explicitly, for issue 05

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "inkling rel attention"
```

If any command returns a match, do not file. Read the match, decide whether our
report is materially different, and if it is, say how in the new report. If it
is not, add our evidence as a comment on the existing issue instead.

## Prior duplicate-check record, 2026-07-21

Queries run via `gh search issues` and `gh issue list`, account jaberjaber23:

- vllm-project/vllm: "inkling rel_bias", "tml-fa4", "inkling attention",
  "rel_bias ignored", "cutlass 4.6 make_fragment", "sheared bias". All empty.
- vllm-project/tml-fa4: full issue list, state all. Zero issues. Open PRs: none.
- vllm-project/flash-attention: **NOT CHECKED.** No query was run against this
  tracker on that date or since.

No existing issue or PR covered any indexed finding on the two trackers that
were checked. That sweep is stale now and does not substitute for the commands
above.

### Known gap, must close before filing

Report 03 targets vllm-project/flash-attention for two of its three defects, and
no duplicate check has ever been run there. The vLLM contribution policy
requires one, and `docs/METHODOLOGY.md` makes it a condition of a defect claim.
Run the six flash-attention commands in the sections above and record the
result, including an empty result, before 03 is filed. There are six, not four:
the two full listings (`gh issue list` and `gh pr list`) in "Full listings",
plus the four in the "03, generic sm_120 path" block of the keyword searches.
Do not file 03 on the strength of the 2026-07-21 sweep; that sweep did not
cover it.

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

Fix artifacts: `scripts/apply_local_sm120_fixes.sh`, `scripts/bootstrap_b200.sh`
(drift section), `kernels/patches/`, `kernels/tml_fa4_modified/`.

## Filing rules

- Do not file from an agent. A human submits, under their own name, and must be
  able to defend every claim.
- Every file already carries the AI-assistance disclosure the vLLM policy
  requires. Do not remove it.
- Do not bundle. Each file is one issue.
- After filing, add the issue URL to the table above.
