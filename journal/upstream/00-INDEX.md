# Upstream findings index (Inkling-turbo)

8 findings across the vLLM day-0 Inkling stack, discovered while building
open-source replacement kernels. Dup-check each tracker before filing
(vLLM AGENTS.md contribution policy).

| # | File | Target repo | Severity |
|---|------|-------------|----------|
| 1 | 01-rel-bias-silently-ignored-non-blackwell.md | tml-fa4 | HIGH, silent wrong output |
| 2 | 02-cutlass-4.6.0-api-drift-cluster.md (4 findings) | tml-fa4 | HIGH, nothing runs vs own pin |
| 3 | 03-vllm-flash-attn-generic-path-bugs.md (3 findings) | vllm-flash-attn (+tml-fa4) | MED, sm_120 users broken |

Evidence chain: journal/local-tier-bringup.md, journal/remote/h100-session1.md
journal/u2-hopper-design.md. Fix artifacts: scripts/apply_local_sm120_fixes.sh
scripts/bootstrap_b200.sh (drift section), kernels/tml_fa4_modified/.

Status: DRAFTED. To file: dup-check trackers -> post -> link back here.

## Duplicate-check record (2026-07-21, per vllm AGENTS.md)

Queries run via `gh search issues` / `gh issue list` (account jaberjaber23):
- vllm-project/vllm: "inkling rel_bias", "tml-fa4", "inkling attention"
  "rel_bias ignored", "cutlass 4.6 make_fragment", "sheared bias" -> ALL EMPTY
- vllm-project/tml-fa4: full issue list (state all) -> ZERO issues; open PRs -> none

No existing issue or PR covers any indexed finding. Trackers are clear to
file after release. Re-run the sweep immediately before actually filing.

## Finding 04 (new, 2026-07-21): pack_gqa x row-semantics interaction

tml-fa4's pack_gqa packs qhead_per_kvhead q-heads into score-tile rows.
Nothing in the sm_90/sm_80 kernel API surfaces this to row-indexed features;
the sheared-bias contract (128-row blocks) and any head-sliced per-row
tensor silently break. Evidence: session 24 stride print (81920 = Hq*padded
= +1 seq row where +8 tile rows expected), then parity 3/3 after forcing
pack_gqa=False. sm_100 handles it via group_tile_bias in the shear writer;
the generic path never packs (finding 03 context). Filing shape: doc/API
note + the working native sm_90 sheared-bias port as the reference fix.

## Finding 05 (2026-07-23): no Inkling rel-attention path exists on SM8x

vllm_flash_attn cute interface.py:722 raises NotImplementedError for any
score_mod on SM8x, and the Inkling serving router's only non-Blackwell path
IS score_mod, so day-0 Inkling attention cannot execute on A100-class GPUs
at all. Evidence: session 26, parity harness on A100-SXM4-40GB: our sheared
generic kernel 3/3 green on the same cases where every day-0 path raises.
Filing shape: gap report + our generic sheared-bias kernel as the working
sm_80 reference. Dup-check: covered by the 2026-07-21 sweep (trackers had
zero issues); re-run before filing.
