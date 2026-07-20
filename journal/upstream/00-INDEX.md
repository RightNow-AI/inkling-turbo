# Upstream findings index (Inkling-turbo)

8 findings across the vLLM day-0 Inkling stack, discovered while building
open-source replacement kernels. Dup-check each tracker before filing
(vLLM AGENTS.md contribution policy).

| # | File | Target repo | Severity |
|---|------|-------------|----------|
| 1 | 01-rel-bias-silently-ignored-non-blackwell.md | tml-fa4 | HIGH — silent wrong output |
| 2 | 02-cutlass-4.6.0-api-drift-cluster.md (4 findings) | tml-fa4 | HIGH — nothing runs vs own pin |
| 3 | 03-vllm-flash-attn-generic-path-bugs.md (3 findings) | vllm-flash-attn (+tml-fa4) | MED — sm_120 users broken |

Evidence chain: journal/local-tier-bringup.md, journal/remote/h100-session1.md,
journal/u2-hopper-design.md. Fix artifacts: scripts/apply_local_sm120_fixes.sh,
scripts/bootstrap_b200.sh (drift section), kernels/tml_fa4_modified/.

Status: DRAFTED. To file: dup-check trackers -> post -> link back here.
