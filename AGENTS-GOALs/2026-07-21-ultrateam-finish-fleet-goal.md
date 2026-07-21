# 2026-07-21 — Ultrateam finish fleet (4 sol lanes + capacity hunters)

State when this fleet launched: main @ 9bf3f6b. U2 sm_90 native GREEN + race
won (session 24); U3 write path proven both archs; serving route patch merged;
8x gate tooling merged; AWS us-east-1 quota 192 vCPUs live but all P-types
ICE; Lambda 8x/1x dry. Spend ~$25.1 of $3k.

## Lanes (model gpt-5.6-sol-2, effort xhigh, launched ~13:1x UTC)

| Lane | Worktree | Branch | Job id | Mission |
|---|---|---|---|---|
| gates-smoke | ../inkling-gsmoke | feat/gates-smoke | task-mrupfmoc-99m044 | env-gated local smoke mode for the 3 gate scripts + SMOKE.md |
| u2-perf | ../inkling-u2perf | perf/u2-splitkv | task-mrupg7cx-1w6lz5 | T1 split-KV sm_90+bias (port sm_100 mechanism, reuse Combine), T2 overlap analysis/enable, T3 packed-gqa notes |
| u3-readpath | ../inkling-u3read | feat/u3-readpath | task-mrupj4rq-pcmedw | fp8 KV read path in generic kernel + interface patch script + harness case |
| upstream-final | ../inkling-upfinal | docs/upstream-final | task-mruphu0y-9es2be | findings 01-04 filing-ready |

Lanes are briefed NO-GIT (sandbox denies .git): orchestrator gates, commits,
merges. Status: `node ~/.claude/skills/ultrateam/scripts/codex-lane.mjs status <job> --lane <worktree> --json`.

## Gate plan per lane
- gates-smoke: I run SMOKE.md in WSL on the 5090 (Qwen2.5-0.5B), fix-loop via
  --resume-last until the smoke passes, THEN the 8x session is de-risked.
- u2-perf: py_compile + diff review here; GPU validation on next parked H100
  (parity 3/3 + probes + split-KV microbench) before merge.
- u3-readpath: run its parity_kv_fp8 extension locally on 5090; merge on green.
- upstream-final: read-review only; merge; file AFTER release flip.

## Background capacity (harness tasks, auto-notify)
- AWS 12h sweep: aws_8x_gates.py --retry-minutes 720 (task bp88tmpve)
- Lambda 8x hunter (task bh9bg9ana); Lambda 1x hunter (task buznkvnz7)
- AWS quota watcher (task bjt9g5xmt); us-west-2/us-east-2 192-vCPU requests PENDING
- On 8x land: bootstrap -> parity -> logit gate auto; e2e is launched by me
  from the parked box (gate_e2e_bench.sh); hard caps enforce termination.

## Collision rule
If BOTH AWS and Lambda land 8x boxes, keep the first-to-bootstrap, terminate
the other immediately (both launchers write scripts/.gates8x_instance.json /
park markers; Lambda id via API list, AWS via describe-instances).
