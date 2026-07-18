# U2-Hopper: relative-bias attention, measured 3.2x headroom

Evidence base (journal/remote/h100-session1.md session 4, sm_90, kv64k
global decode): production score_mod = 2375us; plain attention = 743us;
sheared-style path = 747us but numerically wrong. Prefill 8K global:
score_mod 5372us vs sheared-style 2610us.

## Why the production path is slow

score_mod applies bias via a per-score-element callback gathering from a
materialized (T, H, ext) tensor: a data-dependent gmem read per score
element, serialized against the MMA pipeline. Additionally sm_90 forces
num_splits=1 (fa4_rel_attention.py:71-72), so b1 decode occupies ~8 CTAs
(one per KV head) — the GPU is mostly idle at long context.

## Candidate designs

### A. Fix the existing sheared path on sm_90 (quick validation)

Mechanism already ~free (747 vs 743us) but wrong at a subset of positions
(max err 0.9-1.6, mean 0.02-0.06 — pattern says layout mismatch between
ShearingBias 128-block writer and the sm_90 reader, not math error).
Plan: derive writer convention (shearing_bias.py:357-476) and sm_90 reader
convention (flash_fwd_sm90.py bias consumption) statically; they must agree
on (row, kv-tile) -> padded-column mapping. Local shear-stage parity test
runs on sm_120 (writer is arch-generic). Fix candidate validated in ONE
cheap H100 session (~$0.60).
Risk: low. Reward: ~3.2x decode / ~2x prefill on Hopper. Not novel (path
exists on Blackwell); upstream-quality bugfix.

### B. Register-resident r-projection bias (flagship, novel vs both stacks)

Skip rel_logits materialization AND shear: kernel keeps r (16 fp values per
row-head, registers) and tiles proj (16 x ext bf16 = 40KB max, smem);
bias(i,k) = dot16(r_i, proj[:, i-k]) computed inline = 16 FMA per score vs
128-MAC QK dot (~12% FLOP add, no gmem gather). Kills 3x bias tensor
round-trips in prefill (O(T*H*ext) bytes) and the shear pre-kernel launch.
Needs: qkvr_prep stops projecting (saves that GEMM too); attention API takes
(r, proj) instead of rel_logits; split-KV combine unchanged (bias is
pre-softmax). Prefill win compounds: session-4 gap + materialization traffic.
Risk: medium (new kernel path in CuTe sm_90 pipeline). Reward: >=A on
decode, larger on prefill; transfers to sm_120 and Blackwell (same trick
inside tcgen05 kernel later).

### Split-KV on sm_90 (orthogonal, stacks with A or B)

The num_splits=1 ban exists because score_mod + splits didn't compose.
With A or B the bias is either pre-materialized-sheared or register-local,
both split-compatible. Combine kernel exists (flash_fwd_combine.py). At b1
kv64k: 8 CTAs -> 8*splits CTAs; H100 has 132 SMs -> target 16 splits.
Expected from first principles: approach HBM-bound 268MB/pass; ceiling
~3.35TB/s vs measured 361GB/s -> up to ~9x on the plain kernel alone at b1.
Verify with ncu, not arithmetic.

## Order

1. A (days): validates the 3.2x quickly, upstreamable bugfix.
2. Split-KV enablement on top of A (days): the b1 idle-GPU fix.
3. B (the moat): replaces A's mechanism; A becomes its baseline.
4. U3 FP8 KV multiplies on whichever kernel wins.

## Gates per spec

Parity: harness/parity_fa4_rel.py extended per design; 32-prompt logit gate
when big box lands. Kernel: ncu >=90% of HBM roofline for decode cases
(memory-bound) on H100; profile saved journal/ncu/. No claim without both.
