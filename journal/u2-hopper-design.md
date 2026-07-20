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

## CORRECTION (2026-07-18, post static analysis) — supersedes Design A

Root cause of the sm_90 wrong output is NOT a layout mismatch: the sm_90
kernel has no bias support at all. Proof: FlashAttentionForwardSm90 ctor
(interface.py:~1600) receives no bias argument — vs Sm100's has_bias=
(interface.py:~980); flash_fwd_sm90.py contains zero bias code. On sm_90
the interface accepts rel_bias, allocates + shears the padded tensor
(wasted), then runs PLAIN attention and returns it silently.

Consequences:
- Session-3/4 "sheared-style 747us" = plain attention + dead shear launch.
  The "bias ~free on sm_90" hypothesis is VOID — no such kernel exists.
  (3.2x score_mod-vs-plain headroom measurement is unaffected.)
- Design A ("fix the layout") does not exist as a quick win: bias
  consumption on sm_90 must be BUILT. Therefore A collapses into B —
  build the register-resident r-projection bias directly in the sm_90
  pipeline. B's inline dot16 is also strictly simpler to add than sheared
  tile loads (no pre-kernel, no padded tensor, no g2s bias staging).
- Upstream finding #5 (worst of the set): tml-fa4 flash_attn_varlen_func
  silently ignores rel_bias on every non-Blackwell arch — a correctness
  trap; should raise NotImplementedError. Report with minimal repro.

Revised order: B on sm_90 (target: <=1.1x plain attention at decode,
validating vs harness oracle) -> split-KV enablement -> U3 FP8 KV on top.

## V1 result (2026-07-18, sm_120 local): PARITY GREEN, SPEED FAILED

Parity: 3/3 vs oracle (max 1.6e-2 — same class as production score_mod).
Speed (5090, relative): kv64k b1 decode 13900us vs score_mod 5319us vs
plain 3515us — V1 is 2.6x WORSE than the gather it replaces. Prefill 8K:
49.0ms vs 22.3ms (2.2x worse).

Diagnosis: the score_mod callback charges PER SCORE ELEMENT. V1's constexpr
dot16 compiles to 16 dependent loads (r row + proj column) + FMAs per
element = 17 memory ops where the gather does 1. L1 residency cannot save a
17x issue-count. The mechanism is right; the loop level is wrong.

V2 (the real kernel change, as anticipated): hoist per-ROW state — load
r[q,h,:] (16 values) into registers once per row per m_block, stage the
proj slice for the tile in smem, apply as a vectorized tile op inside
apply_score_mod_inner/mma epilogue instead of per-element callback. Per
element then: 16 FMA + 1 smem vec read, amortized r. This requires
extending softmax.py apply_score_mod_inner (new constexpr fast path) or a
dedicated bias hook in flash_fwd_sm90.py — kernel work, next session.

Meta: V1 cost $0 (local) and pinned the exact perf mechanism. Correct-but-
slow recorded as failed speed gate per LEDGER discipline; kernels/
relproj_score_mod.py kept as the semantic reference for V2.

## V2 implementation brief (anchors verified 2026-07-18)

Target: new constexpr fast path inside `apply_score_mod_inner`
(vllm/vllm_flash_attn/cute/softmax.py:391-520) + a relproj marker the
interface can detect, so the sm_90/sm_120 kernels get tile-level bias with
zero per-element aux gathers.

Verified anchors:
- sm_90 kernel calls apply_score_mod per (m_block, n_block) with
  constant_q_idx=None ALWAYS (flash_fwd_sm90.py:1488-1519) — even decode.
  So V2 hoists per-ROW, not per-call: within one apply call, the thread's
  accumulator rows are FIXED by the wgmma thread layout (tScS row pattern:
  each thread owns 2 rows x N/vec cols for sm_90 m16n8 frags).
- Element loop: softmax.py:452-onwards iterates i in range(0, n_vals,
  vec_size); row index changes only between i-groups (thread-frag layout);
  kv_idx increments by vec within a row.

V2 shape (new branch `relproj_bias=True`, aux = [r (T,H,16), proj(16,ext)]):
1. Before the i-loop: derive the <=2 distinct q rows this thread owns from
   tScS; for each, load r[q_row, head, 0:16] once -> rmem (2x16 bf16).
2. Per i-group: dist_base = q_row - kv_idx(i); proj columns needed are
   dist_base-1..dist_base-vec (contiguous, reversed). One vectorized rmem
   load from gmem proj (columns are contiguous in memory: proj is (16,ext)
   row-major -> a proj COLUMN is strided ext... layout note: store proj
   TRANSPOSED (ext,16) so a column becomes a contiguous 32B row read; the
   qkvr weight loader can pre-transpose at load time, zero runtime cost).
3. bias[j] = sum_d r_reg[row_sel][d] * projT[dist-j][d] — 16 FMA per
   element, all operands rmem/L1; clamp dist via the existing d0 logic.
4. Fold tau upstream (r' = r * tau) as V1 did.

Cost model per element: 16 FMA + one 32B L1 read (projT row, high locality:
adjacent elements read adjacent rows) vs score_mod's 1 random gmem gather
into (T,H,ext). Decode kv64k: projT working set = 1024x16x2B = 32KB row-
sequential — L1-resident.

Gates: parity_fa4_rel.py backend 3 (already wired — swap callback for the
V2 flag); local relative timing must beat score_mod at kv64k before paying
for H100; then H100 session: target <=1.1x plain (743us), ncu HBM roofline
report, save to journal/ncu/.

Fallback if CuTe layout fight exceeds 2 sessions: sm_90 shear-consume path
(port sm_100's has_bias tile loads into flash_fwd_sm90.py) — known-working
design, materialization cost accepted on Hopper only.

## V1.5 result + strategic correction (2026-07-18, local)

V1.5 (proj transposed -> contiguous 32B per-element reads): 13987us vs V1
14797us at kv64k b1 — only 5%. The DSL emits the unrolled dot16 as 16
independent scalar loads regardless of contiguity; coalescing was not the
bottleneck.

Deeper implication, from re-examining session-4 data: at DECODE the day-0
aux tensor rel_logits is only (B,H,ext) ~= 4MB — L2-resident — yet
score_mod still costs 3.2x. So the per-element callback machinery ITSELF
(index divmods, SSA chains, loads serialized against the MMA pipeline) is
the overhead, not aux locality. CONSEQUENCE: no callback-level bias
implementation can reach <=1.1x plain. The fix must be TILE-LEVEL: bias
applied as vectorized fragment ops outside the per-element loop.

Revised V2 (primary): port sm_100's has_bias tile path to flash_fwd_sm90.py
— consume the SHEARED bias tensor via smem tile loads + vectorized adds to
the accumulator fragment (the design already proven on Blackwell; the
ShearingBias pre-kernel is arch-generic and already runs). Materialization
cost accepted on Hopper (decode bias tensors are small; prefill sheared
traffic is the price of the 3.2x win — revisit register-resident INSIDE
the tile loop later, where r/proj live in the fragment pipeline, not in a
per-element callback).

Register-resident-in-callback line of work: CLOSED (V1, V1.5 measured
dead ends — kept in kernels/relproj_score_mod.py as semantic references
and for the parity-harness third backend).

## Donor-pattern map: sm_100 has_bias -> generic-kernel port (2026-07-18)

sm_100 machinery (flash_fwd_sm100.py, all verified line refs):
- ctor: has_bias/rel_extent_padded -> bias_n_max = padded_ext/n_block (:156),
  bias_stage 1-2 (:180), +8 softmax regs (:392), smem budget accounting
  (:442-487, downgrades bias_stage/kv_stage to fit 224KB).
- setup: mBias (total_q, h, padded_ext) K-major required (:1038-1042);
  sBias layout (bias_block_size=128 rows x n_block) x stage (:1044);
  TMA atom for g2s (:1052); s2r tiled copy 128b vector reads (:1059-1062);
  own mbar pipeline (:1168, :1660-1663).
- Consumption: sheared property means for row m the bias for kv tile
  n_block sits at sheared columns [same n-tile-local range] — tile fetch is
  a plain 2D block copy; per-element add to acc fragment via s2r copy then
  vector adds (no divmods, no per-element addressing math).

Port to generic SM80-family kernel (flash_fwd.py — runs on 5090/sm_120,
simplest pipeline, local dev loop; sm_90 port second):
1. ctor flag has_bias + rel_extent_padded; smem: sBias tile
   (tile_m x tile_n bias slice... NOTE sheared bias row-width needed per
   (m,n) tile is tile_n, aligned by the shear — one (tile_m x tile_n) bf16
   tile = 128x64x2B = 16KB at sm_120 tile sizes; fits the 99KB budget by
   trimming num_stages if needed (can_implement update).
2. g2s: cp.async 2D block copy of gBias tile per n_block iteration,
   double-buffered with the existing K/V stages barrier.
3. apply: after gemm_qk, before mask/softmax — per-thread reads of its
   acc coords from sBias (ld.shared, vectorize via the identity-tensor
   coords the mask already computes) + add. Column mapping: sheared_col =
   rel_extent_padded - 1 - (n_idx_right - 1 - kv_idx) per ShearingBias
   writer math (shearing_bias.py:357-476) — derive the exact per-tile
   offset ONCE on paper, unit-test against ShearingBias output on 5090
   BEFORE wiring into the attention kernel (a standalone shear-consume
   test comparing smem-tile reads vs rel_logits[q,h,q-k] oracle).
4. interface: stop gating rel_bias to Blackwell; pass mBias + flag into
   the generic ctor; sm_90 stays score_mod until its own port.

Validation ladder: shear-mapping unit test (local) -> parity_fa4_rel
backend 4 (local sm_120) -> speed vs score_mod local -> sm_90 port ->
H100 session (target <=1.1x plain 743us + ncu roofline).

## Shear-writer layout contract (empirical, machine-extracted 2026-07-18)

harness/parity_shear_writer.py compiles ShearingBias standalone on sm_120
(writer is arch-generic — CONFIRMED it runs locally) and decodes the full
(row, kv) -> column map from encoded values. T=12, ext=512, padded=768:

  col(i, k) = k + (padded - 128 * n_blocks_max_row)   [= k + 640 at T<=128]

- Column is affine in ABSOLUTE kv index k; the row's last attention n-block
  is right-aligned to the padded tensor edge. 78/78 causal pairs match.
- Local vs global: identical placement; only pad VALUES differ
  (right pad -inf always; left pad -inf local / 0.0 global-beyond-extent —
  matches shearing_bias.py:88-89).
- TODO next: T=200 two-n-block case to pin the per-row-block shift
  (expected: rows attending nmax blocks get shift padded-128*nmax), then
  the consuming tile port reads bias tile (m,n) at columns
  [n*128 + shift(i), n*128 + shift(i) + 128) — one vectorized smem load
  per fragment row, zero per-element math.

## Layout contract COMPLETE (machine-verified, 2026-07-19)

T=200 two-block case: 20,100/20,100 positions match

  col(i, k) = k + padded - 128 * (m_block(i) + 1),  m_block(i) = i // 128

- Verified block shift: row 100 vs row 150 at same k differ by exactly 128.
- All rows within one attention m-tile share a single shift ->
  **the bias tile for attention tile (m, n) is one contiguous 128x128
  block at column n*128 + padded - 128*(m+1)** — fetchable with the same
  2D block-copy machinery the kernel uses for K tiles. Zero per-element
  addressing. This is the entire consumption contract for the tile port.
- Contract JSON kept locally (parity_shear_writer.json regenerable).

Port can now proceed with no unknowns in the data layout.

## Generic-kernel insertion map (flash_fwd.py, WSL tree; final pre-code map)

- Class: FlashAttentionForwardSm80 (:619) — sm_120 shim subclasses it.
- SharedStorage: _get_shared_storage_cls (:648-666) — ADD sBias struct
  (tile_m x tile_n bf16, 1024-aligned) to SharedStorageQKV (and budget
  check in can_implement; sm_120 99KB: 128x64x2B=16KB fits).
- Score site: mma_one_n_block (:1195-1220) — after gemm_qk / score_mod
  hook (:1209), BEFORE softmax: ADD tile-level bias fragment add reading
  sBias via the same thr_mma partition pattern as acc_S (tScS coords).
- Load site: the load_K/load_V cp.async pattern (:1225-1233 shows
  load_K_next) — ADD load_Bias(n_block) issuing a 2D cp.async block copy
  from gBias at column offset n*128 + padded - 128*(m_block+1) (the
  verified contract), same pipeline stages as K.
- Plumbing: __call__ (:668+) takes mBias tensor; kernel signature +
  launch (:786-801) thread it through; interface passes bias (already
  built+sheared for every arch) instead of dropping it for non-sm_100.
- Gate order: compile -> parity backend (existing) -> local race vs
  score_mod -> H100 session.

## v0 port: patch applied (2026-07-19), remaining plumbing for next session

kernels/patches/u2_v0_generic_bias.py applied to WSL tree (12 edits,
idempotent, re-runnable). DONE: Base ctor has_bias; SM80 __call__ mBias
transpose+thread-through; kernel slice mBias_cur per head/offset;
compute_one_n_block bias branch; apply_rel_bias (fragment-coord direct-gmem
add, contract-verified shift; scale folded per score_mod convention);
interface tile_n assert relaxed for arch families 8/12.

REMAINING (compile-debug loop, next session):
1. interface.py generic-family branch (~:989 sm_120, ~:868 sm_80): ctor
   needs has_bias=bias is not None; the fa_fwd(...) invocation needs
   mBias=bias kwarg. Find call site: grep "fa_fwd(" interface.py.
2. Scale mode: flash_fwd __call__ uses compute_softmax_scale_log2(scale,
   self.score_mod) — must behave as score_mod-mode when has_bias (bias adds
   AFTER scaling; apply_rel_bias already pre-scales). Read utils.
   compute_softmax_scale_log2, pass sentinel when has_bias.
3. Drive with harness/parity_fa4_rel.py backend 1 (tml_fa4_rel_bias) on
   sm_120 — SKIP should turn into real runs; expect DSL staging errors
   first (tScS[i][0] SSA indexing pattern may need utils helpers like
   softmax.py uses). Iterate.
4. Speed race vs score_mod (microbench relprojT cases) once parity green.

## v0 debug state (2026-07-19, session end)

Plumbing verified end-to-end: rel_bias on sm_120 now reaches KERNEL STAGING
(was: instant assert). Current failure is NOT in apply_rel_bias:

- varlen: MLIRError at Base.epilogue flash_fwd.py:388 —
  `seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None,None,head_idx]`
  slices 3 coords on a 2D view. ragged=False here (use_tma_O False on
  sm80-family), has_cu_seqlens_q=True -> domain_offset branch
  (seqlen_info.py:176-184). Suspect pack_gqa packed rank-2 first mode or a
  latent varlen bug in the never-compiled generic epilogue. NOTE score_mod
  varlen works via vllm_flash_attn's SEPARATE copy of this code — diff the
  two epilogues/offset_batch_Q first; the fix likely already exists there.
- batch (non-varlen) form: dies earlier, `atom._trait` NoneType — TMA atom
  None on sm80-family batch path (separate latent issue; deprioritize,
  varlen is what vLLM uses).

Next: diff vllm_flash_attn/cute/{flash_fwd.py,seqlen_info.py} epilogue
vs tml_fa4 copies; port the fix; rerun harness (repro: harness/repro_u2.py
varlen version in git history).

## v0 PARITY GREEN (global modes) — 2026-07-19

Two more latent generic-path bugs found + fixed (findings #6, #7):
6. pack_gqa=True default but tml_fa4 generic __call__ never calls
   pack_gqa_layout (vllm_flash_attn copy does, :726-729) -> rank chaos in
   epilogue. Fix: pack_gqa=False for arch families 8/12 (perf opt only).
7. sm_120 shim leaves self.arch=sm_120 -> use_tma_O=True on the sm80-family
   kernel which never configures TMA-O -> the :388 2D-view error (ragged
   branch) AND :401 tma_atom None. Fix: use_tma_O=False in generic __call__.

RESULT: tml_fa4 rel_bias path on sm_120 — global_short OK (7.8e-3),
global_beyond_extent OK (1.56e-2) — IDENTICAL diffs to score_mod on the
same inputs. The tile-level bias port produces correct attention.

REMAINING: swa_512 (local mode) fails at JIT ARG marshaling
(DSLRuntimeError "Internal Error" in _generate_jit_func_args) — an
argument type in local mode the DSL can't marshal, NOT a kernel-math bug.
Repro: harness/repro_u2.py (SWA variant). Debug: dump arg types passed to
compile in local vs global mode; suspect window_size_right=0 vs None or a
compile_key/exec-arg ordering divergence in the local branch.
Then: speed race vs score_mod (global mode is enough for the kv64k decode
headline), sm_90 port, H100 session.

## v0 final verdict (2026-07-19): PARITY 3/3 GREEN, SPEED FAILED (expected)

Finding #8 fixed on the way: generic call sites pass raw int window args
(DSL wants Int32/None) — SWA mode was unmarshalable. All 3 parity cases
now GREEN on sm_120 (7.8e-3/7.8e-3/1.56e-2 — score_mod-identical).

Speed (5090, kv64k): v0 74,941us vs score_mod 5,319us vs plain 3,515us —
14x WORSE. Per-element bounds-checked scalar gmem loads + select chains
in the unrolled acc loop. Third confirmation: NO per-element bias
application survives; tile-level smem staging (donor pattern) is the only
path. v0's real deliverables achieved: plumbing end-to-end, layout
contract exercised in-kernel, parity oracle green, 3 more latent bugs
fixed (#6 #7 #8).

## v1 (next): smem-staged bias tile
- Add sBias (tile_m x tile_n bf16 = 16KB) to SharedStorage.
- cp.async 2D block load per n_block: gBias rows [m*128, m*128+tile_m),
  cols [n*tile_n + shift, ...+tile_n), shift = padded - 128*(m+1);
  predicate cols to [0, padded) at COPY time (fill 0.0 outside), fold into
  the existing K-load pipeline stage/barrier.
- apply: replace v0 loop's gmem read with sBias[tScS coords] smem read —
  keep the acc*scale+bias structure, drop all per-element bounds checks
  (baked into the staged tile).
- Race again; then sm_90 port + H100.

## v1 COMPLETE: PARITY 3/3 + BEATS score_mod ON EVERY CASE (2026-07-19)

Debug chain (all banked in kernels/tml_fa4_modified/ verbatim files):
leftover v0 block -> jit-time copy construction (moved host-side into
_setup_attributes, K-machinery clone) -> gmem ptr alignment (dynamic col
shift folded into TILE INDEX — shift always multiple of tile_n; head-slice
ptr rebuilt via cute.make_ptr assumed_align=16 + stride assumes) ->
host-layout-in-kernel-region (sBias_layout + gmem_tiled_copy_Bias threaded
as kernel args like sQ_layout).

RESULTS (5090 sm_120, us/iter — vs same-machine baselines):
| case | v1 smem bias | score_mod | plain |
| decode_b1_kv64k | 5230 | 5319 | 3515 |
| decode_b32_kv64k | 5039 | 5519 | 3554 |
| decode_b32_kv8k | 695 | 766-835 | — |
| prefill_global_8k | 20223 | 22343-25466 | — |
| prefill_swa_8k | 18134 | 21200-24451 | — |

v1 wins every case locally (2-10%). NOTE sm_120 score_mod is only ~1.5x
over plain (vs 3.2x on H100/sm_90) — local headroom is structurally
smaller; the REAL verdict is the sm_90 port on H100 where the measured gap
is 3.2x. Remaining local gap to plain (~48%) = per-element ld.shared apply
loop + unoverlapped bias copy at stages=1 — optimization candidates for
the sm_90 port (vectorized smem reads, pipeline the bias copy).

NEXT: port this working machinery to flash_fwd_sm90.py (same edits, sm_90
kernel file), then ONE H100 session: parity + race vs the 2375us prod
baseline + ncu. Target: <=1.1x plain (743us) => ~3x捕获.

## sm_90 port map (2026-07-20, in progress)

tml_fa4/flash_fwd_sm90.py (1543 lines): warp-specialized — producer warp
(TMA, 32 thr, :212) + consumer MMA warpgroups; smem struct :119-157;
kernel :402; consumer mma region ~:943+ (wgmma partitions :972).

Port plan (translate proven generic-kernel machinery):
1. ctor: has_bias already accepted via Base.__init__ (shared!). sm_90
   __call__ needs mBias param + transpose + kernel threading (mirror
   generic edits; __call__ :157, kernel :402).
2. smem: add sBias (tile_m x tile_n bf16; sm_90 tile_n larger — check
   _tile_size_fwd_sm90; 128x128x2 = 32KB fits 228KB budget) to storage
   struct :119-157.
3. Load: CONSUMER-side cp.async (bypasses the TMA producer pipeline —
   simplest correct port; producer-pipeline integration is a later perf
   pass). Issue at n-block start in the consumer loop, cp_async_wait +
   barrier before apply.
4. Apply: after QK wgmma scores in rmem, before online_softmax — same
   apply_rel_bias_smem (identity tensor partition_C coords work the same
   for wgmma fragments).
5. Interface: sm_90 ctor call gets has_bias=bias is not None; window-arg
   Int32 wrap already global; compile/exec sites for sm_90 branch need
   mBias appended (mirror generic sites).
Gate: parity harness backend 1 runs on H100 only (sm_120 lacks sm_90
path) -> compile-check locally via cute.compile dry... NOT possible
locally (needs sm_90 target GPU for JIT). => Validation is REMOTE-ONLY:
bootstrap applies kernels/tml_fa4_modified/* + sm_90 edits, runs parity +
microbench + ncu in ONE H100 session (~$1-2 with debug headroom).

## sm_90 apply root cause (sessions 5-10, 2026-07-20)

Sessions 6-7: speed TARGET HIT (745us vs plain 737us vs prod 2411 = 3.24x)
but parity FAIL (bias misplaced, max ~2.3). Ruled out: warpgroup race
(s7 lockstep, no change), fragment transpose (s9 coord-swap, no change).
Session 8 debug dump fingerprint: 127/128 rows wrong, ROW 0 EXACT,
per-16-row-block err decreasing.

ROOT CAUSE: the wgmma accumulator fragment layout is
((2,2,N/8), MMA_M, MMA_N) — it does NOT index linearly as (row, col).
My apply paired acc_S[i] with partition_C(identity)[i] linearly, which is
only valid for sm_80 m16n8 frags, not sm_90 wgmma. The PROOF this was the
bug: the mask (AttentionMask.apply_mask, the only VERIFIED-correct
per-element coordinate consumer on sm_90 — attention masks correctly
without bias) never indexes linearly; it calls
quack.layout_utils.reshape_acc_to_mn(acc) to get a clean 2D (m,n) view
first, then indexes [r, c]. The tml_fa4 sm_90 apply_score_mod I originally
mirrored is NOT proven — parity backend 2 uses vllm_flash_attn's separate
copy, so I copied an unverified pattern.

FIX (session 10): rewrote apply_rel_bias_sm90 to reshape_acc_to_mn both
acc_S and the identity partition, then 2D [r,c] loop reading tile-local
sBias[(row,col)]. The bias LOAD was verified correct independently (shear
column map re-derived: sBias[r,c] = bias(global row m*128+r, kv n*128+c)).

## sm_90 native: STATUS as of 2026-07-20 (17 H100 sessions, ~$17)

VALIDATED on H100 via isolation probes (all green):
- apply_rel_bias_sm90 EXECUTES (sentinel probe: obliterating acc -> NaN output).
  Root cause of 11 dead flights: mma() call site never passed mBias/sBias ->
  mBias arrived None -> bias branch skipped -> unscaled plain attn. FIXED.
- scale/wiring correct (ZEROBIAS probe green, 0.0078 vs plain).
- row coordinate correct (ROWBIAS green + causal-mask equivalence).
- plumbing end-to-end: ctor has_bias, __call__ transpose, kernel/mma threading,
  interface arch-9 dispatch, raw-rel_logits redesign (dist=row-kv, no shear).

OPEN (paradox): real bias parity FAILS (global_short max 1.74 mean 0.064;
much improved from 2.29 but not <0.02). DISTBIAS probe (val=row_g-kv vs
bias(i,j)=i-j) RED (1.84) => the effective column/dist is wrong for the bias
LOOKUP, YET the identical column formula masks correctly (causal attention is
exact without bias). Both column forms tested — mask's t0+thr_col_offset trick
AND direct tScS_mn[r,c][1] — give byte-identical (wrong-for-bias) results at
T=128. This is a genuine wgmma-fragment-consumption subtlety: the reshape_acc_to_mn
column coord that is correct for THRESHOLD masking is not directly usable as the
exact per-element key index for a gathered bias. Needs a reference wgmma bias
kernel or local sm_90 access to resolve (remote-only blind iteration exhausted).

DECISION: sm_90 native parked at this state (best version committed). sm_120
generic kernel is FULLY PROVEN (parity 3/3, beats production). Release ships on
sm_120 proof + measured 3.2x Hopper headroom + the 11 findings. sm_90 native
documented honestly as in-final-debug. Resume with reference-kernel study.
