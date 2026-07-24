# Regression: `3b78fc6` broke every sm_90 attention call

Found 2026-07-25 by the first Hopper run of the shear-fusion gate (session 26,
one H100 on Modal, $0.41). Fixed the same day. This file exists because the
defect reached `origin/main` and was public for several hours, and because the
mechanism is a CuTe DSL scoping rule that is easy to trip again.

## Symptom

Every attention call through `vllm/third_party/tml_fa4` failed at trace time:

```
DSLUserCodeError: NameError in `__call__`: cannot access local variable
'n_block' where it is not associated with a value
  --> flash_fwd_sm90.py:893
```

Blast radius on the session-26 run:

- `parity_fa4_rel`: backend `tml_fa4_rel_bias` failed all 3 cases. `score_mod`
  and `relproj_v1` passed all 3, because
  `harness/parity_fa4_rel.py` imports those from `vllm.vllm_flash_attn.cute`,
  which the deploy leaves stock. That contrast is what localised it to our tree.
- `microbench_attn_day0`: all 7 attention cases null. Only the two
  `gate_select_*` cases, which never touch attention, produced numbers.
- `parity_shear_fusion`: 14/16. The 14 writer cases pass; the 2
  `attention_consumes_*` cases fail.
- All 4 `presheared_*` and both `splitkv_*` microbenchmarks failed.

The step verdicts still read PASS for the microbenches, because the harness
catches each case separately and the process exits 0. **The rc was not the
result; the JSON was.** Worth remembering when reading any harness verdict.

## Mechanism

`if has_work:` is not wrapped in `const_expr(...)`, so the CuTe DSL traces each
of its bodies as a **separate Python scope**. `3b78fc6` moved the `n_block` and
`page_idx` bindings from function scope into the first such body:

```
3b78fc6:
  L834  if const_expr(not self.use_block_sparsity):     indent 16
  L848      if has_work:                               indent 20   <- dynamic
  L853          n_block = (...)                        indent 24   <- binding
  L888      if has_work:                               indent 20   <- dynamic
  L893              load_V(block=n_block, ...)                     <- read
  L899              for i in cutlass.range(...): n_block = ...      <- assign
```

The read at 893 and the assignment at 899 sit in the same traced body, so
`n_block` is a **local of that body** and the read precedes it. The only
assignment that dominates it in source order, line 853, lives in the sibling
body and is invisible. Hence "cannot access local variable", not "free
variable in enclosing scope"; the wording is the tell.

The trigger is `const_expr(not self.use_block_sparsity)` being true, which it is
for every Inkling call. **It is not conditional on split-KV.**
`flash_attn_varlen_func` defaults `num_splits=1` and none of the harnesses pass
it, so `is_split_kv` was False and line 845 set `has_work = True`, a plain
Python bool, and it still failed. The comment claiming `if has_work:` was "a
trace-time no-op there" was simply wrong, and it was load-bearing in the review
that let this through.

## Why the earlier version worked

`f585227`, the file that measured 3308.8 / 1223.0 / 852.6 us/iter on an H100 in
session 25, has the same read-then-assign shape but binds at function scope:

```
f585227:
  L800  if const_expr(not self.use_block_sparsity):     indent 16
  L808      n_block = (...)                            indent 20   <- binding
  L843          load_K(block=n_block, ...)                         <- read
```

`git diff d58d523 f585227 -- kernels/tml_fa4_modified/flash_fwd_sm90.py` is
empty, and `git log --all -- <that file>` lists exactly two commits, so the
regression window is precisely `3b78fc6`, whose own message says "Add split-KV
decode, **unvalidated**".

## The fix

Hoist both bindings back to function scope, ahead of both dynamic bodies. Not
initialisation to a dummy: it recomputes the true value, is pure index
arithmetic with no side effects, and every memory operation stays under
`if has_work:`.

```
fixed:
  L834  if const_expr(not self.use_block_sparsity):     indent 16
  L869      n_block = (...)                            indent 20   <- binding
  L891      if has_work:                               indent 20
  L916      if has_work:                               indent 20
  L921              load_V(block=n_block, ...)                     <- read
```

The variant deliberately **avoided** is `n_block = Int32(0)` before the branch.
That also compiles, but it makes "block 0" the fallback, so a wrong guard would
silently read the wrong KV tile instead of crashing. A crash is better than a
quietly wrong kernel.

One addition the hoist forces: on the split-KV path an empty split can leave
`n_block` at -1, and the `mPageTable[batch_idx, n_block]` read now runs before
`has_work` is consulted. The index is clamped under
`const_expr(self.is_split_kv)`; the non-split path keeps the original
expression verbatim so it stays byte-identical to what session 25 measured.

## What this means for the published numbers

The 2.73x / 2.80x / 2.45x / 1.45x speedups and the 1.28x sliding-window loss
were all measured on a kernel byte-identical to `f585227`. They were
**reproducible from `f585227` and not reproducible from `3b78fc6`**, because
`3b78fc6` cannot compile a single sm_90 attention call. Every green sm_90 result
in the record (session 25 parity 3/3, session 26 A100, session 28's 32/32 greedy
end-to-end) predates `3b78fc6`.

The numbers were never wrong. For a few hours the code that was supposed to
produce them was.

## Second defect found in the same diff

`kernels/tml_fa4_modified/interface.py` lines 978 and 986 read
`t.shape == (batch_size)` and `.shape == (1)`. Those are the em-dash cleanup
regex's damage to single-element tuples, at two sites the repair in `8c91a86`
missed. Comparing a `torch.Size` to an `int` is always False, so both asserts
fire for any caller passing `scheduler_metadata` without
`disable_scheduler_metadata`. Session 25 had `(batch_size, )` and `(1, )`.
Fixed, and a tree-wide sweep for the pattern now returns zero.

## Lessons

1. **A commit that says "unvalidated" must not be the tip of a public branch.**
   `3b78fc6` said so in its own message and was pushed to `main` anyway.
2. **A harness exit code is not a result.** Four steps reported PASS while every
   case inside them returned null. Read the artifact.
3. **Static analysis did not settle this.** A scope-aware AST scanner flagged the
   working file and the broken file identically, because the distinguishing fact
   is which scope a binding lands in relative to a *dynamic* branch, and a naive
   walker cannot see the DSL's rewriting rule. What settled it was comparing
   binding placement against the exact commit whose numbers are published, and
   then running it. One small GPU is worth a lot of reasoning here.
4. In CuTe DSL, treat every non-`const_expr` `if` as a function boundary for the
   purposes of variable binding, and bind loop-carried names outside it.
