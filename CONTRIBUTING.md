# Contributing

The bar here is evidence, not style. A patch that is correct and slow is welcome.
A patch that is fast and unproven is not.

## Run the parity gates

From inside the vLLM checkout, with its environment active:

```bash
python harness/parity_rel_chunked_decode.py   # chunked prefill and decode, and the timed head geometry
python harness/parity_rel_varlen_batch.py     # multi-sequence batching, the production call shape
python harness/parity_rel_bias_coverage.py    # bias extent at 64K, where an oracle goes blind
python harness/parity_fa4_rel.py              # full prefill, global and sliding window
python harness/parity_kv_fp8.py               # FP8 KV writes
```

`parity_rel_chunked_decode.py` is the one that matters, and this list changed on
2026-07-26 because an audit found the previous one recommended the wrong files.
It was `parity_fa4_rel.py`, `parity_kv_fp8.py` and `parity_shear_writer.py`, and
two of those three were poor advice:

- **`parity_shear_writer.py` is not a gate and has been removed from this list.**
  It dumps the writer's emitted mapping and prints it. It has no oracle, no
  tolerance and no verdict, and it cannot exit non-zero on a wrong mapping. Under
  the rule two sections below, a check that cannot fail is not a check. It is
  still useful as a probe, and giving it a real pass criterion is an open item.
- **`parity_fa4_rel.py` covers full prefill only**, every case
  `seqlen_q == seqlen_k` at `Hq == Hkv`. It is worth running and it is not the
  one that matters, because the shapes serving actually constructs are chunked
  prefill, decode, and multi-sequence batches, and this project has twice shipped
  a defect that a full-prefill gate was structurally unable to see.

`parity_fa4_rel.py` remains valuable for what it does cover. It compares against an explicit
PyTorch oracle on three cases: a short global case, a global case that crosses
the relative-bias extent, and a sliding-window case. Green means 3/3 inside
tolerance. A skipped backend, a compile failure, a non-finite result, and a
tolerance miss all count as red.

## Rules

**A timing without a passing parity run is not accepted.** Not as a benchmark,
not as a hint, not in a comment. A kernel that returns the wrong answer can be
arbitrarily fast and it means nothing. `harness/tune_sm80.py` enforces this
directly: it refuses to report a configuration's timing unless that
configuration's own parity run was green.

**And that is not enough, which this repository learned the expensive way.** The
parity run has to cover **the shape family the timing is taken on**. The same
`tune_sm80.py` timed decode shapes, `T_q=1` against `T_k=65536`, while its
`parity_ok()` built one `cu_seqlens` and passed it as both `cu_seqlens_q` and
`cu_seqlens_k`, so it verified `seqlen_q == seqlen_k`. The kernel was wrong on
exactly the family the gate never exercised, and the Ampere tile-tuning
percentages it released are withdrawn as a result. An enforced gate aimed at the
wrong shape family is more dangerous than no gate, because it gets cited, and
this paragraph is the citation.

**A shape family is not one axis.** The fix for the above added
`seqlen_q != seqlen_k` cases by hand and left them at `Hq == Hkv`, while every
timed case runs `Hq=64` over `Hkv=8`. A second bias defect, exact at
`qhead_per_kvhead == 1` by construction, then walked through the repaired gate on
the same day. So the rule is not "add the matching parity case", it is **derive
the parity shapes from the timed ones**: `tune_sm80.py` now takes
`(Hq, Hkv, rel_extent, window)` from its own `CASES` list and varies only the
depth, because a hand-written pair can diverge on an axis nobody was watching and
did, twice. If your harness times a shape it cannot derive a parity case from,
that is the bug.

**And say what would have failed it.** A gate has to be able to fail. Record the
margin between the signal it reads and the tolerance it reads it against, per
case, and treat a margin below 1x as a failed gate rather than a passing one. One
case in `parity_rel_varlen_batch.py` prints
`<-- NO POWER: cannot fail on a dropped bias` about itself and is still counted
separately from the 11 that can.
[The record](journal/regression-ampere-tile-sweep.md), and
[the second defect](journal/regression-pack-gqa-shear-granularity.md).

**Numbers are measured or `null`.** No estimates, no roofline arithmetic standing
in for a measurement, no vendor figure quoted as if we ran it. If the gate has
not run, the cell stays `null`.

**Every number names its evidence, and its evidence class.** The strong class is
an artifact under `journal/remote/` that a reader can re-parse. The weak class is
a number transcribed into a journal session with no machine-readable record.
The weak-class numbers are enumerated in `journal/remote/README.md`, which is the
canonical list; do not restate the count here, because every document that did
drifted. Both classes are allowed; silently mixing them is not. If you add a number, add its artifact and link it. If you cannot, say which
journal session it came from and that no artifact exists.

**Hardware claims need that hardware.** An `sm_90` result requires an H100. An
`sm_80` result requires an A100. `sm_120` is not Blackwell and does not stand in
for `sm_100`, whatever the version number suggests. Compiling somewhere is not
running somewhere.

**Report the cases you lose.** We are slower than the shipped path on
sliding-window prefill and the README says so. A benchmark that only shows the
winning shapes is worse than no benchmark.

## Corrections

If a later result invalidates an earlier one, append the correction and leave the
original in place. The journal is a record, not a highlight reel. The full
evidence rules are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
