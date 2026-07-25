# Contributing

The bar here is evidence, not style. A patch that is correct and slow is welcome.
A patch that is fast and unproven is not.

## Run the parity gates

From inside the vLLM checkout, with its environment active:

```bash
python harness/parity_fa4_rel.py       # attention, global and sliding window
python harness/parity_kv_fp8.py        # FP8 KV writes
python harness/parity_shear_writer.py  # shear layout contract
```

`parity_fa4_rel.py` is the one that matters. It compares against an explicit
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
this paragraph is the citation. If you add a case to a timing harness, add the
matching parity case in the same commit.
[The record](journal/regression-ampere-tile-sweep.md).

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
