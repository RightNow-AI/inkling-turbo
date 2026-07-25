#!/usr/bin/env python3
"""Gate: can the relative-bias attention call be captured in a CUDA graph?

WHY THIS FILE EXISTS

Every number in this repository was measured with `--enforce-eager`. Production
vLLM serves with CUDA graphs ON by default, and graph capture is not a free
pass-through: it records a stream of kernel launches once and replays the
recording. Three properties of our attention path are exactly the kind of thing
that breaks or misbehaves under capture, and none of them had ever been tested:

  1. The call launches a SEPARATE `ShearingBias` pre-kernel (plus, on the varlen
     path, `CuSeqlensToBlocksKernel` and `CuBlocksToBatchKernel`) before the
     attention kernel itself. If any of those launches does not land on the
     capture stream, capture either fails outright or, worse, succeeds while
     leaving that kernel OUTSIDE the graph. Replay would then reuse whatever the
     bias buffer happened to contain, which is a silent wrong-answer bug.

  2. `_flash_attn_fwd` allocates its intermediate buffers with `torch.empty`
     on every call, including the padded shear buffer
     `(total_q + tile_m, num_head, rel_extent_padded)`. Allocations during
     capture come from the graph's private memory pool and are frozen at the
     addresses they got during capture.

  3. It is CuTe DSL. The kernels are JIT-compiled on first sight of a shape, and
     `cute.compile` is not capture-safe. Worse, the stream is bound through
     `cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)`, so whether a
     launch lands on the capture stream depends on the tvm-ffi environment stream
     tracking `torch.cuda.current_stream()` at launch time. That is a real
     question, not an obvious yes.

If the answer is "cannot be captured", that is a genuine deployment blocker and
the README has to say so, so this file treats an exception during capture as a
RESULT to be reported rather than a crash.

WHAT IT CHECKS

Shapes and the reference oracle are imported from
`harness/parity_rel_chunked_decode.py`. No new oracle is invented here: the point
of this file is graphs, and reusing an already-calibrated reference means a
failure here cannot be blamed on a fresh untested reference. Cases are a subset
of that file's `CASES`, keeping full prefill (control), chunked prefill, deep
global decode, and sliding-window decode.

Per case:

  eager_determinism   three eager calls on the same static buffers. If those are
                      bit-identical then requiring replay to be bit-identical to
                      eager is legitimate, and the strictest check is used. If
                      eager is itself nondeterministic, its own spread becomes
                      the threshold. This baseline is measured, not assumed.
  eager_vs_reference  sanity that the call being captured is the correct call.
  capture             warm up on a side stream (3 iterations, as the torch CUDA
                      graph contract requires, which also guarantees every
                      `cute.compile` has already happened and is cached), then
                      capture with `torch.cuda.graph`. An exception here is
                      recorded as `capture_ok: false` with its text.
  replay_wrote        the graph's output tensor is filled with NaN, then the
                      graph is replayed. Any NaN left means replay did not
                      write the output at all. This matters because during
                      capture nothing executes, so a graph that recorded
                      nothing would otherwise leave a plausible-looking buffer.
  replay_vs_eager     the sharp check on correctness of the recording.
  mutate_and_replay   THE CHECK THAT CATCHES A HALF-CAPTURED PRE-KERNEL. New
                      values for q, k, v AND rel_bias are copied in place into
                      the same static buffers, a new reference is computed on
                      the host side, and the graph is replayed. If the
                      `ShearingBias` launch were not inside the graph, replay
                      would attend with the OLD bias and miss the new reference
                      by a wide margin. `mutation_signal` records how far the
                      redraw moved the reference output, so a pass here is
                      known to have had the power to fail.

Then one cross-shape section, `shared_pool_multishape`, because that is what
vLLM actually does: it captures many shapes into ONE shared graph memory pool
and replays them interleaved. Two shapes are captured into a shared pool and
replayed in an interleaved order, each output cloned immediately after its own
replay the way vLLM copies its output out.

EACH SECTION RUNS IN ITS OWN SUBPROCESS. A `cudaErrorIllegalAddress` poisons the
CUDA context for the remainder of a process (see
`journal/regression-sm120-varlen-illegal-address.md`, where one fault turned 11
independent cases into a cascade of failures), and a failed graph capture can
leave the context in a capture state. Isolating sections means the first failure
does not manufacture ten more.

NEGATIVE CONTROL, so that a green run means something

A gate that can only pass is not a gate. Two of its checks carry their own proof
of power and one was exercised by hand:

  The stale-bias failure mode. `mutation_signal_over_tol` is recorded per case
  and the run must have it at or above SIGNAL_MARGIN. On the 2026-07-25 sm_120
  run the redraw moved the reference output by 38x to 124x TOL_MEAN, and the
  JSON also records `mutated_replay_vs_stale_ref_mean`, which is literally the
  score a graph that reused the old bias would have posted: 1.9e-02 to 6.2e-02
  against a 5e-04 tolerance, so 38x to 124x over. That mode cannot hide.

  The capture-raises path. Exercised by monkeypatching `Case.call` to perform a
  device-to-host sync (`out.sum().item()`) inside the capture region, which CUDA
  forbids. The harness returned `capture_ok: false`, `pass: false`, and
  "AcceleratorError: CUDA error: operation failed due to a previous error during
  capture / cudaErrorStreamCaptureInvalidated" as the recorded error text,
  without propagating the exception. So "cannot be captured" really does come
  back as a reported result.

WHAT THIS FILE DOES NOT TELL YOU

It tells you about whichever architecture you ran it on, and only that.
`sm_90` runs `flash_fwd_sm90.py`; `sm_80` and `sm_120` run the generic
`flash_fwd.py`. The pre-kernel and the interface-level allocation behaviour are
shared, but the attention kernel is not, so a green run on one says nothing
about the other. The recorded JSON carries the compute capability for exactly
this reason.

It also uses single-sequence `cu_seqlens`, matching the file it borrows its
shapes from. Multi-sequence varlen batches are the production shape and they
currently fault on the generic path for unrelated reasons, so capturing them is
not yet a question that can be asked. See the sm_120 journal entry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_rel_chunked_decode import (  # noqa: E402
    CASES as PARITY_CASES,
    D,
    DEV,
    SIGNAL_MARGIN,
    TOL_MAX,
    TOL_MEAN,
    reference,
)

CASE_KW = dict(PARITY_CASES)

# A subset of parity_rel_chunked_decode.CASES: one full-prefill control, one
# chunked prefill, the deepest global decode that file allows, and one
# sliding-window decode. Enough to cover the shape families that matter without
# paying a fresh process and a fresh JIT compile seven times.
GRAPH_CASES = [
    "control_full_prefill",
    "chunked_global_128_on_1408",
    "decode_global_ctx4095",
    "decode_swa_ctx4095",
]

# The two shapes captured into one shared pool for the vLLM-like section.
SHARED_POOL_CASES = ["chunked_global_128_on_1408", "decode_global_ctx4095"]

WARMUP_ITERS = 3
MUTATION_SEED_OFFSET = 500_000


# --------------------------------------------------------------------------
# tensor construction, byte for byte the same as parity_rel_chunked_decode
# --------------------------------------------------------------------------

def draw(T_q, ctx, Hq, Hkv, rel_extent, seed):
    """Same order and scaling of randn calls as that file's run_case.

    Kept identical on purpose. If the draws diverged, a disagreement between
    this file and the parity gate on the same case name would be ambiguous.
    """
    torch.manual_seed(seed)
    T_k = ctx + T_q
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV)
    r_small = torch.randn(T_q, Hq, 16, dtype=torch.bfloat16, device=DEV) * 0.4
    proj = torch.randn(16, rel_extent, dtype=torch.bfloat16, device=DEV) * 0.3
    rel = (r_small.float() @ proj.float()).to(torch.bfloat16)
    return q, k, v, rel


class Case:
    """One case's STATIC buffers plus the call that runs on them.

    The buffers are allocated once and never rebound. That is what makes a
    replay meaningful: a CUDA graph records addresses, so an input the graph
    reads has to keep living at the address it had during capture.
    """

    def __init__(self, name):
        kw = CASE_KW[name]
        self.name = name
        self.T_q = kw["T_q"]
        self.ctx = kw["ctx"]
        self.T_k = kw["ctx"] + kw["T_q"]
        self.Hq = kw["Hq"]
        self.Hkv = kw["Hkv"]
        self.rel_extent = kw["rel_extent"]
        self.window_left = kw["window_left"]
        self.seed = kw["seed"]
        self.scale = 1.0 / D
        self.window = ((None, None) if self.window_left is None
                       else (self.window_left, 0))

        self.q, self.k, self.v, self.rel = draw(
            self.T_q, self.ctx, self.Hq, self.Hkv, self.rel_extent, self.seed)
        self.cu_q = torch.tensor([0, self.T_q], dtype=torch.int32, device=DEV)
        self.cu_k = torch.tensor([0, self.T_k], dtype=torch.int32, device=DEV)

    def reference(self):
        return reference(self.q, self.k, self.v, self.rel, self.scale,
                         self.ctx, self.window_left)

    def redraw_in_place(self):
        """New values, SAME storage. The graph must not notice a new address."""
        q, k, v, rel = draw(self.T_q, self.ctx, self.Hq, self.Hkv,
                            self.rel_extent, self.seed + MUTATION_SEED_OFFSET)
        self.q.copy_(q)
        self.k.copy_(k)
        self.v.copy_(v)
        self.rel.copy_(rel)

    def call(self):
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func
        out = flash_attn_varlen_func(
            q=self.q, k=self.k, v=self.v,
            rel_bias=self.rel,
            cu_seqlens_q=self.cu_q, cu_seqlens_k=self.cu_k,
            max_seqlen_q=self.T_q, max_seqlen_k=self.T_k,
            softmax_scale=self.scale, causal=True, window_size=self.window,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    def shape_dict(self):
        return {"name": self.name, "T_q": self.T_q, "ctx": self.ctx,
                "T_k": self.T_k, "Hq": self.Hq, "Hkv": self.Hkv,
                "rel_extent": self.rel_extent,
                "window_left": self.window_left, "seed": self.seed}


def diff_stats(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.mean())


def warmup_on_side_stream(case):
    """The torch CUDA graph contract: warm up on a non-default stream first.

    This also forces every `cute.compile` for this shape to happen and land in
    the module-level compile caches, so no compilation can be attempted from
    inside the capture region.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP_ITERS):
            case.call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()


# --------------------------------------------------------------------------
# section: one case, private graph pool
# --------------------------------------------------------------------------

def run_single(name):
    case = Case(name)
    r = {"section": name, "kind": "single", "shape": case.shape_dict()}

    ref = case.reference()

    # Eager determinism baseline, MEASURED. It sets how strict the
    # replay-vs-eager comparison is allowed to be.
    eager = [case.call().clone() for _ in range(3)]
    torch.cuda.synchronize()
    self_max = max(diff_stats(eager[0], e)[0] for e in eager[1:])
    r["eager_self_max"] = self_max
    r["eager_deterministic"] = self_max == 0.0

    e_max, e_mean = diff_stats(eager[0], ref)
    r["eager_vs_ref_max"] = e_max
    r["eager_vs_ref_mean"] = e_mean
    r["eager_matches_ref"] = bool(e_max <= TOL_MAX and e_mean <= TOL_MEAN)

    # Capture. Anything raised here IS the finding.
    graph = torch.cuda.CUDAGraph()
    try:
        warmup_on_side_stream(case)
        with torch.cuda.graph(graph):
            out_g = case.call()
        torch.cuda.synchronize()
        r["capture_ok"] = True
        r["capture_error"] = None
    except Exception as exc:  # noqa: BLE001
        r["capture_ok"] = False
        r["capture_error"] = f"{type(exc).__name__}: {exc}"
        r["capture_traceback"] = traceback.format_exc()
        r["pass"] = False
        return r

    # Did replay write anything at all? During capture nothing executes, so
    # without this the buffer's contents prove nothing.
    try:
        out_g.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        r["replay_ok"] = True
        r["replay_error"] = None
    except Exception as exc:  # noqa: BLE001
        r["replay_ok"] = False
        r["replay_error"] = f"{type(exc).__name__}: {exc}"
        r["replay_traceback"] = traceback.format_exc()
        r["pass"] = False
        return r

    r["replay_wrote_output"] = bool(torch.isfinite(out_g).all().item())

    rp_max, rp_mean = diff_stats(out_g, eager[0])
    r["replay_vs_eager_max"] = rp_max
    r["replay_vs_eager_mean"] = rp_mean
    r["replay_vs_eager_within_eager_noise"] = bool(rp_max <= self_max)

    rr_max, rr_mean = diff_stats(out_g, ref)
    r["replay_vs_ref_max"] = rr_max
    r["replay_vs_ref_mean"] = rr_mean
    r["replay_matches_ref"] = bool(rr_max <= TOL_MAX and rr_mean <= TOL_MEAN)

    # Replay twice more: a graph whose result drifts across replays is as
    # broken as one that is wrong once.
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    st_max, _ = diff_stats(out_g, eager[0])
    r["replay_stable_max"] = st_max
    r["replay_stable"] = bool(st_max <= max(self_max, rp_max))

    # The pre-kernel test. New inputs into the same storage, new host-side
    # reference, replay. A ShearingBias launch left outside the graph shows up
    # here and nowhere else.
    ref_old = ref
    case.redraw_in_place()
    ref_new = case.reference()
    sig_max, sig_mean = diff_stats(ref_new, ref_old)
    r["mutation_signal_max"] = sig_max
    r["mutation_signal_mean"] = sig_mean
    r["mutation_signal_over_tol"] = sig_mean / TOL_MEAN
    r["mutation_informative"] = bool(
        sig_mean / TOL_MEAN >= SIGNAL_MARGIN)

    out_g.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    mu_max, mu_mean = diff_stats(out_g, ref_new)
    r["mutated_replay_vs_ref_max"] = mu_max
    r["mutated_replay_vs_ref_mean"] = mu_mean
    r["mutated_replay_matches_ref"] = bool(
        mu_max <= TOL_MAX and mu_mean <= TOL_MEAN)
    # And how badly it would have scored against the STALE reference, which is
    # what a graph reusing a stale bias buffer would have produced.
    stale_max, stale_mean = diff_stats(out_g, ref_old)
    r["mutated_replay_vs_stale_ref_mean"] = stale_mean
    r["mutated_replay_vs_stale_ref_max"] = stale_max

    r["pass"] = bool(
        r["capture_ok"]
        and r["replay_ok"]
        and r["replay_wrote_output"]
        and r["eager_matches_ref"]
        and r["replay_vs_eager_within_eager_noise"]
        and r["replay_matches_ref"]
        and r["replay_stable"]
        and r["mutation_informative"]
        and r["mutated_replay_matches_ref"]
    )
    return r


# --------------------------------------------------------------------------
# section: two shapes, ONE shared graph pool, interleaved replay
# --------------------------------------------------------------------------

def run_shared_pool():
    """What vLLM does: many shapes, one pool, replayed in any order.

    Output is cloned immediately after each replay, which is what vLLM does
    too, so pool aliasing between the two graphs' output tensors is not a
    confound. Their INTERNAL buffers may well alias, and that is the point:
    each replay has to rebuild everything it depends on.
    """
    r = {"section": "shared_pool_multishape", "kind": "shared_pool",
         "cases": SHARED_POOL_CASES}
    cases = [Case(n) for n in SHARED_POOL_CASES]
    refs = [c.reference() for c in cases]

    eager = []
    for c in cases:
        eager.append(c.call().clone())
    torch.cuda.synchronize()

    pool = torch.cuda.graph_pool_handle()
    graphs = []
    try:
        for c in cases:
            warmup_on_side_stream(c)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                out = c.call()
            graphs.append((g, out))
        torch.cuda.synchronize()
        r["capture_ok"] = True
        r["capture_error"] = None
    except Exception as exc:  # noqa: BLE001
        r["capture_ok"] = False
        r["capture_error"] = f"{type(exc).__name__}: {exc}"
        r["capture_traceback"] = traceback.format_exc()
        r["pass"] = False
        return r

    # Interleaved, and deliberately not in capture order.
    order = [1, 0, 1, 0, 0, 1]
    per_replay = []
    try:
        for step, idx in enumerate(order):
            g, out = graphs[idx]
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            got = out.clone()
            m_max, m_mean = diff_stats(got, refs[idx])
            e_max, _ = diff_stats(got, eager[idx])
            per_replay.append({
                "step": step,
                "case": SHARED_POOL_CASES[idx],
                "vs_ref_max": m_max,
                "vs_ref_mean": m_mean,
                "vs_eager_max": e_max,
                "finite": bool(torch.isfinite(got).all().item()),
                "ok": bool(m_max <= TOL_MAX and m_mean <= TOL_MEAN
                           and torch.isfinite(got).all().item()),
            })
        r["replay_ok"] = True
        r["replay_error"] = None
    except Exception as exc:  # noqa: BLE001
        r["replay_ok"] = False
        r["replay_error"] = f"{type(exc).__name__}: {exc}"
        r["replay_traceback"] = traceback.format_exc()
        r["replays"] = per_replay
        r["pass"] = False
        return r

    r["replays"] = per_replay
    r["pass"] = bool(all(p["ok"] for p in per_replay))
    return r


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

MARK = "@@RESULT@@"
SECTIONS = GRAPH_CASES + ["shared_pool_multishape"]


def run_section(name):
    if name == "shared_pool_multishape":
        return run_shared_pool()
    return run_single(name)


def child(name):
    try:
        r = run_section(name)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        r = {"section": name, "pass": False,
             "harness_error": f"{type(exc).__name__}: {exc}"}
    cc = torch.cuda.get_device_capability(0)
    r["compute_capability"] = f"{cc[0]}.{cc[1]}"
    r["device"] = torch.cuda.get_device_name(0)
    print(MARK + json.dumps(r))
    return 0 if r.get("pass") else 1


def describe(r):
    name = r.get("section", "?")
    if r.get("harness_error"):
        return f"[{name}] FAIL: harness error: {r['harness_error']}"
    if not r.get("capture_ok"):
        return (f"[{name}] FAIL: CAPTURE RAISED: {r.get('capture_error')}"
                "\n            (that is the finding, not a harness bug)")
    if not r.get("replay_ok", True):
        return f"[{name}] FAIL: REPLAY RAISED: {r.get('replay_error')}"
    if r.get("kind") == "shared_pool":
        bad = [p for p in r.get("replays", []) if not p["ok"]]
        head = (f"[{name}] {'OK' if r.get('pass') else 'FAIL'}: captured "
                f"{len(r['cases'])} shapes into one pool, "
                f"{len(r.get('replays', [])) - len(bad)}"
                f"/{len(r.get('replays', []))} interleaved replays match ref")
        for p in r.get("replays", []):
            head += (f"\n    step{p['step']} {p['case']}: "
                     f"max={p['vs_ref_max']:.4e} mean={p['vs_ref_mean']:.4e} "
                     f"{'ok' if p['ok'] else 'BAD'}")
        return head
    lines = [f"[{name}] {'OK' if r.get('pass') else 'FAIL'}: capture ok, "
             f"replay {'wrote' if r.get('replay_wrote_output') else 'DID NOT WRITE'}"]
    lines.append(
        f"    eager vs ref      max={r['eager_vs_ref_max']:.4e} "
        f"mean={r['eager_vs_ref_mean']:.4e} "
        f"{'ok' if r['eager_matches_ref'] else 'BAD'}")
    lines.append(
        f"    eager determinism max spread over 3 calls = "
        f"{r['eager_self_max']:.4e}"
        f"{' (bit-identical)' if r['eager_deterministic'] else ''}")
    lines.append(
        f"    replay vs eager   max={r['replay_vs_eager_max']:.4e} "
        f"mean={r['replay_vs_eager_mean']:.4e} "
        f"{'ok' if r['replay_vs_eager_within_eager_noise'] else 'EXCEEDS EAGER NOISE'}")
    lines.append(
        f"    replay vs ref     max={r['replay_vs_ref_max']:.4e} "
        f"mean={r['replay_vs_ref_mean']:.4e} "
        f"{'ok' if r['replay_matches_ref'] else 'BAD'}")
    if "mutated_replay_vs_ref_max" in r:
        lines.append(
            f"    mutated replay    max={r['mutated_replay_vs_ref_max']:.4e} "
            f"mean={r['mutated_replay_vs_ref_mean']:.4e} "
            f"{'ok' if r['mutated_replay_matches_ref'] else 'BAD'}"
            f"   (stale-bias would score mean="
            f"{r['mutated_replay_vs_stale_ref_mean']:.4e}, "
            f"signal {r['mutation_signal_over_tol']:.1f}x TOL_MEAN)")
        if not r.get("mutation_informative"):
            lines.append("    <- MUTATION TOO SMALL TO PROVE ANYTHING; "
                         "change the redraw, not the tolerance")
    return "\n".join(lines)


def parent():
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")
    cc = torch.cuda.get_device_capability(0)
    path = "flash_fwd_sm90.py" if cc == (9, 0) else "flash_fwd.py"
    print(f"kernel path exercised: {path}")
    print(f"tolerance vs reference: max <= {TOL_MAX}, mean <= {TOL_MEAN} "
          f"(imported from parity_rel_chunked_decode)")
    print("each section runs in its own subprocess: a CUDA fault or a stuck "
          "capture poisons a context")
    print()

    results = {}
    failures = 0
    for name in SECTIONS:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--run", name],
            capture_output=True, text=True)
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(MARK):
                payload = json.loads(line[len(MARK):])
        if payload is None:
            payload = {
                "section": name, "pass": False,
                "harness_error": (
                    f"subprocess produced no result (exit {proc.returncode}); "
                    f"last stderr: "
                    f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'none'}"),
                "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-40:]),
            }
        results[name] = payload
        print(describe(payload))
        print()
        if not payload.get("pass"):
            failures += 1

    out = Path(__file__).with_name(
        f"repro_cuda_graph_capture_sm{cc[0]}{cc[1]}.json")
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "kernel_path": path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tol_max": TOL_MAX,
        "tol_mean": TOL_MEAN,
        "signal_margin": SIGNAL_MARGIN,
        "warmup_iters": WARMUP_ITERS,
        "sections": results,
        "passed": len(SECTIONS) - failures,
        "total": len(SECTIONS),
    }, indent=2), encoding="utf-8")

    captured = sum(1 for r in results.values() if r.get("capture_ok"))
    print(f"capture succeeded on {captured}/{len(SECTIONS)} sections")
    print(f"{len(SECTIONS) - failures}/{len(SECTIONS)} sections pass")
    print(f"saved: {out}")
    print()
    print(f"SCOPE: this result is for compute capability {cc[0]}.{cc[1]} "
          f"({path}). sm_90 runs a different forward kernel "
          f"(flash_fwd_sm90.py) and is NOT covered by this run.")
    print("PASS" if failures == 0 else "FAIL")
    raise SystemExit(1 if failures else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="internal: run one section in this process")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for s in SECTIONS:
            print(s)
        return
    if args.run:
        raise SystemExit(child(args.run))
    parent()


if __name__ == "__main__":
    main()
