# Remote session runbook

How a paid GPU session is run here. Every number in this repository came out of
one of these.

**These scripts rent GPUs and bill by the hour.** An 8x B200 node is roughly
$53/hr. Read the script and set a limit before you launch anything.

## Shape of a session

A session is bounded and has one stated payload. It does not stay alive waiting
for someone to think of the next thing to try.

1. State the hypothesis, the target architecture, the cases, the artifacts you
   expect back, and the spending limit.
2. Freeze the code and the patch payload. The box gets a fixed payload, not a
   live checkout.
3. Launch only an approved instance type.
4. Bootstrap the pinned environment, then verify at runtime that the intended
   source tree is the one being imported. This check has caught a silently stock
   build more than once.
5. Run parity first. Then baselines and candidates. Then profiling, and only if
   parity was green.
6. Copy logs and JSON into `journal/remote/`.
7. Terminate. On success, on failure, on timeout, on interruption.
8. Write the duration, the cost, the result, and the next payload into the
   journal.

## Launchers

| Script | What it does |
|---|---|
| `scripts/grab_b200.py` | Polls for single-GPU capacity, runs the first-contact payload, terminates in a `finally` block. |
| `scripts/grab_8x_gates.py` | Same for an 8-GPU node, running the full gate sequence. Refuses instance types outside its allowlist and prints the price first. |
| `scripts/aws_8x_gates.py` | The AWS equivalent, for when Lambda has no capacity. |
| `scripts/wait_and_grab.py` | Serializes launches so a new session cannot start while an old instance is still terminating. |
| `scripts/watch_b200.py` | Polls capacity only. Costs nothing. |

Capacity check, no spend:

```bash
py scripts/watch_b200.py --interval 60 --max-hours 0.05
```

## Guards, and why each one exists

- **Terminate in a `finally` block.** Any exit path kills the instance.
- **One instance per project.** A launcher refuses to start if an
  `inkling-turbo*` instance already exists. Two grabbers once raced and both
  won.
- **On-box watchdog.** `scripts/gates_watchdog.sh` terminates the instance
  through the provider API at a hard deadline, so a dead laptop cannot leave a
  node billing forever.
- **The bootstrap continues through individual harness failures**, so one paid
  session captures a full diagnostic set instead of stopping at the first error.
  The consequence: the bootstrap exit code is not the gate. The parity lines and
  the artifacts are.

## Two lessons paid for in lost work

**The watchdog deadline belongs to whoever is currently running.** The watchdog
shortens its cap once a completion marker appears. Relaunching a stage
standalone without clearing that marker and re-arming the watchdog means the box
can self-terminate mid-run. That cost us a full serving sweep at session 28.

**Pull artifacts incrementally.** Copy results back after each configuration
rather than at the end. Then a termination costs one configuration instead of
everything. Retrieval should never depend on the box outliving the run.
