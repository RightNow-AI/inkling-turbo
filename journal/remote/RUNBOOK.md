# Remote ops runbook

## Active grabber (detached from Claude session)
- `scripts/grab_b200.py` runs as a detached Windows process (PID recorded below),
  log: `journal/remote/grab_b200_detached.log` (+ `.err`).
- Started 2026-07-17 20:27 UTC, PID 19616, window 168h, types 1x/2x B200.
- On capacity: launches `inkling-turbo-1xb200`, bootstraps pinned fork, runs
  parity + microbench, pulls evidence to journal/remote/, TERMINATES, logs cost.
- Double-launch guard: refuses to start if an `inkling-turbo*` instance exists.
  Keep exactly ONE grabber running. In-session watchers are notify-only.
- Kill it: `Stop-Process -Id <PID>`. Check: `Get-Process -Id <PID>`;
  `Get-Content journal/remote/grab_b200_detached.log -Tail 5`.
- REQUIREMENT: laptop must stay on and awake (Windows sleep pauses polling).

## Manual capacity check
`py scripts/watch_b200.py --interval 60 --max-hours 0.05`
