#!/usr/bin/env python3
"""Hunt an 8x box, run the Inkling-turbo integration gates, terminate.

SPENDS MONEY, AND 8-GPU NODES ARE THE EXPENSIVE ONES. An 8x B200 is roughly
$53/hr. The script prints the price and refuses instance types outside its
allowlist, but it will not second-guess a rate you agreed to. Set
--max-session-hours.

Stage plan (all scripted, artifacts pulled back after each stage):
  1. bootstrap_8x.sh   -- clone/install/drift-fix/deploy + 592GB model dl
  2. parity_fa4_rel.py -- arch-local per-op parity (on 8x B200 this is the
                          FIRST on-arch validation of the sm_100 sheared path)
  3. gate_logit_parity.py -- stock vs ours 32-prompt logit gate + batched==bs1
  4. PARK for review (e2e launched separately) with a hard deadline: the
     process keeps running and terminates the instance at --max-session-hours
     no matter what. Ctrl+C also terminates.

Cost guard: prints the price and refuses types not in the allowlist.
Usage:
  py scripts/grab_8x_gates.py                     # hunt 8x B200 (preferred)
  py scripts/grab_8x_gates.py --allow-8xh100      # also accept 8x H100
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grab_b200 as gb  # api, launch, wait_active, wait_ssh, scp_to, ssh, terminate

REPO = Path(__file__).resolve().parent.parent
PRICE = {"gpu_8x_b200_sxm6": 53.52, "gpu_8x_h100_sxm5": 31.92}


def stamp() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"


def wait_for_capacity(types: list[str], interval: int, max_hours: float):
    deadline = time.monotonic() + max_hours * 3600
    while time.monotonic() < deadline:
        data = gb.api("GET", "/instance-types")["data"]
        for t in types:
            info = data.get(t)
            if info and info["regions_with_capacity_available"]:
                region = info["regions_with_capacity_available"][0]["name"]
                print(f"[{stamp()}] capacity {t} in {region}", flush=True)
                return t, region
        time.sleep(interval)
    raise TimeoutError("no 8x capacity within max-hours")


def push_payload(ip: str) -> None:
    gb.scp_to(ip, REPO / "scripts" / "bootstrap_8x.sh", "~/bootstrap_8x.sh")
    gb.scp_to(ip, REPO / "scripts" / "gate_logit_parity.py", "~/gate_logit_parity.py")
    gb.scp_to(ip, REPO / "scripts" / "gate_e2e_bench.sh", "~/gate_e2e_bench.sh")
    gb.scp_to(ip, REPO / "scripts" / "gate_summarize.py", "~/gate_summarize.py")
    gb.ssh(ip, "mkdir -p ~/tml_fa4_modified", timeout=30)
    for f in sorted((REPO / "kernels" / "tml_fa4_modified").glob("*.py")):
        gb.scp_to(ip, f, f"~/tml_fa4_modified/{f.name}")
    gb.scp_to(ip, REPO / "kernels" / "patches" / "u2_serving_route.py",
              "~/u2_serving_route.py")
    for f in sorted((REPO / "harness").glob("*.py")):
        gb.scp_to(ip, f, f"~/{f.name}")
    print(f"[{stamp()}] payload pushed", flush=True)


def run_stage(ip: str, name: str, cmd: str, timeout: int, outdir: Path) -> int:
    print(f"[{stamp()}] stage {name}: {cmd}", flush=True)
    r = gb.ssh(ip, cmd, timeout=timeout)
    # never assume the pipes decoded: a lost 8x B200 (2026-07-21) came from
    # cp1252 killing the reader thread, leaving stdout None mid-bootstrap.
    out = r.stdout or ""
    err = r.stderr or ""
    log = outdir / f"gates8x_{name}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.log"
    log.write_text(out + ("\n--- STDERR ---\n" + err if err else ""),
                   encoding="utf-8")
    tail = "\n".join(out.splitlines()[-15:])
    print(f"[{stamp()}] stage {name} rc={r.returncode}; log: {log.name}\n{tail}",
          flush=True)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-8xh100", action="store_true",
                    help="also accept 8x H100 (tight fit, W4A16 dequant)")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=96,
                    help="max hours to hunt for capacity")
    ap.add_argument("--max-session-hours", type=float, default=8,
                    help="hard deadline: instance terminated after this")
    args = ap.parse_args()

    types = ["gpu_8x_b200_sxm6"]
    if args.allow_8xh100:
        types.append("gpu_8x_h100_sxm5")

    existing = [i for i in gb.api("GET", "/instances")["data"]
                if i.get("name", "").startswith("inkling-turbo")]
    if existing:
        print(f"[{stamp()}] REFUSING: existing {[(i['name'], i['status']) for i in existing]}",
              flush=True)
        return 2

    # Capacity windows are seconds wide: a failed launch must NOT kill the
    # hunt (2026-07-21: an HTTP 400 crashed the hunter while australia-east-1
    # still had 8x B200 stock). Retry the whole find-then-launch cycle.
    deadline = time.monotonic() + args.max_hours * 3600
    iid = None
    while iid is None and time.monotonic() < deadline:
        itype, region = wait_for_capacity(types, args.interval, args.max_hours)
        print(f"[{stamp()}] launching {itype} @ ${PRICE[itype]}/hr; "
              f"hard cap {args.max_session_hours}h = "
              f"${PRICE[itype] * args.max_session_hours:.0f} max", flush=True)
        try:
            iid = gb.launch(itype, region)
        except Exception as exc:  # noqa: BLE001
            print(f"[{stamp()}] LAUNCH FAILED ({itype}/{region}): {exc}",
                  flush=True)
            print(f"[{stamp()}] resuming hunt in 15s", flush=True)
            time.sleep(15)
    if iid is None:
        print(f"[{stamp()}] no successful launch within max-hours", flush=True)
        return 3
    t0 = time.monotonic()
    outdir = REPO / "journal" / "remote"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        ip = gb.wait_active(iid)
        gb.wait_ssh(ip)
        push_payload(ip)

        rc = run_stage(ip, "bootstrap", "bash ~/bootstrap_8x.sh", 10800, outdir)
        if rc != 0:
            print(f"[{stamp()}] bootstrap FAILED; terminating to stop billing",
                  flush=True)
            return 1
        run_stage(ip, "parity",
                  "cd ~/vllm && source .venv/bin/activate && "
                  "timeout 900 python ~/parity_fa4_rel.py", 1200, outdir)
        run_stage(ip, "logit_gate",
                  "cd ~/vllm && source .venv/bin/activate && "
                  "python ~/gate_logit_parity.py", 10800, outdir)
        for jf in ("gate_logit_parity.json",):
            subprocess.run(["scp", *gb.SSH_ARGS, f"ubuntu@{ip}:~/{jf}",
                            str(outdir / jf)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=120)
        (REPO / "scripts" / ".gates8x_instance.json").write_text(
            f'{{"id": "{iid}", "ip": "{ip}", "type": "{itype}"}}')
        remaining = args.max_session_hours * 3600 - (time.monotonic() - t0)
        print(f"[{stamp()}] PARKED for e2e review: {iid} @ {ip} "
              f"({remaining / 3600:.1f}h until hard-cap termination)", flush=True)
        while time.monotonic() - t0 < args.max_session_hours * 3600:
            time.sleep(60)
        print(f"[{stamp()}] hard session cap reached", flush=True)
        return 0
    finally:
        gb.terminate(iid)
        hours = (time.monotonic() - t0) / 3600
        print(f"[{stamp()}] session: {hours:.2f}h ~= ${hours * PRICE[itype]:.2f} "
              f"({itype})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
