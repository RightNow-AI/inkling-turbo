#!/usr/bin/env python3
"""Grab a 1x B200 on Lambda the moment capacity appears, run the scripted
first-contact unit (bootstrap + FA4 parity on sm_100), then TERMINATE.

SPENDS MONEY. This rents a GPU billed by the hour and can sit polling for
capacity for days before it fires. Set --max-hours and know the rate first.

Guaranteed-kill design: the instance is terminated in a finally block on any
outcome (success, error, Ctrl-C). Evidence lands in journal/remote/.
Passing --park disables that termination and leaves the box billing.

Usage: py scripts/grab_b200.py [--type gpu_1x_b200_sxm6] [--interval 120]
       [--max-hours 72] [--park]  (--park skips termination; NOT default)
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://cloud.lambdalabs.com/api/v1"
REPO = Path(__file__).resolve().parent.parent
SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519")
SSH_ARGS = [
    "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=" + os.devnull, "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]
PRICE_PER_HOUR = {"gpu_1x_b200_sxm6": 6.99, "gpu_2x_b200_sxm6": 13.78,
                  "gpu_8x_b200_sxm6": 53.52, "gpu_1x_h100_sxm5": 4.29,
                  "gpu_2x_h100_sxm5": 8.38}


def api(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("LAMBDA_API_KEY") or (Path.home() / ".lambda" / "api_key").read_text().strip()
    token = base64.b64encode(f"{key}:".encode()).decode()
    req = urllib.request.Request(
        API + path, method=method,
        headers={"Authorization": f"Basic {token}",
                 "Content-Type": "application/json",
                 "User-Agent": "inkling-turbo/1.0"},
        data=json.dumps(body).encode() if body else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        # Lambda puts the real reason in the JSON body; a bare "HTTP 400" is
        # undiagnosable and cost us a capacity window on 2026-07-21.
        try:
            detail = exc.read().decode()[:600]
        except Exception:  # noqa: BLE001
            detail = "<body unreadable>"
        raise RuntimeError(
            f"{method} {path} -> HTTP {exc.code}: {detail}") from exc


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def wait_for_capacity(itypes: list[str], interval: int,
                      max_hours: float) -> tuple[str, str]:
    """Poll all acceptable types; return (type, region) of the first hit
    (list order = priority when several have capacity in the same poll)."""
    deadline = time.monotonic() + max_hours * 3600
    while time.monotonic() < deadline:
        try:
            d = api("GET", "/instance-types")["data"]
            for itype in itypes:
                regions = d[itype].get("regions_with_capacity_available", [])
                if regions:
                    print(f"[{stamp()}] capacity {itype}: "
                          f"{[r['name'] for r in regions]}", flush=True)
                    return itype, regions[0]["name"]
            print(f"[{stamp()}] no capacity ({'/'.join(itypes)})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{stamp()}] API error: {exc}", flush=True)
        time.sleep(interval)
    raise TimeoutError(f"no capacity within {max_hours}h")


def launch(itype: str, region: str) -> str:
    resp = api("POST", "/instance-operations/launch", {
        "region_name": region, "instance_type_name": itype,
        "ssh_key_names": [os.environ.get("LAMBDA_SSH_KEY", "default")],
        "name": "inkling-turbo-" + itype.removeprefix("gpu_"),
        "quantity": 1,
    })
    iid = resp["data"]["instance_ids"][0]
    print(f"[{stamp()}] LAUNCHED {iid} ({itype} in {region})", flush=True)
    return iid


def wait_active(iid: str, timeout_min: int = 25) -> str:
    deadline = time.monotonic() + timeout_min * 60
    while time.monotonic() < deadline:
        inst = api("GET", f"/instances/{iid}")["data"]
        st = inst.get("status")
        if st == "active" and inst.get("ip"):
            print(f"[{stamp()}] active @ {inst['ip']}", flush=True)
            return inst["ip"]
        if st in ("terminated", "terminating", "unhealthy"):
            raise RuntimeError(f"instance entered {st}")
        time.sleep(15)
    raise TimeoutError("not active in time")


def ssh(ip: str, cmd: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_ARGS, f"ubuntu@{ip}", cmd],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def wait_ssh(ip: str, timeout_s: int = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ssh(ip, "echo up", timeout=25).returncode == 0:
            print(f"[{stamp()}] ssh up", flush=True)
            return
        time.sleep(10)
    raise TimeoutError("ssh never came up")


def scp_to(ip: str, local: Path, remote: str) -> None:
    r = subprocess.run(["scp", *SSH_ARGS, str(local), f"ubuntu@{ip}:{remote}"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"scp {local}: {r.stderr.strip()}")


def terminate(iid: str) -> None:
    try:
        api("POST", "/instance-operations/terminate", {"instance_ids": [iid]})
        print(f"[{stamp()}] TERMINATED {iid}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{stamp()}] TERMINATE FAILED for {iid}: {exc}, "
              f"KILL MANUALLY IN LAMBDA CONSOLE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="gpu_1x_b200_sxm6,gpu_2x_b200_sxm6",
                    help="acceptable types, priority order")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=72)
    ap.add_argument("--park", action="store_true",
                    help="leave instance running after bootstrap (NOT default)")
    args = ap.parse_args()

    # Double-launch guard: never act if an inkling-turbo instance already
    # exists (e.g. a second copy of this script, or a previous parked box).
    existing = [i for i in api("GET", "/instances")["data"]
                if i.get("name", "").startswith("inkling-turbo")]
    if existing:
        print(f"[{stamp()}] REFUSING to run: existing instance(s) "
              f"{[(i['name'], i['status']) for i in existing]}", flush=True)
        return 2

    itype, region = wait_for_capacity(args.types.split(","), args.interval,
                                      args.max_hours)
    args.type = itype
    t0 = time.monotonic()
    iid = launch(args.type, region)
    outdir = REPO / "journal" / "remote"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        ip = wait_active(iid)
        wait_ssh(ip)
        scp_to(ip, REPO / "scripts" / "bootstrap_b200.sh", "~/bootstrap.sh")
        for f in sorted((REPO / "harness").glob("*.py")):
            scp_to(ip, f, f"~/{f.name}")
        ssh(ip, "mkdir -p ~/tml_fa4_modified ~/kernels && touch ~/kernels/__init__.py", timeout=30)
        scp_to(ip, REPO / "kernels" / "relproj_score_mod.py", "~/kernels/relproj_score_mod.py")
        scp_to(ip, REPO / "kernels" / "patches" / "u3_fp8_kv.py", "~/u3_fp8_kv.py")
        scp_to(ip, REPO / "kernels" / "patches" / "u2_serving_route.py", "~/u2_serving_route.py")
        for f in sorted((REPO / "kernels" / "tml_fa4_modified").glob("*.py")):
            scp_to(ip, f, f"~/tml_fa4_modified/{f.name}")
        print(f"[{stamp()}] bootstrap starting (~15-25 min)", flush=True)
        r = ssh(ip, "bash ~/bootstrap.sh", timeout=2400)
        log = outdir / f"b200_first_contact_{datetime.now(timezone.utc):%Y%m%d_%H%M}.log"
        log.write_text(r.stdout + ("\n--- STDERR ---\n" + r.stderr if r.stderr else ""),
                       encoding="utf-8")
        # pull microbench JSON evidence if produced
        for jf in ("microbench_attn_day0", "microbench_attn_scoremod"):
            subprocess.run(["scp", *SSH_ARGS,
                            f"ubuntu@{ip}:~/{jf}.json",
                            str(outdir / f"{jf}_{args.type}.json")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        print(f"[{stamp()}] bootstrap rc={r.returncode}; log: {log}", flush=True)
        tail = "\n".join(r.stdout.splitlines()[-25:])
        print(tail, flush=True)
        if args.park:
            (REPO / "scripts" / ".b200_instance.json").write_text(
                json.dumps({"id": iid, "ip": ip, "launched": stamp()}))
            print(f"[{stamp()}] PARKED (billing!): {iid} @ {ip}", flush=True)
            return 0
        return 0 if r.returncode == 0 else 1
    finally:
        if not args.park:
            terminate(iid)
            hours = (time.monotonic() - t0) / 3600
            cost = hours * PRICE_PER_HOUR.get(args.type, 0.0)
            print(f"[{stamp()}] session: {hours:.2f}h ~= ${cost:.2f} "
                  f"({args.type})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
