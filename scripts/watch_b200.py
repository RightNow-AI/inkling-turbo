#!/usr/bin/env python3
"""Poll Lambda for 8x B200 capacity; exit 0 the moment any target type has capacity.

Usage: py scripts/watch_b200.py [--interval 300] [--max-hours 72]
Key: $LAMBDA_API_KEY, or ~/.lambda/api_key
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TARGETS = ["gpu_8x_b200_sxm6", "gpu_2x_b200_sxm6", "gpu_1x_b200_sxm6"]
API = "https://cloud.lambdalabs.com/api/v1/instance-types"


def check(key: str, targets: list[str]) -> dict[str, list[str]]:
    token = base64.b64encode(f"{key}:".encode()).decode()
    req = urllib.request.Request(
        API,
        headers={
            "Authorization": f"Basic {token}",
            "User-Agent": "inkling-turbo-capacity-watch/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())["data"]
    return {
        t: [r["name"] for r in data[t].get("regions_with_capacity_available", [])]
        for t in targets
        if t in data
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=72)
    ap.add_argument("--types", default=",".join(DEFAULT_TARGETS))
    args = ap.parse_args()
    targets = args.types.split(",")

    key = os.environ.get("LAMBDA_API_KEY") or (Path.home() / ".lambda" / "api_key").read_text().strip()
    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            avail = {t: r for t, r in check(key, targets).items() if r}
        except Exception as exc:  # noqa: BLE001 - keep polling through blips
            print(f"[{stamp}] API error: {exc}", flush=True)
            time.sleep(args.interval)
            continue
        if avail:
            print(f"[{stamp}] B200 CAPACITY FOUND: {json.dumps(avail)}", flush=True)
            return 0
        print(f"[{stamp}] no B200 capacity", flush=True)
        time.sleep(args.interval)
    print("watch window expired with no capacity", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
