#!/usr/bin/env python3
"""Book the first available Lambda 8x node and hold it for interactive use.

SPENDS MONEY, AND KEEPS SPENDING IT. This is the one script here with no
auto-terminate. The node is named inkling-user-node, is NEVER killed by this
code, and bills by the hour from the moment it boots until you terminate it
yourself. Prices are printed before launch. Nothing will stop the meter for you.

Priority: 8x H100 > 8x A100-80GB > 8x A100-40GB. 8x B200 is deliberately left
to grab_8x_gates.py, which runs the gates first and then parks the box, serving
both purposes. SSH details are printed on success.

Usage: py scripts/book_user_node.py [--interval 30] [--max-hours 96]
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grab_b200 as gb  # api()

TYPES = ["gpu_8x_h100_sxm5", "gpu_8x_a100_80gb_sxm4", "gpu_8x_a100"]


def stamp() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--max-hours", type=float, default=96)
    args = ap.parse_args()

    existing = [i for i in gb.api("GET", "/instances")["data"]
                if i.get("name") == "inkling-user-node"]
    if existing:
        print(f"[{stamp()}] user node already booked: {existing[0]['id']}")
        return 0

    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        try:
            d = gb.api("GET", "/instance-types")["data"]
            for t in TYPES:
                info = d.get(t)
                if not info:
                    continue
                regions = info["regions_with_capacity_available"]
                if not regions:
                    continue
                price = info["instance_type"]["price_cents_per_hour"] / 100
                region = regions[0]["name"]
                print(f"[{stamp()}] stock: {t} in {region} @ ${price}/hr; "
                      f"booking NOW", flush=True)
                try:
                    r = gb.api("POST", "/instance-operations/launch", {
                        "region_name": region, "instance_type_name": t,
                        "ssh_key_names": [os.environ.get("LAMBDA_SSH_KEY", "default")],
                        "name": "inkling-user-node", "quantity": 1,
                    })
                    iid = r["data"]["instance_ids"][0]
                except Exception as exc:  # noqa: BLE001
                    print(f"[{stamp()}] BOOKING FAILED ({t}/{region}): "
                          f"{str(exc)[:200]}; continuing hunt", flush=True)
                    continue
                print(f"[{stamp()}] USER NODE BOOKED: {iid} ({t} in {region} "
                      f"@ ${price}/hr, billing from now)", flush=True)
                # wait for IP so the caller gets a ready SSH line
                for _ in range(90):
                    inst = next(i for i in gb.api("GET", "/instances")["data"]
                                if i["id"] == iid)
                    if inst["status"] == "active" and inst.get("ip"):
                        print(f"[{stamp()}] ACTIVE @ {inst['ip']} | "
                              f"ssh -i ~/.ssh/id_ed25519 ubuntu@{inst['ip']}",
                              flush=True)
                        return 0
                    time.sleep(10)
                print(f"[{stamp()}] booked but no IP yet; check console",
                      flush=True)
                return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[{stamp()}] poll error: {str(exc)[:160]}", flush=True)
        time.sleep(args.interval)
    print(f"[{stamp()}] no node within {args.max_hours}h", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
