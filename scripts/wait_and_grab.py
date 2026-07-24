#!/usr/bin/env python3
"""Wait until no inkling-turbo instances exist, then run grab_b200.py with
the given args. Serializes session launches against terminating instances."""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    key = os.environ.get("LAMBDA_API_KEY") or (Path.home() / ".lambda" / "api_key").read_text().strip()
    tok = base64.b64encode(f"{key}:".encode()).decode()
    for _ in range(40):
        req = urllib.request.Request(
            "https://cloud.lambdalabs.com/api/v1/instances",
            headers={"Authorization": f"Basic {tok}",
                     "User-Agent": "inkling-turbo/1.0"})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())["data"]
        if not [i for i in d if i.get("name", "").startswith("inkling-turbo")]:
            print("cleared; launching grab", flush=True)
            break
        time.sleep(15)
    repo = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "grab_b200.py"), *sys.argv[1:]])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
