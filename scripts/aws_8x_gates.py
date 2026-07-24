#!/usr/bin/env python3
"""Launch an AWS 8-GPU P-family node and run the Inkling-turbo gates.

SPENDS MONEY, AND 8-GPU NODES ARE THE EXPENSIVE ONES. On-demand P-family
instances bill by the second with no cap of their own. Set
--max-session-hours and check the rate for the type you are requesting.

Mirrors grab_8x_gates.py staging (bootstrap -> parity -> logit gate -> park
with hard cap -> guaranteed terminate) but sources capacity from the AWS
on-demand quota (192 vCPUs, us-east-1) instead of Lambda stock.

Type preference: p6-b200 (8x B200, flagship sm_100) -> p6-b300 -> p5en
(8x H200, safe W4A16 fit) -> p5 (8x H100, tight fit, last resort).
Each type is tried across every AZ that offers it; InsufficientCapacity
falls through to the next.

Safety: InstanceInitiatedShutdownBehavior=terminate, staged auto-terminate
on bootstrap failure, hard --max-session-hours cap in a finally block, and
a terminate-confirm loop. All artifacts land in journal/remote/.

Usage: py scripts/aws_8x_gates.py [--types p6-b200.48xlarge,...]
       [--max-session-hours 5]
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grab_8x_gates as g8  # push_payload, run_stage
import grab_b200 as gb      # scp_to, ssh, SSH_ARGS

REPO = Path(__file__).resolve().parent.parent
REGION = "us-east-1"
QUOTA_VCPUS = 192
EST_PRICE = {"p6-b200.48xlarge": 122.0, "p6-b300.48xlarge": 132.0,
             "p5en.48xlarge": 85.0, "p5.48xlarge": 98.32}
DLAMI_SSM = ("/aws/service/deeplearning/ami/x86_64/"
             "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id")


def stamp() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"


# The aws CLI on this box is a pip script whose shebang CreateProcess cannot
# execute; invoke it through its own interpreter explicitly.
_AWS_WHICH = Path(shutil.which("aws") or "aws")
_AWS_SCRIPT = _AWS_WHICH.parent / "aws"  # extensionless pip script, not .CMD
_PY_FOR_AWS = _AWS_WHICH.parent.parent / "python.exe"
AWS_CMD = ([str(_PY_FOR_AWS), str(_AWS_SCRIPT)]
           if _AWS_SCRIPT.exists() and _PY_FOR_AWS.exists()
           else [str(_AWS_WHICH)])


def aws(*args: str, timeout: int = 120):
    r = subprocess.run([*AWS_CMD, *args, "--region", REGION, "--output", "json"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:3])}: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def wait_quota() -> None:
    while True:
        q = aws("service-quotas", "get-service-quota", "--service-code", "ec2",
                "--quota-code", "L-417A185B")
        val = q["Quota"]["Value"]
        if val >= QUOTA_VCPUS:
            print(f"[{stamp()}] quota live: {val} vCPUs", flush=True)
            return
        print(f"[{stamp()}] quota still {val}, waiting for {QUOTA_VCPUS}...",
              flush=True)
        time.sleep(120)


def ensure_keypair() -> str:
    name = "inkling-turbo"
    try:
        aws("ec2", "describe-key-pairs", "--key-names", name)
        return name
    except RuntimeError:
        pass
    pub = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
    # awscli v1 expects the raw OpenSSH text and base64-encodes it itself
    aws("ec2", "import-key-pair", "--key-name", name,
        "--public-key-material", pub)
    print(f"[{stamp()}] key pair imported: {name}", flush=True)
    return name


def ensure_sg() -> str:
    name = "inkling-turbo-ssh"
    try:
        r = aws("ec2", "describe-security-groups", "--group-names", name)
        return r["SecurityGroups"][0]["GroupId"]
    except RuntimeError:
        pass
    r = aws("ec2", "create-security-group", "--group-name", name,
            "--description", "inkling-turbo ssh")
    sg = r["GroupId"]
    aws("ec2", "authorize-security-group-ingress", "--group-id", sg,
        "--protocol", "tcp", "--port", "22", "--cidr", "0.0.0.0/0")
    print(f"[{stamp()}] security group created: {sg}", flush=True)
    return sg


def get_ami() -> str:
    # IAM user lacks ssm:GetParameter; find the newest DLAMI base via EC2.
    r = aws("ec2", "describe-images", "--owners", "amazon",
            "--filters",
            "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI "
            "(Ubuntu 22.04) *",
            "Name=state,Values=available",
            "--query", "sort_by(Images,&CreationDate)[-1].[ImageId,Name]")
    ami, name = r[0], r[1]
    print(f"[{stamp()}] DLAMI: {ami} ({name})", flush=True)
    return ami


def type_azs(itype: str) -> list[str]:
    r = aws("ec2", "describe-instance-type-offerings",
            "--location-type", "availability-zone",
            "--filters", f"Name=instance-type,Values={itype}")
    return sorted(o["Location"] for o in r["InstanceTypeOfferings"])


def subnet_for_az(az: str) -> str | None:
    r = aws("ec2", "describe-subnets",
            "--filters", f"Name=availability-zone,Values={az}",
            "Name=default-for-az,Values=true")
    subs = r["Subnets"]
    return subs[0]["SubnetId"] if subs else None


def try_launch(itype: str, ami: str, key: str, sg: str) -> str | None:
    for az in type_azs(itype):
        subnet = subnet_for_az(az)
        if subnet is None:
            continue
        try:
            r = aws("ec2", "run-instances", "--instance-type", itype,
                    "--image-id", ami, "--key-name", key,
                    "--security-group-ids", sg, "--subnet-id", subnet,
                    "--count", "1",
                    "--instance-initiated-shutdown-behavior", "terminate",
                    "--block-device-mappings",
                    '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,'
                    '"VolumeType":"gp3","DeleteOnTermination":true}}]',
                    "--tag-specifications",
                    '[{"ResourceType":"instance","Tags":[{"Key":"Name",'
                    '"Value":"inkling-turbo-8x"}]}]',
                    timeout=180)
            iid = r["Instances"][0]["InstanceId"]
            print(f"[{stamp()}] LAUNCHED {iid} ({itype} in {az})", flush=True)
            return iid
        except RuntimeError as exc:
            msg = str(exc)
            if ("InsufficientInstanceCapacity" in msg or "Unsupported" in msg
                    or "InsufficientCapacity" in msg):
                print(f"[{stamp()}] no capacity: {itype} {az}", flush=True)
                continue
            # Transient local/network faults ("Could not connect to the
            # endpoint URL", throttling, timeouts) must NOT end the hunt: on
            # 2026-07-22 one network blip killed a 24h sweep outright.
            if ("Could not connect" in msg or "EndpointConnectionError" in msg
                    or "RequestLimitExceeded" in msg or "Throttling" in msg
                    or "timed out" in msg.lower()
                    or "ServiceUnavailable" in msg):
                print(f"[{stamp()}] transient ({itype} {az}): {msg[:120]}",
                      flush=True)
                time.sleep(20)
                continue
            raise
    return None


def wait_ip(iid: str) -> str:
    for _ in range(60):
        r = aws("ec2", "describe-instances", "--instance-ids", iid)
        inst = r["Reservations"][0]["Instances"][0]
        if inst["State"]["Name"] == "running" and inst.get("PublicIpAddress"):
            print(f"[{stamp()}] running @ {inst['PublicIpAddress']}", flush=True)
            return inst["PublicIpAddress"]
        time.sleep(10)
    raise TimeoutError("instance never reached running+IP")


def wait_ssh(ip: str) -> None:
    for _ in range(60):
        r = subprocess.run(["ssh", *gb.SSH_ARGS, f"ubuntu@{ip}", "true"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if r.returncode == 0:
            print(f"[{stamp()}] ssh up", flush=True)
            return
        time.sleep(10)
    raise TimeoutError("ssh never came up")


def terminate(iid: str) -> None:
    try:
        aws("ec2", "terminate-instances", "--instance-ids", iid)
        for _ in range(40):
            r = aws("ec2", "describe-instances", "--instance-ids", iid)
            state = r["Reservations"][0]["Instances"][0]["State"]["Name"]
            if state in ("terminated", "shutting-down"):
                print(f"[{stamp()}] TERMINATED {iid} ({state})", flush=True)
                return
            time.sleep(15)
        print(f"[{stamp()}] WARNING: {iid} not confirmed terminated, "
              f"CHECK THE EC2 CONSOLE", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{stamp()}] TERMINATE FAILED {iid}: {exc}, "
              f"KILL MANUALLY IN EC2 CONSOLE", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="p6-b200.48xlarge,p6-b300.48xlarge,"
                    "p5en.48xlarge,p5.48xlarge")
    ap.add_argument("--max-session-hours", type=float, default=5)
    ap.add_argument("--retry-minutes", type=float, default=10,
                    help="re-sweep all types on full ICE for this long total")
    args = ap.parse_args()

    wait_quota()
    key = ensure_keypair()
    sg = ensure_sg()
    ami = get_ami()

    iid = None
    itype = None
    sweep_deadline = time.monotonic() + args.retry_minutes * 60
    while iid is None:
        for t in args.types.split(","):
            # belt-and-braces: nothing short of KeyboardInterrupt may end a
            # multi-hour hunt. Unexpected faults log and cost one type-slot.
            try:
                iid = try_launch(t, ami, key, sg)
            except Exception as exc:  # noqa: BLE001
                print(f"[{stamp()}] sweep error on {t}: "
                      f"{str(exc)[:160]}; continuing", flush=True)
                iid = None
            if iid:
                itype = t
                break
        if iid is None:
            if time.monotonic() > sweep_deadline:
                print(f"[{stamp()}] ALL TYPES ICE for {args.retry_minutes} min; "
                      f"giving up (relaunch later)", flush=True)
                return 3
            print(f"[{stamp()}] all types ICE; re-sweeping in 60s", flush=True)
            time.sleep(60)

    price = EST_PRICE.get(itype, 130.0)
    print(f"[{stamp()}] est ${price}/hr; hard cap {args.max_session_hours}h = "
          f"${price * args.max_session_hours:.0f} max", flush=True)
    t0 = time.monotonic()
    outdir = REPO / "journal" / "remote"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        ip = wait_ip(iid)
        wait_ssh(ip)
        g8.push_payload(ip)
        rc = g8.run_stage(
            ip, "bootstrap",
            "MODEL_DIR=/opt/dlami/nvme/models/inkling bash ~/bootstrap_8x.sh",
            10800, outdir)
        if rc != 0:
            print(f"[{stamp()}] bootstrap FAILED; terminating", flush=True)
            return 1
        g8.run_stage(ip, "parity",
                     "cd ~/vllm && source .venv/bin/activate && "
                     "timeout 900 python ~/parity_fa4_rel.py", 1200, outdir)
        g8.run_stage(ip, "logit_gate",
                     "cd ~/vllm && source .venv/bin/activate && "
                     "MODEL_DIR=/opt/dlami/nvme/models/inkling "
                     "python ~/gate_logit_parity.py", 10800, outdir)
        subprocess.run(["scp", *gb.SSH_ARGS,
                        f"ubuntu@{ip}:~/gate_logit_parity.json",
                        str(outdir / "gate_logit_parity.json")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        (REPO / "scripts" / ".gates8x_instance.json").write_text(
            json.dumps({"id": iid, "ip": ip, "type": itype, "cloud": "aws"}))
        remaining = args.max_session_hours * 3600 - (time.monotonic() - t0)
        print(f"[{stamp()}] PARKED for e2e review: {iid} @ {ip} "
              f"({remaining / 3600:.1f}h to hard cap)", flush=True)
        while time.monotonic() - t0 < args.max_session_hours * 3600:
            time.sleep(60)
        print(f"[{stamp()}] hard session cap reached", flush=True)
        return 0
    finally:
        terminate(iid)
        hours = (time.monotonic() - t0) / 3600
        print(f"[{stamp()}] session: {hours:.2f}h ~= ${hours * price:.2f} "
              f"({itype})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
