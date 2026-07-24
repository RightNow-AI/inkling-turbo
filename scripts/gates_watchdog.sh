#!/bin/bash
# ON-BOX orphan protection: terminates THIS instance via the Lambda API at
# the hard cap, so a dead laptop can never leave an 8x box billing forever.
# Launch with: nohup bash ~/gates_watchdog.sh <instance_id> <cap_hours> \
#   > ~/watchdog.log 2>&1 &
# The API key is read from ~/.lambda_key (scp'd at session start).
set -u
IID="$1"
CAP_HOURS="${2:-18}"
KEY=$(cat ~/.lambda_key)
TOKEN=$(printf '%s:' "$KEY" | base64 -w0)
DEADLINE=$(( $(date +%s) + CAP_HOURS * 3600 ))

echo "[watchdog] instance $IID, hard cap ${CAP_HOURS}h, deadline $(date -u -d @${DEADLINE} +%H:%M)Z"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 300
  if [ -f ~/GATES_DONE ] || [ -f ~/GATES_FAILED ]; then
    # results ready: keep the box only 6 more hours for retrieval
    NEW_DEADLINE=$(( $(date +%s) + 6 * 3600 ))
    if [ "$NEW_DEADLINE" -lt "$DEADLINE" ]; then
      DEADLINE=$NEW_DEADLINE
      echo "[watchdog] gates finished; retrieval window 6h, new deadline $(date -u -d @${DEADLINE} +%H:%M)Z"
    fi
    break
  fi
done
while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
echo "[watchdog] hard cap reached - terminating $IID"
curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
  -H "Authorization: Basic $TOKEN" -H "Content-Type: application/json" \
  -d "{\"instance_ids\": [\"$IID\"]}"
