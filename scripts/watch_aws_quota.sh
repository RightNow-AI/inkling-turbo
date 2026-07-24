#!/usr/bin/env bash
# Poll AWS quota request status; emit a line on ANY status change, exit when
# all four requests leave PENDING (approved/denied). Poll: 15 min.
CODES="L-8B23CEF3 L-9C38E5AB L-DA6814F2 L-417A185B"
declare -A last
while true; do
  all_done=1
  for code in $CODES; do
    st=$(aws service-quotas list-requested-service-quota-change-history-by-quota \
      --service-code ec2 --quota-code $code --region us-east-1 \
      --query "RequestedQuotas | sort_by(@, &Created)[-1].Status" --output text 2>/dev/null)
    [ -z "$st" ] && st="QUERY_ERROR"
    if [ "${last[$code]}" != "$st" ]; then
      echo "$(date -u '+%Y-%m-%d %H:%M UTC') $code: ${last[$code]:-initial} -> $st"
      last[$code]=$st
    fi
    case "$st" in PENDING|CASE_OPENED|QUERY_ERROR) all_done=0;; esac
  done
  [ $all_done -eq 1 ] && { echo "ALL REQUESTS RESOLVED"; exit 0; }
  sleep 900
done
