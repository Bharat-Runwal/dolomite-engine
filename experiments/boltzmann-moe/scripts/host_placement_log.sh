#!/bin/bash
# ============================================================================================
# Accumulate a picture of WHERE our runs land and how they fare there, so requeue decisions stop
# being per-incident guesses. Appends one TSV row per (check, job) to
#   experiments/boltzmann-moe/logs/host_placements.tsv
#
#   bash scripts/host_placement_log.sh record <jobid>:<logname>:<arm> [...]   # append a sample
#   bash scripts/host_placement_log.sh report                                # summarise by host
#
# WHY TWO CATEGORIES, and this is the point of the whole file:
#
#   FABRIC_FAULT -- an ncclRemoteError on the FIRST collective (SeqNum=1, "last completed work: -1").
#     This is a property OF THE HOST (or its fabric). Two independent occurrences justify adding it
#     to BAD_HOSTS in submit_train.sh, which is the bar ACCEL_FINDINGS used for p4-r10-n4.
#
#   CONTENDED -- step time regressed >=2x from the arm's own early median because NEIGHBOURS
#     ARRIVED after dispatch. This is NOT a property of the host: any host can fill up, and
#     `select[ut<0.5]` filters only at dispatch. So a contended host must NOT be blacklisted --
#     blacklisting would shrink the candidate pool for no gain. What the history IS good for is
#     spotting hosts that fill up REPEATEDLY, which is a reason to deprioritise, not exclude.
#
# Conflating the two is the mistake this file exists to prevent: on 2026-09-17 two 2-node probes
# hit SeqNum=1 faults on four distinct host pairs, and I initially added two hosts to a suspect
# list -- wrong inference, since drawing genuinely bad hosts twice in four tries among ~500 would
# be very unlikely. It read as general multi-node flakiness instead.
# ============================================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG="$REPO/experiments/boltzmann-moe/logs/host_placements.tsv"
mkdir -p "$(dirname "$LOG")"
[ -f "$LOG" ] || printf 'utc\tjobid\tarm\thost\tslots_run\tstep\tstep_time_med40\tearly_med\tratio\tverdict\n' > "$LOG"

case "${1:-report}" in
record)
  shift
  for spec in "$@"; do
    id=${spec%%:*}; rest=${spec#*:}; nm=${rest%%:*}; arm=${rest#*:}
    f="$HOME/bsub_logs/${nm}_${id}.stderr"
    stat=$(bjobs -noheader -o stat "$id" 2>/dev/null | tr -d ' '); [ -z "$stat" ] && stat=GONE
    host=$(bjobs -noheader -o exec_host "$id" 2>/dev/null | sed 's/[0-9]*\*//' | tr -d ' ')
    [ -z "$host" ] && host="-"
    slots=$(bhosts -noheader -o run "$host" 2>/dev/null | tr -d ' '); [ -z "$slots" ] && slots="-"
    step=$(grep -oE "step = [0-9]+, train-loss" "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+'); [ -z "$step" ] && step=0
    med=$(grep -oE "step_time \(sec\) = [0-9.]+" "$f" 2>/dev/null | sed 's/.*= //' | tail -40 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int(NR/2)+1]}')
    early=$(grep -oE "step_time \(sec\) = [0-9.]+" "$f" 2>/dev/null | sed 's/.*= //' | head -60 | tail -40 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int(NR/2)+1]}')
    ncc=$(tr '\r' '\n' < "$f" 2>/dev/null | grep -ac 'ncclRemoteError' || true)
    verdict=CLEAN
    if [ "${ncc:-0}" -gt 0 ] && [ "$step" -eq 0 ]; then verdict=FABRIC_FAULT
    elif [ -n "$med" ] && [ -n "$early" ]; then
      verdict=$(awk -v m="$med" -v e="$early" 'BEGIN{ if (e>0 && m/e>=2.0) print "CONTENDED"; else if (m+0>6.0) print "SLOW"; else print "CLEAN" }')
    fi
    ratio=$(awk -v m="${med:-0}" -v e="${early:-0}" 'BEGIN{ if(e>0) printf "%.2f", m/e; else print "-" }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$id" "$arm" "$host" "$slots" "$step" "${med:--}" "${early:--}" "$ratio" "$verdict" >> "$LOG"
    echo "  logged $arm ($id) host=$host slots=$slots med=${med:--}s ratio=$ratio -> $verdict"
  done
  ;;
report)
  [ -s "$LOG" ] || { echo "no samples yet"; exit 0; }
  echo "=== per-host history (samples / distinct arms / worst verdict) ==="
  awk -F'\t' 'NR>1 && $4!="-" {n[$4]++; arms[$4]=arms[$4]" "$3;
      if($10=="FABRIC_FAULT") f[$4]++; if($10=="CONTENDED"||$10=="SLOW") c[$4]++}
    END{ for(h in n) printf "  %-14s samples=%-3d fabric_faults=%-3d contended=%-3d\n", h, n[h], (h in f?f[h]:0), (h in c?c[h]:0) }' "$LOG" | sort
  echo
  echo "=== BAD_HOSTS candidates (>=2 fabric faults -- a property of the host) ==="
  awk -F'\t' 'NR>1 && $10=="FABRIC_FAULT" {f[$4]++} END{n=0; for(h in f) if(f[h]>=2){print "  "h" ("f[h]" faults) -> add to BAD_HOSTS in submit_train.sh"; n++} if(!n) print "  none"}' "$LOG"
  echo
  echo "=== repeatedly-busy hosts (>=2 contended -- deprioritise, do NOT blacklist) ==="
  awk -F'\t' 'NR>1 && ($10=="CONTENDED"||$10=="SLOW") {c[$4]++} END{n=0; for(h in c) if(c[h]>=2){print "  "h" ("c[h]" contended placements)"; n++} if(!n) print "  none"}' "$LOG"
  ;;
*) echo "usage: $0 {record <jobid>:<logname>:<arm> ... | report}" >&2; exit 2 ;;
esac
