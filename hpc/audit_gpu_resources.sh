#!/usr/bin/env bash
# Read-only Slurm GPU inventory and access audit for Cognition.

set -uo pipefail

section() {
  printf '\n[%s]\n' "$1"
}

run_optional() {
  "$@" 2>&1 || printf 'Command unavailable or access denied: %s\n' "$*"
}

section "Identity"
printf 'Timestamp (UTC): '
date -u '+%Y-%m-%d %H:%M:%S'
printf 'User: %s\n' "${USER}"
printf 'Login host: %s\n' "$(hostname)"

section "GPU partitions"
run_optional sinfo -o '%20P %10a %14l %8D %12t %40G'

section "GPU nodes"
run_optional sinfo -N -o '%20P %30N %12t %20C %12m %40G'

section "Queue pressure by partition and state"
if command -v squeue >/dev/null 2>&1; then
  squeue -h -o '%P|%T' \
    | awk -F'|' '
        { key=$1 "|" $2; count[key]++ }
        END {
          for (key in count) {
            split(key, fields, "|")
            printf "%-20s %-12s %6d\n", fields[1], fields[2], count[key]
          }
        }
      ' \
    | sort
else
  printf 'squeue is unavailable\n'
fi

section "My jobs"
run_optional squeue -u "${USER}" \
  -o '%.18i %.16P %.20j %.2t %.10M %.19S %.6D %R'

section "Estimated starts for my pending jobs"
run_optional squeue --start -u "${USER}" \
  -o '%.18i %.16P %.20j %.2t %.19S %R'

section "My scheduling priority"
run_optional sprio -u "${USER}" -l

section "My Slurm associations"
run_optional sacctmgr -n -P show assoc where user="${USER}" \
  format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRES,MaxTRESPerJob

section "GPU partition access rules"
if command -v sinfo >/dev/null 2>&1; then
  while IFS='|' read -r partition _gres; do
    partition="${partition%\*}"
    [[ -n "${partition}" ]] || continue
    printf '\nPartition %s\n' "${partition}"
    run_optional scontrol show partition "${partition}" -o
  done < <(
    sinfo -h -o '%P|%G' \
      | awk -F'|' '$2 ~ /gpu/ && !seen[$1]++ { print $1 "|" $2 }'
  )
fi

section "Five-minute one-GPU scheduling estimates"
printf '%s\n' \
  'These are sbatch --test-only checks. They validate access and estimate' \
  'scheduling without submitting or consuming a GPU.'
if command -v sinfo >/dev/null 2>&1; then
  while IFS='|' read -r partition gres_field; do
    partition="${partition%\*}"
    [[ -n "${partition}" ]] || continue
    if [[ "${gres_field}" =~ gpu:([^,:()]+):[0-9]+ ]]; then
      gres_request="gpu:${BASH_REMATCH[1]}:1"
    else
      gres_request="gpu:1"
    fi
    printf '\n%-20s request=%s\n' "${partition}" "${gres_request}"
    run_optional sbatch \
      --test-only \
      --partition="${partition}" \
      --gres="${gres_request}" \
      --cpus-per-task=2 \
      --mem=8G \
      --time=00:05:00 \
      --job-name=gpu-audit \
      --wrap='hostname'
  done < <(
    sinfo -h -o '%P|%G' \
      | awk -F'|' '$2 ~ /gpu/ && !seen[$1]++ { print $1 "|" $2 }'
  )
fi

section "Interpretation"
printf '%s\n' \
  'idle: a node has immediately available capacity.' \
  'mix: some resources are allocated and some may remain available.' \
  'alloc: the node is fully allocated.' \
  'drain/down: the node is unavailable.' \
  'PD(Resources): waiting for requested hardware to become free.' \
  'PD(Priority): resources may exist, but other jobs currently rank higher.' \
  'A short test-only start estimate is useful for choosing a debug partition.'
