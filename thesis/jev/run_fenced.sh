#!/bin/sh
# Run jev_score.py inside a fence: no network (own empty network namespace) and the GitHub and SSH logins
# hidden behind empty folders. The GPU stays visible. Needs no admin rights (user namespaces).
# Run from ~/thesis:  sh bus-embeddings-l2/thesis/jev/run_fenced.sh score --out ...
set -e
here=$(cd "$(dirname "$0")" && pwd)
exec unshare -r -n -m sh -c '
  for d in "$HOME/.ssh" "$HOME/.config/gh"; do
    [ -d "$d" ] && mount -t tmpfs -o size=1m,mode=700 tmpfs "$d"
  done
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
  exec "$0"/.venv/bin/python -I "$0"/../jev_score.py "$@"
' "$here" "$@"
