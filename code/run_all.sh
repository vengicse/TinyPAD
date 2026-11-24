#!/usr/bin/env bash
# Full healthcare pipeline as executed on RunPod (RTX A4000 pod, 14 vCPU).
set -euo pipefail
cd "$(dirname "$0")"
touch /root/RUN_STARTED 2>/dev/null || true
[ -f ../data/wesad.npz ] && [ -f ../data/mitbih.npz ] || python -u prepare_health.py ../raw ../data all
python -u dev_select.py ../data ../results/dev_select.json 6          # design selection on validation people, seed 100
python -u run_health.py --exp main --workers ${WORKERS:-6}
python -u run_health.py --exp ablation --workers ${WORKERS:-6}
python -u run_health.py --exp latency
echo ALL_DONE
