#!/usr/bin/env bash
# Aguarda o processo $1 terminar e roda os estágios seguintes (padrão: judge metrics analyze).
# Logs em results/logs/<estágio>.log; marcos em results/logs/pipeline.log.
# Uso: ./run_pipeline_after.sh <PID> [estágio ...]
set -euo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
pid=$1; shift
stages=("${@:-judge metrics analyze}")
[ $# -eq 0 ] && stages=(judge metrics analyze)
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
for stage in "${stages[@]}"; do
  echo "$(date -Is) iniciando $stage" >> results/logs/pipeline.log
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m llmeval "$stage" > "results/logs/$stage.log" 2>&1
done
echo "$(date -Is) concluído: ${stages[*]}" >> results/logs/pipeline.log
