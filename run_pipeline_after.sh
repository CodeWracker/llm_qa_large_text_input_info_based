#!/usr/bin/env bash
# Aguarda a geração (PID em $1) e roda júri -> métricas -> análise, com logs em results/logs/.
set -euo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
while kill -0 "$1" 2>/dev/null; do sleep 60; done
for stage in judge metrics analyze; do
  echo "$(date -Is) iniciando $stage" >> results/logs/pipeline.log
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m llmeval "$stage" > "results/logs/$stage.log" 2>&1
done
echo "$(date -Is) pipeline concluído" >> results/logs/pipeline.log
