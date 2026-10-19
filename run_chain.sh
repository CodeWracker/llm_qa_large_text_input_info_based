#!/usr/bin/env bash
# Cadeia completa: júri do desenho principal -> (análise intermediária em paralelo) ->
# geração (busca + raciocínio; 2ª passada refaz falhas de servidor, o resto vem do cache) ->
# júri -> métricas -> análise. Marcos em results/logs/pipeline.log.
set -uo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
export CUDA_VISIBLE_DEVICES=1
run() {  # run <estágio> <log>
  echo "$(date -Is) iniciando $1 ($2)" >> results/logs/pipeline.log
  .venv/bin/python -m llmeval "$1" > "results/logs/$2.log" 2>&1
  echo "$(date -Is) fim de $1 ($2), código $?" >> results/logs/pipeline.log
}
run judge judge
( run analyze analyze_interim; echo "$(date -Is) intermediário concluído" >> results/logs/pipeline.log ) &
run generate generate
run generate generate_retry
run judge judge_final
run metrics metrics
wait
run analyze analyze
echo "$(date -Is) pipeline concluído" >> results/logs/pipeline.log
