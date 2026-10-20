#!/usr/bin/env bash
# Cadeia completa, em duas fases. Fase 1, desenho principal (11 condições, sem raciocínio):
# geração (o que falta) -> júri -> métricas -> análise. Fase 2, RQ6: passada com raciocínio
# (2ª passada refaz falhas de servidor) -> júri -> métricas -> análise final.
# Tudo fica em cache: relançar retoma de onde parou. Marcos em results/logs/pipeline.log.
set -uo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
export CUDA_VISIBLE_DEVICES=1
run() {  # run <log> <args...>; código 75 = servidor indisponível: pausa a cadeia (o vigia relança)
  local name=$1; shift
  echo "$(date -Is) iniciando $* ($name)" >> results/logs/pipeline.log
  .venv/bin/python -m llmeval "$@" > "results/logs/$name.log" 2>&1
  local code=$?
  echo "$(date -Is) fim de $* ($name), código $code" >> results/logs/pipeline.log
  if [ "$code" -eq 75 ]; then
    echo "$(date -Is) cadeia pausada: servidor indisponível" >> results/logs/pipeline.log
    exit 75
  fi
}
run generate_main generate --no-think
run judge_main judge
run metrics_main metrics
run analyze_main analyze
echo "$(date -Is) fase 1 (desenho principal) concluída" >> results/logs/pipeline.log
run generate generate
run generate_retry generate
run judge_final judge
run metrics metrics
run analyze analyze
echo "$(date -Is) pipeline concluído" >> results/logs/pipeline.log
