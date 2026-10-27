#!/usr/bin/env bash
# Concordância com humanos no QASPER (deixa um anotador de fora; júri nas respostas livres) e análise.
# Segunda passada refaz votos perdidos por falha do servidor. Marcos em results/logs/pipeline.log.
set -uo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
export CUDA_VISIBLE_DEVICES=1
LOG=results/logs/pipeline.log
run() {
  local name=$1; shift
  echo "$(date -Is) iniciando $* ($name)" >> $LOG
  .venv/bin/python -m llmeval "$@" > "results/logs/$name.log" 2>&1
  local code=$?
  echo "$(date -Is) fim de $* ($name), código $code" >> $LOG
  if [ "$code" -eq 75 ]; then
    echo "$(date -Is) cadeia pausada: servidor indisponível" >> $LOG
    exit 75
  fi
}
run agreement agreement
run agreement_retry agreement
run analyze_agreement analyze
echo "$(date -Is) concordância com humanos concluída" >> $LOG
