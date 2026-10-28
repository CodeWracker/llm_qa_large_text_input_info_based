#!/usr/bin/env bash
# Varredura de orçamento da RQ6 (Qwen3.5-4B e 9B, as cinco condições com raciocínio).
# 1) teste de determinismo: 40 respostas refeitas com teto 8192 devem sair iguais às de 4096;
# 2) tetos 512, 1024 e 2048 (quase tudo deduzido do cache de 4096) e depois 8192 (só os itens cortados);
# 3) júri de cada teto e análise. Tudo em cache: relançar retoma de onde parou.
# Marcos em results/logs/pipeline.log. Uso: CHAIN=./run_budget.sh ./aguarda_servidor.sh (ver HANDOFF)
set -uo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
export CUDA_VISIBLE_DEVICES=1
LOG=results/logs/pipeline.log
run() {  # run <log> <args...>; código 75 = servidor indisponível: pausa a cadeia (o vigia relança)
  local name=$1; shift
  echo "$(date -Is) iniciando $* ($name)" >> $LOG
  .venv/bin/python -m llmeval "$@" > "results/logs/$name.log" 2>&1
  local code=$?
  echo "$(date -Is) fim de $* ($name), código $code" >> $LOG
  if [ "$code" -eq 75 ]; then
    echo "$(date -Is) cadeia pausada: servidor indisponível" >> $LOG
    exit 75
  fi
  if [ "$code" -ne 0 ]; then  # qualquer outra falha para a cadeia: seguir adiante marcaria etapas incompletas como prontas
    echo "$(date -Is) cadeia parada: $* falhou com código $code (ver results/logs/$name.log)" >> $LOG
    exit "$code"
  fi
}
# 27/10: o teste deu 9B 20/20 e 4B 10/20; o 4B entrou em nondeterministic_models (configs) e a cadeia segue
if ! grep -qE "orçamento: determinismo (ok|tratado)" $LOG; then
  .venv/bin/python scripts/check_budget_determinism.py 40 > results/logs/budget_determinism.log 2>&1
  same=$(grep -oP "respostas idênticas: \K[0-9]+" results/logs/budget_determinism.log)
  if [ "${same:-0}" -lt 36 ]; then
    echo "$(date -Is) orçamento: determinismo falhou (${same:-?}/40 respostas idênticas), cadeia parada" >> $LOG
    exit 2
  fi
  echo "$(date -Is) orçamento: determinismo ok (${same}/40)" >> $LOG
fi
for b in 512 1024 2048 8192; do
  run "generate_b$b" generate --budget $b
  run "generate_b${b}_retry" generate --budget $b
done
run generate_b4096 generate --budget 4096
for b in 512 1024 2048 4096 8192; do
  run "judge_b$b" judge --budget $b
done
run analyze analyze
echo "$(date -Is) varredura de orçamento concluída" >> $LOG
