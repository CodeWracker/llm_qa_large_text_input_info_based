#!/usr/bin/env bash
# Experimentos complementares, em ordem: varredura de orçamento (RQ6) e concordância com humanos no QASPER.
# Código 75 de qualquer etapa = servidor indisponível (o vigia relança; o cache retoma de onde parou).
set -uo pipefail
cd "$(dirname "$0")"
./run_budget.sh || exit $?
if [ -x ./run_agreement.sh ]; then
  ./run_agreement.sh || exit $?
else
  echo "$(date -Is) run_agreement.sh ainda não existe: concordância não rodou" >> results/logs/pipeline.log
fi
