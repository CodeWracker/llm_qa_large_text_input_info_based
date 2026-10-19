#!/usr/bin/env bash
# Vigia o servidor Ollama compartilhado e roda run_chain.sh assim que ele fica livre.
# Livre = nenhum modelo alheio ao experimento carregado em /api/ps por LIVRE_N checagens seguidas
# (intervalo de INTERVALO s). Se a cadeia pausar (código 75), volta a vigiar; termina quando a cadeia
# conclui. Uso: setsid nohup ./aguarda_servidor.sh > results/logs/aguarda.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
INTERVALO=${INTERVALO:-60}
LIVRE_N=${LIVRE_N:-3}
LOG=results/logs/pipeline.log
echo $$ > results/logs/aguarda.pid

# modelos alheios carregados agora (vazio = livre); "erro" se /api/ps não respondeu
alheios() {
  .venv/bin/python - <<'EOF'
import json, os, subprocess, yaml
c = yaml.safe_load(open("configs/experiment.yaml"))
nossos = {g["model"] for g in c["generators"]} | {j["model"] for j in c["jury"]["judges"]}
r = subprocess.run(["curl", "-s", "-m", "30", "--netrc", os.environ["OLLAMA_URL"].rstrip("/") + "/api/ps"],
                   capture_output=True, text=True)
try:
    print(" ".join(m["name"] for m in json.loads(r.stdout)["models"] if m["name"] not in nossos))
except (ValueError, KeyError):
    print("erro")
EOF
}

echo "$(date -Is) vigia iniciado (pid $$)" >> "$LOG"
while true; do
  livres=0
  while [ "$livres" -lt "$LIVRE_N" ]; do
    a=$(alheios)
    if [ -z "$a" ]; then livres=$((livres + 1)); else livres=0; fi
    [ "$livres" -lt "$LIVRE_N" ] && sleep "$INTERVALO"
  done
  echo "$(date -Is) vigia: servidor livre, relançando a cadeia" >> "$LOG"
  ./run_chain.sh
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$(date -Is) vigia: cadeia concluída, encerrando" >> "$LOG"; break
  fi
  echo "$(date -Is) vigia: cadeia saiu com código $code, voltando a vigiar" >> "$LOG"
  sleep 600
done
rm -f results/logs/aguarda.pid
