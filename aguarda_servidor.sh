#!/usr/bin/env bash
# Vigia o servidor Ollama compartilhado e roda run_chain.sh assim que ele fica livre.
# Livre = por LIVRE_N checagens seguidas (intervalo de INTERVALO s), nenhum modelo alheio ao experimento
# em /api/ps ou só modelos alheios ociosos (expires_at parado = nenhuma requisição; o Ollama descarrega
# um modelo ocioso quando outro é pedido). Modelo alheio com keep_alive infinito conta sempre como ocupado:
# seu expires_at não muda com o uso. Se a cadeia pausar (código 75), volta a vigiar;
# termina quando a cadeia conclui. Uso: setsid nohup ./aguarda_servidor.sh > results/logs/aguarda.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
source /etc/profile.d/99-persist-env.sh
INTERVALO=${INTERVALO:-60}
LIVRE_N=${LIVRE_N:-3}
LOG=results/logs/pipeline.log
echo $$ > results/logs/aguarda.pid

# modelos alheios carregados agora como "nome@expires_at" (vazio = nenhum); "erro" se /api/ps não respondeu
alheios() {
  .venv/bin/python - <<'EOF'
import json, os, subprocess, time, yaml
c = yaml.safe_load(open("configs/experiment.yaml"))
nossos = {g["model"] for g in c["generators"]} | {j["model"] for j in c["jury"]["judges"]}
r = subprocess.run(["curl", "-s", "-m", "30", "--netrc", os.environ["OLLAMA_URL"].rstrip("/") + "/api/ps"],
                   capture_output=True, text=True)
def marca(m):
    # keep_alive infinito: expires_at fica parado (ano ~2318) mesmo com o modelo em uso, então não serve
    # como sinal de ociosidade; o modelo conta como ocupado (marca muda a cada checagem)
    exp = m.get("expires_at") or ""
    return f"{m['name']}@ocupado-{time.time()}" if exp[:4].isdigit() and int(exp[:4]) > time.gmtime().tm_year + 1 \
        else f"{m['name']}@{exp}"
try:
    print(" ".join(marca(m) for m in json.loads(r.stdout)["models"] if m["name"] not in nossos))
except (ValueError, KeyError):
    print("erro")
EOF
}

echo "$(date -Is) vigia iniciado (pid $$)" >> "$LOG"
while true; do
  livres=0; antes=""
  while [ "$livres" -lt "$LIVRE_N" ]; do
    a=$(alheios)
    if [ "$a" = "erro" ]; then livres=0; a=""
    elif [ -z "$a" ] || [ "$a" = "$antes" ]; then livres=$((livres + 1))
    else livres=0; fi
    antes=$a
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
