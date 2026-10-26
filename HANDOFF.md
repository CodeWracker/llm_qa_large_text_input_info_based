# Handoff (ler primeiro, ~2 min)

Estado do trabalho e progresso: `TODO.md`. Desenho e reprodução: `README.md`.

## O que é
Revisão de um estudo engavetado: LLMs abertos (Ollama UFSC) respondendo perguntas
sobre documentos longos (QASPER, QuALITY), comparando documento completo × RAG,
com júri de 3 LLMs validado contra gabarito. Artigo IEEE em `../latex-llmeval`.

## Repositórios (já existem, nunca rodar `git init`)
- `llm_qa_large_text_input_info_based/`: código, remoto GitHub, autor `CodeWracker`.
- `latex-llmeval/`: artigo, remoto Overleaf. Token em `/home/jovyan/privado/.persist/git/credentials`
  (credential helper `store`, via `~/.gitconfig` -> ZFS; `return_by_death.sh` recria o link).
- Commits em português ("adiciona…", "corrige…"), um por unidade lógica, push logo após.
  **Nunca** adicionar Co-Authored-By nem qualquer atribuição ao Claude.

## Regras do usuário para o artigo
- Fio narrativo contínuo: intro → problema/RQs → método/formalização → setup →
  resultados respondendo às RQs. Antecipar "tá, mas como fez isso?".
- O artigo descreve UM estudo contínuo: nunca mencionar piloto, versões anteriores ou cronologia dos testes.
- Resumo e Conclusão ficam em `\todo` até haver resultados; introdução sem números de resultado.
- `\todo{...}` = só fonte vermelha (`\textcolor{red}`), texto sempre em português, NUNCA caixas/todonotes.
- Resumo e Trabalhos Correlatos ficam como `\todo{}`
  (Trabalhos Correlatos traz as strings de busca).
- Citar só datasets, técnicas, métodos computacionais/estatísticos. Não buscar
  nem inventar referências de apoio a afirmações; no máximo `\todo{buscar referência para isso}`.
- Decisões já tomadas com o usuário: ver o fim do `TODO.md`.
- Arquivos de handoff curtos; se precisar de mais, use `handoffs/` com arquivos pequenos.

## Ambiente (recriar se o container foi resetado)
```bash
source /etc/profile.d/99-persist-env.sh      # OLLAMA_URL
# o container pode rodar como root ou como jovyan; o HOME é efêmero: rode
# /home/jovyan/privado/.persist/return_by_death.sh para recriar os links (~/.ssh, ~/.netrc, ~/.gitconfig)
# se o git acusar "dubious ownership": sudo chown -R jovyan:users /home/jovyan/compartilhado/ralph/llm-eval
# (~/.netrc = auth do Ollama; ~/.gitconfig = credencial do Overleaf; ~/.ssh = chave do GitHub, conta CodeWracker)
cd llm_qa_large_text_input_info_based
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# LaTeX local: /home/jovyan/privado/.persist/bin/tectonic -X compile main.tex
```
O servidor Ollama mantém 1 modelo carregado por vez e é compartilhado com outros projetos: no máximo
8 requisições simultâneas somando todos os processos (`max_inflight`, travas em `data/cache/slots`). Processe um
modelo por vez. Os embeddings rodam localmente na GPU (`CUDA_VISIBLE_DEVICES=1`).

## Pipeline
`python -m llmeval {sample,generate,judge,metrics,analyze}`. Tudo fica em cache em
`data/cache/*.sqlite`, então reexecutar é barato. A análise escreve `tables/*.tex` e
`figures/*.pdf` direto em `../latex-llmeval`. `run_chain.sh` roda a cadeia em duas fases: desenho principal (geração `--no-think` → júri →
métricas → análise) e depois a RQ6 (geração com raciocínio → júri → métricas → análise); marcos em `results/logs/pipeline.log`. Se cair, basta relançar: o cache preserva tudo.

## Incidentes do servidor e loop
- Toda falha do Ollama (HTTP, timeout, conexão) e a recuperação vão automaticamente para
  `/home/jovyan/privado/setic_infra_audit/incidentes_ollama/eventos.jsonl` (o usuário relata à SeTIC).
  `relatorio.py` na mesma pasta gera `RELATORIO_INCIDENTES_OLLAMA.md`.
- Disjuntor: 16 falhas seguidas do servidor encerram o estágio com código 75 e `run_chain.sh` pausa
  ("cadeia pausada" no pipeline.log). Típico quando outro projeto mantém outro modelo carregado.
- `aguarda_servidor.sh` (vigia, PID em `results/logs/aguarda.pid`) espera 3 min sem modelo alheio carregado
  ou só com modelo alheio ocioso (`expires_at` parado; com keep_alive infinito, como o qwen2.5:32b do ClinicalSimilarity,
  conta como ocupado), roda `run_chain.sh` (que pula a fase 1 já concluída) e volta a vigiar se ela pausar. Lançar com
  `setsid nohup ./aguarda_servidor.sh > results/logs/aguarda.log 2>&1 < /dev/null &`.
- Um loop (CronCreate, de hora em hora, expira em 7 dias, morre com a sessão) confere cadeia e vigia e
  relança o vigia se ele morreu. Se a sessão morreu, recrie o loop. Encerrar processos só por PID (nunca `pkill -f`).
- `IGNORAR_ALHEIOS=1` no vigia: servidor cedido pelo usuário (outros modelos carregados não bloqueiam).
- Após reboot da máquina: `OLLAMA_URL` e `~/.netrc` somem até o usuário rodar `/home/jovyan/privado/.persist/return_by_death.sh`.
