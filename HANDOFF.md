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
- Introdução sem números de resultado.
- `\todo{...}` = só fonte vermelha (`\textcolor{red}`), texto sempre em português, NUNCA caixas/todonotes.
- Trabalhos Correlatos fica como `\todo{}` (traz eixos e strings de busca). O resumo já é a versão final.
- Números da validação humana entram por macros `\hum{chave}` (`tables/human_agreement.tex`, gerado por
  `scripts/evaluate_human_results.py`; ver `human_eval/README.md`).
- Citar só datasets, técnicas, métodos computacionais/estatísticos. Não buscar
  nem inventar referências de apoio a afirmações; no máximo `\todo{buscar referência para isso}`.
- Decisões em vigor: ver o fim do `TODO.md`.
- Arquivos de handoff curtos; se precisar de mais, use `handoffs/` com arquivos pequenos.

## Ambiente (recriar se o container foi resetado)
```bash
source /etc/profile.d/99-persist-env.sh      # OLLAMA_URL
# o container pode rodar como root ou como jovyan; o HOME é efêmero: rode
# /home/jovyan/privado/.persist/return_by_death.sh para recriar os links (~/.ssh, ~/.netrc, ~/.gitconfig)
# se o git acusar "dubious ownership": sudo chown -R jovyan:users /home/jovyan/compartilhado/ralph/llm-eval
# (~/.netrc = auth do Ollama; ~/.gitconfig = credencial do Overleaf; ~/.ssh = chave do GitHub, conta CodeWracker)
cd llm_qa_large_text_input_info_based
# o .venv foi recriado como jovyan (27/10); se sumir ou quebrar, recriar:
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# LaTeX local: /home/jovyan/privado/.persist/bin/tectonic -X compile main.tex
```
O servidor Ollama é compartilhado com outros projetos: `max_inflight` (configs) limita as requisições simultâneas
somando todos os processos (travas em `data/cache/slots`); 16 com o servidor cedido, 8 se compartilhado. Um modelo por vez. Os embeddings rodam localmente na GPU (`CUDA_VISIBLE_DEVICES=1`).

## Pipeline
`python -m llmeval {sample,generate,judge,metrics,analyze}`. Tudo fica em cache em
`data/cache/*.sqlite`, então reexecutar é barato. A análise escreve `tables/*.tex` e
`figures/*.pdf` direto em `../latex-llmeval`. Marcos em `results/logs/pipeline.log`; se cair, basta relançar.
- `run_chain.sh`: desenho principal e RQ6 (concluídos).
- `run_extra.sh`: `run_budget.sh` (varredura de orçamento, `generate/judge --budget B`, `--dry-run` só conta chamadas)
  e depois `run_agreement.sh` (concordância com humanos). O cliente deduz do cache de outro teto as respostas cujo
  desfecho já se sabe (`OllamaClient._project`); modelos em `nondeterministic_models` só são projetados para tetos menores.

## Incidentes do servidor e loop
- Toda falha do Ollama (HTTP, timeout, conexão) e a recuperação vão automaticamente para
  `/home/jovyan/privado/setic_infra_audit/incidentes_ollama/eventos.jsonl` (o usuário relata à SeTIC).
  `relatorio.py` na mesma pasta gera `RELATORIO_INCIDENTES_OLLAMA.md`.
- Disjuntor: `circuit_breaker` falhas seguidas do servidor encerram o estágio com código 75 e a cadeia pausa.
- `aguarda_servidor.sh` (vigia, PID em `results/logs/aguarda.pid`) espera 3 min sem modelo alheio em uso
  (`expires_at` parado; keep_alive infinito conta como ocupado), roda `$CHAIN` e volta a vigiar se ela pausar (75);
  qualquer outro código encerra. Variáveis: `CHAIN` (padrão `./run_chain.sh`), `IGNORAR_MODELOS="m1 m2"`
  (modelos alheios liberados pelo usuário, ex. o qwen2.5:32b do ClinicalSimilarity), `IGNORAR_ALHEIOS=1`
  (servidor todo cedido). Comando atual: ver "Se o container cair" no `TODO.md`.
- Um loop (CronCreate, de hora em hora, expira em 7 dias, morre com a sessão) confere cadeia e vigia e
  relança o vigia se ele morreu. Se a sessão morreu, recrie o loop. Encerrar processos só por PID (nunca `pkill -f`).
- Após reboot da máquina: `OLLAMA_URL` e `~/.netrc` somem até o usuário rodar `/home/jovyan/privado/.persist/return_by_death.sh`.
