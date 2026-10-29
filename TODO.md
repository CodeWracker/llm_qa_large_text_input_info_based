# TODO

Só o que vale hoje. `[~]` em andamento, `[ ]` pendente. Histórico: `git log`. Retomar após queda: ver "Se o container cair".

## Experimentos
- [x] Varredura de orçamento da RQ6 e concordância com humanos no QASPER concluídas (29/10); artigo atualizado.
  Nada rodando no servidor. `configs/experiment.yaml` está com concorrência 4 (servidor dividido); com o servidor
  livre, 16 é mais rápido para gerações curtas.

## Artigo (`../latex-llmeval`)
- [ ] Revisar floats (a tabela/figura de orçamento flutuam para as ameaças); conferir metadados do `.bib`
- [ ] Trabalhos Correlatos é do usuário

## Pendências do usuário
- [ ] Segundo anotador (Rodrigo Guedes), se ele fizer: CSV em `human_eval/results/`, rodar
  `scripts/evaluate_human_results.py`; recolocar no artigo a linha "annotator 1 vs. 2" e a concordância entre humanos
- [ ] Avisar o dono/administrador da chave exposta em `/home/jovyan/compartilhado/.ssh/` (não usar: não é do usuário)

## Se o container cair
1. `/home/jovyan/privado/.persist/return_by_death.sh` (usuário) e `source /etc/profile.d/99-persist-env.sh`.
   Se `.venv/bin/python` não existir, recriar como no HANDOFF (Python em `/home/jovyan/privado/.persist/uv-python`).
2. `tail results/logs/pipeline.log`: se não há "varredura de orçamento concluída", relançar
   `IGNORAR_ALHEIOS=1 CHAIN=./run_extra.sh setsid nohup ./aguarda_servidor.sh > results/logs/aguarda_extra.log 2>&1 < /dev/null &`
   (o cache retoma de onde parou).
3. Recriar o loop de verificação horária (CronCreate) se a sessão morreu.

## Decisões em vigor
- Artigo: extração de dados de revisões sistemáticas; QASPER é o alvo e QuALITY o controle; sem RQ7; IEEE conferência.
- Amostra de 400 perguntas; júri de 3 LLMs; raciocínio só em Qwen3.5-4B e 9B, 5 condições.
- Ordem no servidor (27/10): varredura de orçamento antes da concordância com o júri. 28/10: dividir o servidor
  com outros usuários em vez de esperar (IGNORAR_ALHEIOS=1).
- Validação humana: 120 respostas, um anotador (Rodrigo Ferraz, 27/10: júri 85,0%, kappa 0,65); a régua do modelo é a
  concordância entre humanos, não 100%.
- Citar só datasets/métodos, mais Liu 2024, Schick 2023 e Souza & Panisson 2026 (pedidos pelo usuário).
