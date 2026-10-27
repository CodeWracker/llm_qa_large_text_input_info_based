# TODO

Só o que vale hoje. `[~]` em andamento, `[ ]` pendente. Histórico: `git log`. Retomar após queda: ver "Se o container cair".

## Em andamento (servidor)
- [~] **Varredura de orçamento da RQ6** (`run_budget.sh`, via `run_extra.sh`): tetos 512, 1024, 2048 (quase tudo
  deduzido do cache de 4096) e 8192 (Qwen3.5-4B refeito por inteiro, ~3.500 chamadas; 9B só os cortados), júri, análise
  (`tables/budget.tex`, `figures/budget.pdf`). Teste de determinismo (27/10): 9B 20/20 iguais, 4B 10/20 → 4B em
  `nondeterministic_models` (só se projeta para tetos menores). Estimativa ~35–45 h com o servidor livre.
- [ ] **Concordância humano × humano e modelo × humano no QASPER** (`run_agreement.sh` → `python -m llmeval agreement`,
  roda depois da varredura; 6.464 itens, 4.088 chamadas únicas por juiz): o júri julga cada anotação humana contra as demais (deixa um de fora) nas 200 perguntas da
  amostra e nas 1.427 do test com ≥2 anotações; os 6 modelos (documento completo, sem raciocínio) são julgados contra
  os mesmos conjuntos, só nas 200. Mais a parte objetiva sem LLM (sim/não e "tem resposta?") já calculada à parte
  (`human_ceiling` em `analysis.py`, IC por bootstrap, `tables/human_ceiling.tex`). ~6 h de júri.

## Artigo (`../latex-llmeval`): lugares prontos, em vermelho até os dados chegarem
- [ ] RQ6 (V-F): valores e discussão da varredura; `tables/budget.tex` e `figures/budget.pdf` saem do `analyze`;
  rever a frase do resumo, a conclusão da RQ6 e a ameaça "Models and serving"
- [ ] RQ3: coluna "Free text (jury)" de `tables/human_ceiling.tex` e discussão; frase na conclusão
- [ ] Revisar floats; conferir metadados do `.bib`; Trabalhos Correlatos é do usuário

## Pendências do usuário
- [ ] Segundo anotador (Rodrigo Guedes), se ele fizer: CSV em `human_eval/results/`, rodar
  `scripts/evaluate_human_results.py`; recolocar no artigo a linha "annotator 1 vs. 2" e a concordância entre humanos
- [ ] Avisar o dono/administrador da chave exposta em `/home/jovyan/compartilhado/.ssh/` (não usar: não é do usuário)

## Se o container cair
1. `/home/jovyan/privado/.persist/return_by_death.sh` (usuário) e `source /etc/profile.d/99-persist-env.sh`.
   Se `.venv/bin/python` não existir, recriar como no HANDOFF (Python em `/home/jovyan/privado/.persist/uv-python`).
2. `tail results/logs/pipeline.log`: se não há "varredura de orçamento concluída", relançar
   `IGNORAR_MODELOS=qwen2.5:32b CHAIN=./run_extra.sh setsid nohup ./aguarda_servidor.sh > results/logs/aguarda_extra.log 2>&1 < /dev/null &`
   (o cache retoma de onde parou).
3. Recriar o loop de verificação horária (CronCreate) se a sessão morreu.

## Decisões em vigor
- Artigo: extração de dados de revisões sistemáticas; QASPER é o alvo e QuALITY o controle; sem RQ7; IEEE conferência.
- Amostra de 400 perguntas; júri de 3 LLMs; raciocínio só em Qwen3.5-4B e 9B, 5 condições.
- Ordem no servidor (27/10): varredura de orçamento antes da concordância com o júri.
- Validação humana: 120 respostas, um anotador (Rodrigo Ferraz, 27/10: júri 85,0%, kappa 0,65); a régua do modelo é a
  concordância entre humanos, não 100%.
- Citar só datasets/métodos, mais Liu 2024, Schick 2023 e Souza & Panisson 2026 (pedidos pelo usuário).
