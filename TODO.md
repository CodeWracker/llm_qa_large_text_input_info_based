# TODO

Só o que vale hoje. `[~]` em andamento, `[ ]` pendente. Histórico: `git log`. Retomar após queda: ver "Se o container cair".

## Em andamento (servidor)
- [~] **Varredura de orçamento da RQ6** (`run_budget.sh`, via `run_extra.sh`): tetos 512, 1024, 2048 (quase tudo
  deduzido do cache de 4096) e 8192 (Qwen3.5-4B refeito por inteiro, ~3.500 chamadas; 9B só os cortados), júri, análise
  (`tables/budget.tex`, `figures/budget.pdf`). Teste de determinismo (27/10): 9B 20/20 iguais, 4B 10/20 → 4B em
  `nondeterministic_models` (só se projeta para tetos menores). Estimativa ~35–45 h com o servidor livre.
- [ ] **Concordância humano × humano e modelo × humano no QASPER** (`run_agreement.sh`, roda depois da varredura;
  ainda não escrito): o júri julga cada anotação humana contra as demais (deixa um de fora) nas 200 perguntas da
  amostra e nas 1.427 do test com ≥2 anotações; os 6 modelos (documento completo, sem raciocínio) são julgados contra
  os mesmos conjuntos, só nas 200. Mais a parte objetiva sem LLM (sim/não e "tem resposta?") já calculada à parte
  (`scratchpad/human_ceiling2.py`, levar para `analysis.py` com IC por bootstrap). ~9 h de júri.

## Artigo (`../latex-llmeval`), quando os dados chegarem
- [ ] RQ6: tabela/figura de orçamento e texto (III-C, V-F, VI-A, conclusão, resumo)
- [ ] RQ3 (sim/não) e RQ5 ("tem resposta?"): parágrafo e tabela pequena de concordância com humanos; texto livre
- [ ] Ameaças: 51 de 200 perguntas do QASPER com referências contraditórias (16 sim×não, 35 sem resposta×resposta);
  sem elas a acurácia cai 2–5 pontos e nenhuma ordenação muda (análise de sensibilidade)
- [ ] Setup: trocar "answers are reproducible" — o 4B não reproduz com temperatura 0; reproduzível é o cache
- [ ] Revisar floats; conferir metadados do `.bib`; Trabalhos Correlatos é do usuário

## Pendências do usuário
- [ ] Anotar as 120 respostas (Rodrigo Ferraz; Rodrigo Guedes ainda a definir: todas ou um subconjunto), CSVs em
  `human_eval/results/`, rodar `scripts/evaluate_human_results.py` (há um `\todo` na RQ1)
- [ ] Avisar o dono/administrador da chave exposta em `/home/jovyan/compartilhado/.ssh/` (não usar: não é do usuário)

## Se o container cair
1. `/home/jovyan/privado/.persist/return_by_death.sh` (usuário) e `source /etc/profile.d/99-persist-env.sh`.
2. `tail results/logs/pipeline.log`: se não há "varredura de orçamento concluída", relançar
   `IGNORAR_MODELOS=qwen2.5:32b CHAIN=./run_extra.sh setsid nohup ./aguarda_servidor.sh > results/logs/aguarda_extra.log 2>&1 < /dev/null &`
   (o cache retoma de onde parou).
3. Recriar o loop de verificação horária (CronCreate) se a sessão morreu.

## Decisões em vigor
- Artigo: extração de dados de revisões sistemáticas; QASPER é o alvo e QuALITY o controle; sem RQ7; IEEE conferência.
- Amostra de 400 perguntas; júri de 3 LLMs; raciocínio só em Qwen3.5-4B e 9B, 5 condições.
- Ordem no servidor (27/10): varredura de orçamento antes da concordância com o júri.
- Validação humana: 120 respostas; a régua do modelo é a concordância entre humanos, não 100%.
- Citar só datasets/métodos, mais Liu 2024, Schick 2023 e Souza & Panisson 2026 (pedidos pelo usuário).
