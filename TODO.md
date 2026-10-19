# TODO / rastreamento

Atualizar a cada etapa concluída. Formato: `[x]` feito, `[~]` em andamento, `[ ]` pendente.

## Código
- [x] Corrigir bugs do piloto (júri com argumentos trocados, distratores do QuALITY, scores duplicados); mover para `legacy/`
- [x] Pacote `llmeval/`: dados, recuperação (BM25/denso/RRF), cliente Ollama com cache, prompts, júri, métricas, estatística, análise
- [x] Amostra: 400 perguntas, 272 documentos (`data/sample/questions.jsonl`)
- [x] Revocação de evidência (`results/retrieval.jsonl`): com k=5, denso 74% > híbrido 65% > BM25 57%
- [~] Geração: 6 modelos × 7 condições × 400 perguntas = 16.800 respostas (`results/generations.jsonl`, checkpoint por modelo)
- [ ] Júri (`judge`) → métricas (`metrics`) → análise (`analyze`): encadeados por `run_pipeline_after.sh`
- [ ] Inspecionar respostas: taxa de abstenção, falhas de JSON, truncamento
- [ ] Commitar `results/*.jsonl` e `results/analysis/*.csv`
- [ ] Exemplo do carro com o júri real (`tables/car_example.tex`, gerado no `analyze`)

## Artigo (`../latex-llmeval`)
- [x] Preâmbulo, bibliografia (só métodos/datasets), introdução com RQs, metodologia formal, setup, ameaças à validade
- [x] Resumo e Trabalhos Correlatos como `\todo` (PT) com strings de busca
- [x] Tabelas de amostra e modelos; placeholders de tabelas e figuras
- [x] Diagrama principal: versão A aprovada (figura única, formas e cor discreta por tipo)
- [x] Nova RQ2 "parecer certo × estar certo" (decomposição s̄ = μ⁰ + Acc·Δ, inversão de ranking, viés do avaliador); RQs renumeradas para 1–5
- [x] Push para o Overleaf (credencial persistida no ZFS)
- [ ] Escrever a discussão de cada RQ (1–5) com os números reais (substituir os `\todo` da Seção V)
- [ ] Conclusão; revisar o posicionamento dos floats; conferir os metadados do `.bib`
- [ ] Decidir se as figuras antigas do piloto em `images/` devem ser removidas

## Ideias / decisões em aberto
- Anotar à mão ~100 respostas do QASPER para medir a concordância júri × humano (ameaça à validade)
- Se o tempo de servidor apertar: reduzir para `hybrid_k5` + `full` nos modelos grandes
