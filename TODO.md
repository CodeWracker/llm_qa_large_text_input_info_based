# TODO / rastreamento

Atualizar a cada etapa concluída. Formato: `[x]` feito, `[~]` em andamento, `[ ]` pendente.

## Código
- [x] Pacote `llmeval/`: dados, recuperação (BM25/denso/RRF + busca por termos), cliente Ollama com cache, prompts, júri, métricas, estatística, análise
- [x] Amostra: 400 perguntas, 272 documentos (`data/sample/questions.jsonl`)
- [x] Revocação de evidência (`results/retrieval.jsonl`): com k=5, denso 74% > híbrido 65% > BM25 57%
- [x] Geração principal: 6 modelos × 7 condições × 400 = 16.800 respostas (2 falhas de formato)
- [x] Júri sobre a geração principal: 3 juízes × 4.523 itens (20/10 05:37; 1 voto ausente). Servidor tomado por outro projeto 19/10 12:20 → 20/10 03:43 (incidentes em `setic_infra_audit/incidentes_ollama`)
- [x] Análise intermediária do desenho principal (7 condições): reconstruída só do cache fora de `results/`
  (o checkpoint da geração em curso sobrescrevia `generations.jsonl`; corrigido em `generate.py` para as próximas rodadas)
- [~] Cadeia automática (`run_chain.sh` via vigia `aguarda_servidor.sh`, log em `results/logs/pipeline.log`):
  geração das 4 condições de busca + passada com raciocínio → júri → métricas → análise final.
- [ ] Implementar a análise da RQ6 (thinking × sem thinking, pareado; tokens/latência; truncamento)
- [ ] Tabelas com 11 condições: verificar largura (talvez dividir RAG × busca)
- [ ] Inspecionar respostas: abstenção, falhas de JSON, teto de tokens, buscas sem achados
- [ ] Commitar `results/*.jsonl` e `results/analysis/*.csv`

## Artigo (`../latex-llmeval`)
- [x] Preâmbulo, bibliografia (só métodos/datasets), introdução com RQs, metodologia formal, setup, ameaças à validade
- [x] Resumo e Trabalhos Correlatos como `\todo` (PT) com strings de busca
- [x] Diagrama principal: versão A aprovada
- [x] RQ2 "parecer certo × estar certo"
- [x] Busca guiada (método + formalização) e RQ6 (raciocínio) no texto, tabelas e diagrama (validado com Haiku)
- [~] Discussão de RQ1–RQ5 escrita com os números do desenho principal; ao fim da cadeia, atualizar números
  (RQ1/QASPER e RQ2 passam a incluir as 4 condições de busca), escrever busca (RQ4) e RQ6, depois Conclusão e Resumo
- [ ] Revisar floats; conferir metadados do `.bib`

## Decisões do usuário (19/10/2026)
- Sem anotação humana: a validação do júri fica no QuALITY e a limitação vai para as ameaças.
- Alvo de publicação indefinido: escrever no formato IEEE conferência sem cortar; enxugar depois.
- Raciocínio: todos os modelos que suportam, todas as condições, teto de 4096; vira a RQ6; roda após o júri principal.
- Júri mantido (testar autopreferência na RQ1). Amostra mantida em 400.
- Trabalhos Correlatos: só as strings; o usuário pesquisa (Scopus, ScienceDirect...).
- `legacy/` e menções ao piloto removidos também do repositório de código; `images/` removido do artigo.
- Busca guiada: variantes separadas pelo que o modelo leu antes de escolher os termos (nada / RAG / documento).
