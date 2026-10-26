# TODO / rastreamento

Atualizar a cada etapa concluída. Formato: `[x]` feito, `[~]` em andamento, `[ ]` pendente.

## Código
- [x] Pacote `llmeval/`: dados, recuperação (BM25/denso/RRF + busca por termos), cliente Ollama com cache, prompts, júri, métricas, estatística, análise
- [x] Amostra: 400 perguntas, 272 documentos (`data/sample/questions.jsonl`)
- [x] Revocação de evidência (`results/retrieval.jsonl`): com k=5, denso 74% > híbrido 65% > BM25 57%
- [x] Geração principal: 6 modelos × 7 condições × 400 = 16.800 respostas (2 falhas de formato)
- [x] Júri sobre a geração principal: 3 juízes × 4.523 itens (20/10 05:37; 1 voto ausente). Servidor tomado por outro projeto 19/10 12:20 → 20/10 03:43 (incidentes em `setic_infra_audit/incidentes_ollama`)
- [x] Fase 1 da cadeia (20/10 13:29): desenho principal completo, 6 modelos × 11 condições × 400 = 26.400 respostas
  (7 falhas), júri (5.406 itens únicos/juiz, 9 votos ausentes), métricas e análise; resultados commitados
- [x] Fase 2 concluída (24/10 21:02): raciocínio em Qwen3.5-4B e 9B, 5 condições (4.000 respostas; 1.170 perdidas por
  esgotar o orçamento de 4096), júri final (3 votos ausentes), métricas e análise; resultados commitados
- [x] Análise da RQ6: δ pareado com todas as respostas e só com as concluídas no orçamento; tokens; orçamento esgotado
- [x] Tabela/figura de acurácia com 11 condições (cabeçalho agrupado Retrieval × Model-guided search)
- [x] Busca: termos copiados da pergunta, buscas sem achado e revocação dos trechos (calculados à parte, no texto da RQ4)

## Artigo (`../latex-llmeval`)
- [x] Preâmbulo, bibliografia (só métodos/datasets), introdução com RQs, metodologia formal, setup, ameaças à validade
- [x] Resumo e Trabalhos Correlatos como `\todo` (PT) com strings de busca
- [x] Diagrama principal: versão A aprovada
- [x] RQ2 "parecer certo × estar certo"
- [x] Busca guiada (método + formalização) e RQ6 (raciocínio) no texto, tabelas e diagrama (validado com Haiku)
- [x] Discussão de RQ1–RQ5, incluindo a busca guiada (RQ4), com os números finais do desenho principal
- [x] RQ6 e Conclusão escritas com os resultados finais; conclusão em texto corrido guiado pelas RQs
- [x] Figura 4 com painéis (a)/(b) e τ* explicado na legenda
- [ ] Resumo (fica em `\todo` até o usuário pedir) e Trabalhos Correlatos (usuário faz o levantamento)
- [ ] Revisar floats; conferir metadados do `.bib`

## Pendências do usuário
- [x] Acesso SSH da conta CodeWracker reposto em `~/.ssh` (26/10); commits locais enviados. Não usar
  `/home/jovyan/compartilhado/.ssh/id_ed25519_github`: é de outra pessoa (ArthurgLeonida)
- [ ] Avisar o dono ou o administrador de que essa chave privada sem senha está exposta na pasta compartilhada

## Decisões do usuário (19/10/2026)
- Sem anotação humana: a validação do júri fica no QuALITY e a limitação vai para as ameaças.
- Alvo de publicação indefinido: escrever no formato IEEE conferência sem cortar; enxugar depois.
- Raciocínio: teto de 4096; vira a RQ6; roda após o júri principal. 20/10: a ~22 s/item seriam 4–6 dias, então só
  Qwen3.5-4B e 9B (pequenos, mesma família), todas as condições; o texto justifica pelo limite de infraestrutura
  (mostrar a direção do efeito). 23/10: mesmo com 16 req. o servidor dá ~150 itens/h, então só 5 condições
  (closed book, full, hybrid k=5, search+hyb., read→search).
- Júri mantido (testar autopreferência na RQ1). Amostra mantida em 400.
- Trabalhos Correlatos: só as strings; o usuário pesquisa (Scopus, ScienceDirect...).
- `legacy/` e menções ao piloto removidos também do repositório de código; `images/` removido do artigo.
- Busca guiada: variantes separadas pelo que o modelo leu antes de escolher os termos (nada / RAG / documento).
