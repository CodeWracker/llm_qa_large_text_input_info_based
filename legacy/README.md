# Estudo piloto (legado)

Código e artefatos da primeira versão do estudo (2025), mantidos apenas para
rastreabilidade. **Não use estes resultados**: eles foram gerados antes da
correção de três bugs metodológicos (commit `cbb119b`):

1. `ask_ai_jury` recebia os argumentos fora de ordem — o júri avaliava a
   *pergunta* como se fosse a resposta do modelo (visível nas justificativas
   da tabela de exemplo do rascunho do artigo: "EVAL asks ...").
2. No QuALITY, as quatro alternativas (incluindo os três distratores) eram
   tratadas como respostas corretas; `gold_label` era ignorado.
3. Reprocessar um modelo duplicava as entradas de `scores`.

Configuração do piloto: 7 modelos Gemini via API (gerador), júri com
`deepseek-r1:1.5b`, `llama3.2:1b` e `gemma3:1b`, 2.055 pares
pergunta–resposta, similaridade combinada = média(SBERT, BERTScore).

O pipeline atual está em `../llmeval/` (veja o `README.md` da raiz).
