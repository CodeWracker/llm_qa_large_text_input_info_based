# Avaliação de LLMs em QA sobre documentos longos: documento completo vs. RAG

Código do estudo do artigo em `../latex-llmeval`. A pergunta central: quando um
modelo responde a uma pergunta sobre um documento específico (um artigo, um
conto), a resposta está certa? E é melhor colocar o documento inteiro no prompt
ou recuperar só os trechos relevantes (RAG), ou deixar o próprio modelo buscar
no documento?

## Perguntas de pesquisa

| RQ | Pergunta | Onde é respondida |
|----|----------|-------------------|
| RQ1 | Métricas automáticas e um júri de LLMs concordam com a correção objetiva? O júri é confiável e sem viés de família? | `rq1_instruments`, `rq1_reliability`, `fig_roc` |
| RQ2 | Parecer certo × estar certo: modelos com respostas mais similares às referências acertam mais, ou alguns (maiores) erram de forma "plausível"? Formalizado por s̄_m = μ⁰_m + Acc_m·Δ_m | `rq2_plausible` |
| RQ3 | Quão bem modelos abertos de 3B a 125B respondem com o documento inteiro? Como a acurácia varia com o tamanho do documento e a posição da evidência? | `table_accuracy`, `fig_accuracy`, `fig_length_position` |
| RQ4 | RAG (BM25, denso, híbrido; k = 2, 5, 10) e a busca guiada pelo modelo alcançam o documento completo? Com que economia de contexto? Quanto do erro é falha de recuperação? | `rq3_paired` (gera `paired.tex`), `table_gee`, `fig_cost`, `fig_recall` |
| RQ5 | Quando a resposta não está no documento, os modelos dizem isso? | `table_abstention` |
| RQ6 | Raciocinar antes de responder (thinking) ajuda? A que custo em tokens e latência? | (a implementar quando houver dados) |

## Desenho experimental

- **Dados**: amostra estratificada (seed 42) de 400 perguntas em 272 documentos:
  50 por tipo de resposta do QASPER (extractive, abstractive, yes/no,
  unanswerable) e 100 fáceis + 100 difíceis do QuALITY. No máximo 2 perguntas
  por documento.
- **Geradores** (servidor Ollama da UFSC): Llama-3.2-3B, Qwen3.5-4B,
  Mistral-7B, Qwen3.5-9B, Gemma4-31B, Qwen3.5-122B. Temperatura 0, semente
  fixa, `num_ctx` = 32768 para todos (todo documento da amostra cabe), raciocínio
  desligado, saída restrita por JSON Schema.
- **Condições de contexto** (11): sem documento, documento completo, BM25 k=5,
  denso (bge-m3) k=5, híbrido (RRF) k=2/5/10 e quatro de busca guiada pelo
  modelo. Trechos de ~200 palavras que respeitam parágrafos e seções; a busca é
  feita dentro do próprio documento.
- **Busca guiada pelo modelo (Ctrl+F)**: uma chamada de ferramenta única e
  forçada (saída estruturada) em que o gerador lista até 5 termos; uma função
  devolve os trechos que os contêm (sem distinção de maiúsculas, ordenados por
  nº de termos distintos e de ocorrências). As variantes mudam o que o modelo
  leu antes de escolher os termos e como os achados entram na resposta:
  `search_k5` (só a pergunta; só os achados), `search_hybrid_k5` (só a pergunta;
  achados fundidos ao híbrido por RRF), `rag_search_k5` (lê os trechos do
  híbrido; fusão RRF) e `full_search_k5` (lê o documento inteiro; responde com
  o documento seguido dos achados).
- **Raciocínio (RQ6)**: segunda passada, com thinking ligado, nos modelos que o
  suportam (Qwen3.5 e Gemma4), em todas as condições; teto de 4096 tokens
  (raciocínio + resposta). Estourar o teto é registrado como falha.
- **Correção**:
  - QuALITY (múltipla escolha): gabarito, ou seja, rótulo objetivo.
  - QASPER (resposta livre): júri de 3 LLMs de famílias distintas (GPT-OSS-120B,
    Llama-3.3-70B, Qwen3-32B), nenhum deles gerador, com veredito
    correct/partial/incorrect e maioria estrita. Abstenções são decididas por regra.
  - Métricas automáticas: EM, F1 de tokens (SQuAD), ROUGE-L, fuzzy, cosseno
    bge-m3 e BERTScore, todas com máximo sobre as referências.
- **Validação do instrumento (RQ1)**: no QuALITY o gerador escreve a resposta
  com as próprias palavras antes de escolher a letra. Júri e métricas avaliam esse
  texto contra a alternativa correta, e o resultado é comparado ao gabarito.
- **Estatística**: IC de Wilson, McNemar exato + Holm, bootstrap por documento,
  GEE logístico agrupado por documento, Fleiss κ, Krippendorff α, AUROC/Youden.

A formalização completa está na Seção *Methodology* do artigo.

## Estrutura

```
configs/experiment.yaml   # tudo o que define o experimento
llmeval/
  data.py        # download (parquet HF), normalização, amostragem estratificada
  retrieval.py   # chunking, BM25, denso, RRF, revocação de evidência
  embeddings.py  # bge-m3 local (GPU) ou via Ollama, com cache
  ollama.py      # cliente com cache SQLite, JSON Schema, retentativas
  prompts.py     # prompts de geração/júri, detecção de abstenção
  generate.py    # estágio 1: pergunta × modelo × condição -> resposta
  judge.py       # estágio 2: júri + regras de abstenção
  metrics.py     # estágio 3: métricas automáticas
  stats.py       # testes e medidas estatísticas
  analysis.py    # estágio 4: tabelas .tex e figuras .pdf -> ../latex-llmeval
data/sample/questions.jsonl   # amostra (versionada)
results/                      # saídas versionadas (jsonl) e results/analysis/*.csv
```

## Como reproduzir

```bash
# Python >= 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128

# endpoint Ollama (na UFSC); a autenticação Basic vem de ~/.netrc
source /etc/profile.d/99-persist-env.sh

.venv/bin/python -m llmeval sample     # data/sample/{questions,documents}.jsonl
.venv/bin/python -m llmeval generate   # results/generations.jsonl, results/retrieval.jsonl
.venv/bin/python -m llmeval judge      # results/judgments.jsonl
.venv/bin/python -m llmeval metrics    # results/metrics.jsonl
.venv/bin/python -m llmeval analyze    # ../latex-llmeval/{tables,figures}/, results/analysis/
# ou tudo de uma vez:
.venv/bin/python -m llmeval all
```

Toda chamada a modelo fica em cache (`data/cache/*.sqlite`), então reexecutar
um estágio não consulta o servidor de novo. Para trocar modelos, condições ou
tamanho da amostra, edite apenas `configs/experiment.yaml`.

### Notas sobre o servidor compartilhado

O servidor mantém um modelo carregado por vez. Por isso:

- os modelos são processados um de cada vez, com chamadas paralelas (`concurrency`);
- os embeddings rodam localmente (`embedding_backend: local`), sem disputar o servidor;
- `keep_alive` curto, para liberar a memória logo após o uso.
