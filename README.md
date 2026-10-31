# Open-weight LLMs for data extraction from long documents: full document vs. retrieval, with a validated LLM jury

Code of the study *"Reading the Whole Paper or Retrieving What Matters? Open-Weight LLMs for Data Extraction in
Systematic Literature Reviews, Evaluated by a Validated LLM Jury"*. Data (model outputs, verdicts, caches):
Zenodo record (DOI to be added on publication).

The guiding question: can open-weight language models running on a local server fill the data-extraction table of a
systematic literature review reliably, where each cell is a question about one paper? Six models answer 400
questions about long documents under eleven ways of providing the document, and the free-text answers are graded by
a jury of three other LLMs that is first validated against an answer key and a human annotator.

## Research questions

| RQ | Question | Produced by (`llmeval/analysis.py`) |
|----|----------|-------------------------------------|
| RQ1 | Do automatic metrics and an LLM jury agree with objective correctness and with a human annotator? Is the jury reliable and free of family bias? | `rq1_instruments`, `rq1_reliability`, `fig_roc` |
| RQ2 | Do models whose answers look closer to the references answer correctly more often, or do some look right without being right? | `rq2_plausible` |
| RQ3 | How accurately do models of 3B to 125B parameters answer with the whole document, by document length and evidence position, and against the agreement among the QASPER annotators? | `table_accuracy`, `fig_accuracy`, `fig_length_position`, `human_ceiling`, `consensus` |
| RQ4 | Can retrieval (BM25, dense, hybrid; k = 2, 5, 10) or a keyword search chosen by the model match the whole document, with how much less input? | `rq3_paired`, `table_gee`, `fig_cost`, `fig_recall` |
| RQ5 | When the document does not contain the answer, do models say so? | `table_abstention` |
| RQ6 | Does reasoning before answering help, under which generation budget, and at what cost in tokens? | `rq6_reasoning`, `rq6_budget` |

## Experimental design

- **Data**: stratified sample (seed 42) of 400 questions over 272 documents: 50 QASPER questions per answer type
  (extractive, abstractive, yes/no, unanswerable; test split) and 100 easy + 100 hard QuALITY questions (validation
  split); at most two questions per document. Both datasets are downloaded from a pinned Hugging Face commit and
  checked against their SHA-256.
- **Generators**: Llama-3.2-3B, Qwen3.5-4B, Mistral-7B, Qwen3.5-9B, Gemma4-31B, Qwen3.5-122B, served by Ollama.
  Temperature 0, fixed seed, `num_ctx` = 32,768 for every model (every sampled document fits), output constrained by
  a JSON Schema, reasoning off in the main design.
- **Context conditions** (11): closed book, full document, BM25 k=5, dense (BGE-M3) k=5, hybrid (RRF) k=2/5/10, and
  four model-guided keyword searches (Ctrl+F): the model lists up to five terms in a single forced tool call, and the
  passages containing them are returned. The variants differ in what the model reads before choosing the terms and
  in how the passages enter the answer prompt (`search_k5`, `search_hybrid_k5`, `rag_search_k5`, `full_search_k5`).
- **Reasoning (RQ6)**: Qwen3.5-4B and Qwen3.5-9B answer five conditions again with reasoning enabled, under
  generation budgets of 512, 1024, 2048, 4096 and 8192 tokens (reasoning and answer together).
- **Grading**: QuALITY answers by the answer key; QASPER answers by a jury of GPT-OSS-120B, Llama-3.3-70B and
  Qwen3-32B (none of them a generator), with verdicts correct / partially correct / incorrect and a strict majority;
  abstentions by rule. Automatic metrics: exact match, token F1, ROUGE-L, fuzzy, BGE-M3 cosine and BERTScore.
- **Validation of the grader (RQ1)**: on QuALITY each generator first states its answer in its own words; the jury
  and the metrics grade that text against the correct option, and the result is compared with the answer key. On
  QASPER, a human annotator graded 120 answers blind to the model and to the jury.
- **Human reference point (RQ3, RQ5)**: each QASPER annotator is left out in turn and compared with the others; the
  models are compared with the same annotators, and a pairwise analysis shows where models depart from a consensus.
- **Statistics**: Wilson intervals, exact McNemar with Holm correction, document-clustered bootstrap, logistic GEE
  clustered by document, Fleiss' kappa, Krippendorff's alpha, AUROC and Youden thresholds.

## Layout

```
configs/experiment.yaml   everything that defines the experiment (pinned datasets, model tags and digests, conditions, budgets)
llmeval/
  data.py        pinned download, normalisation, stratified sampling
  retrieval.py   chunking, BM25, dense, RRF, keyword search, evidence recall
  embeddings.py  BGE-M3 (local GPU or Ollama), cached
  ollama.py      client with SQLite cache, JSON Schema, retries, budget projection, offline mode
  prompts.py     generation, search and jury prompts; abstention detection
  generate.py    stage 1: question x model x condition -> answer (also --budget B)
  judge.py       stage 2: jury and abstention rules
  metrics.py     stage 3: automatic metrics
  agreement.py   human reference point (leave-one-annotator-out and pairwise)
  stats.py       statistical tests and measures
  analysis.py    stage 4: .tex tables and .pdf figures, results/analysis/*.csv
scripts/         human validation page and agreement, determinism check, Zenodo package
human_eval/      annotation page, hidden key, annotations, agreement report
data/sample/     the sample (questions.jsonl is versioned; documents.jsonl is rebuilt)
results/         model outputs and verdicts (jsonl) and results/analysis/*.csv
```

## Installation

```bash
# Python >= 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128   # GPU build
```

A GPU is needed only to compute embeddings (BGE-M3) and BERTScore; set `CUDA_VISIBLE_DEVICES` as needed.

## Reproducibility

Every model call is cached by a hash of the request (`data/cache/chat.sqlite`), and every stage reads the cache
before calling the server, so re-running a stage is cheap and deterministic. There are three levels of reproduction.

### Level 1: tables and figures from the published outputs (no LLM, no GPU)

```bash
# put the Zenodo files results/, data/sample/ and human_eval/ in place, then:
.venv/bin/python -m llmeval sample     # pinned download of QASPER/QuALITY, rebuilds data/sample/documents.jsonl
LLMEVAL_OFFLINE=1 .venv/bin/python -m llmeval analyze
```

`sample` reproduces the published `questions.jsonl` and `documents.jsonl` byte for byte. `analyze` writes the tables
and figures to the folder given by `paths.paper` in the config (by default `../latex-llmeval`) and the numbers to
`results/analysis/`.

### Level 2: the whole pipeline from the caches (no model server)

Decompress `chat.sqlite` and `embeddings_local.sqlite` from the Zenodo record into `data/cache/`, then:

```bash
export LLMEVAL_OFFLINE=1
.venv/bin/python -m llmeval sample
.venv/bin/python -m llmeval generate
.venv/bin/python -m llmeval judge
.venv/bin/python -m llmeval metrics
for b in 512 1024 2048 4096 8192; do .venv/bin/python -m llmeval generate --budget $b; .venv/bin/python -m llmeval judge --budget $b; done
.venv/bin/python -m llmeval agreement
.venv/bin/python -m llmeval agreement_pairs
.venv/bin/python -m llmeval analyze
```

With `LLMEVAL_OFFLINE=1` nothing is sent to a server: a call that is not in the cache fails and is counted instead.
Run this way, the pipeline reproduces the published outputs exactly. Without the embedding cache, the embeddings are
recomputed on a GPU; tiny floating-point differences can then change a few retrieval rankings, which changes those
prompts and sends them outside the cache.

### Level 3: new inference with your own server

Run the same commands without `LLMEVAL_OFFLINE`, pointing `OLLAMA_URL` to an Ollama server (Basic authentication is
read from `~/.netrc`, or from `OLLAMA_USER`/`OLLAMA_PASSWORD`). Pull the tags listed in `configs/experiment.yaml`;
the `digest` of each model identifies the exact weights we used (compare with `ollama list`). Local settings, such as
`OLLAMA_URL` or `OLLAMA_INCIDENT_LOG` (an optional log of server failures), can go in an untracked `.env.local`,
which the `run_*.sh` scripts source.

Expect the results to differ somewhat from ours, and by more than sampling noise in individual answers:

- **Model build and quantisation**: we used the default quantisation of each Ollama tag (Q4_K_M; MXFP4 for
  GPT-OSS) as served by Ollama 0.40.8. Another quantisation, another runtime (vLLM, llama.cpp, transformers) or a
  re-published tag with a different digest changes the answers.
- **Run-to-run variation**: temperature 0 and a fixed seed do not guarantee identical outputs. Concurrent requests
  are batched on the server, and batching changes the floating-point arithmetic. In our check, Qwen3.5-9B repeated
  20 of 20 reasoning answers token for token, but Qwen3.5-4B only 10 of 20
  (`scripts/check_budget_determinism.py`).
- **Hardware and server version**: GPU type, Ollama version and context handling affect long generations.
- **Grading**: the jury is itself an LLM; another build of the judges changes some verdicts.

The conclusions of the paper rest on differences much larger than these effects, but individual numbers will not
match exactly. Levels 1 and 2 reproduce the published numbers exactly.

### Long runs on a shared server

`run_chain.sh` (main design and reasoning), `run_budget.sh` (budget sweep) and `run_agreement.sh` (human reference
point) chain the stages and log milestones to `results/logs/pipeline.log`. A stage exits with code 75 when the server
keeps failing; `aguarda_servidor.sh` waits until no other model is in use on the server and resumes the chain
(`CHAIN=./run_extra.sh` for the budget sweep and the agreement). Generations longer than 4096 tokens are streamed,
because gateways may close connections that stay silent for minutes. `max_inflight` limits the concurrent requests
across all processes.

## Data and licenses

- Code: MIT (`LICENSE`).
- Data released on Zenodo: CC BY 4.0. They contain text from QASPER and QuALITY (both CC BY 4.0) and outputs of
  Qwen, Gemma, Mistral and GPT-OSS models (Apache 2.0) and of Llama models (Llama 3.2 / 3.3 Community License;
  Built with Llama). The data package is built by `scripts/build_zenodo_package.py`.
- The original datasets are not redistributed here; `llmeval sample` downloads them from the pinned commits.

## Citation

See `CITATION.cff`.
