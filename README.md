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

## Reproducibility: choose your path

There are three ways to use this repository. They differ in what you download and in whether a language model is
called at all.

| Path | What you get | Downloads | Needs |
|---|---|---|---|
| **A. Check our numbers** | the paper's tables, figures and numbers, recomputed from our model outputs | Zenodo record (outputs) | Python; no LLM, no GPU |
| **B. Re-run our pipeline exactly** | every stage re-executed from our cached model calls, reproducing our outputs bit for bit | Zenodo record (outputs + caches) | Python; a GPU only for BERTScore |
| **C. Run your own inference** | the whole experiment re-run with your own models | nothing from Zenodo; QASPER/QuALITY are downloaded automatically | an Ollama server with the models; a GPU |

How it works: every model call is cached by a hash of the request (`data/cache/chat.sqlite`), and every stage looks in
the cache before calling the server. With our cache in place, the code finds our answers and never calls a model;
with an empty cache, it calls your server. `LLMEVAL_OFFLINE=1` makes this explicit: no request leaves the machine,
and a call that is not in the cache fails instead of being sent.

### Path A: check our numbers (no LLM, no GPU)

```bash
# download all files of the Zenodo record (DOI in the paper) into a folder, e.g. ~/llm-eval-data-v1.0, then:
.venv/bin/python scripts/install_zenodo_data.py ~/llm-eval-data-v1.0    # checks SHA-256, copies results/, data/sample/, human_eval/
.venv/bin/python -m llmeval sample                                       # pinned QASPER/QuALITY download; rebuilds documents.jsonl
LLMEVAL_OFFLINE=1 .venv/bin/python -m llmeval analyze
```

`analyze` writes the tables (`.tex`) and figures (`.pdf`) to the folder given by `paths.paper` in
`configs/experiment.yaml` (by default `../latex-llmeval`) and every number to `results/analysis/*.csv`.

### Path B: re-run our pipeline exactly (no model server)

```bash
.venv/bin/python scripts/install_zenodo_data.py ~/llm-eval-data-v1.0 --with-caches   # also installs data/cache/*.sqlite
export LLMEVAL_OFFLINE=1
.venv/bin/python -m llmeval sample
.venv/bin/python -m llmeval generate          # 26,400 answers + 4,000 with reasoning, all read from the cache
.venv/bin/python -m llmeval judge
.venv/bin/python -m llmeval metrics           # recomputes the metrics (GPU for BERTScore)
for b in 512 1024 2048 4096 8192; do
  .venv/bin/python -m llmeval generate --budget $b && .venv/bin/python -m llmeval judge --budget $b
done
.venv/bin/python -m llmeval agreement
.venv/bin/python -m llmeval agreement_pairs
.venv/bin/python -m llmeval analyze
```

We checked this path from a clean folder: every output file and every table of the paper came out identical to the
published ones. The embedding cache matters: without it, the embeddings are recomputed, tiny floating-point
differences can change a few retrieval rankings, and those prompts then fall outside the cache.

### Path C: run your own inference

1. **Start from an empty cache.** Do not install our caches (or delete `data/cache/chat.sqlite`); otherwise the code
   reuses our answers instead of calling your models.
2. **Serve the models with Ollama** (<https://ollama.com>) and pull the tags listed in `configs/experiment.yaml`
   (`generators` and `jury.judges`), for example `ollama pull qwen3.5:9b`. Each entry has the `digest` of the weights
   we used; compare it with `ollama list` (a tag can be re-published with different weights). Approximate memory at
   the default 4-bit quantisation with a 32k context: 3B to 9B models fit in 8 to 16 GB of GPU memory, Gemma4-31B and
   Qwen3-32B need about 24 GB, Llama-3.3-70B about 48 GB, GPT-OSS-120B and Qwen3.5-122B about 80 GB.
3. **Point the code to the server**: `export OLLAMA_URL=http://localhost:11434` (the default). For a remote server
   with Basic authentication, put the credentials in `~/.netrc` or in `OLLAMA_USER`/`OLLAMA_PASSWORD`. These
   settings can go in an untracked `.env.local`, which the `run_*.sh` scripts source.
4. **Run the stages** as in path B, without `LLMEVAL_OFFLINE`, or with the chained scripts below. `sample` downloads
   QASPER and QuALITY from the pinned Hugging Face commits and rebuilds exactly our sample.
5. **Scale it to your hardware.** To run fewer models or conditions, edit `generators`, `conditions` and
   `generation` in `configs/experiment.yaml`. To evaluate a new model next to ours, install our caches, add the model
   to `generators` and run normally: only the new model's calls go to your server. `python -m llmeval generate
   --dry-run` counts the calls a run would send. For reference, our run (30,400 answers, about 6,000 jury items per
   judge, the budget sweep and the human reference point) took several days on a shared university server.

Expect your numbers to differ from ours, even with the same tags:

- **Model build and quantisation**: we used the default quantisation of each Ollama tag (Q4_K_M; MXFP4 for
  GPT-OSS) as served by Ollama 0.40.8. Another quantisation, runtime (vLLM, llama.cpp, transformers) or re-published
  tag changes the answers.
- **Run-to-run variation**: temperature 0 and a fixed seed do not guarantee identical outputs, because the server
  batches concurrent requests and batching changes the floating-point arithmetic. In our check, Qwen3.5-9B repeated
  20 of 20 reasoning answers token for token, Qwen3.5-4B only 10 of 20 (`scripts/check_budget_determinism.py`).
- **Hardware and server version**: GPU type, Ollama version and context handling affect long generations.
- **Grading**: the jury is itself a set of LLMs; another build of the judges changes some verdicts.

The conclusions of the paper rest on differences much larger than these effects; individual numbers will not match
exactly. Paths A and B reproduce the published numbers exactly.

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
