# Data for "Reading the Whole Paper or Retrieving What Matters? Open-Weight LLMs for Data Extraction in Systematic Literature Reviews, Evaluated by a Validated LLM Jury"

Version {{VERSION}}. Code: <https://github.com/CodeWracker/llm_qa_large_text_input_info_based>.

This record holds everything the study produced: the answers of six open-weight language models to 400 questions
about long documents, the verdicts of a three-model LLM jury, automatic similarity metrics, a reasoning-budget
sweep, a human validation of the jury and a comparison of the models with the agreement among the QASPER
annotators. It also holds the caches that let the whole pipeline be re-run offline, reproducing these files exactly.

## Study in brief

- **Questions**: a stratified sample (seed 42) of 200 QASPER questions about NLP papers (50 per answer type:
  extractive, abstractive, yes/no, unanswerable) and 200 QuALITY multiple-choice questions about stories and
  articles (100 easy, 100 hard), over 272 documents of about 650 to 10,300 words.
- **Generators**: Llama-3.2-3B, Qwen3.5-4B, Mistral-7B, Qwen3.5-9B, Gemma4-31B and Qwen3.5-122B, served by Ollama
  0.40.8 with the default quantisation of each tag (Q4_K_M), temperature 0, fixed seed, a 32,768-token context and
  JSON-schema-constrained output. Model tags and digests are in `configs/experiment.yaml`.
- **Context conditions** (11): closed book, full document, BM25 / dense (BGE-M3) / hybrid (RRF) retrieval
  (hybrid with k = 2, 5, 10; the others with k = 5), and four variants of a keyword search chosen by the model.
- **Reasoning**: Qwen3.5-4B and Qwen3.5-9B answered five conditions again with reasoning enabled, under generation
  budgets of 512, 1024, 2048, 4096 and 8192 tokens.
- **Grading**: abstentions by rule; other answers by a jury of GPT-OSS-120B, Llama-3.3-70B and Qwen3-32B (strict
  majority), validated against the QuALITY answer key and against a human annotator on 120 QASPER answers.

## Files

{{FILES}}

Not included:

{{EXCLUDED}}

## Main fields

- `generations*.jsonl`: one answer per line. `qid`, `dataset`, `model`, `condition`, `think` (reasoning on/off),
  `answer`, `abstained`, `context_words`, `evidence_recall` (QASPER, retrieval conditions), `prompt_tokens`,
  `completion_tokens` (reasoning included), `done_reason` (`length` = budget exhausted), `error`; search conditions
  add `search_terms`, `search_matched` and the token counts of the term-selection call; QuALITY adds `choice` and
  `correct` (against the key). Budget files add `budget` and `projected_from` (see below).
- `judgments*.jsonl`: `source` (`rule` or `jury`), `correct` (strict majority of `correct` votes), `score` (mean of
  correct = 1, partially correct = 0.5, incorrect = 0), `votes` and `justifications` per judge.
- `metrics.jsonl`: `em`, `token_f1`, `rouge_l`, `fuzzy`, `embed_cos`, `bertscore`, each the maximum over references.
- `agreement.jsonl`: `candidate` (`human` or a model), `left_out` (index of the annotator left out), `in_sample`,
  `stratum`, `correct` against the remaining annotations. `agreement_pairs.jsonl`: `candidate` (`human:i` or a model),
  `ref` (index of the single annotation it is compared with), `correct`.
- `questions.jsonl`: `references` (one per distinct annotation; `UNANSWERABLE` marks an unanswerable annotation),
  `answer_types`, `evidence_paragraphs` (paragraph indices of `documents.jsonl`), `doc_words`.

Answers under the smaller budgets were derived from the traces generated under larger budgets when the outcome was
already known (`projected_from`): at temperature 0 a trace that finished with t tokens is identical under any budget
of at least t and is cut under any smaller budget. Qwen3.5-4B did not reproduce its own outputs at temperature 0
(10 of 20 identical in a check), so under the budget of 8192 all its answers were generated anew.

## How to use

Clone the code repository and place the files of this record in it (`results/`, `data/sample/`, `human_eval/`,
and the caches, decompressed, in `data/cache/`). The repository README describes three levels of reproduction:

1. **Tables and figures** from these outputs: `python -m llmeval analyze` (no LLM, no GPU).
2. **The whole pipeline from the caches**: with `chat.sqlite` and `embeddings_local.sqlite` in `data/cache/` and
   `LLMEVAL_OFFLINE=1`, every stage re-runs without a model server and reproduces these files exactly.
3. **New inference** with your own Ollama server: the same model tags and settings; results will differ somewhat
   with the model build, quantisation, server version and hardware.

Check the files with `sha256sum -c MANIFEST.sha256`.

## Licenses

The data in this record are released under CC BY 4.0 (`LICENSE`). They contain text from QASPER and QuALITY
(both CC BY 4.0) and outputs of third-party models; see `NOTICE` for attributions and terms. The code is MIT-licensed
in the repository.

## Citation

Please cite the paper (citation to be added on publication) and this record.
