# Handoff (read first, ~2 min)

Internal working notes for whoever continues the work (they mention the study's own machine). Work state and
progress: `TODO.md`. Design and reproduction: `README.md`.

## What this is
Revision of a shelved study: open LLMs (UFSC Ollama server) answering questions about long documents (QASPER,
QuALITY), comparing the full document with RAG and model-guided search, with a 3-LLM jury validated against an answer
key and a human annotator. IEEE paper in `../latex-llmeval`.

## Repositories (they already exist; never run `git init`)
- `llm_qa_large_text_input_info_based/`: code, GitHub remote, author `CodeWracker`.
- `latex-llmeval/`: paper, Overleaf remote. Token in `/home/jovyan/privado/.persist/git/credentials`
  (credential helper `store`, via `~/.gitconfig` -> ZFS; `return_by_death.sh` recreates the link).
- Commit messages in Portuguese ("adiciona…", "corrige…"), one per logical unit, pushed right away.
  **Never** add Co-Authored-By or any attribution to Claude.

## The user's rules for the paper
- One continuous narrative: intro → problem/RQs → method/formalisation → setup → results answering the RQs.
  Anticipate "ok, but how was this done?".
- The paper describes ONE continuous study: never mention pilots, earlier versions or the chronology of the runs.
- No result numbers in the introduction.
- `\todo{...}` = red text only (`\textcolor{red}`), always in Portuguese, NEVER boxes/todonotes.
- Related Work stays as `\todo{}` (axes and search strings); the user writes it.
- Human-validation numbers enter through `\hum{key}` macros (`tables/human_agreement.tex`, written by
  `scripts/evaluate_human_results.py`; see `human_eval/README.md`).
- Cite only datasets, techniques and computational/statistical methods, plus the references the user asked for.
- Decisions in force: see the end of `TODO.md`.
- Keep handoff files short; if more is needed, use small files in `handoffs/`.

## Environment (recreate if the container was reset)
```bash
source /etc/profile.d/99-persist-env.sh      # OLLAMA_URL
# the container may run as root or as jovyan; HOME is ephemeral: run
# /home/jovyan/privado/.persist/return_by_death.sh to recreate the links (~/.ssh, ~/.netrc, ~/.gitconfig)
# if git reports "dubious ownership": sudo chown -R jovyan:users /home/jovyan/compartilhado/ralph/llm-eval
# (~/.netrc = Ollama auth; ~/.gitconfig = Overleaf credential; ~/.ssh = GitHub key, account CodeWracker)
cd llm_qa_large_text_input_info_based
# the .venv uses a Python installed in a persistent folder (HOME is wiped on restart); if it breaks, recreate:
export UV_PYTHON_INSTALL_DIR=/home/jovyan/privado/.persist/uv-python && uv python install 3.12
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# local LaTeX: /home/jovyan/privado/.persist/bin/tectonic -X compile main.tex
```
`.env.local` (untracked) sets `OLLAMA_INCIDENT_LOG`; the `run_*.sh` scripts source it.
The Ollama server is shared with other projects: `max_inflight` (config) caps concurrent requests across all
processes (locks in `data/cache/slots`); 16 with the server to ourselves, 4 when shared (longer generations otherwise
wait in the server queue past the gateway's 600 s and come back as 504). One model at a time. Embeddings run locally
on the GPU (`CUDA_VISIBLE_DEVICES=1`).

## Pipeline
`python -m llmeval {sample,generate,judge,metrics,analyze}` plus `agreement` and `agreement_pairs`. Everything is
cached in `data/cache/*.sqlite`, so re-running is cheap; `LLMEVAL_OFFLINE=1` forbids server calls (reproduction
from the cache, checked on 31/10: generation identical). The analysis writes `tables/*.tex` and `figures/*.pdf`
straight into `../latex-llmeval`. Milestones in `results/logs/pipeline.log`; if a run dies, relaunch it.
- `run_chain.sh`: main design and RQ6 (done).
- `run_extra.sh`: `run_budget.sh` (budget sweep, `generate/judge --budget B`, `--dry-run` only counts calls) and then
  `run_agreement.sh` (human reference point). The client derives from the cache of another budget the answers whose
  outcome is already known (`OllamaClient._project`); models in `nondeterministic_models` are only projected to
  smaller budgets. Generations above 4096 tokens are streamed.
- `scripts/build_zenodo_package.py` builds the data package in `dist/zenodo/`.

## Server incidents and the loop
- Every Ollama failure (HTTP, timeout, connection) and the recovery go to the file in `OLLAMA_INCIDENT_LOG`
  (the user reports them to SeTIC). `relatorio.py` in the same folder writes `RELATORIO_INCIDENTES_OLLAMA.md`.
- Circuit breaker: `circuit_breaker` consecutive server failures end the stage with code 75 and the chain pauses.
- `aguarda_servidor.sh` (watcher, PID in `results/logs/aguarda.pid`) waits 3 min with no other model in use
  (`expires_at` unchanged; infinite keep_alive counts as busy), runs `$CHAIN` and watches again if it pauses (75);
  any other code ends it. Variables: `CHAIN` (default `./run_chain.sh`), `IGNORAR_MODELOS="m1 m2"` (other users'
  models released by the user), `IGNORAR_ALHEIOS=1` (whole server ceded). Current command: see "If the container
  goes down" in `TODO.md`.
- A loop (CronCreate; expires in 7 days; dies with the session) checks the chain and the watcher and relaunches the
  watcher if it died. If the session died, recreate the loop. Kill processes only by PID (never `pkill -f`).
- After a machine reboot: `OLLAMA_URL` and `~/.netrc` disappear until the user runs
  `/home/jovyan/privado/.persist/return_by_death.sh`.
