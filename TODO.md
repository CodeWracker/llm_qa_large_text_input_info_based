# TODO

Only what holds today. `[~]` in progress, `[ ]` pending. History: `git log`. Resuming after a crash: see "If the container goes down".

## Experiments
- [x] RQ6 budget sweep and agreement with the QASPER annotators done (29/10), plus the pairwise consensus analysis (29/10); paper updated. Nothing running on the server. `configs/experiment.yaml` has concurrency 4 (shared server); with the server free, 16 is faster for short generations.

## Publication (31/10)
- [x] Datasets pinned (Hugging Face commit + SHA-256, checked on download; rebuilt sample identical) and model digests in the config; annotator anonymised as "Annotator 1"; MIT license (code) and CC BY 4.0 (data); `scripts/build_zenodo_package.py`; README and markdown in English; offline mode `LLMEVAL_OFFLINE=1`.
- [x] Data published on Zenodo: https://doi.org/10.5281/zenodo.23844724 (v1.0; published files match the local MD5). DOI in the README, `CITATION.cff`, `.zenodo.json` and the paper.
- [ ] Fix the fourth author on the Zenodo record ("D. J. de Macedo, Douglas" -> "de Macedo, Douglas Dyllon J."): Edit, then Publish; metadata changes need no new version (user).
- [ ] The GitHub repository is public. Full history scanned on 31/10: no credentials (only placeholders). Decide whether `HANDOFF.md`/`TODO.md`, which list internal paths, stay tracked (user). The annotator's former name in the history is accepted by the user.

## Paper (`../latex-llmeval`)
- [ ] Review float placement (the budget table/figure float into the threats section); check `.bib` metadata
- [ ] Related Work is the user's

## User's pending items
- [ ] Second annotator ("Annotator 2"), if they do it: CSV in `human_eval/results/`, run `scripts/evaluate_human_results.py`; put back in the paper the "annotator 1 vs. 2" row and the agreement between humans
- [ ] Tell the owner/administrator about the key exposed in `/home/jovyan/compartilhado/.ssh/` (do not use it: it is not the user's)

## If the container goes down
1. `/home/jovyan/privado/.persist/return_by_death.sh` (user) and `source /etc/profile.d/99-persist-env.sh`. If `.venv/bin/python` does not exist, recreate it as in HANDOFF (Python in `/home/jovyan/privado/.persist/uv-python`).
2. If a chain was running (`tail results/logs/pipeline.log`), relaunch it with `IGNORAR_ALHEIOS=1 CHAIN=./run_extra.sh setsid nohup ./aguarda_servidor.sh > results/logs/aguarda_extra.log 2>&1 < /dev/null &` (the cache resumes from where it stopped).
3. Recreate the hourly check loop (CronCreate) if the session died.

## Decisions in force
- Paper: data extraction for systematic reviews; QASPER is the target and QuALITY the control; no RQ7; IEEE conference.
- 400-question sample; 3-LLM jury; reasoning only on Qwen3.5-4B and 9B, 5 conditions, budgets 512–8192.
- Server (28/10): share it with other users instead of waiting (IGNORAR_ALHEIOS=1).
- Human validation: 120 answers, one annotator ("Annotator 1", 27/10: jury 85.0%, kappa 0.65); the yardstick for the models is the agreement among humans, not 100%.
- Cite only datasets/methods, plus Liu 2024, Schick 2023 and Souza & Panisson 2026 (requested by the user).
- Licenses (31/10): code MIT, data CC BY 4.0.
