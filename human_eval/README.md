# Human validation of the jury on QASPER (RQ1)

1. Open `qasper_annotator.html` in a browser (a local file; no server needed), choose the annotator and grade with
   the keys `1`/`2`/`3`. Votes are saved in the browser; the button that downloads the results produces
   `annotations_<annotator>.csv`. **Do not open `sample_key.json` before annotating**: it shows the model, the
   condition and the jury's votes.
2. Put the CSVs in `results/` and run `python scripts/evaluate_human_results.py` with the project's `.venv` (it needs
   `krippendorff` and `scikit-learn`). It writes `agreement_report.json`, `agreement_report.tex` and a copy of the
   macros to `../latex-llmeval/tables/human_agreement.tex`, which fill the instrument table, RQ1, the threats and the
   abstract of the paper (until that file exists, the numbers appear as red dashes).

The sample (`python scripts/build_human_eval_html.py`, seed 42) has 120 answers given without reasoning: 40
extractive, 40 abstractive and 40 yes/no, 20 per generator, half with the full document and half with hybrid
retrieval with k=5, from 120 different questions, and only answers graded by the jury (abstentions, graded by rule,
are left out). The annotation was done by one annotator (`Annotator 1`); the page also offers `Annotator 2`.
