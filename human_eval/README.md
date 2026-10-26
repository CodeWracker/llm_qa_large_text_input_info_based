# Validação humana do júri no QASPER (RQ1)

1. Abra `qasper_annotator.html` no navegador (arquivo local, sem servidor), escolha o anotador e vote com
   as teclas `1`/`2`/`3`. Os votos ficam salvos no navegador; o botão **Baixar CSV de Resultados** gera
   `annotations_<anotador>.csv`. **Não abra `sample_key.json` antes de anotar**: ele traz modelo, condição e votos do júri.
2. Coloque os CSVs em `results/` e rode, com o `.venv` do projeto (precisa de `krippendorff` e `scikit-learn`):
   `python scripts/evaluate_human_results.py`. Saem `agreement_report.json`, `agreement_report.tex` e a cópia
   das macros em `../latex-llmeval/tables/human_agreement.tex`, que preenchem a Tabela de instrumentos, a RQ1,
   as ameaças e o resumo do artigo (enquanto o arquivo não existe, os números aparecem como traços vermelhos).

A amostra (`python scripts/build_human_eval_html.py`, seed 42) tem 120 respostas sem raciocínio: 40 extractive,
40 abstractive e 40 yes/no, 20 por gerador, metade com documento completo e metade com híbrido k=5, 120 perguntas
distintas, só respostas julgadas pelo júri (abstenções decididas por regra ficam de fora).
