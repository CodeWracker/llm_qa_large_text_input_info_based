"""Estágio de julgamento: júri de LLMs (QASPER) e meta-avaliação do júri (QuALITY).

Casos de abstenção são decididos por regra, sem júri:
  modelo abstém  e UNANSWERABLE ∈ referências  -> correto
  modelo abstém  e UNANSWERABLE ∉ referências  -> incorreto
  modelo responde e todas as referências são UNANSWERABLE -> incorreto
Nos demais casos cada juiz j emite v_j ∈ {correct, partially_correct, incorrect};
veredito final = maioria estrita de "correct"; nota graduada = média de s(v_j).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .data import UNANSWERABLE
from .io import read_jsonl, write_jsonl
from .ollama import OllamaClient
from .prompts import JUDGE_SCHEMA, VERDICT_SCORE, judge_messages

log = logging.getLogger(__name__)


def rule_verdict(row: dict, refs: list[str]) -> str | None:
    if row["dataset"] != "qasper":
        return None
    has_unans = UNANSWERABLE in refs
    if row["abstained"]:
        return "correct" if has_unans else "incorrect"
    if not row["answer"] or all(r == UNANSWERABLE for r in refs):
        return "incorrect"
    return None


def run_judging(cfg) -> None:
    sample, results = cfg.path("sample"), cfg.path("results")
    questions = {q["qid"]: q for q in read_jsonl(sample / "questions.jsonl")}
    gens = read_jsonl(results / "generations.jsonl")
    client = OllamaClient(cfg, cfg.path("cache"))
    judges = cfg["jury"]["judges"]
    quality_conds = set(cfg["jury"]["quality_conditions"])

    # 1) decide o que vai para o júri; respostas idênticas à mesma pergunta são julgadas uma vez
    items, todo = [], {}
    for row in gens:
        q = questions[row["qid"]]
        if row["dataset"] == "quality" and (row["condition"] not in quality_conds or not row["answer"]):
            continue
        refs = [r for r in q["references"] if r != UNANSWERABLE]
        rule = rule_verdict(row, q["references"])
        key = None
        if rule is None:
            key = (q["question"], tuple(refs), row["answer"])
            todo[key] = None
        items.append((row, rule, key))

    # 2) consulta cada juiz (um modelo por vez, chamadas em paralelo)
    verdicts: dict[tuple, dict[str, dict]] = {k: {} for k in todo}
    for judge in judges:
        def call(key, judge=judge):
            question, refs, answer = key
            res = client.chat_json(
                judge["model"], judge_messages(question, list(refs), answer), JUDGE_SCHEMA,
                think=judge.get("think", False), num_predict=2048,
            )
            return key, res.get("output")
        with ThreadPoolExecutor(cfg["ollama"]["concurrency"]) as ex:
            futures = [ex.submit(call, k) for k in todo]
            for f in tqdm(as_completed(futures), total=len(futures), desc=judge["label"]):
                key, out = f.result()
                verdicts[key][judge["model"]] = out

    # 3) agrega
    rows = []
    for row, rule, key in items:
        out = {k: row[k] for k in ("qid", "dataset", "model", "condition")}
        if rule is not None:
            out.update(source="rule", verdict=rule, correct=rule == "correct", score=VERDICT_SCORE[rule])
        else:
            votes = {j: (v or {}).get("verdict") for j, v in verdicts[key].items()}
            valid = [v for v in votes.values() if v in VERDICT_SCORE]
            n_correct = sum(v == "correct" for v in valid)
            out.update(
                source="jury",
                votes=votes,
                justifications={j: (v or {}).get("justification") for j, v in verdicts[key].items()},
                correct=bool(valid) and n_correct > len(valid) / 2,
                score=sum(VERDICT_SCORE[v] for v in valid) / len(valid) if valid else None,
            )
        rows.append(out)
    write_jsonl(results / "judgments.jsonl", rows)
    log.info("julgamento concluído: %d itens, %d chamadas únicas por juiz", len(rows), len(todo))
