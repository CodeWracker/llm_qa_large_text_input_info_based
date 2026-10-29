"""Estágio de concordância com humanos no QASPER: quanto os anotadores concordam entre si e os modelos com eles.

Deixa um anotador de fora de cada vez: a anotação dele vira a candidata e as demais, as referências. O júri (ou a
regra de abstenção) decide se a candidata está correta, exatamente como faz com as respostas dos modelos. Assim,
"humano contra os outros humanos" vira uma acurácia na mesma escala que a dos modelos.

  - todas as perguntas do split de teste com >= 2 anotações: só candidatos humanos;
  - perguntas da amostra: também a resposta de cada gerador (documento completo, sem raciocínio), julgada contra
    os MESMOS conjuntos de referências (um por anotador deixado de fora), para a comparação ser justa.

Grava results/agreement.jsonl. Com --dry-run, só conta os itens que iriam ao júri.
"""
from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from .data import UNANSWERABLE, _qasper_answer
from .io import read_jsonl, write_jsonl
from .judge import aggregate, ask_jury, rule_verdict
from .ollama import OllamaClient
from .prompts import UNANSWERABLE_REFERENCE, VERDICT_SCORE

log = logging.getLogger(__name__)


def _annotations(cfg) -> list[dict]:
    """Uma entrada por pergunta do split de teste: tipos e textos de referência de cada anotador, na ordem."""
    df = pd.read_parquet(cfg.path("raw") / f"qasper_{cfg['sample']['qasper_split']}.parquet")
    out = []
    for _, row in df.iterrows():
        for qtext, qid, ans in zip(row.qas["question"], row.qas["question_id"], row.qas["answers"]):
            ann = [_qasper_answer(a) for a in ans["answer"]]
            ann = [(t, r) for t, r in ann if r]  # anotação sem texto não serve de candidata nem de referência
            if len(ann) >= 2:
                out.append({"qid": qid, "question": qtext.strip(), "annotations": ann,
                            "stratum": Counter(t for t, _ in ann).most_common(1)[0][0]})
    return out


def run_agreement(cfg) -> None:
    results = cfg.path("results")
    sample_q = {q["qid"] for q in read_jsonl(cfg.path("sample") / "questions.jsonl") if q["dataset"] == "qasper"}
    models = [g for g in read_jsonl(results / "generations.jsonl")
              if g["dataset"] == "qasper" and not g.get("think") and g["condition"] == "full"]
    by_q: dict[str, list[dict]] = {}
    for g in models:
        by_q.setdefault(g["qid"], []).append(g)

    items, todo = [], {}
    for q in _annotations(cfg):
        in_sample = q["qid"] in sample_q
        for k, (ktype, ktext) in enumerate(q["annotations"]):
            raw_refs = list(dict.fromkeys(r for i, (_, r) in enumerate(q["annotations"]) if i != k))
            refs = [UNANSWERABLE_REFERENCE if r == UNANSWERABLE else r for r in raw_refs]
            cands = [{"candidate": "human", "cand_type": ktype, "answer": ktext, "abstained": ktype == "unanswerable"}]
            if in_sample:
                cands += [{"candidate": g["model"], "cand_type": None, "answer": g["answer"],
                           "abstained": g["abstained"]} for g in by_q.get(q["qid"], [])]
            for c in cands:
                rule = rule_verdict({"dataset": "qasper", **c}, raw_refs)
                key = None
                if rule is None:
                    key = (q["question"], tuple(refs), c["answer"])
                    todo[key] = None
                items.append(({"qid": q["qid"], "in_sample": in_sample, "stratum": q["stratum"], "left_out": k,
                               "n_annotations": len(q["annotations"]), "n_refs": len(raw_refs),
                               "candidate": c["candidate"], "cand_type": c["cand_type"]}, rule, key))
    n_h = sum(1 for it, _, _ in items if it["candidate"] == "human")
    log.info("concordância: %d itens (%d humanos, %d de modelos), %d chamadas únicas por juiz",
             len(items), n_h, len(items) - n_h, len(todo))
    if cfg["generation"].get("dry_run"):
        return

    client = OllamaClient(cfg, cfg.path("cache"))
    verdicts = ask_jury(cfg, client, todo)
    rows = []
    for meta, rule, key in items:
        if rule is not None:
            rows.append({**meta, "source": "rule", "verdict": rule, "correct": rule == "correct",
                         "score": VERDICT_SCORE[rule]})
        else:
            out = aggregate(verdicts[key])
            out.pop("justifications")
            rows.append({**meta, **out})
    missing = sum(1 for v in verdicts.values() for j in cfg["jury"]["judges"] if v.get(j["model"]) is None)
    if missing:
        log.warning("%d votos ausentes; reexecute `agreement` para refazê-los", missing)
    write_jsonl(results / "agreement.jsonl", rows)
    log.info("concordância concluída: %d itens", len(rows))


def run_agreement_pairs(cfg) -> None:
    """Comparações par a par nas perguntas de texto livre da amostra: cada anotador contra cada outro anotador
    sozinho e cada modelo contra cada anotador sozinho. Com dois anotadores isso coincide com o "deixa um de
    fora" (e sai do cache); com três ou mais, separa "bate com todos" de "bate com alguns" sem ambiguidade.
    Grava results/agreement_pairs.jsonl."""
    results = cfg.path("results")
    sample_q = {q["qid"] for q in read_jsonl(cfg.path("sample") / "questions.jsonl") if q["dataset"] == "qasper"}
    by_q: dict[str, list[dict]] = {}
    for g in read_jsonl(results / "generations.jsonl"):
        if g["dataset"] == "qasper" and not g.get("think") and g["condition"] == "full":
            by_q.setdefault(g["qid"], []).append(g)

    items, todo = [], {}
    for q in _annotations(cfg):
        if q["qid"] not in sample_q or q["stratum"] not in ("extractive", "abstractive"):
            continue
        cands = [(f"human:{i}", {"answer": t, "abstained": typ == "unanswerable"}) for i, (typ, t) in
                 enumerate(q["annotations"])]
        cands += [(g["model"], {"answer": g["answer"], "abstained": g["abstained"]}) for g in by_q.get(q["qid"], [])]
        for name, c in cands:
            for j, (_, ref) in enumerate(q["annotations"]):
                if name == f"human:{j}":
                    continue
                refs = [UNANSWERABLE_REFERENCE if ref == UNANSWERABLE else ref]
                rule = rule_verdict({"dataset": "qasper", **c}, [ref])
                key = None
                if rule is None:
                    key = (q["question"], tuple(refs), c["answer"])
                    todo[key] = None
                items.append(({"qid": q["qid"], "candidate": name, "ref": j, "n_annotations": len(q["annotations"])},
                              rule, key))
    log.info("pares: %d comparações, %d chamadas únicas por juiz", len(items), len(todo))
    if cfg["generation"].get("dry_run"):
        return
    verdicts = ask_jury(cfg, OllamaClient(cfg, cfg.path("cache")), todo)
    rows = []
    for meta, rule, key in items:
        if rule is not None:
            rows.append({**meta, "source": "rule", "correct": rule == "correct"})
        else:
            out = aggregate(verdicts[key])
            rows.append({**meta, "source": "jury", "correct": out["correct"], "votes": out["votes"]})
    write_jsonl(results / "agreement_pairs.jsonl", rows)
    log.info("pares concluídos: %d comparações", len(rows))
