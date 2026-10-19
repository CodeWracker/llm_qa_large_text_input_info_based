"""Métricas automáticas de similaridade resposta × referência.

Todas são agregadas como max sobre o conjunto de referências A:  s(â, A) = max_{a∈A} s(â, a).
  - EM e F1 de tokens: normalização SQuAD (minúsculas, sem pontuação/artigos).
  - ROUGE-L (F): maior subsequência comum de tokens.
  - Fuzzy: razão de Levenshtein normalizada sobre caracteres (rapidfuzz).
  - Cosseno de embeddings: bge-m3 (mesmo encoder da recuperação densa).
  - BERTScore F1 (roberta-large, reescalonado pela linha de base).
"""
from __future__ import annotations

import logging
import re
import string
from collections import Counter

import numpy as np
from rapidfuzz import fuzz

from .data import UNANSWERABLE
from .io import read_jsonl, write_jsonl
from .embeddings import Embedder

log = logging.getLogger(__name__)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> list[str]:
    text = _ARTICLES.sub(" ", text.lower().translate(_PUNCT))
    return text.split()


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)
    if not p or not r:
        return float(p == r)
    dp = [0] * (len(r) + 1)
    for tp in p:  # LCS em O(|p||r|) com uma linha
        prev = 0
        for j, tr in enumerate(r, 1):
            cur = dp[j]
            dp[j] = prev + 1 if tp == tr else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def fuzzy(pred: str, ref: str) -> float:
    return fuzz.ratio(pred.lower(), ref.lower()) / 100.0


LEXICAL = {"em": exact_match, "token_f1": token_f1, "rouge_l": rouge_l, "fuzzy": fuzzy}


def _bertscore(pairs: list[tuple[str, str]]) -> np.ndarray:
    from bert_score import BERTScorer
    import torch

    scorer = BERTScorer(
        lang="en", rescale_with_baseline=True, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    _, _, f1 = scorer.score([p for p, _ in pairs], [r for _, r in pairs], batch_size=128)
    return f1.numpy()


def run_metrics(cfg) -> None:
    sample, results = cfg.path("sample"), cfg.path("results")
    questions = {q["qid"]: q for q in read_jsonl(sample / "questions.jsonl")}
    gens = read_jsonl(results / "generations.jsonl")

    # pares (resposta, referência) não triviais; abstenções e referências UNANSWERABLE
    # seguem a convenção do QASPER: 1 se ambos são "não respondível", 0 caso contrário
    pairs, owners = [], []
    for i, g in enumerate(gens):
        refs = [r for r in questions[g["qid"]]["references"] if r != UNANSWERABLE]
        if g["answer"] and not g["abstained"]:
            for r in refs:
                pairs.append((g["answer"], r))
                owners.append(i)
    uniq = sorted(set(pairs))
    pos = {p: j for j, p in enumerate(uniq)}
    log.info("métricas: %d pares únicos", len(uniq))

    texts = sorted({t for p in uniq for t in p})
    e = dict(zip(texts, Embedder(cfg, cfg.path("cache"))(texts)))
    cosine = np.array([float(e[p] @ e[r]) for p, r in uniq])
    bert = _bertscore(uniq)

    per_pair = {}
    for j, (p, r) in enumerate(uniq):
        per_pair[j] = {name: fn(p, r) for name, fn in LEXICAL.items()}
        per_pair[j].update(embed_cos=cosine[j], bertscore=float(bert[j]))

    names = list(LEXICAL) + ["embed_cos", "bertscore"]
    best: dict[int, dict] = {}
    for (p, r), i in zip(pairs, owners):
        s = per_pair[pos[p, r]]
        cur = best.setdefault(i, {n: -np.inf for n in names})
        for n in names:
            cur[n] = max(cur[n], s[n])

    rows = []
    for i, g in enumerate(gens):
        refs = questions[g["qid"]]["references"]
        out = {k: g.get(k, False) for k in ("qid", "dataset", "model", "condition", "think")}
        if i in best:
            out.update({n: float(v) for n, v in best[i].items()})
        else:  # abstenção ou resposta vazia
            hit = float(g["abstained"] and UNANSWERABLE in refs)
            out.update({n: hit for n in names})
        rows.append(out)
    write_jsonl(results / "metrics.jsonl", rows)
