"""Download, normalização e amostragem estratificada de QASPER e QuALITY.

Produz dois arquivos em data/sample/:
  documents.jsonl  {doc_id, dataset, title, paragraphs:[{section, text}]}
  questions.jsonl  {qid, doc_id, dataset, question, stratum, references, ...}

Convenções:
  - Respostas "não respondíveis" do QASPER viram a referência literal UNANSWERABLE.
  - No QuALITY a referência é somente a alternativa correta (gold_label, 1-indexado).
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .io import write_jsonl

log = logging.getLogger(__name__)

UNANSWERABLE = "UNANSWERABLE"
# arquivo de um commit fixo do Hugging Face (configs/experiment.yaml, `sources`): o conteúdo não muda com o tempo
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(sources: dict, raw_dir: Path, splits: dict[str, str]) -> None:
    """Baixa cada (dataset, split) do commit fixado e confere o sha256; um arquivo já presente também é conferido."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        src = sources[name]
        spec = src["files"][split]
        dest = raw_dir / f"{name}_{split}.parquet"
        if not dest.exists():
            url = HF_FILE.format(repo=src["repo"], revision=src["revision"], path=spec["path"])
            log.info("baixando %s", url)
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
        got = _sha256(dest)
        if got != spec["sha256"]:
            raise RuntimeError(f"{dest}: sha256 {got} difere do fixado {spec['sha256']}; apague o arquivo e baixe de novo")


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------- QASPER
def _qasper_answer(a: dict) -> tuple[str, str]:
    """(tipo, texto de referência) de uma anotação do QASPER."""
    if a["unanswerable"]:
        return "unanswerable", UNANSWERABLE
    if a["yes_no"] is not None:
        return "yes_no", "Yes" if a["yes_no"] else "No"
    spans = [s.strip() for s in a["extractive_spans"] if s.strip()]
    if spans:
        return "extractive", ", ".join(spans)
    return "abstractive", a["free_form_answer"].strip()


def parse_qasper(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    docs, questions = [], []
    for _, row in df.iterrows():
        paragraphs = []
        for section, paras in zip(row.full_text["section_name"], row.full_text["paragraphs"]):
            for p in paras:
                if p and p.strip():
                    paragraphs.append({"section": section or "", "text": _norm_ws(p)})
        # legendas de figuras/tabelas fazem parte do texto disponível ao modelo
        for cap in row.figures_and_tables["caption"]:
            if cap and cap.strip():
                paragraphs.append({"section": "Figures and Tables", "text": _norm_ws(cap)})
        docs.append({"doc_id": row.id, "dataset": "qasper", "title": row.title, "paragraphs": paragraphs})

        para_index = {p["text"]: i for i, p in enumerate(paragraphs)}
        for qtext, qid, ans in zip(row.qas["question"], row.qas["question_id"], row.qas["answers"]):
            types, refs, evidence_sets = [], [], []
            for a in ans["answer"]:
                t, ref = _qasper_answer(a)
                types.append(t)
                if ref and ref not in refs:
                    refs.append(ref)
                ev = []
                for e in a["evidence"]:
                    e = _norm_ws(e.replace("FLOAT SELECTED:", ""))
                    if e in para_index:
                        ev.append(para_index[e])
                if ev:
                    evidence_sets.append(sorted(set(ev)))
            if not refs:
                continue
            stratum = Counter(types).most_common(1)[0][0]  # tipo majoritário entre anotadores
            questions.append({
                "qid": qid,
                "doc_id": row.id,
                "dataset": "qasper",
                "question": qtext.strip(),
                "stratum": stratum,
                "answer_types": types,
                "references": refs,
                "evidence_paragraphs": evidence_sets,
            })
    return docs, questions


# --------------------------------------------------------------------- QuALITY
def _quality_paragraphs(article: str) -> list[dict]:
    # parágrafos separados por linha em branco ou por quebra seguida de texto sem recuo;
    # quebras "\n " são apenas quebras de linha do texto original
    parts = re.split(r"\n\s*\n|\n(?=\S)", article)
    return [{"section": "", "text": _norm_ws(p)} for p in parts if p.strip()]


def parse_quality(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    docs, questions = {}, []
    for _, row in df.iterrows():
        doc_id = str(row.article_id)
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "dataset": "quality",
                "title": row.title,
                "paragraphs": _quality_paragraphs(row.article),
            }
        options = [str(o) for o in row.options]
        gold = int(row.gold_label) - 1
        questions.append({
            "qid": row.question_unique_id,
            "doc_id": doc_id,
            "dataset": "quality",
            "question": row.question.strip(),
            "stratum": "hard" if row.difficult else "easy",
            "options": options,
            "gold_index": gold,
            "references": [options[gold]],
        })
    return list(docs.values()), questions


# ------------------------------------------------------------------ amostragem
def stratified_sample(
    questions: list[dict], per_stratum: int, max_per_doc: int, rng: np.random.Generator
) -> list[dict]:
    """Amostra `per_stratum` perguntas por estrato, no máximo `max_per_doc` por documento.

    A ordem é embaralhada uma vez; percorremos estratos do menor para o maior para que
    o limite por documento não esgote os estratos raros.
    """
    order = rng.permutation(len(questions))
    by_stratum: dict[str, list[dict]] = {}
    for i in order:
        by_stratum.setdefault(questions[i]["stratum"], []).append(questions[i])
    per_doc: Counter = Counter()
    chosen = []
    for stratum in sorted(by_stratum, key=lambda s: len(by_stratum[s])):
        taken = 0
        for q in by_stratum[stratum]:
            if taken == per_stratum:
                break
            if per_doc[q["doc_id"]] < max_per_doc:
                per_doc[q["doc_id"]] += 1
                chosen.append(q)
                taken += 1
        if taken < per_stratum:
            log.warning("estrato %s: apenas %d perguntas disponíveis", stratum, taken)
    return chosen


def build_sample(cfg) -> tuple[list[dict], list[dict]]:
    raw = cfg.path("raw")
    s = cfg["sample"]
    download(cfg["sources"], raw, {"qasper": s["qasper_split"], "quality": s["quality_split"]})
    rng = np.random.default_rng(cfg["seed"])

    q_docs, q_qs = parse_qasper(pd.read_parquet(raw / f"qasper_{s['qasper_split']}.parquet"))
    l_docs, l_qs = parse_quality(pd.read_parquet(raw / f"quality_{s['quality_split']}.parquet"))

    questions = stratified_sample(q_qs, s["qasper_per_type"], s["max_questions_per_doc"], rng)
    questions += stratified_sample(l_qs, s["quality_per_difficulty"], s["max_questions_per_doc"], rng)

    used = {q["doc_id"] for q in questions}
    docs = [d for d in q_docs + l_docs if d["doc_id"] in used]
    lengths = {d["doc_id"]: sum(len(p["text"].split()) for p in d["paragraphs"]) for d in docs}
    for q in questions:
        q["doc_words"] = lengths[q["doc_id"]]

    out = cfg.path("sample")
    write_jsonl(out / "documents.jsonl", docs)
    write_jsonl(out / "questions.jsonl", questions)
    log.info("amostra: %d perguntas, %d documentos", len(questions), len(docs))
    return docs, questions


def document_text(doc: dict) -> str:
    """Texto completo do documento, com cabeçalhos de seção, para a condição de contexto completo."""
    lines, current = [f"Title: {doc['title']}"], None
    for p in doc["paragraphs"]:
        if p["section"] and p["section"] != current:
            current = p["section"]
            lines.append(f"\n## {current}")
        lines.append(p["text"])
    return "\n".join(lines)
