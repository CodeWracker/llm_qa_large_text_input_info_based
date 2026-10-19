"""Estágio de geração: (pergunta × modelo × condição) → resposta.

Também grava results/retrieval.jsonl com a revocação de evidência de cada recuperador
para vários k (calculada sem chamar os geradores).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .data import document_text
from .io import read_jsonl, write_jsonl
from .ollama import OllamaClient
from .prompts import LETTERS, generation_messages, is_abstention
from .retrieval import DocumentIndex, chunk_document, evidence_recall, format_chunks

log = logging.getLogger(__name__)
METHODS = ("bm25", "dense", "hybrid")


def build_indices(cfg, client: OllamaClient, docs: list[dict]) -> dict[str, DocumentIndex]:
    ch, rt = cfg["chunking"], cfg["retrieval"]
    indices = {}
    for doc in tqdm(docs, desc="indexando documentos"):
        chunks = chunk_document(doc, ch["target_words"], ch["max_words"])
        emb = client.embed(rt["embedding_model"], [f"{c.section}\n{c.text}".strip() for c in chunks])
        indices[doc["doc_id"]] = DocumentIndex(chunks, emb, rt["bm25_k1"], rt["bm25_b"], rt["rrf_k0"])
    return indices


def retrieval_report(cfg, indices, questions, q_emb) -> list[dict]:
    rows = []
    for q in questions:
        idx = indices[q["doc_id"]]
        for m in METHODS:
            ranking = idx.rank(m, q["question"], q_emb[q["qid"]])
            for k in cfg["retrieval"]["recall_ks"]:
                top = [idx.chunks[i] for i in ranking[:k]]
                rows.append({
                    "qid": q["qid"], "dataset": q["dataset"], "method": m, "k": k,
                    "n_chunks": len(idx.chunks),
                    "recall": evidence_recall(top, q.get("evidence_paragraphs", [])),
                })
    return rows


def run_generation(cfg) -> None:
    sample, results = cfg.path("sample"), cfg.path("results")
    docs = {d["doc_id"]: d for d in read_jsonl(sample / "documents.jsonl")}
    questions = read_jsonl(sample / "questions.jsonl")
    client = OllamaClient(cfg, cfg.path("cache"))

    indices = build_indices(cfg, client, list(docs.values()))
    emb = client.embed(cfg["retrieval"]["embedding_model"], [q["question"] for q in questions])
    q_emb = {q["qid"]: e for q, e in zip(questions, emb)}
    write_jsonl(results / "retrieval.jsonl", retrieval_report(cfg, indices, questions, q_emb))

    # contexto de cada (pergunta, condição) é independente do gerador
    contexts: dict[tuple[str, str], dict] = {}
    for q in questions:
        for cond in cfg["conditions"]:
            kind = cond["context"]
            if kind == "full":
                text, rec = document_text(docs[q["doc_id"]]), None
            elif kind == "rag":
                got = indices[q["doc_id"]].retrieve(cond["retriever"], q["question"], cond["k"], q_emb[q["qid"]])
                text, rec = format_chunks(got), evidence_recall(got, q.get("evidence_paragraphs", []))
            else:
                text, rec = None, None
            contexts[q["qid"], cond["name"]] = {
                "kind": kind, "text": text, "evidence_recall": rec,
                "context_words": len(text.split()) if text else 0,
            }

    def job(gen, q, cond):
        ctx = contexts[q["qid"], cond["name"]]
        messages, schema = generation_messages(q, ctx["kind"], ctx["text"])
        res = client.chat_json(gen["model"], messages, schema, think=False, num_predict=256)
        out = res.get("output") or {}
        answer = (out.get("answer") or "").strip()
        row = {
            "qid": q["qid"], "dataset": q["dataset"], "model": gen["model"], "condition": cond["name"],
            "answer": answer, "abstained": is_abstention(answer),
            "context_words": ctx["context_words"], "evidence_recall": ctx["evidence_recall"],
            **{k: res.get(k) for k in ("prompt_tokens", "completion_tokens", "latency_s", "truncated", "error")},
        }
        if q["dataset"] == "quality":
            choice = out.get("choice")
            row["choice"] = choice
            row["correct"] = choice is not None and LETTERS.index(choice) == q["gold_index"]
        return row

    rows = []
    # um modelo por vez: evita disputar memória do servidor carregando vários modelos
    for gen in cfg["generators"]:
        tasks = [(gen, q, c) for q in questions for c in cfg["conditions"]]
        with ThreadPoolExecutor(cfg["ollama"]["concurrency"]) as ex:
            futures = [ex.submit(job, *t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc=gen["label"]):
                rows.append(f.result())
        write_jsonl(results / "generations.jsonl", rows)  # checkpoint por modelo
    n_err = sum(1 for r in rows if r.get("error"))
    log.info("geração concluída: %d respostas, %d falhas de formato", len(rows), n_err)
