"""Estágio de geração: (pergunta × modelo × condição × raciocínio) → resposta.

Condições de busca: o gerador primeiro escolhe termos (chamada de ferramenta única, via saída
estruturada), depois responde com os trechos que os contêm (opcionalmente fundidos ao RAG).
O raciocínio (think) é uma segunda passada, só nos modelos que o suportam.

Também grava results/retrieval.jsonl com a revocação de evidência de cada recuperador
para vários k (calculada sem chamar os geradores).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .data import document_text
from .io import read_jsonl, write_jsonl
from .embeddings import Embedder
from .ollama import OllamaClient
from .prompts import LETTERS, generation_messages, is_abstention, search_messages
from .retrieval import DocumentIndex, chunk_document, evidence_recall, format_chunks

log = logging.getLogger(__name__)
METHODS = ("bm25", "dense", "hybrid")


def build_indices(cfg, embed: Embedder, docs: list[dict]) -> dict[str, DocumentIndex]:
    ch, rt = cfg["chunking"], cfg["retrieval"]
    chunked = {d["doc_id"]: chunk_document(d, ch["target_words"], ch["max_words"]) for d in docs}
    # um único lote para todos os trechos de todos os documentos
    flat = [(doc_id, c) for doc_id, cs in chunked.items() for c in cs]
    emb = embed([f"{c.section}\n{c.text}".strip() for _, c in flat])
    indices, start = {}, 0
    for doc_id, cs in chunked.items():
        indices[doc_id] = DocumentIndex(cs, emb[start : start + len(cs)], rt["bm25_k1"], rt["bm25_b"], rt["rrf_k0"])
        start += len(cs)
    log.info("indexados %d documentos, %d trechos", len(indices), len(flat))
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

    embed = Embedder(cfg, cfg.path("cache"))
    indices = build_indices(cfg, embed, list(docs.values()))
    emb = embed([q["question"] for q in questions])
    q_emb = {q["qid"]: e for q, e in zip(questions, emb)}
    write_jsonl(results / "retrieval.jsonl", retrieval_report(cfg, indices, questions, q_emb))

    # contexto de cada (pergunta, condição) é independente do gerador, exceto nas condições de busca
    contexts: dict[tuple[str, str], dict] = {}
    for q in questions:
        for cond in cfg["conditions"]:
            kind = cond["context"]
            if kind == "full":
                text, rec = document_text(docs[q["doc_id"]]), None
            elif kind == "rag":
                got = indices[q["doc_id"]].retrieve(cond["retriever"], q["question"], cond["k"], q_emb[q["qid"]])
                text, rec = format_chunks(got), evidence_recall(got, q.get("evidence_paragraphs", []))
            else:  # none; nas condições de busca o contexto é montado em `job`
                text, rec = None, None
            contexts[q["qid"], cond["name"]] = {
                "kind": kind, "text": text, "evidence_recall": rec,
                "context_words": len(text.split()) if text else 0,
            }

    gen_cfg, search_cfg = cfg["generation"], cfg["search"]

    def view_text(q, view):
        if view == "full":
            return document_text(docs[q["doc_id"]])
        if view == "rag":  # o que o gerador lê antes de escolher os termos: os trechos do RAG híbrido
            got = indices[q["doc_id"]].retrieve("hybrid", q["question"], search_cfg["view_k"], q_emb[q["qid"]])
            return format_chunks(got)
        return None

    def search_context(gen, think, q, cond):
        """Passo 1: termos do gerador. Passo 2: busca (Ctrl+F) e, se pedido, fusão RRF com o RAG."""
        messages, schema = search_messages(q, cond["view"], view_text(q, cond["view"]), search_cfg["max_terms"])
        res = client.chat_json(gen["model"], messages, schema, think=think,
                               num_predict=gen_cfg["think_num_predict"] if think else search_cfg["num_predict"])
        raw = (res.get("output") or {}).get("terms") or []
        terms = list(dict.fromkeys(t.strip() for t in raw if isinstance(t, str) and t.strip()))[: search_cfg["max_terms"]]
        idx = indices[q["doc_id"]]
        got, n_matched = idx.search_retrieve(terms, cond["k"], cond.get("fuse"), q["question"], q_emb[q["qid"]])
        found = format_chunks(got) if got else None
        if cond["context"] == "search":
            kind, text, extra = "rag", found or "No passage matched the search terms.", None
        else:  # full_search: documento inteiro seguido dos trechos encontrados
            kind, text, extra = "full_search", document_text(docs[q["doc_id"]]), found
        words = len(text.split()) + (len(extra.split()) if extra else 0)
        meta = {
            "search_terms": terms, "search_matched": n_matched, "search_error": res.get("error"),
            "search_prompt_tokens": res.get("prompt_tokens"), "search_completion_tokens": res.get("completion_tokens"),
            "search_latency_s": res.get("latency_s"),
        }
        return kind, text, extra, evidence_recall(got, q.get("evidence_paragraphs", [])), words, meta

    def job(gen, think, q, cond):
        ctx = contexts[q["qid"], cond["name"]]
        kind, text, extra, rec, words, meta = ctx["kind"], ctx["text"], None, ctx["evidence_recall"], ctx["context_words"], {}
        if kind in ("search", "full_search"):
            kind, text, extra, rec, words, meta = search_context(gen, think, q, cond)
        messages, schema = generation_messages(q, kind, text, extra)
        res = client.chat_json(gen["model"], messages, schema, think=think,
                               num_predict=gen_cfg["think_num_predict"] if think else gen_cfg["num_predict"])
        out = res.get("output") or {}
        answer = (out.get("answer") or "").strip()
        row = {
            "qid": q["qid"], "dataset": q["dataset"], "model": gen["model"], "condition": cond["name"],
            "think": think, "answer": answer, "abstained": is_abstention(answer),
            "context_words": words, "evidence_recall": rec,
            **{k: res.get(k) for k in ("prompt_tokens", "completion_tokens", "thinking_words", "latency_s",
                                       "truncated", "done_reason", "error")},
            **meta,
        }
        if q["dataset"] == "quality":
            choice = out.get("choice")
            row["choice"] = choice
            row["correct"] = choice is not None and LETTERS.index(choice) == q["gold_index"]
        return row

    # passada sem raciocínio para todos; depois, com raciocínio, nos modelos que o suportam
    # (restritos a `think_models`, se definido)
    runs = [(g, False) for g in cfg["generators"]]
    if gen_cfg.get("think_pass"):
        allowed = gen_cfg.get("think_models")
        runs += [(g, True) for g in cfg["generators"] if "thinking" in client.capabilities(g["model"])
                 and (allowed is None or g["model"] in allowed)]
    rows = []
    # um modelo por vez: evita disputar memória do servidor carregando vários modelos
    for gen, think in runs:
        tasks = [(gen, think, q, c) for q in questions for c in cfg["conditions"]]
        with ThreadPoolExecutor(cfg["ollama"]["concurrency"]) as ex:
            futures = [ex.submit(job, *t) for t in tasks]
            desc = gen["label"] + (" +think" if think else "")
            for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
                rows.append(f.result())
        # checkpoint por (modelo, raciocínio) em arquivo à parte: leitores de generations.jsonl
        # (análise intermediária em paralelo) nunca veem uma geração pela metade
        write_jsonl(results / "generations.partial.jsonl", rows)
    write_jsonl(results / "generations.jsonl", rows)
    (results / "generations.partial.jsonl").unlink(missing_ok=True)
    n_err = sum(1 for r in rows if r.get("error"))
    log.info("geração concluída: %d respostas, %d falhas (formato ou teto de tokens)", len(rows), n_err)
