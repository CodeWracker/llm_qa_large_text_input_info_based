"""Testa a premissa da varredura de orçamento (RQ6): com temperatura 0, a resposta não depende do teto.

Refaz no servidor, com teto 8192 e sem projeção pelo cache, N respostas com raciocínio que terminaram dentro
de 4096 tokens (sem documento e com o documento completo, que não dependem da recuperação), e compara com as
de 4096: mesma resposta e mesmo número de tokens gerados. Grava results/logs/budget_determinism.jsonl.

Uso: python scripts/check_budget_determinism.py [N]   (padrão 40, metade por modelo)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmeval.config import load_config  # noqa: E402
from llmeval.data import document_text  # noqa: E402
from llmeval.io import read_jsonl  # noqa: E402
from llmeval.ollama import OllamaClient  # noqa: E402
from llmeval.prompts import generation_messages  # noqa: E402

KIND = {"closed_book": "none", "full": "full"}


def main(n: int) -> None:
    cfg = load_config()
    base, test = cfg["generation"]["think_num_predict"], 8192
    docs = {d["doc_id"]: d for d in read_jsonl(cfg.path("sample") / "documents.jsonl")}
    questions = {q["qid"]: q for q in read_jsonl(cfg.path("sample") / "questions.jsonl")}
    done = [g for g in read_jsonl(cfg.path("results") / "generations.jsonl")
            if g.get("think") and g["condition"] in KIND and g.get("done_reason") != "length" and not g.get("error")]
    rng = np.random.default_rng(42)
    picked = []
    for model in cfg["generation"]["think_models"]:
        pool = [g for g in done if g["model"] == model]
        picked += [pool[i] for i in rng.choice(len(pool), n // 2, replace=False)]

    client = OllamaClient(cfg, cfg.path("cache"))
    client.budgets = []  # sem projeção: a chamada de 8192 vai de fato ao servidor
    rows = []
    for g in picked:
        q = questions[g["qid"]]
        text = document_text(docs[q["doc_id"]]) if KIND[g["condition"]] == "full" else None
        messages, schema = generation_messages(q, KIND[g["condition"]], text)
        a = client.chat_json(g["model"], messages, schema, think=True, num_predict=base)
        b = client.chat_json(g["model"], messages, schema, think=True, num_predict=test)
        rows.append({"qid": g["qid"], "model": g["model"], "condition": g["condition"],
                     "tokens_base": a.get("completion_tokens"), "tokens_test": b.get("completion_tokens"),
                     "same_output": a.get("output") == b.get("output"),
                     "same_tokens": a.get("completion_tokens") == b.get("completion_tokens")})
        print(f"{len(rows):>3} {g['model']:<11} {g['condition']:<11} tokens {rows[-1]['tokens_base']} -> "
              f"{rows[-1]['tokens_test']}  resposta igual: {rows[-1]['same_output']}", flush=True)
    out = cfg.path("results") / "logs" / "budget_determinism.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    same = sum(r["same_output"] for r in rows)
    print(f"\nrespostas idênticas: {same}/{len(rows)}; tokens idênticos: {sum(r['same_tokens'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
