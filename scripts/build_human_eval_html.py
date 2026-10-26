"""Amostra de validação humana do júri no QASPER (RQ1) e o anotador HTML autocontido.

Sorteia (seed 42) 120 respostas geradas sem raciocínio no QASPER: 40 extractive, 40 abstractive e 40 yes/no,
20 por gerador e metade com documento completo, metade com híbrido k=5. Só entram respostas julgadas pelo júri
(abstenções decididas por regra ficam de fora) com os três votos. Dentro de cada tipo, cada pergunta aparece
no máximo uma vez enquanto houver candidatas.

Saídas:
  human_eval/sample_key.json        gabarito oculto: IDs, metadados, resposta e votos dos três juízes
  human_eval/qasper_annotator.html  página única (CSS/JS embutidos) só com os dados cegos

Uso: python scripts/build_human_eval_html.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmeval.config import load_config  # noqa: E402
from llmeval.data import UNANSWERABLE  # noqa: E402
from llmeval.io import read_jsonl  # noqa: E402
from llmeval.prompts import UNANSWERABLE_REFERENCE  # noqa: E402

SEED = 42
TYPES = ["extractive", "abstractive", "yes_no"]
PER_TYPE = 40
CONDITIONS = ["full", "hybrid_k5"]
TYPE_LABEL = {"extractive": "Extractive", "abstractive": "Abstractive", "yes_no": "Yes/No"}
OUT = ROOT / "human_eval"
TEMPLATE = Path(__file__).with_name("human_eval_template.html")


def cell_sizes(n_models: int) -> dict[tuple[int, int], int]:
    """Quantas respostas de cada (tipo, modelo): 40 por tipo e 20 por modelo (7/7/7/7/6/6 em rodízio)."""
    base, extra = divmod(PER_TYPE, n_models)  # 6 e 4
    shift = n_models - extra  # cada tipo desloca a sobra, para que todo modelo some o mesmo total
    return {(t, m): base + ((m - shift * t) % n_models < extra)
            for t in range(len(TYPES)) for m in range(n_models)}


def main() -> None:
    cfg = load_config()
    results = cfg.path("results")
    questions = {q["qid"]: q for q in read_jsonl(cfg.path("sample") / "questions.jsonl")}
    answers = {(g["qid"], g["model"], g["condition"]): g["answer"]
               for g in read_jsonl(results / "generations.jsonl") if not g.get("think")}
    models = [g["model"] for g in cfg["generators"]]
    labels = {g["model"]: g["label"] for g in cfg["generators"]}
    judges = [j["model"] for j in cfg["jury"]["judges"]]

    # candidatas: QASPER, sem raciocínio, julgadas pelo júri com os três votos
    pool: dict[tuple[str, str, str], list[dict]] = {}
    for j in read_jsonl(results / "judgments.jsonl"):
        q = questions[j["qid"]]
        if (j["dataset"] != "qasper" or j.get("think") or j["condition"] not in CONDITIONS
                or j["source"] != "jury" or q["stratum"] not in TYPES):
            continue
        votes = j.get("votes") or {}
        if any(votes.get(m) is None for m in judges):
            continue
        pool.setdefault((q["stratum"], j["model"], j["condition"]), []).append(j)

    rng = np.random.default_rng(SEED)
    sizes = cell_sizes(len(models))
    assert all(sum(sizes[t, m] for m in range(len(models))) == PER_TYPE for t in range(len(TYPES)))
    assert len({sum(sizes[t, m] for t in range(len(TYPES))) for m in range(len(models))}) == 1

    picked, start_full = [], True
    for t, stratum in enumerate(TYPES):
        # metade das respostas de cada célula com o documento completo; nas células ímpares, alterna quem leva a sobra
        quotas = []
        for m in range(len(models)):
            n = sizes[t, m]
            n_full = n // 2 + (n % 2 if start_full else 0)
            if n % 2:
                start_full = not start_full
            quotas += [(models[m], "full")] * n_full + [(models[m], "hybrid_k5")] * (n - n_full)
        used: dict[str, int] = {}
        for i in rng.permutation(len(quotas)):
            model, cond = quotas[i]
            cands = pool[(stratum, model, cond)]
            taken = {(p["qid"], p["model"], p["condition"]) for p in picked}
            cands = [c for c in cands if (c["qid"], model, cond) not in taken]
            least = min(used.get(c["qid"], 0) for c in cands)
            cands = [c for c in cands if used.get(c["qid"], 0) == least]
            choice = cands[rng.integers(len(cands))]
            used[choice["qid"]] = used.get(choice["qid"], 0) + 1
            picked.append(choice)

    # ordem de exibição embaralhada: a posição não revela modelo, condição nem tipo
    order = rng.permutation(len(picked))
    key, blind = [], []
    for pos, i in enumerate(order, 1):
        j = picked[i]
        q = questions[j["qid"]]
        item_id = f"Q{pos:03d}"
        refs = [UNANSWERABLE_REFERENCE if r == UNANSWERABLE else r for r in q["references"]]
        answer = answers[(j["qid"], j["model"], j["condition"])]
        votes = {m: j["votes"][m] for m in judges}
        key.append({"item_id": item_id, "qid": j["qid"], "doc_id": q["doc_id"], "stratum": q["stratum"],
                    "model": j["model"], "model_label": labels[j["model"]], "condition": j["condition"],
                    "question": q["question"], "references": refs, "answer": answer, "votes": votes,
                    "jury_correct": bool(j["correct"]), "jury_score": j["score"]})
        blind.append({"id": item_id, "type": TYPE_LABEL[q["stratum"]], "question": q["question"],
                      "references": refs, "candidate": answer})

    OUT.mkdir(exist_ok=True)
    (OUT / "results").mkdir(exist_ok=True)
    payload = json.dumps(blind, ensure_ascii=False)
    sample_id = hashlib.sha256(payload.encode()).hexdigest()[:12]
    meta = {"seed": SEED, "sample_id": sample_id, "n": len(key), "judges": judges,
            "per_type": {s: sum(k["stratum"] == s for k in key) for s in TYPES},
            "per_model": {labels[m]: sum(k["model"] == m for k in key) for m in models},
            "per_condition": {c: sum(k["condition"] == c for k in key) for c in CONDITIONS},
            "distinct_questions": len({k["qid"] for k in key})}
    with open(OUT / "sample_key.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": key}, f, ensure_ascii=False, indent=1)

    # "</" não pode aparecer cru dentro de <script>
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__SAMPLE_ID__", sample_id).replace("__ITEMS__", payload.replace("</", "<\\/"))
    (OUT / "qasper_annotator.html").write_text(html, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
