"""Concordância entre os anotadores humanos e com o júri na amostra do QASPER (RQ1).

Lê todos os human_eval/results/annotations_*.csv (exportados por human_eval/qasper_annotator.html), cruza com
human_eval/sample_key.json e calcula:
  - entre os anotadores: concordância exata (3 níveis), concordância e kappa de Cohen binários (CORRECT vs. resto)
    e alfa de Krippendorff ordinal (CORRECT=1, PARTIAL=0.5, INCORRECT=0);
  - júri (maioria estrita) e cada juiz contra cada anotador e contra o consenso (itens em que os dois
    anotadores concordam no rótulo binário): acurácia e kappa de Cohen binários.

Saídas: human_eval/agreement_report.json e human_eval/agreement_report.tex (macros \\hum{chave} usadas no artigo,
copiadas também para ../latex-llmeval/tables/human_agreement.tex).

Uso: python scripts/evaluate_human_results.py [--results PASTA] [--out PASTA] [--no-paper]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmeval.config import load_config  # noqa: E402
from llmeval.stats import krippendorff_alpha  # noqa: E402

HE = ROOT / "human_eval"
ANNOTATORS = ["Rodrigo Ferraz", "Rodrigo Guedes"]  # ordem usada no relatório
VERDICT = {"correct": "correct", "partial": "partial", "partially_correct": "partial", "incorrect": "incorrect"}
ORDINAL = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c == "." else "-" for c in text.lower()).strip("-")


def read_annotations(folder: Path, key_ids: set[str]) -> dict[str, dict[str, str]]:
    """{anotador: {item_id: veredito}}; com linhas repetidas, vale a de timestamp mais recente."""
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    files = sorted(folder.glob("annotations_*.csv"))
    if not files:
        sys.exit(f"nenhum annotations_*.csv em {folder}")
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                item, who = row["item_id"].strip(), row["annotator"].strip()
                verdict = VERDICT.get(row["verdict"].strip().lower())
                if item not in key_ids or verdict is None:
                    print(f"aviso: {path.name}: linha ignorada {row}", file=sys.stderr)
                    continue
                ts = row.get("timestamp", "")
                if (who, item) not in latest or latest[(who, item)][1] <= ts:
                    latest[(who, item)] = (verdict, ts)
    out: dict[str, dict[str, str]] = {}
    for (who, item), (verdict, _) in latest.items():
        out.setdefault(who, {})[item] = verdict
    return out


def agreement(a: list[str], b: list[str]) -> dict:
    """Acurácia e kappa de Cohen no rótulo binário (correct vs. resto)."""
    ya = np.array([v == "correct" for v in a], int)
    yb = np.array([v == "correct" for v in b], int)
    return {"n": len(ya), "acc": float((ya == yb).mean()), "kappa": float(cohen_kappa_score(ya, yb))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HE / "results", help="pasta com os annotations_*.csv")
    ap.add_argument("--out", type=Path, default=HE, help="pasta do agreement_report.{json,tex}")
    ap.add_argument("--no-paper", action="store_true", help="não copiar as macros para o artigo")
    args = ap.parse_args()
    cfg = load_config()
    key = {it["item_id"]: it for it in json.load(open(HE / "sample_key.json", encoding="utf-8"))["items"]}
    ann = read_annotations(args.results, set(key))
    names = [n for n in ANNOTATORS if n in ann] + sorted(set(ann) - set(ANNOTATORS))
    judges = {j["model"]: j["label"] for j in cfg["jury"]["judges"]}
    # rótulos do júri por item: maioria estrita e voto de cada juiz (partially_correct conta como não correto)
    machine = {"LLM jury (majority)": {i: "correct" if it["jury_correct"] else "incorrect" for i, it in key.items()}}
    for model, label in judges.items():
        machine[f"Judge: {label}"] = {i: VERDICT[it["votes"][model]] for i, it in key.items()}

    report = {"n_sample": len(key), "annotators": {n: {"n": len(ann[n]), "verdicts": {
        v: sum(x == v for x in ann[n].values()) for v in ORDINAL}} for n in names}}

    # entre anotadores
    pairs = []
    for a, b in itertools.combinations(names, 2):
        common = sorted(set(ann[a]) & set(ann[b]))
        va, vb = [ann[a][i] for i in common], [ann[b][i] for i in common]
        r = agreement(va, vb)
        r["exact"] = float(np.mean([x == y for x, y in zip(va, vb)]))
        r["alpha_ordinal"] = float(krippendorff_alpha(np.array([[ORDINAL[x], ORDINAL[y]] for x, y in zip(va, vb)]),
                                                      "ordinal"))
        pairs.append({"a": a, "b": b, **r})
    report["inter_annotator"] = pairs

    # referência humana: consenso binário dos dois primeiros anotadores (ou o único anotador)
    if len(names) >= 2:
        a, b = names[:2]
        both = set(ann[a]) & set(ann[b])
        consensus = {i: ann[a][i] for i in both if (ann[a][i] == "correct") == (ann[b][i] == "correct")}
        consensus_name = f"consensus ({a} = {b})"
    else:
        consensus, consensus_name = dict(ann[names[0]]), names[0]
    refs = {consensus_name: consensus, **{n: ann[n] for n in names}}

    vs = []
    for ref_name, ref in refs.items():
        ids = sorted(ref)
        for inst, lab in machine.items():
            r = agreement([lab[i] for i in ids], [ref[i] for i in ids])
            vs.append({"reference": ref_name, "instrument": inst, **r})
        # por tipo de pergunta, só para o júri
        for stratum in ("extractive", "abstractive", "yes_no"):
            sid = [i for i in ids if key[i]["stratum"] == stratum]
            if sid:
                r = agreement([machine["LLM jury (majority)"][i] for i in sid], [ref[i] for i in sid])
                vs.append({"reference": ref_name, "instrument": f"LLM jury (majority) [{stratum}]", **r})
    report["consensus_reference"] = consensus_name
    report["machine_vs_human"] = vs
    report["human_accept_rate"] = {n: float(np.mean([v == "correct" for v in ann[n].values()])) for n in names}
    report["jury_accept_rate"] = float(np.mean([it["jury_correct"] for i, it in key.items() if i in consensus]))

    with open(args.out / "agreement_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    # macros para o artigo: \hum{chave}; chaves ausentes aparecem como \pending
    macros = {}
    if pairs:
        p = pairs[0]
        macros |= {"n": f"{p['n']}", "inter-exact": f"{100 * p['exact']:.1f}", "inter-acc": f"{100 * p['acc']:.1f}",
                   "inter-kappa": f"{p['kappa']:.3f}", "inter-alpha": f"{p['alpha_ordinal']:.3f}"}
    macros["cons-n"] = f"{len(consensus)}"
    for r in vs:
        if r["reference"] != consensus_name or "[" in r["instrument"]:
            continue
        k = "jury" if r["instrument"].startswith("LLM jury") else "judge-" + slug(r["instrument"].removeprefix("Judge: "))
        macros[f"{k}-acc"], macros[f"{k}-kappa"] = f"{100 * r['acc']:.1f}", f"{r['kappa']:.3f}"
    lines = ["% Gerado por scripts/evaluate_human_results.py (validação humana do júri no QASPER). Não editar à mão.",
             f"% Anotadores: {', '.join(f'{n} ({len(ann[n])} itens)' for n in names)}; referência do júri: {consensus_name}."]
    lines += [rf"\expandafter\def\csname hum:{k}\endcsname{{{v}}}" for k, v in macros.items()]
    tex = "\n".join(lines) + "\n"
    (args.out / "agreement_report.tex").write_text(tex, encoding="utf-8")
    paper = cfg.path("paper") / "tables"
    if paper.is_dir() and not args.no_paper:
        shutil.copy(args.out / "agreement_report.tex", paper / "human_agreement.tex")

    # resumo no terminal
    print(f"Amostra: {len(key)} itens. Anotadores: " + ", ".join(f"{n} = {len(ann[n])}" for n in names))
    for p in pairs:
        print(f"\n{p['a']} x {p['b']} (n={p['n']}): exata {100 * p['exact']:.1f}% | binária {100 * p['acc']:.1f}% "
              f"| kappa {p['kappa']:.3f} | alfa ordinal {p['alpha_ordinal']:.3f}")
    for ref_name in refs:
        print(f"\nReferência: {ref_name}")
        for r in vs:
            if r["reference"] == ref_name:
                print(f"  {r['instrument']:<38} n={r['n']:>3}  acc {100 * r['acc']:5.1f}%  kappa {r['kappa']:6.3f}")
    print(f"\nRelatório: {args.out / 'agreement_report.json'}; macros: {args.out / 'agreement_report.tex'}")


if __name__ == "__main__":
    main()
