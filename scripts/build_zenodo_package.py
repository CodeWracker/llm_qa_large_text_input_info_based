"""Builds the Zenodo data package of the study: model outputs, verdicts, metrics, analyses, the sample, the human
validation and the caches needed to re-run the pipeline offline.

The package goes to dist/zenodo/llm-eval-data-<version>/ with a README (data card), LICENSE, NOTICE (third-party
terms), the exact configuration and a MANIFEST.sha256. The SQLite caches are gzip-compressed. Upload every file of
that folder to one Zenodo record.

Usage: python scripts/build_zenodo_package.py [--version v1.0] [--no-embeddings]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (published path, source path, description)
FILES = [
    ("results/generations.jsonl", "results/generations.jsonl",
     "Every answer of the main design (6 generators x 11 conditions x 400 questions) and of the reasoning pass "
     "at a budget of 4096 tokens (2 generators x 5 conditions x 400)."),
    *[(f"results/generations_b{b}.jsonl", f"results/generations_b{b}.jsonl",
       f"Reasoning pass under a generation budget of {b} tokens (budget sweep, RQ6).")
      for b in (512, 1024, 2048, 4096, 8192)],
    ("results/judgments.jsonl", "results/judgments.jsonl",
     "Grading of each answer of generations.jsonl: abstention rule or the three judges' verdicts and justifications."),
    *[(f"results/judgments_b{b}.jsonl", f"results/judgments_b{b}.jsonl",
       f"Grading of generations_b{b}.jsonl.") for b in (512, 1024, 2048, 4096, 8192)],
    ("results/metrics.jsonl", "results/metrics.jsonl",
     "Automatic metrics per answer (exact match, token F1, ROUGE-L, fuzzy, embedding cosine, BERTScore)."),
    ("results/retrieval.jsonl", "results/retrieval.jsonl",
     "Evidence recall of BM25, dense and hybrid retrieval for k in {1, 2, 3, 5, 10, 20} (QASPER)."),
    ("results/agreement.jsonl", "results/agreement.jsonl",
     "Human reference point: each QASPER annotation, and each model answer, graded against the other annotations "
     "with one annotator left out in turn."),
    ("results/agreement_pairs.jsonl", "results/agreement_pairs.jsonl",
     "Pairwise comparisons (answer vs. one annotator) used by the consensus analysis."),
    ("data/sample/questions.jsonl", "data/sample/questions.jsonl",
     "The 400 sampled questions with references, answer types and evidence paragraphs."),
    ("data/sample/documents.jsonl", "data/sample/documents.jsonl",
     "The 272 sampled documents (QASPER papers and QuALITY texts), split into paragraphs."),
    ("human_eval/qasper_annotator.html", "human_eval/qasper_annotator.html",
     "Self-contained annotation page used for the human validation (blind: no model, no jury verdict)."),
    ("human_eval/sample_key.json", "human_eval/sample_key.json",
     "Key of the 120 answers of the human validation: question, references, answer, model, condition, votes."),
    ("human_eval/annotations_annotator_1.csv", "human_eval/results/annotations_annotator_1.csv",
     "Verdicts of the human annotator (anonymised as Annotator 1)."),
    ("human_eval/agreement_report.json", "human_eval/agreement_report.json",
     "Agreement between the jury, each judge and the human annotator."),
    ("configs/experiment.yaml", "configs/experiment.yaml",
     "The exact configuration: pinned dataset revisions, model tags and digests, conditions, budgets, seeds."),
]
CACHES = [
    ("cache/chat.sqlite.gz", "data/cache/chat.sqlite",
     "Every model call (generators and judges) keyed by a hash of the request, with the raw response. With it the "
     "whole pipeline re-runs offline (LLMEVAL_OFFLINE=1) and reproduces the published outputs exactly."),
    ("cache/embeddings_local.sqlite.gz", "data/cache/embeddings_local.sqlite",
     "BGE-M3 embeddings of chunks, questions and answers. Needed only to reproduce dense/hybrid retrieval "
     "bit-for-bit; otherwise they are recomputed on a GPU."),
]
EXCLUDED = """- `data/raw/`: the original QASPER and QuALITY files. They are downloaded from a pinned Hugging Face
  commit and checked against their SHA-256 (`configs/experiment.yaml`, `sources`).
- `embeddings.sqlite` (an early Ollama embedding cache, unused), logs and intermediate files."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> str:
    if path.suffix in (".jsonl", ".csv"):
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        return str(n - 1 if path.suffix == ".csv" else n)
    if path.name.endswith(".sqlite"):
        return str(sqlite3.connect(path).execute("SELECT COUNT(*) FROM kv").fetchone()[0])
    return ""


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--no-embeddings", action="store_true", help="leave the 0.5 GB embedding cache out")
    args = ap.parse_args()
    out = ROOT / "dist" / "zenodo" / f"llm-eval-data-{args.version}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    table = []
    for dest, src, desc in FILES:
        s = ROOT / src
        if not s.exists():
            sys.exit(f"missing: {src}")
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, out / dest)
        table.append((dest, human(s.stat().st_size), rows(s), desc))
    for name in sorted((ROOT / "results" / "analysis").glob("*.csv")):
        dest = f"results/analysis/{name.name}"
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, out / dest)
    n_csv = len(list((out / "results" / "analysis").glob("*.csv")))
    table.append(("results/analysis/*.csv", human(sum(p.stat().st_size for p in (out / "results/analysis").glob("*"))),
                  f"{n_csv} files", "Every number of the paper's tables and figures (written by `llmeval analyze`)."))
    for dest, src, desc in CACHES:
        if args.no_embeddings and "embeddings" in src:
            continue
        s = ROOT / src
        (out / dest).parent.mkdir(parents=True, exist_ok=True)
        with open(s, "rb") as fi, gzip.open(out / dest, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        table.append((dest, f"{human((out / dest).stat().st_size)} (gzip; {human(s.stat().st_size)} raw)",
                      f"{rows(s)} entries", desc))

    template = (ROOT / "scripts" / "zenodo_README.md").read_text(encoding="utf-8")
    lines = ["| File | Size | Rows | Content |", "|---|---|---|---|"]
    lines += [f"| `{d}` | {sz} | {r} | {desc} |" for d, sz, r, desc in table]
    readme = (template.replace("{{VERSION}}", args.version).replace("{{FILES}}", "\n".join(lines))
              .replace("{{EXCLUDED}}", EXCLUDED))
    (out / "README.md").write_text(readme, encoding="utf-8")
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / "scripts" / f"zenodo_{name}.txt", out / name)

    manifest = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            manifest.append(f"{sha256(p)}  {p.relative_to(out)}")
    (out / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"package": str(out), "files": len(manifest) + 1, "size": human(total)}, indent=1))


if __name__ == "__main__":
    main()
