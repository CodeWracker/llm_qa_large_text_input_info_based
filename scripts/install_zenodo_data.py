"""Installs the files of the Zenodo data record into this repository.

Checks the record against its MANIFEST.sha256, copies the outputs to results/, data/sample/ and human_eval/, and,
with --with-caches, decompresses the caches into data/cache/ (needed only to re-run the pipeline offline).

Usage: python scripts/install_zenodo_data.py PATH_TO_RECORD [--with-caches]
PATH_TO_RECORD is the downloaded llm-eval-data-<version>.zip, or the folder extracted from it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# folder in the record -> folder in the repository
TARGETS = {"results": "results", "data/sample": "data/sample", "human_eval": "human_eval"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=Path)
    ap.add_argument("--with-caches", action="store_true", help="also install data/cache/*.sqlite (offline re-run)")
    args = ap.parse_args()
    rec = args.record.resolve()
    if rec.suffix == ".zip":  # the archive downloaded from Zenodo: extract it next to itself
        with zipfile.ZipFile(rec) as z:
            z.extractall(rec.parent)
            rec = rec.parent / z.namelist()[0].split("/")[0]

    bad = []
    for line in (rec / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        p = rec / name
        if name.startswith("cache/") and not args.with_caches:
            continue
        if not p.exists() or sha256(p) != digest:
            bad.append(name)
    if bad:
        sys.exit("files missing or corrupted: " + ", ".join(bad))

    n = 0
    for src_dir, dst_dir in TARGETS.items():
        for p in (rec / src_dir).rglob("*"):
            if p.is_file():
                rel = p.relative_to(rec / src_dir)
                # human_eval/annotations_*.csv lives in human_eval/results/ in the repository
                if src_dir == "human_eval" and p.name.startswith("annotations_"):
                    rel = Path("results") / rel
                dest = ROOT / dst_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                n += 1
    print(f"{n} files installed in results/, data/sample/ and human_eval/")

    if args.with_caches:
        cache = ROOT / "data" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        for gz in sorted((rec / "cache").glob("*.sqlite.gz")):
            dest = cache / gz.name.removesuffix(".gz")
            with gzip.open(gz, "rb") as fi, open(dest, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            print(f"cache installed: {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
