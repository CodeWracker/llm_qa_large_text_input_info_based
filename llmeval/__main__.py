"""CLI: python -m llmeval {sample,generate,judge,metrics,analyze,all} [--config ...]"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config

STAGES = ("sample", "generate", "judge", "metrics", "analyze")


def main() -> None:
    ap = argparse.ArgumentParser(prog="llmeval")
    ap.add_argument("stage", choices=STAGES + ("all",))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    from .ollama import ServerUnavailable

    try:
        run_stages(cfg, args.stage)
    except ServerUnavailable as exc:
        logging.getLogger("llmeval").error("servidor indisponível, estágio interrompido: %s", exc)
        sys.exit(75)  # EX_TEMPFAIL: tentar de novo mais tarde (o cache preserva o progresso)


def run_stages(cfg, stage_arg: str) -> None:
    for stage in STAGES if stage_arg == "all" else (stage_arg,):
        if stage == "sample":
            from .data import build_sample
            build_sample(cfg)
        elif stage == "generate":
            from .generate import run_generation
            run_generation(cfg)
        elif stage == "judge":
            from .judge import run_judging
            run_judging(cfg)
        elif stage == "metrics":
            from .metrics import run_metrics
            run_metrics(cfg)
        elif stage == "analyze":
            from .analysis import run_analysis
            run_analysis(cfg)


if __name__ == "__main__":
    main()
