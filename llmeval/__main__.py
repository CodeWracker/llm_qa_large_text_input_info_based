"""CLI: python -m llmeval {sample,generate,judge,metrics,analyze,all} [--config ...]"""
from __future__ import annotations

import argparse
import logging

from .config import load_config

STAGES = ("sample", "generate", "judge", "metrics", "analyze")


def main() -> None:
    ap = argparse.ArgumentParser(prog="llmeval")
    ap.add_argument("stage", choices=STAGES + ("all",))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    for stage in STAGES if args.stage == "all" else (args.stage,):
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
