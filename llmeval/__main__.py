"""CLI: python -m llmeval {sample,generate,judge,metrics,analyze,all,agreement} [--config ...] [--no-think] [--budget B] [--dry-run]"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config

STAGES = ("sample", "generate", "judge", "metrics", "analyze")


def main() -> None:
    ap = argparse.ArgumentParser(prog="llmeval")
    ap.add_argument("stage", choices=STAGES + ("all", "agreement", "agreement_pairs"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-think", action="store_true", help="geração sem a passada com raciocínio (RQ6)")
    ap.add_argument("--budget", type=int, default=None,
                    help="generate/judge: só a passada com raciocínio, com este teto (varredura de orçamento, RQ6)")
    ap.add_argument("--dry-run", action="store_true", help="generate/agreement: só conta as chamadas que iriam ao servidor")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.no_think:
        cfg["generation"]["think_pass"] = False
    cfg["generation"]["budget_run"] = args.budget
    cfg["generation"]["dry_run"] = args.dry_run
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
        elif stage == "agreement":  # concordância com humanos no QASPER (fora de "all")
            from .agreement import run_agreement
            run_agreement(cfg)
        elif stage == "agreement_pairs":  # comparações par a par na amostra (fora de "all")
            from .agreement import run_agreement_pairs
            run_agreement_pairs(cfg)
        elif stage == "analyze":
            from .analysis import run_analysis
            run_analysis(cfg)


if __name__ == "__main__":
    main()
