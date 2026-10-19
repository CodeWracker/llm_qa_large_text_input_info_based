"""Carregamento da configuração do experimento (configs/experiment.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"


class Config(dict):
    """Dicionário com acesso por atributo e caminhos resolvidos a partir da raiz do projeto."""

    def __getattr__(self, key):
        value = self[key]
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        p = ROOT / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def condition(self, name: str) -> dict:
        return next(c for c in self["conditions"] if c["name"] == name)


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
