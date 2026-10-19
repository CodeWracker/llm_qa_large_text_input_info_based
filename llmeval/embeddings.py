"""Embeddings de texto (recuperação densa e métrica de cosseno), com cache persistente.

backend "local": sentence-transformers na GPU local (padrão; mesmos pesos do bge-m3 do Ollama).
backend "ollama": endpoint /api/embed. Num servidor compartilhado que mantém um modelo por vez,
cada troca entre o embedder e os LLMs força uma recarga; por isso o padrão é local.
"""
from __future__ import annotations

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ollama import OllamaClient, SQLiteCache

_LOCK = threading.Lock()


@lru_cache(maxsize=None)
def _local_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")


class Embedder:
    def __init__(self, cfg, cache_dir: Path):
        rt = cfg["retrieval"]
        self.backend = rt.get("embedding_backend", "local")
        self.model = rt["embedding_model"]
        self.cache = SQLiteCache(cache_dir / f"embeddings_{self.backend}.sqlite")
        self._client = OllamaClient(cfg, cache_dir) if self.backend == "ollama" else None

    def _key(self, text: str) -> str:
        return hashlib.sha256(json.dumps([self.backend, self.model, text]).encode()).hexdigest()

    def __call__(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        out: list = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            todo = [texts[i] for i in missing]
            if self.backend == "local":
                with _LOCK:
                    vecs = _local_model(self.model).encode(
                        todo, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
                    )
            else:
                vecs = self._client.embed(self.model, todo)
            for i, v in zip(missing, vecs):
                out[i] = [float(x) for x in v]
                self.cache.set(keys[i], out[i])
        mat = np.asarray(out, dtype=np.float32)
        return mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
