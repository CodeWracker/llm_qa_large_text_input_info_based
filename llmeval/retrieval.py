"""Segmentação em trechos (chunks) e recuperação: BM25, densa (bge-m3) e híbrida (RRF).

Notação (seguida no artigo):
  d = (p_1..p_n) parágrafos;  c_j trechos formados por parágrafos consecutivos;
  BM25(q, c)   = sum_t IDF(t) * f(t,c)(k1+1) / (f(t,c) + k1(1 - b + b|c|/avgdl))
  dense(q, c)  = cos(E(q), E(c))
  RRF(c)       = sum_r 1 / (k0 + rank_r(c))
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
STOPWORDS = frozenset(
    "a an the of and or in on at to for from by with as is are was were be been being this that "
    "these those it its into than then there their they we our you your he she his her them which "
    "who whom what when where why how do does did not no so such can could would should may might "
    "will shall also about over under between".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    idx: int
    section: str
    text: str
    paragraphs: list[int] = field(default_factory=list)  # índices dos parágrafos de origem
    start_word: int = 0  # posição (em palavras) do início do trecho no documento


def chunk_document(doc: dict, target_words: int = 200, max_words: int = 300) -> list[Chunk]:
    """Agrupa parágrafos consecutivos da mesma seção até ~target_words palavras.

    Parágrafos maiores que max_words são quebrados em grupos de sentenças. Cada trecho
    guarda os parágrafos de origem, o que permite medir a revocação das evidências.
    """
    units: list[tuple[str, str, int]] = []  # (seção, texto, parágrafo de origem)
    for i, p in enumerate(doc["paragraphs"]):
        words = p["text"].split()
        if len(words) <= max_words:
            units.append((p["section"], p["text"], i))
            continue
        buf: list[str] = []
        for sent in _SENT.split(p["text"]):
            if buf and len(" ".join(buf).split()) + len(sent.split()) > target_words:
                units.append((p["section"], " ".join(buf), i))
                buf = []
            buf.append(sent)
        if buf:
            units.append((p["section"], " ".join(buf), i))

    chunks: list[Chunk] = []
    cur: Chunk | None = None
    cur_words = offset = 0
    for section, text, pidx in units:
        n = len(text.split())
        if cur is None or section != cur.section or cur_words + n > target_words:
            if cur is not None:
                chunks.append(cur)
            cur = Chunk(len(chunks), section, text, [pidx], offset)
            cur_words = n
        else:
            cur.text += "\n" + text
            if pidx not in cur.paragraphs:
                cur.paragraphs.append(pidx)
            cur_words += n
        offset += n
    if cur is not None:
        chunks.append(cur)
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.len.mean() if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        # IDF de Robertson–Spärck Jones com suavização (sempre positivo)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avgdl, 1e-9))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


def rrf(rankings: list[np.ndarray], k0: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion. `rankings`: arrays de índices ordenados do melhor ao pior."""
    n = len(rankings[0])
    fused = np.zeros(n)
    for ranking in rankings:
        fused[ranking] += 1.0 / (k0 + np.arange(1, n + 1))
    return fused


class DocumentIndex:
    """Índice de um único documento: o espaço de busca é o próprio documento da pergunta."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, k1: float, b: float, k0: int):
        self.chunks = chunks
        self.emb = embeddings
        self.bm25 = BM25([tokenize(f"{c.section} {c.text}") for c in chunks], k1, b)
        self.k0 = k0

    def rank(self, method: str, query: str, query_emb: np.ndarray | None = None) -> np.ndarray:
        """Índices dos trechos do mais ao menos relevante (desempate estável pela ordem no texto)."""
        if method == "bm25":
            s = self.bm25.scores(tokenize(query))
        elif method == "dense":
            s = self.emb @ query_emb
        elif method == "hybrid":
            s = rrf([self.rank("bm25", query), self.rank("dense", query, query_emb)], self.k0)
        else:
            raise ValueError(method)
        return np.argsort(-s, kind="stable")

    def retrieve(self, method: str, query: str, k: int, query_emb=None) -> list[Chunk]:
        top = self.rank(method, query, query_emb)[:k]
        # trechos reapresentados na ordem do documento: preserva a narrativa para o leitor-modelo
        return [self.chunks[i] for i in sorted(top)]


def evidence_recall(retrieved: list[Chunk], evidence_sets: list[list[int]]) -> float | None:
    """Revocação de evidência: máximo, entre anotadores, da fração de parágrafos-evidência
    cobertos pelos trechos recuperados. None se a pergunta não tem evidência textual mapeável."""
    if not evidence_sets:
        return None
    covered = {p for c in retrieved for p in c.paragraphs}
    return max(len(covered & set(ev)) / len(ev) for ev in evidence_sets)


def format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        head = f"[Excerpt {i}" + (f" | Section: {c.section}]" if c.section else "]")
        parts.append(f"{head}\n{c.text}")
    return "\n\n".join(parts)
