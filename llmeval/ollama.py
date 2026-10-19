"""Cliente Ollama com cache persistente, saída estruturada (JSON Schema) e retentativas.

Toda chamada é identificada por um hash de (modelo, mensagens, schema, opções). O resultado
fica em SQLite, então reexecutar qualquer estágio é idempotente e não gasta o servidor.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import socket
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from requests.utils import get_netrc_auth
from tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential

log = logging.getLogger(__name__)


class TransientError(RuntimeError):
    pass


class SQLiteCache:
    """Cache chave→JSON seguro para múltiplas threads."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()


def _error_detail(text: str) -> str:
    """Resumo legível de uma resposta de erro: o <title> de páginas HTML ou o início do texto."""
    m = re.search(r"<title>(.*?)</title>", text or "", re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()[:200]


class IncidentLog:
    """Registro de falhas do servidor (uma linha JSON por tentativa falha), para relatar à SeTIC.

    O formato é compartilhado com outros experimentos: ver o README da pasta de incidentes.
    Só host, rota, modelo e erro são gravados; nunca credenciais nem o conteúdo das mensagens.
    """

    def __init__(self, path: str | None, experiment: str, url: str):
        self.path = Path(path) if path else None
        if self.path is not None and not self.path.parent.is_dir():
            self.path = None  # pasta ausente (outra máquina): não registra
        self.experiment, self.host = experiment, urlparse(url).netloc
        self.client = socket.gethostname()
        self._lock = threading.Lock()
        self._failing = False

    def _write(self, **event) -> None:
        if self.path is None:
            return
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "experimento": self.experiment,
               "servico": self.host, "cliente": self.client, "fonte": "automatico", **event}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def failure(self, endpoint: str, model: str | None, tipo: str, status: int | None, detalhe: str, dur: float):
        self._failing = True
        self._write(evento="falha", rota=endpoint, modelo=model, tipo=tipo, status=status,
                    detalhe=detalhe, duracao_s=round(dur, 2))

    def success(self, endpoint: str, model: str | None) -> None:
        if self._failing:  # primeira resposta boa após falhas: marca o fim do episódio
            self._failing = False
            self._write(evento="recuperado", rota=endpoint, modelo=model)


class GlobalSlots:
    """Limite de requisições simultâneas ao servidor somando TODOS os processos do experimento.

    São `n` arquivos de trava (flock); cada requisição segura um enquanto está no servidor.
    O servidor é compartilhado com outros projetos, então o limite vale para a soma dos estágios.
    """

    def __init__(self, lock_dir: Path, n: int):
        self.dir, self.n = Path(lock_dir), n
        self.dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def hold(self):
        while True:
            for i in range(self.n):
                f = open(self.dir / f"slot_{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
                return
            time.sleep(0.25)


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class OllamaClient:
    def __init__(self, cfg, cache_dir: Path):
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        user, pwd = os.environ.get("OLLAMA_USER"), os.environ.get("OLLAMA_PASSWORD")
        self.auth = (user, pwd) if user else get_netrc_auth(self.url)
        self.opts = cfg["ollama"]
        self.session = requests.Session()
        self.chat_cache = SQLiteCache(cache_dir / "chat.sqlite")
        self.slots = GlobalSlots(cache_dir / "slots", self.opts.get("max_inflight", 8))
        self.incidents = IncidentLog(os.environ.get("OLLAMA_INCIDENT_LOG", self.opts.get("incident_log")),
                                     self.opts.get("experiment", "llm-eval"), self.url)
        self.emb_cache = SQLiteCache(cache_dir / "embeddings.sqlite")

    # ------------------------------------------------------------------ HTTP
    @retry(
        retry=retry_if_exception_type(TransientError),
        wait=wait_exponential(multiplier=2, max=120),
        stop=stop_after_delay(1800),  # aguenta quedas do servidor compartilhado de até ~30 min
        reraise=True,
    )
    def _post(self, endpoint: str, payload: dict) -> dict:
        model, t0 = payload.get("model"), time.time()
        try:
            with self.slots.hold():
                t0 = time.time()  # o tempo na fila local não conta como latência do servidor
                r = self.session.post(
                    f"{self.url}{endpoint}", json=payload, auth=self.auth, timeout=self.opts["timeout_s"]
                )
        except requests.RequestException as exc:  # rede, timeout
            tipo = "timeout" if isinstance(exc, requests.Timeout) else "conexao"
            self.incidents.failure(endpoint, model, tipo, None, f"{type(exc).__name__}: {str(exc)[:160]}", time.time() - t0)
            raise TransientError(str(exc)) from exc
        if r.status_code >= 400:
            tipo = "http_5xx" if r.status_code >= 500 else ("http_429" if r.status_code == 429 else "http_4xx")
            self.incidents.failure(endpoint, model, tipo, r.status_code,
                                  _error_detail(r.content.decode("utf-8", "replace")), time.time() - t0)
        if r.status_code >= 500 or r.status_code == 429:
            raise TransientError(f"{r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        self.incidents.success(endpoint, model)
        return r.json()

    @lru_cache(maxsize=None)
    def capabilities(self, model: str) -> tuple[str, ...]:
        return tuple(self._post("/api/show", {"model": model}).get("capabilities", []))

    # ------------------------------------------------------------------ chat
    def chat_json(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        think=False,
        num_predict: int = 512,
    ) -> dict:
        """Chamada com saída restrita a `schema`. Retorna o JSON parseado + metadados.

        `think`: False/True ou nível ("low"/"medium"/"high"); só é enviado se o modelo suporta.
        """
        options = {
            "temperature": self.opts["temperature"],
            "seed": self.opts["seed"],
            "num_ctx": self.opts["num_ctx"],
            "num_predict": num_predict,
        }
        payload = {
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "keep_alive": self.opts["keep_alive"],
            "options": options,
        }
        if "thinking" in self.capabilities(model):
            payload["think"] = think
        key = _hash(payload)
        cached = self.chat_cache.get(key)
        if cached is not None:
            return cached

        last_err = None
        for attempt in range(self.opts["max_parse_retries"]):
            if attempt:  # nova semente para escapar de uma saída malformada determinística
                payload["options"] = {**options, "seed": options["seed"] + attempt}
            t0 = time.time()
            try:
                resp = self._post("/api/chat", payload)
            except TransientError as exc:
                # servidor fora do ar além da janela de retentativas: a chamada falha sozinha, sem
                # derrubar o estágio e sem ir para o cache (uma nova execução a refaz)
                log.error("servidor indisponível para %s: %s", model, str(exc)[:120])
                return {"output": None, "error": f"servidor: {str(exc)[:120]}", "parse_attempts": attempt}
            msg = resp.get("message", {})
            content = msg.get("content", "")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                last_err = f"{exc}: {content[:200]!r}"
                log.warning("JSON inválido de %s (tentativa %d): %s", model, attempt + 1, last_err)
                if resp.get("done_reason") == "length":
                    # o teto de tokens (em geral consumido pelo raciocínio) se repetiria com outra semente
                    # é um desfecho determinístico (temperatura 0): vai para o cache como falha
                    failed = {"output": None, "error": f"num_predict={num_predict} esgotado",
                              "done_reason": "length", "prompt_tokens": resp.get("prompt_eval_count"),
                              "completion_tokens": resp.get("eval_count"),
                              "latency_s": round(time.time() - t0, 3), "parse_attempts": attempt + 1}
                    self.chat_cache.set(key, failed)
                    return failed
                continue
            result = {
                "output": parsed,
                "prompt_tokens": resp.get("prompt_eval_count"),
                "completion_tokens": resp.get("eval_count"),  # inclui os tokens de raciocínio
                "thinking_words": len((msg.get("thinking") or "").split()),
                "done_reason": resp.get("done_reason"),
                "latency_s": round(time.time() - t0, 3),
                "truncated": (resp.get("prompt_eval_count") or 0) >= options["num_ctx"] - 8,
                "parse_attempts": attempt + 1,
            }
            self.chat_cache.set(key, result)
            return result
        # falhas de formato não vão para o cache: uma nova execução tenta de novo
        return {"output": None, "error": last_err, "parse_attempts": self.opts["max_parse_retries"]}

    # ------------------------------------------------------------- embeddings
    def embed(self, model: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embeddings L2-normalizados, com cache por texto."""
        keys = [_hash([model, t]) for t in texts]
        out: list = [self.emb_cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        for s in range(0, len(missing), batch_size):
            idx = missing[s : s + batch_size]
            resp = self._post("/api/embed", {"model": model, "input": [texts[i] for i in idx]})
            for i, vec in zip(idx, resp["embeddings"]):
                out[i] = vec
                self.emb_cache.set(keys[i], vec)
        mat = np.asarray(out, dtype=np.float32)
        return mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
