"""Local Ollama backend for the Cyprus Solution Landscape AI dashboard.

This module has no cloud dependencies. It queries a local Ollama server and
retrieves relevant passages from the SQLite index created by
``build_local_index.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
from urllib import error, request

import numpy as np


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_INDEX_PATH = Path("local_index/index.sqlite3")


class LocalAIError(RuntimeError):
    """A user-facing error raised by the local AI backend."""


@dataclass(frozen=True)
class RetrievedChunk:
    title: str
    year: str
    document_type: str
    page_number: int
    source_path: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        year = f" ({self.year})" if self.year else ""
        return f"{self.title}{year}, p. {self.page_number}"


def _ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def _post_ollama(endpoint: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        f"{_ollama_url()}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", str(exc))
        except Exception:
            detail = str(exc)
        raise LocalAIError(f"Ollama rejected the request: {detail}") from exc
    except error.URLError as exc:
        raise LocalAIError(
            "Ollama could not be reached. Start Ollama and confirm that "
            f"{_ollama_url()} is available."
        ) from exc
    except json.JSONDecodeError as exc:
        raise LocalAIError("Ollama returned an unreadable response.") from exc


def _embed(texts: list[str], model: str) -> list[list[float]]:
    result = _post_ollama(
        "/api/embed",
        {"model": model, "input": texts},
    )
    embeddings = result.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise LocalAIError(
            f"No usable embeddings were returned by Ollama model '{model}'."
        )
    return embeddings


def _load_index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    except sqlite3.Error as exc:
        raise LocalAIError(
            "The local index is incomplete. Re-run build_local_index.py."
        ) from exc


def retrieve_chunks(
    question: str,
    *,
    index_path: Path | str | None = None,
    top_k: int = 8,
    maximum_per_document: int = 2,
) -> list[RetrievedChunk]:
    """Retrieve semantically relevant passages with basic source diversity."""
    path = Path(index_path or os.getenv("LOCAL_INDEX_PATH", str(DEFAULT_INDEX_PATH)))
    if not path.exists():
        raise LocalAIError(
            f"Local index not found at '{path}'. Run build_local_index.py first."
        )

    try:
        connection = sqlite3.connect(path)
        metadata = _load_index_metadata(connection)
        rows = connection.execute(
            """
            SELECT title, year, document_type, page_number, source_path, text, embedding
            FROM chunks
            ORDER BY id
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise LocalAIError(f"Could not read the local index: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    if not rows:
        raise LocalAIError("The local index contains no searchable passages.")

    embed_model = metadata.get("embedding_model", DEFAULT_EMBED_MODEL)
    query_vector = np.asarray(_embed([question], embed_model)[0], dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0:
        raise LocalAIError("The embedding model returned an empty query vector.")
    query_vector /= query_norm

    vectors = np.vstack([np.frombuffer(row[6], dtype=np.float32) for row in rows])
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vector_norms[vector_norms == 0] = 1.0
    scores = (vectors / vector_norms) @ query_vector

    candidate_count = min(len(rows), max(top_k * 8, top_k))
    candidate_indices = np.argsort(scores)[-candidate_count:][::-1]

    selected: list[RetrievedChunk] = []
    per_document: dict[str, int] = {}
    for index in candidate_indices:
        row = rows[int(index)]
        source_path = str(row[4])
        if per_document.get(source_path, 0) >= maximum_per_document:
            continue
        selected.append(
            RetrievedChunk(
                title=str(row[0]),
                year=str(row[1] or ""),
                document_type=str(row[2]),
                page_number=int(row[3]),
                source_path=source_path,
                text=str(row[5]),
                score=float(scores[int(index)]),
            )
        )
        per_document[source_path] = per_document.get(source_path, 0) + 1
        if len(selected) >= top_k:
            break

    return selected


def _format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for number, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[SOURCE {number}: {chunk.citation}; type={chunk.document_type}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(sections)


def ask_local(
    *,
    question: str,
    mode: str,
    data_context: str = "",
    index_path: Path | str | None = None,
) -> dict[str, Any]:
    """Answer with the local model using data, local retrieval, or both."""
    if mode not in {"data_only", "theory_only", "combined"}:
        raise ValueError(f"Unsupported local answer mode: {mode}")

    chunks: list[RetrievedChunk] = []
    if mode in {"theory_only", "combined"}:
        chunks = retrieve_chunks(question, index_path=index_path)

    if mode == "data_only":
        source_rule = (
            "Use only the supplied dashboard dataset. Do not use the papers, book, "
            "or outside knowledge."
        )
    elif mode == "theory_only":
        source_rule = (
            "Use only the retrieved passages from the published-paper corpus and "
            "Conflict and Change. Do not use outside knowledge."
        )
    else:
        source_rule = (
            "Use both the supplied dashboard dataset and the retrieved passages. "
            "Clearly separate empirical findings from theoretical interpretation. "
            "Do not use outside knowledge."
        )

    retrieved_context = _format_retrieved_context(chunks) or "No paper or book passages supplied."
    dashboard_context = data_context or "No dashboard dataset supplied in this mode."

    system_prompt = """
You are the local AI interpretation assistant for the Cyprus Solution Landscape Dashboard.
Use only the sources supplied in the user's message. Do not invent facts, years, percentages,
sample sizes, causal claims, titles, or citations. If the available sources do not support an
answer, say so clearly. Distinguish 'in favor' from 'accepted'; accepted = in_favor + tolerate.
When using dashboard data, state the relevant year, community, solution, and percentage.
When using retrieved scholarship, cite its SOURCE number and the displayed title/page.
Answer in the same language as the question when possible. Give the answer, not private
chain-of-thought reasoning.
""".strip()

    user_prompt = f"""
SOURCE RULE FOR THIS REQUEST:
{source_rule}

DASHBOARD DATA:
{dashboard_context}

RETRIEVED PAPERS AND BOOK PASSAGES:
{retrieved_context}

QUESTION:
{question}
""".strip()

    chat_model = os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    try:
        context_window = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
    except ValueError:
        context_window = 32768

    result = _post_ollama(
        "/api/chat",
        {
            "model": chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": context_window},
        },
        timeout=600,
    )

    answer = (result.get("message") or {}).get("content", "").strip()
    if not answer:
        raise LocalAIError(f"Ollama model '{chat_model}' returned an empty answer.")

    return {
        "answer": answer,
        "model": chat_model,
        "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        "sources": [
            {
                "citation": chunk.citation,
                "title": chunk.title,
                "year": chunk.year,
                "page": chunk.page_number,
                "document_type": chunk.document_type,
                "score": chunk.score,
            }
            for chunk in chunks
        ],
    }
