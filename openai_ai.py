"""OpenAI File Search backend for the Cyprus Solution Landscape dashboard."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-5-mini"


class OpenAIBackendError(RuntimeError):
    """A user-facing error raised by the OpenAI backend."""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _text_from_result(result: dict[str, Any]) -> str:
    text = result.get("text")
    if isinstance(text, str):
        return text
    content = result.get("content") or []
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _page_from_text(text: str) -> int | None:
    matches = re.findall(r"===\s*PDF PAGE\s+(\d+)\s*===", text, flags=re.I)
    return int(matches[-1]) if matches else None


def extract_sources(response: Any) -> list[dict[str, Any]]:
    data = _as_dict(response)
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "file_search_call":
            for result in item.get("results", []) or []:
                if not isinstance(result, dict):
                    continue
                filename = str(result.get("filename") or result.get("file_name") or "Indexed source")
                file_id = str(result.get("file_id") or "")
                excerpt = _text_from_result(result).strip()
                page = _page_from_text(excerpt)
                key = (file_id or filename, page)
                if key in seen:
                    continue
                seen.add(key)
                citation = f"{filename}, PDF p. {page}" if page is not None else filename
                sources.append({
                    "citation": citation, "title": filename, "page": page,
                    "document_type": "book" if "conflict" in filename.casefold() else "paper",
                    "score": result.get("score"), "excerpt": excerpt[:900], "file_id": file_id,
                })
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if not isinstance(annotation, dict) or annotation.get("type") != "file_citation":
                    continue
                filename = str(annotation.get("filename") or "Indexed source")
                file_id = str(annotation.get("file_id") or "")
                if any((s.get("file_id") or s["title"]) == (file_id or filename) for s in sources):
                    continue
                sources.append({
                    "citation": filename, "title": filename, "page": None,
                    "document_type": "book" if "conflict" in filename.casefold() else "paper",
                    "score": None, "excerpt": "", "file_id": file_id,
                })
    for number, source in enumerate(sources, start=1):
        source["source_number"] = number
    return sources


def ask_openai(*, question: str, mode: str, data_context: str = "", api_key: str,
               vector_store_id: str = "", model: str | None = None) -> dict[str, Any]:
    if mode not in {"data_only", "theory_only", "combined"}:
        raise ValueError(f"Unsupported answer mode: {mode}")
    if mode != "data_only" and not vector_store_id:
        raise OpenAIBackendError("OPENAI_VECTOR_STORE_ID is required for paper/book questions.")
    if mode == "data_only":
        source_rule = "Use only the supplied dashboard tables. Do not use outside knowledge."
    elif mode == "theory_only":
        source_rule = "Use only passages retrieved from the indexed 45 papers and Conflict and Change. Do not use outside knowledge."
    else:
        source_rule = "Use dashboard tables for empirical findings and retrieved papers/book for interpretation. Separate them clearly."
    instructions = """You are the research assistant for the Cyprus Solution Landscape Dashboard.
Use only evidence allowed for the selected mode. Never invent studies, titles, years, percentages,
samples, causal claims, quotations, or citations. If evidence is insufficient, say so. Distinguish
'in favor' from 'accepted': accepted = in_favor + tolerate. Joint acceptance is the lower of GC and
TC acceptance. For scholarly claims, name the retrieved source and PDF page when available. Separate
general theory from Cyprus-specific evidence. Answer in the question's language."""
    prompt = f"SOURCE RULE:\n{source_rule}\n\nDASHBOARD DATA:\n{data_context or 'No dashboard data supplied.'}\n\nQUESTION:\n{question}"
    client = OpenAI(api_key=api_key)
    model_name = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    request: dict[str, Any] = {"model": model_name, "instructions": instructions, "input": prompt}
    if mode != "data_only":
        request["tools"] = [{"type": "file_search", "vector_store_ids": [vector_store_id], "max_num_results": 10}]
        request["include"] = ["file_search_call.results"]
    started = time.perf_counter()
    try:
        response = client.responses.create(**request)
    except Exception as exc:
        raise OpenAIBackendError(f"OpenAI request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    answer = str(getattr(response, "output_text", "") or "").strip()
    if not answer:
        raise OpenAIBackendError("OpenAI returned an empty answer.")
    usage = _as_dict(getattr(response, "usage", {}))
    return {"answer": answer, "model": model_name, "sources": extract_sources(response),
            "performance": {"total_seconds": elapsed, "prompt_tokens": usage.get("input_tokens"),
                            "output_tokens": usage.get("output_tokens")}}

