"""OpenAI File Search backend for the Cyprus Solution Landscape dashboard."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-5-mini"

# Responses API citations can be rendered as private-use marker sequences such
# as "\ue200filecite\ue202turn1file6\ue201".  Browsers without the matching renderer show
# those characters as boxes.  The app replaces them with its own SOURCE labels.
_FILE_CITATION_MARKER = re.compile(
    r"[\ue000-\uf8ff\ufffd\u25a1]*filecite[\ue000-\uf8ff\ufffd\u25a1]*"
    r"turn\d+file\d+[\ue000-\uf8ff\ufffd\u25a1]*",
    flags=re.I,
)


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


def _file_citation_annotations(response: Any) -> list[dict[str, Any]]:
    """Return file-citation annotations in the order emitted by the model."""
    annotations: list[dict[str, Any]] = []
    for item in _as_dict(response).get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if isinstance(annotation, dict) and annotation.get("type") == "file_citation":
                    annotations.append(annotation)
    return annotations


def clean_answer_citations(
    answer: str,
    response: Any,
    sources: list[dict[str, Any]],
) -> str:
    """Replace OpenAI's internal citation markers with visible SOURCE labels."""
    labels: list[str] = []
    for annotation in _file_citation_annotations(response):
        file_id = str(annotation.get("file_id") or "")
        filename = str(annotation.get("filename") or "")
        source = next(
            (
                item
                for item in sources
                if (file_id and item.get("file_id") == file_id)
                or (filename and item.get("title") == filename)
            ),
            None,
        )
        labels.append(
            f"[SOURCE {source['source_number']}]" if source else "[indexed source]"
        )

    label_index = 0

    def replace_marker(_: re.Match[str]) -> str:
        nonlocal label_index
        label = labels[label_index] if label_index < len(labels) else "[indexed source]"
        label_index += 1
        return label

    cleaned = _FILE_CITATION_MARKER.sub(replace_marker, answer)
    # Remove any unmatched citation-renderer control glyphs, but retain ordinary
    # non-ASCII text used in Greek, Turkish, names, and titles.
    cleaned = re.sub(r"[\ue000-\uf8ff]", "", cleaned).replace("\ufffd", "")
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _authoritative_dashboard_finding(data_context: str) -> str:
    """Extract the precomputed maximum line supplied by the dashboard."""
    match = re.search(
        r"AUTHORITATIVE PRECOMPUTED MAXIMUM JOINT ACCEPTANCE:\s*\n\s*([^\n]+)",
        data_context,
        flags=re.I,
    )
    return match.group(1).strip() if match else ""


def _ensure_authoritative_finding(answer: str, data_context: str, mode: str) -> str:
    """Guarantee that a requested precomputed dashboard maximum is not omitted."""
    if mode not in {"data_only", "combined"}:
        return answer
    finding = _authoritative_dashboard_finding(data_context)
    if not finding:
        return answer

    year = re.search(r"\b(?:19|20)\d{2}\b", finding)
    joint = re.search(r"joint=min\(GC,\s*TC\)=([0-9.]+%)", finding, flags=re.I)
    has_required_value = bool(
        year
        and joint
        and year.group(0) in answer
        and joint.group(1) in answer
    )
    if has_required_value:
        return answer
    return f"**Authoritative dashboard finding:** {finding}\n\n{answer}"


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
TC acceptance.

The supplied dashboard tables are the sole authority for dashboard percentages, rankings, years,
and solution comparisons. If an AUTHORITATIVE PRECOMPUTED result is supplied, reproduce it exactly
at the beginning of the answer. Never replace a requested dashboard value with a number found in a
paper or book. Do not infer a rising or falling trend unless the supplied dashboard rows establish it.

For combined answers, use two explicit sections: "Dashboard finding" and "Interpretation from the
indexed scholarship". The second section may explain the finding but must not alter it. Do not say a
solution is the only viable or acceptable solution unless a retrieved passage explicitly supports
that wording. Distinguish normative argument, theoretical interpretation, and empirical evidence.

For scholarly claims, name the retrieved source and PDF page when available. Do not print internal
platform markers or the word "filecite"; cite sources in plain prose. Separate general theory from
Cyprus-specific evidence. Answer in the question's language."""
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
    raw_answer = str(getattr(response, "output_text", "") or "").strip()
    if not raw_answer:
        raise OpenAIBackendError("OpenAI returned an empty answer.")
    sources = extract_sources(response)
    answer = clean_answer_citations(raw_answer, response, sources)
    answer = _ensure_authoritative_finding(answer, data_context, mode)
    usage = _as_dict(getattr(response, "usage", {}))
    return {"answer": answer, "model": model_name, "sources": sources,
            "performance": {"total_seconds": elapsed, "prompt_tokens": usage.get("input_tokens"),
                            "output_tokens": usage.get("output_tokens")}}
