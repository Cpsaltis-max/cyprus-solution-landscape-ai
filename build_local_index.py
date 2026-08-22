"""Build a private local semantic index from the papers and book PDFs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3

import numpy as np
from pypdf import PdfReader

from local_ai import DEFAULT_EMBED_MODEL, _embed


def clean_title(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[_-]+", " ", stem).strip()


def inferred_year(filename: str) -> str:
    match = re.search(r"(?:19|20)\d{2}", filename)
    return match.group(0) if match else ""


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").replace("\\", "/").lower()
            if filename:
                records[filename] = {key: (value or "").strip() for key, value in row.items()}
    return records


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalised = re.sub(r"\s+", " ", text).strip()
    if not normalised:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalised):
        target_end = min(len(normalised), start + chunk_size)
        end = target_end
        if target_end < len(normalised):
            sentence_break = normalised.rfind(". ", start + chunk_size // 2, target_end)
            word_break = normalised.rfind(" ", start + chunk_size // 2, target_end)
            if sentence_break > start:
                end = sentence_break + 1
            elif word_break > start:
                end = word_break
        chunk = normalised[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalised):
            break
        start = max(start + 1, end - overlap)
    return chunks


def document_metadata(
    pdf_path: Path,
    corpus_root: Path,
    manifest: dict[str, dict[str, str]],
) -> dict[str, str]:
    relative = pdf_path.relative_to(corpus_root).as_posix()
    record = manifest.get(relative.lower()) or manifest.get(pdf_path.name.lower()) or {}
    parent_names = {part.lower() for part in pdf_path.relative_to(corpus_root).parts[:-1]}
    inferred_type = "book" if "book" in parent_names else "paper"
    return {
        "source_path": relative,
        "title": record.get("title") or clean_title(pdf_path.name),
        "year": record.get("year") or inferred_year(pdf_path.name),
        "document_type": record.get("document_type") or inferred_type,
    }


def initialise_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            source_path TEXT NOT NULL,
            title TEXT NOT NULL,
            year TEXT,
            document_type TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_number INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );

        CREATE INDEX idx_chunks_source ON chunks(source_path);
        """
    )


def build_index(
    corpus_root: Path,
    index_path: Path,
    manifest_path: Path,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> None:
    pdf_paths = sorted(corpus_root.rglob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found under '{corpus_root}'.")

    manifest = load_manifest(manifest_path)
    passages: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for document_number, pdf_path in enumerate(pdf_paths, start=1):
        metadata = document_metadata(pdf_path, corpus_root, manifest)
        print(f"Extracting {document_number}/{len(pdf_paths)}: {metadata['title']}")
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            warnings.append({"file": metadata["source_path"], "error": str(exc)})
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                warnings.append(
                    {
                        "file": metadata["source_path"],
                        "page": page_number,
                        "error": str(exc),
                    }
                )
                continue

            if len(page_text.strip()) < 40:
                warnings.append(
                    {
                        "file": metadata["source_path"],
                        "page": page_number,
                        "warning": "Very little extractable text; OCR may be required.",
                    }
                )

            for chunk_number, text in enumerate(
                split_text(page_text, chunk_size, overlap), start=1
            ):
                passages.append(
                    {
                        **metadata,
                        "page_number": page_number,
                        "chunk_number": chunk_number,
                        "text": text,
                    }
                )

    if not passages:
        raise SystemExit("No extractable text was found in the PDF corpus.")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(index_path.suffix + ".building")
    if temporary_path.exists():
        temporary_path.unlink()

    connection = sqlite3.connect(temporary_path)
    initialise_database(connection)

    try:
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            embeddings = _embed([str(item["text"]) for item in batch], embedding_model)

            records = []
            for item, embedding in zip(batch, embeddings):
                vector = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if norm:
                    vector /= norm
                records.append(
                    (
                        item["source_path"],
                        item["title"],
                        item["year"],
                        item["document_type"],
                        item["page_number"],
                        item["chunk_number"],
                        item["text"],
                        vector.tobytes(),
                    )
                )

            connection.executemany(
                """
                INSERT INTO chunks (
                    source_path, title, year, document_type, page_number,
                    chunk_number, text, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            connection.commit()
            completed = min(start + len(batch), len(passages))
            print(f"Embedded {completed}/{len(passages)} passages")

        metadata_rows = {
            "embedding_model": embedding_model,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "document_count": str(len(pdf_paths)),
            "passage_count": str(len(passages)),
            "chunk_size": str(chunk_size),
            "overlap": str(overlap),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata_rows.items(),
        )
        connection.commit()
    except Exception:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    else:
        connection.close()

    os.replace(temporary_path, index_path)
    report_path = index_path.parent / "index_report.json"
    report_path.write_text(
        json.dumps(
            {
                "documents_found": len(pdf_paths),
                "passages_indexed": len(passages),
                "embedding_model": embedding_model,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nLocal index created: {index_path}")
    print(f"Extraction report: {report_path}")
    print(f"Documents: {len(pdf_paths)} | passages: {len(passages)} | warnings: {len(warnings)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("private_corpus"))
    parser.add_argument("--index", type=Path, default=Path("local_index/index.sqlite3"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("private_corpus/corpus_manifest.csv"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )
    parser.add_argument("--chunk-size", type=int, default=2200)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        corpus_root=args.corpus,
        index_path=args.index,
        manifest_path=args.manifest,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
