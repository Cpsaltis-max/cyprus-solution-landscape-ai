"""Upload the private PDF corpus to an OpenAI File Search vector store.

Each PDF is converted to page-marked UTF-8 text before upload so retrieved
passages can retain the original PDF page number. PDFs, extracted text, API
keys, and the resulting store manifest are never committed to GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import time

from openai import OpenAI
from pypdf import PdfReader


def safe_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_.")
    return f"{stem or 'document'}.txt"


def extract_page_marked_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = [f"=== SOURCE PDF: {pdf_path.name} ==="]
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        parts.append(f"\n=== PDF PAGE {page_number} ===\n{text}")
    return "\n".join(parts)


def wait_until_ready(client: OpenAI, vector_store_id: str, file_id: str) -> str:
    deadline = time.time() + 1800
    while time.time() < deadline:
        item = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=file_id,
        )
        status = str(item.status)
        if status == "completed":
            return status
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Indexing failed for {file_id}: {item}")
        time.sleep(2)
    raise TimeoutError(f"Timed out while indexing {file_id}")


def build(corpus: Path, vector_store_id: str, store_name: str) -> None:
    pdfs = sorted(corpus.rglob("*.pdf")) + sorted(corpus.rglob("*.PDF"))
    pdfs = sorted(set(pdfs))
    if not pdfs:
        raise SystemExit(f"No PDFs found below {corpus}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not vector_store_id:
        vector_store_id = client.vector_stores.create(name=store_name).id
        print(f"Created vector store: {vector_store_id}")
    else:
        print(f"Using vector store: {vector_store_id}")

    uploaded = []
    with tempfile.TemporaryDirectory(prefix="cyprus-openai-index-") as temp_dir:
        temp_root = Path(temp_dir)
        for number, pdf_path in enumerate(pdfs, start=1):
            upload_path = temp_root / f"{number:03d}_{safe_name(pdf_path)}"
            upload_path.write_text(extract_page_marked_text(pdf_path), encoding="utf-8")
            with upload_path.open("rb") as handle:
                file_object = client.files.create(file=handle, purpose="user_data")
            client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_object.id,
            )
            wait_until_ready(client, vector_store_id, file_object.id)
            uploaded.append({"pdf": str(pdf_path), "file_id": file_object.id})
            print(f"Indexed {number}/{len(pdfs)}: {pdf_path.name}")

    manifest = {
        "vector_store_id": vector_store_id,
        "documents": uploaded,
        "document_count": len(uploaded),
    }
    Path("openai_index_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nIndex complete.")
    print(f'Add this Streamlit secret:\nOPENAI_VECTOR_STORE_ID = "{vector_store_id}"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("private_corpus"))
    parser.add_argument("--vector-store-id", default="")
    parser.add_argument("--name", default="Cyprus Solution Landscape — 45 papers and book")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.corpus, args.vector_store_id, args.name)

