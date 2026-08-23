# Local AI experiment (Windows, 64 GB laptop)

This branch adds an optional local Ollama backend to the existing Streamlit AI
dashboard. The Gemini backend remains available and unchanged.

## Privacy

The PDFs and generated semantic index remain on the laptop. The folders
`private_corpus/` and `local_index/` are excluded from Git. Do not commit the
papers, the book, the index, or `.streamlit/secrets.toml`.

When **Local AI** is selected, questions, dashboard data, and retrieved passages
are sent only to Ollama at `http://127.0.0.1:11434`. They are not sent to Gemini.

## 1. Install the application

Open PowerShell in the repository folder. The simplest setup is:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_local_ai.ps1
```

The script creates a private Python environment, installs the requirements,
creates the private folders, and downloads both Ollama models. Alternatively,
the equivalent manual commands are:

```powershell
git switch local-agent-experiment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

Install Ollama for Windows from <https://ollama.com/download/windows>, then run:

```powershell
ollama pull qwen3:8b
ollama pull bge-m3
```

`qwen3:8b` is the initial answer model. `bge-m3` creates multilingual semantic
embeddings for English, Greek, and Turkish retrieval.

## 2. Add the private corpus

Create these folders without committing their contents:

```text
private_corpus/
├── papers/    # the 45 published-paper PDFs
└── book/      # the Conflict and Change PDF
```

PDF filenames are used as titles by default. For accurate titles and years,
copy `corpus_manifest.example.csv` to `private_corpus/corpus_manifest.csv` and
add one row per document. `filename` is relative to `private_corpus/`.

## 3. Build the local index

Make sure Ollama is running, then execute:

```powershell
python build_local_index.py
```

The command searches all subfolders, extracts page text, creates embeddings,
and writes `local_index/index.sqlite3`. Each page is first read with pypdf. If
missing-font boxes, replacement characters, or unusually little text are
detected, the page is independently re-read with PyMuPDF and the demonstrably
better extraction is used. The generated `local_index/index_report.json`
separately lists pages repaired by the fallback and pages that still require
OCR. Rebuilding is safe: the existing index is replaced only after the new
index has been completed successfully.

## 4. Run the dashboard

```powershell
streamlit run app.py
```

Or use the included launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_local_ai.ps1
```

In **Ask the data and theory**, select **Local AI** and choose one of the three
existing answer modes. Gemini continues to require the existing Streamlit
secrets; Local AI does not require an API key.

## Optional model settings

The defaults can be changed for one PowerShell session:

```powershell
$env:OLLAMA_CHAT_MODEL = "qwen3:14b"
$env:OLLAMA_EMBED_MODEL = "bge-m3"
$env:OLLAMA_NUM_CTX = "8192"
$env:LOCAL_INDEX_PATH = "local_index/index.sqlite3"
streamlit run app.py
```

If the embedding model changes, rebuild the index before asking theory-based
questions.
