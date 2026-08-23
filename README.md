📘 Cyprus Solution Landscape Dashboard (AI Version)

An interactive, trilingual (English–Greek–Turkish) visualisation platform presenting longitudinal bicommunal survey data (2010–2025) on public preferences regarding political solutions to the Cyprus issue.

This AI-enabled version integrates:

📊 Empirical data analysis (dashboard)
📚 Theoretical interpretation grounded in Cambridge University Book by Psaltis & Wagoner (2025).
Conflict and Change: Integrating Social and Developmental Psychology

🤖 OpenAI Responses API module for querying both data and theory

🔍 Features
Interactive visualisations of:

Full distributions (Against / Tolerate / In favor)
Accepted vs Rejected (referendum viability)
Cross-solution comparison
Joint (bicommual) acceptance

Trilingual interface:

English
Greek (Ελληνικά)
Turkish (Türkçe)

AI module:

Ask questions about the data

Receive grounded answers based on:

the dataset

the book (Conflict and Change)

Separate:
empirical findings
theoretical interpretation (GSP framework)

🧠 Conceptual Framework

The dashboard is informed by Genetic Social Psychology (GSP), integrating:

Microgenesis (interactional processes)
Ontogenesis (developmental trajectories)
Sociogenesis (historical and societal dynamics)

The AI module uses this framework to interpret patterns in public opinion and intergroup relations.

📊 Data

The dataset is based on:

Representative bicommunal surveys (2010–2025)
Greek Cypriot and Turkish Cypriot communities

Key variables:

Support for political solutions:

Bizonal Bicommunal Federation (BBF)
Unitary State
Two States
Status Quo

Response categories:

Against
Tolerate
In favor

Derived indicator:

Accepted = In favor + Tolerate

Joint acceptance:

Minimum accepted level across both communities
🤖 AI Module

The AI component uses the OpenAI Responses API with:

Dataset grounding (structured CSV input)

Retrieval-Augmented Generation (RAG)

Indexed book:
Conflict and Change

The model is instructed to:

Use only:

dashboard data

retrieved book passages

Avoid hallucination

Clearly distinguish:

empirical findings

theoretical interpretation

📖 Citation

If you use this dashboard, dataset, or outputs, please cite:

Psaltis, C. (2026).
Cyprus Solution Landscape Dashboard (2010–2025).
Genetic Social Psychology Lab, University of Cyprus.
https://cyprus-solution-landscape.streamlit.app

And for theoretical framework:

Psaltis, C., & Wagoner, B. (2025).
Conflict and Change: Integrating Social and Developmental Psychology.
Cambridge University Press.

⚖️ Copyright & Use

© 2026 Charis Psaltis
Genetic Social Psychology Lab – University of Cyprus

Permitted use:

Academic research
Teaching
Policy analysis

Conditions:

Proper citation required
No misrepresentation of data or findings

Not permitted without permission:
Commercial use

Redistribution of dataset as a standalone product

Reproduction of AI-generated interpretations without attribution

🔐 Data & AI Integrity

The dataset is read-only within the application

Users cannot modify underlying data

AI responses are constrained to:

the dataset
indexed book content

🧪 Technical Stack
Streamlit
Pandas
Plotly
OpenAI Responses API (RAG with File Search)

## OpenAI corpus setup

Keep the 45 papers and the searchable *Conflict and Change* PDF outside GitHub
under `private_corpus/papers/` and `private_corpus/book/`. Set the API key for
the current PowerShell session and build the OpenAI vector store once:

```powershell
$env:OPENAI_API_KEY = "your-key"
python build_openai_index.py
```

The indexer uploads page-marked text, not repository files, so retrieved
passages can retain their original PDF page number. When it finishes, add the
printed values to Streamlit secrets:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_VECTOR_STORE_ID = "vs_..."
OPENAI_MODEL = "gpt-5-mini"
```

Never commit the PDFs, `openai_index_manifest.json`, or
`.streamlit/secrets.toml`.

📬 Contact

For collaboration, data access, or academic inquiries:

👉 email me at: cpsaltis@ucy.ac.cy

🚀 Live App

👉 (https://cyprus-solution-landscape-ai-fqtppmt23fqxz93nq9dfre.streamlit.app/)

🎯 Notes

This platform is designed as both:

a research tool
and a public engagement interface

bridging data, theory, and AI-assisted interpretation of the Cyprus problem.
