"""Local benchmark and human evaluation helpers for the research assistant."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

import pandas as pd
import streamlit as st


BENCHMARK_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "B01",
        "domain": "Intergroup contact",
        "recommended_mode": "Papers + book only",
        "question": (
            "According to the indexed papers and Conflict and Change, how does "
            "intergroup contact between Greek Cypriots and Turkish Cypriots "
            "contribute to prejudice reduction, and what conditions limit its "
            "effects? Cite the sources and PDF page numbers."
        ),
    },
    {
        "id": "B02",
        "domain": "Trust and threat",
        "recommended_mode": "Papers + book only",
        "question": (
            "What roles do intergroup trust, realistic threat, and symbolic "
            "threat play in relations between Greek Cypriots and Turkish "
            "Cypriots? Distinguish direct evidence from theoretical "
            "interpretation and cite sources with PDF pages."
        ),
    },
    {
        "id": "B03",
        "domain": "Direct and indirect contact",
        "recommended_mode": "Papers + book only",
        "question": (
            "Compare the effects of direct contact, cross-group friendship, "
            "extended contact, and online intergroup friendship on prejudice "
            "and willingness for renewed cohabitation. Cite the retrieved "
            "sources and PDF page numbers."
        ),
    },
    {
        "id": "B04",
        "domain": "Collective memory",
        "recommended_mode": "Papers + book only",
        "question": (
            "How do collective memory and representations of the past shape "
            "intergroup relations and prospects for reconciliation in Cyprus? "
            "Identify the mechanisms supported by the indexed scholarship and "
            "cite sources with PDF pages."
        ),
    },
    {
        "id": "B05",
        "domain": "Social representations",
        "recommended_mode": "Papers + book only",
        "question": (
            "How does the Genetic Social Psychology approach use social "
            "representations to explain stability and change in understandings "
            "of the Cyprus conflict? Cite the relevant papers or book passages "
            "and their PDF pages."
        ),
    },
    {
        "id": "B06",
        "domain": "History teaching",
        "recommended_mode": "Papers + book only",
        "question": (
            "What does the indexed scholarship suggest about history teaching, "
            "history educators, and the treatment of difficult pasts in the two "
            "Cypriot communities? Separate empirical findings from normative "
            "recommendations and cite PDF pages."
        ),
    },
    {
        "id": "B07",
        "domain": "Transitional justice",
        "recommended_mode": "Papers + book only",
        "question": (
            "How are transitional justice, acknowledgment of suffering, and "
            "acceptance of renewed cohabitation related in the Cyprus context? "
            "State what is directly supported and cite sources with PDF pages."
        ),
    },
    {
        "id": "B08",
        "domain": "Identity and recognition",
        "recommended_mode": "Papers + book only",
        "question": (
            "How do identity, recognition, and representations of the other "
            "community affect support for reconciliation or political "
            "settlement in Cyprus? Cite the strongest retrieved evidence and "
            "its PDF pages."
        ),
    },
    {
        "id": "B09",
        "domain": "Genetic Social Psychology",
        "recommended_mode": "Papers + book only",
        "question": (
            "Explain how microgenesis, ontogenesis, and sociogenesis are "
            "integrated in Genetic Social Psychology, using examples supported "
            "by the indexed papers and Conflict and Change. Cite PDF pages and "
            "avoid claims not present in the retrieved passages."
        ),
    },
    {
        "id": "B10",
        "domain": "Data–theory integration",
        "recommended_mode": "Data + papers + book interpretation",
        "question": (
            "Which political solution has the highest joint acceptance in "
            "2025, what are the Greek Cypriot and Turkish Cypriot acceptance "
            "percentages, and how might the result be interpreted using the "
            "indexed scholarship? Clearly separate the dashboard finding from "
            "theoretical interpretation and do not make a causal claim."
        ),
    },
]


def set_evaluation_candidate(
    *,
    benchmark: dict[str, str],
    provider: str,
    answer_mode: str,
    model: str,
    answer: str,
    response_seconds: float | None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """Store the latest generated answer for human evaluation."""
    source_items = sources or []
    st.session_state["evaluation_candidate"] = {
        "run_id": uuid.uuid4().hex,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": benchmark.get("id", "MANUAL"),
        "domain": benchmark.get("domain", "Manual question"),
        "recommended_mode": benchmark.get("recommended_mode", ""),
        "question": benchmark.get("question", ""),
        "provider": provider,
        "answer_mode": answer_mode,
        "model": model,
        "response_seconds": response_seconds,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "source_count": len(source_items),
        "sources": " | ".join(
            str(source.get("citation", "")) for source in source_items
        ),
        "answer": answer,
    }


def render_evaluation_panel() -> None:
    """Render ratings for the latest answer and an in-memory CSV export."""
    candidate = st.session_state.get("evaluation_candidate")
    if not candidate:
        return

    st.divider()
    st.subheader("Evaluate the latest answer")
    st.caption(
        "Rate the answer against the retrieved evidence, not only whether it "
        "sounds plausible. Records remain in this browser session until you "
        "download them or close the session."
    )
    st.markdown(
        f"**{candidate['benchmark_id']} — {candidate['domain']}** · "
        f"{candidate['provider']} · {candidate['model']}"
    )

    with st.form(f"evaluation_form_{candidate['run_id']}"):
        accuracy = st.slider(
            "Factual and theoretical accuracy",
            1,
            5,
            3,
            help="1 = seriously inaccurate; 5 = fully accurate.",
        )
        source_relevance = st.slider(
            "Relevance of retrieved sources",
            1,
            5,
            3,
            help="1 = largely irrelevant; 5 = directly addresses the question.",
        )
        citation_support = st.slider(
            "Do the cited passages support the claims?",
            1,
            5,
            3,
            help="1 = citations do not support claims; 5 = claims are well supported.",
        )
        completeness = st.slider(
            "Completeness and appropriate qualification",
            1,
            5,
            3,
            help="1 = major omissions/overclaims; 5 = complete and appropriately cautious.",
        )
        notes = st.text_area(
            "Evaluator notes",
            placeholder=(
                "Record unsupported claims, missing sources, incorrect page "
                "references, important omissions, or especially strong features."
            ),
        )
        save_rating = st.form_submit_button("Save evaluation")

    if save_rating:
        record = {
            **candidate,
            "evaluation_utc": datetime.now(timezone.utc).isoformat(),
            "accuracy_1_5": accuracy,
            "source_relevance_1_5": source_relevance,
            "citation_support_1_5": citation_support,
            "completeness_1_5": completeness,
            "mean_rating": round(
                (accuracy + source_relevance + citation_support + completeness) / 4,
                2,
            ),
            "notes": notes.strip(),
        }
        records = st.session_state.setdefault("evaluation_records", [])
        records[:] = [
            item for item in records if item.get("run_id") != candidate["run_id"]
        ]
        records.append(record)
        st.success("Evaluation saved in this browser session.")

    records = st.session_state.get("evaluation_records", [])
    if records:
        evaluation_frame = pd.DataFrame(records)
        summary_columns = [
            "benchmark_id",
            "domain",
            "provider",
            "model",
            "response_seconds",
            "mean_rating",
        ]
        st.dataframe(
            evaluation_frame[
                [column for column in summary_columns if column in evaluation_frame]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download evaluation CSV",
            evaluation_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="local_ai_benchmark_evaluations.csv",
            mime="text/csv",
        )
