"""
evidence.py — Stage 2 of the gap-assessment pipeline.

Reads ``sub_queries`` from session state, runs each through Discovery Engine,
deduplicates the returned chunks by ``doc_id + text[:100]``, and writes the
merged list to state as ``evidence_chunks``.

This is a custom BaseAgent so the retrieval loop is fully deterministic — no
LLM calls are made here.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from .tools.discovery_engine import search_chunks as _search_chunks

logger = logging.getLogger(__name__)

# How many chunks to request per sub-query
_TOP_K = 5

# Minimum retrieval score to keep a chunk. Chunks with score == 0.0 are kept
# only when the DE serving config has no ranker (all scores are 0); once a
# ranker is enabled, anything below this threshold is discarded.
_SCORE_THRESHOLD = 0.6


class EvidenceAgent(BaseAgent):
    """Retrieves evidence chunks for all sub-queries and writes them to state."""

    # Pydantic requires fields to be declared; no custom fields needed here.
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        import os

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = os.getenv("DATASTORE_LOCATION", "us")
        datastore_id = os.getenv("DATASTORE_ID", "")

        if not project_id or not datastore_id:
            err = (
                "EvidenceAgent: GOOGLE_CLOUD_PROJECT and DATASTORE_ID must be set."
            )
            logger.error(err)
            ctx.session.state["evidence_chunks"] = []
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=err)],
                ),
            )
            return

        # ── Read sub_queries written by decompose_agent ──────────────────────
        raw = ctx.session.state.get("sub_queries", "[]")
        if isinstance(raw, str):
            try:
                sub_queries: list[str] = json.loads(raw)
            except json.JSONDecodeError:
                # Sometimes the LLM wraps the JSON in markdown fences
                import re
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                sub_queries = json.loads(match.group(0)) if match else []
        else:
            sub_queries = list(raw) if raw else []

        if not sub_queries:
            logger.warning("EvidenceAgent: no sub_queries found in state.")
            ctx.session.state["evidence_chunks"] = []
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text="No sub-queries to search.")],
                ),
            )
            return

        # ── Search DE for each sub-query and deduplicate ─────────────────────
        seen: set[str] = set()
        all_chunks = []
        all_scores_zero = True  # track whether ranker is returning real scores

        for query in sub_queries:
            try:
                chunks = _search_chunks(
                    query,
                    project_id=project_id,
                    location=location,
                    datastore_id=datastore_id,
                    top_k=_TOP_K,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("DE search failed for query %r: %s", query, exc)
                continue

            for chunk in chunks:
                if chunk.retrieval_score != 0.0:
                    all_scores_zero = False
                key = f"{chunk.doc_id}||{chunk.text[:100]}"
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(chunk)

        # Apply threshold only when the ranker is returning real scores.
        # If every score is 0.0 the serving config has no ranker enabled yet —
        # skip filtering so we don't discard all results silently.
        if not all_scores_zero:
            before = len(all_chunks)
            all_chunks = [c for c in all_chunks if c.retrieval_score >= _SCORE_THRESHOLD]
            logger.info(
                "Score threshold %.2f: kept %d / %d chunk(s).",
                _SCORE_THRESHOLD, len(all_chunks), before,
            )

        # ── Write to state ────────────────────────────────────────────────────
        ctx.session.state["evidence_chunks"] = [
            c.model_dump() for c in all_chunks
        ]

        summary = (
            f"Retrieved {len(all_chunks)} unique evidence chunk(s) "
            f"across {len(sub_queries)} sub-quer{'y' if len(sub_queries)==1 else 'ies'}."
        )
        logger.info(summary)

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=summary)],
            ),
        )


evidence_agent = EvidenceAgent(
    name="evidence_agent",
    description=(
        "Retrieves evidence chunks from Discovery Engine for each sub-query "
        "and stores deduplicated results in state as 'evidence_chunks'."
    ),
)
