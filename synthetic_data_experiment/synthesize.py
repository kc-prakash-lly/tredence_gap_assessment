"""
synthesize.py — Stage 3 of the gap-assessment pipeline.

Reads the question + evidence_chunks from state, produces a GapFinding
(verdict + rationale + verbatim citations), and then runs an
after_agent_callback to:
  1. Compute confidence deterministically from retrieval scores.
  2. Validate the finding (no confabulated citations, no citations on
     NO_EVIDENCE verdict, no citations with unmatched snippets).
"""

from __future__ import annotations

import json
import logging
import os

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

from .models import Citation, EvidenceChunk, GapFinding

logger = logging.getLogger(__name__)

# ── How many top-k chunks to average for confidence ──────────────────────────
_CONFIDENCE_TOP_K = 5


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate(finding: GapFinding, chunks: list[EvidenceChunk]) -> list[str]:
    """
    Return a list of violation messages (empty = clean).

    Rules enforced:
    1. verdict != NO_EVIDENCE → citations must be non-empty.
    2. verdict == NO_EVIDENCE → citations must be empty.
    3. Every citation snippet must be a verbatim substring of some chunk text.
    """
    violations: list[str] = []

    if finding.verdict != "NO_EVIDENCE" and not finding.citations:
        violations.append(
            f"CONFABULATION RISK: verdict={finding.verdict!r} but citations is empty."
        )

    if finding.verdict == "NO_EVIDENCE" and finding.citations:
        violations.append(
            f"INVALID: verdict=NO_EVIDENCE but {len(finding.citations)} citation(s) present."
        )

    all_texts = [c.text for c in chunks]
    for cit in finding.citations:
        if not any(cit.snippet in text for text in all_texts):
            violations.append(
                f"HALLUCINATED SNIPPET: citation snippet not found verbatim in "
                f"any retrieved chunk.  doc_id={cit.doc_id!r}  "
                f"snippet={cit.snippet[:80]!r}"
            )

    return violations


# ---------------------------------------------------------------------------
# Confidence callback
# ---------------------------------------------------------------------------


def compute_confidence_callback(callback_context: CallbackContext) -> None:
    """
    after_agent_callback on synthesize_agent.

    Reads evidence_chunks from state, computes mean retrieval score over
    top-k chunks, rounds to 4dp, and writes it onto the gap_finding in state.
    Also runs validation and logs violations loudly.
    """
    state = callback_context.state

    # ── Parse evidence chunks ─────────────────────────────────────────────────
    raw_chunks = state.get("evidence_chunks", [])
    try:
        chunks = [EvidenceChunk(**c) for c in raw_chunks]
    except Exception as exc:  # noqa: BLE001
        logger.error("compute_confidence_callback: could not parse evidence_chunks: %s", exc)
        chunks = []

    # ── Compute confidence ────────────────────────────────────────────────────
    if not chunks:
        confidence = 0.0
    else:
        top_scores = sorted(
            (c.retrieval_score for c in chunks), reverse=True
        )[:_CONFIDENCE_TOP_K]
        confidence = round(sum(top_scores) / len(top_scores), 4)

    # ── Parse the GapFinding written by the LLM ──────────────────────────────
    raw_finding = state.get("gap_finding")
    if raw_finding is None:
        logger.error("compute_confidence_callback: gap_finding not in state.")
        return

    try:
        if isinstance(raw_finding, str):
            finding = GapFinding.model_validate_json(raw_finding)
        elif isinstance(raw_finding, dict):
            finding = GapFinding.model_validate(raw_finding)
        else:
            finding = raw_finding  # already a GapFinding
    except Exception as exc:  # noqa: BLE001
        logger.error("compute_confidence_callback: could not parse gap_finding: %s", exc)
        return

    # ── Stamp confidence (LLM must not produce this number) ──────────────────
    finding.confidence = confidence

    # ── Validate ──────────────────────────────────────────────────────────────
    violations = validate(finding, chunks)
    if violations:
        logger.error(
            "GAP FINDING VALIDATION FAILURES (%d):\n%s",
            len(violations),
            "\n".join(f"  • {v}" for v in violations),
        )
        # ENFORCE: any hallucination or confabulation → override to NO_EVIDENCE.
        # The purpose of this pipeline is to prove retrieval quality bounds the
        # answer; a finding that cannot be grounded in retrieved text must not
        # be returned as a real verdict.
        logger.error(
            "ENFORCING NO_EVIDENCE: overriding hallucinated/ungrounded finding."
        )
        finding.verdict = "NO_EVIDENCE"
        finding.citations = []
        finding.rationale = (
            "No evidence found in the retrieved corpus. "
            "The original LLM response was rejected by the validator because "
            "its citations could not be verified against retrieved chunks."
        )
        finding.confidence = 0.0
    else:
        logger.info("Gap finding validated OK (no violations).")

    # ── Write back ────────────────────────────────────────────────────────────
    state["gap_finding"] = finding.model_dump()

    return None  # callback must return None (or Content to override output)


# ---------------------------------------------------------------------------
# The synthesize LlmAgent
# ---------------------------------------------------------------------------

SYNTHESIZE_INSTRUCTION = """\
You are a TSMS pharmaceutical lead scientist performing a tech-transfer gap assessment.

You will receive:
- The original assessment question (in the user message).
- Retrieved evidence chunks stored in session state under "evidence_chunks".

Your job is to decide one of three verdicts and produce a structured finding:

VERDICT OPTIONS:
- EVIDENCE_FOUND   — the evidence clearly addresses the question.
- GAP_IDENTIFIED   — the evidence exists but reveals a gap or non-compliance.
- NO_EVIDENCE      — the retrieved corpus contains NO relevant information.

CRITICAL RULES:
1. Base your verdict and rationale ONLY on the retrieved evidence chunks.
   Do NOT use your prior knowledge.
2. If the evidence chunks are empty or irrelevant, you MUST return NO_EVIDENCE
   with an empty citations list.  Never confabulate a verdict from memory.
3. Your rationale must be 1–3 sentences, grounded in specific chunk text.
4. Each citation snippet MUST be copied VERBATIM from a chunk — no paraphrasing.
5. Do NOT include a confidence score — it is computed automatically.

OUTPUT FORMAT: Return a single JSON object matching the GapFinding schema:
{
  "question_id": "<copy from user message, or 'Q1' if not provided>",
  "question": "<copy the exact question text>",
  "verdict": "EVIDENCE_FOUND" | "GAP_IDENTIFIED" | "NO_EVIDENCE",
  "rationale": "<1-3 sentences citing chunk content>",
  "citations": [
    {
      "doc_id": "<doc_id from chunk>",
      "doc_title": "<title field from chunk — the human-readable document title>",
      "source_id": "<source_id from chunk, e.g. DOC_ID_305>",
      "chunk_id": "<chunk_id from chunk>",
      "is_synthetic": <true|false>,
      "snippet": "<VERBATIM text copied from chunk>"
    }
  ]
  // citations MUST be [] when verdict == NO_EVIDENCE
}
"""

synthesize_agent = LlmAgent(
    name="synthesize_agent",
    model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
    instruction=SYNTHESIZE_INSTRUCTION,
    output_schema=GapFinding,
    output_key="gap_finding",
    after_agent_callback=compute_confidence_callback,
    description=(
        "Synthesizes a GapFinding from the question and evidence chunks; "
        "confidence is computed deterministically by the callback."
    ),
)
