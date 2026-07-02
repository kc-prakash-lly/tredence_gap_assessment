"""Pydantic schemas for the gap assessment pipeline."""

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    source_id: str          # e.g. "DOC_ID_305" from DE metadata field source_id
    chunk_id: str           # DE chunk ID (last segment of chunk.name)
    is_synthetic: bool
    # MUST be verbatim substring of a retrieved chunk — validated by callback
    snippet: str


class EvidenceChunk(BaseModel):
    doc_id: str
    doc_title: str
    source_id: str          # source_id field from DE document metadata
    chunk_id: str           # DE chunk ID
    uri: str
    text: str
    retrieval_score: float
    is_synthetic: bool


class GapFinding(BaseModel):
    question_id: str
    question: str
    verdict: Literal["EVIDENCE_FOUND", "GAP_IDENTIFIED", "NO_EVIDENCE"]
    rationale: str
    # Empty ONLY when verdict == NO_EVIDENCE
    citations: list[Citation]
    # Set deterministically by after_agent_callback — NOT by the LLM
    confidence: float = 0.0
