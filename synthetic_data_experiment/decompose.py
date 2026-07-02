"""
decompose.py — Stage 1 of the gap-assessment pipeline.

Turns one assessment question into 1–4 focused retrieval sub-queries and
writes them to session state under ``sub_queries``.
"""

import os

from google.adk.agents import LlmAgent

DECOMPOSE_INSTRUCTION = """\
You are a TSMS pharmaceutical lead scientist performing a tech-transfer gap assessment.

Your task:  Given an assessment question, produce 1–4 short, specific retrieval
sub-queries that together cover all aspects of the question.  These sub-queries
will be sent verbatim to a document search engine, so make each one a
self-contained, search-engine-style phrase.

Rules:
- Output ONLY the JSON array of strings — nothing else.
- Minimum 1, maximum 4 sub-queries.
- Each sub-query must be distinct and cover a different aspect of the question.
- Keep each sub-query under 20 words.

Input: the assessment question is provided in the user message.

Example output:
["elemental impurity risk assessment drug substance",
 "ICH Q3D compliance manufacturing equipment",
 "elemental impurities permitted daily exposure limits"]
"""

decompose_agent = LlmAgent(
    name="decompose_agent",
    model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
    instruction=DECOMPOSE_INSTRUCTION,
    output_key="sub_queries",
    description=(
        "Breaks an assessment question into 1-4 focused retrieval sub-queries "
        "and stores them in session state as 'sub_queries'."
    ),
)
