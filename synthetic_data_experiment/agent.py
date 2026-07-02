"""
agent.py — root_agent entry point for `adk web`.

The pipeline is a SequentialAgent of three stages:
  1. decompose_agent  — question → sub_queries (state)
  2. evidence_agent   — sub_queries → evidence_chunks (state)
  3. synthesize_agent — question + evidence_chunks → gap_finding (state)

Usage
-----
  adk web
  # then open http://localhost:8000 and send e.g.:
  # {"question_id": "Q1", "question": "Does the product introduce capacity concerns?"}
"""

import os

from dotenv import load_dotenv
from google.adk.agents import SequentialAgent

from .decompose import decompose_agent
from .evidence import evidence_agent
from .synthesize import synthesize_agent

# Load .env from repo root (no-op if already set)
load_dotenv()

# Vertex AI routing — required so ADK uses Vertex rather than AI Studio
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

root_agent = SequentialAgent(
    name="gap_assessment_pipeline",
    description=(
        "Three-stage pipeline: decompose question → retrieve evidence from "
        "Discovery Engine → synthesize a GapFinding with deterministic confidence."
    ),
    sub_agents=[
        decompose_agent,
        evidence_agent,
        synthesize_agent,
    ],
)
