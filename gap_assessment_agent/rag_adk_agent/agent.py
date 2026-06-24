"""
Facility Gap RAG Agent — Google ADK
Serves via:  adk web --port 8000
"""
import os
from typing import Any, Dict, List

import google.auth
import vertexai
from google.adk.agents import Agent
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from vertexai.generative_models import GenerationConfig, GenerativeModel

# ---------------------------------------------------------------------------
# Bootstrap — read config from environment / ADC
# ---------------------------------------------------------------------------

_, _project_id = google.auth.default()

_DATASTORE_ID = os.environ.get("DATASTORE_ID", "")
_DATASTORE_LOCATION = os.environ.get("DATASTORE_LOCATION", "us")
_MODEL_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# Required by ADK to route model calls through Vertex AI
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", _MODEL_LOCATION)

# Required by vertexai.generative_models (used in _expand_queries)
vertexai.init(project=_project_id, location=_MODEL_LOCATION)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# CONVERSATIONAL_PROMPT = """\
# You are a TS/MS (Technical Services/Manufacturing Science) Lead Scientist supporting a drug product technology transfer for an aseptic injectable (multiple-dose cartridge) from a Sending Site to a Receiving Site.

# A collection of technical documents is available to you: Technical Study Summary Reports, Process Descriptions, Qualification Strategies, and Elemental Impurity Risk Assessments.

# Always call the search_datastore tool first to retrieve relevant document content before answering. Answer ONLY from the retrieved contexts. Do not fabricate information not present in the documents.

# Your task: For each relevant facility area found in the documents, produce a detailed analysis covering:
# - The facility area name
# - Sending site requirements (with source document and section)
# - Global / regulatory requirements (with source document and section)
# - Evidence available at the Receiving Site (with source document and section)
# - Any gaps between requirements and available evidence

# Facility areas include (but are not limited to): facility design and capacity, HVAC/RABS/Isolator qualification, people/material/waste flow, cross-contamination controls, environmental controls, utilities (WFI/Clean Steam/N2), dispensing/glove box equipment, storage and staging, APS readiness, equipment cleaning validation.

# Be precise and cite the source document and section for every requirement and evidence item.\
# """


CONVERSATIONAL_PROMPT = """\
You are a TS/MS (Technical Services/Manufacturing Science) Lead Scientist supporting a drug product technology transfer of Multiple-dose [COMPOUND_1] Injection (3 mL cartridge) from the Sending Site to a Receiving Site, for Registration Stability / Process Validation batches, as described in the technology transfer gap assessment report (LQP-230-1).

A collection of technical documents is available to you, including: Gap Assessment / Risk Assessment reports (per LQP-230-1), Global Process Flow Documents (GLO_gPFD), Technical Study Summary Reports (VPHP evaluation, semi-dried residues downtime), Qualification Strategies (visual inspection, APS), and site-specific assessments.

IMPORTANT - Tool Use:
Call search_datastore exactly once, passing the user's question as the query. The tool internally handles query expansion and retrieves all relevant passages in a single call. After receiving the results, compile your complete analysis and produce a single, final response. Do not call search_datastore multiple times.

Your Task:
- Identify all facility-related requirements documented at the Sending Site and from global standards (EU Annex 1, GQS202, LQP-230-1).
- Identify all facility-related evidence available from the Receiving Site.
- Analyse requirements and evidence, compare them, and prepare the gaps.

For each gap found, extract and present the following in a structured JSON:

{
  "facility_area": "",
  "sending_site_requirements": [{"requirement": "", "source": ""}, ...],
  "global_requirements": [{"requirement": "", "source": ""}, ...],
  "receiving_site_evidence": [{"evidence": "", "source": ""}, ...],
  "gaps": [{"gap_description": "", "risk_to_product_quality": "", "cqa_at_risk": [], "owner": "", "due_date": ""}]
}

Facility areas to cover include (but are not limited to):

- Facility design, capacity, and room classifications (including dedicated line status, aseptic design meeting EU Annex 1 and GQS202)
- Air pressurization schemes and HVAC / RABS / Isolator qualification (including Grade A isolator qualification status)
- People, material, waste, and equipment flow (including flows meeting EU Annex 1 and GQS202 requirements)
- Cross-contamination risk and product mix-up controls (particularly relevant where the Receiving Site shares building space with other product lines)
- Environmental controls (humidity, temperature - e.g., NMT 40% RH for [COMPOUND_1] dispensing and DS handling)
- Utilities: WFI (target 20 degrees C for formulation, added via spray ball), Clean Steam, Process Air, Nitrogen (not required for [COMPOUND_1])
- Dispensing areas and glove box / PTB equipment readiness (including operator training status and placebo handling tests)
- Storage, staging, and sampling areas (including DS hold times, TOR, and hold time qualification)
- Aseptic Process Simulation (APS) readiness (including local aseptic hold time definition and challenge)
- Equipment parts / consumables preparation and cleaning validation (including autoclave / washer loading patterns, clean/dirty hold times, and cleaning validation qualification status)

Ground Rules:
- Be precise. Cite the source document and section for each requirement, evidence entry, and gap.
- Do not fabricate gaps not present in the documents.
- You must populate the full JSON structure for every facility area.
- If there are no gaps for a given area, return the gaps key as an empty list [] and provide a brief explanation of why no gap exists.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_queries(query: str, n: int = 4) -> List[str]:
    """Use Vertex AI Gemini to generate n alternative search sub-queries."""
    model = GenerativeModel("gemini-2.5-flash")
    prompt = (
        f"You are helping with pharmaceutical document retrieval.\n"
        f'Original question: "{query}"\n\n'
        f"Generate {n} short, specific search queries (different phrasings/aspects) "
        f"to help retrieve all relevant documents. "
        f"Output only the queries, one per line, no numbering."
    )
    resp = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=0.2),
    )
    return [line.strip() for line in (resp.text or "").splitlines() if line.strip()][:n]


def _chunk_search_passages(
    query: str,
    page_size: int = 8,
) -> List[Dict[str, Any]]:
    """Chunk-mode Vertex AI Search, returns list of passage dicts."""
    endpoint = (
        "discoveryengine.googleapis.com"
        if _DATASTORE_LOCATION.lower() == "global"
        else f"{_DATASTORE_LOCATION}-discoveryengine.googleapis.com"
    )
    serving_config = (
        f"projects/{_project_id}/locations/{_DATASTORE_LOCATION}"
        f"/collections/default_collection"
        f"/dataStores/{_DATASTORE_ID}/servingConfigs/default_config"
    )
    spec = discoveryengine.SearchRequest.ContentSearchSpec(
        search_result_mode=(
            discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
        ),
    )
    req = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=page_size,
        content_search_spec=spec,
    )
    client = discoveryengine.SearchServiceClient(
        client_options=ClientOptions(api_endpoint=endpoint)
    )

    passages: List[Dict[str, Any]] = []
    for r in client.search(req).results:
        chunk = getattr(r, "chunk", None)
        if chunk is None:
            continue
        content = str(getattr(chunk, "content", "") or "").strip()
        if not content:
            continue
        doc_meta = getattr(chunk, "document_metadata", None)
        title = str(getattr(doc_meta, "title", "") or "") if doc_meta else ""
        uri = str(getattr(doc_meta, "uri", "") or "") if doc_meta else ""
        chunk_name = str(getattr(chunk, "name", "") or "")
        parts = chunk_name.split("/")
        doc_id = parts[-3] if len(parts) >= 3 else ""
        passages.append(
            {"doc_id": doc_id, "title": title, "uri": uri, "content": content}
        )
    return passages


# ---------------------------------------------------------------------------
# ADK tool
# ---------------------------------------------------------------------------


def search_datastore(query: str) -> str:
    """Search the document datastore for relevant chunks matching the query.

    Internally expands the query into multiple sub-queries for better recall,
    deduplicates results, and returns all retrieved passages.

    Args:
        query: The search query to find relevant document chunks.

    Returns:
        Formatted document chunks with citation IDs, titles, URIs, and content.
    """
    if not _DATASTORE_ID:
        return "Error: DATASTORE_ID environment variable is not set."

    sub_queries = _expand_queries(query, n=4)

    seen: set = set()
    all_passages: List[Dict[str, Any]] = []
    for sq in [query] + sub_queries:
        for p in _chunk_search_passages(sq):
            key = (p["doc_id"], p["content"][:80])
            if key not in seen:
                seen.add(key)
                all_passages.append(p)

    if not all_passages:
        return "No relevant documents found."

    lines = []
    for idx, p in enumerate(all_passages, 1):
        lines.append(
            f"[C{idx}]\n"
            f"title: {p['title']}\n"
            f"uri: {p['uri']}\n"
            f"content: {p['content']}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Root agent — discovered by `adk web`
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.5-flash",
    name="facility_rag_agent",
    description="Retrieves facility gap information from technical documents.",
    instruction=CONVERSATIONAL_PROMPT,
    tools=[search_datastore],
)
