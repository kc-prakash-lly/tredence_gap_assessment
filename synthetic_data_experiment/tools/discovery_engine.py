"""
Discovery Engine search tool for the gap-assessment pipeline.

Auth: Application Default Credentials (SA or gcloud user credential).
      NEVER accept an API key.

Score handling
--------------
In DE *chunk mode*, per-result scores come from:
  1. r.relevance_score        – set when the serving config has Ranking/Re-ranker
                                 enabled.  Most common; this is what the notebook
                                 observes (currently returning 0.0 because the
                                 serving config has no ranker enabled yet).
  2. chunk.relevance_score    – set on some store configurations.
  3. chunk.page_span           – no numeric score; falls back to 0.0.

We read both fields and take whichever is non-zero.  Run the standalone
__main__ block first to confirm real scores before wiring into agents.
"""

from __future__ import annotations

import json
import os
from typing import Any

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as de

from ..models import EvidenceChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proto_to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort protobuf → plain dict conversion."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        from google.protobuf.json_format import MessageToDict
        parsed = MessageToDict(obj._pb) if hasattr(obj, "_pb") else MessageToDict(obj)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _de_endpoint(location: str) -> str:
    if location.lower() == "global":
        return "discoveryengine.googleapis.com"
    return f"{location}-discoveryengine.googleapis.com"


def _is_synthetic(details: dict[str, Any]) -> bool:
    """Return True when doc metadata contains a SYNTHETIC marker."""
    for key in ("classification_rds", "classificationRds", "information_classification__c"):
        val = str(details.get(key, "") or "")
        if "SYNTHETIC" in val.upper():
            return True
    # also check a flat "synthetic" boolean field some stores stamp
    if details.get("synthetic") is True or str(details.get("is_synthetic", "")).upper() == "TRUE":
        return True
    return False


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------


def search_chunks(
    query: str,
    *,
    project_id: str,
    location: str,
    datastore_id: str,
    top_k: int = 8,
    debug: bool = False,
) -> list[EvidenceChunk]:
    """
    Run a chunk-mode DE search and return structured EvidenceChunks.

    Parameters
    ----------
    query       : The retrieval query string.
    project_id  : GCP project ID.
    location    : DE location, e.g. "us" or "global".
    datastore_id: DE data-store ID.
    top_k       : Max results to fetch.
    debug       : If True, print raw per-chunk score info to stdout.

    Returns
    -------
    List of EvidenceChunk with retrieval_score populated from the best
    available numeric score field.
    """
    endpoint = _de_endpoint(location)
    client = de.SearchServiceClient(
        client_options=ClientOptions(api_endpoint=endpoint)
        # ADC is picked up automatically — no explicit credentials argument
    )

    serving_config = (
        f"projects/{project_id}/locations/{location}/collections/default_collection/"
        f"dataStores/{datastore_id}/servingConfigs/default_config"
    )

    content_spec = de.SearchRequest.ContentSearchSpec(
        search_result_mode=de.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,
    )

    request = de.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=top_k,
        content_search_spec=content_spec,
    )

    response = client.search(request=request)

    chunks: list[EvidenceChunk] = []
    for r in response.results:
        chunk = getattr(r, "chunk", None)
        document = getattr(r, "document", None)

        # ── doc_id ──────────────────────────────────────────────────────────
        doc_id = (
            str(getattr(document, "id", "") or "").strip()
            or str(getattr(document, "name", "") or "").split("/")[-1].strip()
            or (
                str(getattr(chunk, "name", "") or "").split("/")[-3].strip()
                if chunk and getattr(chunk, "name", "")
                else ""
            )
        )

        # ── chunk_id ─────────────────────────────────────────────────────────
        chunk_id = (
            str(getattr(chunk, "id", "") or "").strip()
            or str(getattr(chunk, "name", "") or "").split("/")[-1].strip()
        )

        # ── content text ─────────────────────────────────────────────────────
        content = str(getattr(chunk, "content", "") or "").strip()
        if not content:
            continue

        # ── score: prefer r.relevance_score, fall back to chunk field ────────
        result_score = float(getattr(r, "relevance_score", 0.0) or 0.0)
        chunk_score = 0.0
        if chunk is not None:
            chunk_score = float(getattr(chunk, "relevance_score", 0.0) or 0.0)
        score = result_score if result_score != 0.0 else chunk_score

        # ── URI ───────────────────────────────────────────────────────────────
        uri = str(getattr(document, "uri", "") or "").strip()

        # ── metadata for title + synthetic flag ───────────────────────────────
        details: dict[str, Any] = {}
        if document is not None:
            dd = _proto_to_dict(document)
            for key in ("structData", "derivedStructData"):
                sub = dd.get(key)
                if isinstance(sub, dict):
                    details.update(sub)
            if not uri:
                uri = str(details.get("uri") or details.get("link") or "").strip()

        if chunk is not None:
            cmd = getattr(chunk, "document_metadata", None)
            if cmd is not None:
                md = _proto_to_dict(cmd)
                for key in ("structData", "derivedStructData"):
                    sub = md.get(key)
                    if isinstance(sub, dict):
                        details.update(sub)
                if not uri:
                    uri = str(md.get("uri", "") or "").strip()

        doc_title = str(
            details.get("title")
            or details.get("name")
            or uri.split("/")[-1]
            or doc_id
        )

        # source_id comes from the DE metadata field "source_id" (e.g. "DOC_ID_305")
        source_id = str(details.get("source_id") or details.get("sourceId") or "")

        if debug:
            print(
                f"[DEBUG] doc_id={doc_id!r:40s}  chunk_id={chunk_id!r:10s}  "
                f"source_id={source_id!r:15s}  "
                f"result_score={result_score:.6f}  chunk_score={chunk_score:.6f}  "
                f"→ used={score:.6f}"
            )

        chunks.append(
            EvidenceChunk(
                doc_id=doc_id,
                doc_title=doc_title,
                source_id=source_id,
                chunk_id=chunk_id,
                uri=uri,
                text=content,
                retrieval_score=score,
                is_synthetic=_is_synthetic(details),
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# ADK-compatible tool wrapper (reads env at call time)
# ---------------------------------------------------------------------------


def search_discovery_engine(query: str) -> str:
    """
    ADK tool: search Discovery Engine for chunks relevant to *query*.

    Reads GOOGLE_CLOUD_PROJECT, DATASTORE_LOCATION, and DATASTORE_ID
    from the environment at call time (ADK does not inject custom vars).

    DATASTORE_LOCATION is the DE store location (e.g. "us", "global") and is
    intentionally separate from GOOGLE_CLOUD_LOCATION, which ADK uses for
    Vertex AI model routing (e.g. "us-central1").

    Returns a JSON string — a list of EvidenceChunk dicts — or an error
    message string if configuration is missing.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("DATASTORE_LOCATION", "us")
    datastore_id = os.getenv("DATASTORE_ID", "")

    if not project_id or not datastore_id:
        return (
            "Configuration error: GOOGLE_CLOUD_PROJECT and DATASTORE_ID "
            "must be set in the environment."
        )

    try:
        chunks = search_chunks(
            query,
            project_id=project_id,
            location=location,
            datastore_id=datastore_id,
            debug=False,
        )
        return json.dumps([c.model_dump() for c in chunks], indent=2)
    except Exception as exc:  # noqa: BLE001
        return f"Discovery Engine search failed: {exc}"


# ---------------------------------------------------------------------------
# Standalone test — run with:  python -m synthetic_data_experiment.tools.discovery_engine
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()  # picks up .env at repo root

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "dev-mq-tech-transfer")
    location = os.getenv("DATASTORE_LOCATION", "us")
    datastore_id = os.getenv("DATASTORE_ID", "")

    if not datastore_id:
        print(
            "ERROR: set DATASTORE_ID in .env (e.g. masked-data_1781603945004)",
            file=sys.stderr,
        )
        sys.exit(1)

    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Does the product introduce any capacity concerns?"
    )

    print(f"Query : {test_query!r}")
    print(f"Store : {datastore_id}")
    print(f"Proj  : {project_id}  location={location}")
    print("-" * 70)

    results = search_chunks(
        test_query,
        project_id=project_id,
        location=location,
        datastore_id=datastore_id,
        top_k=8,
        debug=True,  # prints raw scores
    )

    print(f"\nReturned {len(results)} chunk(s):\n")
    for i, c in enumerate(results, 1):
        print(
            f"[{i}] score={c.retrieval_score:.4f}  synthetic={c.is_synthetic}\n"
            f"     doc_id  : {c.doc_id}\n"
            f"     title   : {c.doc_title}\n"
            f"     uri     : {c.uri}\n"
            f"     text[:200]: {c.text[:200]!r}\n"
        )
