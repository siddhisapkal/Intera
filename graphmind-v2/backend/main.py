from __future__ import annotations

import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import BackgroundTasks, Cookie, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import DatabaseError
from pydantic import BaseModel, Field

from .auth_store import AuthUser, authenticate_user, create_session, delete_session, get_user_by_session_token, register_user
from .chat_history_store import ensure_conversation, get_chat_history, list_conversations, save_message as save_chat_message
from .db import get_session
from .event_store import delete_user_events, log_promotions, log_raw_event, recent_raw_events
from .gemini_chat import analyze_strength_weakness_profile, classify_profile_graph_signals, classify_relation_with_llm, configured_models, evaluate_response_relevance, extract_triple_candidates, generate_company_planner, generate_reply_bundle, generate_interview_question, evaluate_interview_answer, transcribe_audio
from .profile_store import delete_user_profile, fetch_profile_summary, upsert_profile_observations
from .graph.service import graph_memory_service
from .prompt_router import route_prompt
from .relation_semantics import classify_relation_semantics, should_background_enrich, store_llm_relation_semantics
from .resume_analyzer import analyze_resume, extract_text_from_pdf
from .section_resolver import resolve_sections
from .topic_router import topic_semantic_router
from .vector_store import add_message, delete_user_messages, search as vector_search, warm_user_indexes
from .web_research import SearchPlan, build_search_plan, infer_memory_signals_from_plan, search_from_plan

app = FastAPI(title="Intera", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=4)
MEMORY_EXECUTOR = ThreadPoolExecutor(max_workers=2)
SESSION_COOKIE_NAME = "graphmind_session"


class ChatRequest(BaseModel):
    user_id: str | None = None
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    source: str = "chat"
    allow_web_search: bool = False


class SearchRequest(BaseModel):
    user_id: str | None = None
    query: str
    conversation_id: str | None = None
    k: int = 5


class MemorySignalInput(BaseModel):
    entity: str
    relation: str
    confidence: float = Field(ge=0.0, le=1.0)
    entity_type: str = "Entity"
    linked_to_action: bool = False
    raw_text: str | None = None


class MemoryIngestRequest(BaseModel):
    user_id: str | None = None
    source: str = "external_api"
    signals: list[MemorySignalInput] = Field(default_factory=list)


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class CompanyPlannerRequest(BaseModel):
    user_id: str | None = None
    company: str = Field(min_length=2)
    days_left: int | None = Field(default=14)


class InterviewStartRequest(BaseModel):
    mode: str = "mix"


class InterviewAnswerRequest(BaseModel):
    answer: str
    mode: str = "mix"


# ─────────────────────────── startup ────────────────────────────────────────

@app.on_event("startup")
def _startup_create_constraints() -> None:
    try:
        with get_session() as session:
            _deduplicate_user_nodes(session)
            _drop_legacy_chat_graph_schema(session)
            _cleanup_legacy_chat_graph(session)
            _ensure_constraint(
                session,
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
                "user_id_unique",
            )
            graph_memory_service.ensure_schema(session)
            topic_semantic_router.refresh_from_session(session)
        warm_user_indexes(limit_users=100)
    except Exception as e:
        print(f"WARNING: Neo4j startup tasks failed (is Neo4j running?): {e}")


def _ensure_constraint(session, query: str, name: str) -> None:
    try:
        session.run(query)
    except DatabaseError as exc:
        print(f"Skipping constraint {name}: {exc}")


def _drop_legacy_chat_graph_schema(session) -> None:
    constraints = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes")
    for record in constraints:
        labels = set(record.get("labelsOrTypes") or [])
        if labels & {"Conversation", "Message"}:
            session.run(f"DROP CONSTRAINT `{record['name']}` IF EXISTS")
    indexes = session.run("SHOW INDEXES YIELD name, labelsOrTypes")
    for record in indexes:
        labels = set(record.get("labelsOrTypes") or [])
        if labels & {"Conversation", "Message"}:
            session.run(f"DROP INDEX `{record['name']}` IF EXISTS")


def _cleanup_legacy_chat_graph(session) -> None:
    session.run("MATCH (n) WHERE n:Conversation OR n:Message DETACH DELETE n")


def _ensure_user_node(session, *, user_id: str) -> None:
    session.run(
        """
        MERGE (u:User {id: $user_id})
        ON CREATE SET u.created_at = datetime()
        SET u.last_seen = datetime()
        """,
        user_id=user_id,
    )


def _deduplicate_user_nodes(session) -> None:
    rows = session.run(
        """
        MATCH (u:User)
        WHERE u.id IS NOT NULL
        WITH u.id AS id, collect(elementId(u)) AS ids
        WHERE size(ids) > 1
        RETURN id, ids
        """
    )
    for row in rows:
        ids = list(row["ids"])
        if len(ids) < 2:
            continue
        keep_id = ids[0]
        for duplicate_id in ids[1:]:
            _merge_duplicate_user_into(session, keep_id=keep_id, duplicate_id=duplicate_id)


def _merge_duplicate_user_into(session, *, keep_id: str, duplicate_id: str) -> None:
    session.run(
        """
        MATCH (keep:User), (dup:User)
        WHERE elementId(keep) = $keep_id AND elementId(dup) = $duplicate_id
        SET keep += properties(dup)
        """,
        keep_id=keep_id, duplicate_id=duplicate_id,
    )
    outgoing = session.run(
        "MATCH (dup:User)-[r]->(target) WHERE elementId(dup) = $duplicate_id RETURN type(r) AS rel_type, properties(r) AS rel_props, elementId(target) AS target_id",
        duplicate_id=duplicate_id,
    )
    for record in outgoing:
        _merge_relationship(session, start_id=keep_id, end_id=record["target_id"], rel_type=record["rel_type"], rel_props=record["rel_props"] or {})
    incoming = session.run(
        "MATCH (source)-[r]->(dup:User) WHERE elementId(dup) = $duplicate_id RETURN elementId(source) AS source_id, type(r) AS rel_type, properties(r) AS rel_props",
        duplicate_id=duplicate_id,
    )
    for record in incoming:
        _merge_relationship(session, start_id=record["source_id"], end_id=keep_id, rel_type=record["rel_type"], rel_props=record["rel_props"] or {})
    session.run("MATCH (dup:User) WHERE elementId(dup) = $duplicate_id DETACH DELETE dup", duplicate_id=duplicate_id)


def _merge_relationship(session, *, start_id: str, end_id: str, rel_type: str, rel_props: dict) -> None:
    escaped_type = rel_type.replace("`", "``")
    session.run(
        f"""
        MATCH (start_node), (end_node)
        WHERE elementId(start_node) = $start_id AND elementId(end_node) = $end_id
        MERGE (start_node)-[r:`{escaped_type}`]->(end_node)
        SET r += $rel_props
        """,
        start_id=start_id, end_id=end_id, rel_props=rel_props,
    )


def _resolve_user_id(requested_user_id: str | None, current_user: AuthUser | None) -> str:
    requested = (requested_user_id or "").strip()
    if current_user is not None:
        if requested and requested != current_user.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Authenticated users can only access their own memory space.")
        return current_user.user_id
    if not requested:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required or provide a user_id explicitly.")
    return requested


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(key=SESSION_COOKIE_NAME, value=token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)


def _auth_payload(user: AuthUser) -> dict[str, object]:
    return {"authenticated": True, "user": {"user_id": user.user_id, "username": user.username, "created_at": user.created_at}}


def _message_requests_web(message: str) -> bool:
    lowered = " ".join((message or "").lower().split())
    return any(phrase in lowered for phrase in ("from web", "from the web", "search web", "latest", "current", "today", "recent"))


# ─────────────────────────── helpers ────────────────────────────────────────

def _compress_snippet(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3].rstrip() + "..."


def _fetch_graph_bundle(*, user_id, query, section_tags=None, section_families=None, focus_entity=None):
    graph_evidence: dict = {"facts": [], "paths": [], "citations": []}
    graph_context: list = []
    start = time.time()
    with get_session() as session:
        graph_evidence = graph_memory_service.fetch_graph_evidence(session=session, user_id=user_id, query=query, limit=6)
        if section_tags or section_families:
            graph_context = graph_memory_service.fetch_section_context(session=session, user_id=user_id, section_tags=section_tags or [], section_families=section_families or [], focus_entity=focus_entity, query=query, limit=6)
        if not graph_context and list(graph_evidence.get("paths") or []):
            graph_context = graph_memory_service.fetch_graph_context(session=session, user_id=user_id, limit=6)
    return graph_evidence, graph_context, int((time.time() - start) * 1000)


def _fetch_vector_bundle(*, query, user_id, conversation_id, k):
    start = time.time()
    results = vector_search(query=query, user_id=user_id, conversation_id=conversation_id, k=k)
    return results, int((time.time() - start) * 1000)


def _fetch_web_bundle(*, queries, intent, reason):
    start = time.time()
    plan = SearchPlan(should_search=True, intent=intent, confidence=1.0, entities={}, queries=queries, reason=reason)
    items = [{"title": r.title, "snippet": r.snippet, "url": r.url} for r in search_from_plan(plan, limit=4)]
    return items, int((time.time() - start) * 1000)


def _planner_queries(company: str) -> list[str]:
    c = " ".join((company or "").split()).strip()
    return [f"{c} recruitment process rounds", f"{c} previous interview questions", f"{c} aptitude technical hr questions", f"{c} placement preparation topics", f"{c} role interview experience"]


def _planner_memory_context(*, user_id: str) -> list[str]:
    with get_session() as session:
        records = graph_memory_service.fetch_graph_memory(user_id=user_id, session=session, limit=8)
    lines: list[str] = []
    for item in records:
        entity = str(item.get("entity") or "").strip()
        relation = str(item.get("relation") or "").strip()
        entity_type = str(item.get("entity_type") or "").strip()
        if entity and relation:
            lines.append(f"{relation} -> {entity} ({entity_type})")
    profile = fetch_profile_summary(user_id=user_id, limit=4)
    for item in list(profile.get("strengths") or [])[:3]:
        lines.append(f"STRENGTH_PROFILE -> {item['entity']} ({item['entity_type']}, score {float(item['score']):.2f})")
    for item in list(profile.get("weaknesses") or [])[:3]:
        lines.append(f"WEAKNESS_PROFILE -> {item['entity']} ({item['entity_type']}, score {float(item['score']):.2f})")
    return lines


def _profile_signal_queries(*, message, observations, triples):
    lowered = f" {' '.join((message or '').lower().split())} "
    if " i " not in lowered and " my " not in lowered and " me " not in lowered:
        return []
    entities: list[str] = []
    for item in list(observations or [])[:5]:
        entity = " ".join(str(item.get("entity") or "").split()).strip()
        if entity:
            entities.append(entity)
    if not entities:
        for triple in list(triples or [])[:8]:
            if getattr(triple, "subject_type", "").strip().lower() != "user":
                continue
            entity = " ".join(str(getattr(triple, "object_name", "") or "").split()).strip()
            if entity:
                entities.append(entity)
    deduped: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(entity)
    queries: list[str] = []
    for entity in deduped[:3]:
        queries.append(f"{entity} interview preparation important topics")
        queries.append(f"{entity} common interview questions concepts")
    return queries[:4]


def _profile_signal_web_facts(*, message, observations, triples):
    queries = _profile_signal_queries(message=message, observations=observations, triples=triples)
    if not queries:
        return []
    plan = SearchPlan(should_search=True, intent="general_learning", confidence=0.78, entities={}, queries=queries, reason="profile signal extraction context")
    return [f"{r.title}: {r.snippet} ({r.url})" for r in search_from_plan(plan, limit=4) if r.title]


def _process_memory_pipeline(*, user_id, conversation_id, message, source, source_event_id, created_at, inferred_raw_signals=None):
    existing_profile_summary = fetch_profile_summary(user_id=user_id, limit=8)
    extracted_triples = extract_triple_candidates(user_id=user_id, message=message, source=source)
    base_profile_observations = analyze_strength_weakness_profile(message=message, triples=extracted_triples, existing_profile_summary=existing_profile_summary)
    profile_web_facts = _profile_signal_web_facts(message=message, observations=base_profile_observations, triples=extracted_triples)
    profile_observations = analyze_strength_weakness_profile(message=message, triples=extracted_triples, web_facts=profile_web_facts, seed_observations=base_profile_observations, existing_profile_summary=existing_profile_summary) or base_profile_observations
    if profile_observations:
        upsert_profile_observations(user_id=user_id, observations=profile_observations)
    profile_graph_signals = classify_profile_graph_signals(message=message, observations=profile_observations, web_facts=profile_web_facts)
    profile_raw_signals = [
        {"user_id": user_id, "entity": str(item.get("entity") or "").strip(), "entity_type": str(item.get("entity_type") or "Skill").strip() or "Skill", "relation": str(item.get("relation") or "").strip(), "confidence": float(item.get("confidence") or 0.8), "linked_to_action": bool(item.get("linked_to_action")), "source": "profile_graph_classifier", "raw_text": message, "source_event_id": source_event_id}
        for item in profile_graph_signals
        if str(item.get("entity") or "").strip() and str(item.get("relation") or "").strip()
    ]
    profile_summary = fetch_profile_summary(user_id=user_id, limit=5)
    extracted_raw_signals = [
        {"user_id": triple.user_id, "entity": triple.object_name, "entity_type": triple.object_type, "relation": triple.relation, "confidence": triple.confidence, "linked_to_action": triple.linked_to_action, "source": triple.source, "raw_text": triple.raw_text, "source_event_id": source_event_id}
        for triple in extracted_triples
        if triple.subject_type.strip().lower() == "user"
    ]
    all_raw_signals = [*extracted_raw_signals, *profile_raw_signals, *(inferred_raw_signals or [])]
    with get_session() as session:
        promotion_summary = {"ephemeral_count": 0, "promoted_count": 0, "promoted_items": []}
        if extracted_triples:
            promotion_summary = graph_memory_service.process_triples(session=session, triples=extracted_triples)
        if inferred_raw_signals:
            inferred_summary = graph_memory_service.process_signals(session=session, raw_signals=inferred_raw_signals)
            promotion_summary = {"ephemeral_count": int(promotion_summary.get("ephemeral_count") or 0) + int(inferred_summary.get("ephemeral_count") or 0), "promoted_count": int(promotion_summary.get("promoted_count") or 0) + int(inferred_summary.get("promoted_count") or 0), "promoted_items": [*list(promotion_summary.get("promoted_items") or []), *list(inferred_summary.get("promoted_items") or [])]}
        if profile_raw_signals:
            psr = graph_memory_service.process_signals(session=session, raw_signals=profile_raw_signals)
            promotion_summary = {"ephemeral_count": int(promotion_summary.get("ephemeral_count") or 0) + int(psr.get("ephemeral_count") or 0), "promoted_count": int(promotion_summary.get("promoted_count") or 0) + int(psr.get("promoted_count") or 0), "promoted_items": [*list(promotion_summary.get("promoted_items") or []), *list(psr.get("promoted_items") or [])]}
        if int(promotion_summary.get("promoted_count") or 0) > 0:
            topic_semantic_router.refresh_from_session(session)
    log_promotions(user_id=user_id, source_event_id=source_event_id, created_at=created_at, raw_signals=all_raw_signals, summary=promotion_summary)
    for signal in all_raw_signals:
        relation = str(signal.get("relation") or "").strip()
        entity_type = str(signal.get("entity_type") or "").strip()
        semantics = classify_relation_semantics(relation, entity_type=entity_type)
        if not should_background_enrich(semantics):
            continue
        enriched = classify_relation_with_llm(relation=relation, entity_type=entity_type)
        if not enriched:
            continue
        store_llm_relation_semantics(relation=relation, entity_type=entity_type, family=str(enriched.get("family") or "general"), polarity=str(enriched.get("polarity") or "neutral"), section_tags=[str(tag) for tag in list(enriched.get("section_tags") or [])], strength=float(enriched.get("strength") or 0.5))
    return {"signals_extracted": len(extracted_raw_signals), "promotion_summary": promotion_summary, "profile_summary": profile_summary, "profile_web_facts": profile_web_facts}


def _run_memory_pipeline_background(*, user_id, conversation_id, message, source, source_event_id, created_at, inferred_raw_signals=None):
    try:
        _process_memory_pipeline(user_id=user_id, conversation_id=conversation_id, message=message, source=source, source_event_id=source_event_id, created_at=created_at, inferred_raw_signals=inferred_raw_signals)
    except Exception as exc:
        print(f"Background memory pipeline failed: {exc}")


# ─────────────────────────── routes ─────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, object]:
    models = configured_models()
    return {"status": "ok", "models": models, "ephemeral_backend": graph_memory_service.ephemeral_backend}


@app.post("/auth/register")
def auth_register(req: AuthCredentials, response: Response) -> dict[str, object]:
    user = register_user(username=req.username, password=req.password)
    token, _ = create_session(user_id=user.user_id)
    _set_session_cookie(response, token)
    return _auth_payload(user)


@app.post("/auth/login")
def auth_login(req: AuthCredentials, response: Response) -> dict[str, object]:
    user = authenticate_user(username=req.username, password=req.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")
    token, _ = create_session(user_id=user.user_id)
    _set_session_cookie(response, token)
    return _auth_payload(user)


@app.post("/auth/logout")
def auth_logout(response: Response, graphmind_session: str | None = Cookie(default=None)) -> dict[str, object]:
    if graphmind_session:
        delete_session(graphmind_session)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"authenticated": False}


@app.get("/auth/me")
def auth_me(graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    user = get_user_by_session_token(graphmind_session)
    if not user:
        return {"authenticated": False}
    return _auth_payload(user)


@app.get("/graph/memory/{user_id}")
def get_graph_memory(user_id: str, limit: int = 10, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        items = graph_memory_service.fetch_graph_memory(user_id=resolved_user_id, session=session, limit=limit)
    return {"user_id": resolved_user_id, "items": items}


@app.get("/graph/view/{user_id}")
def get_graph_view(user_id: str, limit: int = 24, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        graph = graph_memory_service.fetch_graph_view(user_id=resolved_user_id, session=session, limit=limit)
    return {"user_id": resolved_user_id, **graph}


@app.get("/memory/ephemeral/{user_id}")
def get_ephemeral_memory(user_id: str, limit: int = 20, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    items = graph_memory_service.fetch_ephemeral_memory(user_id=resolved_user_id, limit=limit)
    return {"user_id": resolved_user_id, "backend": graph_memory_service.ephemeral_backend, "items": items}


@app.get("/profile/summary/{user_id}")
def get_profile_summary(user_id: str, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    return {"user_id": resolved_user_id, "profile": fetch_profile_summary(user_id=resolved_user_id, limit=8)}


@app.get("/events/{user_id}")
def get_recent_events(user_id: str, limit: int = 20, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    return {"user_id": resolved_user_id, "items": recent_raw_events(user_id=resolved_user_id, limit=limit)}


@app.delete("/memory/reset/{user_id}")
def reset_user_memory(user_id: str, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        graph_summary = graph_memory_service.reset_user_memory(session=session, user_id=resolved_user_id)
    event_summary = delete_user_events(user_id=resolved_user_id)
    deleted_messages = delete_user_messages(user_id=resolved_user_id)
    profile_summary = delete_user_profile(user_id=resolved_user_id)
    return {"user_id": resolved_user_id, "graph": graph_summary, "events": event_summary, "vector": {"deleted_messages": deleted_messages}, "profile": profile_summary}


@app.post("/search")
def search(req: SearchRequest, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    results = vector_search(query=req.query, user_id=resolved_user_id, conversation_id=req.conversation_id, k=req.k)
    return {"results": results}


@app.get("/chat/history/{conversation_id}")
def chat_history(conversation_id: str, limit: int = 100, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    current_user = get_user_by_session_token(graphmind_session)
    resolved_user_id = _resolve_user_id(None, current_user)
    items = get_chat_history(conversation_id=conversation_id, user_id=resolved_user_id, limit=limit)
    return {"conversation_id": conversation_id, "user_id": resolved_user_id, "messages": items}


@app.get("/chat/conversations")
def chat_conversations(limit: int = 50, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    current_user = get_user_by_session_token(graphmind_session)
    resolved_user_id = _resolve_user_id(None, current_user)
    items = list_conversations(user_id=resolved_user_id, limit=limit)
    return {"user_id": resolved_user_id, "items": items}


@app.post("/memory/signals")
def ingest_memory_signals(req: MemoryIngestRequest, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    created_at = datetime.now(timezone.utc).isoformat()
    source_event_id = log_raw_event(user_id=resolved_user_id, conversation_id=None, source_type=req.source, source_ref="api:memory_signals", role="system", content=f"memory_signals:{len(req.signals)}", metadata={"signal_count": len(req.signals)}, created_at=created_at)
    raw_signals = [{"user_id": resolved_user_id, "entity": s.entity, "relation": s.relation, "confidence": s.confidence, "entity_type": s.entity_type, "linked_to_action": s.linked_to_action, "source": req.source, "raw_text": s.raw_text or s.entity} for s in req.signals]
    with get_session() as session:
        summary = graph_memory_service.process_signals(session=session, raw_signals=raw_signals)
    log_promotions(user_id=resolved_user_id, source_event_id=source_event_id, created_at=created_at, raw_signals=raw_signals, summary=summary)
    return {"user_id": resolved_user_id, "source": req.source, "summary": summary}


@app.post("/planner/company")
def company_planner(req: CompanyPlannerRequest, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    company = " ".join(req.company.split()).strip()
    days_left = max(1, min(int(req.days_left or 14), 60))
    if not company:
        raise HTTPException(status_code=400, detail="Company is required.")
    plan = SearchPlan(should_search=True, intent="prep_guidance", confidence=1.0, entities={"entity": company, "entity_type": "organization", "role": "software engineer", "topic": ""}, queries=_planner_queries(company), reason="company planner research")
    web_results = [{"title": r.title, "snippet": r.snippet, "url": r.url} for r in search_from_plan(plan, limit=6)]
    memory_facts = _planner_memory_context(user_id=resolved_user_id)
    profile_summary = fetch_profile_summary(user_id=resolved_user_id, limit=5)
    planner = generate_company_planner(company=company, days_left=days_left, web_results=web_results, memory_facts=memory_facts, profile_summary=profile_summary)
    return {"user_id": resolved_user_id, "company": company, "days_left": days_left, "planner": planner, "sources": web_results, "memory_facts": memory_facts, "profile_summary": profile_summary}


@app.post("/resume/analyze")
async def resume_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, object]:
    current_user = get_user_by_session_token(graphmind_session)
    resolved_user_id = _resolve_user_id(None, current_user)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB allowed.")

    result = analyze_resume(user_id=resolved_user_id, file_bytes=file_bytes)
    
    # Store full text for interview context
    full_text = extract_text_from_pdf(file_bytes)
    graph_memory_service.ephemeral_store._client.set(
        f"graphmind:resume_context:{resolved_user_id}", 
        full_text, 
        ex=3600 # 1 hour
    )

    signals = result.get("signals", [])

    # Ingest signals into graph and profile
    if signals:
        raw_signals = [
            {"user_id": resolved_user_id, "entity": s["entity"], "entity_type": s["entity_type"], "relation": s["relation"], "confidence": s["confidence"], "linked_to_action": s.get("linked_to_action", True), "source": "resume_upload", "raw_text": s.get("raw_text", s["entity"])}
            for s in signals
            if s.get("entity") and s.get("relation")
        ]
        created_at = datetime.now(timezone.utc).isoformat()
        source_event_id = log_raw_event(user_id=resolved_user_id, conversation_id=None, source_type="resume_upload", source_ref="resume:pdf_upload", role="system", content=f"resume_signals:{len(raw_signals)}", metadata={"filename": file.filename, "signal_count": len(raw_signals)}, created_at=created_at)
        
        with get_session() as session:
            _ensure_user_node(session, user_id=resolved_user_id)
            summary = graph_memory_service.process_signals(session=session, raw_signals=raw_signals)
        
        log_promotions(user_id=resolved_user_id, source_event_id=source_event_id, created_at=created_at, raw_signals=raw_signals, summary=summary)
        
        # Trigger background profile and memory sync
        background_tasks.add_task(
            _run_memory_pipeline_background, 
            user_id=resolved_user_id, 
            conversation_id=None, 
            message=f"Resume Analysis for {result.get('name', 'User')}. Skills: {', '.join(result.get('skills', [])[:10])}", 
            source="resume_upload", 
            source_event_id=source_event_id, 
            created_at=created_at, 
            inferred_raw_signals=raw_signals
        )
    else:
        summary = {"ephemeral_count": 0, "promoted_count": 0, "promoted_items": []}

    return {
        "user_id": resolved_user_id,
        "filename": file.filename,
        "name": result.get("name"),
        "email": result.get("email"),
        "skills": result.get("skills", []),
        "companies": result.get("companies", []),
        "education": result.get("education", []),
        "signals_extracted": len(signals),
        "graph_summary": summary,
        "text_preview": result.get("text_preview", ""),
    }


@app.post("/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks, graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)) -> dict[str, object]:
    resolved_user_id = _resolve_user_id(req.user_id, get_user_by_session_token(graphmind_session))
    with get_session() as session:
        _ensure_user_node(session, user_id=resolved_user_id)
    start = time.time()
    conversation_id = req.conversation_id or f"convo-{uuid4().hex[:10]}"
    ensure_conversation(conversation_id=conversation_id, user_id=resolved_user_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    source_event_id = log_raw_event(user_id=resolved_user_id, conversation_id=conversation_id, source_type=req.source, source_ref="chat:user_message", role="user", content=req.message, metadata={"conversation_id": conversation_id}, created_at=now_iso)
    user_msg_id = str(uuid4())
    add_message(message_id=user_msg_id, text=req.message, metadata={"user_id": resolved_user_id, "conversation_id": conversation_id, "role": "user", "created_at": now_iso})
    save_chat_message(conversation_id=conversation_id, user_id=resolved_user_id, role="user", content=req.message)
    recent_history = get_chat_history(conversation_id=conversation_id, user_id=resolved_user_id, limit=10)
    topic_match = topic_semantic_router.detect(req.message)
    route_decision = route_prompt(req.message, semantic_topic=topic_match.topic if topic_match else None)
    search_plan = build_search_plan(message=req.message, semantic_topic=topic_match.topic if topic_match else None, route_intent=route_decision.intent)
    section_plan = resolve_sections(message=req.message, route_intent=route_decision.intent, semantic_topic=topic_match.topic if topic_match else None, target_entity=str(search_plan.entities.get("entity") or "").strip() or None)
    graph_evidence: dict = {"facts": [], "paths": [], "citations": []}
    graph_context: list = []
    graph_retrieval_time_ms = 0
    retrieved: list = []
    retrieval_time_ms = 0
    retrieval_mode = "skipped"
    web_results: list = []
    web_retrieval_time_ms = 0
    inferred_memory_signals = infer_memory_signals_from_plan(message=req.message, user_id=resolved_user_id, plan=search_plan)
    graph_future = RETRIEVAL_EXECUTOR.submit(_fetch_graph_bundle, user_id=resolved_user_id, query=req.message, section_tags=section_plan.query_tags(), section_families=section_plan.query_families(), focus_entity=section_plan.focus_entity)
    vector_future = RETRIEVAL_EXECUTOR.submit(_fetch_vector_bundle, query=req.message, user_id=resolved_user_id, conversation_id=conversation_id, k=5)
    effective_allow_web_search = bool(req.allow_web_search or _message_requests_web(req.message))
    web_future = None
    web_search_used = False
    if effective_allow_web_search and search_plan.should_search and search_plan.queries:
        web_future = RETRIEVAL_EXECUTOR.submit(_fetch_web_bundle, queries=search_plan.queries, intent=search_plan.intent, reason=search_plan.reason)
    if graph_future is not None:
        graph_evidence, graph_context, graph_retrieval_time_ms = graph_future.result()
    graph_max_score = max((float(item.get("score") or 0.0) for item in list(graph_evidence.get("facts") or [])), default=0.0)
    if vector_future is not None:
        retrieved, retrieval_time_ms = vector_future.result()
    if web_future is not None:
        web_results, web_retrieval_time_ms = web_future.result()
        web_search_used = bool(web_results)
    snippets: list[str] = []
    relevant_retrieved_count = 0
    for result in retrieved:
        score = float(result.get("score") or 0.0)
        if score < 0.5:
            continue
        text = _compress_snippet((result.get("text") or "").strip())
        if text and text != req.message:
            snippets.append(text)
            relevant_retrieved_count += 1
    web_facts = [_compress_snippet(f"{item.get('title')}: {item.get('snippet')} ({item.get('url')})", limit=220) for item in web_results if item.get("title")]
    if graph_max_score < 0.45:
        graph_context = []
        graph_evidence = {"facts": [], "paths": [], "citations": []}
    memory_hit = bool(graph_max_score >= 0.45 or relevant_retrieved_count > 0)
    llm_start = time.time()
    reply_bundle = generate_reply_bundle(user_message=req.message, retrieved_snippets=snippets, recent_history=recent_history, graph_facts=graph_context, evidence_paths=list(graph_evidence.get("paths") or []), web_facts=web_facts if web_search_used else None, memory_found=memory_hit)
    llm_generation_time_ms = int((time.time() - llm_start) * 1000)
    answer = reply_bundle["text"]
    relevance = evaluate_response_relevance(query=req.message, response=answer)
    retrieval_mode = "memory_hit" if memory_hit else "direct_reply"
    if web_search_used:
        retrieval_mode = "memory_plus_web"
    mem_result = _process_memory_pipeline(user_id=resolved_user_id, conversation_id=conversation_id, message=req.message, source=req.source, source_event_id=user_msg_id, created_at=now_iso, inferred_raw_signals=inferred_memory_signals)
    bot_msg_id = str(uuid4())
    add_message(message_id=bot_msg_id, text=answer, metadata={"user_id": resolved_user_id, "conversation_id": conversation_id, "role": "assistant", "created_at": datetime.now(timezone.utc).isoformat()})
    save_chat_message(conversation_id=conversation_id, user_id=resolved_user_id, role="assistant", content=answer)
    log_raw_event(user_id=resolved_user_id, conversation_id=conversation_id, source_type="assistant_reply", source_ref="chat:assistant_reply", role="assistant", content=answer, metadata={"conversation_id": conversation_id, "llm_provider": reply_bundle.get("provider", "unknown"), "llm_model": reply_bundle.get("model", "unknown"), "graph_paths": list(graph_evidence.get("paths") or [])}, created_at=datetime.now(timezone.utc).isoformat())
    return {"user_id": resolved_user_id, "conversation_id": conversation_id, "answer": answer, "retrieved_count": len(snippets), "retrieval_time_ms": retrieval_time_ms, "graph_retrieval_time_ms": graph_retrieval_time_ms, "web_retrieval_time_ms": web_retrieval_time_ms, "retrieval_mode": retrieval_mode, "memory_found": memory_hit, "web_search_used": web_search_used, "suggest_web_search": (not memory_hit and not effective_allow_web_search and bool(search_plan.queries)), "graph_confidence": round(graph_max_score, 4), "graph_promotion_time_ms": 0, "signal_extraction_time_ms": 0, "signals_extracted": mem_result.get("signals_extracted", 0), "promotion_summary": mem_result.get("promotion_summary", {}), "route": route_decision.to_dict(), "topic_match": ({"topic": topic_match.topic, "score": topic_match.score, "source": topic_match.source} if topic_match else None), "search_plan": search_plan.to_dict(), "section_plan": {"sections": section_plan.sections, "focus_entity": section_plan.focus_entity, "tags": section_plan.query_tags(), "families": section_plan.query_families()}, "web_results": web_results, "graph_evidence": graph_evidence, "llm_provider": reply_bundle.get("provider", "unknown"), "llm_model": reply_bundle.get("model", "unknown"), "llm_generation_time_ms": llm_generation_time_ms, "answer_relevance": relevance, "memory_update_mode": "synchronous", "ephemeral_backend": graph_memory_service.ephemeral_backend, "time_ms": int((time.time() - start) * 1000)}


@app.post("/audio/transcribe")
async def audio_transcribe(
    file: UploadFile = File(...),
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)
) -> dict[str, str]:
    current_user = get_user_by_session_token(graphmind_session)
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    return {"text": text}


@app.post("/resume/interview/start")
async def interview_start(
    req: InterviewStartRequest,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)
) -> dict[str, object]:
    try:
        current_user = get_user_by_session_token(graphmind_session)
        resolved_user_id = _resolve_user_id(None, current_user)
        
        mode = req.mode
        
        resume_context = graph_memory_service.ephemeral_store._client.get(f"graphmind:resume_context:{resolved_user_id}")
        if not resume_context:
            raise HTTPException(status_code=400, detail="No resume analyzed yet. Please upload a resume first.")

        if isinstance(resume_context, bytes):
            resume_context = resume_context.decode("utf-8")

        # Fetch additional context for personalization if mode is "mix"
        profile_text = ""
        graph_text = ""
        if mode == "mix":
            try:
                profile_data = fetch_profile_summary(user_id=resolved_user_id, limit=10)
                profile_text = json.dumps(profile_data, indent=2)
                
                with get_session() as session:
                    graph_facts = graph_memory_service.fetch_graph_context(session=session, user_id=resolved_user_id, limit=10)
                graph_text = "\n".join(graph_facts)
            except Exception as e:
                print(f"Personalization error: {e}")
                # Don't fail the whole thing, just fall back to no personalization
                pass

        # Clear previous history
        graph_memory_service.ephemeral_store._client.delete(f"graphmind:interview_history:{resolved_user_id}")
        
        question = generate_interview_question(
            resume_context=resume_context, 
            history=[], 
            profile_context=profile_text, 
            graph_context=graph_text
        )
        
        # Save the question as history
        history = [{"role": "assistant", "content": question}]
        graph_memory_service.ephemeral_store._client.set(
            f"graphmind:interview_history:{resolved_user_id}", 
            json.dumps(history), 
            ex=3600
        )
        
        return {"question": question}
    except Exception as e:
        print(f"Interview start error: {e}")
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume/interview/answer")
async def interview_answer(
    req: InterviewAnswerRequest,
    graphmind_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME)
) -> dict[str, object]:
    try:
        current_user = get_user_by_session_token(graphmind_session)
        resolved_user_id = _resolve_user_id(None, current_user)
        
        answer = req.answer
        mode = req.mode
        
        resume_context = graph_memory_service.ephemeral_store._client.get(f"graphmind:resume_context:{resolved_user_id}")
        history_raw = graph_memory_service.ephemeral_store._client.get(f"graphmind:interview_history:{resolved_user_id}")
        
        if not resume_context or not history_raw:
            raise HTTPException(status_code=400, detail="Interview session not found or expired.")
            
        if isinstance(resume_context, bytes):
            resume_context = resume_context.decode("utf-8")
        if isinstance(history_raw, bytes):
            history_raw = history_raw.decode("utf-8")
            
        history = json.loads(history_raw)
        last_question = history[-1]["content"] if history else ""
        
        # Fetch personalization context if mode is "mix"
        profile_text = ""
        graph_text = ""
        if mode == "mix":
            try:
                profile_data = fetch_profile_summary(user_id=resolved_user_id, limit=10)
                profile_text = json.dumps(profile_data, indent=2)
                
                with get_session() as session:
                    graph_facts = graph_memory_service.fetch_graph_context(session=session, user_id=resolved_user_id, limit=10)
                graph_text = "\n".join(graph_facts)
            except Exception as e:
                print(f"Personalization error: {e}")
                pass

        # Evaluate the answer
        feedback_data = evaluate_interview_answer(
            question=last_question, 
            answer=answer, 
            resume_context=resume_context,
            profile_context=profile_text,
            graph_context=graph_text
        )
        
        # Add to history
        history.append({"role": "user", "content": answer})
        
        # Generate next question
        next_question = generate_interview_question(
            resume_context=resume_context, 
            history=history,
            profile_context=profile_text,
            graph_context=graph_text
        )
        history.append({"role": "assistant", "content": next_question})
        
        # Save history
        graph_memory_service.ephemeral_store._client.set(
            f"graphmind:interview_history:{resolved_user_id}", 
            json.dumps(history), 
            ex=3600
        )
        
        return {
            "feedback": feedback_data,
            "next_question": next_question
        }
    except Exception as e:
        print(f"Interview answer error: {e}")
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────── UI ─────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    import os
    candidates = [
        "/app/frontend/index.html",
        os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>Frontend not found</h1><p>Place frontend/index.html in the project root.</p>"
