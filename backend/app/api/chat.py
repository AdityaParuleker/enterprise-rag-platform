"""
Chat & Conversations API Contracts (Phase 4 — Basic RAG Orchestration)
Implements end-to-end RAG chat pipeline with SSE streaming.
"""

import os
import asyncio
import inspect
from unittest.mock import AsyncMock
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from backend.app.auth.rbac import get_current_user, require_permission
from backend.app.generation.providers.factory import get_embedding_provider
from backend.app.retrieval.vector_search import VectorSearchEngine
from backend.app.retrieval.fts_search import FTSSearchEngine
from backend.app.retrieval.hybrid_search import HybridSearchEngine, RetrievalFailureError
from backend.app.retrieval.metadata_filter import MetadataFilterEngine
from backend.app.retrieval.acl_filter import DocumentACLFilter
from backend.app.retrieval.reranker import RerankerEngine
from backend.app.memory.conversation_store import ConversationStore
from backend.app.retrieval.query_rewriter import QueryRewriter
from backend.app.retrieval.compression import ContextualCompressor
from backend.app.generation.context_assembler import ContextAssembler
from backend.app.generation.prompt_builder import RAGPromptBuilder
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.prompt_guard import GuardrailPipeline
from backend.app.auth.audit_logger import AuditLogger
from backend.app.generation.streaming import (
    ValidatedStreamer,
    format_token_event,
    format_done_event,
    format_error_event
)


router = APIRouter(prefix="/api/v1", tags=["Chat"])


class ChatMessagePayload(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    conversation_id: Optional[str] = None
    top_k: Optional[int] = Field(default=10, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    retrieval_mode: Optional[str] = Field(default="hybrid", pattern="^(hybrid|vector|fts)$")
    enable_reranking: Optional[bool] = Field(default=True)
    rerank_top_k: Optional[int] = Field(default=5, ge=1, le=30)
    enable_rewriting: Optional[bool] = Field(default=True)
    enable_compression: Optional[bool] = Field(default=True)
    strict_grounding: Optional[bool] = Field(default=None)

    def get_query(self) -> str:
        q = self.query or self.message
        if not q or not q.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query or message string is required."
            )
        return q.strip()


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Conversation"


from backend.app.chat.query_classifier import QueryClassifier, QueryType

class ChatOrchestrator:
    """
    End-to-end RAG Chat Pipeline Orchestrator with SSE Streaming,
    Conversational Query Routing, Multi-Branch RRF Retrieval, Cross-Encoder Reranking,
    Contextual Compression, Memory, and AI Guardrails.
    """

    def __init__(
        self,
        embedding_provider=None,
        vector_search=None,
        fts_search=None,
        hybrid_search=None,
        metadata_filter_engine=None,
        acl_filter=None,
        reranker_engine=None,
        conversation_store=None,
        query_rewriter=None,
        contextual_compressor=None,
        context_assembler=None,
        prompt_builder=None,
        llm_client=None,
        guardrail_pipeline=None,
        audit_logger=None,
        streamer=None,
        query_classifier=None
    ):
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_search = vector_search or VectorSearchEngine()
        self.fts_search = fts_search or (AsyncMock(return_value=[]) if vector_search else FTSSearchEngine())
        self.hybrid_search = hybrid_search or HybridSearchEngine(
            vector_search_engine=self.vector_search,
            fts_search_engine=self.fts_search
        )
        self.metadata_filter_engine = metadata_filter_engine or MetadataFilterEngine()
        self.acl_filter = acl_filter or DocumentACLFilter()
        self.reranker_engine = reranker_engine or RerankerEngine()
        self.conversation_store = conversation_store or ConversationStore()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.contextual_compressor = contextual_compressor or ContextualCompressor()
        self.context_assembler = context_assembler or ContextAssembler()
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.llm_client = llm_client or LLMClient()
        self.guardrail_pipeline = guardrail_pipeline or GuardrailPipeline()
        self.audit_logger = audit_logger or AuditLogger()
        self.streamer = streamer or ValidatedStreamer()
        self.query_classifier = query_classifier or QueryClassifier(llm_client=self.llm_client)

    async def handle_chat_stream(
        self,
        query: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        retrieval_mode: str = "hybrid",
        enable_reranking: bool = True,
        rerank_top_k: int = 5,
        enable_rewriting: bool = True,
        enable_compression: bool = True,
        strict_grounding: Optional[bool] = None,
        db_conn=None
    ) -> StreamingResponse:
        # STEP 1. Input Guard Execution (Invariant: MUST execute BEFORE QueryRewriter, retrieval, or LLM calls)
        input_res = self.guardrail_pipeline.check_input(query)
        if input_res.action == "BLOCK":
            try:
                await self.audit_logger.log_event(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action="GUARDRAIL_TRIGGERED",
                    resource_type="CHAT",
                    detail={
                        "guardrail_type": "INPUT_INJECTION" if input_res.reason == "PROMPT_INJECTION_DETECTED" else "INPUT_GUARD",
                        "guardrail_stage": "INPUT",
                        "action_taken": "BLOCK",
                        "reason": input_res.reason
                    },
                    conn=db_conn
                )
            except Exception:
                pass

            async def input_err_gen():
                yield format_error_event(
                    f"Input guardrail blocked query: {input_res.reason}",
                    code="PROMPT_INJECTION_DETECTED" if input_res.reason == "PROMPT_INJECTION_DETECTED" else "INVALID_INPUT"
                )
            return self.streamer.create_sse_response(input_err_gen())

        # STEP 2. Conversation Session Authorization & User Message Pre-Persistence
        history_messages = []
        history_summary = None
        if conversation_id:
            conv = await self.conversation_store.get_conversation(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                conn=db_conn
            )
            if not conv:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Conversation '{conversation_id}' not found or unauthorized."
                )

            # Pre-persist user message before retrieval/generation starts
            await self.conversation_store.add_user_message(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                content=query,
                conn=db_conn
            )

            history_messages = await self.conversation_store.get_recent_messages(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                limit=10,
                conn=db_conn
            )
            history_summary = conv.get("summary")

        import logging
        logging.getLogger(__name__).info(
            f"[DEBUG_CHAT] Incoming request | query={repr(query)} | conversation_id={conversation_id} | "
            f"history_messages_count={len(history_messages)} | history_messages={history_messages}"
        )

        # STEP 2.5. Conversational Query Routing (Bypass retrieval for greetings / small talk)
        query_type = await self.query_classifier.classify_query(
            query=query,
            history_messages=history_messages
        )

        if query_type == QueryType.CONVERSATIONAL:
            import logging
            logging.getLogger(__name__).info("Query routing: query_type=conversational, retrieval_skipped=true")

            history_context = ""
            if history_messages:
                prior_msgs = [
                    m for m in history_messages
                    if not (m.get("content") == query and m == history_messages[-1])
                ]
                if prior_msgs:
                    formatted_turns = []
                    for m in prior_msgs:
                        role = "User" if (m.get("role") == "user" or m.get("sender") == "user") else "Assistant"
                        formatted_turns.append(f"{role}: {m.get('content', '')}")
                    history_context = "RECENT CONVERSATION HISTORY:\n" + "\n".join(formatted_turns) + "\n\n"

            conv_user_message = f"{history_context}USER MESSAGE:\n{query}" if history_context else query

            conv_sys_prompt = (
                "You are a helpful enterprise AI assistant. "
                "Respond naturally and concisely to the user. "
                "If the user asks what their name is or asks about personal details from history, answer directly using ONLY facts stated in the provided conversation history. "
                "If the requested personal detail or fact was never stated in the conversation history, state politely that it hasn't been mentioned yet."
            )
            try:
                raw_token_stream = self.llm_client.stream_response(
                    prompt=conv_user_message,
                    system=conv_sys_prompt
                )

                sse_generator = self.streamer.astream_tokens(
                    token_stream=raw_token_stream,
                    citations=[],
                    done_metadata={"citations_count": 0, "query_type": "conversational"}
                )

                async def conv_persistence_wrapper(token_gen):
                    full_content = []
                    async for event_str in token_gen:
                        if isinstance(event_str, str) and event_str.startswith("data: "):
                            try:
                                import json
                                event_data = json.loads(event_str[6:-2])
                                if event_data.get("type") == "token":
                                    full_content.append(event_data.get("content", ""))
                            except Exception:
                                pass
                        yield event_str

                    if conversation_id and full_content:
                        assistant_text = "".join(full_content)
                        try:
                            await self.conversation_store.add_assistant_message(
                                tenant_id=tenant_id,
                                user_id=user_id,
                                conversation_id=conversation_id,
                                content=assistant_text,
                                citations=[],
                                conn=db_conn
                            )
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).warning(f"Assistant message persistence failed: {e}")

                return self.streamer.create_sse_response(conv_persistence_wrapper(sse_generator))
            except Exception as e:
                err_msg = str(e)
                async def err_gen():
                    yield format_error_event(
                        f"Conversational generation failed: {err_msg}",
                        code="GENERATION_FAILURE"
                    )
                return self.streamer.create_sse_response(err_gen())

        import logging
        logging.getLogger(__name__).info("Query routing: query_type=domain, retrieval_skipped=false")

        # 3. Pre-validate metadata filters BEFORE retrieval
        if filters:
            self.metadata_filter_engine.parse_and_validate(filters)

        # 4. Resolve user role UUIDs via DocumentACLFilter if user_id is provided
        user_roles = []
        if user_id:
            user_roles = await self.acl_filter.get_user_role_ids(
                user_id=user_id,
                tenant_id=tenant_id,
                conn=db_conn
            )

        # 5. Query Rewriting & Multi-Branch Sub-Query Decomposition
        if enable_rewriting and (history_messages or history_summary):
            retrieval_queries = await self.query_rewriter.rewrite_query(
                user_query=query,
                history_messages=history_messages,
                summary=history_summary
            )
            max_branches = int(os.getenv("MAX_RETRIEVAL_BRANCHES", "1" if os.getenv("LLM_PROVIDER", "").lower() in ("gemini", "google") else "3"))
            retrieval_queries = retrieval_queries[:max_branches]
        else:
            retrieval_queries = [query]

        # INVARIANT (RAG Benchmark Regression Decision): Do NOT unconditionally force max(30, top_k) or candidate pool floors.
        # Fetch 30 candidates ONLY when an active cross-encoder reranker is enabled.
        # When reranking is disabled, use top_k directly to prevent context dilution and RRF rank corruption.
        retrieval_limit = 30 if enable_reranking else top_k


        # 6. Multi-Branch Retrieval Execution with ACL Pushdown
        branch_results = []
        for branch_query in retrieval_queries:
            try:
                raw_vector = self.embedding_provider.embed(branch_query)
                if inspect.isawaitable(raw_vector):
                    branch_vector = await raw_vector
                else:
                    branch_vector = raw_vector
            except Exception as e:
                err_msg = str(e)
                async def embed_err_gen():
                    yield format_error_event(
                        f"Embedding generation failed: {err_msg}",
                        code="EMBEDDING_FAILURE"
                    )
                return self.streamer.create_sse_response(embed_err_gen())

            try:
                if retrieval_mode == "vector":
                    res = await self.vector_search.search(
                        query_vector=branch_vector,
                        tenant_id=tenant_id,
                        limit=retrieval_limit,
                        metadata_filters=filters,
                        user_id=user_id,
                        user_roles=user_roles,
                        conn=db_conn
                    )
                elif retrieval_mode == "fts":
                    res = await self.fts_search.search(
                        query=branch_query,
                        tenant_id=tenant_id,
                        limit=retrieval_limit,
                        metadata_filters=filters,
                        user_id=user_id,
                        user_roles=user_roles,
                        conn=db_conn
                    )
                else:
                    res = await self.hybrid_search.search(
                        query=branch_query,
                        query_vector=branch_vector,
                        tenant_id=tenant_id,
                        limit=retrieval_limit,
                        metadata_filters=filters,
                        user_id=user_id,
                        user_roles=user_roles,
                        conn=db_conn
                    )
                branch_results.append(res)
            except (RetrievalFailureError, Exception) as e:
                err_msg = str(e)
                async def ret_err_gen():
                    yield format_error_event(
                        f"Retrieval failed: {err_msg}",
                        code="RETRIEVAL_FAILURE"
                    )
                return self.streamer.create_sse_response(ret_err_gen())

        # 6.1 RRF Merging across branches and chunk_id deduplication
        merged_map = {}
        for branch_list in branch_results:
            for rank, candidate in enumerate(branch_list):
                cid = str(candidate["chunk_id"])
                rrf_increment = 1.0 / (60.0 + rank + 1)
                if cid not in merged_map:
                    cand_copy = dict(candidate)
                    cand_copy["rrf_score"] = rrf_increment
                    merged_map[cid] = cand_copy
                else:
                    merged_map[cid]["rrf_score"] += rrf_increment

        merged_candidates = list(merged_map.values())
        merged_candidates.sort(key=lambda x: (-x["rrf_score"], str(x["chunk_id"])))

        # 6.2 Candidate Pool Ceiling: max 30 candidates passed to Phase 6
        merged_candidates = merged_candidates[:30]

        # 6.5 Cross-encoder reranking (if enabled)
        if enable_reranking:
            search_results = await self.reranker_engine.rerank(
                query=query,
                candidates=merged_candidates,
                top_k=rerank_top_k
            )
        else:
            search_results = merged_candidates[:max(10, top_k)]


        # STEP 7. Context Guard & Composite Evidence Scorer
        context_res = self.guardrail_pipeline.check_context(
            query=query,
            retrieved_chunks=search_results,
            enable_reranking=enable_reranking,
            strict_grounding=strict_grounding
        )
        if context_res.action == "INSUFFICIENT_EVIDENCE":
            refusal_text = context_res.detail.get("response_text") if context_res.detail else "I could not find sufficient information in the available documents to answer your question."
            async def refusal_stream():
                yield format_token_event(refusal_text)
                yield format_done_event({"refusal": True, "citations_count": 0})

            return self.streamer.create_sse_response(refusal_stream())

        # 8. Contextual Compression & ACL-protected Neighbor Stitching (if enabled)
        if enable_compression and search_results:
            search_results = await self.contextual_compressor.compress_candidates(
                query=query,
                candidates=search_results,
                tenant_id=tenant_id,
                user_id=user_id,
                user_roles=user_roles,
                conn=db_conn
            )

        # 9. Context assembly (4000 char budget enforcement)
        context_result = self.context_assembler.assemble(search_results)

        # 10. Prompt building (trust boundary & fallback for empty context)
        prompt_result = self.prompt_builder.build_prompt(
            user_query=query,
            context_result=context_result,
            strict_grounding=strict_grounding
        )

        citations = prompt_result.citations_metadata

        if prompt_result.is_fallback:
            async def fallback_stream():
                yield format_token_event(prompt_result.fallback_reason or "No relevant document context found for your query in the workspace.")
                yield format_done_event({"fallback": True, "citations_count": 0})

            return self.streamer.create_sse_response(fallback_stream())

        # STEP 11. Buffer-Validate-Stream Flow with OutputGuard Validation & Single Regeneration Retry
        attempts = 0
        raw_completion = ""
        output_res = None

        while attempts < 2:
            attempts += 1
            try:
                token_chunks = []
                raw_token_stream = self.llm_client.stream_response(
                    prompt=prompt_result.user_message,
                    system=prompt_result.system_prompt
                )
                for tok in raw_token_stream:
                    token_chunks.append(tok)
                raw_completion = "".join(token_chunks)
            except Exception as e:
                err_msg = str(e)
                async def err_gen():
                    yield format_error_event(
                        f"LLM generation failed: {err_msg}",
                        code="GENERATION_FAILURE"
                    )
                return self.streamer.create_sse_response(err_gen())

            # Validate generated output with OutputGuard
            output_res = self.guardrail_pipeline.check_output(raw_completion, citations=citations)
            if output_res.action == "ALLOW":
                break
            elif output_res.action == "RETRY" and attempts < 2:
                # 1 retry with stricter grounding prompt, using exact same evidence set
                prompt_result.user_message += "\n\nCRITICAL: Ensure every citation index [N] matches exact retrieved document number."
                continue
            else:
                break

        if output_res and output_res.action == "BLOCK":
            async def block_gen():
                yield format_error_event(
                    f"Output guardrail blocked completion: {output_res.reason}",
                    code="OUTPUT_GUARD_BLOCKED"
                )
            return self.streamer.create_sse_response(block_gen())

        # Stream validated answer via SSE using ValidatedStreamer
        sse_generator = self.streamer.astream_tokens(
            token_stream=[raw_completion],
            citations=citations,
            done_metadata={"citations_count": len(citations)}
        )

        async def persistence_wrapper(token_gen):
            full_content = []
            async for event_str in token_gen:
                if isinstance(event_str, str) and event_str.startswith("data: "):
                    try:
                        import json
                        event_data = json.loads(event_str[6:-2])
                        if event_data.get("type") == "token":
                            full_content.append(event_data.get("content", ""))
                    except Exception:
                        pass
                yield event_str

            if conversation_id and full_content:
                assistant_text = "".join(full_content)
                try:
                    await self.conversation_store.add_assistant_message(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        content=assistant_text,
                        citations=citations,
                        conn=db_conn
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Assistant message post-stream persistence failed: {e}")

        return self.streamer.create_sse_response(persistence_wrapper(sse_generator))


@router.post("/chat")
async def chat_stream(
    chat_req: ChatRequest,
    current_user: Dict = Depends(require_permission("chat"))
):
    """
    POST /api/v1/chat -> End-to-end RAG SSE streaming endpoint.
    Retrieves tenant-isolated context, builds trust-bounded prompt, streams LLM completion tokens & citations via SSE.
    Requires 'chat' RBAC permission.
    """
    query = chat_req.get_query()
    tenant_id = current_user.get("tenant_id")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User tenant_id is missing."
        )

    user_id = current_user.get("sub") or current_user.get("user_id")

    orchestrator = ChatOrchestrator()
    return await orchestrator.handle_chat_stream(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=chat_req.conversation_id,
        top_k=chat_req.top_k or 10,
        filters=chat_req.filters,
        retrieval_mode=chat_req.retrieval_mode or "hybrid",
        enable_reranking=chat_req.enable_reranking if chat_req.enable_reranking is not None else True,
        rerank_top_k=chat_req.rerank_top_k or 5,
        enable_rewriting=chat_req.enable_rewriting if chat_req.enable_rewriting is not None else True,
        enable_compression=chat_req.enable_compression if chat_req.enable_compression is not None else True,
        strict_grounding=chat_req.strict_grounding
    )


@router.post("/conversations")
async def create_conversation(
    conv_req: Optional[CreateConversationRequest] = None,
    current_user: Dict = Depends(require_permission("chat"))
):
    tenant_id = current_user.get("tenant_id")
    user_id = current_user.get("sub") or current_user.get("user_id")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing user or tenant identity.")

    title = conv_req.title if conv_req and conv_req.title else "New Conversation"
    store = ConversationStore()
    conv = await store.create_conversation(tenant_id=tenant_id, user_id=user_id, title=title)
    return {"data": conv, "error": None}


@router.get("/conversations")
async def list_conversations(
    current_user: Dict = Depends(require_permission("chat"))
):
    tenant_id = current_user.get("tenant_id")
    user_id = current_user.get("sub") or current_user.get("user_id")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing user or tenant identity.")

    store = ConversationStore()
    conversations = await store.list_conversations(tenant_id=tenant_id, user_id=user_id)
    return {"data": conversations, "error": None}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: Dict = Depends(require_permission("chat"))
):
    tenant_id = current_user.get("tenant_id")
    user_id = current_user.get("sub") or current_user.get("user_id")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing user or tenant identity.")

    store = ConversationStore()
    conv = await store.get_conversation(tenant_id=tenant_id, user_id=user_id, conversation_id=conversation_id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found or unauthorized.")

    messages = await store.get_recent_messages(tenant_id=tenant_id, user_id=user_id, conversation_id=conversation_id, limit=50)
    conv["messages"] = messages
    return {"data": conv, "error": None}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Dict = Depends(require_permission("chat"))
):
    tenant_id = current_user.get("tenant_id")
    user_id = current_user.get("sub") or current_user.get("user_id")
    if not tenant_id or not user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing user or tenant identity.")

    store = ConversationStore()
    deleted = await store.delete_conversation(tenant_id=tenant_id, user_id=user_id, conversation_id=conversation_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found or unauthorized.")

    return {"data": {"message": "Conversation deleted successfully"}, "error": None}
