"""
Response Composer - Single authority for response generation
"""
import hashlib
from datetime import date
from typing import List, Dict, Optional
import logging
import numpy as np

from models.dharmic_query import DharmicQueryObject
from models.memory_context import ConversationMemory
from models.session import ConversationPhase
from config import settings
from services.profile_builder import build_user_profile
from services.cache_service import get_cache_service

logger = logging.getLogger(__name__)


class ResponseComposer:

    def __init__(self):
        from llm.service import get_llm_service
        from services.response_validator import get_response_validator
        self.llm = get_llm_service()
        # Final-stage gate. Runs after llm.generate_response and may trigger
        # one regeneration with corrective hints if a HIGH-severity check
        # fails. See services/response_validator.py for the contract.
        self.validator = get_response_validator()
        self.available = self.llm.available
        self._embedding_model = None
        logger.info(f"ResponseComposer initialized (LLM available={self.available})")

    def _get_embedding_model(self):
        """Lazy-load embedding model from RAG pipeline (shared singleton)."""
        if self._embedding_model is None:
            try:
                from rag.pipeline import get_rag_pipeline
                pipeline = get_rag_pipeline()
                if pipeline and pipeline._embedding_model is not None:
                    self._embedding_model = pipeline._embedding_model
            except Exception:
                pass
        return self._embedding_model

    def _user_fingerprint(self, memory: ConversationMemory, user_id: Optional[str]) -> str:
        """Hash of user identity + profile fields that affect personalization.

        The cache key MUST be scoped per-user. Without this, cached responses
        with personalized text (name, rashi, deity, mantra suggestions) leak
        across users — found Apr 8 2026 in a live multi-persona test where
        the first persona's cached response was returned to all subsequent
        users with their personal data baked in.

        Authenticated users use their user_id directly. Anonymous users get a
        fingerprint based on profile fields visible to the engine so two
        anonymous users with different profiles don't collide either.
        """
        if user_id:
            return user_id[:16]
        story = memory.story if memory and memory.story else None
        parts = [
            (memory.user_name if memory else "") or "",
            (getattr(story, "preferred_deity", "") if story else "") or "",
            (getattr(story, "rashi", "") if story else "") or "",
            (getattr(story, "nakshatra", "") if story else "") or "",
            (getattr(story, "gender", "") if story else "") or "",
        ]
        raw = "|".join(parts).strip("|")
        if not raw:
            return "anon"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _build_cache_key(
        self,
        query: str,
        phase: Optional[ConversationPhase],
        emotion: str,
        life_domain: str,
        turn_count: int = 0,
        user_fingerprint: str = "anon",
    ) -> str:
        """Build a deterministic cache key from query semantics + context + user identity.

        Includes turn_count so same-session queries never hit stale cached responses.
        Includes user_fingerprint so cached personalized text never leaks across users.
        Includes a format version (`fmt=md1`) so the cache invalidates when the
        response format contract changes — bumped Apr 2026 when we switched
        from plain-text to restricted-markdown responses. Plain-text cached
        entries will miss and regenerate cleanly.
        """
        phase_val = phase.value if phase else "unknown"
        key_str = (
            f"{query.strip().lower()}|{phase_val}|{emotion}|{life_domain}"
            f"|turn{turn_count}|u{user_fingerprint}|fmt=md1|d={date.today().isoformat()}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _check_response_cache(
        self,
        query: str,
        phase: Optional[ConversationPhase],
        memory: ConversationMemory,
        turn_count: int = 0,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Check if a semantically similar query has a cached response (per-user)."""
        if not settings.RESPONSE_CACHE_ENABLED:
            return None

        emotion = (memory.story.emotional_state or "").lower()
        life_domain = (memory.story.life_area or "").lower()
        fp = self._user_fingerprint(memory, user_id)

        cache = get_cache_service()
        cache_key = self._build_cache_key(query, phase, emotion, life_domain, turn_count, fp)
        cached = await cache.get("response_semantic", key=cache_key)
        if cached and isinstance(cached, dict):
            logger.info(f"Response cache HIT for query='{query[:40]}' user_fp={fp}")
            return cached.get("response")
        return None

    async def _store_response_cache(
        self,
        query: str,
        phase: Optional[ConversationPhase],
        memory: ConversationMemory,
        response: str,
        turn_count: int = 0,
        user_id: Optional[str] = None,
    ) -> None:
        """Cache a generated response (scoped to the calling user via fingerprint)."""
        if not settings.RESPONSE_CACHE_ENABLED:
            return
        # Don't cache very short or fallback responses
        if len(response) < 30:
            return

        emotion = (memory.story.emotional_state or "").lower()
        life_domain = (memory.story.life_area or "").lower()
        fp = self._user_fingerprint(memory, user_id)

        cache = get_cache_service()
        cache_key = self._build_cache_key(query, phase, emotion, life_domain, turn_count, fp)
        await cache.set(
            "response_semantic",
            {"response": response, "query": query},
            ttl=settings.RESPONSE_CACHE_TTL,
            key=cache_key,
        )
        logger.debug(f"Response cache SET for query='{query[:40]}' user_fp={fp}")

    async def compose_with_memory(
        self,
        dharmic_query: DharmicQueryObject,
        memory: ConversationMemory,
        retrieved_verses: List[Dict],
        conversation_history: List[Dict],
        reduce_scripture: bool = False,
        phase: Optional[ConversationPhase] = None,
        original_query: Optional[str] = None,
        user_id: Optional[str] = None,
        past_memories: List[str] = None,
        model_override: Optional[str] = None,
        config_override: Optional[Dict] = None,
        session=None,
        response_mode: Optional[str] = None,
    ) -> str:
        """
        Compose a response using:
        - synthesized dharmic query (for RAG context)
        - rich conversation memory
        - verses retrieved via RAG
        - full conversation history (for context continuity)
        - current conversation phase
        - original user query (for natural response)
        - session (optional) — surfaces last_suggestions, suggested_verses,
          and recent_products into the LLM prompt. Without it those fields
          are silently dropped (a bug fixed Apr 2026 — see profile_builder).
        """

        # Use original query for the LLM prompt if available,
        # otherwise fallback to build_search_query
        llm_query = original_query
        if not llm_query:
            llm_query = dharmic_query.build_search_query()

        if not llm_query:
            logger.error("No query text available for ResponseComposer")
            return self._compose_fallback(dharmic_query)

        # Check response cache before expensive LLM call
        # Per-user caching: user_id (or anonymous fingerprint) is part of the
        # key so cached personalized text never leaks across users.
        _turn_count = getattr(memory, "turn_count", 0) or 0
        cached_response = await self._check_response_cache(
            llm_query, phase, memory,
            turn_count=_turn_count,
            user_id=user_id,
        )
        if cached_response:
            return cached_response

        # Optionally thin out the scripture context when a user is very
        # distressed – we still keep a couple of strong anchors.
        context_docs = retrieved_verses
        _distress_cap = max(1, settings.RERANK_TOP_K - 1)
        if reduce_scripture and len(retrieved_verses) > _distress_cap:
            context_docs = retrieved_verses[:_distress_cap]

        # Build user profile from memory + session (session unlocks
        # last_suggestions, suggested_verses, and recent_products fields)
        user_profile = build_user_profile(memory, session=session)

        # Inject past memories into profile
        if past_memories:
            user_profile["past_memories"] = past_memories

        if self.llm.available:
            response = await self.llm.generate_response(
                query=llm_query,
                context_docs=context_docs,
                conversation_history=conversation_history, # Use the explicit history passed from session
                user_profile=user_profile,
                phase=phase,
                memory_context=memory,
                model_override=model_override,
                config_override=config_override,
                response_mode=response_mode,
            )
            # Final gate: run ResponseValidator and optionally regenerate
            # once with corrective hints if a HIGH-severity check fails.
            response = await self._validate_and_repair(
                response,
                phase=phase,
                response_mode=response_mode,
                regenerate=lambda hints: self.llm.generate_response(
                    query=llm_query,
                    context_docs=context_docs,
                    conversation_history=conversation_history,
                    user_profile=user_profile,
                    phase=phase,
                    memory_context=memory,
                    model_override=model_override,
                    config_override=self._merge_correction_hints(config_override, hints),
                    response_mode=response_mode,
                ),
            )
            # Cache the response for future similar queries (per-user)
            await self._store_response_cache(
                llm_query, phase, memory, response,
                turn_count=_turn_count,
                user_id=user_id,
            )
            return response

        logger.info("LLM unavailable, using fallback")
        return self._compose_fallback(dharmic_query)

    async def _validate_and_repair(
        self,
        response: str,
        *,
        phase: Optional[ConversationPhase],
        regenerate,
        response_mode: Optional[str] = None,
    ) -> str:
        """Run the ResponseValidator and optionally regenerate.

        Behavior:
            * Apply any in-place repairs (e.g. verse auto-wrap) to the text.
            * If a HIGH-severity check fails AND retries remain in the budget
              (config.LLM_REGENERATION_RETRIES), call ``regenerate`` with the
              correction hints, then re-validate.
            * Return the best version we have. We never return raw text that
              still contains a known scratchpad leak — that's the contract.

        ``response_mode`` is threaded through to the validator so mode-aware
        checks (presence_first / closure banned-empathy enforcement) can
        fire with mode-specific corrective hints.
        """
        phase_value = phase.value if phase else None
        report = self.validator.validate(response, phase=phase_value, response_mode=response_mode)
        attempts = 0
        max_attempts = max(0, int(settings.LLM_REGENERATION_RETRIES))

        while not report.passed and report.needs_regeneration and attempts < max_attempts:
            attempts += 1
            hints = report.correction_hints
            logger.info(
                f"ResponseComposer: regenerating (attempt {attempts}/{max_attempts}) — hints: {hints}"
            )
            try:
                response = await regenerate(hints)
            except Exception as exc:
                logger.warning(f"Regeneration failed: {exc} — keeping original response")
                break
            report = self.validator.validate(response, phase=phase_value, response_mode=response_mode)

        # Verse auto-wrap and similar in-place repairs are baked into report.text
        return report.text

    @staticmethod
    def _merge_correction_hints(
        config_override: Optional[Dict],
        hints: List[str],
    ) -> Dict:
        """Inject correction hints into the LLM config_override so the next
        generation pass sees them as additional system instructions. Keeps
        the regenerate path declarative — no string surgery on prompts here.
        """
        merged = dict(config_override or {})
        if hints:
            existing = merged.get("correction_hints") or []
            merged["correction_hints"] = list(existing) + list(hints)
        return merged

    async def compose_stream(
        self,
        dharmic_query: DharmicQueryObject,
        memory: ConversationMemory,
        retrieved_verses: List[Dict],
        conversation_history: List[Dict],
        reduce_scripture: bool = False,
        phase: Optional[ConversationPhase] = None,
        original_query: Optional[str] = None,
        user_id: Optional[str] = None,
        past_memories: List[str] = None,
        model_override: Optional[str] = None,
        config_override: Optional[Dict] = None,
        session=None,
        response_mode: Optional[str] = None,
    ):
        """
        Stream response synthesis using LLMService.generate_response_stream.

        ``session`` (optional) is forwarded to ``build_user_profile`` so the
        prompt sees last_suggestions, suggested_verses, and recent_products.
        Without it those fields are silently dropped (Apr 2026 fix).
        """
        llm_query = original_query or dharmic_query.build_search_query()

        if not llm_query:
            yield self._compose_fallback(dharmic_query)
            return

        # Check response cache (per-user) — if hit, yield in word-sized chunks
        # to preserve streaming UX.
        _turn_count = getattr(memory, "turn_count", 0) or 0
        cached_response = await self._check_response_cache(
            llm_query, phase, memory,
            turn_count=_turn_count,
            user_id=user_id,
        )
        if cached_response:
            for word in cached_response.split(' '):
                yield word + ' '
            return

        context_docs = retrieved_verses
        _distress_cap = max(1, settings.RERANK_TOP_K - 1)
        if reduce_scripture and len(retrieved_verses) > _distress_cap:
            context_docs = retrieved_verses[:_distress_cap]

        # Build user profile with session so last_suggestions /
        # suggested_verses / recent_products reach the LLM (Apr 2026 fix).
        user_profile = build_user_profile(memory, session=session)
        if past_memories:
            user_profile["past_memories"] = past_memories

        if self.llm.available:
            full_response_parts = []
            async for chunk in self.llm.generate_response_stream(
                query=llm_query,
                context_docs=context_docs,
                conversation_history=conversation_history,
                user_profile=user_profile,
                phase=phase,
                memory_context=memory,
                model_override=model_override,
                config_override=config_override,
                response_mode=response_mode,
            ):
                full_response_parts.append(chunk)
                yield chunk
            # Cache the full response after streaming completes (per-user)
            full_response = "".join(full_response_parts)
            await self._store_response_cache(
                llm_query, phase, memory, full_response,
                turn_count=_turn_count,
                user_id=user_id,
            )
        else:
            yield self._compose_fallback(dharmic_query)

    def _compose_fallback(self, dq: DharmicQueryObject) -> str:
        response = (
            f"I understand what you're going through.\n\n"
            f"From a dharmic perspective, concepts like "
            f"{', '.join(dq.dharmic_concepts[:2])} remind us to move step by step.\n\n"
            "Take a slow breath. You do not need to solve everything at once."
        )
        logger.info(f"Composed fallback response ({len(response)} chars)")
        return response


_response_composer: Optional[ResponseComposer] = None


def get_response_composer() -> ResponseComposer:
    global _response_composer
    if _response_composer is None:
        _response_composer = ResponseComposer()
    return _response_composer
