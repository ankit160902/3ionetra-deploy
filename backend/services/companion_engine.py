import asyncio
import logging
import random
import re
import time
from typing import Tuple, Optional, TYPE_CHECKING, Dict, List

from config import settings
from constants import TRIVIAL_MESSAGES
from models.session import SessionState, ConversationPhase, SignalType, IntentType
from models.memory_context import ConversationMemory
from services.panchang_service import get_panchang_service

from rag.scoring_utils import get_doc_score
from services.cost_tracker import get_cost_tracker
from services.intent_agent import get_intent_agent
from services.memory_service import get_memory_service
from services.model_router import get_model_router
from services.product_service import get_product_service
from services.conversation_fsm import ConversationFSM
from services.memory_updater import update_memory as _update_memory_impl
from services.product_recommender import ProductRecommender
from services.profile_builder import build_user_profile
from services.signal_collector import collect_signals_from_analysis
from services.retrieval_judge import get_retrieval_judge
if TYPE_CHECKING:
    from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Emotions requiring extra listening turns before guidance transition
_DISTRESS_EMOTIONS = frozenset({"shame", "grief", "guilt", "fear", "humiliation", "trauma", "panic"})

class CompanionEngine:
    """
    Empathetic front-line companion.

    Responsibilities:
    - Listen and update ConversationMemory
    - Decide when we're ready for dharmic wisdom
    - Generate grounded empathetic responses

    Product maps and recommendation logic extracted to product_recommender.py.
    Memory update logic extracted to memory_updater.py.
    Signal collection logic extracted to signal_collector.py.
    Phase transition logic extracted to conversation_fsm.py.
    """

    def __init__(
        self,
        rag_pipeline: Optional["RAGPipeline"] = None,
        *,
        llm=None,
        intent_agent=None,
        memory_service=None,
        product_service=None,
        panchang=None,
        model_router=None,
    ) -> None:
        # Accept injected ports or fall back to singleton factories
        if llm is not None:
            self.llm = llm
        else:
            from llm.service import get_llm_service
            self.llm = get_llm_service()

        self.rag_pipeline = rag_pipeline
        self.panchang = panchang if panchang is not None else get_panchang_service()
        self.intent_agent = intent_agent if intent_agent is not None else get_intent_agent()
        self.memory_service = memory_service if memory_service is not None else get_memory_service(rag_pipeline)
        self.product_service = product_service if product_service is not None else get_product_service()
        self.product_recommender = ProductRecommender(self.product_service)
        self.model_router = model_router if model_router is not None else get_model_router()
        self.available = self.llm.available
        logger.info(f"CompanionEngine initialized (LLM available={self.available})")

    def set_rag_pipeline(self, rag_pipeline: "RAGPipeline") -> None:
        self.rag_pipeline = rag_pipeline
        self.memory_service.set_rag_pipeline(rag_pipeline)
        logger.info("RAG pipeline connected to CompanionEngine")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _build_product_context(self, session: SessionState) -> str:
        """Build product interaction context string for IntentAgent.

        Loads the RelationalProfile (Redis-cached) and appends the
        session-level product count. Used by both streaming and
        non-streaming code paths.
        """
        product_context = ""
        _uid = getattr(session.memory, "user_id", None) or getattr(session, "user_id", None)
        if _uid:
            try:
                from services.memory_reader import load_relational_profile
                _profile = await load_relational_profile(_uid)
                product_context = _profile.to_product_context()
            except Exception as e:
                logger.debug(f"Product context load failed (non-fatal): {e}")

        _session_products = len(getattr(session, "shown_product_ids", set()))
        if _session_products > 0:
            product_context += f" Products shown this session: {_session_products}."

        return product_context

    async def generate_response_stream(
        self,
        session: SessionState,
        message: str,
    ):
        """
        Streaming version of conversational processing.
        Yields: (token, is_metadata_chunk, metadata_dict)
        """
        turn_topics = self._update_memory(session.memory, session, message)

        # Build context with product interaction history for LLM
        memory_summary = session.memory.get_memory_summary()
        product_context = await self._build_product_context(session)
        context_with_products = memory_summary
        if product_context:
            context_with_products += "\n" + product_context
        # Turn count helps the IntentAgent make turn-aware product decisions
        context_with_products += f"\n[Turn {session.turn_count}. Products shown this session: {len(session.shown_product_ids)}]"

        # 🚀 Parallel Task Execution: Intent Analysis + Memory Retrieval
        user_id = getattr(session, 'user_id', None) or getattr(session.memory, 'user_id', None)
        tasks = [
            self.intent_agent.analyze_intent(message, context_with_products),
        ]

        if user_id:
            tasks.append(self.memory_service.retrieve_relevant_memories(user_id, message))
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        analysis, past_memories = await asyncio.gather(*tasks)

        # 1. Update session signals and story from intent analysis
        collect_signals_from_analysis(session, analysis)

        # 1a. Off-topic short-circuit — yield the redirect from YAML and stop.
        # Detector wraps IntentAgent.is_off_topic so the call site never has
        # to know which dict key holds the boolean.
        from services.off_topic_detector import get_off_topic_detector
        _off_topic = get_off_topic_detector()
        if _off_topic.is_off_topic(analysis):
            logger.info(
                f"Off-topic intent in stream (session={session.session_id}); "
                f"yielding canned redirect"
            )
            yield {
                "type": "control",
                "is_ready_for_wisdom": False,
                "phase": ConversationPhase.LISTENING.value,
                "turn_topics": turn_topics,
            }
            yield {
                "type": "token",
                "content": _off_topic.get_redirect_message(),
            }
            return

        # 🚀 Decide phase (via FSM)
        intent = analysis.get("intent")
        fsm = ConversationFSM(session)
        is_ready, readiness_trigger = fsm.evaluate(analysis, turn_topics)

        current_phase = ConversationPhase.CLOSURE if intent == IntentType.CLOSURE else ConversationPhase.CLARIFICATION

        if is_ready:
            # Metadata for frontend
            yield {
                "type": "control",
                "is_ready_for_wisdom": True,
                "phase": ConversationPhase.GUIDANCE.value,
                "turn_topics": turn_topics,
                "intent": intent
            }
            return # main.py will handle RAG + Synthesis stream

        # Listening Phase → Stream LLM response
        yield {
            "type": "control",
            "is_ready_for_wisdom": False,
            "phase": current_phase.value,
            "turn_topics": turn_topics
        }

        if self.available:
            # Start streaming with proper model routing + token budget
            user_profile = build_user_profile(session.memory, session)
            if past_memories:
                user_profile["past_memories"] = past_memories

            # Route through model_router for correct token budget (was missing → truncation)
            routing = self.model_router.route(
                intent_analysis=analysis,
                phase=current_phase,
                session=session,
                has_rag_context=False,
            )

            async for token in self.llm.generate_response_stream(
                query=message,
                context_docs=[],
                conversation_history=session.conversation_history,
                user_profile=user_profile,
                phase=current_phase,
                memory_context=session.memory,
                model_override=routing.model_name,
                config_override=routing.config_override,
                response_mode=analysis.get("response_mode", "exploratory"),
            ):
                yield {"type": "token", "content": token}
        else:
            yield {"type": "token", "content": "I'm here with you. Could you tell me a little more?"}

    async def process_message_preamble(
        self,
        session: SessionState,
        message: str,
    ) -> Dict:
        """
        All analysis, memory, signals, RAG, products — everything EXCEPT LLM response generation.
        Returns dict with: is_ready_for_wisdom, context_docs, turn_topics, recommended_products,
        active_phase, user_profile, past_memories, analysis, acknowledgement (if guidance).
        """
        turn_topics = self._update_memory(session.memory, session, message)

        # Intent classification runs first. MemoryReader (below) depends on
        # the analysis (response_mode, emotion) for its mode gate and tone
        # filter, and we want to defer the reader until AFTER the crisis
        # and off-topic short-circuits so those paths don't pay the ~100ms
        # reader cost for responses that won't use the result anyway.
        # Build context with product interaction history for LLM
        memory_summary = session.memory.get_memory_summary()
        product_context = await self._build_product_context(session)
        context_with_products = memory_summary
        if product_context:
            context_with_products += "\n" + product_context
        context_with_products += f"\n[Turn {session.turn_count}. Products shown this session: {len(session.shown_product_ids)}]"

        _t_intent_start = time.perf_counter()
        analysis = await self.intent_agent.analyze_intent(
            message, context_with_products
        )
        _intent_ms = (time.perf_counter() - _t_intent_start) * 1000

        user_id = getattr(session, 'user_id', None) or getattr(session.memory, 'user_id', None)

        # Guard: if IntentAgent returned an exception, fall back to a safe dict
        # so downstream code doesn't crash.
        if isinstance(analysis, Exception):
            logger.warning(f"IntentAgent failed (non-fatal): {analysis}. Using fallback.")
            analysis = {
                "intent": IntentType.OTHER, "emotion": "neutral",
                "life_domain": "unknown", "entities": {}, "urgency": "normal",
                "summary": "", "needs_direct_answer": False,
                "product_signal": {"intent": "none", "confidence": 0.0, "type_filter": "any",
                                   "search_keywords": [], "max_results": 0, "sensitivity_note": ""},
                "recommend_products": False, "product_search_keywords": [],
                "product_rejection": False, "query_variants": [],
                "expected_length": "moderate", "is_off_topic": False,
            }

        # Initialize reader-backed state — will be populated after the
        # crisis/off-topic short-circuits fail to fire.
        relational_profile_text: str = ""
        past_memories: list = []

        # 1. Update session signals from LLM analysis
        collect_signals_from_analysis(session, analysis)

        # Observability: log the chosen response_mode so runtime behavior is
        # traceable. Follows the existing IntentAgent logger.info pattern.
        logger.info(
            f"Mode selected: {analysis.get('response_mode', 'exploratory')} "
            f"for '{message[:50]}' (session={session.session_id}, turn={session.turn_count})"
        )

        # 1a. LLM-based crisis detection — catches typos/misspellings that the
        # keyword-based check in _preflight missed. The IntentAgent sees the
        # full message and understands "i dnt wnt 2 liv" = crisis, regardless
        # of spelling. This is the STRUCTURAL safety net — keyword check is
        # just a speed optimization for obvious cases. Apr 2026.
        if isinstance(analysis, dict) and analysis.get("urgency") == "crisis":
            logger.warning(
                f"LLM-based crisis detected (session={session.session_id}): "
                f"IntentAgent urgency=crisis for message '{message[:50]}'"
            )
            from services.crisis_response_composer import get_crisis_response_composer
            from services.crisis_memory_hook import dispatch_crisis_meta_fact
            session.crisis_turn_count += 1
            # Fire-and-forget meta-fact write: set prior_crisis_flag + bump
            # count on user_profiles. NEVER stores verbatim crisis text.
            dispatch_crisis_meta_fact(session.memory.user_id)
            crisis_response = get_crisis_response_composer().compose(session, message)
            return {
                "is_ready_for_wisdom": False,
                "readiness_trigger": "crisis",
                "context_docs": [],
                "turn_topics": turn_topics,
                "recommended_products": [],
                "active_phase": ConversationPhase.LISTENING,
                "user_profile": build_user_profile(session.memory, session),
                "past_memories": past_memories or [],
                "analysis": analysis,
                "model_override": None,
                "config_override": None,
                "crisis_response": crisis_response,
                "response_mode": analysis.get("response_mode", "presence_first"),
            }

        # 1b. Off-topic short-circuit — if the intent agent (or fast-path) flagged
        # the message as outside spiritual/life-guidance scope, return a gentle
        # redirect immediately. This avoids running RAG, product search, or
        # routing the LLM through the guidance pipeline for queries the
        # companion shouldn't engage with at all (coding help, sports scores, etc.).
        from services.off_topic_detector import get_off_topic_detector
        _off_topic_detector = get_off_topic_detector()
        if _off_topic_detector.is_off_topic(analysis):
            logger.info(
                f"Off-topic intent detected (session={session.session_id}); "
                f"returning canned redirect"
            )
            return {
                "is_ready_for_wisdom": False,
                "readiness_trigger": "off_topic",
                "context_docs": [],
                "turn_topics": turn_topics,
                "recommended_products": [],
                "active_phase": ConversationPhase.LISTENING,
                "user_profile": build_user_profile(session.memory, session),
                "past_memories": past_memories or [],
                "analysis": analysis,
                "model_override": None,
                "config_override": None,
                "off_topic_response": _off_topic_detector.get_redirect_message(),
                "response_mode": analysis.get("response_mode", "exploratory"),
            }

        # 1c. Dynamic MemoryReader — profile load (always-on, Redis-cached)
        # + mode-gated episodic retrieval (scored, tone-filtered). Runs only
        # for authenticated users AFTER crisis/off-topic short-circuits fail
        # to fire — those paths never use memory so we save ~100ms on them.
        if user_id:
            _t_reader_start = time.perf_counter()
            try:
                from services import memory_reader
                read_result = await memory_reader.load_and_retrieve(
                    user_id=user_id,
                    query=message,
                    response_mode=analysis.get("response_mode"),
                    analysis=analysis,
                    session=session,
                )
                relational_profile_text = read_result.profile.to_prompt_text()
                past_memories = [
                    sm.memory.get("text", "")
                    for sm in read_result.episodic
                    if sm.memory.get("text")
                ]
            except Exception as exc:
                logger.warning(
                    f"MemoryReader failed (non-fatal): "
                    f"{type(exc).__name__}: {exc}"
                )
            _reader_ms = (time.perf_counter() - _t_reader_start) * 1000
            logger.info(
                f"PERF_PREAMBLE intent={_intent_ms:.0f}ms reader={_reader_ms:.0f}ms "
                f"profile_len={len(relational_profile_text)} "
                f"past_mem_count={len(past_memories)}"
            )

        # 2. Store in Long-Term Memory if significant
        is_significant_update = (
            len(message) > 30 or
            analysis.get("urgency") in ["high", "crisis"] or
            (analysis.get("emotion") and analysis["emotion"] not in [None, "neutral"]) or
            (analysis.get("entities", {}).get("ritual")) or
            (analysis.get("life_domain") and analysis.get("life_domain") != "unknown")
        )

        if is_significant_update:
            if user_id:
                logger.info(f"💾 Preserving semantic memory anchor for user {user_id} (Reason: significant update)")
                mem_anchor = analysis.get("summary", message)
                await self.memory_service.store_memory(user_id, mem_anchor)

        # 🚀 CORE LOGIC: Decide if we should go to GUIDANCE phase (via FSM)
        intent = analysis.get("intent")
        fsm = ConversationFSM(session)
        is_ready, readiness_trigger = fsm.evaluate(analysis, turn_topics)
        session.readiness_trigger = readiness_trigger

        current_phase = ConversationPhase.CLOSURE if intent == IntentType.CLOSURE else ConversationPhase.CLARIFICATION

        # is_direct_ask preserved for downstream use (acknowledgement selection)
        is_direct_ask = is_ready and readiness_trigger in ("explicit_request", "user_asked_for_guidance")

        # MODE OVERRIDE for GUIDANCE phase: presence_first and exploratory both suppress dharmic
        # content in different ways — presence_first forbids mantras/verses entirely, exploratory
        # asks another clarifying question instead of giving guidance. When FSM decides GUIDANCE
        # is appropriate, both modes are wrong. Override to teaching so the full dharmic toolkit
        # (mantras, verses, remedies, profile-aware practices) is available.
        if is_ready and analysis.get("response_mode") in ("presence_first", "exploratory"):
            analysis["response_mode"] = "teaching"

        # ------------------------------------------------------------------
        # Ready for wisdom → prepare acknowledgement + context docs + products
        # ------------------------------------------------------------------
        if is_ready:
            acknowledgements = [
                "Thank you for sharing. Let me reflect on this through the lens of Dharma.",
                "I appreciate your honesty. I'm looking into the scriptures for guidance.",
                "I hear you deeply. Please give me a moment to gather wisdom for your situation.",
                "Namaste. Your words have touched me. I am seeking the right dharmic path for you."
            ]

            if is_direct_ask:
                acknowledgements = [
                    "I understand your question. Let me provide the specific guidance for this.",
                    "That's a very clear request. I'm gathering the relevant wisdom for you right now.",
                    "I see what you are seeking. Let me look that up for you."
                ]

            # PRODUCT RECOMMENDATIONS — DISABLED
            # # 🛍️ PRODUCT RECOMMENDATION (kicked off in parallel with the
            # # context-doc work below). The recommender reads only intent +
            # # turn_topics + life_domain — none of which depend on the RAG
            # # context that's about to be computed. Running both in parallel
            # # cuts ~2-3s off slow-path latency. The recommender catches its
            # # own exceptions and returns [] on failure, so awaiting the task
            # # at the end is safe.
            # product_task = asyncio.create_task(
            #     self.product_recommender.recommend(session, message, analysis)
            # )

            # 📚 SCRIPTURE RETRIEVAL — mode-gated, sequential (post-Apr-2026).
            context_docs = []
            is_verse_request = "Verse Request" in turn_topics
            is_product_request = "Product Inquiry" in turn_topics
            # Gate RAG based on response_mode (LLM-classified, not keyword-matched).
            # practical_first / closure → skip RAG entirely; these modes never
            # surface scripture.
            # presence_first on early turns → skip RAG; the user needs presence,
            # not citations.
            # teaching → always use RAG (full scripture power).
            # exploratory → let RAG run if it has something concrete to surface.
            _response_mode = analysis.get("response_mode", "exploratory")
            _skip_rag_for_mode = (
                _response_mode in ("practical_first", "closure")
                or (_response_mode == "presence_first" and session.turn_count <= 2)
            )
            should_get_verses = (
                is_verse_request
                or (is_ready and not is_product_request and not _skip_rag_for_mode)
            )

            if should_get_verses and self.rag_pipeline and self.rag_pipeline.available:
                # Sequential RAG: with speculative dispatch removed (Apr 2026),
                # we only run RAG here, only for modes that need it. This costs
                # ~1-2s on teaching queries but saves 60-170s on practical/closure
                # turns that used to pay speculative reranker cost for nothing.
                try:
                    context_docs, _top_score = await self._retrieve_and_validate(
                        session, message, analysis, intent, "guidance",
                    )
                except Exception as e:
                    logger.warning(f"Guidance-phase RAG/validation failed: {e}")

            # PRODUCT RECOMMENDATIONS — DISABLED
            # # Now await the product recommendation that's been running in
            # # parallel with the RAG/validation work above.
            # products = await product_task
            products = []

            # Model routing decision
            routing = self.model_router.route(
                intent_analysis=analysis,
                phase=ConversationPhase.GUIDANCE,
                session=session,
                has_rag_context=bool(context_docs),
            )

            _guidance_user_profile = build_user_profile(session.memory, session)
            if relational_profile_text:
                _guidance_user_profile["relational_profile"] = relational_profile_text
            if past_memories:
                _guidance_user_profile["past_memories"] = past_memories
            return {
                "is_ready_for_wisdom": True,
                "readiness_trigger": readiness_trigger,
                "context_docs": context_docs,
                "turn_topics": turn_topics,
                "recommended_products": products,
                "active_phase": ConversationPhase.GUIDANCE,
                "user_profile": _guidance_user_profile,
                "past_memories": past_memories,
                "analysis": analysis,
                "acknowledgement": random.choice(acknowledgements),
                "model_override": routing.model_name,
                "config_override": routing.config_override,
                "response_mode": analysis.get("response_mode", "exploratory"),
            }

        # ------------------------------------------------------------------
        # Not ready → prepare context for listening-phase LLM call
        # ------------------------------------------------------------------
        context_docs = []

        if self.available and self.rag_pipeline and self.rag_pipeline.available:
            # Listening-phase RAG gating — layered. Mode-based skips come first
            # (practical_first / closure / early presence_first all skip RAG
            # entirely); then the pre-existing intent-based skips for greetings,
            # short acknowledgments, and panchang queries act as belt-and-
            # suspenders guards in case the mode classification is unavailable.
            _response_mode_listen = analysis.get("response_mode", "exploratory")
            _skip_rag_listen_mode = (
                _response_mode_listen in ("practical_first", "closure")
                or (_response_mode_listen == "presence_first" and session.turn_count <= 2)
            )
            skip_rag_intents = {IntentType.GREETING, IntentType.CLOSURE}
            # Skip RAG for emotional expressions in early turns — they need presence, not scripture
            skip_rag_early = session.turn_count <= 2 and intent == IntentType.EXPRESSING_EMOTION
            # Skip RAG for short acknowledgments (ok, thanks, sure, fine) — no scripture needed
            _msg_words = len(message.strip().split())
            _positive_emotions = {"joy", "gratitude", "hope", "neutral"}
            _detected_emotion = (analysis.get("emotion") or "").lower()
            _is_short_ack = _msg_words <= 3  # "ok", "ok fine", "sure", "thanks"
            skip_rag_short = _is_short_ack or (_msg_words <= 8 and _detected_emotion in _positive_emotions)
            panchang_keywords = ["panchang", "tithi", "nakshatra", "muhurat", "today's day", "calendar"]
            _is_panchang = any(k in message.lower() for k in panchang_keywords)
            if _skip_rag_listen_mode:
                logger.info(f"Skipping RAG for mode={_response_mode_listen} in listening phase")
            elif intent in skip_rag_intents or skip_rag_early or skip_rag_short:
                logger.info(f"Skipping RAG for {intent} intent in listening phase (turn={session.turn_count}, words={_msg_words}, emotion={_detected_emotion})")
            elif _is_panchang:
                logger.info("Skipping RAG for Panchang-related query in listening phase")
            else:
                # Sequential RAG: run the search only when the mode and intent
                # checks all agree the query needs scripture context.
                try:
                    context_docs, _top_score = await self._retrieve_and_validate(
                        session, message, analysis, intent, "listening",
                    )
                except Exception as e:
                    logger.warning(f"Listening-phase RAG/validation failed: {e}")

        user_profile = build_user_profile(session.memory, session)
        if relational_profile_text:
            user_profile["relational_profile"] = relational_profile_text
        if past_memories:
            user_profile["past_memories"] = past_memories

        # PRODUCT RECOMMENDATIONS — DISABLED
        # # 🛍️ PRODUCT RECOMMENDATION (listening phase — delegated to ProductRecommender)
        # products = await self.product_recommender.recommend(session, message, analysis)
        products = []

        # Model routing decision
        routing = self.model_router.route(
            intent_analysis=analysis,
            phase=current_phase,
            session=session,
            has_rag_context=bool(context_docs),
        )

        return {
            "is_ready_for_wisdom": False,
            "readiness_trigger": readiness_trigger,
            "context_docs": context_docs,
            "turn_topics": turn_topics,
            "recommended_products": products,
            "active_phase": current_phase,
            "user_profile": user_profile,
            "past_memories": past_memories,
            "analysis": analysis,
            "model_override": routing.model_name,
            "config_override": routing.config_override,
            "response_mode": analysis.get("response_mode", "exploratory"),
        }

    async def process_message(
        self,
        session: SessionState,
        message: str,
    ) -> Tuple:
        """
        Returns:
            (assistant_text, is_ready_for_wisdom, context_docs_used, turn_topics,
             recommended_products, active_phase, model_override, config_override,
             past_memories, response_mode, analysis)

        The element at index 9 (``response_mode``) is the IntentAgent-
        classified response shape for this turn — used by routers/chat.py to
        pass the mode through to ResponseComposer for guidance-phase synthesis.

        The element at index 10 (``analysis``) is the full IntentAgent output
        dict — used by routers/chat.py to gate the fire-and-forget memory
        extraction dispatch (skip crisis / off-topic / greeting / closure
        turns). Existing scripts and regression tests that access the tuple
        by index only read items 0, 5, 9 and are unaffected by this addition.
        """
        meta = await self.process_message_preamble(session, message)
        _mode = meta.get("response_mode", "exploratory")
        _analysis = meta.get("analysis", {})

        # LLM-based crisis short-circuit: IntentAgent detected urgency=crisis
        # (catches typos/misspellings that keyword check in _preflight missed).
        if meta.get("crisis_response"):
            return (
                meta["crisis_response"], False, [],
                meta["turn_topics"], [], ConversationPhase.LISTENING,
                None, None, meta.get("past_memories", []), _mode, _analysis,
            )

        # Off-topic short-circuit: return canned redirect, skip LLM call entirely.
        if meta.get("off_topic_response"):
            return (
                meta["off_topic_response"], False, [],
                meta["turn_topics"], [], ConversationPhase.LISTENING,
                None, None, meta.get("past_memories", []), _mode, _analysis,
            )

        if meta["is_ready_for_wisdom"]:
            return (
                meta["acknowledgement"], True, meta["context_docs"],
                meta["turn_topics"], meta["recommended_products"], ConversationPhase.GUIDANCE,
                meta.get("model_override"), meta.get("config_override"),
                meta.get("past_memories", []), _mode, _analysis,
            )

        # Listening phase — make the LLM call
        if self.available:
            reply = await self.llm.generate_response(
                query=message,
                context_docs=meta["context_docs"],
                conversation_history=session.conversation_history,
                user_profile=meta["user_profile"],
                phase=meta["active_phase"],
                memory_context=session.memory,
                model_override=meta.get("model_override"),
                config_override=meta.get("config_override"),
                response_mode=meta.get("response_mode"),
            )

            # Cost tracking
            if settings.MODEL_COST_TRACKING_ENABLED:
                try:
                    usage = self.llm.last_usage
                    analysis = meta.get("analysis", {})
                    routing_model = meta.get("model_override", settings.GEMINI_MODEL)
                    get_cost_tracker().log(
                        session_id=session.session_id,
                        model_name=routing_model,
                        tier="standard",
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        intent=str(analysis.get("intent", "")),
                        phase=meta["active_phase"].value,
                        response_mode=str(analysis.get("response_mode", "")),
                    )
                except Exception as e:
                    logger.debug(f"Cost tracking failed: {e}")

            return (reply, False, meta["context_docs"], meta["turn_topics"],
                    meta["recommended_products"], meta["active_phase"],
                    meta.get("model_override"), meta.get("config_override"),
                    meta.get("past_memories", []), _mode, _analysis)

        # Fallback (no LLM)
        return (
            "I'm here with you. Could you tell me a little more about what feels most heavy right now?",
            False,
            [],
            meta["turn_topics"],
            [],
            ConversationPhase.LISTENING,
            None,
            None,
            [],
            _mode,
            _analysis,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    async def _retrieve_and_validate(
        self, session, message: str, analysis: Dict, intent, phase_label: str,
    ) -> tuple:
        """Shared RAG retrieval + context validation for both guidance and listening phases."""
        from services.context_validator import get_context_validator
        ctx_validator = get_context_validator()

        search_query = self._build_listening_query(message, session.memory)
        _type_exclusions = self._get_doc_type_exclusions(intent)
        _dq = session.dharmic_query
        _scripture_filter = None
        if settings.DHARMIC_QUERY_RAG_ENABLED and _dq:
            _scripture_filter = getattr(_dq, 'allowed_scriptures', None)

        search_kwargs = dict(
            scripture_filter=_scripture_filter, language="en",
            top_k=settings.RETRIEVAL_TOP_K, intent=analysis.get("intent"),
            min_score=settings.MIN_SIMILARITY_SCORE,
            doc_type_filter=_type_exclusions,
            life_domain=analysis.get("life_domain"),
            query_variants=analysis.get("query_variants"),
        )

        judge = get_retrieval_judge()
        if judge.available:
            context_docs = await judge.enhanced_retrieve(
                query=search_query, intent_analysis=analysis,
                rag_pipeline=self.rag_pipeline, search_kwargs=search_kwargs,
            )
        else:
            context_docs = await self.rag_pipeline.search(query=search_query, **search_kwargs)

        # Compute allowed_scriptures from dharmic_query or life_domain
        _life_domain = analysis.get("life_domain", "")
        if session.dharmic_query and getattr(session.dharmic_query, 'allowed_scriptures', None):
            _allowed_scriptures = session.dharmic_query.allowed_scriptures
        elif _life_domain and _life_domain not in ("unknown", ""):
            from services.context_synthesizer import LIFE_DOMAIN_TO_SCRIPTURES
            _allowed_scriptures = LIFE_DOMAIN_TO_SCRIPTURES.get(_life_domain, None)
        else:
            _allowed_scriptures = None

        context_docs = ctx_validator.validate(
            docs=context_docs, intent=intent, allowed_scriptures=_allowed_scriptures,
            temple_interest=getattr(session.memory.story, 'temple_interest', False),
            query=message,
            min_score=settings.MIN_SIMILARITY_SCORE,
            max_per_source=settings.MAX_DOCS_PER_SOURCE,
            max_docs=settings.RERANK_TOP_K,
        )

        _top_score = max((get_doc_score(d) for d in context_docs), default=0)
        logger.info(
            f"RAG_OBS phase={phase_label} query='{search_query[:50]}' "
            f"validated={len(context_docs)} top_score={_top_score:.3f} "
            f"sources={[d.get('scripture','?') for d in context_docs]} "
            f"intent={analysis.get('intent')} domain={analysis.get('life_domain')}"
        )
        return context_docs, _top_score

    def _get_doc_type_exclusions(self, intent: Optional[str]) -> Optional[List[str]]:
        """
        Return doc type strings to exclude from RAG for the given intent.
        Returns None if no exclusions apply.
        """
        if not intent:
            return None
        exclusions = {
            IntentType.EXPRESSING_EMOTION: ["temple"],
            IntentType.OTHER: ["temple"],
            IntentType.PRODUCT_SEARCH: ["scripture"],
            IntentType.GREETING: ["temple", "scripture"],
        }
        return exclusions.get(intent)

    def _build_listening_query(
        self, message: str, memory: ConversationMemory
    ) -> str:
        """
        Build a semantically-rich search query for the RAG pipeline.
        Prioritises the CURRENT turn message, then enriches with confirmed
        memory signals (emotion, domain, deity) for better recall.
        Falls back to the raw message if no memory context is available.
        """
        parts = [message.strip()]
        story = memory.story

        # Enrich with confirmed emotional context
        if story.emotional_state and story.emotional_state not in ("unknown", "neutral", ""):
            parts.append(f"feeling {story.emotional_state}")

        # Enrich with the confirmed life domain
        if story.life_area and story.life_area not in ("unknown", ""):
            parts.append(f"life domain: {story.life_area}")

        # Enrich with deity if user has expressed one
        if story.preferred_deity:
            parts.append(f"deity: {story.preferred_deity}")

        # Enrich with the core spiritual/ritual entity if extracted
        if story.primary_concern and len(story.primary_concern) > 10:
            parts.append(story.primary_concern[:100])

        # Enrich with dharmic concepts for better embedding alignment
        if memory.relevant_concepts:
            parts.append("concepts: " + " ".join(memory.relevant_concepts[:3]))

        enriched = " | ".join(parts)[:400]  # Cap for embedding efficiency
        logger.debug(f"Enriched RAG query: {enriched[:120]}")
        return enriched

    async def reconstruct_memory(self, session: SessionState, history: list, snapshot: Optional[Dict] = None) -> None:
        """Reconstruct high-level memory from historical messages and optional snapshot"""
        if snapshot:
            from models.memory_context import ConversationMemory
            session.memory = ConversationMemory.from_dict(snapshot)
            logger.info("Initialized memory from persistent snapshot")

        if not history:
            return
            
        logger.info(f"Reconstructing/Refining deep memory from {len(history)} past messages...")
        
        user_msg_count = 0
        # If we had a snapshot, we might want to only process new messages, 
        # but for safety we re-process to ensure consistency if the history is short.
        # However, to avoid double-counting signals from the snapshot, 
        # we only process messages if memory was fresh.
        
        for msg in history:
            if msg.get("role") == "user":
                user_msg_count += 1
                if not snapshot: # Only auto-extract if we didn't have a snapshot
                    self._update_memory(session.memory, session, msg.get("content", ""))
        
        session.turn_count = user_msg_count
        logger.info(f"Memory reconstruction complete. User messages: {user_msg_count}. Story concern: {session.memory.story.primary_concern[:50]}...")

    def _update_memory(self, memory: ConversationMemory, session: SessionState, text: str) -> List[str]:
        """Extract signals and update narrative story. Delegates to memory_updater module."""
        return _update_memory_impl(memory, session, text)



# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_companion_engine: Optional[CompanionEngine] = None


def get_companion_engine() -> CompanionEngine:
    global _companion_engine
    if _companion_engine is None:
        _companion_engine = CompanionEngine()
    return _companion_engine
