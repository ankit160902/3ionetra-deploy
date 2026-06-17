"""Declarative conversation phase state machine.

Replaces the 87-line _assess_readiness() and inline if/else readiness
logic in CompanionEngine with a formal pytransitions FSM.

Thresholds (cooldown, distress listen turns) are read from
``config.settings`` rather than module constants so the FSM has a single
source of truth and operations can tune them without code changes.

The emotion / topic / keyword sets below are *behavioral heuristics* applied
to LLM-classified output (IntentAgent's emotion + topics). They are not
domain-knowledge data tables. A long-term migration would have the
IntentAgent emit a structured ``severity`` field directly so the FSM doesn't
need any local emotion mapping at all — tracked separately.

Usage:
    fsm = ConversationFSM(session)
    is_ready, trigger = fsm.evaluate(analysis, turn_topics)
"""
import logging
from typing import Dict, List, Tuple

from transitions import Machine

from config import settings
from models.session import SessionState, IntentType

logger = logging.getLogger(__name__)

# Emotions requiring extra listening turns before guidance
DISTRESS_EMOTIONS = frozenset({
    "shame", "grief", "guilt", "fear", "humiliation", "trauma", "panic",
})

# High-intensity emotions used inside _assess_readiness (broader set)
HIGH_INTENSITY_EMOTIONS = frozenset({
    "sadness", "anger", "anxiety", "hopelessness", "grief", "despair",
    "loneliness", "frustration",
})

# Urgent keywords that bypass oscillation cooldown
URGENT_KEYWORDS = frozenset({
    "please guide", "suggest me", "tell me what",
    "give me some", "show me", "what should i",
})

# Explicit topics that bypass turn threshold entirely
EXPLICIT_TOPICS = frozenset({"Verse Request", "Product Inquiry"})

# Topics treated as guidance asks (need min turns)
GUIDANCE_TOPICS = frozenset({"Routine Request", "Puja Guidance", "Diet Plan"})

# Planetary / astrological keywords — unambiguous dharmic context, high readiness
PLANETARY_KEYWORDS = frozenset({
    "shani", "saturn", "rahu", "ketu", "mangal", "guru", "brihaspati",
    "budh", "shukra", "surya", "chandra", "graha", "gochara",
    "nakshatra", "rashi", "kundali", "lagna", "dasha", "antardasha",
    "mahadasha", "sadesati", "sade sati", "dhrishti", "jyotish",
    "planetary", "vedic astrology", "kundli", "janma",
})


class ConversationFSM:
    """Conversation phase state machine.

    States: LISTENING, GUIDANCE, CLOSURE
    Evaluates intent analysis + session state to decide phase transitions.
    """

    states = ["LISTENING", "GUIDANCE", "CLOSURE"]

    def __init__(self, session: SessionState):
        self.session = session
        self._analysis: Dict = {}
        self._turn_topics: List[str] = []
        self._last_message: str = ""
        self._trigger_reason: str = "listening"

        # Map current session phase to FSM state
        initial = "LISTENING"
        if session.phase and session.phase.value == "guidance":
            initial = "GUIDANCE"
        elif session.phase and session.phase.value == "closure":
            initial = "CLOSURE"

        self.machine = Machine(
            model=self,
            states=ConversationFSM.states,
            initial=initial,
            send_event=True,
            auto_transitions=False,
        )

        # Transitions are evaluated in order — first match wins
        # 1. Closure from any state
        self.machine.add_transition(
            trigger="step",
            source=["LISTENING", "GUIDANCE"],
            dest="CLOSURE",
            conditions=["is_closure_intent"],
            before="set_trigger_closure",
        )
        # 1b. Return to LISTENING after guidance — only when the user genuinely
        # needs a listening turn (acute new emotion without dharmic context).
        # When the user continues with dharmic signals (graha, verse request,
        # puja, etc.) the companion stays in GUIDANCE. This enables the
        # "stay-dharmic" mode where guidance is the default after turn 2.
        self.machine.add_transition(
            trigger="step",
            source="GUIDANCE",
            dest="LISTENING",
            conditions=["needs_re_listening"],
            unless=["is_closure_intent"],
            before="set_trigger_back_to_listening",
        )
        # 1c. Stay in GUIDANCE when user continues with dharmic or spiritual context.
        self.machine.add_transition(
            trigger="step",
            source="GUIDANCE",
            dest=None,  # internal transition — no state change
            unless=["is_closure_intent", "needs_re_listening"],
            before="set_trigger_guidance_continue",
        )
        # 2a. Explicit user-initiated request (panchang, product, verse — bypasses all gates)
        # These represent unambiguous user signals where the user has explicitly named
        # what they want. Listening-first does not apply because the user is not venting.
        self.machine.add_transition(
            trigger="step",
            source="LISTENING",
            dest="GUIDANCE",
            conditions=["is_explicit_user_request"],
            unless=["is_closure_intent", "is_greeting_intent"],
            before="set_trigger_explicit",
        )
        # 2b. LLM-inferred direct-answer flag (less reliable signal, needs listening guard)
        # The intent agent's needs_direct_answer flag is heuristic and often fires for
        # emotional vents. Block it for distressed users on early turns.
        self.machine.add_transition(
            trigger="step",
            source="LISTENING",
            dest="GUIDANCE",
            conditions=["is_inferred_direct_answer"],
            unless=["is_closure_intent", "is_greeting_intent", "needs_listening_first"],
            before="set_trigger_explicit",
        )
        # 3. Guidance ask with min turns met
        # Defense-in-depth: even when intent==SEEKING_GUIDANCE, a distressed
        # opener still needs at least one listening turn. The IntentAgent can
        # misclassify emotional vents as guidance asks (e.g. "I lost my
        # mother and can't stop crying" → SEEKING_GUIDANCE), and without
        # this guard the FSM jumps straight to scripture prescription.
        self.machine.add_transition(
            trigger="step",
            source="LISTENING",
            dest="GUIDANCE",
            conditions=["is_guidance_ask", "min_turns_met_for_ask"],
            unless=["is_closure_intent", "is_greeting_intent", "needs_listening_first"],
            before="set_trigger_guidance_ask",
        )
        # 4. Signal-based readiness (score + turns + cooldown)
        # Same emotional guard — high readiness scores can fire on the very
        # first turn for users in acute distress, which is exactly when
        # listening matters most.
        self.machine.add_transition(
            trigger="step",
            source="LISTENING",
            dest="GUIDANCE",
            conditions=["readiness_threshold_met", "min_turns_met_for_signals", "cooldown_passed"],
            unless=["is_closure_intent", "is_greeting_intent", "needs_listening_first"],
            before="set_trigger_signals",
        )
        # 5. Force transition (max turns)
        self.machine.add_transition(
            trigger="step",
            source="LISTENING",
            dest="GUIDANCE",
            conditions=["should_force_transition"],
            unless=["is_closure_intent", "is_greeting_intent"],
            before="set_trigger_force",
        )
        # 6. Stay in LISTENING (no-op fallback, keeps state)
        self.machine.add_transition(
            trigger="step",
            source="LISTENING",
            dest=None,  # internal transition, no state change
            before="set_trigger_listening",
        )
        # 7. Return to listening after guidance
        self.machine.add_transition(
            trigger="return_to_listening",
            source="GUIDANCE",
            dest="LISTENING",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, analysis: Dict, turn_topics: List[str]) -> Tuple[bool, str]:
        """Evaluate whether to transition to GUIDANCE.

        Returns (is_ready_for_wisdom, readiness_trigger).
        """
        self._analysis = analysis
        self._turn_topics = turn_topics

        # Extract last user message for keyword checks
        history = self.session.conversation_history
        if history and history[-1].get("role") == "user":
            self._last_message = history[-1].get("content", "").lower()
        else:
            self._last_message = ""

        self._trigger_reason = "listening"

        # Fire the FSM — first matching transition wins
        self.step()

        is_ready = self.state == "GUIDANCE"

        logger.info(
            f"FSM session={self.session.session_id}: "
            f"state={self.state}, trigger={self._trigger_reason}, "
            f"turns={self.session.turn_count}, "
            f"readiness={self.session.memory.readiness_for_wisdom:.2f}"
        )

        return is_ready, self._trigger_reason

    # ------------------------------------------------------------------
    # Guard conditions (all receive EventData from pytransitions)
    # ------------------------------------------------------------------

    def is_greeting_intent(self, event=None) -> bool:
        """Greetings should ALWAYS stay in LISTENING — never jump to guidance."""
        return self._analysis.get("intent") == IntentType.GREETING

    def is_closure_intent(self, event=None) -> bool:
        return self._analysis.get("intent") == IntentType.CLOSURE

    def needs_listening_first(self, event=None) -> bool:
        """Distressed users on early turns need listening before guidance.

        Used as an ``unless`` guard on transitions 2b (LLM-inferred direct
        answer), 3 (guidance ask), and 4 (signal-based readiness). When a
        user opens with a high-intensity emotion or the IntentAgent flags
        urgency as ``high``/``crisis`` on their first ``MIN_DISTRESS_LISTEN_TURNS``
        turns, the companion holds in LISTENING regardless of any other
        signal that would otherwise fire a guidance transition.

        Explicit user requests (verse request, product search, panchang)
        are not affected because the user has unambiguously named what
        they want. Force transitions are also exempt because they already
        require maximum-clarification turns.
        """
        if self.session.turn_count >= settings.MIN_DISTRESS_LISTEN_TURNS:
            return False
        detected_emotion = (self._analysis.get("emotion") or "").lower()
        urgency = (self._analysis.get("urgency") or "").lower()
        # Distress can come from either an emotion classification or an
        # explicit urgency signal — both are LLM-derived, no local lookup.
        if urgency in {"high", "crisis"}:
            return True
        return detected_emotion in HIGH_INTENSITY_EMOTIONS

    def is_explicit_user_request(self, event=None) -> bool:
        """User explicitly named what they want — bypasses listening turns.

        These are unambiguous user signals: panchang query, product search,
        verse request, product inquiry, or LLM-detected product recommendation
        intent. None of these are emotional vents, so listening-first does
        not apply.
        """
        intent = self._analysis.get("intent")
        if intent in (IntentType.ASKING_PANCHANG, IntentType.PRODUCT_SEARCH):
            return True
        if any(t in EXPLICIT_TOPICS for t in self._turn_topics):
            return True
        if self._analysis.get("recommend_products", False):
            return True
        return False

    def is_inferred_direct_answer(self, event=None) -> bool:
        """LLM-inferred flag that the user wants a direct answer.

        This flag is heuristic and often fires for emotional messages where
        the user is venting rather than asking a procedural question. The
        `needs_listening_first` guard on the corresponding transition prevents
        this from short-circuiting the listening phase for distressed users.
        """
        return self._analysis.get("needs_direct_answer", False)

    def is_explicit_request(self, event=None) -> bool:
        """Backwards-compatible composite predicate.

        Retained so existing call sites (analytics, debug logging, tests)
        continue to work. Prefer the more specific
        `is_explicit_user_request` and `is_inferred_direct_answer` for new
        callers.
        """
        return self.is_explicit_user_request() or self.is_inferred_direct_answer()

    def is_guidance_ask(self, event=None) -> bool:
        intent = self._analysis.get("intent")
        if intent in (IntentType.SEEKING_GUIDANCE, IntentType.ASKING_INFO):
            return True
        if any(t in GUIDANCE_TOPICS for t in self._turn_topics):
            return True
        return False

    def min_turns_met_for_ask(self, event=None) -> bool:
        """Min turns gate for guidance asks (distress-aware)."""
        detected_emotion = (self._analysis.get("emotion") or "").lower()
        min_turns = self.session.min_clarification_turns if detected_emotion in DISTRESS_EMOTIONS else 2
        return self.session.turn_count >= min_turns

    def min_turns_met_for_signals(self, event=None) -> bool:
        """Min turns gate for signal-based transitions (uses _assess_readiness logic)."""
        emotional_state = getattr(self.session.memory.story, "emotional_state", "")
        requires_extra = emotional_state in HIGH_INTENSITY_EMOTIONS

        # Replicate the tiered min-turns logic from _assess_readiness
        msg_lower = self._last_message
        explicit_spiritual_keywords = [
            "mantra", "shloka", "verse", "prayer", "pooja", "puja", "vrat",
            "chant", "suggest a practice", "spiritual help", "koi upay",
            "mantra batao", "kuch batao", "what should i chant",
            "give me a mantra", "suggest me", "guide me spiritually",
            # Planetary / astrological signals — unambiguous dharmic context
            *PLANETARY_KEYWORDS,
        ]
        guidance_phrases = [
            "what should i do", "what can i do", "how do i fix",
            "help me with", "give me advice", "suggest a solution",
            "way out", "way to deal", "how to overcome",
        ]

        if any(kw in msg_lower for kw in explicit_spiritual_keywords):
            min_turns = 2 if requires_extra else 1
        elif any(phrase in msg_lower for phrase in guidance_phrases):
            min_turns = 4 if requires_extra else 3
        else:
            min_turns = 3 if requires_extra else 2

        return self.session.turn_count >= min_turns

    def cooldown_passed(self, event=None) -> bool:
        """Oscillation cooldown: N turns since last guidance, unless urgent.

        N comes from ``settings.GUIDANCE_OSCILLATION_COOLDOWN`` so the value
        is editable without code change.
        """
        last_guidance = getattr(self.session, "last_guidance_turn", 0) or 0
        if last_guidance <= 0:
            return True

        turns_since = self.session.turn_count - last_guidance
        if turns_since >= settings.GUIDANCE_OSCILLATION_COOLDOWN:
            return True

        # Check urgent keyword override
        if any(kw in self._last_message for kw in URGENT_KEYWORDS):
            logger.info(
                f"FSM session={self.session.session_id}: urgent request bypasses cooldown"
            )
            return True

        return False

    def readiness_threshold_met(self, event=None) -> bool:
        return self.session.memory.readiness_for_wisdom >= 0.50

    def needs_re_listening(self, event=None) -> bool:
        """True only when the user genuinely needs a listening response after guidance.

        After turn 2, the companion defaults to staying in GUIDANCE unless
        the user opens with an acute new emotional crisis that has no dharmic
        context. Dharmic signals (planetary, verse, puja, guidance ask) always
        keep the companion in GUIDANCE regardless of emotional intensity.
        """
        # Very early turns always need listening
        if self.session.turn_count < 2:
            return True

        # Any dharmic signal → stay in GUIDANCE, never re-listen
        has_dharmic_signal = (
            self._analysis.get("needs_direct_answer", False)
            or self._analysis.get("intent") in (
                IntentType.SEEKING_GUIDANCE, IntentType.ASKING_INFO,
                IntentType.ASKING_PANCHANG, IntentType.PRODUCT_SEARCH,
            )
            or any(t in EXPLICIT_TOPICS | GUIDANCE_TOPICS for t in self._turn_topics)
            or any(kw in self._last_message for kw in PLANETARY_KEYWORDS)
            or "Planetary Guidance" in self._turn_topics
        )
        if has_dharmic_signal:
            return False

        # Re-listen only for acute new emotional crisis without dharmic context
        urgency = (self._analysis.get("urgency") or "").lower()
        emotion = (self._analysis.get("emotion") or "").lower()
        return urgency == "crisis" or emotion in ("grief", "despair")

    def should_force_transition(self, event=None) -> bool:
        return self.session.should_force_transition()

    # ------------------------------------------------------------------
    # Trigger reason callbacks (called via `before=`)
    # ------------------------------------------------------------------

    def set_trigger_closure(self, event=None):
        self._trigger_reason = "closure"

    def set_trigger_explicit(self, event=None):
        self._trigger_reason = "explicit_request"

    def set_trigger_guidance_ask(self, event=None):
        self._trigger_reason = "user_asked_for_guidance"

    def set_trigger_signals(self, event=None):
        self._trigger_reason = "signals_accumulated"

    def set_trigger_force(self, event=None):
        self._trigger_reason = "forced_transition"

    def set_trigger_listening(self, event=None):
        self._trigger_reason = "listening"

    def set_trigger_back_to_listening(self, event=None):
        self._trigger_reason = "post_guidance_listening"

    def set_trigger_guidance_continue(self, event=None):
        self._trigger_reason = "guidance_continue"
