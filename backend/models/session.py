"""
Session models for conversation state management
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import uuid


class ConversationPhase(str, Enum):
    """Phases of the conversation flow"""
    CLARIFICATION = "clarification"
    SYNTHESIS = "synthesis"
    ANSWERING = "answering"
    LISTENING = "listening"
    GUIDANCE = "guidance"
    CLOSURE = "closure"


class SignalType(str, Enum):
    """Types of signals collected during conversation"""
    EMOTION = "emotion"
    TRIGGER = "trigger"
    LIFE_DOMAIN = "life_domain"
    MENTAL_STATE = "mental_state"
    USER_GOAL = "user_goal"
    INTENT = "intent"
    SEVERITY = "severity"


class IntentType(str, Enum):
    """Intent categories from the IntentAgent classifier"""
    GREETING = "GREETING"
    SEEKING_GUIDANCE = "SEEKING_GUIDANCE"
    EXPRESSING_EMOTION = "EXPRESSING_EMOTION"
    ASKING_INFO = "ASKING_INFO"
    ASKING_PANCHANG = "ASKING_PANCHANG"
    PRODUCT_SEARCH = "PRODUCT_SEARCH"
    CLOSURE = "CLOSURE"
    OTHER = "OTHER"


@dataclass
class Signal:
    """Represents a collected signal with its value"""
    signal_type: SignalType
    value: str
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "signal_type": self.signal_type.value,
            "value": self.value,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Signal':
        return cls(
            signal_type=SignalType(data["signal_type"]),
            value=data["value"],
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class SessionState:
    """
    Represents the state of a conversation session.
    Tracks signals, conversation history, and phase transitions.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    phase: ConversationPhase = ConversationPhase.LISTENING
    turn_count: int = 0
    signals_collected: Dict[SignalType, Signal] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Thresholds for phase transition
    min_signals_threshold: int = 4
    min_clarification_turns: int = 3
    max_clarification_turns: int = 6
    
    # Oscillation control
    last_guidance_turn: int = -1  # Turn number when guidance was last given
    
    # Returning user detection
    is_returning_user: bool = False  # True if past conversation history was loaded

    # Proactive product suggestion anti-spam guard
    last_proactive_product_turn: int = -1

    # Session-level dedup: tracks which product IDs have been shown
    shown_product_ids: Set[str] = field(default_factory=set)

    # Last 5 products recommended this session — surfaced to the LLM so it
    # can answer follow-ups like "how do I use that mala?" with awareness
    # of the actual item that was suggested. FIFO capped at 5 in
    # ProductRecommender._record_shown. See profile_builder and
    # llm.service._build_prompt for the consumer side.
    recent_products: List[Dict[str, Any]] = field(default_factory=list)

    # Product recommendation throttling — split counters (Apr 2026 redesign)
    product_event_count: int = 0            # Proactive/contextual product events (cap 3)
    explicit_product_count: int = 0         # Explicit user requests (cap 10, generous)
    product_rejection_turn: int = -1        # Turn when user last rejected products
    product_rejection_count: int = 0        # Total rejections
    user_dismissed_products: bool = False   # Hard kill: user said "stop products"

    # Crisis progression — incremented every time the user trips a safety
    # gate. Used by services.crisis_response_composer to choose a different
    # response variant on consecutive crisis turns so the bot does not
    # return a byte-identical canned reply across turns.
    crisis_turn_count: int = 0
    crisis_resolved: bool = False  # Set True after de-escalation; tells check_crisis_signals to skip history scan

    # Why the phase transition happened — injected into LLM phase prompt
    readiness_trigger: str = "listening"

    # Track what practices Mitra suggested (for acceptance-based product inference)
    # Each entry: {"turn": int, "practice": str, "product_keywords": ["rudraksha mala", "japa mala"]}
    # FIFO capped at 3 entries
    last_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    # Track suggested verses/mantras for diversity (anti-repetition)
    # Each entry: {"turn": int, "mantras": ["Om Namah Shivaya"], "references": ["BG 2.47"]}
    suggested_verses: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "phase": self.phase.value,
            "turn_count": self.turn_count,
            "signals_collected": {st.value: s.to_dict() for st, s in self.signals_collected.items()},
            "conversation_history": self.conversation_history[-30:],
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "min_signals_threshold": self.min_signals_threshold,
            "min_clarification_turns": self.min_clarification_turns,
            "max_clarification_turns": self.max_clarification_turns,
            "last_guidance_turn": self.last_guidance_turn,
            "is_returning_user": self.is_returning_user,
            "last_proactive_product_turn": self.last_proactive_product_turn,
            "shown_product_ids": list(self.shown_product_ids),
            "recent_products": self.recent_products,
            "product_event_count": self.product_event_count,
            "explicit_product_count": self.explicit_product_count,
            "product_rejection_turn": self.product_rejection_turn,
            "product_rejection_count": self.product_rejection_count,
            "user_dismissed_products": self.user_dismissed_products,
            "crisis_turn_count": self.crisis_turn_count,
            "crisis_resolved": self.crisis_resolved,
            "last_suggestions": self.last_suggestions,
            "readiness_trigger": self.readiness_trigger,
            "suggested_verses": self.suggested_verses,
            "memory": self.memory.to_dict() if self.memory else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionState':
        from models.memory_context import ConversationMemory
        
        signals = {}
        for k, v in data.get("signals_collected", {}).items():
            try:
                signals[SignalType(k)] = Signal.from_dict(v)
            except ValueError:
                continue

        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            phase=ConversationPhase(data["phase"]),
            turn_count=data["turn_count"],
            signals_collected=signals,
            conversation_history=data.get("conversation_history", []),
            created_at=data["created_at"] if isinstance(data["created_at"], datetime) else datetime.fromisoformat(data["created_at"]),
            last_activity=data["last_activity"] if isinstance(data["last_activity"], datetime) else datetime.fromisoformat(data["last_activity"]),
            min_signals_threshold=data.get("min_signals_threshold", 4),
            min_clarification_turns=data.get("min_clarification_turns", 3),
            max_clarification_turns=data.get("max_clarification_turns", 6),
            last_guidance_turn=data.get("last_guidance_turn", -1),
            is_returning_user=data.get("is_returning_user", False),
            last_proactive_product_turn=data.get("last_proactive_product_turn", -1),
            shown_product_ids=set(data.get("shown_product_ids", [])),
            recent_products=data.get("recent_products", []),
            product_event_count=data.get("product_event_count", 0),
            explicit_product_count=data.get("explicit_product_count", 0),
            product_rejection_turn=data.get("product_rejection_turn", -1),
            product_rejection_count=data.get("product_rejection_count", 0),
            user_dismissed_products=data.get("user_dismissed_products", False),
            crisis_turn_count=data.get("crisis_turn_count", 0),
            crisis_resolved=data.get("crisis_resolved", False),
            last_suggestions=data.get("last_suggestions", []),
            readiness_trigger=data.get("readiness_trigger", "listening"),
            suggested_verses=data.get("suggested_verses", []),
            memory=ConversationMemory.from_dict(data.get("memory", {}))
        )
        return session

    # Memory context for rich understanding
    memory: Optional[Any] = field(default=None)

    # Synthesized query for RAG
    dharmic_query: Optional[Any] = field(default=None)

    def __post_init__(self):
        # Initialize memory if not provided
        if self.memory is None:
            from models.memory_context import ConversationMemory
            self.memory = ConversationMemory()

    def add_signal(self, signal_type: SignalType, value: str, confidence: float = 1.0) -> None:
        """Add or update a signal"""
        self.signals_collected[signal_type] = Signal(
            signal_type=signal_type,
            value=value,
            confidence=confidence
        )

    def get_signal(self, signal_type: SignalType) -> Optional[Signal]:
        """Get a specific signal"""
        return self.signals_collected.get(signal_type)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_activity = datetime.utcnow()

    def get_signals_summary(self) -> Dict[str, str]:
        """Get a summary of collected signals as simple key-value pairs"""
        return {
            signal_type.value: signal.value
            for signal_type, signal in self.signals_collected.items()
        }

    def should_force_transition(self) -> bool:
        """Check if we should force transition to answering phase"""
        from config import settings
        # Always read live config value — session.max_clarification_turns may be stale
        # for sessions restored from Redis that were created before a config change.
        _force_at = settings.MAX_CLARIFICATION_TURNS

        # If we recently gave guidance, DO NOT force transition.
        # We need at least 2 turns of listening between wisdom.
        if self.last_guidance_turn > 0 and (self.turn_count - self.last_guidance_turn) < 2:
            return False

        # Force transition after max turns (absolute cap)
        if self.turn_count >= _force_at:
            # If we haven't given guidance for a while, allow it
            if self.last_guidance_turn == -1 or (self.turn_count - self.last_guidance_turn) > 3:
                return True

        # Check if enough signals have been collected after min turns
        if self.turn_count >= self.min_clarification_turns:
            if len(self.signals_collected) >= self.min_signals_threshold:
                 # Again, ensure cooldown
                if self.last_guidance_turn == -1 or (self.turn_count - self.last_guidance_turn) > 2:
                    return True

        return False

    @property
    def memory_readiness(self) -> float:
        """Get the memory's readiness for wisdom score"""
        if self.memory and hasattr(self.memory, 'readiness_for_wisdom'):
            return self.memory.readiness_for_wisdom
        return 0.0

    def is_ready_for_transition(self) -> bool:
        """Check if session is ready to transition to answering phase"""
        if self.memory_readiness >= 0.7:
            return True

        # Check turn count
        if self.turn_count >= self.min_clarification_turns:
            # Check signal count
            if len(self.signals_collected) >= self.min_signals_threshold:
                return True

        return False
