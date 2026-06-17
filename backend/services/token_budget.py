"""Adaptive token budget calculator.

Single source of truth for max_output_tokens. Uses the LLM's expected_length
estimate + conversation phase to set generous ceilings.

The prompt controls actual response length. This only prevents runaway generation.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from config import settings
from models.session import ConversationPhase, IntentType

logger = logging.getLogger(__name__)

# Ceiling per expected_length category
CEILING_MAP = {
    "brief": 512,       # greetings, closures, acknowledgments
    "moderate": 2048,   # emotional shares, simple questions, mantra + meaning
    "detailed": 4096,   # guidance, how-to, itineraries, explanations
    "full_text": 8192,  # chalisa, stotra, complete prayers, step-by-step
}

# Intents that should never be budget-starved
_MIN_BUDGET_INTENTS = frozenset({
    IntentType.ASKING_INFO,
    IntentType.ASKING_PANCHANG,
    IntentType.SEEKING_GUIDANCE,
})

# Default ceiling when no expected_length is provided
DEFAULT_CEILING = CEILING_MAP["moderate"]


@dataclass
class TokenBudget:
    """Token budget decision."""
    ceiling: int
    reason: str


def calculate_budget(
    analysis: Optional[Dict] = None,
    phase: Optional[ConversationPhase] = None,
) -> TokenBudget:
    """Calculate adaptive token ceiling from intent analysis and phase.

    Priority:
    1. Closure phase → always brief (512)
    2. LLM's expected_length estimate → maps to ceiling
    3. Intent-based safety floor → explicit requests get at least 1024
    4. Default → 1024 (moderate)

    The prompt controls actual response length. This ceiling is a safety net
    that prevents mid-sentence truncation without wasting tokens.
    """
    if analysis is None:
        analysis = {}

    expected_length = analysis.get("expected_length", "moderate")
    intent = analysis.get("intent")

    # Normalize intent to enum
    if isinstance(intent, str):
        try:
            intent = IntentType(intent)
        except ValueError:
            intent = IntentType.OTHER

    # Phase override: closure is always brief
    if phase == ConversationPhase.CLOSURE:
        return TokenBudget(ceiling=512, reason="closure_phase")

    # Primary: use LLM's length estimate
    ceiling = CEILING_MAP.get(expected_length, DEFAULT_CEILING)

    # Cap long responses in guidance phase — conversational guidance should be concise.
    # full_text is NOT capped: when expected_length="full_text" the intent classifier
    # already identified an explicit full-prayer/stotra request (chalisa, tandav, etc.).
    # Devanagari text tokenises at 2-3 tokens/char, so 8192 ceiling is needed.
    if phase == ConversationPhase.GUIDANCE:
        if expected_length == "detailed":
            ceiling = min(ceiling, 2048)

    # Safety floor: explicit requests should never be budget-starved
    if intent in _MIN_BUDGET_INTENTS:
        ceiling = max(ceiling, 1024)

    # Log for observability
    logger.info(
        f"TOKEN_BUDGET | ceiling={ceiling} "
        f"reason=length={expected_length} intent={intent} phase={phase}"
    )

    return TokenBudget(
        ceiling=ceiling,
        reason=f"length={expected_length}",
    )
