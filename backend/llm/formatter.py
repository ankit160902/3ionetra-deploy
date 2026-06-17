"""
Response Formatter, Reformatter, and Query Refiner
Uses AWS Bedrock (Claude via Converse API).
"""

import logging
from typing import Optional

from config import settings
from llm.bedrock import bedrock_generate

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Response Formatter (high‑intelligence restructuring)
# ------------------------------------------------------------------

class ResponseFormatter:
    def __init__(self):
        self.available = True
        logger.info("ResponseFormatter ready (Bedrock)")

    async def reformulate_response(
        self,
        original_response: str,
        user_query: str,
        context_verses: str
    ) -> str:
        prompt = f"""
You are a wise Bhagavad Gita teacher.

USER QUESTION:
{user_query}

AVAILABLE SCRIPTURE:
{context_verses}

ROUGH RESPONSE:
{original_response}

Rebuild the response using this structure:

1. Brief acknowledgment
2. Relevant Verse (optional)
3. One sentence explanation
4. Gentle question

Max Length: 75 words.

Rules:
- Plain text
- Short paragraphs
- No markdown
"""

        try:
            response = bedrock_generate(
                prompt, model=settings.GEMINI_MODEL,
                temperature=settings.RESPONSE_TEMPERATURE, max_tokens=settings.RESPONSE_MAX_TOKENS)
            return response.text.strip()
        except Exception:
            logger.exception("Bedrock formatter failed")
            raise RuntimeError("LLM formatter unavailable")


# ------------------------------------------------------------------
# Response Reformatter (light rewrite only)
# ------------------------------------------------------------------

class ResponseReformatter:
    def __init__(self, api_key: str | None = None):
        # api_key retained for signature compatibility; Bedrock uses IAM auth.
        self.available = True
        logger.info("✅ ResponseReformatter ready (Bedrock)")

    async def reformulate_response(
        self,
        original_response: str,
        user_query: str,
        context_verses: str,
    ) -> str:
        if not self.available:
            return original_response

        prompt = f"""
You are a compassionate Sanatan Dharma guide.

USER QUESTION:
{user_query}

SCRIPTURAL CONTEXT:
{context_verses}

RAW RESPONSE:
{original_response}

Rewrite the response to be:
- empathetic
- clear
- grounded in Sanatan wisdom
- calm and reassuring

Do not repeat verses verbatim unless necessary.
"""

        try:
            response = bedrock_generate(
                prompt, model=settings.GEMINI_MODEL,
                temperature=settings.RESPONSE_TEMPERATURE, max_tokens=settings.RESPONSE_MAX_TOKENS)
            return response.text.strip()
        except Exception:
            return original_response


# ------------------------------------------------------------------
# Query Refiner (RAG search optimization)
# ------------------------------------------------------------------

class QueryRefiner:
    def __init__(self, api_key: str | None = None):
        # api_key retained for signature compatibility; Bedrock uses IAM auth.
        self.available = True
        logger.info("✅ QueryRefiner ready (Bedrock)")

    async def refine_query(self, query: str, language: str = "en") -> str:
        if not self.available or len(query.split()) < 3:
            return query

        prompt = f"""
Convert the following user input into a concise spiritual search query
for Sanatan Dharma / Bhagavad Gita context.

User input:
{query}

Return only 3–6 keyword phrase.
"""

        try:
            response = bedrock_generate(
                prompt, model=settings.GEMINI_FAST_MODEL,
                temperature=settings.QUERY_REFINE_TEMPERATURE if hasattr(settings, "QUERY_REFINE_TEMPERATURE") else 0.1,
                max_tokens=64)
            return response.text.strip()
        except Exception:
            return query


# ------------------------------------------------------------------
# Singletons
# ------------------------------------------------------------------

_formatter: Optional[ResponseFormatter] = None
_reformatter: Optional[ResponseReformatter] = None
_refiner: Optional[QueryRefiner] = None


def get_formatter() -> ResponseFormatter:
    global _formatter
    if _formatter is None:
        _formatter = ResponseFormatter()
    return _formatter


def get_reformatter(api_key: str | None = None) -> ResponseReformatter:
    global _reformatter
    if _reformatter is None:
        _reformatter = ResponseReformatter(api_key)
    return _reformatter


def get_refiner(api_key: str | None = None) -> QueryRefiner:
    global _refiner
    if _refiner is None:
        _refiner = QueryRefiner(api_key)
    return _refiner
