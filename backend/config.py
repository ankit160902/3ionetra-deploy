"""
Configuration management for 3ioNetra Spiritual Companion
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List, Optional


class Settings(BaseSettings):
    """Application settings"""

    # ------------------------------------------------------------------
    # API Settings
    # ------------------------------------------------------------------
    API_TITLE: str = "3ioNetra Spiritual Companion API"
    API_VERSION: str = "1.1.3"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    THREAD_POOL_SIZE: int = Field(default=64, env="THREAD_POOL_SIZE")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # ------------------------------------------------------------------
    # LLM Settings (AWS Bedrock — Claude)
    # ------------------------------------------------------------------
    # NOTE: The LLM moved from Google Gemini to AWS Bedrock. The *_MODEL fields
    # below are plain Bedrock model/inference-profile id strings consumed by the
    # bedrock helper — the GEMINI_* names are retained only to avoid touching the
    # ~12 call sites that already read them. Override per-env via the same keys.
    # Defaults target ap-south-1; ENABLE MODEL ACCESS in the Bedrock console.
    # Region: ap-south-1 (Mumbai) — lowest latency for Indian users.
    # apac-regional profiles only offer legacy Sonnet 4 / Haiku 3, so the live
    # Claude models use `global.*` inference profiles (callable from Mumbai,
    # routed to nearest active region). Nova uses the apac-regional profile.
    BEDROCK_REGION: str = Field(default="ap-south-1", env="BEDROCK_REGION")
    GEMINI_MODEL: str = Field(default="global.anthropic.claude-sonnet-4-6", env="GEMINI_MODEL")  # Main responses + vision/OCR — latest Sonnet (best output/latency balance)
    GEMINI_FAST_MODEL: str = Field(default="global.anthropic.claude-haiku-4-5-20251001-v1:0", env="GEMINI_FAST_MODEL")  # Intent, query expansion, quick tasks
    BEDROCK_VIDEO_MODEL: str = Field(default="apac.amazon.nova-pro-v1:0", env="BEDROCK_VIDEO_MODEL")  # Video ingestion (Claude has no video)
    GEMINI_CACHE_TTL: int = Field(default=0, env="GEMINI_CACHE_TTL")  # Disabled — no Bedrock equivalent of Gemini explicit context cache

    # Per-task LLM temperatures
    RESPONSE_TEMPERATURE: float = 0.7
    RESPONSE_MAX_TOKENS: int = 2048
    INTENT_TEMPERATURE: float = 0.1

    # LLM thinking-mode budget — Gemini 2.5/3 models default to thinking ON
    # which (a) adds 5-15s latency, (b) can leak chain-of-thought into responses
    # if SDK part filtering misses a frame. Default 0 disables thinking for the
    # main conversational path. Internal callers (intent classifier, retrieval
    # judge) that genuinely benefit from reasoning can opt in by passing an
    # explicit thinking_budget kwarg to LLMService methods.
    LLM_THINKING_BUDGET_DEFAULT: int = 0

    # ResponseValidator regeneration budget — number of times the response
    # composer is allowed to ask the LLM to rewrite a response that fails a
    # validator check (length, hollow phrase, scratchpad leak, etc.). Hard
    # backstop, not a target. After this many retries the last response is
    # returned with a logged metric.
    LLM_REGENERATION_RETRIES: int = 2  # Apr 2026: bumped from 1 to give 2 chances to avoid hollow phrases after HIGH-severity detection

    # Token budget ceilings (used by TokenBudgetCalculator — adaptive system)
    TOKEN_CEILING_BRIEF: int = 512       # greetings, closures
    TOKEN_CEILING_MODERATE: int = 1024   # emotional shares, simple Qs
    TOKEN_CEILING_DETAILED: int = 2048   # guidance, how-to
    TOKEN_CEILING_FULL_TEXT: int = 4096  # chalisa, stotra, complete prayers

    # Context Budget Manager — prevents response truncation
    PROMPT_TARGET_INPUT_TOKENS: int = 8000   # Max input tokens — lower = faster Gemini response
    PROMPT_SAFETY_MARGIN: int = 500          # Buffer for model overhead
    PROMPT_MIN_HISTORY_MESSAGES: int = 4     # Never trim below 4 history messages
    PROMPT_MIN_RAG_DOCS: int = 1             # Never trim below 1 RAG doc

    QUERY_TRANSLATE_TEMPERATURE: float = 0.1
    QUERY_EXPAND_TEMPERATURE: float = 0.3
    QUERY_SUMMARIZE_TEMPERATURE: float = 0.1
    HYDE_TEMPERATURE: float = 0.5
 
    # ------------------------------------------------------------------
    # Embedding Settings
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"  # benchmark-validated: instruct variant scored lower (0.784 vs 0.840)
    EMBEDDING_DIM: int = 1024
    EMBEDDING_ONNX_ENABLED: bool = Field(default=False, env="EMBEDDING_ONNX_ENABLED")  # ONNX INT8 for ~30-50ms faster embedding
    EMBEDDING_ONNX_PATH: str = Field(default="./models/e5-large-onnx", env="EMBEDDING_ONNX_PATH")  # exported ONNX model path
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # ------------------------------------------------------------------
    # Vector DB (Qdrant)
    # ------------------------------------------------------------------
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "sanatan_scriptures"
    VECTOR_DB_PATH: str = "./data/vector_db"

    # ------------------------------------------------------------------
    # RAG Settings
    # ------------------------------------------------------------------
    RETRIEVAL_TOP_K: int = 5          # benchmark: Hit@5 = Hit@7, no accuracy loss
    RERANK_TOP_K: int = 3
    MIN_SIMILARITY_SCORE: float = 0.28
    MAX_DOCS_PER_SOURCE: int = 2
    RAG_SEARCH_CACHE_TTL: int = Field(default=3600, env="RAG_SEARCH_CACHE_TTL")

    # RAG scoring & fusion
    RERANKER_WEIGHT: float = 0.75           # semantic weight = 1 - this (was 0.7, updated per RAKS report Section 8)
    INTENT_WEIGHT_SCALE: float = 0.3       # how much intent adjusts final score
    SOFT_FLOOR_RATIO: float = 0.75         # min_score * this = soft floor in search()
    CONTEXT_VERSE_SCORE_RATIO: float = 0.5 # parent-child context verse discount
    MIN_CANDIDATE_POOL: int = 15           # reduced from 40 — fewer candidates = faster reranking
    CANDIDATE_POOL_MULTIPLIER: int = 3     # top_k * this = candidate pool target (legacy fallback)
    CANDIDATE_POOL_KEYWORD: int = 10       # simple ASKING_INFO queries (was 25)
    CANDIDATE_POOL_DEFAULT: int = 15       # most queries (was 40)
    CANDIDATE_POOL_THEMATIC: int = 20      # emotional/thematic queries (was 50)
    CANDIDATE_POOL_COMPARATIVE: int = 25   # comparative queries needing broad retrieval (was 60)
    CURATED_SLOT_LIMIT: int = 4            # max curated docs in slot reservation (was 10, reduced for latency)
    CURATED_VIABILITY_THRESHOLD: float = 0.5  # min score for curated doc injection
    EXPAND_TOP_N: int = 2                  # how many top docs get parent-child expansion
    CURATED_FLOOR: float = 0.35
    CURATED_RATIO: float = 0.6
    TRADITION_BONUS: float = 0.05
    SECTION_CHUNKS_ENABLED: bool = Field(default=True, env="SECTION_CHUNKS_ENABLED")
    SPLADE_ENABLED: bool = Field(default=True, env="SPLADE_ENABLED")
    # When True AND SPLADE_ENABLED is True, a missing SPLADE index causes
    # a hard startup failure with an actionable error. Use this in
    # production deployments where silent degradation to BM25 would mask
    # a corrupted ingest. Default False so dev environments without an
    # index still boot in degraded mode.
    SPLADE_REQUIRED: bool = Field(default=False, env="SPLADE_REQUIRED")
    SPLADE_MODEL: str = "naver/splade-cocondenser-ensembledistil"
    RERANKER_ENABLED: bool = Field(default=True, env="RERANKER_ENABLED")
    MAX_RERANK_CANDIDATES: int = Field(default=12, env="MAX_RERANK_CANDIDATES")  # cap candidates sent to CrossEncoder

    # Reranker skip — when top candidate is decisive, skip neural reranking
    SKIP_RERANK_THRESHOLD: float = Field(default=0.75, env="SKIP_RERANK_THRESHOLD")
    SKIP_RERANK_GAP: float = Field(default=0.15, env="SKIP_RERANK_GAP")

    # Content validation
    DYNAMIC_RELEVANCE_RATIO: float = 0.5   # top_score * this = dynamic floor
    MIN_CONTENT_LENGTH: int = 10           # min chars for content gate
    RELEVANCE_RATIO_EMOTIONAL: float = 0.40
    RELEVANCE_RATIO_CITATION: float = 0.55
    RELEVANCE_RATIO_COMPARATIVE: float = 0.25
    RELEVANCE_RATIO_GUIDANCE: float = 0.45
    RELEVANCE_RATIO_DEFAULT: float = 0.50
    MEMORY_DEDUP_THRESHOLD: float = 0.85
    MEMORY_MAX_RESULTS: int = 5
    MAX_DOCS_PER_TRADITION: int = 3
    MEMORY_SIMILARITY_THRESHOLD: float = 0.35  # min cosine sim for memory recall

    # Dynamic memory system (Apr 2026 — see 2026-04-12-dynamic-memory-design.md).
    # Scoring weights and thresholds for the new MemoryReader + MemoryExtractor
    # + ReflectionService pipeline. All tunable without code changes.
    MEMORY_HALF_LIFE_DAYS: int = 30                    # recency decay half-life
    MEMORY_IMPORTANCE_FLOOR_THRESHOLD: int = 8         # importance >= this: never fully decays
    MEMORY_IMPORTANCE_FLOOR_VALUE: float = 0.3         # minimum recency floor for important memories
    MEMORY_WEIGHT_RECENCY: float = 0.5                 # Generative Agents used 1.0 — we lower
    MEMORY_WEIGHT_IMPORTANCE: float = 1.0              # slightly stronger than relevance
    MEMORY_WEIGHT_RELEVANCE: float = 1.0               # cosine similarity weight
    MEMORY_EPISODIC_TOP_K: int = 3                     # top-k episodic memories injected per turn
    MEMORY_SCORE_FLOOR: float = 0.4                    # absolute score floor — weak matches aren't shown
    MEMORY_EXTRACTION_TIMEOUT_SECONDS: int = 30        # hard cap on async extraction task
    MEMORY_UPDATER_SIMILAR_K: int = 5                  # top-k similar memories shown to the Mem0 decider
    MEMORY_PROFILE_CACHE_TTL_SECONDS: int = 300        # RelationalProfile Redis cache TTL (5 min per spec §8.2)
    MEMORY_PRUNE_IMPORTANCE_SAFETY_FLOOR: int = 8      # the ONE hardcoded safety floor — prune never touches this

    # Tone families for tone-aware retrieval (spec §10.4). Sensitive memories
    # only surface when the current turn's tone sits in the same family as
    # the memory's tone_marker. Edit this dict to retune without code
    # changes — e.g. move "relief" from recovering → warm if that feels
    # closer. Unknown tones are treated as neutral.
    MEMORY_TONE_FAMILIES: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "heavy": [
                "grief", "despair", "anxiety", "overwhelm", "confusion",
                "shame", "fear", "loneliness", "doubt", "anger",
            ],
            "recovering": ["resolve", "healing", "hope", "relief"],
            "warm": ["gratitude", "joy", "devotion", "curiosity", "peace"],
            "neutral": ["neutral", ""],
        }
    )

    REFLECTION_THRESHOLD: int = 30                     # importance sum that triggers consolidation
    REFLECTION_EPISODIC_WINDOW: int = 20               # how many recent memories reflection reads
    REFLECTION_MODEL: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # Bedrock id — reflection/consolidation
    BACKFILL_CONVERSATION_COUNT: int = 10              # how many prior conversations cold-start backfill processes

    # Semantic Response Cache (saves 5-15s on repeat patterns)
    RESPONSE_CACHE_ENABLED: bool = Field(default=True, env="RESPONSE_CACHE_ENABLED")
    RESPONSE_CACHE_TTL: int = Field(default=21600, env="RESPONSE_CACHE_TTL")  # 6 hours
    RESPONSE_CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.92, env="RESPONSE_CACHE_SIMILARITY_THRESHOLD")

    # HyDE (Hypothetical Document Embedding)
    HYDE_ENABLED: bool = Field(default=True, env="HYDE_ENABLED")
    HYDE_COUNT: int = Field(default=2, env="HYDE_COUNT")
    HYDE_CACHE_TTL: int = Field(default=86400, env="HYDE_CACHE_TTL")

    # Adaptive Fusion Logging
    LOG_FUSION_WEIGHTS: bool = Field(default=True, env="LOG_FUSION_WEIGHTS")

    # Parent-Child Verse Retrieval
    PARENT_CHILD_ENABLED: bool = Field(default=True, env="PARENT_CHILD_ENABLED")
    VERSE_CONTEXT_WINDOW: int = Field(default=1, env="VERSE_CONTEXT_WINDOW")

    # Long Query Summarization
    LONG_QUERY_SUMMARIZATION_ENABLED: bool = Field(default=True, env="LONG_QUERY_SUMMARIZATION_ENABLED")
    LONG_QUERY_THRESHOLD: int = Field(default=20, env="LONG_QUERY_THRESHOLD")

    # Query Expansion — LLM-based expansion for short queries (adds ~800ms latency)
    QUERY_EXPANSION_ENABLED: bool = Field(default=True, env="QUERY_EXPANSION_ENABLED")

    # DharmicQueryObject → RAG pre-filtering
    DHARMIC_QUERY_RAG_ENABLED: bool = Field(default=True, env="DHARMIC_QUERY_RAG_ENABLED")

    # ------------------------------------------------------------------
    # Hybrid RAG / Retrieval Judge
    # ------------------------------------------------------------------
    HYBRID_RAG_ENABLED: bool = Field(default=True, env="HYBRID_RAG_ENABLED")
    JUDGE_MIN_SCORE: int = Field(default=4, env="JUDGE_MIN_SCORE")
    JUDGE_MAX_RETRIES: int = Field(default=1, env="JUDGE_MAX_RETRIES")
    JUDGE_CACHE_TTL: int = Field(default=86400, env="JUDGE_CACHE_TTL")
    GROUNDING_ENABLED: bool = Field(default=True, env="GROUNDING_ENABLED")
    GROUNDING_MIN_CONFIDENCE: float = Field(default=0.7, env="GROUNDING_MIN_CONFIDENCE")

    # ------------------------------------------------------------------
    # Conversation Flow
    # ------------------------------------------------------------------
    MIN_SIGNALS_THRESHOLD: int = 2
    MIN_CLARIFICATION_TURNS: int = 1
    MAX_CLARIFICATION_TURNS: int = 2
    SESSION_TTL_MINUTES: int = 120  # 2hr — prevents mid-conversation expiry on user pauses
    READINESS_POST_GUIDANCE: float = 0.3   # readiness reset after guidance phase

    # FSM thresholds — promoted from conversation_fsm.py module constants so
    # the FSM has a single source of truth and these are tunable without code
    # changes. Used by ConversationFSM transition prerequisites.
    MIN_DISTRESS_LISTEN_TURNS: int = 1          # turns to stay in LISTENING when latest signal severity is HIGH
    GUIDANCE_OSCILLATION_COOLDOWN: int = 0      # turns required between successive GUIDANCE phases
    EMOTIONAL_SEVERITIES_REQUIRE_LISTEN: str = "high,severe,crisis"  # comma-separated severity names that block early guidance

    # ------------------------------------------------------------------
    # Query preprocessing (QueryNormalizer seam)
    # ------------------------------------------------------------------
    # Spell correction was previously implemented inline in rag/pipeline.py
    # using whole-string Levenshtein distance against a dharmic vocabulary.
    # That approach mangled common English words into Sanskrit terms (e.g.
    # "How"→"homa", "Give"→"gita") and biased off-topic queries toward
    # scripture matches. Disabled by default — embedding model + reranker
    # already handle morphology and fuzzy matching at the semantic layer.
    SPELL_CORRECTION_ENABLED: bool = Field(default=False, env="SPELL_CORRECTION_ENABLED")

    # ------------------------------------------------------------------
    # Reranker cache
    # ------------------------------------------------------------------
    # Caches CrossEncoder rerank scores keyed by (query, doc_id) so repeated
    # phrases in long conversations don't re-run the encoder. Backed by the
    # existing Redis cache_service (DB 1).
    RERANKER_CACHE_ENABLED: bool = Field(default=True, env="RERANKER_CACHE_ENABLED")
    RERANKER_CACHE_TTL: int = Field(default=86400, env="RERANKER_CACHE_TTL")  # 24h

    # Emotions where product recommendations are suppressed entirely.
    # PRODUCT_SUPPRESS_EMOTIONS REMOVED (Apr 2026 adaptive architecture).
    # The IntentAgent's `product_signal.intent` field now makes the contextual
    # decision about whether products are appropriate — it sees the emotion,
    # intent, and full message. The hardcoded emotion list was over-blocking
    # (e.g. "I'm anxious about my puja, what items do I need?" was blocked).
    # Crisis-level blocking remains in ProductRecommender (safety, non-negotiable).
    #
    # PRODUCT_GUIDANCE_CONTEXT_ENABLED and PRODUCT_LISTENING_PROACTIVE_ENABLED
    # also removed — they gated Path 3 (proactive inference) which is deleted.
    PRODUCT_MIN_RELEVANCE_SCORE: float = 15.0         # Min combined score to include a product (15 ≈ one name keyword match)
    PRODUCT_RELEVANCE_GAP_RATIO: float = 0.55         # A product must score at least N% of the top item's score
    # The gap ratio is the structural fix for "always returns 3 padded products."
    # Tighter (0.7) = stricter (fewer products); looser (0.4) = more permissive.
    # Drives both search_products and search_by_metadata. Replaces the
    # "fill remaining slots regardless of relevance" fallback that was padding
    # results with weakly-related items. Apr 2026 fix.

    # ------------------------------------------------------------------
    # Safety / Crisis
    # ------------------------------------------------------------------
    ENABLE_CRISIS_DETECTION: bool = True
    CRISIS_HELPLINE_IN: str = (
        "iCall: 9152987821, Vandrevala: 1860-2662-345"
    )

    # ------------------------------------------------------------------
    # Scripture Data Paths
    # ------------------------------------------------------------------
    DATA_DIR: str = "./data"
    SCRIPTURES_DIR: str = "./data/scriptures"
    PROCESSED_DIR: str = "./data/processed"
    TRANSLATION_CACHE_PATH: str = "./data/cache/translations"

    # ------------------------------------------------------------------
    # MongoDB Settings
    # ------------------------------------------------------------------
    MONGODB_URI: str = Field(default="", env="MONGODB_URI")
    DATABASE_NAME: str = Field(default="", env="DATABASE_NAME")
    DATABASE_PASSWORD: str = Field(default="", env="DATABASE_PASSWORD")
    MONGO_MAX_POOL_SIZE: int = Field(default=50, env="MONGO_MAX_POOL_SIZE")
    MONGO_MIN_POOL_SIZE: int = Field(default=5, env="MONGO_MIN_POOL_SIZE")
    MONGO_MAX_IDLE_TIME_MS: int = Field(default=30000, env="MONGO_MAX_IDLE_TIME_MS")
    MONGO_READ_PREFERENCE: str = Field(default="primaryPreferred", env="MONGO_READ_PREFERENCE")
    ENABLE_AUTO_MIGRATIONS: bool = Field(default=True, env="ENABLE_AUTO_MIGRATIONS")
    MAX_CONVERSATIONS_PER_USER: int = Field(default=100, env="MAX_CONVERSATIONS_PER_USER")

    # ------------------------------------------------------------------
    # Redis Settings
    # ------------------------------------------------------------------
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # ------------------------------------------------------------------
    # External API Keys
    # ------------------------------------------------------------------
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    COHERE_API_KEY: str = Field(default="", env="COHERE_API_KEY")
    HUGGINGFACE_TOKEN: str = Field(default="", env="HUGGINGFACE_TOKEN")

    # ------------------------------------------------------------------
    # Evaluation Model Settings
    # ------------------------------------------------------------------
    EVAL_GEMINI_MODEL: str = "gemini-2.5-pro"
    EVAL_CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    EVAL_OPENAI_MODEL: str = "gpt-4o"
    EVAL_GEMINI_FAST_MODEL: str = "gemini-2.5-flash"
    EVAL_CLAUDE_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
    EVAL_OPENAI_MINI_MODEL: str = "gpt-4o-mini"

    # ------------------------------------------------------------------
    # Reranker Settings
    # ------------------------------------------------------------------
    # bge-reranker-v2-m3: multilingual, accurate but heavier (~200ms)
    # BAAI/bge-reranker-base: English-focused, lighter (~50ms) — use if most queries are English/Hindi
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", env="RERANKER_MODEL")
    RERANKER_ONNX_ENABLED: bool = Field(default=False, env="RERANKER_ONNX_ENABLED")  # ONNX for 2-4x faster reranking
    RERANKER_TYPE: str = "cross-encoder"  # "cross-encoder" or "api"

    # ------------------------------------------------------------------
    # Model Routing Settings
    # ------------------------------------------------------------------
    MODEL_ROUTING_ENABLED: bool = Field(default=True, env="MODEL_ROUTING_ENABLED")
    MODEL_ECONOMY: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"   # Bedrock ids
    MODEL_STANDARD: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    MODEL_PREMIUM: str = "global.anthropic.claude-sonnet-4-6"
    MODEL_COST_TRACKING_ENABLED: bool = Field(default=False, env="MODEL_COST_TRACKING_ENABLED")

    # ------------------------------------------------------------------
    # Circuit Breaker Settings
    # ------------------------------------------------------------------
    CIRCUIT_BREAKER_THRESHOLD: int = 1   # Trip after 1 failure — fastest possible fail-fast
    CIRCUIT_BREAKER_TIMEOUT: int = 15    # Recover after 15s (was 60) — retry sooner
    GEMINI_MAX_CONCURRENT: int = 10      # Max concurrent Gemini API calls (semaphore)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # ------------------------------------------------------------------
    # Query Logging
    # ------------------------------------------------------------------
    QUERY_LOG_ENABLED: bool = Field(default=True, env="QUERY_LOG_ENABLED")
    QUERY_LOG_PATH: str = "./data/logs/queries.db"

    # ------------------------------------------------------------------
    # Pydantic Settings Config
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    allowed_origins: str = Field(default="http://localhost:3000", env="allowed_origins")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# ----------------------------------------------------------------------
# Global settings instance
# ----------------------------------------------------------------------
settings = Settings()
