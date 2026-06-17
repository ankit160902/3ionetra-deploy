# 3ioNetra — Spiritual Companion

Production AI spiritual companion rooted in Sanatan Dharma. Users converse with "Mitra" (a warm spiritual friend) who listens empathetically, then guides with scriptures, mantras, and practices.

**Stack:** FastAPI (Python 3.11) + Next.js (TypeScript) + MongoDB + Redis + Qdrant (optional) + Google Gemini 2.5 Pro
**Version:** 1.1.4
**Live:** https://ionetra-frontend-688398835360.asia-south1.run.app | https://3io-netra.vercel.app

---

## Quick Start

```bash
# Full stack via Docker Compose
docker compose up --build        # Qdrant :6333, Redis :6379, Backend :8080, Frontend :3000

# Manual (backend)
cd backend
pip install -r requirements.txt
# Set .env: GEMINI_API_KEY, MONGODB_URI, DATABASE_NAME, DATABASE_PASSWORD, REDIS_HOST
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Manual (frontend)
cd frontend
npm install && npm run dev       # http://localhost:3000

# Data ingestion (required before first run)
cd backend && python scripts/ingest_all_data.py
```

---

## Architecture Overview

```
Frontend (Next.js)  →  FastAPI Routers  →  CompanionEngine (orchestrator)
                                              ├── IntentAgent (LLM classifier)
                                              ├── MemoryService (semantic recall)
                                              ├── ProductService (e-commerce)
                                              └── PanchangService (astrology)
                                          ↓
                                    ResponseComposer
                                      ├── LLMService (Gemini) ← PromptManager (YAML)
                                      └── RAGPipeline (scripture search)
                                            ├── SentenceTransformer (embeddings)
                                            ├── CrossEncoder (reranker)
                                            └── ContextValidator (5-gate filter)
                                          ↓
                                    SessionManager (Redis > MongoDB > InMemory)
                                    AuthService (MongoDB users/tokens/conversations)
                                    CacheService (Redis DB 1)
                                    SafetyValidator (crisis detection)
                                    CircuitBreaker (Gemini resilience)
```

---

## Directory Structure

```
backend/
├── main.py                     # FastAPI app bootstrap, lifespan, CORS, router registration
├── config.py                   # Pydantic Settings (all env vars and defaults)
├── llm/service.py              # LLMService — Gemini integration, prompt building, phase detection
├── rag/pipeline.py             # RAGPipeline — embedding search, BM25, reranking, query expansion
├── models/
│   ├── session.py              # SessionState, ConversationPhase, Signal, SignalType
│   ├── memory_context.py       # ConversationMemory, UserStory dataclasses
│   ├── api_schemas.py          # Pydantic request/response models
│   ├── dharmic_query.py        # DharmicQueryObject for RAG context synthesis
│   └── product.py              # Product model
├── routers/
│   ├── chat.py                 # /api/conversation, /api/session/*, /api/text/query, /api/user/*
│   ├── auth.py                 # /api/auth/register, login, verify, logout
│   └── admin.py                # /api/health, /api/scripture/search, /api/panchang, /api/tts
├── services/
│   ├── companion_engine.py     # Core orchestrator (~900 lines) — process_message, readiness logic
│   ├── intent_agent.py         # LLM-based intent classification (9 fields JSON output)
│   ├── response_composer.py    # Assembles final LLM response with memory + RAG + profile
│   ├── session_manager.py      # Redis/Mongo/InMemory session backends
│   ├── context_validator.py    # 5-gate RAG filter (relevance, content, type, scripture, diversity)
│   ├── context_synthesizer.py  # Synthesizes DharmicQueryObject from session memory
│   ├── prompt_manager.py       # Loads YAML prompts from backend/prompts/
│   ├── auth_service.py         # MongoDB auth + ConversationStorage (history persistence)
│   ├── cache_service.py        # Redis async cache (DB 1, separate from sessions)
│   ├── resilience.py           # CircuitBreaker (CLOSED→OPEN→HALF_OPEN)
│   ├── safety_validator.py     # Crisis detection, professional help referrals
│   ├── memory_service.py       # Semantic long-term memory retrieval
│   ├── product_service.py      # Product recommendations from 3ioNetra store
│   ├── panchang_service.py     # Hindu calendar / astrology data
│   └── tts_service.py          # Text-to-speech synthesis
├── prompts/
│   └── spiritual_mitra.yaml    # Persona definition, system instruction, phase prompts, domain compass
├── scripts/
│   ├── ingest_all_data.py      # Main ingestion pipeline (CSV/JSON/PDF/video → embeddings)
│   ├── pdf_ingester.py         # PDF → structured verses via Gemini
│   ├── video_ingester.py       # Video → structured segments via Gemini Files API
│   └── download_models.py      # Pre-downloads embedding/reranker models for Docker
├── tests/
│   └── qa_evaluator.py         # Automated QA evaluation suite
├── data/
│   ├── raw/                    # Source scripture files (CSV, JSON, PDF, video)
│   └── processed/
│       ├── verses.json         # Verse metadata (no embeddings)
│       └── embeddings.npy      # Pre-computed embeddings (memory-mapped at runtime)
├── Dockerfile                  # Production — bakes models into image, offline mode
└── Dockerfile.lite             # Dev — lighter dependencies, no baked models

frontend/
├── pages/
│   ├── index.tsx               # Main chat UI (messages, product cards, phase indicator)
│   └── _app.tsx                # Next.js app wrapper
├── components/
│   ├── LoginPage.tsx           # Auth UI (register/login)
│   ├── PhaseIndicator.tsx      # Conversation phase badge
│   └── TTSButton.tsx           # Text-to-speech playback button
├── hooks/
│   ├── useSession.ts           # Session management, API calls, streaming
│   └── useAuth.ts              # Auth state, token management
└── Dockerfile                  # Frontend container
```

---

## Backend Services

### CompanionEngine (`services/companion_engine.py`)
Core orchestrator. Entry point: `process_message(session, message)` and `generate_response_stream(session, message)`.
- Runs IntentAgent + MemoryService retrieval in parallel
- Updates session signals (emotion, life_domain, entities) from intent analysis
- Determines readiness for wisdom (intent-based or signal-threshold)
- Handles greeting, panchang, product search, closure as special intents
- Returns: `(response, is_ready_for_wisdom, context_docs, topics, products, phase)`
- **Product recommendations DISABLED (May 2026):** Both call sites (`guidance` path and `listening` path) are commented out. `products` always returns `[]`. `ProductService` class is intact but not called. Do not re-enable without explicit instruction.

### IntentAgent (`services/intent_agent.py`)
LLM-based classifier returning 9-field JSON:
- `intent`: GREETING | SEEKING_GUIDANCE | EXPRESSING_EMOTION | ASKING_INFO | ASKING_PANCHANG | PRODUCT_SEARCH | CLOSURE | OTHER
- `emotion`, `life_domain`, `entities`, `urgency`, `summary`
- `needs_direct_answer` (bool), `recommend_products` (bool), `product_search_keywords` (list)
- Falls back to keyword matching if LLM unavailable

### ResponseComposer (`services/response_composer.py`)
Assembles the final LLM prompt with: dharmic query, conversation memory, RAG verses, conversation history, user profile (including panchang, past memories). Delegates to `LLMService.generate_response()`.

### LLMService (`llm/service.py`)
Google Gemini integration. Key responsibilities:
- `_extract_context()` — signal detection from query text
- `_detect_phase()` — phase state machine
- `_build_prompt()` — large structured prompt with profile, facts, history, phase instructions, RAG context
- `generate_response()` / `generate_response_stream()` — Gemini API calls via CircuitBreaker
- `extract_text_from_image()` — OCR for spiritual texts
- `analyze_video()` — video analysis via Gemini Files API
- Loads system instruction and phase prompts from PromptManager (YAML)

### SessionManager (`services/session_manager.py`)
Abstract base with 3 backends (singleton factory auto-selects):
1. **RedisSessionManager** — preferred (async, TTL-based expiry)
2. **MongoSessionManager** — fallback (TTL index)
3. **InMemorySessionManager** — dev fallback

### RAGPipeline (`rag/pipeline.py`)
- **Initialize:** loads `verses.json` (metadata) + `embeddings.npy` (memory-mapped via `mmap_mode='r'` to avoid OOM)
- **Search:** query expansion → hybrid search (70% semantic + 30% BM25) → neural reranking (CrossEncoder) with intent-based weighting → min_score gate → doc-type filter
- **Query:** RAG-augmented QA with Redis caching (1h TTL)
- **Streaming:** `query_stream()` yields metadata chunk then answer tokens

### ContextValidator (`services/context_validator.py`)
5-gate sequential filter between RAG retrieval and LLM prompt:
1. **Relevance Gate** — drop below `min_score` (default 0.12)
2. **Content Gate** — drop empty/placeholder/too-short text
3. **Type Gate** — exclude spatial/temple docs for emotional intents; exclude meditation templates for emotional queries
4. **Scripture Gate** — hard-filter to allowed scriptures (graceful fallback)
5. **Diversity Gate** — max N docs per source to prevent echo-chamber

### Other Services
| Service | Role |
|---------|------|
| `PromptManager` | Loads YAML from `backend/prompts/`, dot-notation access (`get_prompt('spiritual_mitra', 'phase_prompts.guidance')`) |
| `AuthService` | MongoDB user CRUD, PBKDF2 password hashing, 30-day token expiry |
| `ConversationStorage` | Persistent conversation history in MongoDB, Redis-cached list queries |
| `CacheService` | Redis DB 1, deterministic MD5 keys, async get/set with TTL |
| `CircuitBreaker` | States: CLOSED→OPEN→HALF_OPEN. Default: 5 failures, 60s recovery |
| `SafetyValidator` | Crisis keyword detection, professional help referral (iCall, Vandrevala, NIMHANS) |
| `MemoryService` | Semantic long-term memory retrieval via RAG embeddings |
| `ProductService` | Product search from 3ioNetra store for recommendations |
| `PanchangService` | Hindu calendar data (tithi, nakshatra, special days) |
| `ContextSynthesizer` | Builds `DharmicQueryObject` from session memory for RAG queries |
| `TTSService` | Text-to-speech via gTTS |

---

## Conversation Flow

### Phase State Machine
```
LISTENING → (enough signals OR direct ask) → GUIDANCE → CLOSURE
     ↑                                           |
     └───────── (cooldown: 2+ turns) ────────────┘
```

**Phases** (defined in `ConversationPhase` enum):
- `LISTENING` — greeting, initial exploration, building rapport
- `CLARIFICATION` — gathering more context (internal, rarely exposed in API)
- `SYNTHESIS` — reflecting back understanding
- `GUIDANCE` — scripture, mantras, practices, product recommendations
- `ANSWERING` — (legacy/internal phase, rarely used directly)
- `CLOSURE` — wrap-up, blessing, "come back"

### Readiness Assessment
Transition to GUIDANCE happens when:
1. **Direct ask detected:** `needs_direct_answer=True`, or intent is SEEKING_GUIDANCE / ASKING_INFO / ASKING_PANCHANG / PRODUCT_SEARCH
2. **Signal threshold met:** `turn_count >= min_clarification_turns` AND `len(signals) >= min_signals_threshold`
3. **Force transition:** `turn_count >= max_clarification_turns` with cooldown check
4. **Memory readiness:** `readiness_for_wisdom >= 0.7`

### Oscillation Control
After guidance is given, `last_guidance_turn` is set. At least 2 turns of listening must pass before next guidance to prevent flip-flopping.

---

## Data Models

### SessionState (`models/session.py`)
```python
session_id: str (UUID)
phase: ConversationPhase
turn_count: int
signals_collected: Dict[SignalType, Signal]   # EMOTION, TRIGGER, LIFE_DOMAIN, MENTAL_STATE, USER_GOAL, INTENT, SEVERITY
conversation_history: List[Dict[str, str]]     # {role, content, timestamp}
memory: ConversationMemory
dharmic_query: DharmicQueryObject              # synthesized RAG query
is_returning_user: bool
```

### ConversationMemory (`models/memory_context.py`)
```python
story: UserStory          # primary_concern, emotional_state, life_area, trigger, demographics, spiritual profile
readiness_for_wisdom: float  # 0.0–1.0
user_quotes: List[Dict]   # significant user quotes with turn numbers
emotional_arc: List[Dict]  # emotion trajectory
relevant_concepts: List[str]  # dharmic concepts detected
user_id, user_name, user_email, user_phone, user_dob, user_created_at
```

### API Schemas (`models/api_schemas.py`)
- **Request:** `ConversationalQuery(session_id, message, language, user_profile)`
- **Response:** `ConversationalResponse(session_id, phase, response, signals_collected, turn_count, is_complete, sources, recommended_products, flow_metadata)`

---

## RAG Pipeline Details

### Data Format
- `data/processed/verses.json` — verse metadata (scripture, reference, text, meaning, type, topic)
- `data/processed/embeddings.npy` — float32 numpy array, memory-mapped at runtime
- Verse types: `scripture`, `temple`, `procedural`

### Search Pipeline
1. **Query Expansion** — LLM generates 2 alternative search terms for short queries (<4 words)
2. **Hybrid Search** — semantic cosine similarity (70%) + BM25 keyword scores (30%)
3. **Candidate Retrieval** — top 20 candidates via `np.argpartition`
4. **Neural Reranking** — CrossEncoder (`bge-reranker-v2-m3`, multilingual) with intent-based weight adjustments
5. **Post-filters** — min_score gate, doc_type exclusion, final top_k cut

### Embedding Model
- `intfloat/multilingual-e5-large` (1024-dim, with query/passage prefix)
- Pre-normalized during ingestion for fast dot-product similarity

---

## Data Ingestion

```bash
cd backend
python scripts/ingest_all_data.py   # Processes all files in data/raw/
```

**Supported formats:** CSV, JSON, PDF (via Gemini OCR), Video (via Gemini Files API)

**Pipeline:** find files → parse by type → infer scripture name → infer topic → deduplicate by reference → generate embeddings → save split format (verses.json + embeddings.npy)

**Special handling:**
- Temple JSON files → converted to verse-like objects with `type: "temple"`
- PDF files → processed via `PDFIngester` using Gemini multimodal
- Video files → processed via `VideoIngester` using Gemini Files API (segments, transcription, shlokas)

---

## API Endpoints

### Auth (`/api/auth`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Create account (name, email, password, profile fields) |
| POST | `/login` | Login, returns Bearer token (30-day expiry) |
| GET | `/verify` | Verify token, returns user info |
| POST | `/logout` | Invalidate token |

### Chat (`/api`)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/conversation` | Main conversational endpoint (auth optional but enriches profile) |
| POST | `/session/create` | Create new session, returns welcome message |
| GET | `/session/{id}` | Get session state |
| DELETE | `/session/{id}` | Delete session |
| POST | `/text/query` | Standalone RAG query with citations |
| GET | `/user/conversations` | List user's conversation history |
| GET | `/user/conversations/{id}` | Get specific conversation |
| POST | `/user/conversations` | Save conversation |
| DELETE | `/user/conversations/{id}` | Delete conversation |

### Admin (`/api`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (RAG status) |
| GET | `/ready` | Readiness probe |
| GET | `/scripture/search` | Direct scripture search |
| POST | `/embeddings/generate` | Generate embeddings for text |
| GET | `/panchang/today` | Current panchang data |
| POST | `/tts` | Text-to-speech |

---

## Frontend

Next.js app with Tailwind CSS. Key structure:

- `pages/index.tsx` — Main chat interface: message list, input, phase indicator, conversation history sidebar; sticky header shows BETA badge next to app name (May 2026)
- `components/LoginPage.tsx` — Registration and login forms; registration CTA branded as "Know Your Bhakt" (not "Create Account") in all 3 touchpoints (May 2026)
- `components/PhaseIndicator.tsx` — Shows current conversation phase
- `components/TTSButton.tsx` — Audio playback of bot responses
- `hooks/useSession.ts` — Session management, API calls, SSE streaming
- `hooks/useAuth.ts` — Auth state, token storage (localStorage)
- `next.config.js` — CSP allows `http://localhost:8080` in `NODE_ENV=development` only; production CSP is strict (May 2026 fix for local dev login)

---

## Configuration

All settings in `backend/config.py` via `pydantic-settings`. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Model for all LLM calls |
| `MONGODB_URI` | (required for auth) | MongoDB connection string |
| `DATABASE_NAME` | (required for auth) | MongoDB database name |
| `DATABASE_PASSWORD` | (optional) | Replaces `<db_password>` in URI |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | (optional) | Redis auth |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | SentenceTransformer model (1024-dim, E5 prefix) |
| `RETRIEVAL_TOP_K` | `7` | Number of RAG results |
| `RERANK_TOP_K` | `3` | Results after reranking |
| `MIN_SIMILARITY_SCORE` | `0.15` | Minimum cosine similarity |
| `MIN_SIGNALS_THRESHOLD` | `2` | Signals needed for transition |
| `SESSION_TTL_MINUTES` | `60` | Session expiry |
| `API_PORT` | `8080` | Backend port |

---

## Deployment

### Docker Compose (dev)
4 services: `qdrant`, `redis`, `backend` (Dockerfile.lite), `frontend`

### Production Dockerfile
- Python 3.11 slim, bakes embedding + reranker models into image
- `TRANSFORMERS_OFFLINE=1` at runtime — no network model downloads
- Cloud Run compatible: uses `$PORT` env var, 1 worker, 65s keep-alive
- Memory-mapped embeddings (`mmap_mode='r'`) to prevent OOM on constrained instances

---

## Prompt System (`backend/prompts/spiritual_mitra.yaml`)

### Structure
- `persona` — name ("3ioNetra — Mitra"), description
- `system_instruction` — full persona definition loaded as Gemini system instruction
- `phase_prompts` — per-phase instructions (listening, synthesis, guidance, closure)
- `rag_synthesis` — instructions for using RAG context

### Key Persona Rules
- Warm spiritual friend, not therapist or chatbot
- Flowing sentences only — no markdown, no bullet points, no numbered lists
- Natural spiritual hooks: mantras, practices, temple visits, verses
- Never force spirituality — read the room
- When rejected, pivot to alternative (don't just empathize)
- Domain compass: 20 life domains mapped to specific dharmic concepts, mantras, and anchors
- **Product queries (May 2026):** When user wants to acquire a spiritual item (murti, rudraksha, mala, yantra, etc.), treat it as a spiritual question — respond with the item's significance, deity connection, mantra, consecration method, and placement. Never give prices or store names.
- **PRICE RULE:** If asked how much something costs, give no price (not even approximate). Redirect to intention: "What matters more is the intention you bring to it."
- **WHERE-TO-BUY RULE:** If asked where to buy online, suggest a local temple shop or spiritual store in their city instead. Pivot to spiritual preparation, not shopping logistics.

---

## Testing

```bash
cd backend
python tests/qa_evaluator.py                    # Automated QA evaluation
python scripts/test_context_validation.py       # Context validator tests
python scripts/validator_suite.py               # Full validation suite
python scripts/verify_pdf_ingestion.py          # PDF ingestion verification
python scripts/verify_video_ingestion.py        # Video ingestion verification
```

---

## Coding Conventions

- **Singleton pattern** — all services use `get_*()` factory functions with module-level `_instance` variables
- **Async-first** — all service methods are async, Gemini calls are awaited
- **Circuit breaker** — wraps all Gemini API calls (`CircuitBreaker.call()`)
- **Parallel execution** — `asyncio.gather()` for independent tasks (intent + memory, semantic + BM25)
- **Graceful degradation** — every service has `.available` flag, fallbacks for LLM/Redis/MongoDB unavailability
- **Memory-mapped data** — embeddings loaded with `np.load(mmap_mode='r')` to avoid OOM

---

## Critical Rules

1. **Response length:** Adaptive — match query complexity. Simple support: 50-100 words. Detailed requests (itinerary, step-by-step, educational): up to 300 words. Never artificially truncate. (Apr 2026 — was "30-100 words" hardcoded.)
2. **Restricted markdown in LLM responses.** Default is flowing sentences. Allowed: `**bold**` (sparing — 1-2 per response, for deity names or key practices), `- ` bullet lists (2-5 items max, for short step sequences only), `---` horizontal rules (at most one per response, for a real subject change). Forbidden: `# headers`, `> blockquotes`, `*italic*`, `` `inline code` ``, `1. numbered lists`, tables, links. The persona is a warm friend, not an article writer. (Apr 2026 — was previously "no markdown".)
3. **Verse format:** `[VERSE]...[/VERSE]` tags for original Sanskrit/Hindi only. Max one per response.
4. **Safety protocol:** Crisis → direct compassion + helpline numbers (iCall, Vandrevala, NIMHANS). Never spiritual-reframe active danger.
5. **Product recommendations DISABLED (May 2026):** Both call sites in `companion_engine.py` are commented out — `products` always returns `[]`. Do not re-enable without explicit instruction. LLM text must never mention products, prices, URLs, or store names. Product-seeking queries are answered as spiritual guidance (significance, mantra, consecration) with no price or purchase information.
6. **No hollow phrases:** Never say "I hear you", "I understand", "It sounds like", "everything happens for a reason". Never attribute suffering to "past life karma". Never say "just be positive" or "others have it worse".
7. **Pivot on rejection:** If user rejects a suggestion, immediately offer an alternative spiritual path — never just empathize and back off.
