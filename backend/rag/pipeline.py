import asyncio
import hashlib
import json
import math
import os
import re
import logging
import shelve
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np

from config import settings
from constants import TRIVIAL_MESSAGES
from llm.bedrock import bedrock_generate
from llm.service import get_llm_service
from models.session import IntentType
from rag.scoring_utils import get_doc_score
from services.cache_service import get_cache_service

logger = logging.getLogger(__name__)

# GPU access lock — serializes embedding model inference to prevent
# VRAM contention on small GPUs (e.g., GTX 1650 4GB).
#
# Implementation note: asyncio.Lock instances bind to the running event loop
# on first use (Python 3.10+).  A module-level Lock() created at import time
# will raise "bound to a different event loop" if the event loop is replaced
# (e.g. in tests or after uvicorn reload).  We therefore store one lock per
# event-loop identity and create it lazily on first access.
_gpu_locks: dict = {}


def _gpu_lock() -> asyncio.Lock:
    """Return the asyncio.Lock for the current event loop, creating it if needed."""
    loop = asyncio.get_event_loop()
    loop_id = id(loop)
    if loop_id not in _gpu_locks:
        _gpu_locks[loop_id] = asyncio.Lock()
    return _gpu_locks[loop_id]


def _get_onnx_providers() -> List[str]:
    """Return the explicit ONNX Runtime execution provider list.

    Why this exists
    ---------------
    By default, ONNX Runtime probes every available execution provider
    on the host. On Apple Silicon Macs the CoreML provider probe fails
    with ``EP Error SystemError : 20``, then the runtime falls back to
    CPU. The fallback works but emits 4 lines of stderr noise on every
    cold start, hiding real errors.

    Setting an explicit provider list short-circuits the probe and
    documents the supported execution paths. CUDA is preferred when
    available; otherwise we use CPU directly. CoreML is intentionally
    excluded — sentence-transformers' CoreML support is incomplete and
    the speedup on Apple Silicon doesn't outweigh the noise + risk of
    silent fallback.
    """
    providers: List[str] = []
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
    except Exception:
        pass
    providers.append("CPUExecutionProvider")
    return providers


def _sanitize_for_prompt(text: str, max_len: int = 500) -> str:
    """Sanitize user input before interpolation into LLM prompts.
    Strips characters that could be interpreted as prompt structure."""
    clean = text[:max_len].replace('"', "'").replace("\\", "")
    # Remove lines that look like prompt injection attempts
    lines = clean.split("\n")
    safe_lines = [l for l in lines if not l.strip().upper().startswith(("[SYSTEM", "[INST", "<<SYS"))]
    return " ".join(safe_lines).strip()

# Domain-to-scripture affinity map for reranking boost (80+ entries across all 11 scriptures)
DOMAIN_SCRIPTURE_AFFINITY = {
    # --- Original 24 entries (refined with multi-scripture coverage) ---
    "health": {"charaka samhita": 0.6, "atharva veda": 0.3, "patanjali yoga sutras": 0.2},
    "career": {"bhagavad gita": 0.3, "mahabharata": 0.2},
    "career_work": {"bhagavad gita": 0.3, "mahabharata": 0.2},
    "grief": {"bhagavad gita": 0.3, "ramayana": 0.3},
    "anger": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.2, "mahabharata": 0.2},
    "meditation": {"patanjali yoga sutras": 0.5, "bhagavad gita": 0.2},
    "yoga": {"patanjali yoga sutras": 0.6, "bhagavad gita": 0.2},
    "ayurveda": {"charaka samhita": 0.7},
    "ayurveda_specific": {"charaka samhita": 0.7},
    "family": {"ramayana": 0.4, "bhagavad gita": 0.2, "mahabharata": 0.2},
    "devotion": {"bhagavad gita": 0.3, "ramayana": 0.3, "rig veda": 0.2},
    "liberation": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.3},
    "self-worth": {"bhagavad gita": 0.5, "ramayana": 0.2},
    "fear": {"bhagavad gita": 0.3, "atharva veda": 0.2},
    "addiction": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.3},
    "relationships": {"ramayana": 0.3, "bhagavad gita": 0.2, "mahabharata": 0.2},
    "parenting": {"ramayana": 0.4, "mahabharata": 0.2},
    "ethics": {"bhagavad gita": 0.4, "mahabharata": 0.3},
    "ethics_moral": {"bhagavad gita": 0.4, "mahabharata": 0.3},
    "education": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.2},
    "education_exam": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.2},
    "shame": {"bhagavad gita": 0.5, "ramayana": 0.2},
    "guilt": {"bhagavad gita": 0.5, "ramayana": 0.2},
    "jealousy": {"bhagavad gita": 0.4, "mahabharata": 0.3},
    "loneliness": {"bhagavad gita": 0.5, "ramayana": 0.2},
    "frustration": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.2},
    "confusion": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.2},
    "hopelessness": {"bhagavad gita": 0.5, "ramayana": 0.2},
    "financial_stress": {"bhagavad gita": 0.3, "atharva veda": 0.2},
    "pregnancy_fertility": {"bhagavad gita": 0.2, "atharva veda": 0.3, "charaka samhita": 0.3},
    "digital_life": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.3},
    "habits_lust": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.3},

    # --- Emotions missing from original map ---
    "anxiety": {"patanjali yoga sutras": 0.5, "bhagavad gita": 0.3},
    "despair": {"bhagavad gita": 0.5, "ramayana": 0.3},
    "joy": {"rig veda": 0.3, "bhagavad gita": 0.2, "ramayana": 0.2},
    "gratitude": {"rig veda": 0.4, "bhagavad gita": 0.2},
    "hope": {"bhagavad gita": 0.3, "ramayana": 0.3},
    "curiosity": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.2},
    "neutral": {"bhagavad gita": 0.2},

    # --- Life domains with multi-scripture coverage ---
    "marriage": {"ramayana": 0.5, "atharva veda": 0.3, "bhagavad gita": 0.2},
    "death": {"bhagavad gita": 0.5, "atharva veda": 0.2, "ramayana": 0.2},
    "sleep": {"charaka samhita": 0.6, "patanjali yoga sutras": 0.3},
    "diet": {"charaka samhita": 0.7, "bhagavad gita": 0.2},
    "pilgrimage": {"ramayana": 0.4, "mahabharata": 0.3},
    "mantra": {"atharva veda": 0.4, "rig veda": 0.4, "bhagavad gita": 0.2},
    "puja_ritual": {"atharva veda": 0.4, "rig veda": 0.3},
    "fasting": {"atharva veda": 0.3, "bhagavad gita": 0.2},
    "protection": {"atharva veda": 0.5, "ramayana": 0.3},
    "prosperity": {"atharva veda": 0.4, "rig veda": 0.3, "bhagavad gita": 0.2},
    "concentration": {"patanjali yoga sutras": 0.6, "bhagavad gita": 0.3},
    "forgiveness": {"ramayana": 0.4, "bhagavad gita": 0.3, "mahabharata": 0.2},
    "courage": {"ramayana": 0.4, "bhagavad gita": 0.4, "mahabharata": 0.3},
    "patience": {"bhagavad gita": 0.4, "ramayana": 0.3, "patanjali yoga sutras": 0.2},
    "discipline": {"patanjali yoga sutras": 0.5, "bhagavad gita": 0.3},
    "identity": {"bhagavad gita": 0.5, "patanjali yoga sutras": 0.2},
    "rebirth": {"bhagavad gita": 0.5, "atharva veda": 0.2},
    "sacrifice": {"rig veda": 0.4, "bhagavad gita": 0.3, "atharva veda": 0.2},
    "ego": {"bhagavad gita": 0.5, "patanjali yoga sutras": 0.3},
    "attachment": {"bhagavad gita": 0.5, "patanjali yoga sutras": 0.3},
    "desire": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.3},
    "contentment": {"patanjali yoga sutras": 0.4, "bhagavad gita": 0.3},
    "self_improvement": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.4},
    "yoga_practice": {"patanjali yoga sutras": 0.6, "bhagavad gita": 0.2},
    "ayurveda_wellness": {"charaka samhita": 0.7, "atharva veda": 0.2},
    "meditation_mind": {"patanjali yoga sutras": 0.5, "bhagavad gita": 0.3},
    "dharma_duty": {"bhagavad gita": 0.5, "ramayana": 0.3, "mahabharata": 0.2},
    "spiritual_practice": {"bhagavad gita": 0.3, "patanjali yoga sutras": 0.3, "rig veda": 0.2},
    "worship_bhakti": {"rig veda": 0.4, "bhagavad gita": 0.3, "ramayana": 0.2},
    "knowledge_learning": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.2},
    "teacher_guru": {"bhagavad gita": 0.3, "ramayana": 0.3, "mahabharata": 0.2},
    "nonviolence": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.3, "mahabharata": 0.2},
    "mental_health": {"patanjali yoga sutras": 0.5, "bhagavad gita": 0.3, "charaka samhita": 0.2},
    "stress": {"patanjali yoga sutras": 0.4, "bhagavad gita": 0.3, "charaka samhita": 0.2},
    "depression": {"bhagavad gita": 0.4, "patanjali yoga sutras": 0.3, "ramayana": 0.2},

    # --- Additional entries to reach 80+ ---
    "temple_visit": {"ramayana": 0.3, "mahabharata": 0.3},
    "astrology": {"atharva veda": 0.4, "rig veda": 0.3},
    "dreams": {"atharva veda": 0.4, "charaka samhita": 0.3},
    "old_age": {"bhagavad gita": 0.4, "charaka samhita": 0.3},
    "children": {"ramayana": 0.4, "mahabharata": 0.3, "atharva veda": 0.2},
    "friendship": {"ramayana": 0.4, "mahabharata": 0.3},
    "success": {"bhagavad gita": 0.4, "mahabharata": 0.2},
    "creativity": {"rig veda": 0.3, "bhagavad gita": 0.3},
}

# Keywords indicating user is asking about temples/pilgrimage
_TEMPLE_KEYWORDS = {"temple", "mandir", "pilgrimage", "tirtha", "yatra",
                    "visit", "darshan", "jyotirlinga", "shrine", "dham",
                    "मंदिर", "तीर्थ", "दर्शन", "ज्योतिर्लिंग", "धाम", "यात्रा"}

# Citation lookup patterns for direct verse reference short-circuit (item 1.3)
_CITATION_PATTERNS = [
    re.compile(r'(?:bhagavad\s*gita|gita|bg)\s*(\d+)[.:\s]+(\d+)', re.I),
    re.compile(r'(?:yoga\s*sutra|ys|patanjali)\s*(\d+)[.:\s]+(\d+)', re.I),
    re.compile(r'(?:rig\s*veda|rv)\s*(\d+)[.:\s]+(\d+)', re.I),
    re.compile(r'(?:atharva\s*veda|av)\s*(\d+)[.:\s]+(\d+)', re.I),
    re.compile(r'(?:ramayana)\s*(\d+)[.:\s]+(\d+)', re.I),
    re.compile(r'(?:mahabharata)\s*(\d+)[.:\s]+(\d+)', re.I),
]




# ------------------------------------------------------------------
# Query normalization is handled by services.query_normalizer.QueryNormalizer.
# The previous Levenshtein-based spell corrector and dharmic vocabulary
# tables that lived here mangled common English words into Sanskrit terms
# (e.g. "How" → "homa", "Give" → "gita") because they operated on whole
# strings without a stopword mask. The semantic infrastructure (E5 + BGE
# CrossEncoder) handles morphology and fuzzy matching far better.
# ------------------------------------------------------------------


_BM25_STOPWORDS = frozenset({
    "a", "i", "an", "is", "it", "in", "on", "to", "of", "or", "by",
    "at", "if", "do", "so", "up", "no", "be", "we", "he", "me",
    "my", "am", "as",
})


def _simple_stem(word: str) -> str:
    """Lightweight English suffix stemmer for BM25 recall improvement.
    Handles common inflections: meditating/meditation → meditate, chanting → chant, etc.
    """
    if len(word) <= 4:
        return word
    for suffix, replacement in [
        ('ational', 'ate'), ('tional', 'tion'), ('ation', 'ate'),
        ('izing', 'ize'), ('ating', 'ate'), ('eness', ''),
        ('ness', ''), ('ment', ''), ('ance', ''), ('ence', ''),
        ('ting', 't'), ('ing', ''), ('ied', 'y'), ('ies', 'y'),
        ('ed', ''), ('er', ''), ('es', ''), ('s', ''),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[:-len(suffix)] + replacement
    return word


class RAGPipeline:
    """
    Optimized RAG pipeline that:
    - loads pre‑computed verse embeddings from data/processed
    - performs efficient cosine‑similarity search with pre-normalized vectors
    - filters results based on configuration thresholds for precision
    - can optionally call the LLM to synthesize answers

    The API is designed to match how `main.py` uses it:
    - await initialize()
    - await query(...)
    - async for chunk in query_stream(...)
    - await search(...)
    - await generate_embeddings(...)
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        global _rag_pipeline_instance
        self.verses: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.dim: int = 0
        self.available: bool = False
        self._embedding_model = None
        self._reranker_model = None
        self._llm = get_llm_service()
        # Single source of truth for query string preprocessing.
        # See services/query_normalizer.py for the contract.
        from services.query_normalizer import get_query_normalizer
        self._query_normalizer = get_query_normalizer()
        self._data_dir_override: Optional[Path] = Path(data_dir) if data_dir is not None else None
        # Register as the module-level singleton so get_rag_pipeline() works
        _rag_pipeline_instance = self
        # Per-instance semaphore for embedding model concurrency (max 4 parallel calls)
        self._embed_sem = asyncio.Semaphore(4)

    def __bool__(self) -> bool:
        return self.available

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Load RAG data efficiently using memory mapping for embeddings.
        This allows handling massive datasets (2GB+) without OOM crashes.
        """
        try:
            if self._data_dir_override is not None:
                processed_dir = self._data_dir_override
            else:
                base_dir = Path(__file__).parent.parent
                processed_dir = base_dir / "data" / "processed"
            metadata_path = processed_dir / "verses.json"
            embeddings_path = processed_dir / "embeddings.npy"

            # Check for high-efficiency split format
            if metadata_path.exists() and embeddings_path.exists():
                logger.info(f"RAGPipeline: Loading metadata from {metadata_path}...")
                with metadata_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.verses = payload.get("verses", [])
                
                logger.info(f"RAGPipeline: Memory-mapping embeddings from {embeddings_path}...")
                # mmap_mode='r' is the critical fix for Cloud Run OOM
                raw_embeddings = np.load(embeddings_path, mmap_mode='r')
            else:
                logger.warning("RAGPipeline: No processed data found (verses.json + embeddings.npy). Run scripts/ingest_all_data.py.")
                self.available = False
                return

            if not self.verses or raw_embeddings.size == 0:
                logger.warning("RAGPipeline: Empty dataset loaded")
                self.available = False
                return

            # Validate verse schema — check first verse has required fields
            _required_fields = {"text", "scripture"}
            _sample = self.verses[0]
            _missing = _required_fields - set(_sample.keys())
            if _missing:
                logger.error(f"RAGPipeline: verses.json schema invalid — first verse missing fields: {_missing}")
                self.available = False
                return

            # Validate embedding/verse count alignment
            if raw_embeddings.shape[0] != len(self.verses):
                logger.error(
                    f"RAGPipeline: embeddings/verses count mismatch — "
                    f"{raw_embeddings.shape[0]} embeddings vs {len(self.verses)} verses"
                )
                self.available = False
                return

            # Validate embeddings contain no NaN (corrupted ingestion)
            # Check a sample of rows to avoid full-array scan on mmap
            _check_indices = np.linspace(0, raw_embeddings.shape[0] - 1, min(100, raw_embeddings.shape[0]), dtype=int)
            if np.isnan(raw_embeddings[_check_indices]).any():
                logger.error("RAGPipeline: embeddings.npy contains NaN values — run re-ingestion")
                self.available = False
                return

            self.dim = raw_embeddings.shape[1]
            self.embeddings = raw_embeddings
            self.available = True

            # Precompute curated concept doc indices for slot reservation during search
            self._curated_indices = np.array([
                i for i, v in enumerate(self.verses)
                if v.get("type") == "curated_concept" or v.get("source") == "curated_concept"
            ], dtype=np.int64)
            if len(self._curated_indices) > 0:
                logger.info(f"RAGPipeline: {len(self._curated_indices)} curated concept docs indexed for slot reservation")

            # Build scripture → indices mapping for pre-filtering
            self._scripture_indices: Dict[str, np.ndarray] = {}
            scripture_counts = {}
            for i, v in enumerate(self.verses):
                s = v.get("scripture", "Unknown")
                scripture_counts[s] = scripture_counts.get(s, 0) + 1
                if s not in self._scripture_indices:
                    self._scripture_indices[s] = []
                self._scripture_indices[s].append(i)
            for k in self._scripture_indices:
                self._scripture_indices[k] = np.array(self._scripture_indices[k], dtype=np.int64)
            logger.info(f"RAGPipeline: scripture index built for {len(self._scripture_indices)} scriptures")

            # Build neighbor index for parent-child verse retrieval
            if settings.PARENT_CHILD_ENABLED:
                self._build_neighbor_index()

            logger.info(
                f"RAGPipeline initialized with {len(self.verses)} verses "
                f"(dim={self.dim}). Memory Map: {'Active' if metadata_path.exists() else 'Inactive (Legacy Mode)'}"
            )

            # Eagerly load ML models at startup to avoid per-request delays
            self._load_ml_models()

            # Note: the previous implementation built a hardcoded "sanskrit
            # term lookup" set used by ``_contains_sanskrit_term`` to decide
            # whether to trigger query expansion. That heuristic has been
            # removed — bilingual / synonym handling is now the responsibility
            # of services.intent_agent.IntentAgent which produces query
            # variants threaded into ``RAGPipeline.search(query_variants=...)``
            # upstream. Short queries still trigger expansion via the LLM
            # path; long queries go through embedding/reranker which handle
            # multilingual matching natively.

            # Build reference index for citation short-circuit (item 1.3)
            self._reference_index: Dict[str, int] = {}
            for i, v in enumerate(self.verses):
                ref = v.get("reference", "")
                if ref:
                    self._reference_index[ref.lower()] = i
            logger.info(f"Reference index: {len(self._reference_index)} entries")

            # Build tradition → indices mapping (item 2.1)
            self._tradition_indices: Dict[str, np.ndarray] = {}
            tradition_counts_map: Dict[str, int] = {}
            for i, v in enumerate(self.verses):
                t = v.get("tradition", "general")
                tradition_counts_map[t] = tradition_counts_map.get(t, 0) + 1
                if t not in self._tradition_indices:
                    self._tradition_indices[t] = []
                self._tradition_indices[t].append(i)
            for k in self._tradition_indices:
                self._tradition_indices[k] = np.array(self._tradition_indices[k], dtype=np.int64)
            logger.info(f"Tradition index: {len(self._tradition_indices)} traditions — {tradition_counts_map}")

            # Load corpus manifest if available (item 2.4)
            manifest_path = processed_dir / "corpus_manifest.json"
            if manifest_path.exists():
                try:
                    with manifest_path.open("r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    logger.info(f"Corpus manifest v{manifest.get('version', '?')}: "
                                f"{manifest.get('verse_count', '?')} verses, "
                                f"generated {manifest.get('generated_at', '?')}")
                except Exception as e:
                    logger.warning(f"Failed to load corpus manifest: {e}")

            # Load cross-reference index if available (item 2.5)
            self._cross_refs: Dict[str, List[str]] = {}
            cross_refs_path = processed_dir / "cross_refs.json"
            if cross_refs_path.exists():
                try:
                    with cross_refs_path.open("r", encoding="utf-8") as f:
                        self._cross_refs = json.load(f)
                    logger.info(f"Cross-reference index: {len(self._cross_refs)} entries loaded")
                except Exception as e:
                    logger.warning(f"Failed to load cross-refs: {e}")

            # Load section chunks if available (item 2.6)
            self._section_verses: List[Dict] = []
            self._section_embeddings: Optional[np.ndarray] = None
            sections_path = processed_dir / "sections.json"
            sections_emb_path = processed_dir / "sections_embeddings.npy"
            if settings.SECTION_CHUNKS_ENABLED and sections_path.exists() and sections_emb_path.exists():
                try:
                    with sections_path.open("r", encoding="utf-8") as f:
                        section_payload = json.load(f)
                    self._section_verses = section_payload.get("sections", [])
                    self._section_embeddings = np.load(sections_emb_path, mmap_mode='r')
                    logger.info(f"Section chunks: {len(self._section_verses)} loaded")
                except Exception as e:
                    logger.warning(f"Failed to load section chunks: {e}")

        except Exception as exc:
            logger.exception(f"Failed to initialize RAGPipeline: {exc}")
            self.available = False

    # ------------------------------------------------------------------
    # ML Model Loading (eager at startup, cached locally)
    # ------------------------------------------------------------------

    def _load_ml_models(self) -> None:
        """Load embedding and reranker models at startup.
        Also pre-computes BM25 statistics and sets HuggingFace Hub to offline
        mode to prevent HTTP requests during request handling.
        """
        self._ensure_embedding_model()
        self._ensure_reranker_model()  # Eager: eliminates first-query cold start

        # Pre-compute BM25 doc stats so first search is instant
        self._precompute_bm25_stats()

        if self._embedding_model is not None:
            try:
                self._embedding_model.encode(["warmup"], show_progress_bar=False)
                logger.info("RAGPipeline: embedding model pre-warmed")
            except Exception as e:
                logger.warning(f"Embedding warmup failed (non-fatal): {e}")

        if self._reranker_model is not None:
            try:
                self._reranker_model.predict([("warmup query", "warmup document")])
                logger.info("RAGPipeline: reranker model pre-warmed")
            except Exception as e:
                logger.warning(f"Reranker warmup failed (non-fatal): {e}")

        # Load SPLADE model if enabled and index exists (item 3.2)
        self._splade_model = None
        self._splade_tokenizer = None
        self._splade_index = None
        if settings.SPLADE_ENABLED:
            base_dir = Path(__file__).parent.parent
            splade_index_path = base_dir / "data" / "processed" / "splade_index.npz"
            if splade_index_path.exists():
                try:
                    from scipy.sparse import load_npz
                    from transformers import AutoModelForMaskedLM, AutoTokenizer
                    self._splade_index = load_npz(str(splade_index_path))
                    local_splade = base_dir / "models" / "splade"
                    model_name = str(local_splade) if local_splade.exists() and os.listdir(str(local_splade)) else settings.SPLADE_MODEL
                    self._splade_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self._splade_model = AutoModelForMaskedLM.from_pretrained(model_name)
                    self._splade_model.eval()
                    logger.info(f"SPLADE: loaded model + index ({self._splade_index.shape})")
                except Exception as e:
                    logger.warning(f"SPLADE: failed to load — falling back to BM25: {e}")
                    if settings.SPLADE_REQUIRED:
                        raise RuntimeError(
                            f"SPLADE_REQUIRED=True but SPLADE failed to load: {e}. "
                            "Either fix the SPLADE model/index or set "
                            "SPLADE_ENABLED=False / SPLADE_REQUIRED=False."
                        ) from e
            else:
                # The index is missing. Surface this loudly — silent
                # degradation to BM25 hides corrupted ingest pipelines.
                msg = (
                    f"SPLADE: enabled but no index found at {splade_index_path}. "
                    "To fix, run: python scripts/build_splade_index.py"
                )
                if settings.SPLADE_REQUIRED:
                    raise RuntimeError(
                        msg + " (SPLADE_REQUIRED=True — refusing to start in degraded mode)"
                    )
                logger.error(msg + " — falling back to BM25 (set SPLADE_REQUIRED=True to fail fast)")

        # Block all future HuggingFace Hub HTTP requests
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info(
            "RAGPipeline: ML models loaded at startup (embedding + reranker) — "
            "HuggingFace Hub set to offline mode for runtime"
        )

    def _try_citation_lookup(self, query: str) -> Optional[List[Dict]]:
        """Short-circuit: if query is a direct verse citation (e.g. 'BG 2.47'),
        look it up by reference instead of running the full embedding pipeline."""
        if not hasattr(self, '_reference_index') or not self._reference_index:
            return None

        for pattern in _CITATION_PATTERNS:
            m = pattern.search(query)
            if m:
                chapter, verse = m.group(1), m.group(2)
                # Build candidate reference strings to search for
                candidates = [
                    f"{chapter}.{verse}",
                    f"chapter {chapter}, verse {verse}",
                    f"ch {chapter} v {verse}",
                ]
                for ref_key, idx in self._reference_index.items():
                    for candidate in candidates:
                        if candidate in ref_key:
                            verse_data = self.verses[idx].copy()
                            verse_data["score"] = 1.0
                            verse_data["final_score"] = 1.0
                            verse_data["_global_idx"] = idx
                            logger.info(f"Citation short-circuit: '{query}' → {verse_data.get('reference', ref_key)}")
                            return [verse_data]
        return None

    def _get_candidate_pool_size(self, query: str, intent: Optional[IntentType]) -> int:
        """Return adaptive candidate pool size based on query type and intent."""
        _comparison_words = {"compare", "difference", "vs", "versus", "better", "contrast"}
        if any(w in query.lower().split() for w in _comparison_words):
            return settings.CANDIDATE_POOL_COMPARATIVE
        if intent == IntentType.ASKING_INFO:
            return settings.CANDIDATE_POOL_KEYWORD
        if intent == IntentType.EXPRESSING_EMOTION:
            return settings.CANDIDATE_POOL_THEMATIC
        return settings.CANDIDATE_POOL_DEFAULT

    @staticmethod
    def _local_dev_model_dir() -> Path:
        """Return <backend>/models — the local dev model cache directory."""
        return Path(__file__).resolve().parent.parent / "models"

    # ------------------------------------------------------------------
    # Embedding utilities
    # ------------------------------------------------------------------

    def _load_model(self, attr_name: str, model_cls: type, subdir: str, hub_name: str, label: str) -> None:
        """Generic 3-step model loader: Docker path -> dev cache -> Hub download."""
        if getattr(self, attr_name) is not None:
            return
        try:
            # 1. Docker production path
            local_path = f"/app/models/{subdir}"
            if os.path.isdir(local_path):
                logger.info(f"RAGPipeline: loading {label} from LOCAL path: {local_path}")
                setattr(self, attr_name, model_cls(local_path))
                return

            # 2. Local dev cache path
            dev_path = str(self._local_dev_model_dir() / subdir)
            if os.path.isdir(dev_path) and os.listdir(dev_path):
                logger.info(f"RAGPipeline: loading {label} from DEV cache: {dev_path}")
                setattr(self, attr_name, model_cls(dev_path))
                return

            # 3. Download from Hub (first time only), then cache locally
            logger.info(f"RAGPipeline: downloading {label} '{hub_name}' from Hub (one-time)")
            model = model_cls(hub_name)
            os.makedirs(dev_path, exist_ok=True)
            model.save(dev_path)
            setattr(self, attr_name, model)
            logger.info(f"RAGPipeline: cached {label} to {dev_path}")
        except Exception as exc:
            logger.exception(f"RAGPipeline: failed to load {label}: {exc}")
            setattr(self, attr_name, None)

    def _ensure_embedding_model(self) -> None:
        if self._embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        if settings.EMBEDDING_ONNX_ENABLED:
            try:
                providers = _get_onnx_providers()
                logger.info(
                    "RAGPipeline: loading embedding model with ONNX backend "
                    f"(providers={providers})"
                )
                self._embedding_model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    backend="onnx",
                    model_kwargs={"provider": providers[0]},
                )
                return
            except Exception as e:
                logger.warning(f"ONNX embedding load failed, falling back to PyTorch: {e}")
        self._load_model("_embedding_model", SentenceTransformer, "embeddings", settings.EMBEDDING_MODEL, "embedding model")

    def _ensure_reranker_model(self) -> None:
        if not settings.RERANKER_ENABLED:
            self._reranker_model = None
            return
        if self._reranker_model is not None:
            return

        _local_reranker = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "reranker"
        )
        _docker_reranker = "/app/models/reranker"
        _model_dir = _docker_reranker if os.path.isdir(_docker_reranker) else _local_reranker

        # 1. Try ONNX on CPU (preferred — no VRAM contention)
        _onnx_path = os.path.join(_model_dir, "onnx", "model.onnx")
        if os.path.exists(_onnx_path):
            try:
                from rag.onnx_reranker import ONNXReranker
                self._reranker_model = ONNXReranker(_model_dir)
                logger.info("RAGPipeline: reranker loaded via ONNX (CPU)")
                return
            except Exception as e:
                logger.warning(f"ONNX reranker load failed, falling back to PyTorch: {e}")

        # 2. PyTorch fallback — force CPU to prevent VRAM contention with embedding model
        from sentence_transformers import CrossEncoder
        logger.info(f"RAGPipeline: loading reranker via PyTorch (CPU) from {_model_dir}")
        try:
            if os.path.isdir(_model_dir) and os.listdir(_model_dir):
                self._reranker_model = CrossEncoder(_model_dir, device="cpu")
            else:
                self._reranker_model = CrossEncoder(settings.RERANKER_MODEL, device="cpu")
            logger.info(f"RAGPipeline: reranker loaded on {self._reranker_model.device}")
        except Exception as exc:
            logger.exception(f"RAGPipeline: failed to load reranker: {exc}")
            self._reranker_model = None

    def _needs_instruction_prefix(self) -> bool:
        """Check if the embedding model requires query/passage prefixes (e.g., E5 models)."""
        return "e5" in settings.EMBEDDING_MODEL.lower()

    async def generate_embeddings(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Public utility used by /api/embeddings/generate and search.
        For E5 models, automatically prepends 'query: ' or 'passage: ' prefix.
        """
        self._ensure_embedding_model()
        if self._embedding_model is None:
            dim = self.dim or settings.EMBEDDING_DIM
            logger.warning("RAGPipeline: embedding model unavailable, returning zeros")
            return np.zeros((dim,), dtype="float32")

        # Normalize text: strip whitespace, replace newlines
        clean_text = text.strip().replace("\n", " ")

        # E5 models require instruction prefix for optimal performance
        if self._needs_instruction_prefix():
            prefix = "query: " if is_query else "passage: "
            clean_text = prefix + clean_text

        async with self._embed_sem:
            vec = (await asyncio.to_thread(
                self._embedding_model.encode, [clean_text],
                convert_to_tensor=False, show_progress_bar=False,
            ))[0]
        return np.asarray(vec, dtype="float32")

    async def generate_embeddings_batch(self, texts: list, is_query: bool = True) -> list:
        """Batch-encode multiple texts in a single model.encode() call.
        Far faster than N separate generate_embeddings() calls (avoids GIL serialization)."""
        self._ensure_embedding_model()
        if self._embedding_model is None:
            dim = self.dim or settings.EMBEDDING_DIM
            return [np.zeros((dim,), dtype="float32") for _ in texts]
        prefix = ""
        if self._needs_instruction_prefix():
            prefix = "query: " if is_query else "passage: "
        clean_texts = [prefix + t.strip().replace("\n", " ") for t in texts]
        async with self._embed_sem:
            vecs = await asyncio.to_thread(
                self._embedding_model.encode, clean_texts,
                convert_to_tensor=False, show_progress_bar=False, batch_size=len(clean_texts),
            )
        return [np.asarray(v, dtype="float32") for v in vecs]

    # ------------------------------------------------------------------
    # Query Expansion
    # ------------------------------------------------------------------

    # Words that should never trigger LLM-based query expansion
    _SKIP_EXPANSION = TRIVIAL_MESSAGES

    # ------------------------------------------------------------------
    # Hindi / Devanagari detection and translation
    # ------------------------------------------------------------------

    def _is_devanagari_text(self, text: str) -> bool:
        """Check if >50% of non-whitespace chars are Devanagari (for adaptive BM25 weighting)."""
        if not text:
            return False
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return False
        devanagari = sum(1 for c in chars if '\u0900' <= c <= '\u097F')
        return devanagari / len(chars) > 0.5

    def _is_hindi_or_devanagari(self, text: str) -> bool:
        """Check if >30% of chars are Devanagari."""
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return False
        devanagari = sum(1 for c in chars if '\u0900' <= c <= '\u097F')
        return devanagari / len(chars) > 0.3

    def _is_transliterated_hindi(self, text: str) -> bool:
        """Detect Hinglish via common Hindi marker words in Latin script."""
        hindi_markers = {"kya", "hai", "kaise", "kare", "mein", "ka", "ki", "ke",
                         "ko", "se", "hota", "nahi", "bahut", "aur", "bhi",
                         "mandir", "bhagwan", "puja", "dhyan", "katha", "arth",
                         "raasta", "tarika", "mahatva", "matlab", "sunao",
                         "batao", "gita", "ram", "raha", "rahi"}
        words = set(text.lower().split())
        return len(words & hindi_markers) >= 2

    async def _translate_query(self, query: str) -> str:
        """Translate Hindi/Hinglish query to English via Gemini Flash. Cached 24h."""
        if not self._is_hindi_or_devanagari(query) and not self._is_transliterated_hindi(query):
            return query
        if not self._llm.available:
            return query

        cache = get_cache_service()
        cache_key = query.strip().lower()

        # Persistent disk cache (survives Redis restarts)
        _disk_key = hashlib.md5(cache_key.encode()).hexdigest()
        _disk_path = settings.TRANSLATION_CACHE_PATH
        try:
            with shelve.open(_disk_path) as db:
                if _disk_key in db:
                    logger.info("Translation disk-cache HIT")
                    return db[_disk_key]
        except Exception:
            pass

        cached = await cache.get("query_translation", query=cache_key)
        if cached:
            return cached

        prompt = f'Translate this Hindi/Hinglish spiritual query to English. Return ONLY the English translation.\nQuery: "{_sanitize_for_prompt(query)}"'
        try:
            def _sync():
                return bedrock_generate(
                    prompt, model=settings.GEMINI_FAST_MODEL,
                    temperature=settings.QUERY_TRANSLATE_TEMPERATURE, max_tokens=100)
            response = await asyncio.to_thread(_sync)
            translation = (response.text or "").strip()
            if translation:
                logger.info(f"Query translated: '{query[:40]}' -> '{translation[:60]}'")
                await cache.set("query_translation", translation, ttl=86400, query=cache_key)
                # Write to disk cache
                try:
                    os.makedirs(os.path.dirname(_disk_path), exist_ok=True)
                    with shelve.open(_disk_path) as db:
                        db[_disk_key] = translation
                except Exception:
                    pass
                return translation
        except Exception as e:
            logger.error(f"Query translation failed: {e}")
        return query

    # ------------------------------------------------------------------
    # Query Expansion
    # ------------------------------------------------------------------

    async def _expand_query(self, query: str, intent: Optional[str] = None, life_domain: Optional[str] = None) -> List[str]:
        """Use LLM to expand the query for better recall.
        Skips expansion for greetings and trivial messages to save a Gemini call.
        Results are cached for 24 hours.
        """
        # Fast-path: skip expansion for greetings / trivial messages
        if query.strip().lower() in self._SKIP_EXPANSION:
            return [query]

        if not self._llm.available:
            return [query]

        # Check cache first
        cache = get_cache_service()
        cache_key = query.strip().lower()
        cached = await cache.get("query_expansion", query=cache_key)
        if cached:
            return cached

        intent_hint = f"\nUser's intent: {intent}" if intent else ""
        domain_hint = f"\nLife domain: {life_domain}" if life_domain else ""
        prompt = f"""Expand the following spiritual query into 2 alternative search terms that capture different aspects.{intent_hint}{domain_hint}
Original: "{_sanitize_for_prompt(query)}"
If emotional intent, focus on comfort and healing verses. If informational, focus on specific teachings.
Respond ONLY with 2 terms, separated by a newline."""
        try:
            # Use the fast model for query expansion (lightweight task)
            def _sync_expand():
                return bedrock_generate(
                    prompt, model=settings.GEMINI_FAST_MODEL,
                    temperature=settings.QUERY_EXPAND_TEMPERATURE, max_tokens=100)

            response = await asyncio.to_thread(_sync_expand)
            expansion = response.text if response.text else ""
            expanded_terms = [t.strip() for t in expansion.split("\n") if t.strip()]
            result = [query] + expanded_terms[:2]
            logger.info(f"Query expansion: {expanded_terms}")

            # Cache for 24 hours
            await cache.set("query_expansion", result, ttl=86400, query=cache_key)

            return result
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    # ------------------------------------------------------------------
    # Long Query Summarization
    # ------------------------------------------------------------------

    async def _summarize_long_query(self, query: str) -> str:
        """Summarize long queries to core spiritual question for better embedding."""
        if not self._llm.available:
            return query

        cache = get_cache_service()
        cache_key = query.strip().lower()[:200]
        cached = await cache.get("query_summarize", query=cache_key)
        if cached:
            return cached

        prompt = (
            "Extract the core spiritual question from this text in under 15 words. "
            "Keep key concepts and emotions. Return ONLY the summary.\n"
            f'Text: "{_sanitize_for_prompt(query)}"'
        )
        try:
            def _sync():
                return bedrock_generate(
                    prompt, model=settings.GEMINI_FAST_MODEL,
                    temperature=settings.QUERY_SUMMARIZE_TEMPERATURE, max_tokens=50)
            response = await asyncio.to_thread(_sync)
            summary = (response.text or "").strip()
            if summary and len(summary.split()) <= 20:
                await cache.set("query_summarize", summary, ttl=86400, query=cache_key)
                logger.info(f"Long query summarized: '{query[:40]}...' → '{summary}'")
                return summary
        except Exception as e:
            logger.error(f"Query summarization failed: {e}")
        return query

    # ------------------------------------------------------------------
    # HyDE (Hypothetical Document Embedding)
    # ------------------------------------------------------------------

    def _should_use_hyde(self, query: str, intent: Optional[IntentType] = None) -> bool:
        """Decide whether to generate hypothetical document embeddings for this query.
        HyDE adds 4-6s latency (Gemini Flash API + embedding). Only use for
        ambiguous or complex queries where the embedding model struggles."""
        if not settings.HYDE_ENABLED:
            return False
        if not self._llm.available:
            return False
        # Skip trivial/single-word queries
        if query.strip().lower() in self._SKIP_EXPANSION or len(query.split()) < 2:
            return False
        # Skip intents where HyDE adds no value
        skip_intents = {
            IntentType.GREETING, IntentType.CLOSURE,
            IntentType.PRODUCT_SEARCH, IntentType.ASKING_PANCHANG,
        }
        if intent in skip_intents:
            return False
        # Skip specific verse reference requests (e.g. "2.47", "Gita 3.19")
        if re.search(r'\d+\.\d+', query):
            return False
        # Skip short, clear queries — HyDE helps ambiguous queries, not direct ones.
        # Use raw message (before first | separator) for word count.
        raw_part = query.split(" | ")[0] if " | " in query else query
        raw_words = len(raw_part.strip().split())
        _clear_intents = {IntentType.SEEKING_GUIDANCE, IntentType.ASKING_INFO}
        # Also skip when intent is None (speculative RAG) and query is short
        if raw_words <= 12 and (intent in _clear_intents or intent is None):
            logger.info(f"HyDE: skipped for short clear query ({raw_words} words, intent={intent})")
            return False
        return True

    async def _generate_hyde(self, query: str, life_domain: Optional[str] = None) -> List[np.ndarray]:
        """Generate hypothetical scripture passages and embed them for HyDE retrieval."""
        cache = get_cache_service()
        cache_key = query.strip().lower()

        # Check cache for text passages (not embeddings — model changes invalidate)
        cached_passages = await cache.get("hyde_passages", query=cache_key)
        if cached_passages and isinstance(cached_passages, list):
            passages = cached_passages
        else:
            domain_hint = f"\nTheir life area: {life_domain}." if life_domain else ""
            prompt = (
                f'A person asks: "{_sanitize_for_prompt(query)}"{domain_hint}\n\n'
                f'Write {settings.HYDE_COUNT} short scripture-style passages (2-3 sentences each) '
                f'that would perfectly answer this question. Write as if excerpts from '
                f'Bhagavad Gita, Upanishads, Yoga Sutras, or Ramayana. Include the spiritual '
                f'teaching and its practical meaning. Separate passages with "---".'
            )
            try:
                _t_hyde_llm = time.perf_counter()
                def _sync():
                    return bedrock_generate(
                        prompt, model=settings.GEMINI_FAST_MODEL,
                        temperature=settings.HYDE_TEMPERATURE, max_tokens=500)
                response = await asyncio.to_thread(_sync)
                _hyde_llm_ms = (time.perf_counter() - _t_hyde_llm) * 1000
                logger.info(f"PERF_HYDE gemini_call={_hyde_llm_ms:.0f}ms")
                raw = (response.text or "").strip()
                passages = [p.strip() for p in raw.split("---") if p.strip()]
                passages = passages[:settings.HYDE_COUNT]
                if passages:
                    await cache.set("hyde_passages", passages, ttl=settings.HYDE_CACHE_TTL, query=cache_key)
                    logger.info(f"HyDE: generated {len(passages)} hypothetical passage(s) for '{query[:40]}'")
            except Exception as e:
                logger.error(f"HyDE generation failed: {e}")
                return []

        if not passages:
            return []

        # Batch-embed HyDE passages (single encode call, not N separate calls)
        hyde_vecs = await self.generate_embeddings_batch(passages, is_query=False)
        if hyde_vecs:
            logger.info(f"HyDE: {len(hyde_vecs)} hypothetical embedding(s) added to search pool")
        return hyde_vecs

    # ------------------------------------------------------------------
    # Adaptive Fusion Weights
    # ------------------------------------------------------------------

    _KEYWORD_HEAVY_DOMAINS = frozenset({"ayurveda", "ayurveda_specific", "yoga", "health"})
    _EMOTIONAL_DOMAINS = frozenset({
        "grief", "shame", "fear", "guilt", "loneliness",
        "hopelessness", "anger", "confusion",
    })

    def _get_fusion_weights(
        self,
        query: str,
        intent: Optional[IntentType],
        life_domain: Optional[str],
        is_devanagari_query: bool,
        is_hinglish_query: bool,
        devanagari_result_ratio: float,
    ) -> tuple:
        """Compute adaptive semantic/BM25 fusion weights based on query characteristics."""
        sem_w = 0.80  # base
        reasons = []

        # Language-based adjustments
        if is_devanagari_query:
            sem_w += 0.08
            reasons.append("devanagari_query+0.08")
        elif is_hinglish_query:
            sem_w += 0.04
            reasons.append("hinglish_query+0.04")

        if devanagari_result_ratio >= 0.7:
            sem_w += 0.05
            reasons.append("devanagari_results+0.05")

        # Intent-based adjustments
        if intent == IntentType.EXPRESSING_EMOTION:
            sem_w += 0.05
            reasons.append("emotion_intent+0.05")
        elif intent == IntentType.ASKING_INFO:
            sem_w -= 0.05
            reasons.append("info_intent-0.05")

        # Domain-based adjustments
        domain_lower = (life_domain or "").lower()
        if domain_lower in self._KEYWORD_HEAVY_DOMAINS:
            sem_w -= 0.05
            reasons.append(f"keyword_domain({domain_lower})-0.05")
        elif domain_lower in self._EMOTIONAL_DOMAINS:
            sem_w += 0.03
            reasons.append(f"emotional_domain({domain_lower})+0.03")

        # Query length adjustments
        word_count = len(query.split())
        if word_count <= 2:
            sem_w -= 0.04
            reasons.append("short_query-0.04")
        elif word_count >= 8:
            sem_w += 0.03
            reasons.append("long_query+0.03")

        # Clamp
        sem_w = max(0.50, min(0.98, sem_w))
        bm25_w = 1.0 - sem_w

        metadata = {"sem_w": sem_w, "bm25_w": bm25_w, "reasons": reasons}
        return sem_w, bm25_w, metadata

    # ------------------------------------------------------------------
    # Parent-Child Verse Retrieval
    # ------------------------------------------------------------------

    def _build_neighbor_index(self) -> None:
        """Build (scripture, chapter, section) → sorted [(verse_number_int, global_index)] index
        for O(log n) adjacent verse lookups via bisect.
        """
        import bisect as _bisect  # noqa: F841

        _SKIP_TYPES = {"temple", "curated_concept", "curated_narrative", "meditation_template"}
        neighbor_idx: Dict[tuple, list] = {}
        skipped = 0

        for i, v in enumerate(self.verses):
            vtype = (v.get("type") or v.get("source") or "scripture").lower()
            if vtype in _SKIP_TYPES:
                skipped += 1
                continue
            vn = str(v.get("verse_number") or "")
            if not vn.isdigit():
                skipped += 1
                continue
            key = (
                v.get("scripture", ""),
                str(v.get("chapter") or ""),
                str(v.get("section") or ""),
            )
            if key not in neighbor_idx:
                neighbor_idx[key] = []
            neighbor_idx[key].append((int(vn), i))

        # Sort each group by verse_number
        for key in neighbor_idx:
            neighbor_idx[key].sort(key=lambda x: x[0])

        self._neighbor_index = neighbor_idx
        total_indexed = sum(len(v) for v in neighbor_idx.values())
        logger.info(
            f"Neighbor index: {len(neighbor_idx)} groups, "
            f"{total_indexed} verses indexed, {skipped} skipped"
        )

    def _expand_with_neighbors(
        self,
        results: List[Dict],
        expand_top_n: int = 2,
        window: int = 1,
    ) -> List[Dict]:
        """Expand the top N results with ±window adjacent verses for narrative continuity."""
        import bisect

        if not hasattr(self, '_neighbor_index') or not self._neighbor_index:
            return results

        _SKIP_TYPES = {"temple", "curated_concept", "curated_narrative", "meditation_template"}
        seen_indices = {r.get("_global_idx") for r in results if r.get("_global_idx") is not None}
        expanded = []
        total_added = 0

        for rank, doc in enumerate(results):
            expanded.append(doc)

            if rank >= expand_top_n:
                continue

            doc_type = (doc.get("type") or doc.get("source") or "scripture").lower()
            if doc_type in _SKIP_TYPES:
                continue

            vn = str(doc.get("verse_number") or "")
            if not vn.isdigit():
                continue
            vn_int = int(vn)

            key = (
                doc.get("scripture", ""),
                str(doc.get("chapter") or ""),
                str(doc.get("section") or ""),
            )
            group = self._neighbor_index.get(key)
            if not group:
                continue

            # Binary search for position
            verse_numbers = [g[0] for g in group]
            pos = bisect.bisect_left(verse_numbers, vn_int)

            # Collect ±window neighbors
            neighbors = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                npos = pos + offset
                if 0 <= npos < len(group):
                    neighbor_vn, neighbor_global_idx = group[npos]
                    if abs(neighbor_vn - vn_int) <= window and neighbor_global_idx not in seen_indices:
                        neighbors.append((neighbor_vn, neighbor_global_idx))
                        seen_indices.add(neighbor_global_idx)

            parent_ref = doc.get("reference", "")
            parent_score = get_doc_score(doc)

            for _nvn, nidx in neighbors:
                neighbor_verse = self.verses[nidx].copy()
                neighbor_verse["is_context_verse"] = True
                neighbor_verse["context_parent_ref"] = parent_ref
                neighbor_verse["score"] = parent_score * settings.CONTEXT_VERSE_SCORE_RATIO
                neighbor_verse["_global_idx"] = nidx
                expanded.append(neighbor_verse)
                total_added += 1

        # Cross-reference expansion: add top cross-scripture matches as context verses
        if hasattr(self, '_cross_refs') and self._cross_refs:
            cross_added = 0
            for rank, doc in enumerate(results[:expand_top_n]):
                ref = doc.get("reference", "")
                cross_list = self._cross_refs.get(ref, [])
                for cross_entry in cross_list[:2]:  # max 2 cross-refs per primary
                    cross_ref = cross_entry.get("reference", "") if isinstance(cross_entry, dict) else cross_entry
                    cross_ref_lower = cross_ref.lower() if cross_ref else ""
                    if cross_ref_lower in (getattr(self, '_reference_index', {}) or {}):
                        cross_idx = self._reference_index[cross_ref_lower]
                        if cross_idx not in seen_indices:
                            seen_indices.add(cross_idx)
                            cross_verse = self.verses[cross_idx].copy()
                            cross_verse["is_context_verse"] = True
                            cross_verse["context_parent_ref"] = ref
                            cross_verse["cross_scripture_ref"] = True
                            parent_score = get_doc_score(doc)
                            cross_verse["score"] = parent_score * settings.CONTEXT_VERSE_SCORE_RATIO
                            cross_verse["_global_idx"] = cross_idx
                            expanded.append(cross_verse)
                            cross_added += 1
            if cross_added:
                total_added += cross_added
                logger.info(f"Cross-ref expansion: added {cross_added} cross-scripture verse(s)")

        if total_added:
            logger.info(f"Neighbor expansion: added {total_added} total context verse(s)")
        return expanded

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def _cosine_similarities(self, query_vec: np.ndarray) -> np.ndarray:
        if self.embeddings is None or not self.available:
            return np.zeros((0,), dtype="float32")

        # Normalize the query vector
        q = query_vec.astype("float32")
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Dot product with pre-normalized document vectors
        return self.embeddings @ q

    def _precompute_bm25_stats(self) -> None:
        """Pre-compute BM25 index at startup using rank_bm25."""
        if not self.verses:
            return

        from rank_bm25 import BM25Okapi

        corpus = []
        meanings_found = 0
        for verse in self.verses:
            parts = [
                verse.get("text") or "",
                verse.get("meaning") or "",
                verse.get("translation") or "",
                verse.get("topic") or "",
                verse.get("scripture") or "",
            ]
            if verse.get("meaning"):
                meanings_found += 1
            text = " ".join(parts).lower()
            corpus.append([_simple_stem(w) for w in text.split() if len(w) > 1 and w not in _BM25_STOPWORDS])

        self._bm25 = BM25Okapi(corpus)
        meanings_pct = round(100 * meanings_found / len(corpus), 1) if corpus else 0
        logger.info(
            f"BM25 index built for {len(corpus)} documents "
            f"(meanings={meanings_found}/{len(corpus)} = {meanings_pct}%)"
        )

    def _keyword_score_bm25(self, query: str) -> np.ndarray:
        """BM25 keyword scoring via rank_bm25 (vectorised, replaces per-doc loop)."""
        if not self.verses:
            return np.zeros((0,))

        query_tokens = [_simple_stem(t.lower()) for t in query.split() if len(t) > 1 and t.lower() not in _BM25_STOPWORDS]
        if not query_tokens:
            return np.zeros(len(self.verses))

        # Lazy fallback if not pre-computed (shouldn't happen after startup)
        if not hasattr(self, "_bm25"):
            self._precompute_bm25_stats()

        scores = self._bm25.get_scores(query_tokens)

        # Normalize scores to [0, 1] for fusion
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score

        return scores

    def _keyword_score_splade(self, query: str) -> np.ndarray:
        """SPLADE sparse retrieval scoring (when enabled and index available)."""
        if not hasattr(self, '_splade_index') or self._splade_index is None:
            return self._keyword_score_bm25(query)
        if not hasattr(self, '_splade_model') or self._splade_model is None:
            return self._keyword_score_bm25(query)

        try:
            import torch
            tokens = self._splade_tokenizer(
                [query], return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            with torch.no_grad():
                output = self._splade_model(**tokens)
                logits = output.logits
                splade_rep = torch.log1p(torch.relu(logits))
                attention_mask = tokens["attention_mask"].unsqueeze(-1)
                query_sparse = (splade_rep * attention_mask).max(dim=1).values

            from scipy.sparse import csr_matrix
            query_csr = csr_matrix(query_sparse.cpu().numpy())
            scores = np.array((self._splade_index @ query_csr.T).todense()).flatten()

            max_score = np.max(scores) if scores.size > 0 else 0
            if max_score > 0:
                scores = scores / max_score
            return scores
        except Exception as e:
            logger.warning(f"SPLADE scoring failed, falling back to BM25: {e}")
            return self._keyword_score_bm25(query)

    async def _rerank_results(self, query: str, results: List[Dict], intent: Optional[IntentType] = None, life_domain: Optional[str] = None) -> List[Dict]:
        """Neural re-ranking for higher precision with intent-based weighting"""
        if not results:
            return results

        _t0 = time.perf_counter()

        # Check reranker cache before running CrossEncoder
        cache = get_cache_service()
        _doc_refs = tuple(sorted(r.get("reference", str(i)) for i, r in enumerate(results)))
        _rerank_cache_key = f"rerank:{query.strip().lower()}:{_doc_refs}:{intent}:{life_domain or ''}"
        try:
            cached_scores = await cache.get("rerank", key=_rerank_cache_key)
            if cached_scores is not None:
                for i, score_data in enumerate(cached_scores):
                    if i < len(results):
                        results[i]["rerank_score"] = score_data["rerank_score"]
                        results[i]["final_score"] = score_data["final_score"]
                results.sort(key=lambda x: x["final_score"], reverse=True)
                logger.info("Reranker cache HIT — skipped CrossEncoder inference")
                return results
        except Exception:
            pass  # Cache miss or unavailable — proceed with reranking

        _t1 = time.perf_counter()
        logger.info(f"PERF_RERANK cache_check={round((_t1-_t0)*1000)}ms")

        self._ensure_reranker_model()

        # Cap candidates to reduce reranking time
        if len(results) > settings.MAX_RERANK_CANDIDATES:
            results = results[:settings.MAX_RERANK_CANDIDATES]

        # Pairs for cross-encoder
        # Include topic as semantic hint when meaning is absent (97% of verses)
        pairs = []
        for doc in results:
            parts = [doc.get('text', ''), doc.get('meaning', '')]
            if not doc.get('meaning'):
                parts.append(doc.get('topic', ''))
            content = " ".join(p for p in parts if p)
            pairs.append([query, content])

        _t2 = time.perf_counter()
        logger.info(f"PERF_RERANK pairs_built={round((_t2-_t1)*1000)}ms n_pairs={len(pairs)}")

        try:
            # Cross-encoder scores (duck-typing supports API-based rerankers)
            if self._reranker_model is not None:
                if hasattr(self._reranker_model, 'predict'):
                    re_scores = await asyncio.to_thread(self._reranker_model.predict, pairs)
                elif hasattr(self._reranker_model, 'rerank'):
                    documents = [pair[1] for pair in pairs]
                    re_scores = self._reranker_model.rerank(query, documents)
                else:
                    re_scores = [0.0] * len(results)
            else:
                re_scores = [0.0] * len(results)

            _t3 = time.perf_counter()
            logger.info(f"PERF_RERANK predict={round((_t3-_t2)*1000)}ms model={type(self._reranker_model).__name__}")

            # Attach and re-sort
            for i, score in enumerate(re_scores):
                results[i]["rerank_score"] = float(score)

                # Sigmoid-normalize reranker score to [0, 1]
                norm_rerank = 1.0 / (1.0 + math.exp(-float(score)))

                # Dynamic Intent-Based Weighting
                weighting_adjustment = 0.0
                doc_type = results[i].get("type", "scripture")
                doc_text = (results[i].get("text", "") + " " + results[i].get("reference", "")).lower()

                # 1. Penalize Temple/Locations for non-spatial intents
                spatial_keywords = ["maidan", "ground", "complex", "road", "street", "near"]
                is_spatial = any(k in doc_text for k in spatial_keywords) or doc_type == "temple"

                query_lower = query.lower()
                is_story_request = any(k in query_lower for k in ["story", "legend", "tale", "katha", "parable"])

                if intent in (IntentType.SEEKING_GUIDANCE, IntentType.EXPRESSING_EMOTION, IntentType.OTHER) or is_story_request:
                    if is_spatial:
                        # Purely spatial (address/directions) gets full penalty; temple knowledge docs get softer penalty
                        _pure_spatial_markers = {"located", "address", "km", "bus", "railway", "airport", "distance", "route"}
                        is_pure_spatial = any(k in doc_text for k in _pure_spatial_markers)
                        weighting_adjustment -= 0.8 if is_pure_spatial else 0.3
                    if doc_type == "procedural" or doc_type == "scripture":
                        weighting_adjustment += 0.5

                # 2. Boost Procedural for Guidance
                if intent == IntentType.SEEKING_GUIDANCE and doc_type == "procedural":
                    weighting_adjustment += 0.3

                # 3. Boost Temples ONLY for specific intents
                if intent == IntentType.ASKING_INFO and is_spatial:
                    weighting_adjustment += 0.5

                # 4. Explicit "How-to" boost for procedural rituals
                is_howto = any(k in query_lower for k in ["how", "step", "procedure", "ritual", "guide", "method"])
                if is_howto and doc_type == "procedural":
                    weighting_adjustment += 0.5

                # Domain-aware scripture affinity boost
                if life_domain and life_domain.lower() in DOMAIN_SCRIPTURE_AFFINITY:
                    doc_scripture_lower = (results[i].get("scripture") or "").lower()
                    for scripture_key, boost in DOMAIN_SCRIPTURE_AFFINITY[life_domain.lower()].items():
                        if scripture_key in doc_scripture_lower:
                            weighting_adjustment += boost
                            break

                # Cap intent adjustment to [-1.0, +1.0]
                weighting_adjustment = max(-1.0, min(1.0, weighting_adjustment))

                # Bounded final score: blend retrieval + normalized reranker, then scale by intent
                base_score = ((1 - settings.RERANKER_WEIGHT) * results[i]["score"]) + (settings.RERANKER_WEIGHT * norm_rerank)
                results[i]["final_score"] = base_score * (1.0 + settings.INTENT_WEIGHT_SCALE * weighting_adjustment)

                # Tradition bonus: slight nudge for docs with labeled tradition metadata
                doc_tradition = (results[i].get("tradition") or "").lower()
                if doc_tradition and doc_tradition != "general":
                    results[i]["final_score"] += settings.TRADITION_BONUS

            # Sort by final score
            results.sort(key=lambda x: x["final_score"], reverse=True)
            logger.info(f"Neural re-ranking with intent='{intent}' complete")

            # Cache reranker scores for future hits
            try:
                _score_data = [{"rerank_score": r.get("rerank_score", 0), "final_score": r["final_score"]} for r in results]
                await cache.set("rerank", _score_data, ttl=settings.RAG_SEARCH_CACHE_TTL, key=_rerank_cache_key)
            except Exception:
                pass  # Cache write failure is non-fatal

            return results
        except Exception as e:
            logger.error(f"Re-ranking failed, falling back to retrieval-score ordering: {e}")
            # Ensure consistent ordering: sort by original retrieval score
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results

    async def rerank(
        self,
        query: str,
        candidates: List[Dict],
        intent: Optional[IntentType] = None,
        life_domain: Optional[str] = None,
    ) -> List[Dict]:
        """Public reranking entry point for the rerank-once pattern.

        Called by retrieval_judge after merging results from parallel
        skip_rerank=True searches. Caps candidates at MAX_RERANK_CANDIDATES,
        runs CrossEncoder, and returns sorted results.
        """
        if not candidates:
            return candidates
        if len(candidates) > settings.MAX_RERANK_CANDIDATES:
            candidates = candidates[:settings.MAX_RERANK_CANDIDATES]
        return await self._rerank_results(query, candidates, intent=intent, life_domain=life_domain)

    @staticmethod
    def _mmr_diversify(
        results: List[Dict],
        query_embedding: Optional[np.ndarray] = None,
        lambda_param: float = 0.7,
        top_k: int = 7,
    ) -> List[Dict]:
        """Maximal Marginal Relevance — re-order results for relevance AND diversity.

        Score = λ * relevance(doc) - (1-λ) * max_similarity(doc, already_selected)

        λ=1.0 → pure relevance (no diversity), λ=0.5 → balanced, λ=0.7 → recommended
        """
        if not results or len(results) <= 1:
            return results

        selected = [results[0]]  # Always pick the most relevant first
        remaining = list(results[1:])

        while remaining and len(selected) < top_k:
            best_score = -float('inf')
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance component: use final_score from reranking
                relevance = candidate.get("final_score", candidate.get("score", 0))

                # Diversity component: penalize similarity to already-selected docs
                max_sim_to_selected = 0.0
                cand_scripture = (candidate.get("scripture") or "").lower()
                cand_topic = (candidate.get("topic") or "").lower()
                cand_ref = (candidate.get("reference") or "").lower()

                for sel in selected:
                    sel_scripture = (sel.get("scripture") or "").lower()
                    sel_topic = (sel.get("topic") or "").lower()
                    sel_ref = (sel.get("reference") or "").lower()

                    # Content similarity: same scripture + same topic = very similar
                    sim = 0.0
                    if cand_scripture == sel_scripture:
                        sim += 0.4
                    if cand_topic == sel_topic:
                        sim += 0.3
                    if cand_ref == sel_ref:
                        sim = 1.0  # exact same verse = maximum penalty

                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    async def search(
        self,
        query: str,
        scripture_filter: Optional[List[str]] = None,
        language: str = "en",
        top_k: int = settings.RETRIEVAL_TOP_K,
        intent: Optional[IntentType] = None,
        min_score: float = settings.MIN_SIMILARITY_SCORE,
        doc_type_filter: Optional[List[str]] = None,
        life_domain: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
        exclude_references: Optional[List[str]] = None,
        skip_rerank: bool = False,
        response_mode: Optional[str] = None,
    ) -> List[Dict]:
        """
        Advanced RAG Search:
        1. Contextual Query Expansion
        2. Hybrid Search (Semantic + BM25)
        3. Initial Candidate Retrieval
        4. Neural Re-ranking with Intent-Based Weighting
        """
        if not self.available or not self.verses:
            return []

        if not query.strip():
            return []

        # Citation short-circuit: direct verse reference bypasses full pipeline
        citation_results = self._try_citation_lookup(query)
        if citation_results is not None:
            return citation_results

        # Fix A: Empty scripture filter means "no scriptures match" — skip search entirely
        if scripture_filter is not None and len(scripture_filter) == 0:
            logger.info("Scripture pre-filter: empty allowed list — skipping RAG search")
            return []

        # --- Full RAG search caching ---
        cache = get_cache_service()
        _cache_params = {
            "q": query.strip().lower(),
            "intent": str(intent) if intent else "",
            "domain": life_domain or "",
            "scripture": ",".join(sorted(scripture_filter)) if scripture_filter else "",
            "min_score": str(min_score),
            "doc_type": ",".join(sorted(doc_type_filter)) if doc_type_filter else "",
            "top_k": str(top_k),
        }
        cached = await cache.get("rag_search", **_cache_params)
        if cached is not None:
            logger.info(f"RAG search cache HIT for query='{query[:40]}'")
            return cached

        _t_start = time.perf_counter()
        _timings = {}

        # Query normalization (NFKC + whitespace) is delegated to
        # services/query_normalizer.py. The normalizer never substitutes user
        # tokens and never appends domain-specific synonyms — bilingual /
        # synonym handling lives in IntentAgent.query_variants which threads
        # through this method's ``query_variants`` parameter.
        normalized = self._query_normalizer.normalize(query)
        query = normalized.normalized

        # Auto-exclude temple docs for non-temple queries
        if doc_type_filter is None and intent is not None:
            query_lower = query.lower()
            if not any(kw in query_lower for kw in _TEMPLE_KEYWORDS):
                doc_type_filter = ["temple"]

        # === Unified Preprocessing: translate + expand + HyDE (all parallel) ===
        # Key: query is NEVER overwritten with translation.
        # query = original (for reranker + HyDE + cache)
        # bm25_query = English translation (for BM25 keyword scoring only)
        original_query = query
        bm25_query = query

        # Language detection flags (reused by fusion weights)
        is_devanagari_query = self._is_hindi_or_devanagari(query)
        is_hinglish_query = not is_devanagari_query and self._is_transliterated_hindi(query)
        needs_translation = is_devanagari_query or is_hinglish_query

        # Use pre-computed variants from IntentAgent if available (saves a Gemini Flash round-trip)
        # IntentAgent is the single source of bilingual / synonym variants —
        # no local hardcoded vocabulary lookup. If no variants were provided
        # upstream, fall back to expanding short queries via the LLM path.
        _has_precomputed_variants = bool(query_variants)
        needs_expansion = (
            settings.QUERY_EXPANSION_ENABLED
            and not _has_precomputed_variants
            and len(query.split()) < 6
            and query.strip().lower() not in self._SKIP_EXPANSION
            and self._llm.available
        )

        # Launch ALL async preprocessing in parallel
        _t_preprocess = time.perf_counter()
        _async_tasks = {}
        if needs_translation:
            _async_tasks['translate'] = self._translate_query(query)
        if needs_expansion:
            _async_tasks['expand'] = self._expand_query(query, intent=intent.value if intent else None, life_domain=life_domain)
        # Use raw message length (before | enrichment) to avoid inflated word count
        _raw_query = query.split(" | ")[0] if " | " in query else query
        if settings.LONG_QUERY_SUMMARIZATION_ENABLED and len(_raw_query.split()) > settings.LONG_QUERY_THRESHOLD:
            _async_tasks['summarize'] = self._summarize_long_query(query)
        if self._should_use_hyde(query, intent=intent):
            _async_tasks['hyde'] = self._generate_hyde(query, life_domain=life_domain)

        if _async_tasks:
            _results = await asyncio.gather(*_async_tasks.values(), return_exceptions=True)
            _result_map = dict(zip(_async_tasks.keys(), _results))
        else:
            _result_map = {}

        # Unpack — translation is AUGMENTATION only, never replaces query
        translated_query = None
        if 'translate' in _result_map and not isinstance(_result_map['translate'], Exception):
            translated_query = _result_map['translate']
            bm25_query = translated_query  # BM25 needs English tokens

        expanded_queries = [query]  # Original always first
        _seen_lower = {query.strip().lower()}  # Case-insensitive dedup set
        # Fix 1: Insert long-query summary as primary embedding target
        if 'summarize' in _result_map and not isinstance(_result_map['summarize'], Exception):
            summary = _result_map['summarize']
            if summary and summary.strip().lower() not in _seen_lower:
                expanded_queries.insert(0, summary)
                _seen_lower.add(summary.strip().lower())
                logger.info(f"Long query: using summary as primary embedding")
        # Use pre-computed query variants from IntentAgent (combined call)
        if _has_precomputed_variants:
            for qv in query_variants:
                if qv and qv.strip().lower() not in _seen_lower:
                    expanded_queries.append(qv)
                    _seen_lower.add(qv.strip().lower())
            logger.info(f"Using {len(query_variants)} pre-computed query variants from IntentAgent")
        elif 'expand' in _result_map and not isinstance(_result_map['expand'], Exception):
            for eq in _result_map['expand']:
                if eq and eq.strip().lower() not in _seen_lower:
                    expanded_queries.append(eq)
                    _seen_lower.add(eq.strip().lower())
        if translated_query and translated_query.strip().lower() not in _seen_lower:
            expanded_queries.append(translated_query)
            _seen_lower.add(translated_query.strip().lower())

        # SCO: Sanskrit Concept Ontology expansion — limited to 1 concept, 2 terms (latency cap)
        try:
            from services.concept_ontology import get_concept_ontology
            sco = get_concept_ontology()
            if sco.available:
                detected = sco.detect_concepts(query)
                for concept in detected[:1]:  # Was [:3] — reduced to cap query count
                    related = sco.get_related_concepts(concept, depth=1, max_results=2)  # Was depth=2, max=3
                    for term in related:
                        term_lower = term.strip().lower()
                        if term_lower not in _seen_lower:
                            expanded_queries.append(term)
                            _seen_lower.add(term_lower)
                if detected:
                    logger.info(f"SCO expansion: detected {detected[:1]}, added related concepts to query pool")
        except Exception as e:
            logger.debug(f"SCO expansion skipped: {e}")

        _timings['preprocess_ms'] = round((time.perf_counter() - _t_preprocess) * 1000)

        hyde_embeddings = []
        if 'hyde' in _result_map and not isinstance(_result_map['hyde'], Exception):
            hyde_embeddings = _result_map['hyde'] or []

        # Batch Execution: Single model.encode() call for all expansions (avoids GIL serialization)
        async def get_all_semantic_scores():
            _t_embed = time.perf_counter()
            vecs = await self.generate_embeddings_batch(expanded_queries)
            _embed_ms = (time.perf_counter() - _t_embed) * 1000
            logger.info(f"PERF_EMBED batch={len(expanded_queries)} {_embed_ms:.0f}ms")
            all_scores = [self._cosine_similarities(v) for v in vecs]
            # Add HyDE pseudo-doc embeddings to MAX-pool
            for hyde_vec in hyde_embeddings:
                all_scores.append(self._cosine_similarities(hyde_vec))
            return np.max(np.array(all_scores), axis=0)

        # 3. Hybrid Search (Parallel EXECUTION)
        _t_search = time.perf_counter()
        semantic_task = get_all_semantic_scores()
        # Use SPLADE when available, otherwise BM25
        _keyword_fn = self._keyword_score_splade if settings.SPLADE_ENABLED and hasattr(self, '_splade_index') and self._splade_index is not None else self._keyword_score_bm25
        keyword_task = asyncio.to_thread(_keyword_fn, bm25_query)

        semantic_scores, keyword_scores = await asyncio.gather(semantic_task, keyword_task)
        _timings['hybrid_search_ms'] = round((time.perf_counter() - _t_search) * 1000)

        # Adaptive fusion weights based on query/intent/language characteristics
        top_sem_indices = np.argpartition(-semantic_scores, min(10, len(semantic_scores) - 1))[:10]
        devanagari_count = sum(
            1 for i in top_sem_indices
            if self._is_devanagari_text(self.verses[int(i)].get('text', ''))
        )
        devanagari_result_ratio = devanagari_count / min(10, len(semantic_scores))

        sem_w, bm25_w, fusion_meta = self._get_fusion_weights(
            query=query, intent=intent, life_domain=life_domain,
            is_devanagari_query=is_devanagari_query, is_hinglish_query=is_hinglish_query,
            devanagari_result_ratio=devanagari_result_ratio,
        )
        # If BM25 produced no matches, shift to pure semantic (avoid diluting signal)
        if np.max(keyword_scores) == 0:
            fused_scores = semantic_scores
            if settings.LOG_FUSION_WEIGHTS:
                logger.info("FUSION_OBS: BM25 all-zeros, using pure semantic scores")
        else:
            fused_scores = (sem_w * semantic_scores) + (bm25_w * keyword_scores)

        if settings.LOG_FUSION_WEIGHTS:
            logger.info(f"FUSION_OBS sem={sem_w:.3f} bm25={bm25_w:.3f} reasons={fusion_meta['reasons']}")

        # Section chunk score boosting: if section-level retrieval finds matching chapters,
        # give a small additive boost to contained verses (item 2.6)
        if (settings.SECTION_CHUNKS_ENABLED
                and hasattr(self, '_section_embeddings') and self._section_embeddings is not None
                and len(self._section_verses) > 0
                and (len(query.split()) > 10 or "chapter" in query.lower())):
            try:
                query_vec = await self.generate_embeddings(query)
                q = query_vec.astype("float32")
                q_norm = np.linalg.norm(q)
                if q_norm > 0:
                    q = q / q_norm
                section_scores = self._section_embeddings @ q
                top_section_idx = int(np.argmax(section_scores))
                if section_scores[top_section_idx] > 0.5:
                    top_section = self._section_verses[top_section_idx]
                    verse_refs = set(r.lower() for r in top_section.get("verse_refs", []))
                    if verse_refs and hasattr(self, '_reference_index'):
                        boosted = 0
                        for ref_lower, vidx in self._reference_index.items():
                            if ref_lower in verse_refs:
                                fused_scores[vidx] += 0.05
                                boosted += 1
                        if boosted:
                            logger.info(f"Section chunk boost: +0.05 to {boosted} verse(s) from {top_section.get('reference', '?')}")
            except Exception as e:
                logger.debug(f"Section chunk search failed (non-fatal): {e}")

        # Scripture pre-filtering: mask non-matching before candidate selection
        scripture_prefiltered = False
        if scripture_filter is not None and len(scripture_filter) > 0 and hasattr(self, '_scripture_indices'):
            allowed_mask = np.zeros(len(self.verses), dtype=bool)
            for s in scripture_filter:
                indices = self._scripture_indices.get(s)
                if indices is not None:
                    allowed_mask[indices] = True
            if allowed_mask.any():
                fused_scores = np.where(allowed_mask, fused_scores, -np.inf)
                scripture_prefiltered = True
                logger.info(f"Scripture pre-filter: {int(allowed_mask.sum())} candidate docs from {scripture_filter}")

        # 4. Adaptive candidate pool based on query type and intent
        candidates_k = self._get_candidate_pool_size(original_query, intent)
        k_search = min(candidates_k, fused_scores.shape[0])

        if k_search <= 0:
            return []

        candidate_indices = np.argpartition(-fused_scores, k_search - 1)[:k_search]

        # Curated concept slot reservation: ensure top curated docs reach the reranker
        # even if the 96k-verse corpus dominates initial candidate selection
        # Dynamic threshold: max(CURATED_FLOOR, top_fused * CURATED_RATIO)
        if hasattr(self, '_curated_indices') and len(self._curated_indices) > 0:
            curated_scores = fused_scores[self._curated_indices]
            _top_fused = float(np.max(fused_scores)) if fused_scores.size > 0 else 0.5
            _dynamic_curated_threshold = max(settings.CURATED_FLOOR, _top_fused * settings.CURATED_RATIO)
            curated_top_k = min(settings.CURATED_SLOT_LIMIT, len(self._curated_indices))
            curated_top_local = np.argsort(-curated_scores)[:curated_top_k]
            candidate_set = set(candidate_indices.tolist())
            injected = 0
            for local_idx in curated_top_local:
                global_idx = self._curated_indices[local_idx]
                if curated_scores[local_idx] > _dynamic_curated_threshold and global_idx not in candidate_set:
                    candidate_set.add(int(global_idx))
                    injected += 1
            if injected > 0:
                candidate_indices = np.array(list(candidate_set), dtype=np.int64)
                logger.info(f"Curated slot reservation: injected {injected} concept doc(s) (threshold={_dynamic_curated_threshold:.3f})")

        sorted_indices = candidate_indices[np.argsort(-fused_scores[candidate_indices])]

        results: List[Dict] = []
        for idx in sorted_indices:
            score = float(fused_scores[int(idx)])
            if not np.isfinite(score):
                continue  # Skip masked-out documents from pre-filtering
            verse = self.verses[int(idx)]

            # Fallback scripture filter (only when pre-filtering wasn't applied)
            if scripture_filter and not scripture_prefiltered:
                verse_scripture = verse.get("scripture")
                if isinstance(scripture_filter, list):
                    if verse_scripture not in scripture_filter:
                        continue
                elif verse_scripture != scripture_filter:
                    continue

            results.append({**verse, "score": score, "_global_idx": int(idx)})
            if len(results) >= candidates_k:
                break

        # 5. Neural Re-ranking (Cross-Encoder) — skip when top result is already decisive
        # Adaptive skip: modes that don't cite specific verses skip reranking
        _skip_modes = frozenset({"presence_first", "closure"})
        _effective_skip = skip_rerank or (response_mode in _skip_modes)
        logger.info(f"PERF_RAG candidates={len(results)} before reranking (skip={_effective_skip}, mode={response_mode})")
        _t_rerank = time.perf_counter()
        if _effective_skip:
            # Caller will rerank merged results later (rerank-once pattern)
            for r in results:
                r["rerank_score"] = r.get("score", 0)
                r["final_score"] = r.get("score", 0)
        elif results and len(results) >= 2:
            top_score = results[0].get("score", 0)
            second_score = results[1].get("score", 0)
            gap = top_score - second_score
            if top_score >= settings.SKIP_RERANK_THRESHOLD and gap >= settings.SKIP_RERANK_GAP:
                logger.info(f"Skipping reranker: top={top_score:.3f} gap={gap:.3f} (threshold={settings.SKIP_RERANK_THRESHOLD})")
                for r in results:
                    r["rerank_score"] = r.get("score", 0)
                    r["final_score"] = r.get("score", 0)
            else:
                results = await self._rerank_results(query, results, intent=intent, life_domain=life_domain)
        elif results:
            results = await self._rerank_results(query, results, intent=intent, life_domain=life_domain)
        _timings['rerank_ms'] = round((time.perf_counter() - _t_rerank) * 1000)

        # 5.5. Parent-child verse expansion (after reranking, before min_score gate)
        if settings.PARENT_CHILD_ENABLED and hasattr(self, '_neighbor_index'):
            results = self._expand_with_neighbors(
                results, expand_top_n=settings.EXPAND_TOP_N, window=settings.VERSE_CONTEXT_WINDOW,
            )

        # 6. Soft min_score floor — only drop obvious garbage; let ContextValidator do real filtering
        if min_score > 0:
            soft_floor = min_score * settings.SOFT_FLOOR_RATIO
            before = len(results)
            results = [r for r in results if r.get("is_context_verse") or get_doc_score(r) >= soft_floor]
            dropped = before - len(results)
            if dropped:
                logger.info(f"Soft min_score floor: dropped {dropped} doc(s) below {soft_floor:.3f}")

        # 7. Doc-type exclusion filter
        if doc_type_filter:
            excluded_types = {t.lower() for t in doc_type_filter}
            before = len(results)
            results = [r for r in results if (r.get("type") or "scripture").lower() not in excluded_types]
            dropped = before - len(results)
            if dropped:
                logger.info(f"doc_type_filter: dropped {dropped} doc(s) of types={doc_type_filter}")

        # Limit to final top_k
        final_results = results[:top_k]
        
        _timings['total_ms'] = round((time.perf_counter() - _t_start) * 1000)
        _perf_msg = (
            f"RAG_LATENCY total={_timings['total_ms']}ms "
            f"preprocess={_timings.get('preprocess_ms', 0)}ms "
            f"search={_timings.get('hybrid_search_ms', 0)}ms "
            f"rerank={_timings.get('rerank_ms', 0)}ms "
            f"| docs={len(final_results)} queries={len(expanded_queries)} hyde={len(hyde_embeddings)} query='{query[:40]}' intent={intent}"
        )
        logger.info(_perf_msg)
        print(_perf_msg, flush=True)  # Ensure visibility in Cloud Run logs

        # Exclude previously-suggested verses (session diversity)
        if exclude_references and final_results:
            _exclude_set = {r.lower() for r in exclude_references}
            final_results = [
                doc for doc in final_results
                if (doc.get("reference") or "").lower() not in _exclude_set
            ]

        # MMR diversity — re-order for relevance + diversity balance
        if final_results and len(final_results) > 1:
            final_results = self._mmr_diversify(
                final_results, lambda_param=0.7, top_k=top_k
            )

        # Cache search results for repeat queries
        await cache.set("rag_search", final_results, ttl=settings.RAG_SEARCH_CACHE_TTL, **_cache_params)

        # Fire-and-forget query logging for analytics
        if settings.QUERY_LOG_ENABLED:
            try:
                from services.query_logger import get_query_logger
                _top_score = self._get_best_score(final_results[0]) if final_results else 0.0
                asyncio.create_task(get_query_logger().log(
                    query=original_query,
                    intent=str(intent) if intent else None,
                    life_domain=life_domain,
                    num_results=len(final_results),
                    top_score=_top_score,
                    latency_ms=_timings.get('total_ms', 0),
                    expanded_queries=expanded_queries[1:] if len(expanded_queries) > 1 else None,
                ))
            except Exception:
                pass  # Query logging failure must never break search

        return final_results

    # ------------------------------------------------------------------
    # High‑level text QA (used by /api/text/query)
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        language: str = "en",
        include_citations: bool = True,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        RAG‑augmented QA for standalone text queries with caching.
        """
        cache = get_cache_service()
        
        # 1. Try to get from cache
        # Note: we use a limited history slice for cache key stability
        history_key = str(conversation_history[-2:]) if conversation_history else ""
        cache_params = {
            "query": query,
            "language": language,
            "citations": include_citations,
            "history": history_key
        }
        
        cached_res = await cache.get("rag_query", **cache_params)
        if cached_res:
            return cached_res

        # Retrieve context first
        docs = await self.search(
            query=query, 
            scripture_filter=None, 
            language=language, 
            top_k=settings.RETRIEVAL_TOP_K
        )

        # If LLM is available, let it synthesize an answer
        if self._llm.available:
            answer = await self._llm.generate_response(
                query=query,
                context_docs=docs,
                conversation_history=conversation_history or [],
            )
        else:
            # Very simple fallback that just echoes top verse text
            if docs:
                top = docs[0]
                answer = top.get("text") or top.get("meaning") or "I found a relevant verse for you."
            else:
                answer = "I couldn't find a specific verse, but I'm here to listen to what you're going through."

        citations: List[Dict] = []
        if include_citations:
            for doc in docs[:settings.RERANK_TOP_K]:
                citations.append(
                    {
                        "reference": doc.get("reference", ""),
                        "scripture": doc.get("scripture", ""),
                        "text": (doc.get("text") or "")[:200],
                        "score": doc.get("score", 0.0),
                    }
                )

        confidence = 1.0 if docs and docs[0].get('score', 0) > 0.4 else (0.5 if docs else 0.0)

        result = {
            "answer": answer,
            "citations": citations,
            "language": language,
            "confidence": confidence,
        }

        # 3. Store in cache (expire after 1 hour by default)
        await cache.set("rag_query", result, ttl=3600, **cache_params)

        return result

    async def query_stream(
        self,
        query: str,
        language: str = "en",
        include_citations: bool = True,
        conversation_history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        True streaming version of standalone text query.
        """
        # 1. Retrieval
        docs = await self.search(
            query=query, 
            scripture_filter=None, 
            language=language, 
            top_k=settings.RETRIEVAL_TOP_K
        )

        # 2. Build metadata chunk
        citations: List[Dict] = []
        if include_citations:
            for doc in docs[:settings.RERANK_TOP_K]:
                citations.append(
                    {
                        "reference": doc.get("reference", ""),
                        "scripture": doc.get("scripture", ""),
                        "text": (doc.get("text") or "")[:200],
                        "score": doc.get("score", 0.0),
                    }
                )

        confidence = 1.0 if docs and docs[0].get('score', 0) > 0.4 else (0.5 if docs else 0.0)

        # Yield metadata first
        yield {
            "type": "meta",
            "citations": citations,
            "confidence": confidence,
        }

        # 3. Stream Synthesis
        if self._llm.available:
            # Use generate_response_stream for token-by-token delivery
            async for token in self._llm.generate_response_stream(
                query=query,
                context_docs=docs,
                conversation_history=conversation_history or [],
            ):
                yield {
                    "type": "answer",
                    "text": token,
                }
        else:
            # Simple fallback
            if docs:
                top = docs[0]
                answer = top.get("text") or top.get("meaning") or "I found a relevant verse for you."
            else:
                answer = "I'm here to listen to what you're going through."

            yield {
                "type": "answer",
                "text": answer,
            }


# ---------------------------------------------------------------------------
# Singleton factory — consistent with all other service singletons in this
# codebase. Returns the module-level RAGPipeline instance when available,
# or None if the pipeline has not been initialised yet.
# ---------------------------------------------------------------------------

_rag_pipeline_instance: Optional["RAGPipeline"] = None


def get_rag_pipeline() -> Optional["RAGPipeline"]:
    """Return the module-level RAGPipeline singleton.

    Returns None when the pipeline has not been initialised (e.g. in unit
    tests that never call ``RAGPipeline()``). Callers should guard:

        pipeline = get_rag_pipeline()
        if pipeline is None or not pipeline.available:
            return []
    """
    return _rag_pipeline_instance
