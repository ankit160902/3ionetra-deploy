import asyncio
import logging
import json
from typing import Dict, Any, Optional
from config import settings
from models.session import IntentType
from models.llm_schemas import IntentAnalysis, extract_json
from services.cache_service import _LRUCache

logger = logging.getLogger(__name__)

class IntentAgent:
    """
    LLM-based Intent Agent for deep understanding of user messages.
    Categorizes intent, emotion, life domains, and extracts key entities.
    """

    INTENT_PROMPT = """
    You are an expert intent classifier for 3ioNetra, a spiritual companion bot.
    Analyze the user message below and return a JSON object.

    USER MESSAGE: "{message}"
    CONVERSATION CONTEXT: {context}

    Return ONLY a JSON object with these fields:

    1. "intent": Choose ONE from:
       [GREETING, SEEKING_GUIDANCE, EXPRESSING_EMOTION, ASKING_INFO, ASKING_PANCHANG, PRODUCT_SEARCH, CLOSURE, OTHER]
       - ASKING_INFO: factual "what is", "how does", "tell me about" questions
       - SEEKING_GUIDANCE: planning/advice requests like "give me a routine", "plan me a puja", "how should I"
       - PRODUCT_SEARCH: ONLY when user explicitly asks to buy/find/order something

    2. "emotion": The dominant emotion. Detect IMPLICIT emotions — look beyond surface words:
       "zero friends", "eat lunch alone", "nobody would notice" → loneliness
       "cant walk properly", "stuck at home" → grief, frustration
       "dont want to live", "nothing matters" → despair
       "cant stop drinking", "always relapse" → shame
       NEVER return "neutral" when the user describes a painful situation. Choose from: grief, anxiety, loneliness, frustration, anger, despair, hopelessness, shame, confusion, joy, gratitude, hope, curiosity, neutral.

    3. "life_domain": Describe the primary area of concern in 2-5 words. Be specific to
       the user's actual situation — do NOT force-fit into a generic category.
       Examples: "career growth frustration", "marriage conflict", "board exam preparation",
       "health anxiety chest pain", "Kedarnath pilgrimage planning", "financial debt stress",
       "daily spiritual practice", "grief after parent loss", "anger management",
       "travel and temple visit planning", "parenting a teenager", "addiction recovery"

    4. "entities": Dict of extracted nouns e.g. {{"deity": "Shiva", "ritual": "Abhishekam", "item": "mala"}}

    5. "urgency": One of [low, normal, high, crisis]
       - crisis: user expresses suicidal ideation, self-harm intent, or desire to end their life
         (even with typos/misspellings like "i dnt wnt 2 liv", "suicde", "kil myself")
       - high: deep existential despair WITHOUT explicit self-harm intent — "giving up on
         everything", "nothing makes sense", "what's the point of going on", "can't take it
         anymore". These need extra care but NOT helpline numbers.
       - normal: emotional but not in danger — stress, frustration, sadness, confusion
       - low: factual/informational queries, casual conversation

    6. "summary": One-sentence summary of the user's core need.

    7. "needs_direct_answer": boolean
       - True ONLY when the user asks a specific HOW-TO, factual WHAT-IS, or planning/procedure
         question with NO emotional weight.
         Examples: "What is karma yoga?", "How to perform Satyanarayan puja?",
         "Which mantra for Ganesh?", "Steps to do pranayama"
       - False when the user is venting, sharing feelings, expressing pain, or describing a
         difficult situation, even if the message contains question words.
         Examples: "I feel lost about my career", "What should I do, I'm so confused",
         "Why does this hurt so much?"
       - HARD RULE: If the message contains explicit emotion words (sad, angry, hurt, lost,
         scared, anxious, hopeless, lonely, depressed, worthless, broken, overwhelmed) OR
         the emotion you classified above is one of grief/anxiety/anger/fear/despair/loneliness/
         hopelessness/shame, return False.

    8. "product_signal": A structured object describing the user's product interest.
       This replaces the old boolean recommend_products. DEFAULT intent is "none".
       {{
         "intent": one of ["none", "explicit_search", "contextual_need", "casual_mention", "negative"],
         "confidence": float 0.0-1.0,
         "type_filter": one of ["any", "physical_only", "consultation_only", "service_only"],
         "search_keywords": list of 3-4 specific product search terms (empty if intent is "none"),
         "max_results": integer 0-5 (0 = don't recommend any products),
         "sensitivity_note": brief note if emotional sensitivity matters (empty string otherwise)
       }}

       ## Product Awareness

       IMPORTANT: When you set product_signal.intent to anything other than "none",
       the system will show product cards to the user in the chat interface. This is
       the ONLY way products appear — the text response itself must NEVER mention
       product names, prices, or shopping URLs. Your job is to decide WHETHER to show
       product cards by setting the right intent and search_keywords.

       You are a friend who happens to know a shop (my3ionetra.com). You never bring up
       products unless the moment genuinely calls for it. Here is how you think about it:

       - If someone is hurting, you LISTEN. You do not hand them a catalog.
       - If someone is exploring spirituality casually, you talk. You do not sell.
       - If someone has been discussing a practice across several turns and naturally
         reaches the point where they would need something (e.g. "I want to start doing
         japa daily"), THAT is when you might set product_signal to show them a mala.
       - If someone directly asks "do you have products?", "where can I buy X?",
         "can you recommend items for Y?", or "I want to buy Z", you MUST set
         product_signal.intent to "explicit_search" with relevant search_keywords.
         This is how the user gets product cards. Do NOT just describe where to shop
         in your text response — set the signal so the system can show actual products.
       - If someone has told you before that they do not want product suggestions,
         you respect that permanently.

       The bar is: would a thoughtful human friend mention a product here, or would
       it feel forced? When in doubt, set intent to "none".

       You will receive the user's product interaction history (how many times they have
       been shown products, whether they have rejected them, their preference). Use this
       to calibrate. A user who has been shown products 5 times this month needs a higher
       bar than a new user.

       product_signal output rules:
       - "explicit_search": user asks to buy, find, or get product recommendations. ALWAYS use this when user says "buy", "purchase", "where can I get", "recommend items/products", "I need [item] for [practice]", "suggest products", "product batao", "kuch chahiye [practice] ke liye", "khareedna hai", "order karna hai". confidence 0.8-1.0, max_results 3-5. search_keywords MUST contain specific product terms (e.g. ["rudraksha mala", "mala"] or ["puja thali", "diya", "incense"]).
       - "contextual_need": the conversation has naturally reached a point where a product genuinely helps and the user has expressed readiness (e.g. "I want to start doing japa daily" after several turns of discussion, or "kuch suggest karo jo help kare", "kuch batao jo kaam aaye", "kuch chahiye stress ke liye"). confidence 0.5-0.7, max_results 1-2.
       - "casual_mention": products tangentially relevant but not the user's ask. confidence 0.2-0.4, max_results 1.
       - "negative": user rejects or criticizes products. confidence 0.8-1.0, max_results 0.
       - "none": DEFAULT. No product interest detected. This includes emotional sharing, advice-seeking, scripture questions, greetings, closures, and any turn where recommending would feel forced. confidence 0.0, max_results 0.

       TURN-AWARENESS: Check the context for turn count. When a user has been
       sharing for 3+ turns and explicitly asks for suggestions ("suggest
       something", "is there anything", "what can help", "kuch suggest karo"),
       consider "contextual_need" with max_results=1-2 even if the context is
       emotional. By turn 3+, the user has been heard and is now seeking
       actionable help. This does NOT apply to pure venting without an
       explicit ask for suggestions.

       TYPE FILTER RULES:
       - "physical_only": User asks for objects/items (murti, mala, diya, incense)
       - "consultation_only": User asks for expert/astrologer/consultation
       - "service_only": User asks for puja booking, temple seva, workshop
       - "any": Not specified or mixed context

       WHEN IN DOUBT: intent="none", max_results=0. It is ALWAYS better to NOT
       show products than to show irrelevant ones. Products should feel like a
       rare, relevant surprise — not a default on every response.

    11. "query_variants": List of EXACTLY 2 alternative search phrasings for the user's spiritual question.
        - You MUST generate 2 variants for ALL intents EXCEPT GREETING, CLOSURE, PRODUCT_SEARCH, and trivial/single-word messages.
        - For EXPRESSING_EMOTION: include comfort/healing oriented terms (e.g. "finding peace in grief", "overcoming anxiety through dharma").
        - For SEEKING_GUIDANCE / ASKING_INFO: include specific Sanskrit/English teaching terms.
        - Each variant should capture a different semantic angle of the user's concern.
        - Return empty list [] ONLY for GREETING, CLOSURE, PRODUCT_SEARCH, or trivial messages.

    12. "expected_length": How long the response should be. Choose ONE:
        - "brief": greetings, yes/no answers, acknowledgments, closures, "ok", "thanks"
        - "moderate": emotional sharing, simple questions, most conversation turns
        - "detailed": guidance requests, how-to questions, explanations, deeper topics
        - "full_text": user explicitly asks for complete text (full chalisa, full prayer, complete stotra, all steps of a ritual)

    13. "is_off_topic": boolean
        - True ONLY when the query is clearly outside spiritual/life-guidance scope.
          Examples: writing code, debugging software, sports scores, weather forecasts,
          stock prices, cooking recipes, movie reviews, generic web searches, news headlines.
        - False for anything related to life, emotions, relationships, family, career stress,
          health, dharma, spirituality, rituals, philosophy, personal growth, or anything
          a spiritual companion might reasonably engage with.
        - When in doubt, return False — let the conversation flow naturally.

    14. "response_mode": How the response should be shaped. Choose ONE:
        - "practical_first": User brought a solvable real-world problem (study
          strategy, job search, budgeting, health logistics, relationship
          communication, time management, exam prep, interview prep, financial
          triage). Answer is mostly PRACTICAL — direct, actionable, friend-voice.
          Minimal or no explicit spirituality. Spiritual content appears ONLY as
          a brief optional tip that genuinely helps the immediate problem (e.g.
          "30 seconds of silence before the exam"), framed as a practical hack.
        - "presence_first": User is in raw emotional pain (grief, despair,
          loneliness, shame, overwhelm, hopelessness) and is VENTING or SHARING,
          NOT asking for advice. Answer is acknowledgment + holding space, no
          scripture, no practical pivot, no mantras. Short (3-5 sentences).
        - "teaching": User is explicitly asking about dharma, scripture,
          philosophy, mantras, rituals, festivals, deities, or panchang. This
          is full spiritual mode — rich answer, scripture allowed, verses and
          mantras allowed, domain compass applies.
        - "exploratory": User is vague, searching, doesn't know what they need
          ("I feel lost", "I don't know what to do"). Answer is warmth + ONE
          clarifying question. No solutions, no scripture yet.
        - "closure": User is signaling the END of this exchange — gratitude,
          acceptance, "I feel better now", "thanks for listening", "ok I'll
          try that", "thank you, I needed to hear that", or a short positive
          acknowledgement of advice given. The signal is acceptance, not more
          sharing. Classify as closure when the CURRENT TURN is a wind-down,
          REGARDLESS of how emotional the prior turns were. This rule dominates
          conversation-history bias — if the prior 3 turns were about grief
          and this turn is "thank you, that helped", the mode is closure, not
          presence_first. The response should be 1-2 sentences of warmth,
          not an invitation to continue.

        TIE-BREAKER: If a query has both dimensions ("I'm starting a new job
        on Monday, should I do a puja?"), pick the DOMINANT mode — usually the
        ASK (puja -> teaching) over the context (job -> practical_first).
        Another example: "I'm anxious about my exam tomorrow" has emotion but
        the dominant need is practical exam strategy -> practical_first.

        WHEN IN DOUBT: default to "exploratory" — it's the safest fallback
        because it invites more context rather than committing to a direction.

    Respond ONLY with the valid JSON object.
    """

    def __init__(self):
        from llm.service import get_llm_service
        self.llm = get_llm_service()
        self.available = self.llm.available
        self._cache = _LRUCache(max_size=500)  # Expanded for better hit rate

    # ── Fast-path constants ──
    _GREETING_SET = frozenset({
        "hi", "hey", "hello", "namaste", "pranam", "hii", "hiii",
        "hi mitra", "hey mitra", "hello mitra", "namaste mitra", "pranam mitra",
        "namaskar", "namaskar mitra", "jai shree ram", "har har mahadev",
        "jai shri ram", "ram ram", "radhe radhe", "hare krishna",
        "good morning", "good evening", "good afternoon",
        "good morning mitra", "good evening mitra",
        "kaise ho", "kaise ho mitra", "kya haal hai",
    })
    _CLOSURE_SET = frozenset({
        "thank you", "thanks", "bye", "goodbye", "ok bye", "dhanyavaad", "shukriya",
        "alvida", "ok thanks", "thanks bye", "good bye", "ok thank you",
        "thank you mitra", "thanks mitra", "bye mitra", "goodbye mitra",
        "bye bye", "accha bye", "theek hai bye", "chalo bye",
        "thank you so much", "thanks a lot", "bahut dhanyavaad",
    })
    _PANCHANG_KEYWORDS = ("panchang", "tithi", "nakshatra", "muhurat")
    _INFO_PREFIXES = ("what is ", "what are ", "tell me about ", "who is ", "explain ")
    _EMOTION_MAP = {
        "sad": "grief", "unhappy": "grief", "crying": "grief",
        "anxious": "anxiety", "worried": "anxiety", "nervous": "anxiety",
        "stressed": "frustration", "frustrated": "frustration", "overwhelmed": "frustration",
        "lonely": "loneliness", "alone": "loneliness", "isolated": "loneliness",
        "angry": "anger", "furious": "anger", "irritated": "anger",
        "lost": "confusion", "confused": "confusion",
        "depressed": "hopelessness", "hopeless": "hopelessness",
        "scared": "anxiety", "afraid": "anxiety", "fearful": "anxiety",
    }
    _EMOTION_PATTERNS = (
        "i feel ", "i'm feeling ", "im feeling ", "i am feeling ",
        "feeling ", "i'm so ", "im so ", "i am so ",
        "i feel so ", "i'm ", "im ", "i am ",
    )
    # Words that indicate emotional weight — used to suppress fast-path INFO classification
    # and to gate the fallback's needs_direct_answer heuristic. Kept as a frozenset for O(1)
    # lookup; updated alongside _EMOTION_MAP whenever new emotion vocabulary is added.
    _EMOTIONAL_INDICATORS = frozenset({
        "feel", "feeling", "felt", "hurt", "hurts", "scared", "afraid", "anxious",
        "sad", "angry", "lost", "hopeless", "lonely", "depressed", "overwhelmed",
        "worthless", "broken", "crying", "miserable", "exhausted", "drained",
        "empty", "stuck", "trapped", "ashamed", "guilty", "grieving",
    })
    # Off-topic phrases — narrow keyword list to avoid false positives. The LLM-level
    # is_off_topic field provides the broader semantic check; this fast-path catches
    # the most obvious non-spiritual queries without an LLM round-trip.
    _OFF_TOPIC_PHRASES = frozenset({
        "python code", "javascript", "java code", "write code for", "debug my",
        "write a function", "regex for", "html tag", "css for",
        "stock price", "share price", "weather forecast", "temperature today",
        "movie review", "football match", "cricket score", "ipl match",
        "recipe for", "how to cook", "calorie count",
    })

    def _has_emotional_weight(self, msg_lower: str) -> bool:
        """Return True if the message contains explicit emotion vocabulary.

        Used by the fast-path to suppress INFO classification when the user
        is venting, and by the fallback to gate needs_direct_answer.
        """
        tokens = set(msg_lower.split())
        return bool(tokens & self._EMOTIONAL_INDICATORS)

    def _fast_path(self, msg_lower: str) -> Optional[Dict[str, Any]]:
        """Return a fast-path result dict if the message matches a known pattern, else None."""
        _default_product_signal = {
            "intent": "none", "confidence": 0.0, "type_filter": "any",
            "search_keywords": [], "max_results": 0, "sensitivity_note": "",
        }
        _base = {
            "entities": {}, "urgency": "low", "summary": "",
            "needs_direct_answer": False,
            "product_signal": _default_product_signal,
            # Legacy compat (all default to no-products)
            "recommend_products": False,
            "product_search_keywords": [], "product_rejection": False,
            "query_variants": [], "expected_length": "moderate",
            "is_off_topic": False,
            # Response mode — default is exploratory so an unmatched fast-path
            # short-circuit leaves the LLM in the safest "ask one question" mode
            "response_mode": "exploratory",
        }

        # (a) GREETING — exact match or short greeting pattern
        if msg_lower in self._GREETING_SET:
            return {**_base, "intent": IntentType.GREETING, "emotion": "neutral",
                    "life_domain": "unknown", "expected_length": "brief",
                    "response_mode": "exploratory"}
        first_word = msg_lower.split()[0] if msg_lower else ""
        if first_word in {"hi", "hey", "hello", "namaste", "namaskar", "pranam", "hii", "hiii"} and len(msg_lower.split()) <= 4:
            return {**_base, "intent": IntentType.GREETING, "emotion": "neutral",
                    "life_domain": "unknown", "expected_length": "brief",
                    "response_mode": "exploratory"}

        # (b) CLOSURE — route to the dedicated closure mode so the response is
        # a 1-2 sentence warm wind-down instead of an invitation to keep talking.
        if msg_lower in self._CLOSURE_SET:
            return {**_base, "intent": IntentType.CLOSURE, "emotion": "gratitude",
                    "life_domain": "unknown", "expected_length": "brief",
                    "response_mode": "closure"}

        # (c) OFF_TOPIC — short-circuit before normal classification.
        # We skip this check if the message also has emotional weight (e.g.
        # "I'm so stressed I can't even cook a recipe"), letting the LLM
        # handle the disambiguation.
        if not self._has_emotional_weight(msg_lower):
            for phrase in self._OFF_TOPIC_PHRASES:
                if phrase in msg_lower:
                    return {**_base, "intent": IntentType.OTHER, "emotion": "neutral",
                            "life_domain": "unknown", "is_off_topic": True,
                            "expected_length": "brief",
                            "response_mode": "exploratory"}

        # (d) ASKING_PANCHANG — substring check
        if any(kw in msg_lower for kw in self._PANCHANG_KEYWORDS):
            return {**_base, "intent": IntentType.ASKING_PANCHANG, "emotion": "neutral",
                    "life_domain": "spiritual", "needs_direct_answer": True,
                    "expected_length": "detailed", "response_mode": "teaching"}

        # (e) EXPRESSING_EMOTION — short messages with emotion keywords
        words = msg_lower.split()
        if len(words) <= 8:
            for pattern in self._EMOTION_PATTERNS:
                if msg_lower.startswith(pattern):
                    remainder = msg_lower[len(pattern):].strip().rstrip(".")
                    for keyword, emotion in self._EMOTION_MAP.items():
                        if keyword in remainder:
                            return {**_base, "intent": IntentType.EXPRESSING_EMOTION,
                                    "emotion": emotion, "life_domain": "unknown",
                                    "summary": msg_lower, "expected_length": "moderate",
                                    "response_mode": "presence_first"}
                    break

        # (f) ASKING_INFO — starts with question prefix.
        # Suppress fast-path classification when the message has emotional weight,
        # so the LLM can decide whether the user wants a factual answer or empathy.
        for prefix in self._INFO_PREFIXES:
            if msg_lower.startswith(prefix):
                if self._has_emotional_weight(msg_lower):
                    break  # fall through to LLM analysis
                return {**_base, "intent": IntentType.ASKING_INFO, "emotion": "curiosity",
                        "life_domain": "unknown", "needs_direct_answer": True,
                        "summary": msg_lower, "expected_length": "detailed",
                        "response_mode": "teaching"}

        return None

    async def analyze_intent(self, message: str, context_summary: str = "") -> Dict[str, Any]:
        """Analyze intent using LLM (fast model for low latency)."""
        msg_lower = message.strip().lower()

        # Fast-path: skip LLM for obvious intents (saves 800ms-2s)
        fast = self._fast_path(msg_lower)
        if fast is not None:
            logger.info(f"Fast-path intent: {fast['intent']} for '{message[:30]}'")
            return fast

        # LRU cache: return cached result for identical messages (saves 800ms-2s on repeats)
        cached = self._cache.get(msg_lower)
        if cached is not None:
            logger.info(f"Intent cache HIT for '{message[:30]}'")
            return cached

        if not self.available:
            return self._fallback_analysis(message)

        prompt = self.INTENT_PROMPT.format(message=message, context=context_summary)

        try:
            # Use fast model (Haiku) for low-latency JSON classification
            from llm.bedrock import bedrock_generate

            def _sync_call():
                return bedrock_generate(
                    prompt,
                    model=settings.GEMINI_FAST_MODEL,
                    temperature=settings.INTENT_TEMPERATURE,
                    max_tokens=1024,
                    json_mode=True,
                )

            # Run in thread pool — let Bedrock complete
            response_text = await asyncio.to_thread(_sync_call)

            raw_text = response_text.text.strip()
            parsed = extract_json(raw_text)
            if parsed is None:
                logger.warning(f"IntentAgent: JSON extraction failed for '{message[:30]}...'")
                return self._fallback_analysis(message)

            # Validate with Pydantic model (coerces enums, applies defaults).
            # Wrap in its own try/except because Pydantic validation can fail
            # on unexpected JSON shapes (e.g. Gemini returns product_signal as
            # a nested object but omits legacy flat fields). On failure, merge
            # parsed JSON with defaults manually.
            try:
                analysis = IntentAnalysis(**parsed)
                data = analysis.to_dict()
            except Exception as pydantic_err:
                logger.warning(f"IntentAgent Pydantic validation failed: {pydantic_err}. Using raw parsed with defaults.")
                # Manually build a valid dict from parsed JSON + defaults
                _valid_modes = {"practical_first", "presence_first", "teaching", "exploratory", "closure"}
                _raw_mode = str(parsed.get("response_mode", "exploratory")).strip().lower()
                data = {
                    "intent": parsed.get("intent", "OTHER"),
                    "emotion": parsed.get("emotion", "neutral"),
                    "life_domain": parsed.get("life_domain", "unknown"),
                    "entities": parsed.get("entities", {}),
                    "urgency": parsed.get("urgency", "normal"),
                    "summary": parsed.get("summary", ""),
                    "needs_direct_answer": parsed.get("needs_direct_answer", False),
                    "product_signal": parsed.get("product_signal", {
                        "intent": "none", "confidence": 0.0, "type_filter": "any",
                        "search_keywords": [], "max_results": 0, "sensitivity_note": "",
                    }),
                    "recommend_products": parsed.get("recommend_products", False),
                    "product_search_keywords": parsed.get("product_search_keywords", []),
                    "product_rejection": parsed.get("product_rejection", False),
                    "query_variants": parsed.get("query_variants", []),
                    "expected_length": parsed.get("expected_length", "moderate"),
                    "is_off_topic": parsed.get("is_off_topic", False),
                    "response_mode": _raw_mode if _raw_mode in _valid_modes else "exploratory",
                }
            data["intent"] = IntentType(data.get("intent", "OTHER"))
            logger.info(
                f"LLM Intent Analysis for '{message[:30]}...': {data.get('intent')} "
                f"| mode: {data.get('response_mode', 'n/a')} "
                f"| signal: {data.get('product_signal', {}).get('intent', 'n/a')}"
            )
            self._cache.set(msg_lower, data, 1800)  # 30-min TTL
            return data

        except Exception as e:
            logger.error(f"Error in LLM intent analysis: {e}")
            return self._fallback_analysis(message)

    def _fallback_analysis(self, message: str) -> Dict[str, Any]:
        """Keyword-based fallback if LLM is unavailable"""
        message_lower = message.lower()

        intent = IntentType.OTHER
        if any(w in message_lower for w in ["hi", "hello", "hey", "namaste"]):
            intent = IntentType.GREETING
        elif "?" in message or any(w in message_lower for w in ["how", "what", "guide", "help"]):
            intent = IntentType.SEEKING_GUIDANCE
        elif any(w in message_lower for w in ["panchang", "tithi", "nakshatra"]):
            intent = IntentType.ASKING_PANCHANG
        elif any(w in message_lower for w in ["bye", "thanks", "thank you", "ok", "no", "nothing"]):
            intent = IntentType.CLOSURE

        # Suppress needs_direct_answer when the message has emotional weight, even if
        # it contains a question word. Without this, "What should I do? I'm so lost"
        # would short-circuit straight to guidance instead of being heard first.
        has_emotion = self._has_emotional_weight(message_lower)
        needs_direct = (not has_emotion) and (
            "?" in message or any(w in message_lower for w in ["how", "what", "where", "why"])
        )

        # Product signal — minimal fallback (LLM handles the real logic)
        _buy_words = {
            "buy", "purchase", "order", "shop", "shopping", "price", "cost",
            "khareed", "khareedna", "kharidna", "mangwa", "mangwao",
            "product", "products", "chahiye",
        }
        _reject_phrases = {"stop suggesting products", "stop recommending", "no more products",
                          "don't show products", "dont show products"}
        msg_lower = message.lower()

        if any(p in msg_lower for p in _reject_phrases):
            product_signal = {
                "intent": "negative", "confidence": 1.0, "type_filter": "any",
                "search_keywords": [], "max_results": 0, "sensitivity_note": "",
            }
        elif any(w in msg_lower.split() for w in _buy_words):
            product_signal = {
                "intent": "explicit_search", "confidence": 0.6, "type_filter": "any",
                "search_keywords": [w for w in msg_lower.split() if len(w) > 3 and w not in _buy_words],
                "max_results": 3, "sensitivity_note": "",
            }
        else:
            product_signal = {
                "intent": "none", "confidence": 0.0, "type_filter": "any",
                "search_keywords": [], "max_results": 0, "sensitivity_note": "",
            }

        # Derive response_mode from intent + emotional weight. Used when the
        # LLM classifier is unavailable — keeps mode-aware downstream code
        # functioning with sensible defaults.
        mode_map = {
            IntentType.GREETING: "exploratory",
            IntentType.CLOSURE: "closure",
            IntentType.ASKING_INFO: "teaching",
            IntentType.ASKING_PANCHANG: "teaching",
            IntentType.PRODUCT_SEARCH: "teaching",
            IntentType.SEEKING_GUIDANCE: "practical_first",
            IntentType.EXPRESSING_EMOTION: "presence_first",
            IntentType.OTHER: "exploratory",
        }
        response_mode = mode_map.get(intent, "exploratory")
        # If the message has emotional weight but intent was misread as
        # SEEKING_GUIDANCE (question word + feelings), lean presence_first.
        if has_emotion and intent == IntentType.SEEKING_GUIDANCE:
            response_mode = "presence_first"

        return {
            "intent": intent,
            "emotion": "neutral",
            "life_domain": "unknown",
            "entities": {},
            "urgency": "normal",
            "summary": message[:100],
            "needs_direct_answer": needs_direct,
            "product_signal": product_signal,
            # Legacy compat (derived from product_signal)
            "recommend_products": product_signal["intent"] not in ("none", "negative"),
            "product_search_keywords": product_signal["search_keywords"],
            "product_rejection": product_signal["intent"] == "negative",
            "query_variants": [],
            "is_off_topic": False,
            "response_mode": response_mode,
        }

_intent_agent = None

def get_intent_agent() -> IntentAgent:
    global _intent_agent
    if _intent_agent is None:
        _intent_agent = IntentAgent()
    return _intent_agent
