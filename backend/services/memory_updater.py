"""Keyword-based signal detection and memory updates.

Extracted from CompanionEngine._update_memory() — detects emotions,
life domains, and special intents from user message text using regex
keyword matching with weighted scoring and tie-breakers.

This is a pure function of (memory, session, text) — no service dependencies.
"""
import re
from typing import List

from models.memory_context import ConversationMemory
from models.session import SessionState, SignalType


def _has_word(keywords: list, text: str) -> bool:
    """Check if any keyword appears as a whole word in text."""
    pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    return bool(re.search(pattern, text))


def update_memory(memory: ConversationMemory, session: SessionState, text: str) -> List[str]:
    """Extract signals from user message and update narrative story.

    Returns list of turn_topics detected in this message.
    Side-effects: updates memory.story, memory.readiness_for_wisdom, session signals.
    """
    text = text.lower().strip()
    turn_topics = []

    if not memory.story.primary_concern and len(text) > 10:
        memory.story.primary_concern = text[:200]

    # ------------------------------------------------------------------
    # 1. EMOTIONAL STATES (scored with tie-breakers)
    # ------------------------------------------------------------------
    emotions = {
        "Sadness & Grief": ["sad", "low", "lonely", "depressed", "hurt", "grief", "despair", "mourning", "loss", "lost", "crying", "tears", "heavy", "hopeless", "empty", "alone", "ache", "inadequate", "unhappy", "hurts", "loneliness", "irritable", "disconnected"],
        "Anxiety & Fear": ["anxious", "anxiety", "worried", "stressed", "overwhelmed", "panic", "fear", "scared", "nervous", "tension", "uneasy", "restless", "deadline", "deadlines", "fraud", "fraudulent", "fail", "failing", "burnout", "burned out", "burning out", "panic", "panic attack", "guilty", "guilt", "burn", "burning", "paralyzed", "concentration", "exam", "exams", "insomnia", "high-stress"],
        "Anger & Frustration": ["angry", "frustrated", "irritated", "furious", "mad", "stupid", "annoying", "rage", "resentment", "hate", "fight", "yell", "hostile", "credit", "irritable"],
        "Confusion & Doubt": ["confused", "lost", "doubt", "uncertain", "directionless", "stuck", "don't know", "unsure", "clarity", "missing", "purpose", "meaning", "existential", "fraud", "fraudulent", "unethical", "guilty", "guilt", "mirror", "wondering", "failing", "ethics", "void", "search"],
        "Gratitude & Peace": ["happy", "grateful", "peace", "calm", "content", "blessed", "thankful", "joy", "serene", "better", "morning", "inspiration", "humility", "humble", "meditation"],
    }

    emotion_scores = {}
    for label, keywords in emotions.items():
        matched_keywords = [kw for kw in keywords if _has_word([kw], text)]
        if matched_keywords:
            score = len(matched_keywords)
            if label == "Confusion & Doubt":
                if _has_word(["unethical", "existential", "mirror", "wondering", "if i should", "purpose", "meaning", "failing", "fail", "lost"], text):
                    score += 5
                if "failing as a parent" in text or "fail as a parent" in text or "dharma" in text:
                    score += 15
                if _has_word(["lost"], text) and _has_word(["happiness", "dream", "success", "reached", "bought"], text):
                    score += 15
            if label == "Anxiety & Fear":
                if _has_word(["fraud", "fraudulent", "panic", "guilty", "guilt", "burnout", "burned out", "deadline", "deadlines", "fraud"], text):
                    score += 5
                if "burning me out" in text or "burning out" in text:
                    score += 10
            if label == "Sadness & Grief":
                if _has_word(["lonely", "ache", "empty", "loneliness"], text):
                    score += 3

            emotion_scores[label] = score
            if label not in turn_topics:
                turn_topics.append(label)
            if label not in memory.story.detected_topics:
                memory.story.detected_topics.append(label)

    if emotion_scores:
        ids = list(emotion_scores.keys())

        if "paralyzed" in text or "exams" in text or "exam" in text:
            emotion_scores["Anxiety & Fear"] = emotion_scores.get("Anxiety & Fear", 0) + 30
        if "irritable" in text or "fight" in text or "stressed" in text:
            if "stressed" in text:
                emotion_scores["Anxiety & Fear"] = emotion_scores.get("Anxiety & Fear", 0) + 5
            if "irritable" in text:
                emotion_scores["Anger & Frustration"] = emotion_scores.get("Anger & Frustration", 0) + 20

        if "Anxiety & Fear" in ids and "Confusion & Doubt" in ids:
            if _has_word(["fraud", "fraudulent", "lead", "promoted"], text):
                emotion_scores["Anxiety & Fear"] += 30
            if _has_word(["values", "traditions", "learn our", "screens", "parent"], text):
                emotion_scores["Confusion & Doubt"] += 40
            if _has_word(["care", "parents", "aging", "balancing", "burn", "burning"], text):
                emotion_scores["Anxiety & Fear"] += 30

        if "Confusion & Doubt" in ids and "Sadness & Grief" in ids:
            if _has_word(["purpose", "meaning", "reached", "bought", "happiness", "void"], text):
                emotion_scores["Confusion & Doubt"] += 40
            if _has_word(["grandfather", "lost my", "last week"], text):
                emotion_scores["Sadness & Grief"] += 40
            if _has_word(["startup", "start over", "should i", "savings", "confidence"], text):
                emotion_scores["Confusion & Doubt"] += 40

        if "Gratitude & Peace" in ids:
            if _has_word(["humility", "success", "doer", "full of myself"], text):
                emotion_scores["Gratitude & Peace"] += 40
            if (_has_word(["morning", "meditation", "inspiration", "beautiful"], text)
                    and not _has_word(["fear", "anxious", "low", "lost", "stuck", "angry", "hostile", "resentful", "frustrated"], text)
                    and "Routine Request" not in turn_topics):
                emotion_scores["Gratitude & Peace"] += 40

        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            memory.story.emotional_state = best_emotion
            session.add_signal(SignalType.EMOTION, best_emotion, 0.85)

    # ------------------------------------------------------------------
    # 2. LIFE DOMAINS (scored with tie-breakers)
    # ------------------------------------------------------------------
    domains = {
        "Career & Finance": ["work", "job", "office", "career", "boss", "colleague", "promotion", "salary", "money", "finance", "debt", "business", "startup", "interview", "hiring", "deadline", "deadlines", "workplace", "hostile", "inflation", "balance", "desk"],
        "Relationships": ["relationship", "partner", "marriage", "wife", "husband", "dating", "boyfriend", "girlfriend", "breakup", "divorce", "love", "crush", "ex", "fight", "social circle"],
        "Family": ["family", "parents", "children", "mother", "father", "son", "daughter", "sister", "brother", "kids", "mom", "dad", "grandparents", "home", "grandfather", "grandmother", "traditions", "parenting", "elderly", "parents", "aging", "kids", "baby", "sleep", "birthday", "gift"],
        "Physical Health": ["health", "digestion", "tired", "sleep", "body", "pain", "disease", "symptom", "weight", "exercise", "energy", "fatigue", "sick", "hurting", "burnout", "fever", "weak", "gut", "insomnia"],
        "Ayurveda & Wellness": ["ayurveda", "dosha", "pitta", "kapha", "vata", "herbs", "remedy", "cleanse", "routine", "dinacharya", "oil", "massage", "natural", "tea", "rejuvenate", "supplement", "diet", "meal"],
        "Yoga Practice": ["yoga", "asana", "posture", "flexibility", "strength", "surya", "namaskar", "hatha", "vinyasa", "routine", "yogic"],
        "Meditation & Mind": ["meditation", "focus", "mind", "concentration", "mindfulness", "dhyana", "awareness", "stillness", "thoughts", "distraction", "mental", "soul", "eternal", "meditated", "mindful"],
        "Spiritual Growth": ["dharma", "karma", "god", "soul", "spirit", "enlightenment", "purpose", "meaning", "faith", "prayer", "devotion", "bhakti", "divine", "sacred", "scripture", "gita", "missing", "ethical", "unethical", "values", "house", "dream", "existential", "void", "search", "philosophy", "upanishad", "humility", "humble"],
        "Panchang & Astrology": ["panchang", "tithi", "nakshatra", "muhurat", "shubh", "calendar", "festival", "vedic astrology", "jyotish", "moon", "waxing", "waning"],
        "Self-Improvement": ["discipline", "growth", "learning", "habits", "productivity", "goals", "confidence", "motivation", "success", "failure", "study", "instagram", "fraud", "fraudulent", "started", "starting", "inadequate", "startup", "fail", "failed", "exam", "exams", "concentration", "focus", "journal", "reflection"],
        "General Life": ["lonely", "moving", "city", "new place", "weekend", "weekends", "phone", "staring", "everyone", "anyone", "understand", "understands"],
    }

    domain_scores = {}
    for label, keywords in domains.items():
        matches = [kw for kw in keywords if _has_word([kw], text)]
        if matches:
            score = len(matches)
            if label == "Spiritual Growth" and _has_word(["unethical", "ethical", "dream", "house", "meaning", "purpose"], text):
                score += 5
            if label == "Self-Improvement" and _has_word(["instagram", "fraud", "fail", "failed", "startup"], text):
                score += 5
            if label == "General Life" and _has_word(["understand", "understands", "lonely"], text):
                score += 5
            domain_scores[label] = score
            if label not in turn_topics:
                turn_topics.append(label)
            if label not in memory.story.detected_topics:
                memory.story.detected_topics.append(label)

    found_domain = False
    if domain_scores:
        dids = domain_scores.keys()
        if _has_word(["exam", "exams", "concentration", "focus", "study", "studies"], text):
            if "Self-Improvement" in dids:
                domain_scores["Self-Improvement"] += 30
        if _has_word(["gita", "upanishad", "dharma", "ethics", "unethical", "ethics", "humility", "void", "meaning", "purpose"], text):
            if "Spiritual Growth" in dids:
                domain_scores["Spiritual Growth"] += 50
        if "Diet Plan" in turn_topics and "Ayurveda & Wellness" in dids:
            domain_scores["Ayurveda & Wellness"] += 40
        if "Routine Request" in turn_topics:
            if "Yoga Practice" in dids:
                domain_scores["Yoga Practice"] += 40
            if "Meditation & Mind" in dids:
                domain_scores["Meditation & Mind"] += 40
        if "Puja Guidance" in turn_topics:
            if "Family" in dids:
                domain_scores["Family"] += 40
            if "Spiritual Growth" in dids:
                domain_scores["Spiritual Growth"] += 40
        if "Family" in dids and _has_word(["baby", "parenting", "kids"], text):
            domain_scores["Family"] += 30

        best_domain = max(domain_scores, key=domain_scores.get)
        memory.story.life_area = best_domain
        session.add_signal(SignalType.LIFE_DOMAIN, best_domain, 0.9)
        found_domain = True

    if not found_domain and len(text.split()) > 5:
        if not memory.story.life_area:
            memory.story.life_area = "General Life"
            session.add_signal(SignalType.LIFE_DOMAIN, "General Life", 0.5)
            if "General Life" not in turn_topics:
                turn_topics.append("General Life")

    # ------------------------------------------------------------------
    # 3. SPECIAL INTENTS
    # ------------------------------------------------------------------
    temple_keywords = ["temple", "mandir", "pilgrimage", "shrine", "darshan", "puri", "kashi", "tirupati", "badrinath", "kedarnath", "dwarka", "rameswaram", "somnath", "visit"]
    if _has_word(temple_keywords, text):
        memory.story.temple_interest = text[:100]
        session.add_signal(SignalType.INTENT, "Temple & Pilgrimage", 0.8)
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.35)

    # Planetary / astrological signals — clear dharmic context, high readiness
    planetary_keywords = [
        "shani", "saturn", "rahu", "ketu", "mangal", "guru", "brihaspati",
        "budh", "shukra", "surya", "chandra", "graha", "gochara",
        "nakshatra", "rashi", "kundali", "lagna", "dasha", "antardasha",
        "mahadasha", "sadesati", "sade sati", "dhrishti", "jyotish",
        "planetary", "vedic astrology", "kundli", "janma",
    ]
    if _has_word(planetary_keywords, text):
        session.add_signal(SignalType.INTENT, "Planetary Guidance", 0.9)
        if "Planetary Guidance" not in turn_topics:
            turn_topics.append("Planetary Guidance")
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.60)

    breathing_keywords = ["breath", "breathing", "pranayama", "inhale", "exhale", "lungs", "air"]
    if _has_word(breathing_keywords, text):
        session.add_signal(SignalType.INTENT, "Pranayama (Breathwork)", 0.8)
        if not found_domain:
            memory.story.life_area = "Yoga Practice"
            session.add_signal(SignalType.LIFE_DOMAIN, "Yoga Practice", 0.9)

    # 4. VERSE & SCRIPTURE INTENTS
    verse_keywords = ["verse", "verses", "scripture", "scriptures", "gita", "upanishad", "upanishads", "mantra", "mantras", "remind me", "wisdom", "sloka", "shloka", "philosophy"]
    intent_keywords = ["give", "tell", "provide", "share", "need", "want", "love", "send", "suggest", "provide me", "how to", "show", "read", "?"]
    if _has_word(verse_keywords, text) and (any(w in text for w in intent_keywords)):
        session.add_signal(SignalType.INTENT, "Verse Request", 0.9)
        if "Verse Request" not in turn_topics:
            turn_topics.append("Verse Request")
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.6)

    # 5. PRODUCT & SERVICE INTENTS
    if "Verse Request" not in turn_topics:
        buy_keywords = ["buy", "purchase", "order", "price", "cost", "shop", "store", "where can i", "how much", "products", "item", "items"]
        product_items = ["rudraksha", "mala", "diya", "incense", "dhoop", "havan", "idol", "thali", "book", "yantra", "murti", "gangajal", "oil", "tea", "supplement", "herbs", "ayurvedic", "journal", "pendant", "bracelet"]
        if _has_word(buy_keywords, text) or (_has_word(product_items, text) and ("?" in text or "want" in text or "need" in text or "is there" in text or "suggest" in text or "love" in text or "get" in text)):
            session.add_signal(SignalType.INTENT, "Product Inquiry", 0.9)
            if "Product Inquiry" not in turn_topics:
                turn_topics.append("Product Inquiry")
            memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.6)

    # 6. PROCEDURAL & ROUTINE INTENTS
    routine_keywords = ["routine", "plan", "program", "schedule", "daily", "day", "morning", "evening", "night", "breaks", "habit", "starter"]
    routine_activity = ["yoga", "meditation", "breaks", "exercise", "sleep", "nidra", "yogic", "meditated", "mindful", "moon", "phases", "alignment"]
    if _has_word(routine_keywords, text) and (_has_word(routine_activity, text) or _has_word(["how", "give", "create", "provide"], text)):
        session.add_signal(SignalType.INTENT, "Routine Request", 0.9)
        if "Routine Request" not in turn_topics:
            turn_topics.append("Routine Request")
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.5)

    puja_keywords = ["puja", "pooja", "ritual", "ceremony", "altar", "mandir", "home temple", "worship", "spiritual corner"]
    puja_action = ["plan", "how", "steps", "items", "setup", "prepare", "perform", "instructions", "direction", "essential"]
    if _has_word(puja_keywords, text) and (_has_word(puja_action, text) or "?" in text):
        session.add_signal(SignalType.INTENT, "Puja Guidance", 0.9)
        if "Puja Guidance" not in turn_topics:
            turn_topics.append("Puja Guidance")
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.5)
        if "Product Inquiry" in turn_topics and _has_word(["setup", "direction", "corner"], text):
            turn_topics.remove("Product Inquiry")

    diet_keywords = ["diet", "food", "eat", "meal", "meals", "breakfast", "lunch", "dinner", "prep", "nutrition"]
    diet_context = ["plan", "routine", "ayurvedic", "sattvic", "pitta", "kapha", "vata", "dosha"]
    if _has_word(diet_keywords, text) and _has_word(diet_context, text):
        session.add_signal(SignalType.INTENT, "Diet Plan", 0.9)
        if "Diet Plan" not in turn_topics:
            turn_topics.append("Diet Plan")
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.5)

    # ------------------------------------------------------------------
    # Readiness boosters
    # ------------------------------------------------------------------
    memory.add_user_quote(session.turn_count, text[:500])

    if memory.story.emotional_state:
        memory.record_emotion(session.turn_count, memory.story.emotional_state, "moderate")
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.15)
        # Cumulative boost: second+ emotional turn signals sustained context → ready for dharmic guidance
        if session.turn_count >= 2:
            memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.20)

    if len(text) > 100:
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.2)

    wellness_query_keywords = ["how do i", "what is", "routine", "technique", "practice", "method", "steps"]
    if any(w in text for w in wellness_query_keywords) and "?" in text:
        memory.readiness_for_wisdom = min(1.0, memory.readiness_for_wisdom + 0.3)

    return turn_topics
