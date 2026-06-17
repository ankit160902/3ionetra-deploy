#!/usr/bin/env python3
"""
Generate expanded curated concept documents (target: 200 concepts).

Uses Gemini to create high-quality dharmic concept documents that cover:
- All 20 domain compass domains
- Core Vedanta, Yoga/Samkhya, Ritual/Dharma, Bhakti concepts
- Psychological Sanskrit vocabulary
- Practical life applications
- Meditation concepts
- Temple knowledge

Each document is 200-500 words with concept, explanation, scripture anchors,
and practical relevance. Output is a JSON file ready for ingestion.

IMPORTANT: Generated drafts need human review before ingestion.

Run:  cd backend && python scripts/generate_curated_concepts.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Concept categories with seed concepts
CONCEPT_SEEDS = {
    "core_vedanta": [
        "Brahman", "Atman", "Maya", "Moksha", "Avidya", "Viveka", "Vairagya",
        "Satchidananda", "Advaita", "Turiya", "Nirguna Brahman", "Saguna Brahman",
        "Jivanmukti", "Videhamukti", "Panchakosha", "Antahkarana", "Chit",
        "Ananda", "Sat", "Paramatman", "Ishvara", "Jiva", "Upadhi",
        "Adhyasa", "Neti Neti", "Tat Tvam Asi", "Aham Brahmasmi",
        "Prajnanam Brahma", "Ayam Atma Brahma", "Satyam Jnanam Anantam",
    ],
    "yoga_samkhya": [
        "Ashtanga Yoga", "Yama", "Niyama", "Asana", "Pranayama", "Pratyahara",
        "Dharana", "Dhyana", "Samadhi", "Chitta Vritti Nirodha", "Purusha",
        "Prakriti", "Gunas", "Sattva", "Rajas", "Tamas", "Kaivalya",
        "Abhyasa", "Ishvara Pranidhana", "Tapas", "Svadhyaya",
        "Kleshas", "Avidya", "Asmita", "Raga", "Dvesha", "Abhinivesha",
    ],
    "ritual_dharma": [
        "Yajna", "Havan", "Puja Vidhi", "Sandhya Vandana", "Agnihotra",
        "Shraddha Ceremony", "Upanayana", "Vivah Sanskar", "Annaprasana",
        "Navagraha Shanti", "Rudra Abhishekam", "Satyanarayan Katha",
        "Ganesh Chaturthi Puja", "Lakshmi Puja", "Durga Puja",
        "Maha Shivaratri", "Navratri", "Diwali Significance",
        "Holi Spiritual Meaning", "Raksha Bandhan",
    ],
    "bhakti": [
        "Navadha Bhakti", "Shravana", "Kirtana", "Smarana", "Pada Sevana",
        "Archana", "Vandana", "Dasya", "Sakhya", "Atma Nivedana",
        "Prema Bhakti", "Saranagati", "Prapatti", "Madhurya Bhava",
        "Dasya Bhava", "Vatsalya Bhava", "Sakhya Bhava", "Shanta Bhava",
        "Nama Japa", "Kirtan",
    ],
    "psychological_sanskrit": [
        "Manas", "Buddhi", "Ahamkara", "Chitta", "Samskaras", "Vasanas",
        "Kama", "Krodha", "Lobha", "Moha", "Mada", "Matsarya",
        "Shoka", "Vishada", "Bhaya", "Chinta", "Dvesha", "Irshya",
        "Santosha", "Shanti", "Sukha", "Ananda", "Karuna", "Mudita",
        "Upeksha",
    ],
    "practical_life": [
        "Nishkama Karma", "Svadharma", "Purushartha", "Artha", "Dharma in Daily Life",
        "Grihastha Dharma", "Vanaprastha", "Brahmacharya", "Ahimsa in Practice",
        "Satya in Relationships", "Dana and Charity", "Seva and Service",
        "Work-Life Balance Dharmic View", "Dealing with Failure",
        "Overcoming Anger", "Managing Anxiety through Dharma",
        "Grief and Loss in Hindu Philosophy", "Marriage and Dharma",
        "Parenting in Hindu Tradition", "Student Life and Gurukul",
        "Financial Ethics", "Environmental Dharma", "Digital Detox Dharmic",
        "Sleep and Ayurveda", "Diet in Ayurveda", "Fasting and Vrata",
        "Pilgrimage Significance", "Morning Routine Dinacharya",
        "Evening Routine Sandhya", "Meditation for Beginners",
    ],
    "meditation": [
        "Dhyana Types", "Trataka", "Yoga Nidra", "Ajapa Japa",
        "So-Ham Meditation", "Chakra Dhyana", "Kundalini Meditation",
        "Vipassana and Hindu Parallels", "Mantra Meditation",
        "Breath Awareness Anapanasati", "Walking Meditation",
        "Loving Kindness Maitri Bhavana", "Body Scan Kosha Meditation",
        "Mindfulness Smriti", "Concentration Dharana Techniques",
        "Witness Consciousness Sakshi Bhava", "Third Eye Ajna Meditation",
        "Om Meditation", "Mahamrityunjaya Mantra Japa",
        "Gayatri Mantra Significance",
    ],
    "temple_knowledge": [
        "Temple Architecture Vastu", "Garbhagriha Significance",
        "Gopuram Symbolism", "Pradakshina", "Abhishekam Ritual",
        "Temple Bells Significance", "Deepam and Light",
        "Prasadam", "Tirtha Water", "Temple Tank Pushkarini",
        "Dvajasthambam Flag Post", "Nandi Bull Symbolism",
        "Garuda Symbolism", "Vimana Tower", "Mandapa Hall",
        "Rangoli Kolam Significance", "Temple Music Nagaswaram",
        "Daily Temple Rituals", "Festival Calendar",
        "Jyotirlinga Significance", "Char Dham Yatra",
        "Shakti Peethas", "Divya Desams", "Arupadai Veedu",
        "Pancha Bhoota Sthalams", "Navagraha Temples",
        "Ashtavinayak", "Sapta Puri", "Temple Etiquette",
        "Donation Dharma in Temples",
    ],
}


def main():
    base = Path(__file__).resolve().parent.parent
    output_path = base / "data" / "raw" / "curated_concepts_generated.json"

    total = sum(len(v) for v in CONCEPT_SEEDS.values())
    print(f"Concept seed list: {total} concepts across {len(CONCEPT_SEEDS)} categories")

    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = base / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"')
                    break

    if not api_key:
        print("WARNING: No GEMINI_API_KEY found. Generating stub documents only.")
        print("Set GEMINI_API_KEY in environment or .env for LLM-generated content.")
        _generate_stubs(output_path)
        return

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        print("Falling back to stub generation.")
        _generate_stubs(output_path)
        return

    concepts = []
    for category, seed_list in CONCEPT_SEEDS.items():
        print(f"\nCategory: {category} ({len(seed_list)} concepts)")
        for i, concept in enumerate(seed_list):
            print(f"  [{i+1}/{len(seed_list)}] {concept}...", end=" ", flush=True)

            prompt = f"""Write a comprehensive spiritual knowledge document about the Hindu/Dharmic concept: "{concept}" (category: {category.replace('_', ' ')}).

Structure:
1. Sanskrit/Hindi term and meaning
2. Core explanation (what it is, why it matters)
3. Scripture references (cite specific verses from Bhagavad Gita, Upanishads, Yoga Sutras, Ramayana, Mahabharata, or Vedas)
4. Practical relevance (how to apply in modern life)

Requirements:
- 200-500 words
- Include at least 2 scripture references with approximate verse numbers
- Write in clear, accessible English
- Include the Sanskrit term in Devanagari if applicable
- Focus on practical spiritual wisdom, not just academic knowledge
- Write as flowing prose, not bullet points"""

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.4, "max_output_tokens": 800},
                )
                text = (response.text or "").strip()
                if text:
                    concepts.append({
                        "text": text,
                        "scripture": "Curated Concepts",
                        "reference": f"concept_{category}_{concept.lower().replace(' ', '_')}",
                        "type": "curated_concept",
                        "source": "curated_concept",
                        "topic": f"{category}, {concept}",
                        "tradition": _category_to_tradition(category),
                    })
                    print("OK")
                else:
                    print("EMPTY")
            except Exception as e:
                print(f"FAILED: {e}")

            # Rate limiting
            time.sleep(0.5)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(concepts)} concept documents → {output_path}")
    print("IMPORTANT: Review generated content before ingesting with ingest_all_data.py")


def _category_to_tradition(category: str) -> str:
    mapping = {
        "core_vedanta": "vedanta",
        "yoga_samkhya": "yoga",
        "ritual_dharma": "smriti",
        "bhakti": "vedanta",
        "psychological_sanskrit": "vedanta",
        "practical_life": "smriti",
        "meditation": "yoga",
        "temple_knowledge": "kshetra",
    }
    return mapping.get(category, "general")


def _generate_stubs(output_path: Path):
    """Generate minimal stub documents when Gemini is unavailable."""
    concepts = []
    for category, seed_list in CONCEPT_SEEDS.items():
        for concept in seed_list:
            concepts.append({
                "text": f"{concept} — a key concept in {category.replace('_', ' ')}. [STUB: Replace with LLM-generated content]",
                "scripture": "Curated Concepts",
                "reference": f"concept_{category}_{concept.lower().replace(' ', '_')}",
                "type": "curated_concept",
                "source": "curated_concept",
                "topic": f"{category}, {concept}",
                "tradition": _category_to_tradition(category),
            })

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(concepts)} STUB concept documents → {output_path}")


if __name__ == "__main__":
    main()
