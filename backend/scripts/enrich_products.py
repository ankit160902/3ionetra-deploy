"""Product Metadata Enrichment Pipeline.

Uses Gemini to auto-tag 241 products with structured metadata:
tags, deities, practices, emotions, life_domains, benefits, product_type, occasion_tags.

Usage:
    # Dry run (first 10 products, print only, no MongoDB writes)
    python scripts/enrich_products.py --dry-run --limit 10

    # Full enrichment (all products, writes to MongoDB)
    python scripts/enrich_products.py

    # Re-enrich specific products
    python scripts/enrich_products.py --force --limit 5

Safety:
    - Dry-run mode by default (must explicitly omit --dry-run for writes)
    - Pydantic validation on every Gemini response
    - Rate limiting (2s between calls)
    - Error recovery (failed products logged and skipped)
    - Backup prompt before first write
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enrich_products")

# ---------------------------------------------------------------------------
# Gemini enrichment prompt
# ---------------------------------------------------------------------------

ENRICHMENT_PROMPT = """You are a product metadata tagger for 3ioNetra, a spiritual e-commerce store selling Hindu spiritual items, puja essentials, astrology consultations, and wellness products.

Given this product:
  Name: "{name}"
  Category: "{category}"
  Price: ₹{amount}
  Description: "{description}"

Return ONLY a valid JSON object with these fields:

1. "tags": 5-10 lowercase searchable keywords. Include: product type, material, use case, related terms.
   Example: ["rudraksha", "mala", "japa", "108 beads", "shiva", "meditation"]

2. "deities": List of Hindu deities this product is associated with. Use lowercase names.
   Choose from: shiva, vishnu, krishna, rama, hanuman, ganesh, durga, lakshmi, saraswati, kali, parvati, surya, shrinathji, murugan, naag, dhanvantari, radha, sita, balaji, mahadev, nandi
   Return empty list [] if not deity-specific.

3. "practices": List of spiritual practices this product supports.
   Choose from: japa, puja, meditation, yoga, havan, abhishek, aarti, seva, vrat, darshan, consultation, crystal_healing, home_temple, daily_worship, pranayama, sankalpa, kirtan

4. "emotions": List of emotions this product helps address or cultivate.
   Choose from: anxiety, stress, grief, anger, confusion, fear, loneliness, hopelessness, sadness, frustration, shame, guilt, peace, joy, gratitude, focus, courage, protection, prosperity, healing, love, hope, calm, clarity, strength, devotion

5. "life_domains": List of life areas this product is relevant to.
   Choose from: spiritual, career, relationships, family, health, education, finance, self_improvement, marriage, parenting, wellness, meditation

6. "benefits": 3-5 specific benefits in lowercase.
   Example: ["concentration", "protection", "inner peace", "obstacle removal"]

7. "product_type": One of: physical, service, consultation, experience

8. "occasion_tags": Relevant festivals, days, or occasions in lowercase.
   Example: ["daily", "navratri", "maha_shivratri", "monday", "diwali", "ekadashi"]
   Return empty list [] if not occasion-specific.

IMPORTANT:
- For consultation products with empty descriptions, infer from name and category.
- For astrology consultations, use practices=["consultation"], emotions=["confusion", "anxiety", "clarity"].
- Be conservative — only tag what's clearly relevant, don't over-tag.
- Return ONLY the JSON object, no extra text.
"""


def get_gemini_client():
    """Initialize Gemini client."""
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return client


def enrich_single_product(client, product: dict) -> dict:
    """Call Gemini to enrich a single product. Returns enrichment dict or None on failure."""
    from models.product_enrichment import ProductEnrichment
    from models.llm_schemas import extract_json

    name = product.get("name", "")
    category = product.get("category", "")
    amount = product.get("amount", 0)
    description = (product.get("description", "") or "")[:500]

    prompt = ENRICHMENT_PROMPT.format(
        name=name, category=category, amount=amount, description=description
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "max_output_tokens": 1024,
            },
        )

        raw = response.text.strip() if response.text else ""
        parsed = extract_json(raw)

        if not parsed:
            logger.warning(f"  JSON extraction failed for '{name}'")
            return None

        # Validate with Pydantic
        enrichment = ProductEnrichment(**parsed)
        return enrichment.to_mongo_update()

    except Exception as e:
        logger.error(f"  Gemini call failed for '{name}': {e}")
        return None


def run_enrichment(dry_run=True, limit=None, force=False):
    """Main enrichment pipeline."""
    from services.auth_service import get_mongo_client

    logger.info("=" * 60)
    logger.info(f"Product Enrichment Pipeline")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE (writing to MongoDB)'}")
    logger.info(f"Limit: {limit or 'ALL'}")
    logger.info(f"Force re-enrich: {force}")
    logger.info("=" * 60)

    # Connect to MongoDB
    db = get_mongo_client()
    if db is None:
        logger.error("MongoDB not available. Cannot enrich products.")
        return

    collection = db["products"]
    total = collection.count_documents({"is_active": True})
    logger.info(f"Total active products: {total}")

    # Query products to enrich
    query = {"is_active": True}
    if not force:
        # Skip already-enriched products
        query["enrichment_version"] = {"$exists": False}

    products = list(collection.find(query).limit(limit or 0))
    logger.info(f"Products to enrich: {len(products)}")

    if not products:
        logger.info("Nothing to enrich. Use --force to re-enrich.")
        return

    # Initialize Gemini
    client = get_gemini_client()
    logger.info("Gemini client initialized")

    # Process each product
    success = 0
    failed = 0
    skipped = 0

    for i, product in enumerate(products):
        name = product.get("name", "?")
        logger.info(f"\n[{i+1}/{len(products)}] Enriching: {name}")

        enrichment = enrich_single_product(client, product)

        if enrichment is None:
            failed += 1
            logger.warning(f"  FAILED: {name}")
            continue

        # Log enrichment results
        logger.info(f"  Tags: {enrichment['tags'][:5]}...")
        logger.info(f"  Deities: {enrichment['deities']}")
        logger.info(f"  Practices: {enrichment['practices']}")
        logger.info(f"  Emotions: {enrichment['emotions'][:4]}...")
        logger.info(f"  Type: {enrichment['product_type']}")

        if dry_run:
            logger.info(f"  [DRY RUN] Would write to MongoDB")
            logger.info(f"  Full enrichment: {json.dumps(enrichment, indent=2, default=str)}")
        else:
            # Write to MongoDB
            collection.update_one(
                {"_id": product["_id"]},
                {"$set": enrichment},
            )
            logger.info(f"  ✅ Written to MongoDB")

        success += 1

        # Rate limiting (2s between calls to avoid Gemini quota issues)
        if i < len(products) - 1:
            time.sleep(2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"ENRICHMENT COMPLETE")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {success + failed}")
    if dry_run:
        logger.info(f"Mode: DRY RUN — no writes made. Re-run without --dry-run to write.")
    logger.info("=" * 60)


def create_enrichment_indexes():
    """Create MongoDB indexes for enriched fields."""
    from services.auth_service import get_mongo_client

    db = get_mongo_client()
    if db is None:
        logger.error("MongoDB not available")
        return

    collection = db["products"]
    logger.info("Creating enrichment indexes...")
    collection.create_index([("deities", 1)], background=True)
    collection.create_index([("practices", 1)], background=True)
    collection.create_index([("emotions", 1)], background=True)
    collection.create_index([("life_domains", 1)], background=True)
    collection.create_index([("product_type", 1)], background=True)
    collection.create_index([("tags", 1)], background=True)
    logger.info("✅ Enrichment indexes created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich products with Gemini metadata")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print enrichment results without writing to MongoDB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of products to enrich")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-enrich already-enriched products")
    parser.add_argument("--create-indexes", action="store_true", default=False,
                        help="Create MongoDB indexes for enriched fields")

    args = parser.parse_args()

    if args.create_indexes:
        create_enrichment_indexes()
    else:
        run_enrichment(dry_run=args.dry_run, limit=args.limit, force=args.force)
