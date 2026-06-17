"""Backfill empty product descriptions via Gemini.

Queries MongoDB for products where description is empty/null, generates
a 2-3 sentence description from name + category + amount, and writes
it back. Safe to re-run (skips products that already have descriptions).

Usage:
    python scripts/backfill_descriptions.py              # dry-run (prints, no writes)
    python scripts/backfill_descriptions.py --write       # actually write to MongoDB
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from services.auth_service import get_mongo_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_description(name: str, category: str, amount: float, currency: str = "INR") -> str:
    """Use Gemini to generate a product description."""
    try:
        from google import genai

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        prompt = (
            f"Write a 2-3 sentence product description for a spiritual/religious product.\n"
            f"Product name: {name}\n"
            f"Category: {category}\n"
            f"Price: {currency} {amount}\n\n"
            f"Write in a warm, inviting tone suitable for a spiritual store. "
            f"Mention what the product is, its spiritual significance, and who it's for. "
            f"Do NOT use markdown. Plain flowing sentences only. Max 60 words."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini call failed for '{name}': {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Backfill empty product descriptions")
    parser.add_argument("--write", action="store_true", help="Actually write to MongoDB (default is dry-run)")
    args = parser.parse_args()

    db = get_mongo_client()
    if db is None:
        logger.error("MongoDB not available. Set MONGODB_URI in .env")
        return 1

    collection = db["products"]

    # Find products with empty or missing descriptions
    query = {
        "is_active": True,
        "$or": [
            {"description": {"$exists": False}},
            {"description": None},
            {"description": ""},
        ],
    }
    empty_products = list(collection.find(query))
    logger.info(f"Found {len(empty_products)} products with empty descriptions")

    if not empty_products:
        logger.info("Nothing to backfill!")
        return 0

    updated = 0
    failed = 0
    for i, product in enumerate(empty_products, 1):
        name = product.get("name", "Unknown")
        category = product.get("category", "")
        amount = product.get("amount", 0)
        currency = product.get("currency", "INR")

        logger.info(f"[{i}/{len(empty_products)}] Generating description for: {name}")
        description = generate_description(name, category, amount, currency)

        if not description:
            logger.warning(f"  Failed to generate description for: {name}")
            failed += 1
            continue

        logger.info(f"  Generated: {description[:80]}...")

        if args.write:
            collection.update_one(
                {"_id": product["_id"]},
                {"$set": {"description": description}},
            )
            updated += 1
        else:
            logger.info(f"  [DRY-RUN] Would write description to MongoDB")

        # Rate limit Gemini calls
        time.sleep(0.5)

    mode = "WRITTEN" if args.write else "DRY-RUN"
    logger.info(f"\n{mode}: {updated} updated, {failed} failed, {len(empty_products)} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
