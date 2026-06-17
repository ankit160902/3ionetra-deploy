
import json
import urllib.request
import re
import os

def clean_text(text):
    if not text:
        return ""
    if not isinstance(text, str):
        return text
    # Remove HTML tags if any
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, ' ', text)
    # Remove surrogates and non-utf8 characters
    text = "".join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))
    # Replace common issues
    return ' '.join(text.split())

def fetch_and_update():
    url = "https://my3ionetra.com/products.json?limit=250"
    print(f"Fetching products from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            site_products = data.get("products", [])
            print(f"Found {len(site_products)} products.")
            
            formatted_products = []
            for p in site_products:
                name = clean_text(p.get("title", "Unknown Product"))
                handle = p.get("handle", "")
                product_url = f"https://my3ionetra.com/products/{handle}"
                
                # Get price
                variants = p.get("variants", [])
                amount = 0.0
                if variants:
                    try:
                        amount = float(variants[0].get("price", 0))
                    except (ValueError, TypeError):
                        amount = 0.0
                
                # Get image
                images = p.get("images", [])
                image_url = ""
                if images:
                    image_url = images[0].get("src", "")
                
                # Get category
                category = clean_text(p.get("product_type", ""))
                if not category and p.get("tags"):
                    category = clean_text(p.get("tags")[0])
                if not category:
                    category = "Spiritual"
                
                description = clean_text(p.get("body_html", ""))
                
                formatted_products.append({
                    "name": name,
                    "category": category,
                    "amount": amount,
                    "currency": "INR",
                    "description": description,
                    "image_url": image_url,
                    "product_url": product_url,
                    "is_active": True
                })
            
            # Save to JSON file
            json_path = os.path.abspath("backend/scripts/products.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(formatted_products, f, indent=4, ensure_ascii=False)
            
            print(f"Saved {len(formatted_products)} products to {json_path}")
            
            # Now generate the seed_products.py content to load this JSON
            seed_script_content = '''import os
import json
from pymongo import MongoClient
from datetime import datetime, timezone

# MongoDB Connection (read from environment — never hardcode credentials)
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ.get("DATABASE_NAME", "spiritual_voice_bot")

def get_db():
    import time
    for i in range(3):
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            return client[DB_NAME]
        except Exception as e:
            print(f"Connection attempt {i+1} failed: {e}")
            if i < 2:
                time.sleep(2)
            else:
                raise

def seed_products():
    db = get_db()
    products_collection = db["products"]
    
    # Load from JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "products.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    print("Cleanup: Removing existing products...")
    products_collection.delete_many({})
    
    print(f"Seeding {len(products)} products...")
    
    # Add timestamps before seeding
    for p in products:
        p["created_at"] = datetime.now(timezone.utc)
        p["updated_at"] = datetime.now(timezone.utc)
        
    if products:
        # We'll try to insert in batches to handle any remaining encoding errors gracefully
        batch_size = 50
        for i in range(0, len(products), batch_size):
            batch = products[i:i+batch_size]
            try:
                products_collection.insert_many(batch)
                print(f"Inserted batch {i//batch_size + 1} ({len(batch)} products)")
            except Exception as e:
                print(f"Batch {i//batch_size + 1} failed, trying individually: {e}")
                for p in batch:
                    try:
                        products_collection.insert_one(p)
                    except Exception as ie:
                        print(f"Failed to insert '{p['name']}': {ie}")

    print("Seeding process finished.")

if __name__ == "__main__":
    seed_products()
'''
            
            target_path = os.path.abspath("backend/scripts/seed_products.py")
            with open(target_path, "w") as f:
                f.write(seed_script_content)
            
            print(f"Successfully updated {target_path}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_and_update()
