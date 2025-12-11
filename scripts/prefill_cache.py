# scripts/prefill_cache.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level up from scripts/
sys.path.insert(0, str(PROJECT_ROOT))
import json, requests, time
from pathlib import Path
from src.kb import PLACES_KB
# make script importable when run from scripts/ directory


CACHE_PATH = Path("data/place_cache.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CACHE_PATH.exists():
    CACHE_PATH.write_text("{}", encoding="utf-8")

def wiki_summary(query):
    q = requests.utils.requote_uri(query.strip())
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent":"VoyageHelper/1.0"})
        if r.status_code == 200:
            j = r.json()
            return {
                "title": j.get("title"),
                "description": j.get("extract"),
                "image": (j.get("thumbnail") or {}).get("source") or ""
            }
    except Exception as e:
        print("wiki error", e)
    return None

cache = json.loads(CACHE_PATH.read_text(encoding="utf-8") or "{}")

for key in list(PLACES_KB.keys()):
    entry = PLACES_KB[key]
    has_desc = isinstance(entry, dict) and entry.get("description")
    if key in cache or has_desc:
        print("skip", key)
        continue
    print("Fetching wiki for", key)
    w = wiki_summary(key)
    if w:
        card = {
            "type":"place_card",
            "key": key,
            "title": w.get("title") or key.title(),
            "description": w.get("description") or "",
            "sights": PLACES_KB[key] if isinstance(PLACES_KB[key], list) else [],
            "best_time": "",
            "how_to_reach": "",
            "hotels": [],
            "food": [],
            "image": w.get("image") or ""
        }
        cache[key] = card
        time.sleep(0.5)
    else:
        print("no wiki for", key)

CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
print("Done. Cached", len(cache), "entries.")
