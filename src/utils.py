# src/utils.py
import re
import time
from functools import wraps
from difflib import get_close_matches

# Optional: Wikipedia integration
# Requires: pip install wikipedia
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
    wikipedia.set_lang("en")
except Exception:
    WIKIPEDIA_AVAILABLE = False

from src.kb import PLACES_KB

# Base city list from KB + extra known places
BASE_CITIES = list(set(PLACES_KB.keys()) | {
    "goa", "mangalore", "bangalore", "udupi", "delhi",
    "mumbai", "kerala", "mysore", "chennai", "pune"
})

def _normalize_text(text: str) -> str:
    return text.lower().strip()

def _words_from_text(text: str):
    return re.findall(r"[a-zA-Z]+(?:['-][a-zA-Z]+)?", text.lower())

def extract_cities(text: str):
    """
    Extract city/place names from free text.

    Strategy:
    1. Exact whole-word matches against BASE_CITIES.
    2. Fuzzy matches for single words using difflib.get_close_matches.
    3. Substring check for multi-word places (keeps previous behavior).
    Returns a list preserving the order of first appearance in the text.
    """
    if not text:
        return []

    text_l = _normalize_text(text)
    words = _words_from_text(text_l)

    found = []
    found_set = set()

    # 1) Exact whole-word matches (respects word boundaries)
    for city in BASE_CITIES:
        pattern = r"\b" + re.escape(city.lower()) + r"\b"
        if re.search(pattern, text_l):
            if city not in found_set:
                found.append(city)
                found_set.add(city)

    # 2) Fuzzy match each token against BASE_CITIES
    for w in words:
        match = get_close_matches(w, BASE_CITIES, n=1, cutoff=0.72)
        if match:
            candidate = match[0]
            if candidate not in found_set:
                found.append(candidate)
                found_set.add(candidate)

    # 3) Multi-word / substring detection (covers cases like 'south goa', 'new delhi')
    for city in BASE_CITIES:
        if city.lower() in text_l and city not in found_set:
            found.append(city)
            found_set.add(city)

    return found

# -----------------------
# Simple TTL cache helper
# -----------------------
def ttl_cache(ttl=3600, maxsize=256):
    """
    Simple function-level TTL cache. Not thread-safe — OK for small dev setups.
    Use Redis or functools.lru_cache+timestamp for production.
    """
    def decorator(func):
        cache = {}
        order = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()

            # hit and not expired
            if key in cache:
                entry_time, value = cache[key]
                if now - entry_time < ttl:
                    return value
                # expired
                try:
                    order.remove(key)
                except ValueError:
                    pass
                cache.pop(key, None)

            # compute and store
            value = func(*args, **kwargs)
            # enforce maxsize
            if len(order) >= maxsize:
                oldest = order.pop(0)
                cache.pop(oldest, None)
            cache[key] = (now, value)
            order.append(key)
            return value

        return wrapper
    return decorator

# -----------------------
# Wikipedia helper
# -----------------------
@ttl_cache(ttl=1800, maxsize=512)
def fetch_wikipedia_summary(query: str, sentences: int = 3):
    """
    Search Wikipedia for `query` and return (title, summary, url).
    Returns (None, None, None) if no suitable page found or wikipedia is unavailable.

    Behavior:
    - Try wikipedia.search to get candidate titles.
    - For hotel/accommodation queries, prefer pages mentioning hotel/accommodation/resort.
    - Fall back to the first sensible page (e.g., city page).
    """
    if not WIKIPEDIA_AVAILABLE:
        return (None, None, None)

    q = (query or "").strip()
    if not q:
        return (None, None, None)

    try:
        results = wikipedia.search(q, results=6)
        if not results:
            return (None, None, None)

        fallback = None
        query_lower = q.lower()
        prefer_terms = ["hotel", "accommodation", "resort", "lodg", "inn", "hostel", "stay"]

        for title in results:
            try:
                # Avoid auto_suggest to reduce unexpected redirects; use direct titles
                page = wikipedia.page(title, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                # try the first non-ambiguous option if available
                opts = e.options
                if opts:
                    try:
                        page = wikipedia.page(opts[0], auto_suggest=False)
                    except Exception:
                        continue
                else:
                    continue
            except Exception:
                continue

            # get a short summary
            try:
                summary = wikipedia.summary(page.title, sentences=sentences)
            except Exception:
                summary = (page.content[:500] + "...") if page.content else ""

            # Decide whether this page is a strong match
            page_text = (page.title + " " + (summary or "")).lower()
            if any(term in query_lower for term in prefer_terms):
                # If user explicitly asked about hotels/accommodation prefer pages with those words
                if any(term in page_text for term in prefer_terms):
                    return (page.title, summary, page.url)
                else:
                    # keep as fallback (city/tourism page)
                    if not fallback:
                        fallback = (page.title, summary, page.url)
            else:
                # For generic queries accept the first meaningful page
                return (page.title, summary, page.url)

        if fallback:
            return fallback

        return (None, None, None)

    except Exception:
        # any wiki-level error -> graceful None
        return (None, None, None)

# -----------------------
# KB convenience functions
# -----------------------
def get_kb_hotels_for_city(city_name: str):
    """
    Return hotels list from PLACES_KB for a city (case-insensitive), or None.
    """
    if not city_name:
        return None
    city_lower = city_name.lower()
    # PLACES_KB keys might be lowercase or mixed; try direct then lower-key lookup
    if city_name in PLACES_KB:
        kb_city = PLACES_KB[city_name]
    elif city_lower in PLACES_KB:
        kb_city = PLACES_KB[city_lower]
    else:
        # try close matches
        match = get_close_matches(city_lower, list(PLACES_KB.keys()), n=1, cutoff=0.72)
        if match:
            kb_city = PLACES_KB[match[0]]
        else:
            kb_city = None

    if not kb_city:
        return None

    return kb_city.get("hotels") or kb_city.get("accommodation") or None
