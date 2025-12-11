# src/app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import logging
import requests
import json
from urllib.parse import quote
from pathlib import Path
from src.utils import extract_cities, fetch_wikipedia_summary, get_kb_hotels_for_city
from src.kb import PLACES_KB, TRAIN_SCHEDULES, BUS_SCHEDULES, TICKET_STEPS

# -----------------------
# Logging & model loading
# -----------------------
logger = logging.getLogger("voyagehelper")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("voyagehelper.log")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(ch)

MODEL_PATH = "model/intent_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    logger.error("Model not found at %s. Run src/train_intent.py first.", MODEL_PATH)
    raise FileNotFoundError("Model not found. Run src/train_intent.py first.")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded model from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    raise

app = Flask(__name__, static_folder="static")
CORS(app)  # allow frontend origins (for dev)

# -----------------------
# Runtime state & config
# -----------------------
contexts = {}  # in-memory context per remote_addr

def ctx_key():
    return request.remote_addr or "anonymous"

# When intent_proba < CONFIDENCE_THRESHOLD -> give low-confidence fallback
CONFIDENCE_THRESHOLD = float(os.getenv("VH_CONFIDENCE_THRESHOLD", 0.35))

# Entity-first only when intent_proba >= ENTITY_FIRST_CONF
ENTITY_FIRST_CONF = float(os.getenv("VH_ENTITY_FIRST_CONF", 0.75))

# Cache + wiki settings
CACHE_PATH = Path("data/place_cache.json")
WIKI_API_TIMEOUT = 5.0

if not CACHE_PATH.parent.exists():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CACHE_PATH.exists():
    try:
        CACHE_PATH.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.warning("Could not initialize cache file at %s", CACHE_PATH)

def load_cache():
    try:
        raw = CACHE_PATH.read_text(encoding="utf-8") or "{}"
        return json.loads(raw)
    except Exception as e:
        logger.warning("Could not load cache: %s", e)
        return {}

def save_cache(cache):
    try:
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save cache: %s", e)

# -----------------------
# Place card builders (KB / Wiki)
# -----------------------
def build_place_card_from_kb(key):
    key_l = key.lower()
    if key_l not in PLACES_KB:
        return None
    entry = PLACES_KB[key_l]
    if isinstance(entry, list):
        return {
            "type": "place_card",
            "key": key_l,
            "title": key_l.title(),
            "description": "",
            "sights": entry,
            "best_time": "",
            "how_to_reach": "",
            "hotels": [],
            "food": [],
            "image": "",
            "source": "kb",
            "source_url": None
        }
    if isinstance(entry, dict):
        return {
            "type": "place_card",
            "key": key_l,
            "title": entry.get("title", key_l.title()),
            "description": entry.get("description", ""),
            "sights": entry.get("sights", []),
            "best_time": entry.get("best_time", ""),
            "how_to_reach": entry.get("how_to_reach", ""),
            "hotels": entry.get("hotels", []),
            "food": entry.get("food", []),
            "image": entry.get("image", ""),
            "source": "kb",
            "source_url": entry.get("source_url")
        }
    return None

def build_place_card_from_wiki(token, source_url=None):
    """
    Return a place_card built from Wikipedia summary. Attempts to use fetch_wikipedia_summary util if available,
    otherwise fallback to REST summary endpoint. Also tries to include thumbnail when available.
    """
    title = summary = url = image = None

    # util may or may not be present; guard it
    try:
        if callable(fetch_wikipedia_summary):
            title, summary, url = fetch_wikipedia_summary(token, sentences=4)
    except Exception:
        title = summary = url = None

    if not title:
        try:
            q = requests.utils.requote_uri(token.strip())
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
            r = requests.get(wiki_url, timeout=WIKI_API_TIMEOUT, headers={"User-Agent": "VoyageHelper/1.0"})
            if r.status_code == 200:
                j = r.json()
                title = j.get("title")
                summary = j.get("extract")
                # try to fetch image if present
                thumb = (j.get("thumbnail") or {}).get("source") or (j.get("originalimage") or {}).get("source")
                image = thumb or ""
                url = j.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{q}"
        except Exception as e:
            logger.debug("wiki REST fallback failed for %s: %s", token, e)

    if not title:
        return None

    return {
        "type": "place_card",
        "key": token.lower().replace(" ", "_"),
        "title": title or token.title(),
        "description": summary or "",
        "sights": [],
        "best_time": "",
        "how_to_reach": "",
        "hotels": [],
        "food": [],
        "image": image or "",
        "source": "wikipedia",
        "source_url": url or source_url
    }

# -----------------------
# Hotels handler (wiki -> kb fallback)
# -----------------------
def handle_hotel_query(city):
    """
    Return a consistent card for hotel/accommodation queries:
    - Try Wikipedia queries prioritizing 'Hotels in {city}', 'Accommodation in {city}', and 'Tourism in {city}'.
    - If wiki returns a summary, return it as a place_card with source=wikipedia and include KB hotels (if any) in 'hotels'.
    - If Wikipedia yields nothing useful, return KB hotels if present.
    - If still nothing, return a helper card with a pre-built web search URL so the user can continue.
    """
    if not city:
        return None

    cache = load_cache()
    pk = city.lower()

    # cached card: merge KB hotels if present and return
    try:
        if pk in cache:
            cached_card = cache[pk]
            kb_hotels = []
            try:
                kb_hotels = get_kb_hotels_for_city(pk) or []
            except Exception:
                kb_hotels = []
            if kb_hotels:
                cached_card["hotels"] = kb_hotels
            return cached_card
    except Exception:
        logger.debug("cache access issue for %s", pk)

    # try a set of wiki queries (hotel-specific first)
    queries = [
        f"Hotels in {city}",
        f"Accommodation in {city}",
        f"Lodging in {city}",
        f"Tourism in {city}",
        city
    ]

    wiki_card = None
    for q in queries:
        try:
            c = build_place_card_from_wiki(q)
            if not c:
                continue
            desc = (c.get("description") or "").lower()
            # if query is hotel-specific, prefer pages mentioning hotels
            if any(term in q.lower() for term in ["hotel", "accommodation", "lodg"]):
                if any(term in desc for term in ["hotel", "accommodation", "lodg", "resort", "inn", "hostel"]):
                    wiki_card = c
                    break
                else:
                    if not wiki_card:
                        wiki_card = c
            else:
                wiki_card = c
                break
        except Exception:
            continue

    # merge KB hotels if present
    kb_hotels = []
    try:
        kb_hotels = get_kb_hotels_for_city(city) or []
    except Exception:
        kb_hotels = []

    if wiki_card:
        if kb_hotels:
            wiki_card["hotels"] = kb_hotels
        cache[pk] = wiki_card
        save_cache(cache)
        return wiki_card

    # fallback to KB hotels only (no wikipedia summary)
    if kb_hotels:
        kb_card = build_place_card_from_kb(pk) or {
            "type": "place_card",
            "key": pk,
            "title": city.title(),
            "description": "",
            "sights": [],
            "best_time": "",
            "how_to_reach": "",
            "hotels": kb_hotels,
            "food": [],
            "image": "",
            "source": "kb",
            "source_url": None
        }
        cache[pk] = kb_card
        save_cache(cache)
        return kb_card

    # --- helpful fallback card pointing to external hotel search ---
    query = quote(f"hotels in {city}")
    google_search = f"https://www.google.com/search?q={query}"
    booking_search = f"https://www.booking.com/searchresults.html?ss={quote(city)}"
    tripadvisor_search = f"https://www.tripadvisor.in/Search?q={query}"

    fallback_desc = (
        f"Sorry — I don't have hotel listings for {city.title()} in my KB. "
        "You can try the external search links below (Google / Booking / TripAdvisor)."
    )

    fallback_card = {
        "type": "place_card",
        "key": pk + "_hotels_fallback",
        "title": f"Hotels — {city.title()}",
        "description": fallback_desc,
        "sights": [],
        "best_time": "",
        "how_to_reach": "",
        "hotels": [],
        "food": [],
        "image": "",
        "source": "web_search",
        "source_url": google_search,
        "help_urls": {
            "google": google_search,
            "booking": booking_search,
            "tripadvisor": tripadvisor_search
        }
    }

    cache[pk] = fallback_card
    save_cache(cache)
    return fallback_card

# -----------------------
# Textual reply builder (unchanged behavior)
# -----------------------
def build_reply(intent, cities, message, ctx):
    cities = [c.lower() for c in cities] if cities else []
    if intent == "greeting":
        if ctx.get("last_intent") == "greeting":
            return "Hi again! How can I help?"
        return "Hello! I'm VoyageHelper. Where do you want to travel?"
    if intent == "places_to_visit":
        city = (cities[0] if cities else None) or ctx.get("last_city")
        if city:
            city_l = city.lower()
            if city_l in PLACES_KB:
                entry = PLACES_KB[city_l]
                sights = entry.get("sights") if isinstance(entry, dict) else (entry if isinstance(entry, list) else [])
                if sights:
                    return f"Top places in {city.title()}:\n- " + "\n- ".join(sights)
                return f"Sorry, I don't have places data for {city.title()}."
            return f"Sorry, I don't have data for {city.title()}. Try another city."
        return "Which city? e.g., 'Places to visit in Goa'"
    if intent == "ticket_booking":
        return ("Train (IRCTC):\n1. Open IRCTC website/app\n2. Login\n3. Enter From/To & date\n4. Choose train & class\n5. Enter passenger details\n6. Pay and download ticket\n\n"
                "Bus (RedBus/KSRTC):\n1. Open website/app\n2. Search route\n3. Select bus & seat\n4. Pay and get e-ticket")
    if intent == "train_timing":
        if len(cities) >= 2:
            origin, dest = cities[0].lower(), cities[1].lower()
            key = (origin, dest)
            if key in TRAIN_SCHEDULES:
                return f"Train schedule {origin.title()} → {dest.title()}:\n- " + "\n- ".join(TRAIN_SCHEDULES[key])
            key_rev = (dest, origin)
            if key_rev in TRAIN_SCHEDULES:
                return f"Train schedule {dest.title()} → {origin.title()}:\n- " + "\n- ".join(TRAIN_SCHEDULES[key_rev])
            return f"I don't have schedules for {origin.title()} → {dest.title()}."
        if ctx.get("last_city") and len(cities) == 1:
            origin = ctx.get("last_city").lower()
            dest = cities[0].lower()
            key = (origin, dest)
            if key in TRAIN_SCHEDULES:
                return f"Train schedule {origin.title()} → {dest.title()}:\n- " + "\n- ".join(TRAIN_SCHEDULES[key])
        return "Tell me origin and destination (e.g., 'Mangalore to Bangalore')"
    if intent == "bus_timing":
        if len(cities) >= 2:
            origin, dest = cities[0].lower(), cities[1].lower()
            key = (origin, dest)
            if key in BUS_SCHEDULES:
                return f"Bus schedule {origin.title()} → {dest.title()}:\n" + str(BUS_SCHEDULES[key])
            key_rev = (dest, origin)
            if key_rev in BUS_SCHEDULES:
                return f"Bus schedule {dest.title()} → {origin.title()}:\n" + str(BUS_SCHEDULES[key_rev])
            return f"I don't have bus schedules for {origin.title()} → {dest.title()}."
        if ctx.get("last_city") and len(cities) == 1:
            origin = ctx.get("last_city").lower()
            dest = cities[0].lower()
            key = (origin, dest)
            if key in BUS_SCHEDULES:
                return f"Bus schedule {origin.title()} → {dest.title()}:\n" + str(BUS_SCHEDULES[key])
        return "Tell me origin and destination (e.g., 'Mangalore to Udupi')"
    if intent == "goodbye":
        return "Goodbye! Safe travels!"
    if intent == "thanks":
        return "You're welcome! Happy travels!"
    return "Sorry, I didn't understand. Can you rephrase?"

# -----------------------
# Utility: consistent response with both reply + card
# -----------------------
def respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, card):
    reply_text = build_reply(predicted_intent, cities or ([ctx.get("last_city")] if ctx.get("last_city") else []), text, ctx)
    logger.info(
        "RESPOND card: text=%s intent=%s conf=%s cities=%s card_key=%s",
        text, predicted_intent, intent_proba, cities, (card.get("key") if card else None)
    )
    return jsonify({
        "message": text,
        "intent": predicted_intent,
        "confidence": intent_proba,
        "entities": {"cities": cities},
        "reply": reply_text,
        "card": card
    })

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET"])
def index():
    """
    Serve the chat UI index. Flask static folder is src/static by default here.
    """
    try:
        base = Path(__file__).resolve().parent
        return send_from_directory(base / "static", "index.html")
    except Exception as e:
        logger.exception("Failed to serve index.html: %s", e)
        return "VoyageHelper backend running. UI not found.", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Debugging endpoint: POST { "text": "..." } -> returns predicted intent and probabilities.
    Useful when tuning ENTITY_FIRST_CONF and classifier behavior.
    """
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip().lower()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        pred = model.predict([text])[0]
        proba = None
        classes = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                proba = dict(zip(classes, [float(x) for x in probs]))
            else:
                proba = [float(x) for x in probs]
        return jsonify({"text": text, "predicted": pred, "proba": proba, "classes": classes})
    except Exception as e:
        logger.exception("predict error: %s", e)
        return jsonify({"error": str(e)}), 500

# -----------------------
# Updated /chat handler (improved place follow-ups)
# -----------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "No message provided"}), 400

    key = ctx_key()
    ctx = contexts.get(key, {"last_intent": None, "last_city": None})

    # --- Immediate chip handlers (override classifier for UI chip clicks) ---
    # These run BEFORE intent prediction so UI chips always get a response.
    text_lower = text.lower()

    def pick_city_from_text_or_ctx():
        cities_local = extract_cities(text)
        if cities_local:
            return cities_local[0]
        return ctx.get("last_city")

    # Hotels immediate patterns
    hotels_patterns = [
        "hotels in ", "show hotels", "find hotels", "hotel listings", "find a hotel", "show me hotels", "list hotels"
    ]
    if any(p in text_lower for p in hotels_patterns) or text_lower.startswith("hotels "):
        place = pick_city_from_text_or_ctx()
        if not place:
            return jsonify({
                "message": text,
                "intent": "hotels",
                "confidence": 1.0,
                "entities": {"cities": []},
                "reply": "Which city do you want hotels for? e.g., 'Hotels in Mangalore'.",
                "card": None
            })

        card = handle_hotel_query(place)
        ctx["last_intent"] = "hotels"
        ctx["last_city"] = place.lower()
        contexts[key] = ctx

        reply_text = f"Here are hotel results for {place.title()} — see the card for details."
        if card and card.get("help_urls"):
            reply_text += " I also included quick links (Google / Booking / TripAdvisor)."

        logger.info("Immediate hotels handler returned for %s", place)
        return jsonify({
            "message": text,
            "intent": "hotels",
            "confidence": 1.0,
            "entities": {"cities": [place]},
            "reply": reply_text,
            "card": card
        })

    # Best time immediate patterns
    best_time_patterns = ["best time", "when to visit", "when to go", "season to visit", "best months", "which month"]
    if any(p in text_lower for p in best_time_patterns) or text_lower.startswith("best time"):
        place = pick_city_from_text_or_ctx()
        if not place:
            return jsonify({
                "message": text,
                "intent": "best_time",
                "confidence": 1.0,
                "entities": {"cities": []},
                "reply": "Which city do you mean? e.g., 'Best time to visit Goa'.",
                "card": None
            })

        pk = place.lower()
        kb_card = build_place_card_from_kb(pk)
        if kb_card and kb_card.get("best_time"):
            kb_card["description"] = (kb_card.get("description") or "") + f"\n\nBest time to visit: {kb_card.get('best_time')}"
            ctx["last_intent"] = "best_time"
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Immediate best_time from KB for %s", pk)
            return jsonify({
                "message": text,
                "intent": "best_time",
                "confidence": 1.0,
                "entities": {"cities": [place]},
                "reply": f"Best time to visit {place.title()}: {kb_card.get('best_time')}",
                "card": kb_card
            })

        wiki_card = build_place_card_from_wiki(place)
        best_time_text = "Generally October — March is a comfortable season to visit."
        if wiki_card and wiki_card.get("description"):
            desc = wiki_card["description"].lower()
            if "monsoon" in desc or "rain" in desc:
                best_time_text = "Post-monsoon and winter months (October — March) are usually best — avoid the heavy monsoon months."
            elif "winter" in desc and ("cold" in desc or "cool" in desc):
                best_time_text = "Winter months (November — February) are often pleasant."
            elif "summer" in desc or "hot" in desc:
                best_time_text = "Avoid peak summer. Late autumn to early spring (Oct — Mar) is generally better."
            wiki_card["description"] = (wiki_card.get("description") or "") + f"\n\nBest time to visit: {best_time_text}"
            wiki_card["best_time"] = best_time_text
            ctx["last_intent"] = "best_time"
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Immediate best_time from Wiki for %s", pk)
            return jsonify({
                "message": text,
                "intent": "best_time",
                "confidence": 1.0,
                "entities": {"cities": [place]},
                "reply": f"Best time to visit {place.title()}: {best_time_text}",
                "card": wiki_card
            })

        # final fallback
        ctx["last_intent"] = "best_time"
        ctx["last_city"] = pk
        contexts[key] = ctx
        logger.info("Immediate best_time fallback for %s", pk)
        return jsonify({
            "message": text,
            "intent": "best_time",
            "confidence": 1.0,
            "entities": {"cities": [place]},
            "reply": f"Best time to visit {place.title()} is usually October — March.",
            "card": None
        })

    # How to reach immediate patterns
    how_patterns = ["how to reach", "how to get to", "how do i reach", "how to go to", "how to reach "]
    if any(p in text_lower for p in how_patterns):
        place = pick_city_from_text_or_ctx()
        if not place:
            return jsonify({
                "message": text,
                "intent": "how_to_reach",
                "confidence": 1.0,
                "entities": {"cities": []},
                "reply": "Which city do you want directions to? e.g., 'How to reach Udupi'.",
                "card": None
            })

        pk = place.lower()
        kb_card = build_place_card_from_kb(pk)
        if kb_card and kb_card.get("how_to_reach"):
            how_text = kb_card.get("how_to_reach")
            kb_card["description"] = (kb_card.get("description") or "") + f"\n\nHow to reach: {how_text}"
            ctx["last_intent"] = "how_to_reach"
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Immediate how_to_reach from KB for %s", pk)
            return jsonify({
                "message": text,
                "intent": "how_to_reach",
                "confidence": 1.0,
                "entities": {"cities": [place]},
                "reply": how_text,
                "card": kb_card
            })

        wiki_card = build_place_card_from_wiki(place)
        if wiki_card:
            how_text = f"Most visitors reach {place.title()} by air, train or road. Check local schedules for exact routes."
            wiki_card["description"] = (wiki_card.get("description") or "") + f"\n\nHow to reach: {how_text}"
            ctx["last_intent"] = "how_to_reach"
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Immediate how_to_reach from Wiki for %s", pk)
            return jsonify({
                "message": text,
                "intent": "how_to_reach",
                "confidence": 1.0,
                "entities": {"cities": [place]},
                "reply": how_text,
                "card": wiki_card
            })

        ctx["last_intent"] = "how_to_reach"
        ctx["last_city"] = pk
        contexts[key] = ctx
        logger.info("Immediate how_to_reach fallback for %s", pk)
        return jsonify({
            "message": text,
            "intent": "how_to_reach",
            "confidence": 1.0,
            "entities": {"cities": [place]},
            "reply": f"You can reach {place.title()} by air, rail, and road. Tell me which mode you prefer for detailed steps.",
            "card": None
        })

    # ---------- End immediate chip handlers ----------
    # Continue with normal intent prediction and flows

    # Predict intent
    try:
        predicted_intent = model.predict([text.lower()])[0]
    except Exception:
        logger.exception("Model predict failed for text: %s", text)
        return jsonify({"error": "Model prediction error"}), 500

    # Compute intent probability
    intent_proba = 1.0
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text.lower()])[0]
            if hasattr(model, "classes_"):
                try:
                    idx = list(model.classes_).index(predicted_intent)
                    intent_proba = float(probs[idx])
                except ValueError:
                    intent_proba = float(max(probs))
            else:
                intent_proba = float(max(probs))
        else:
            intent_proba = 1.0
    except Exception:
        logger.exception("predict_proba failed, defaulting probability to 1.0")

    cities = extract_cities(text)
    logger.info("msg=%s intent=%s conf=%s entities=%s ctx_last=%s", text, predicted_intent, intent_proba, cities, ctx.get("last_city"))

    # Helper: pick current city (explicit entity or last city)
    def pick_city():
        return (cities[0] if cities else (ctx.get("last_city") if ctx.get("last_city") else None))

    # QUICK: if user explicitly asks for hotels using keywords, keep old behavior
    if any(k in text.lower() for k in ["hotel", "hotels", "stay", "accommodation", "lodging", "lodg", "resort", "inn", "hostel"]):
        place_key = pick_city()
        if place_key:
            card = handle_hotel_query(place_key)
            if card:
                ctx["last_intent"] = predicted_intent
                ctx["last_city"] = place_key.lower()
                contexts[key] = ctx
                logger.info("Returning hotel card for %s (hotel-specific flow)", place_key)
                return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, card)
        else:
            reply = "Which city are you looking to find hotels for? e.g., 'Hotels in Mangalore'."
            logger.info("hotel query without city for msg=%s", text)
            return jsonify({
                "message": text, "intent": predicted_intent, "confidence": intent_proba,
                "entities": {"cities": cities}, "reply": reply, "card": None
            })

    # EXPLICIT INTENT HANDLING: hotels & best_time (don't rely on keyword-only)
    # 1) hotels intent
    if predicted_intent == "hotels":
        place_key = pick_city()
        if place_key:
            card = handle_hotel_query(place_key)
            if card:
                ctx["last_intent"] = predicted_intent
                ctx["last_city"] = place_key.lower()
                contexts[key] = ctx
                logger.info("Returning hotel card for %s (intent-based)", place_key)
                return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, card)
        else:
            reply = "Which city do you want hotels for? e.g., 'Hotels in Udupi'."
            return jsonify({
                "message": text, "intent": predicted_intent, "confidence": intent_proba,
                "entities": {"cities": cities}, "reply": reply, "card": None
            })

    # 2) best_time intent
    if predicted_intent in ("best_time", "season", "when_to_visit"):
        place_key = pick_city()
        if not place_key:
            return jsonify({
                "message": text, "intent": predicted_intent, "confidence": intent_proba,
                "entities": {"cities": cities}, "reply": "Which city do you mean? e.g., 'Best time to visit Goa'.", "card": None
            })

        pk = place_key.lower()
        # Try KB first
        kb_card = build_place_card_from_kb(pk)
        if kb_card and kb_card.get("best_time"):
            # add a small card that highlights best_time for the place
            kb_card["description"] = (kb_card.get("description") or "") + f"\n\nBest time to visit: {kb_card.get('best_time')}"
            ctx["last_intent"] = predicted_intent
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Returning KB best_time for %s", pk)
            return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, kb_card)

        # Try wiki summary and heuristics
        wiki_card = build_place_card_from_wiki(place_key)
        best_time_text = None
        if wiki_card and wiki_card.get("description"):
            desc = (wiki_card.get("description") or "").lower()
            # heuristics
            if "monsoon" in desc or "rain" in desc or "wet" in desc:
                best_time_text = "Post-monsoon and winter months (October — March) are usually best to visit — avoid the heavy monsoon months."
            elif "winter" in desc and ("cold" in desc or "cool" in desc):
                best_time_text = "Winter months (November — February) are often pleasant."
            elif "summer" in desc or "hot" in desc:
                best_time_text = "Avoid peak summer. Late autumn to early spring (Oct — Mar) is generally better."
            elif "best time" in desc or "best months" in desc:
                # try to extract phrase around 'best'
                idx = desc.find("best")
                snippet = desc[idx: idx + 200] if idx >= 0 else desc[:200]
                best_time_text = snippet.strip().capitalize()
            else:
                # fallback default recommendation
                best_time_text = "Generally October — March is a comfortable period to visit most parts of India."
        else:
            best_time_text = "I don't have a precise seasonal note for that place; usually October — March is a safe recommendation."

        # build a small card containing best_time info + helpful external link
        wiki_card = wiki_card or {
            "type": "place_card",
            "key": pk + "_best_time",
            "title": place_key.title(),
            "description": "",
            "sights": [],
            "best_time": "",
            "how_to_reach": "",
            "hotels": [],
            "food": [],
            "image": "",
            "source": "wiki" if wiki_card else "unknown",
            "source_url": wiki_card.get("source_url") if wiki_card else None
        }
        wiki_card["description"] = (wiki_card.get("description") or "") + f"\n\nBest time to visit: {best_time_text}"
        wiki_card["best_time"] = best_time_text

        ctx["last_intent"] = predicted_intent
        ctx["last_city"] = pk
        contexts[key] = ctx
        logger.info("Returning best_time card for %s (heuristic)", pk)
        return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, wiki_card)

    # ENTITY-FIRST: only when model confidence is high (existing behavior)
    place_key = cities[0] if cities else None
    cache = load_cache()

    if place_key and intent_proba >= ENTITY_FIRST_CONF:
        pk = place_key.lower()

        # KB card preferred
        card = build_place_card_from_kb(pk)
        if card:
            ctx["last_intent"] = predicted_intent
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Returning local KB card for %s (entity-first, high confidence)", pk)
            return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, card)

        # cached card
        if pk in cache:
            try:
                card = cache[pk]
                ctx["last_intent"] = predicted_intent
                ctx["last_city"] = pk
                contexts[key] = ctx
                logger.info("Returning cached card for %s (entity-first, high confidence)", pk)
                return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, card)
            except Exception:
                logger.debug("Failed to use cached card for %s", pk)

        # wiki fallback
        wiki_card = build_place_card_from_wiki(place_key)
        if wiki_card:
            try:
                kb_hotels = get_kb_hotels_for_city(pk) or []
            except Exception:
                kb_hotels = []
            if kb_hotels:
                wiki_card["hotels"] = kb_hotels
            cache[pk] = wiki_card
            save_cache(cache)
            ctx["last_intent"] = predicted_intent
            ctx["last_city"] = pk
            contexts[key] = ctx
            logger.info("Returning wiki card for %s (entity-first, high confidence)", pk)
            return respond_with_card_and_reply(text, predicted_intent, intent_proba, cities, ctx, wiki_card)

    # Low confidence fallback
    if intent_proba < CONFIDENCE_THRESHOLD:
        reply_low = "I'm not sure what you mean. Can you rephrase or try a different question (places, timings, booking)?"
        logger.info("low_confidence reply for msg=%s conf=%s", text, intent_proba)
        return jsonify({
            "message": text, "intent": "unknown", "confidence": intent_proba,
            "entities": {"cities": cities}, "reply": reply_low, "card": None
        })

    # Intent-first / normal handling for other intents (preserve existing behavior)
    if cities:
        ctx["last_city"] = cities[0].lower()

    reply_text = build_reply(predicted_intent, cities or ([ctx["last_city"]] if ctx.get("last_city") else []), text, ctx)
    ctx["last_intent"] = predicted_intent
    contexts[key] = ctx
    logger.info("reply=%s", reply_text)

    return jsonify({
        "message": text, "intent": predicted_intent, "confidence": intent_proba,
        "entities": {"cities": cities}, "reply": reply_text, "card": None
    })

if __name__ == "__main__":
    # Run on 127.0.0.1:5000 for local dev (same as frontend default)
    app.run(host="127.0.0.1", debug=True, port=5000)
