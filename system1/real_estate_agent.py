"""
LangGraph-based agentic real estate assistant for Apartments.com searches.

The agent:
* Converses with the user to refine preferences.
* Builds Apartments.com search URLs with `url_complex.ApartmentsSearchQuery`.
* Uses the provided Apify scraper wrapper (`scraper.py`) to fetch listings.
* Ranks and summarizes the top matches.
* Enriches listings with nearby points of interest via Google Maps.

Environment variables:
* OPENAI_API_KEY (required)
* GOOGLE_MAPS_API_KEY (optional but recommended)
* APIFY_API_TOKEN or APIFY_TOKEN (required when scraping)
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

import scraper
from url_complex import ApartmentsSearchQuery, Lifestyle, PetType, PropertyType

# Defaults can be overridden via env vars
load_dotenv()
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_PAGES = 1
DEFAULT_FILTER_OPTION = "all"

STATE_ABBREVIATIONS = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

DEFAULT_PERSONA_MODE = "auto"

PERSONAS = {
    "naturalist": {
        "name": "The Neighborhood Naturalist",
        "temperature": 0.9,
        "top_p": 1.0,
        "system": (
            "You are The Neighborhood Naturalist, a friendly local who knows every tree, coffee shop, and dog park in town. "
            "Blend sensory detail with honest market realities. Highlight vibe, walkability, green spaces, and everyday life details. "
            "Be warm, pragmatic, supportive, and realistic—never salesy."
        ),
    },
    "data": {
        "name": "The Data Whisperer",
        "temperature": 0.3,
        "top_p": 1.0,
        "system": (
            "You are The Data Whisperer, a concise, analytical, and reassuring rental guide. "
            "Lean into metrics (price per sq ft, comps, average rents, trends) and explain them plainly. "
            "Structure explanations, compare clearly, and demystify numbers without sounding robotic."
        ),
    },
    "deal": {
        "name": "The Deal Navigator",
        "temperature": 0.5,
        "top_p": 1.0,
        "system": (
            "You are The Deal Navigator, a confident but calm guide through trade-offs and negotiations. "
            "Highlight opportunities and risks without pressure. Frame options so the user feels in control. "
            "Discuss strategy: when to move fast, when to wait, and what to compromise on."
        ),
    },
}
class AgentState(TypedDict, total=False):
    """Shared conversational state passed between graph nodes."""

    messages: List[Dict[str, str]]
    preferences: Dict[str, Any]
    preferences_updated: bool
    persona_mode: str
    active_persona: str
    clarifying_questions: List[str]
    need_more_info: bool
    search_query: Optional[Dict[str, Any]]
    search_url: Optional[str]
    listings: List[Dict[str, Any]]
    enriched_listings: List[Dict[str, Any]]
    ranked_listings: List[Dict[str, Any]]
    reply: str


def get_openai_client() -> OpenAI:
    """Create an OpenAI client, raising a helpful error when the key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run the real estate agent.")
    return OpenAI()


def get_gmaps_client():
    """Return a Google Maps client if configured; otherwise None."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    try:
        import googlemaps  # type: ignore
    except ImportError:
        return None
    return googlemaps.Client(key=api_key)


def merge_preferences(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge preferences, keeping prior values when the new ones are None."""
    merged = dict(old or {})
    for key, value in (new or {}).items():
        if value is not None:
            merged[key] = value
    return merged


def _normalize_city_state(prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure city/state are populated, attempting to split a legacy `location` string.
    Does not override explicit city/state provided by the user.
    """
    merged = dict(prefs or {})
    city = merged.get("city")
    state = merged.get("state")
    location = merged.get("location")

    if (not city or not state) and location:
        parsed_city, parsed_state = _parse_city_state(location)
        if parsed_city and not city:
            merged["city"] = parsed_city
        if parsed_state and not state:
            merged["state"] = parsed_state
    if merged.get("state"):
        raw_state = str(merged["state"])
        abbr = _abbreviate_state(raw_state)
        if abbr:
            merged["state"] = abbr
        else:
            # Unknown state format; drop to force a clarifying question.
            cleaned = raw_state.strip()
            if cleaned and not (len(cleaned) == 2 and cleaned.isalpha()):
                merged.pop("state", None)
    return merged


def _parse_city_state(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort splitter for inputs like "Los Angeles, CA" or "Austin TX".
    Returns (city, state) when it can, otherwise (None, None).
    """
    if not text:
        return None, None
    cleaned = text.replace(",", " ").strip()
    parts = [p for p in cleaned.split() if p]
    if len(parts) < 2:
        return None, None
    state = parts[-1]
    city = " ".join(parts[:-1])
    return city or None, state or None


def _abbreviate_state(state: str) -> Optional[str]:
    """Convert full state names to 2-letter abbreviations; pass through existing abbreviations."""
    cleaned = state.strip().lower()
    if not cleaned:
        return None
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned.upper()
    return STATE_ABBREVIATIONS.get(cleaned)


PERSONA_ALIASES = {
    "1": "naturalist",
    "naturalist": "naturalist",
    "neighbor": "naturalist",
    "neighborhood": "naturalist",
    "local": "naturalist",
    "2": "data",
    "data": "data",
    "numbers": "data",
    "metrics": "data",
    "3": "deal",
    "deal": "deal",
    "navigator": "deal",
    "4": "auto",
    "auto": "auto",
}


def _normalize_persona_mode(raw: Optional[str]) -> str:
    """Normalize user input persona mode."""
    if not raw:
        return DEFAULT_PERSONA_MODE
    key = raw.strip().lower()
    return PERSONA_ALIASES.get(key, DEFAULT_PERSONA_MODE)


def _choose_auto_persona(user_text: str, prefs: Dict[str, Any]) -> str:
    """Heuristic to pick a persona when in auto mode."""
    text = (user_text or "").lower()
    # Prioritize negotiation/trade-off intent
    deal_keywords = ["negot", "offer", "trade", "compromise", "concession", "counter", "leverage"]
    data_keywords = ["price", "rent", "trend", "value", "deal", "good deal", "compare", "worth", "market", "comps", "per sq", "per sqft"]
    vibe_keywords = ["neighborhood", "vibe", "walk", "park", "coffee", "feel", "safe", "quiet", "lively", "schools", "beach"]

    if any(k in text for k in deal_keywords):
        return "deal"
    if any(k in text for k in data_keywords):
        return "data"
    if any(k in text for k in vibe_keywords):
        return "naturalist"
    # Fall back to naturalist for explorations, data for price-specific prompts
    if "budget" in text or "price" in text:
        return "data"
    return "naturalist"


def resolve_persona(mode: str, user_text: str, prefs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Determine the active persona based on mode and context.
    Returns (persona_key, persona_config).
    """
    normalized_mode = _normalize_persona_mode(mode)
    if normalized_mode != "auto" and normalized_mode in PERSONAS:
        key = normalized_mode
    else:
        key = _choose_auto_persona(user_text, prefs)
    return key, PERSONAS[key]

def compute_missing_preferences(prefs: Dict[str, Any]) -> List[str]:
    """
    Identify key missing preferences that block a search.
    Location (unless near_me) and at least one of price/beds/baths are expected.
    """
    missing: List[str] = []
    if not prefs.get("near_me"):
        city = prefs.get("city")
        state = prefs.get("state")
        if not city and not state:
            missing.append("city and state (e.g., Austin, TX)")
        elif not city:
            missing.append("city (e.g., Austin)")
        elif not state:
            missing.append(f"state for {city} (e.g., TX)")
    if not any(prefs.get(f) is not None for f in ["max_rent", "min_rent", "min_beds", "max_beds", "min_baths", "max_baths"]):
        missing.append("budget or bedroom/bathroom preferences")
    return missing


def normalize_enum(value: Optional[str], enum_cls):
    """Transform a string into an enum member when possible."""
    if value is None:
        return None
    try:
        return enum_cls(value.lower())
    except Exception:
        return None


def generate_persona_reply(
    state: AgentState,
    intent: str,
    listing_summaries: Optional[List[Dict[str, Any]]] = None,
    notes: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Create a persona-aware reply using the latest user message and optional listing context.
    Returns (reply_text, persona_key).
    """
    listing_summaries = listing_summaries or []
    prefs = state.get("preferences", {})
    messages = state.get("messages", [])
    latest_user_message = messages[-1]["content"] if messages else ""
    persona_key, persona_cfg = resolve_persona(
        state.get("persona_mode", DEFAULT_PERSONA_MODE), latest_user_message, prefs
    )

    safety_instructions = (
        "You are acting as a U.S. rental real estate agent for Apartments.com listings. "
        "Stay within the provided data (listings, preferences, maps outputs). "
        "Do NOT invent listing details; if something is unknown, say so briefly. "
        "Keep responses concise, friendly, and aligned with your persona style. "
        "Use tools only when necessary; otherwise rely on the provided context."
    )

    system_prompt = f"{persona_cfg['system']} {safety_instructions}"
    payload = {
        "intent": intent,
        "preferences": prefs,
        "clarifying_questions": state.get("clarifying_questions", []),
        "listings": listing_summaries,
        "recent_messages": messages[-6:],
        "latest_user_message": latest_user_message,
        "notes": notes,
    }

    client = get_openai_client()
    response = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=persona_cfg.get("temperature", 0.5),
        top_p=persona_cfg.get("top_p", 1.0),
    )
    reply = response.choices[0].message.content or "Let me know how you'd like to adjust the search."
    return reply, persona_key


def build_query_from_preferences(prefs: Dict[str, Any]) -> ApartmentsSearchQuery:
    """Create an ApartmentsSearchQuery instance from user preferences."""
    normalized = _normalize_city_state(prefs)
    query = ApartmentsSearchQuery()
    query.city = normalized.get("city")
    query.state = normalized.get("state")
    query.near_me = bool(normalized.get("near_me"))
    query.property_type = normalize_enum(normalized.get("property_type"), PropertyType)
    query.lifestyle = normalize_enum(normalized.get("lifestyle"), Lifestyle)
    query.rooms_for_rent = bool(normalized.get("rooms_for_rent", False))
    query.min_beds = _safe_int(normalized.get("min_beds"))
    query.max_beds = _safe_int(normalized.get("max_beds"))
    query.min_baths = _safe_float(normalized.get("min_baths"))
    query.max_baths = _safe_float(normalized.get("max_baths"))
    query.min_rent = _safe_int(normalized.get("min_rent"))
    query.max_rent = _safe_int(normalized.get("max_rent"))
    query.pet_friendly = bool(normalized.get("pet_friendly", False))
    query.pet_type = normalize_enum(normalized.get("pet_type"), PetType)
    query.cheap_only = bool(normalized.get("cheap_only", False))
    query.utilities_included = bool(normalized.get("utilities_included", False))
    query.amenity_slugs = normalized.get("amenity_slugs") or []
    query.keyword = normalized.get("keyword")
    query.frbo_only = bool(normalized.get("frbo_only", False))
    query.page = _safe_int(normalized.get("page"), default=1) or 1
    return query


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


def analyze_preferences(state: AgentState) -> AgentState:
    """LLM node: extract structured preferences from the latest user message."""
    client = get_openai_client()
    existing = state.get("preferences", {})
    user_message = state["messages"][-1]["content"]

    system_prompt = (
        "You are a friendly U.S. rental real estate agent. "
        "Extract the user's preferences for Apartments.com searches. "
        "Only include fields that are clearly implied. "
        "Use JSON with a top-level 'preferences' object and optional 'clarifying_questions' list."
        "Preferences keys allowed: city, state, location, near_me, property_type, lifestyle, rooms_for_rent, "
        "min_rent, max_rent, min_beds, max_beds, min_baths, max_baths, pet_friendly, pet_type, "
        "cheap_only, utilities_included, amenity_slugs, keyword, frbo_only, page, filter_option, max_pages."
    )

    extraction_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "existing_preferences": existing,
                    "message": user_message,
                },
                ensure_ascii=False,
            ),
        },
    ]

    response = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        messages=extraction_messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    new_prefs = parsed.get("preferences", parsed)
    merged = merge_preferences(existing, new_prefs)
    merged = _normalize_city_state(merged)
    preferences_updated = merged != existing
    missing = compute_missing_preferences(merged)
    clarifying = parsed.get("clarifying_questions", [])

    if missing and not clarifying:
        clarifying = [f"Could you share your {item}?" for item in missing]

    return {
        **state,
        "preferences": merged,
        "preferences_updated": preferences_updated,
        "clarifying_questions": clarifying,
        "need_more_info": bool(missing),
    }


def clarify(state: AgentState) -> AgentState:
    """Generate a simple clarification message based on outstanding questions."""
    questions = state.get("clarifying_questions") or []
    notes = None
    if not questions:
        notes = "Encourage the user to share city, state, budget, beds, and baths before searching."
    reply, persona_key = generate_persona_reply(state, intent="clarify", listing_summaries=[], notes=notes)
    return {**state, "reply": reply, "active_persona": persona_key}


def build_query_node(state: AgentState) -> AgentState:
    """Build the Apartments.com search URL from preferences."""
    prefs = state.get("preferences", {})
    # Avoid rebuilding if nothing changed and we already have a search URL.
    if state.get("search_url") and not state.get("preferences_updated"):
        return state
    try:
        query = build_query_from_preferences(prefs)
        search_url = query.build_url()
        next_state = {
            **state,
            "search_query": asdict(query),
            "search_url": search_url,
            "need_more_info": False,
        }
    except Exception as exc:  # URL creation failed
        next_state = {
            **state,
            "need_more_info": True,
            "clarifying_questions": [f"To build the search URL, I need a valid city/state. Details: {exc}"],
        }
    return next_state


def scrape_listings(state: AgentState) -> AgentState:
    """Call the scraper actor to fetch listings from Apartments.com."""
    search_url = state.get("search_url")
    if state.get("listings") and not state.get("preferences_updated"):
        return state
    prefs = state.get("preferences", {})
    if not search_url:
        reply, persona_key = generate_persona_reply(
            state,
            intent="error",
            notes="No search URL available; ask the user to verify city and state.",
        )
        return {**state, "reply": reply, "active_persona": persona_key}

    run_input = {
        "search_url": search_url,
        "max_pages": _safe_int(prefs.get("max_pages"), DEFAULT_MAX_PAGES) or DEFAULT_MAX_PAGES,
        "filter_option": prefs.get("filter_option") or DEFAULT_FILTER_OPTION,
    }

    try:
        items = scraper.fetch_listings_from_actor(run_input)
    except Exception as exc:
        reply, persona_key = generate_persona_reply(
            state,
            intent="error",
            notes=f"Scraper failed: {exc}. Apologize briefly and ask to confirm preferences or try again.",
        )
        return {**state, "reply": reply, "listings": [], "active_persona": persona_key}

    if not items:
        reply, persona_key = generate_persona_reply(
            state,
            intent="no_results",
            notes="No listings were found; ask to adjust location or price range.",
        )
        return {**state, "reply": reply, "listings": [], "active_persona": persona_key}

    return {**state, "listings": items}


def enrich_with_maps(state: AgentState) -> AgentState:
    """Augment listings with nearby points of interest via Google Maps."""
    gmaps_client = get_gmaps_client()
    listings = state.get("listings", []) or []
    if not gmaps_client:
        return {**state, "enriched_listings": listings}

    enriched: List[Dict[str, Any]] = []
    for listing in listings:
        location_text = (
            listing.get("about", {}).get("location")
            or listing.get("about", {}).get("title")
            or listing.get("url")
        )
        lat_lng = _geocode_address(gmaps_client, location_text)
        pois = _find_pois(gmaps_client, lat_lng) if lat_lng else []
        enriched.append({**listing, "nearby_pois": pois, "geocoded_location": lat_lng})
    return {**state, "enriched_listings": enriched}


def _geocode_address(gmaps_client, address: Optional[str]) -> Optional[Tuple[float, float]]:
    """Geocode an address string to (lat, lng)."""
    if not address:
        return None
    try:
        result = gmaps_client.geocode(address)
        if result and "geometry" in result[0]:
            loc = result[0]["geometry"]["location"]
            return loc.get("lat"), loc.get("lng")
    except Exception:
        return None
    return None


def _find_pois(gmaps_client, lat_lng: Optional[Tuple[float, float]]) -> List[str]:
    """Find nearby points of interest for the given coordinates."""
    if not lat_lng:
        return []
    lat, lng = lat_lng
    categories = [
        ("gyms", "gym"),
        ("schools", "school"),
        ("universities", "university"),
        ("parks", "park"),
        ("shopping", "shopping_mall"),
        ("hospitals", "hospital"),
        ("beaches", "beach"),
    ]
    poi_summaries: List[str] = []
    for label, place_type in categories:
        try:
            results = gmaps_client.places_nearby(location=(lat, lng), radius=2000, type=place_type)
            names = [p.get("name") for p in results.get("results", [])[:2] if p.get("name")]
            if names:
                poi_summaries.append(f"Nearby {label}: {', '.join(names)}")
        except Exception:
            continue
    return poi_summaries


def rank_and_format(state: AgentState) -> AgentState:
    """Score listings, select the top 3, and craft the user-facing reply."""
    prefs = state.get("preferences", {})
    listings = state.get("enriched_listings") or state.get("listings") or []
    ranked = sorted(listings, key=lambda item: _score_listing(item, prefs), reverse=True)[:5]
    if not ranked:
        reply, persona_key = generate_persona_reply(
            state,
            intent="no_results",
            listing_summaries=[],
            notes="No listings were found; ask to adjust location, price, or beds/baths.",
        )
        return {**state, "reply": reply, "active_persona": persona_key, "preferences_updated": False}

    listing_summaries: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked):
        view = _listing_prompt_view(item)
        view["rank"] = idx + 1
        view["score"] = _score_listing(item, prefs)
        view["why_match"] = _reason_tags(item, prefs, view.get("amenities", []))
        listing_summaries.append(view)

    notes = "Present the top matches in a natural, persona-aligned tone. Offer to refine price, beds/baths, or area."
    reply, persona_key = generate_persona_reply(
        state, intent="results", listing_summaries=listing_summaries, notes=notes
    )
    return {
        **state,
        "reply": reply,
        "ranked_listings": ranked,
        "active_persona": persona_key,
        "preferences_updated": False,
    }


def _score_listing(listing: Dict[str, Any], prefs: Dict[str, Any]) -> float:
    """Simple heuristic scoring based on price and bed/bath alignment."""
    price_min, price_max = _extract_price_range(listing)
    min_budget = prefs.get("min_rent")
    max_budget = prefs.get("max_rent")
    min_beds, max_beds = prefs.get("min_beds"), prefs.get("max_beds")

    score = 0.0
    if max_budget and price_min and price_min <= max_budget:
        score += 3
    if min_budget and price_max and price_max >= min_budget:
        score += 2
    beds, baths = _extract_beds_baths(listing)
    if min_beds and beds and beds >= min_beds:
        score += 2
    if max_beds and beds and beds <= max_beds:
        score += 1
    if prefs.get("pet_friendly"):
        if "pet" in json.dumps(listing).lower():
            score += 0.5
    return score


def _extract_price_range(listing: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Parse a price range from the scraper's pricing section."""
    pricing = listing.get("pricingAndFloorPlans") or []
    prices: List[int] = []
    for plan in pricing:
        rent_label = plan.get("rent_label") or ""
        prices.extend(int(p) for p in re.findall(r"\$?(\d[\d,]*)", rent_label.replace(",", "")))
        for unit in plan.get("units") or []:
            if unit.get("price"):
                prices.extend(int(p) for p in re.findall(r"\d+", unit["price"].replace(",", "")))
    if not prices:
        return None, None
    return min(prices), max(prices)


def _extract_beds_baths(listing: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Parse bed/bath counts from the details section."""
    pricing = listing.get("pricingAndFloorPlans") or []
    beds = baths = None
    for plan in pricing:
        for detail in plan.get("details") or []:
            if "bed" in detail.lower():
                match = re.search(r"(\d+)", detail)
                if match:
                    beds = max(beds or 0, float(match.group(1)))
            if "bath" in detail.lower():
                match = re.search(r"(\d+(\.\d+)?)", detail)
                if match:
                    baths = max(baths or 0, float(match.group(1)))
    return beds, baths


def _listing_prompt_view(listing: Dict[str, Any]) -> Dict[str, Any]:
    """Compact listing view for LLM context."""
    about = listing.get("about") or {}
    price_min, price_max = _extract_price_range(listing)
    beds, baths = _extract_beds_baths(listing)
    amenities = listing.get("amenities") or []
    amenity_titles: List[str] = []
    for item in amenities:
        amenity_titles.extend(item.get("amenities", [])[:3])
    return {
        "title": about.get("title") or "Listing",
        "location": about.get("location") or "",
        "price_min": price_min,
        "price_max": price_max,
        "beds": beds,
        "baths": baths,
        "nearby_pois": listing.get("nearby_pois") or [],
        "features": (about.get("Unique Features") or [])[:5],
        "amenities": amenity_titles[:6],
        "url": listing.get("url"),
    }


def _reason_tags(listing: Dict[str, Any], prefs: Dict[str, Any], amenity_titles: List[str]) -> List[str]:
    """Build short reason tags explaining fit."""
    reasons: List[str] = []
    price_min, price_max = _extract_price_range(listing)
    min_budget, max_budget = prefs.get("min_rent"), prefs.get("max_rent")
    if max_budget and price_min and price_min <= max_budget:
        reasons.append("fits your budget ceiling")
    if min_budget and price_max and price_max >= min_budget:
        reasons.append("in your budget range")
    beds, baths = _extract_beds_baths(listing)
    min_beds, max_beds = prefs.get("min_beds"), prefs.get("max_beds")
    if min_beds and beds and beds >= min_beds:
        reasons.append("meets your bedroom need")
    if max_beds and beds and beds <= max_beds:
        reasons.append("not oversized on bedrooms")
    if prefs.get("pet_friendly") and "pet" in json.dumps(listing).lower():
        reasons.append("pet-friendly potential")
    if amenity_titles:
        reasons.append(f"has amenities like {amenity_titles[0]}")
    return reasons


def _format_listing(listing: Dict[str, Any], prefs: Dict[str, Any], rank: int) -> str:
    """Create a human-friendly summary for one listing."""
    about = listing.get("about") or {}
    title = about.get("title") or "Listing"
    location = about.get("location") or "Address not provided"
    price_min, price_max = _extract_price_range(listing)
    beds, baths = _extract_beds_baths(listing)
    price_str = (
        f"${price_min:,} - ${price_max:,}" if price_min and price_max else "Price not listed"
    )
    pois = listing.get("nearby_pois") or []
    features = about.get("Unique Features") or []
    amenities = listing.get("amenities") or []

    amenity_titles = []
    for item in amenities:
        amenity_titles.extend(item.get("amenities", [])[:3])
    amenity_preview = ", ".join(amenity_titles[:5]) if amenity_titles else ""

    reasons = _reason_tags(listing, prefs, amenity_titles)
    reason_line = f"Good match because it {', '.join(reasons)}." if reasons else ""

    parts = [
        f"{rank}. {title}",
        f"   Location: {location}",
        f"   Price: {price_str}",
        f"   Bedrooms/Baths: {beds or 'N/A'} bd / {baths or 'N/A'} ba",
    ]
    if amenity_preview:
        parts.append(f"   Amenities: {amenity_preview}")
    if features:
        parts.append(f"   Notable features: {', '.join(features[:5])}")
    if pois:
        parts.append(f"   Nearby: {', '.join(pois[:3])}")
    if reason_line:
        parts.append(f"   Why it could work: {reason_line}")
    parts.append(f"   Link: {listing.get('url') or 'N/A'}")
    return "\n".join(parts)


def _persona_label(mode: str) -> str:
    key = _normalize_persona_mode(mode)
    if key in PERSONAS:
        return PERSONAS[key]["name"]
    return "Auto"


def prompt_persona_choice() -> str:
    """CLI helper to pick a persona mode."""
    print("Choose a persona:")
    print("  1: Neighborhood Naturalist")
    print("  2: Data Whisperer")
    print("  3: Deal Navigator")
    print("  4: Auto (let me choose based on the conversation)")
    choice = input("Enter 1-4 (default auto): ").strip()
    return _normalize_persona_mode(choice or DEFAULT_PERSONA_MODE)


def handle_persona_command(state: AgentState, user_input: str) -> Optional[str]:
    """Switch personas mid-conversation if the user issues a /persona command."""
    if not user_input.lower().startswith("/persona"):
        return None
    parts = user_input.split()
    if len(parts) < 2:
        return "Please specify a persona: naturalist, data, deal, or auto."
    new_mode = _normalize_persona_mode(parts[1])
    state["persona_mode"] = new_mode
    label = _persona_label(new_mode)
    ack = f"Switched persona to {label}."
    msgs = state.get("messages") or []
    msgs.append({"role": "user", "content": user_input})
    msgs.append({"role": "assistant", "content": ack})
    state["messages"] = msgs
    return ack


def answer_with_existing(state: AgentState) -> AgentState:
    """Use existing listings and history to answer follow-ups without re-scraping."""
    listings = state.get("enriched_listings") or state.get("listings") or []
    if not listings:
        reply, persona_key = generate_persona_reply(
            state,
            intent="no_cached",
            notes="No cached listings yet; ask for city/state/budget to start a search.",
        )
        return {**state, "reply": reply, "active_persona": persona_key}

    preview_listings = listings[:5]
    listing_summaries = [_listing_prompt_view(item) for item in preview_listings]
    notes = "Answer the user's follow-up using cached listings; compare briefly and be transparent about trade-offs."
    reply, persona_key = generate_persona_reply(
        state, intent="follow_up", listing_summaries=listing_summaries, notes=notes
    )
    return {**state, "reply": reply, "active_persona": persona_key}


def build_graph() -> StateGraph:
    """Construct the LangGraph workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("analyze_preferences", analyze_preferences)
    graph.add_node("clarify", clarify)
    graph.add_node("answer_existing", answer_with_existing)
    graph.add_node("build_query", build_query_node)
    graph.add_node("scrape", scrape_listings)
    graph.add_node("enrich", enrich_with_maps)
    graph.add_node("rank_and_format", rank_and_format)

    graph.set_entry_point("analyze_preferences")

    graph.add_conditional_edges(
        "analyze_preferences",
        lambda s: "clarify"
        if s.get("need_more_info")
        else "answer_existing"
        if s.get("listings") and not s.get("preferences_updated")
        else "build_query",
        {"clarify": "clarify", "build_query": "build_query", "answer_existing": "answer_existing"},
    )

    graph.add_conditional_edges(
        "build_query",
        lambda s: "clarify" if s.get("need_more_info") else "scrape",
        {"clarify": "clarify", "scrape": "scrape"},
    )

    graph.add_conditional_edges(
        "scrape",
        lambda s: "enrich" if s.get("listings") else "end",
        {"enrich": "enrich", "end": END},
    )

    graph.add_edge("enrich", "rank_and_format")
    graph.add_edge("rank_and_format", END)
    graph.add_edge("clarify", END)
    graph.add_edge("answer_existing", END)

    return graph.compile()


def main() -> None:
    """Simple REPL interface."""
    try:
        get_openai_client()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    persona_mode = prompt_persona_choice()
    print(f"Great. Persona set to {_persona_label(persona_mode)}.")
    print("Hi! I'm your Apartments.com rental helper. Tell me the city, state, budget, beds/baths, and must-haves.")
    state: AgentState = {"messages": [], "preferences": {}, "persona_mode": persona_mode, "preferences_updated": False}
    app = build_graph()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        command_reply = handle_persona_command(state, user_input)
        if command_reply:
            print(f"Agent: {command_reply}\n")
            continue

        state_messages = state.get("messages") or []
        state_messages.append({"role": "user", "content": user_input})
        state["messages"] = state_messages

        state = app.invoke(state)
        reply = state.get("reply") or "I didn't catch that—could you rephrase?"
        print(f"Agent: {reply}\n")

        # Maintain conversation memory
        state["messages"].append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
