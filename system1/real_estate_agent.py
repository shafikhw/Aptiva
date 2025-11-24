"""
LangGraph-based agentic real estate assistant for Apartments.com searches.

The agent:
* Converses with the user to refine preferences.
* Builds Apartments.com search URLs with `url_complex.ApartmentsSearchQuery`.
* Uses the provided Apify scraper wrapper (`scraper.py`) to fetch listings.
* Ranks and summarizes the top matches.
* Enriches listings with nearby points of interest via Google Maps.
* Generates draft lease agreements with compliance checks on demand.

Environment variables:
* OPENAI_API_KEY (required)
* GOOGLE_MAPS_API_KEY (optional but recommended)
* APIFY_API_TOKEN or APIFY_TOKEN (required when scraping)
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

try:
    from . import lease_drafter, scraper
    from .url_complex import ApartmentsSearchQuery, Lifestyle, PetType, PropertyType
except ImportError:
    import lease_drafter  # type: ignore
    import scraper  # type: ignore
    from url_complex import ApartmentsSearchQuery, Lifestyle, PetType, PropertyType  # type: ignore

# Defaults can be overridden via env vars
load_dotenv()
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_PAGES = 1
DEFAULT_FILTER_OPTION = "all"
VALID_FILTER_OPTIONS = {"all", "bed0", "bed1", "bed2", "bed3", "bed4", "bed5"}
SCRAPER_OUTPUT_PATH = "actor_output.json"
SCRAPER_OUTPUT_FILE = Path(__file__).resolve().parent / SCRAPER_OUTPUT_PATH
THUMBNAIL_WIDTH_PX = 280

DOMAIN_GUARD_SYSTEM_MESSAGE = (
    "You are a specialized US residential real estate rental agent. "
    "You only discuss topics directly related to finding and comparing rental properties in the United States; "
    "property details, neighborhoods, and local amenities; scheduling viewings and tours; "
    "lease terms, applications, approvals, move-in logistics, and follow-up questions about scraped listings. "
    "If the user asks for anything outside this scope (general coding, medical advice, politics, personal counseling, etc.), "
    "politely refuse, explain that you only handle US residential rentals, and invite a housing-related question instead. "
    "Do not answer unrelated questions even if you know the answer, and use general knowledge only to support US rental decisions."
)

OFF_TOPIC_REFUSAL = (
    "I am designed only to help with US residential rentals and related topics like property matching, viewings, and leases. "
    "Please ask me something about finding or evaluating a place to live in the United States."
)

REAL_ESTATE_KEYWORDS = [

    # Core rental vocabulary
    "rent", "rental", "renting", "for rent", "rented",
    "apartment", "apartments", "apt",
    "condo", "condominium",
    "house", "single family home", "single-family",
    "townhouse", "townhome", "rowhouse",
    "loft", "duplex", "triplex", "quadplex", "multi family",
    "multifamily", "property", "unit", "listing", "complex",
    "building", "community", "residence", "residential",

    # Listing attributes
    "bedroom", "bedrooms", "bathroom", "bathrooms",
    "bed", "bath", "beds", "baths", "1br", "2br", "3br", "4br",
    "bd", "ba", "studio", "efficiency unit",
    "floor plan", "layout", "sqft", "square feet", "square ft",
    "flooring", "hardwood floors", "laminate", "vinyl floors",
    "tile floors", "walk in closet", "walk-in closet",
    "open concept", "open layout", "kitchen island",
    "granite countertops", "stainless steel appliances",
    "updated kitchen", "renovated", "newly renovated",

    # Amenities
    "amenities", "pool", "heated pool", "gym", "fitness center",
    "sauna", "steam room", "hot tub", "spa",
    "laundry", "in unit laundry", "in-unit laundry",
    "washer", "dryer", "washer dryer", "laundry room",
    "parking", "garage", "assigned parking", "covered parking",
    "carport", "street parking", "off street parking",
    "elevator", "doorman", "24 hour concierge",
    "door staff", "concierge",
    "balcony", "patio", "yard", "terrace", "porch",
    "rooftop", "rooftop deck", "roof deck",
    "storage", "bike storage", "secure package room",
    "package locker", "mail room", "clubhouse",
    "business center", "coworking space", "coworking",
    "high speed internet", "fiber internet", "wifi included",
    "hvac", "central air", "air conditioning", "heating",
    "fireplace", "ceiling fan", "soundproofing",

    # Pet terminology
    "pet friendly", "pet-friendly", "dog friendly", "cat friendly",
    "pet policy", "dog", "cat", "pets allowed", "pet deposit",
    "breed restrictions", "weight limit",

    # Location / walkability / neighborhood context
    "neighborhood", "walkability", "walk score",
    "bike score", "transit score",
    "commute", "transit", "subway", "metro", "train", "bus",
    "transportation", "bike lane", "bikeable",
    "downtown", "uptown", "midtown",
    "school district", "zoning",
    "safe neighborhood", "quiet neighborhood",
    "crime rate", "low crime", "low noise", "noise level",

    # Viewing / touring
    "viewing", "tour", "schedule a tour", "book a tour",
    "open house", "openhouse", "virtual tour",
    "in person tour", "self guided tour", "self-guided",

    # Leasing / applications
    "application", "apply", "apply now",
    "rental application", "tenant application",
    "credit check", "background check", "criminal check",
    "guarantor", "guardian", "cosigner",
    "pre approval", "pre-approval", "approval process",
    "application fee", "processing fee",
    "income requirement", "income verification",
    "employment verification", "proof of income",
    "pay stubs", "w2", "tax return", "bank statements",

    # Fees & financial terms
    "deposit", "security deposit", "holding deposit",
    "first month rent", "last month rent",
    "move in", "move-in", "move out", "move-out",
    "renters insurance", "utilities included",
    "furnished", "unfurnished",
    "broker fee", "no fee", "one month free",
    "concession", "special offer", "promo",
    "late fee", "rent increase", "rent control",
    "stabilized", "market rent",

    # Rental contract / agreement terminology
    "lease agreement", "lease contract", "rental agreement",
    "lease terms", "sublease", "sublet",
    "lease takeover", "lease break",
    "early termination", "termination fee",
    "month to month", "month-to-month",
    "short term lease", "long term lease",
    "renewal", "lease renewal",

    # Maintenance & building operations
    "maintenance", "maintenance request", "work order",
    "onsite manager", "property manager",
    "property management", "landlord", "tenant", "tenancy",
    "24 hour maintenance", "emergency maintenance",
    "pest control", "trash pickup", "recycling",

    # Special housing categories
    "rooms for rent", "roommate", "shared housing",
    "student housing", "off campus housing",
    "corporate housing", "section 8", "voucher",
    "affordable housing", "low income", "income restricted",
    "senior housing", "55+", "age restricted",

    # Searching / filtering on sites like Apartments.com
    "apartments.com", "zillow", "trulia", "hotpads",
    "search url", "listing url", "availability",
    "units available", "vacancy", "vacant",
    "move in date", "available now", "available soon",

    # Surrounding points of interest (POI)
    "near", "close to", "walking distance", "minutes away",
    "grocery store", "supermarket", "trader joes",
    "whole foods", "schools", "parks", "playground",
    "coffee shop", "cafe", "mall", "shopping center",
    "restaurants", "hospital", "clinic",
    "university", "campus",

    # Maps + geolocation
    "google maps", "distance", "miles", "mile", "minutes",
    "commute time", "directions", "route",
    "geocode", "address lookup", "zip code", "zipcode",

    # Relocation & moving logistics
    "relocate", "relocating", "moving", "moving truck",
    "moving service", "move assistance", "storage unit",
    "pods", "uhaul", "u-haul",

    # Rental market / pricing context
    "market trend", "market rate", "comps", "comparable",
    "price per sqft", "median rent", "average rent",
    "absorption rate", "inventory", "vacancy rate",
    "supply", "demand",

    # Legal terms (non-advice, but part of rental vocabulary)
    "eviction", "eviction record",
    "fair housing", "fair housing act",
    "tenant rights", "landlord responsibilities",
    "habitability", "rental license", "inspection",

    # General signals of rental intent
    "find a place", "find housing", "looking to rent",
    "searching for an apartment", "new place",
    "place to live", "need housing", "move to",
    "moving to", "housing options"
]

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
    scraped_listings: List[Dict[str, Any]]
    enriched_listings: List[Dict[str, Any]]
    ranked_listings: List[Dict[str, Any]]
    listing_summaries: List[Dict[str, Any]]
    listing_lookup: Dict[str, Dict[str, Any]]
    focused_listing: Dict[str, Any]
    reply: str
    reply_streamed: bool
    lease_drafts: List[Dict[str, Any]]
    pending_lease_choice: Optional[int]
    pending_lease_waiting_name: bool
    pending_lease_waiting_plan: bool
    pending_lease_plan_options: List[Dict[str, Any]]
    pending_lease_selected_plan: Dict[str, Any]
    pending_lease_waiting_unit: bool
    pending_lease_unit_options: List[Dict[str, Any]]
    pending_lease_selected_unit: Dict[str, Any]
    pending_lease_waiting_start: bool
    pending_lease_waiting_duration: bool
    pending_lease_duration_bounds: Tuple[Optional[int], Optional[int]]
    last_listing: Dict[str, Any]
    last_overrides: Dict[str, Any]
    last_duration_bounds: Tuple[Optional[int], Optional[int]]
    off_topic: bool


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


def is_real_estate_related(user_message: str) -> bool:
    """
    Return True if the message is about US residential real estate renting,
    property matching, listings, neighborhoods, amenities, viewing scheduling,
    leases, or related logistics. Return False otherwise.
    """
    if not user_message:
        return False
    text = user_message.lower()
    off_topic_phrases = [

        # Vehicle rentals / leasing
        "car rental", "rent a car", "rental car", "car hire",
        "truck rental", "rent a truck", "moving truck rental",
        "box truck rental", "pickup rental", "pickup truck rental",
        "van rental", "cargo van rental", "passenger van rental",
        "sprinter van rental", "minivan rental",
        "motorcycle rental", "scooter rental", "moped rental",
        "bike rental", "bicycle rental", "e bike rental", "e-bike rental",
        "ebike rental", "electric bike rental",
        "boat rental", "kayak rental", "canoe rental",
        "paddleboard rental", "paddle board rental",
        "jet ski rental", "jetski rental",
        "yacht rental", "sailboat rental",
        "rv rental", "camper rental", "camper van rental",
        "motorhome rental", "caravan rental",
        "atv rental", "utv rental", "side by side rental",
        "snowmobile rental", "golf cart rental",
        "limousine rental", "limo rental",

        # Transportation services / ride services
        "ride share", "rideshare", "uber", "lyft",
        "taxi rental", "taxi hire", "chauffeur service",
        "airport shuttle", "airport transfer", "car service",
        "black car service", "driver for hire",

        # Entertainment & media rentals
        "dvd rental", "movie rental", "rent a movie",
        "film rental", "video rental", "video store rental",
        "blockbuster rental", "redbox rental",
        "game rental", "video game rental",
        "console game rental", "blu ray rental", "bluray rental",

        # Party / event rentals
        "party rental", "event rental", "wedding rental",
        "wedding tent rental", "marquee rental",
        "tent rental", "canopy rental",
        "chair rental", "table rental",
        "linen rental", "tablecloth rental",
        "glassware rental", "china rental", "flatware rental",
        "decor rental", "wedding decor rental",
        "centerpiece rental",
        "bounce house rental", "bouncy castle rental",
        "inflatable rental", "inflatable slide rental",
        "photo booth rental", "photobooth rental",
        "stage rental", "platform rental", "runway rental",
        "lighting rental", "uplighting rental",
        "sound system rental", "pa system rental",
        "audiovisual rental", "av equipment rental",
        "pipe and drape rental",

        # Tool / equipment rentals (construction, DIY, industrial)
        "tool rental", "power tool rental",
        "generator rental", "portable generator rental",
        "pressure washer rental", "power washer rental",
        "ladder rental", "scaffold rental", "scaffolding rental",
        "chainsaw rental", "jackhammer rental",
        "demolition hammer rental",
        "excavator rental", "mini excavator rental",
        "bobcat rental", "skid steer rental",
        "forklift rental", "pallet jack rental",
        "tractor rental", "backhoe rental",
        "bulldozer rental", "crane rental", "boom lift rental",
        "scissor lift rental", "lift rental",
        "cement mixer rental", "concrete mixer rental",
        "dump trailer rental", "equipment trailer rental",
        "air compressor rental", "nailer rental",
        "tile saw rental", "floor sander rental",
        "carpet cleaner rental", "rug doctor rental",
        "heavy equipment rental", "construction rental",
        "industrial equipment rental",

        # Storage and non-residential spaces
        "storage unit rental", "rent storage unit",
        "self storage rental", "self storage",
        "mini storage rental", "storage locker rental",
        "storage container rental", "pod rental",
        "pods rental", "shipping container rental",
        "warehouse rental", "industrial storage rental",
        "locker rental", "gym locker rental",
        "boat slip rental", "dock rental", "marina slip rental",
        "hangar rental", "airplane hangar rental",

        # Hospitality / short stay travel (non long term housing)
        "hotel room", "book a hotel", "hotel booking",
        "motel room", "hostel bed", "hostel booking",
        "resort stay", "all inclusive resort",
        "nightly rental", "weekend rental",
        "airbnb", "vrbo", "vacation rental", "holiday rental",
        "holiday let", "short stay rental",
        "vacation cabin rental", "cabin rental",
        "beach house rental", "lake house rental",
        "ski cabin rental", "ski chalet rental",
        "bnb rental", "bed and breakfast booking",

        # Commercial / non-residential leases
        "commercial lease", "office lease", "office rental",
        "coworking space rental", "coworking membership",
        "shared office rental",
        "warehouse lease", "industrial lease",
        "retail lease", "shopping mall lease",
        "mall kiosk lease", "storefront lease",
        "restaurant space lease", "bar lease",
        "land lease", "ground lease",
        "billboard rental", "ad space rental",
        "advertising space rental", "sign rental",

        # Intellectual property / abstract "property"
        "intellectual property", "ip rights",
        "copyright", "patent", "trademark",
        "software license", "license key rental",
        "code property", "class property",
        "javascript property", "css property",
        "object property", "property in python",
        "property decorator",

        # Computing / server / cloud rentals
        "server rental", "rent a server",
        "game server rental", "minecraft server rental",
        "ark server rental",
        "vps rental", "virtual server rental",
        "cloud server rental", "dedicated server rental",
        "gpu server rental", "compute instance rental",
        "web hosting rental",

        # Finance / economics uses of "rent" or "lease"
        "rent seeking", "economic rent",
        "financial lease", "capital lease",
        "operating lease", "sale leaseback",
        "lease financing", "lease accounting",
        "lease liability", "ifrs lease", "asc 842 lease",
        "equipment finance", "fleet lease",
        "business lease", "corporate lease",

        # Social / novelty rentals
        "rent a friend", "rent a girlfriend", "rent a boyfriend",
        "rent a partner", "rent a family",
        "rent a maid", "rent a butler",
        "rent a chef", "rent a clown",
        "rent a santa", "rent an elf",
        "rent a crowd", "rent a fanbase",

        # Fashion / clothing / accessories rentals
        "clothing rental", "clothes rental",
        "fashion rental", "designer dress rental",
        "dress rental", "gown rental", "tuxedo rental",
        "suit rental", "costume rental",
        "cosplay costume rental",
        "handbag rental", "purse rental",
        "jewelry rental", "watch rental",

        # Animal / pet boarding or stalls
        "dog kennel rental", "dog boarding",
        "cat boarding", "pet boarding",
        "horse stable rental", "horse boarding",
        "stall rental", "kennel rental",

        # Sports / recreation equipment and space
        "ski rental", "snowboard rental",
        "golf cart rental", "golf club rental",
        "surfboard rental", "windsurf rental", "kiteboard rental",
        "climbing gear rental", "hiking gear rental",
        "camping gear rental", "tent gear rental",
        "kayak gear rental",
        "tennis court rental", "tennis court booking",
        "basketball court rental", "volleyball court rental",
        "pickleball court rental", "soccer field rental",
        "sports equipment rental", "bowling lane rental",
        "ice rink rental", "skate rental",

        # Medical / health equipment
        "hospital bed rental", "medical bed rental",
        "medical equipment rental", "durable medical equipment rental",
        "wheelchair rental", "mobility scooter rental",
        "walker rental", "crutches rental",
        "oxygen tank rental", "oxygen concentrator rental",
        "cpap rental", "hospital equipment rental",

        # Music / audio / production gear
        "instrument rental", "band instrument rental",
        "guitar rental", "piano rental", "keyboard rental",
        "drum rental", "violin rental", "cello rental",
        "amp rental", "amplifier rental",
        "speaker rental", "sound system rental",
        "dj equipment rental", "dj gear rental",
        "microphone rental", "mic rental",
        "recording equipment rental",

        # Film / photography production
        "camera rental", "lens rental", "dslr rental",
        "mirrorless camera rental", "cinema camera rental",
        "video camera rental", "gimbal rental",
        "tripod rental", "slider rental",
        "lighting kit rental", "light kit rental",
        "film equipment rental", "production gear rental",
        "boom mic rental", "lav mic rental",
        "steadicam rental", "drone rental",
        "studio rental", "photo studio rental",
        "soundstage rental",

        # Education / school related rentals
        "school locker rental", "band instrument rental",
        "lab equipment rental", "scientific equipment rental",

        # Business services / misc kiosks
        "vending machine rental", "atm rental",
        "arcade machine rental", "pinball rental",
        "claw machine rental", "photo kiosk rental",
        "3d printer rental", "laser cutter rental",

        # Agriculture / land equipment (non residential)
        "farm equipment rental", "tractor rental",
        "combine rental", "harvester rental",
        "plow rental", "baler rental",

        # Insurance / legal that clearly refer to non-housing situations
        "rental car insurance", "car rental insurance",
        "equipment rental insurance",
        "event rental insurance",

        # Overly abstract or metaphorical "tenant / landlord" uses
        "tenant of life", "landlord metaphor",
        "tenant metaphor", "intellectual tenant",

        # Misc strongly non-housing rentals
        "karaoke machine rental", "slot machine rental",
        "cot rental", "crib rental", "stroller rental",
        "snow cone machine rental", "cotton candy machine rental",
        "popcorn machine rental"
    ]
    if any(phrase in text for phrase in off_topic_phrases):
        return False
    if any(keyword in text for keyword in REAL_ESTATE_KEYWORDS):
        return True
    if re.search(r"\b\d+\s*(?:bed|bd|bath|ba|br|bdrm)s?\b", text):
        return True
    if re.search(r"\b\d{5}\b", text):
        return True
    if re.search(r"\b[a-z][a-z\s]+,\s*[a-z]{2}\b", text):
        return True
    if "move" in text and ("city" in text or "state" in text or "apartment" in text or "house" in text or "neighborhood" in text):
        return True
    return True


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


def _extract_chunk_text(chunk: Any) -> str:
    """Pull text content from a streaming delta, handling list/str shapes."""
    try:
        if not getattr(chunk, "choices", None):
            return ""
        delta = getattr(chunk.choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    text_val = getattr(item, "text", None)
                    if text_val:
                        parts.append(str(text_val))
            return "".join(parts)
        if isinstance(content, dict):
            return str(content.get("text", ""))
    except Exception:
        return ""
    return ""


def _inject_domain_guardrail(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure the domain restriction system message is always prepended."""
    guard = {"role": "system", "content": DOMAIN_GUARD_SYSTEM_MESSAGE}
    cleaned = [
        msg for msg in (messages or []) if not (msg.get("role") == "system" and msg.get("content") == DOMAIN_GUARD_SYSTEM_MESSAGE)
    ]
    return [guard, *cleaned]


def stream_chat_completion(
    client: OpenAI,
    *,
    stream_label: Optional[str] = None,
    stream_to_stdout: bool = False,
    **kwargs: Any,
) -> Tuple[str, bool]:
    """
    Dispatch a streaming chat completion.

    Returns a tuple of (full_text, emitted_to_stdout) where emitted_to_stdout indicates
    whether anything was printed in real time.
    """
    messages = kwargs.get("messages") or []
    kwargs["messages"] = _inject_domain_guardrail(list(messages))
    kwargs["stream"] = True  # guarantee streaming on every call
    content_parts: List[str] = []
    emitted_to_stdout = False
    label_printed = False

    try:
        response = client.chat.completions.create(**kwargs)
        for chunk in response:
            text = _extract_chunk_text(chunk)
            if not text:
                continue
            content_parts.append(text)
            if stream_to_stdout:
                if stream_label is not None and not label_printed:
                    print(stream_label, end="", flush=True)
                    label_printed = True
                print(text, end="", flush=True)
                emitted_to_stdout = True
        if stream_to_stdout and label_printed:
            print()  # newline after streamed tokens
    except Exception:
        had_content = bool(content_parts)
        if stream_to_stdout:
            if stream_label is not None and not label_printed:
                print(stream_label, end="", flush=True)
                label_printed = True
            notice = "\n[stream interrupted—partial reply above; you can ask me to continue]"
            print(notice, flush=True)
        fallback = "".join(content_parts)
        if not fallback:
            fallback = "Sorry, the response was interrupted. Could you ask again?"
        else:
            fallback += "\n\n(Streaming was interrupted; message may be partial.)"
        return fallback, emitted_to_stdout or had_content
    return "".join(content_parts), emitted_to_stdout


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
    focused_listing: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, bool]:
    """
    Create a persona-aware reply using the latest user message and optional listing context.
    Returns (reply_text, persona_key, streamed_to_stdout).
    """
    listing_summaries = listing_summaries or []
    prefs = state.get("preferences", {})
    messages = state.get("messages", [])
    latest_user_message = messages[-1]["content"] if messages else ""
    persona_key, persona_cfg = resolve_persona(
        state.get("persona_mode", DEFAULT_PERSONA_MODE), latest_user_message, prefs
    )

    render_views: List[Dict[str, Any]] = [dict(view) for view in listing_summaries]
    if not render_views and focused_listing:
        focus_view = _listing_prompt_view(focused_listing, prefs)
        focus_view["rank"] = 1
        focus_view["why_match"] = _reason_tags(focused_listing, prefs, focus_view.get("amenities", []))
        render_views = [focus_view]
        if not listing_summaries:
            listing_summaries = render_views
    for idx, view in enumerate(render_views, start=1):
        view.setdefault("rank", idx)
    rendered_listings_md = _render_listings_markdown(render_views) if render_views else ""
    if rendered_listings_md:
        notes = (notes + " " if notes else "") + "Include the rendered_listings_md block verbatim."

    safety_instructions = (
        "You are acting as a U.S. rental real estate agent for Apartments.com listings. "
        "Stay within the provided data (listings, preferences, maps outputs). "
        "Do NOT invent listing details; if something is unknown, say so briefly. "
        "Keep responses concise, friendly, and aligned with your persona style. "
        "Use tools only when necessary; otherwise rely on the provided context. "
        "When you recommend listings (top matches, follow-ups, or persona summaries), format each with a clickable title linked to the Apartments.com URL, "
        "an inline thumbnail directly under the title when an image URL is available (skip only if missing), and bullet points for price, beds, baths, amenities, and why it matches. "
        "If rendered_listings_md is provided in the payload, include that block verbatim - it already follows this format with a compact inline image."
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
        "rendered_listings_md": rendered_listings_md,
    }

    client = get_openai_client()
    reply, streamed = stream_chat_completion(
        client,
        model=DEFAULT_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=persona_cfg.get("temperature", 0.5),
        top_p=persona_cfg.get("top_p", 1.0),
        stream_label="Agent: ",
        stream_to_stdout=True,
    )
    reply = reply or "Let me know how you'd like to adjust the search."
    return reply, persona_key, streamed


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

    if not is_real_estate_related(user_message):
        return {
            **state,
            "reply": OFF_TOPIC_REFUSAL,
            "reply_streamed": False,
            "off_topic": True,
            "preferences_updated": False,
            "clarifying_questions": [],
            "need_more_info": False,
        }
    state = {**state, "off_topic": False}

    system_prompt = (
        "You are a friendly U.S. rental real estate agent. "
        "Extract the user's preferences for Apartments.com searches. "
        "Only include fields that are clearly implied. "
        "Use JSON with a top-level 'preferences' object and optional 'clarifying_questions' list."
        "Preferences keys allowed: city, state, location, near_me, property_type, lifestyle, rooms_for_rent, "
        "min_rent, max_rent, min_beds, max_beds, min_baths, max_baths, pet_friendly, pet_type, "
        "cheap_only, utilities_included, amenity_slugs, keyword, frbo_only, page, filter_option, max_pages, tenant_name."
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

    raw, _ = stream_chat_completion(
        client,
        model=DEFAULT_OPENAI_MODEL,
        messages=extraction_messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = raw or "{}"
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
    reply, persona_key, streamed = generate_persona_reply(
        state, intent="clarify", listing_summaries=[], notes=notes
    )
    return {**state, "reply": reply, "active_persona": persona_key, "reply_streamed": streamed}


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
        reply, persona_key, streamed = generate_persona_reply(
            state,
            intent="error",
            notes="No search URL available; ask the user to verify city and state.",
        )
        return {**state, "reply": reply, "active_persona": persona_key, "reply_streamed": streamed}

    run_input = {
        "search_url": search_url,
        "max_pages": _safe_int(prefs.get("max_pages"), DEFAULT_MAX_PAGES) or DEFAULT_MAX_PAGES,
        "filter_option": prefs.get("filter_option") or DEFAULT_FILTER_OPTION,
    }

    try:
        items = scraper.fetch_listings_from_actor(run_input)
    except Exception as exc:
        reply, persona_key, streamed = generate_persona_reply(
            state,
            intent="error",
            notes=f"Scraper failed: {exc}. Apologize briefly and ask to confirm preferences or try again.",
        )
        return {
            **state,
            "reply": reply,
            "scraped_listings": [],
            "listings": [],
            "active_persona": persona_key,
            "reply_streamed": streamed,
        }

    items = items or []

    try:
        SCRAPER_OUTPUT_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    except Exception:
        pass

    if not items:
        reply, persona_key, streamed = generate_persona_reply(
            state,
            intent="no_results",
            notes="No listings were found; ask to adjust location or price range.",
        )
        return {
            **state,
            "reply": reply,
            "scraped_listings": [],
            "listings": [],
            "active_persona": persona_key,
            "reply_streamed": streamed,
        }

    return {**state, "listings": items, "scraped_listings": items}


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
        reply, persona_key, streamed = generate_persona_reply(
            state,
            intent="no_results",
            listing_summaries=[],
            notes="No listings were found; ask to adjust location, price, or beds/baths.",
        )
        return {
            **state,
            "reply": reply,
            "active_persona": persona_key,
            "preferences_updated": False,
            "reply_streamed": streamed,
        }

    listing_summaries: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked):
        view = _listing_prompt_view(item, prefs)
        view["rank"] = idx + 1
        view["score"] = _score_listing(item, prefs)
        view["why_match"] = _reason_tags(item, prefs, view.get("amenities", []))
        listing_summaries.append(view)

    notes = "Present the top matches in a natural, persona-aligned tone. Offer to refine price, beds/baths, or area."
    reply, persona_key, streamed = generate_persona_reply(
        state, intent="results", listing_summaries=listing_summaries, notes=notes
    )
    return {
        **state,
        "reply": reply,
        "ranked_listings": ranked,
        "listing_summaries": listing_summaries,
        "active_persona": persona_key,
        "preferences_updated": False,
        "listing_lookup": _build_listing_lookup(ranked),
        "reply_streamed": streamed,
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


def _listing_prompt_view(listing: Dict[str, Any], prefs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compact listing view for LLM context."""
    prefs = prefs or {}
    about = listing.get("about") or {}
    price_min, price_max = _extract_price_range(listing)
    beds, baths = _extract_beds_baths(listing)
    amenities = listing.get("amenities") or []
    amenity_titles: List[str] = []
    for item in amenities:
        amenity_titles.extend(item.get("amenities", [])[:3])
    amenity_preview = amenity_titles[:6]
    images = about.get("images") or []
    image_url = listing.get("image") or about.get("image") or (images[0] if images else None)
    listing_url = listing.get("listing_url") or listing.get("url")
    return {
        "title": about.get("title") or "Listing",
        "location": about.get("location") or _derive_listing_location(listing, prefs),
        "price_min": price_min,
        "price_max": price_max,
        "price_text": _format_price_label(price_min, price_max),
        "beds": beds,
        "baths": baths,
        "beds_text": _format_numeric(beds),
        "baths_text": _format_numeric(baths),
        "nearby_pois": listing.get("nearby_pois") or [],
        "features": (about.get("Unique Features") or [])[:5],
        "amenities": amenity_preview,
        "amenity_preview": ", ".join(amenity_preview) if amenity_preview else "",
        "url": listing_url,
        "listing_url": listing_url,
        "image": image_url,
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


def _format_numeric(value: Optional[float]) -> str:
    """Render numeric counts without trailing zeros."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return str(num).rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _format_price_label(price_min: Optional[int], price_max: Optional[int]) -> str:
    """Human-friendly price range text."""
    if price_min and price_max:
        return f"${int(price_min):,} - ${int(price_max):,}"
    if price_min:
        return f"${int(price_min):,}+"
    if price_max:
        return f"Up to ${int(price_max):,}"
    return "Price not listed"


def _render_listings_markdown(listing_views: List[Dict[str, Any]]) -> str:
    """Build markdown with clickable titles and inline thumbnails for each listing."""
    blocks: List[str] = []
    for idx, view in enumerate(listing_views, start=1):
        rank = view.get("rank") or idx
        title = view.get("title") or "Listing"
        listing_url = view.get("listing_url") or view.get("url")
        header = f"{rank}. **[{title}]({listing_url})**" if listing_url else f"{rank}. **{title}**"
        lines = [header]

        image_url = view.get("image")
        if image_url:
            alt_text = f"{title} - primary photo"
            lines.append(f'   <img src="{image_url}" alt="{alt_text}" width="{THUMBNAIL_WIDTH_PX}" />')

        price_text = view.get("price_text") or _format_price_label(view.get("price_min"), view.get("price_max"))
        beds_text = view.get("beds_text") or _format_numeric(view.get("beds"))
        baths_text = view.get("baths_text") or _format_numeric(view.get("baths"))
        location = view.get("location")
        amenity_preview = view.get("amenity_preview") or ", ".join(view.get("amenities") or [])
        reasons = view.get("why_match")

        if location:
            lines.append(f"   - **Location**: {location}")
        if price_text:
            lines.append(f"   - **Price**: {price_text}")
        lines.append(f"   - **Beds**: {beds_text}")
        lines.append(f"   - **Baths**: {baths_text}")
        if amenity_preview:
            lines.append(f"   - **Amenities**: {amenity_preview}")
        reason_text = ""
        if isinstance(reasons, list) and reasons:
            reason_text = "; ".join(reasons)
        elif isinstance(reasons, str) and reasons.strip():
            reason_text = reasons.strip()
        if reason_text:
            lines.append(f"   - **Why it matches**: {reason_text}")
        features = view.get("features") or []
        if features:
            lines.append(f"   - **Notable features**: {', '.join(features[:5])}")
        nearby = view.get("nearby_pois") or []
        if nearby:
            lines.append(f"   - **Nearby**: {', '.join(nearby[:3])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parse_price_value(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        cleaned = re.sub(r"[^\d]", "", str(value))
        return int(cleaned) if cleaned else None
    except Exception:
        return None


def _parse_rent_label(label: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not label:
        return None, None
    values = []
    for match in re.findall(r"\$?\s*([\d,]+)", label):
        try:
            values.append(int(match.replace(",", "")))
        except ValueError:
            continue
    if not values:
        return None, None
    return min(values), max(values)


def _clean_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"\s+", " ", name).strip()


def _extract_plan_options(listing: Dict[str, Any]) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    plans = listing.get("pricingAndFloorPlans") or []
    for idx, plan in enumerate(plans, start=1):
        if not isinstance(plan, dict):
            continue
        rent_min, rent_max = _parse_rent_label(plan.get("rent_label"))
        text_blob = " ".join(plan.get("details") or [])
        rent_text = " ".join([plan.get("rent_label") or "", text_blob]).lower()
        per_person = "per person" in rent_text
        option = {
            "index": idx,
            "model_name": _clean_model_name(plan.get("model_name")) or plan.get("model_name"),
            "details": plan.get("details") or [],
            "rent_label": plan.get("rent_label"),
            "rent_min": rent_min,
            "rent_max": rent_max,
            "availability": plan.get("availability"),
            "deposit": plan.get("deposit") or plan.get("deposit_label"),
            "units": plan.get("units") or [],
            "per_person": per_person,
        }
        options.append(option)
    return options


def _match_plan_selection(user_text: str, plan_options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    lowered = user_text.lower()
    numbers = re.findall(r"\d+", lowered)
    if numbers:
        try:
            idx = int(numbers[-1])
            for option in plan_options:
                if option.get("index") == idx:
                    return option
        except ValueError:
            pass
    for option in plan_options:
        name = (option.get("model_name") or "").lower()
        if name and name in lowered:
            return option
    return None


def _format_plan_prompt(plan_options: List[Dict[str, Any]], option_number: int) -> str:
    lines = [f"Option {option_number} has multiple floor plans. Which one should I use for the lease?"]
    for option in plan_options[:8]:
        label_parts = []
        if option.get("model_name"):
            label_parts.append(option["model_name"])
        if option.get("details"):
            label_parts.append("/".join(option["details"]))
        label = " - ".join(label_parts) if label_parts else "Plan"
        rent = option.get("rent_label") or "rent TBD"
        availability = option.get("availability") or "availability not listed"
        deposit = option.get("deposit")
        pieces = [f"{option['index']}. {label}", rent, availability]
        if deposit:
            pieces.append(f"Deposit: {deposit}")
        if option.get("per_person"):
            pieces.append("price per person")
        lines.append(" | ".join(pieces))
    lines.append("Reply with the plan number or name (e.g., 'plan 2' or 'B06').")
    return "\n".join(lines)


def _format_unit_prompt(unit_options: List[Dict[str, Any]], option_number: int, plan_name: Optional[str]) -> str:
    header = f"Option {option_number}"
    if plan_name:
        header += f" ({plan_name})"
    header += " has multiple units. Which one should I use for the lease?"
    lines = [header]
    for option in unit_options[:10]:
        idx = option.get("index")
        label = option.get("unit") or "Unit"
        price = option.get("price")
        sqft = option.get("square_feet")
        availability = option.get("availability")
        details = option.get("details")
        parts = [f"{idx}. {label}"]
        if price:
            parts.append(f"Base Price: ${price:,}")
        if sqft:
            parts.append(f"Sq Ft: {sqft}")
        if availability:
            parts.append(f"Availability: {availability}")
        if details:
            parts.append(f"Details: {details}")
        lines.append(" | ".join(parts))
    lines.append("Reply with the unit number (e.g., '1' or '2').")
    return "\n".join(lines)


def _match_unit_selection(user_text: str, unit_options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    lowered = user_text.lower().strip()
    numbers = re.findall(r"\d+", lowered)
    if numbers:
        try:
            idx = int(numbers[-1])
            for option in unit_options:
                if option.get("index") == idx:
                    return option
        except ValueError:
            pass
    return None


def _derive_listing_location(listing: Dict[str, Any], prefs: Dict[str, Any]) -> str:
    """Best-effort location string using Apartments.com listing metadata."""
    about = listing.get("about") or {}
    location = about.get("location")
    if location:
        return location
    breadcrumbs = about.get("breadcrumbs") or []
    state_full = breadcrumbs[0] if breadcrumbs else prefs.get("state")
    city = ""
    neighborhood = ""
    if len(breadcrumbs) >= 3:
        city = breadcrumbs[2]
    elif len(breadcrumbs) >= 2:
        city = breadcrumbs[1]
    if len(breadcrumbs) >= 4:
        neighborhood = breadcrumbs[3]
    state_abbr = _abbreviate_state(state_full) if state_full else None
    parts: List[str] = []
    if neighborhood:
        parts.append(neighborhood)
    if city or prefs.get("city"):
        parts.append(city or prefs.get("city"))
    if state_abbr or state_full:
        parts.append(state_abbr or state_full)
    return ", ".join(p for p in parts if p)


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


def _normalize_lookup_key(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _listing_identity(listing: Dict[str, Any]) -> Dict[str, Optional[str]]:
    about = listing.get("about") or {}
    return {
        "title": about.get("title"),
        "location": about.get("location"),
        "url": listing.get("listing_url") or listing.get("url"),
    }


def _build_listing_lookup(listings: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for idx, listing in enumerate(listings, start=1):
        identity = _listing_identity(listing)
        keys = {
            str(idx),
            f"option {idx}",
            f"listing {idx}",
            f"choice {idx}",
        }
        title_key = _normalize_lookup_key(identity.get("title") or "")
        if title_key:
            keys.add(title_key)
        location_key = _normalize_lookup_key(identity.get("location") or "")
        if location_key:
            keys.add(location_key)
        if title_key and location_key:
            keys.add(_normalize_lookup_key(f"{identity.get('title')} {identity.get('location')}"))
        url = identity.get("url")
        if url:
            keys.add(url.lower())
            keys.add(url.split("?")[0].lower())
        record = {**identity, "index": idx - 1}
        for key in keys:
            lookup[key] = record
    return lookup


def _load_scraped_output(state: AgentState, *, force_reload: bool = False) -> List[Dict[str, Any]]:
    if not force_reload and state.get("scraped_listings") is not None:
        return state.get("scraped_listings") or []
    try:
        data = json.loads(SCRAPER_OUTPUT_FILE.read_text(encoding="utf-8"))
        listings = data if isinstance(data, list) else []
    except FileNotFoundError:
        listings = state.get("scraped_listings") or []
    except Exception:
        listings = state.get("scraped_listings") or []
    state["scraped_listings"] = listings
    return listings


def _identify_listing_from_message(
    message: str, lookup: Dict[str, Dict[str, Any]], listings: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    text = (message or "").lower()
    if not text:
        return None
    match = re.search(r"(?:option|listing|choice)\s*(\d+)", text)
    if match:
        try:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(listings):
                identity = _listing_identity(listings[idx])
                identity["index"] = idx  # type: ignore[index]
                return identity
        except ValueError:
            pass
    for key, record in lookup.items():
        if not key:
            continue
        if key.isdigit():
            if re.search(rf"\b{re.escape(key)}\b", text):
                return record
        elif key in text:
            return record
    url_match = re.search(r"https?://\S+", text)
    if url_match:
        url = url_match.group(0).lower()
        for record in lookup.values():
            record_url = record.get("url")
            if record_url and url in record_url.lower():
                return record
    return None


def _find_listing_in_data(listings: List[Dict[str, Any]], identity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not identity:
        return None
    index = identity.get("index")
    if isinstance(index, int) and 0 <= index < len(listings):
        return listings[index]
    target_url = identity.get("url")
    target_title = _normalize_lookup_key(identity.get("title"))
    target_location = _normalize_lookup_key(identity.get("location"))
    for listing in listings:
        meta = _listing_identity(listing)
        url = meta.get("url")
        if target_url and url and target_url.lower().split("?")[0] == url.lower().split("?")[0]:
            return listing
        if target_title and _normalize_lookup_key(meta.get("title")) == target_title:
            if not target_location or _normalize_lookup_key(meta.get("location")) == target_location:
                return listing
    return None


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


def _resolve_tenant_name(state: AgentState) -> Optional[str]:
    """Best-effort extraction of the tenant's name supplied by the user."""
    preferences = state.get("preferences") or {}
    for key in ("tenant_name", "name", "user_name", "contact_name"):
        value = preferences.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _store_tenant_name(state: AgentState, name: str) -> None:
    """Persist the captured tenant name into preferences."""
    cleaned = name.strip().strip('"').strip()
    if not cleaned:
        return
    _store_preference(state, "tenant_name", cleaned)


def _extract_name_from_message(message: str) -> Optional[str]:
    text = message.strip()
    if not text:
        return None
    pattern = re.search(r"(?:my\s+full\s+legal\s+name\s+is|my\s+name\s+is|name:)\s*(.+)", text, re.IGNORECASE)
    if pattern:
        candidate = pattern.group(1).strip()
        candidate = candidate.rstrip(".")
        return candidate or None
    return text if text else None


def _store_preference(state: AgentState, key: str, value: Any) -> None:
    prefs = state.get("preferences") or {}
    prefs[key] = value
    state["preferences"] = prefs


def _parse_move_in_date(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    cleaned = text.replace(",", " ").strip()
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.date().isoformat()
        except Exception:
            continue
    partial_formats = ["%B %d", "%b %d", "%m/%d", "%d %B", "%d %b"]
    today = date.today()
    for fmt in partial_formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            dt = dt.replace(year=today.year)
            return dt.date().isoformat()
        except Exception:
            continue
    numbers = re.findall(r"\d{4}-\d{1,2}-\d{1,2}", cleaned)
    if numbers:
        return numbers[0]
    return None


def _parse_duration_months(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(month|mo)", text.lower())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    numbers = re.findall(r"\d+", text)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            return None
    return None


def _collect_text_fragments(value: Any) -> List[str]:
    fragments: List[str] = []
    if isinstance(value, str):
        fragments.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            fragments.extend(_collect_text_fragments(item))
    elif isinstance(value, list):
        for item in value:
            fragments.extend(_collect_text_fragments(item))
    return fragments


def _extract_lease_duration_bounds(listing: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    texts: List[str] = []
    fees = listing.get("feesAndPolicies") or {}
    texts.extend(_collect_text_fragments(fees))
    about = listing.get("about") or {}
    description = about.get("description")
    if description:
        texts.append(str(description))

    min_months: Optional[int] = None
    max_months: Optional[int] = None

    for text in texts:
        lower = text.lower()
        for match in re.finditer(r"(\d+)\s*(?:-|to|–|—)\s*(\d+)\s*(?:month)", lower):
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                start, end = end, start
            if min_months is None or start < min_months:
                min_months = start
            if max_months is None or end > max_months:
                max_months = end
        for match in re.finditer(r"(\d+)\s*(?:month)", lower):
            value = int(match.group(1))
            if min_months is None or value < min_months:
                min_months = value
            if max_months is None or value > max_months:
                max_months = value

    return min_months, max_months


def _select_reference_listing(state: AgentState, choice_index: int = 0) -> Optional[Dict[str, Any]]:
    """Pick a listing for lease drafting context, honoring the user's chosen option."""
    for key in ("ranked_listings", "enriched_listings", "listings"):
        listings = state.get(key) or []
        if listings:
            normalized_index = max(0, min(choice_index, len(listings) - 1))
            return listings[normalized_index]
    return None


def handle_lease_command(state: AgentState, user_input: str) -> Optional[str]:
    """
    Generate a lease draft when the user explicitly requests it.
    Trigger phrases: "lease draft", "draft lease", "option 1".
    """
    normalized = user_input.strip().lower()
    if not normalized:
        return None

    waiting_for_plan = bool(state.get("pending_lease_waiting_plan"))
    pending_choice = state.get("pending_lease_choice")
    if waiting_for_plan and pending_choice is not None:
        plan_options = state.get("pending_lease_plan_options") or []
        selection = _match_plan_selection(user_input, plan_options)
        if not selection:
            prompt = _format_plan_prompt(plan_options, pending_choice + 1)
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": prompt})
            state["messages"] = history
            return prompt
        state["pending_lease_selected_plan"] = selection
        state["pending_lease_waiting_plan"] = False
        state["pending_lease_plan_options"] = []
        normalized = f"option {pending_choice + 1} lease draft"
        user_input = normalized

    waiting_for_unit = bool(state.get("pending_lease_waiting_unit"))
    if waiting_for_unit and pending_choice is not None:
        unit_options = state.get("pending_lease_unit_options") or []
        selection = _match_unit_selection(user_input, unit_options)
        if not selection:
            plan_name = state.get("pending_lease_selected_plan", {}).get("model_name")
            prompt = _format_unit_prompt(unit_options, pending_choice + 1, plan_name)
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": prompt})
            state["messages"] = history
            return prompt
        state["pending_lease_selected_unit"] = selection
        state["pending_lease_waiting_unit"] = False
        state["pending_lease_unit_options"] = []
        normalized = f"option {pending_choice + 1} lease draft"
        user_input = normalized

    waiting_for_start = bool(state.get("pending_lease_waiting_start"))
    if waiting_for_start and pending_choice is not None:
        move_date = _parse_move_in_date(user_input)
        if not move_date:
            reply = "Please provide the move-in date in YYYY-MM-DD format."
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            state["messages"] = history
            return reply
        _store_preference(state, "lease_start_date", move_date)
        state["pending_lease_waiting_start"] = False
        normalized = f"option {pending_choice + 1} lease draft"
        user_input = normalized

    waiting_for_duration = bool(state.get("pending_lease_waiting_duration"))
    if waiting_for_duration and pending_choice is not None:
        bounds = state.get("pending_lease_duration_bounds") or (None, None)
        lower_bound, upper_bound = bounds
        user_lower = user_input.lower()
        months = None
        if ("all" in user_lower or "max" in user_lower or "full" in user_lower) and upper_bound:
            months = upper_bound
        else:
            months = _parse_duration_months(user_input)
        if not months:
            if upper_bound and lower_bound:
                reply = f"How many months should the lease run? (between {lower_bound} and {upper_bound} months)"
            elif upper_bound:
                reply = f"How many months should the lease run? (up to {upper_bound} months)"
            else:
                reply = "How many months should the lease run? (e.g., 12 months)"
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            state["messages"] = history
            return reply
        if upper_bound and months > upper_bound:
            reply = f"The community lists lease terms up to {upper_bound} months. Please choose a duration within that range."
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            state["messages"] = history
            return reply
        if lower_bound and months < lower_bound:
            reply = f"Lease terms start at {lower_bound} months for this community. Please choose a duration within that range."
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            state["messages"] = history
            return reply
        _store_preference(state, "lease_duration_months", months)
        state["pending_lease_waiting_duration"] = False
        normalized = f"option {pending_choice + 1} lease draft"
        user_input = normalized

    waiting_for_name = bool(state.get("pending_lease_waiting_name"))
    pending_choice = state.get("pending_lease_choice")
    tenant_name: Optional[str] = _resolve_tenant_name(state)

    if waiting_for_name and pending_choice is not None:
        extracted = _extract_name_from_message(user_input)
        if not extracted:
            reply = "I didn’t catch your full legal name. Could you please type it clearly?"
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            state["messages"] = history
            return reply
        _store_tenant_name(state, extracted)
        tenant_name = extracted
        normalized = f"option {pending_choice + 1} lease draft"
        state["pending_lease_waiting_name"] = False
        state["pending_lease_choice"] = None

    triggers = ("lease draft", "draft lease", "option 1", "generate lease", "lease option")
    if not any(trigger in normalized for trigger in triggers):
        return None

    match = re.search(r"option\s*(\d+)", normalized)
    if not match:
        reply = "Let me know which option number you'd like me to draft a lease for (e.g., 'option 2 lease draft')."
        history = state.get("messages") or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        state["messages"] = history
        return reply
    choice_index = max(0, int(match.group(1)) - 1)

    tenant_name = tenant_name or _resolve_tenant_name(state)
    if not tenant_name:
        reply = "Please share your full legal name so I can include it in the lease draft."
        history = state.get("messages") or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        state["messages"] = history
        state["pending_lease_waiting_name"] = True
        state["pending_lease_choice"] = choice_index
        return reply

    listing = _select_reference_listing(state, choice_index)
    if not listing:
        reply = "I don't have details for that option yet. Please run a search and pick one of the listed properties (e.g., 'option 2 lease draft')."
        history = state.get("messages") or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        state["messages"] = history
        return reply

    if not _resolve_tenant_name(state):
        _store_tenant_name(state, tenant_name)

    selected_plan = state.get("pending_lease_selected_plan")
    plan_options = _extract_plan_options(listing)
    if not selected_plan:
        if len(plan_options) == 1:
            selected_plan = plan_options[0]
        elif len(plan_options) > 1:
            prompt = _format_plan_prompt(plan_options, choice_index + 1)
            state["pending_lease_waiting_plan"] = True
            state["pending_lease_plan_options"] = plan_options
            state["pending_lease_choice"] = choice_index
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": prompt})
            state["messages"] = history
            return prompt
        else:
            state["pending_lease_plan_options"] = []
            state["pending_lease_waiting_plan"] = False
    state["pending_lease_selected_plan"] = selected_plan
    state["pending_lease_plan_options"] = []
    state["pending_lease_choice"] = choice_index

    # Prompt for unit immediately after plan selection
    plan_units_source = selected_plan or {}
    selected_unit = state.get("pending_lease_selected_unit")
    plan_units = plan_units_source.get("units") or []
    if not selected_unit and plan_units:
        unit_options = []
        for idx, unit in enumerate(plan_units, start=1):
            if not isinstance(unit, dict):
                continue
            price = _parse_price_value(unit.get("price"))
            details = unit.get("details")
            if isinstance(details, list):
                details = ", ".join(details)
            unit_options.append(
                {
                    "index": idx,
                    "unit": unit.get("unit") or unit.get("label") or "",
                    "price": price,
                    "square_feet": unit.get("square_feet"),
                    "availability": unit.get("availability") or unit.get("available") or unit.get("availabilityStatus"),
                    "details": details,
                }
            )
        priced_options = [u for u in unit_options if u.get("price")]
        target_options = priced_options or unit_options
        if target_options:
            prompt = _format_unit_prompt(
                target_options,
                choice_index + 1,
                selected_plan.get("model_name") if selected_plan else None,
            )
            state["pending_lease_waiting_unit"] = True
            state["pending_lease_unit_options"] = target_options
            state["pending_lease_choice"] = choice_index
            history = state.get("messages") or []
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": prompt})
            state["messages"] = history
            return prompt
    state["pending_lease_selected_unit"] = selected_unit

    duration_bounds = _extract_lease_duration_bounds(listing)
    if duration_bounds == (None, None):
        duration_bounds = (12, 12)
    state["pending_lease_duration_bounds"] = duration_bounds

    preferences = state.get("preferences") or {}
    preferences = state.get("preferences") or {}
    if not preferences.get("lease_start_date"):
        prompt = "When should the lease start? Please reply with a move-in date (YYYY-MM-DD)."
        state["pending_lease_waiting_start"] = True
        state["pending_lease_choice"] = choice_index
        history = state.get("messages") or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": prompt})
        state["messages"] = history
        return prompt

    if not preferences.get("lease_duration_months"):
        lower_bound, upper_bound = state.get("pending_lease_duration_bounds") or (None, None)
        if lower_bound and upper_bound:
            prompt = f"How many months should the lease run? (the community lists {lower_bound}-{upper_bound} month terms)"
        elif upper_bound:
            prompt = f"How many months should the lease run? (up to {upper_bound} months)"
        else:
            prompt = "How many months should the lease run? (e.g., 12 months)"
        state["pending_lease_waiting_duration"] = True
        state["pending_lease_choice"] = choice_index
        history = state.get("messages") or []
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": prompt})
        state["messages"] = history
        return prompt

    listing_summary: Optional[Dict[str, Any]] = None
    summaries = state.get("listing_summaries") or []
    if 0 <= choice_index < len(summaries):
        listing_summary = summaries[choice_index]

    overrides: Dict[str, Any] = {"tenant_name": tenant_name}
    if selected_plan:
        if selected_plan.get("rent_min"):
            overrides["monthly_rent"] = selected_plan["rent_min"]
        if selected_plan.get("model_name"):
            overrides["selected_plan_name"] = selected_plan.get("model_name")
        details_text = ", ".join(selected_plan.get("details") or [])
        if details_text:
            overrides["selected_plan_details"] = details_text
        overrides["selected_plan_availability"] = selected_plan.get("availability")
        overrides["selected_plan_rent_label"] = selected_plan.get("rent_label")
        overrides["selected_plan_deposit"] = selected_plan.get("deposit")
        overrides["selected_plan_price_per_person"] = selected_plan.get("per_person", False)
    selected_unit = state.get("pending_lease_selected_unit")
    if selected_unit:
        if selected_unit.get("price"):
            overrides["monthly_rent"] = selected_unit["price"]
        overrides["selected_unit_label"] = selected_unit.get("unit")
        overrides["selected_unit_sqft"] = selected_unit.get("square_feet")
        overrides["selected_unit_price"] = selected_unit.get("price")
        overrides["selected_unit_availability"] = selected_unit.get("availability")
        overrides["selected_unit_details"] = selected_unit.get("details")
    if listing_summary:
        rent_choice = listing_summary.get("price_min")
        if rent_choice is None:
            rent_choice = listing_summary.get("price_max")
        if rent_choice is not None and "monthly_rent" not in overrides:
            overrides["monthly_rent"] = rent_choice
        location_choice = listing_summary.get("location")
        if location_choice:
            overrides.setdefault("property_address", location_choice)

    if preferences.get("lease_start_date"):
        overrides["lease_start_date"] = preferences["lease_start_date"]
    if preferences.get("lease_duration_months"):
        overrides["lease_term_months"] = preferences["lease_duration_months"]

    property_location = ""
    if listing:
        property_location = _derive_listing_location(listing, state.get("preferences", {}))
    if property_location and not overrides.get("property_address"):
        overrides["property_address"] = property_location

    lease_inputs = lease_drafter.infer_inputs(
        preferences=state.get("preferences"),
        listing=listing,
        overrides=overrides,
    )
    package = lease_drafter.build_lease_package(lease_inputs)
    compliance = package["compliance"]

    state["last_listing"] = listing
    state["last_overrides"] = copy.deepcopy(overrides)
    state["last_duration_bounds"] = state.get("pending_lease_duration_bounds")

    source_path = None
    if listing:
        base_path = Path(package["file_path"])
        source_path = base_path.parent / "option_source.json"
        source_path.write_text(json.dumps(listing, indent=2, ensure_ascii=True), encoding="ascii", errors="ignore")
        package["listing_source"] = str(source_path)

    issues = compliance.get("issues", [])
    warnings = compliance.get("warnings", [])
    listing_title = ""
    listing_location = ""
    if listing:
        about = listing.get("about") or {}
        listing_title = about.get("title") or "Listing"
        listing_location = about.get("location") or ""

    summary_lines = [
        "Lease draft generated from the Apartments.com context.",
        f"Selected option: {choice_index + 1}" if listing else "Selected option: n/a (no listing context).",
        f"Listing: {listing_title} {f'({listing_location})' if listing_location else ''}".strip()
        if listing
        else "Listing: n/a",
        f"Saved text: {package['file_path']}",
        f"Saved PDF: {package.get('pdf_path', 'n/a')}",
        f"Listing JSON: {str(source_path)}" if source_path else "Listing JSON: n/a",
        f"Tenant name: {lease_inputs.tenant_name}",
    ]
    if issues:
        summary_lines.append("Compliance issues detected:")
        summary_lines.extend(f"- {issue}" for issue in issues)
    else:
        summary_lines.append("No blocking compliance issues detected.")
    if warnings:
        summary_lines.append("Warnings:")
        summary_lines.extend(f"- {warn}" for warn in warnings)
    if compliance.get("rules_checked"):
        summary_lines.append(
            "Rules referenced: " + ", ".join(compliance["rules_checked"])
        )
    summary_lines.append("Review and customize before sending to tenants.")
    reply = "\n".join(summary_lines)

    history = state.get("messages") or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    state["messages"] = history

    drafts = state.get("lease_drafts") or []
    drafts.append(package)
    state["lease_drafts"] = drafts
    state["pending_lease_waiting_name"] = False
    state["pending_lease_waiting_plan"] = False
    state["pending_lease_plan_options"] = []
    state["pending_lease_selected_plan"] = None
    state["pending_lease_waiting_unit"] = False
    state["pending_lease_unit_options"] = []
    state["pending_lease_selected_unit"] = None
    state["pending_lease_waiting_start"] = False
    state["pending_lease_waiting_duration"] = False
    state["pending_lease_choice"] = None
    state["pending_lease_waiting_name"] = False
    state["pending_lease_waiting_plan"] = False
    state["pending_lease_waiting_unit"] = False
    state["pending_lease_unit_options"] = []
    state["pending_lease_selected_unit"] = None
    state["pending_lease_plan_options"] = []
    state["pending_lease_selected_plan"] = None
    state["pending_lease_waiting_start"] = False
    state["pending_lease_waiting_duration"] = False
    state["pending_lease_duration_bounds"] = (None, None)
    return reply


def handle_lease_update_request(state: AgentState, user_input: str) -> Optional[str]:
    lowered = user_input.lower().strip()
    if not lowered:
        return None
    keywords = ["update lease", "change lease", "edit lease", "modify lease", "lease update"]
    if not any(keyword in lowered for keyword in keywords):
        return None

    last_listing = state.get("last_listing")
    last_overrides = state.get("last_overrides")
    if not last_listing or not last_overrides:
        return "I don't have a recent lease draft to update yet. Please generate one first."

    overrides = copy.deepcopy(last_overrides)
    preferences = state.get("preferences") or {}
    updates_applied: List[str] = []

    move_date = _parse_move_in_date(user_input)
    if move_date:
        overrides["lease_start_date"] = move_date
        preferences["lease_start_date"] = move_date
        updates_applied.append(f"Move-in date set to {move_date}.")

    duration_bounds = state.get("last_duration_bounds") or (None, None)
    if "month" in lowered or "term" in lowered or "duration" in lowered:
        months = _parse_duration_months(user_input)
        if months:
            lower_bound, upper_bound = duration_bounds
            if upper_bound and months > upper_bound:
                return f"The community allows terms up to {upper_bound} months. Please choose a duration within that range."
            if lower_bound and months < lower_bound:
                return f"The community requires at least {lower_bound} months. Please choose a duration within that range."
            overrides["lease_term_months"] = months
            preferences["lease_duration_months"] = months
            updates_applied.append(f"Lease term set to {months} months.")

    if "rent" in lowered or "$" in user_input:
        rent_value = _parse_price_value(user_input)
        if rent_value:
            overrides["monthly_rent"] = rent_value
            updates_applied.append(f"Monthly rent updated to ${rent_value:,}.")

    if "name" in lowered:
        name = _extract_name_from_message(user_input)
        if name:
            overrides["tenant_name"] = name
            preferences["tenant_name"] = name
            updates_applied.append(f"Tenant name updated to {name}.")

    if not updates_applied:
        return "Let me know what you'd like to change (e.g., move-in date, lease term, or rent)."

    state["preferences"] = preferences
    lease_inputs = lease_drafter.infer_inputs(
        preferences=preferences,
        listing=last_listing,
        overrides=overrides,
    )
    package = lease_drafter.build_lease_package(lease_inputs)
    state["last_overrides"] = copy.deepcopy(overrides)
    state["last_listing"] = last_listing
    state["last_duration_bounds"] = duration_bounds
    source_path = None
    base_path = Path(package["file_path"])
    source_path = base_path.parent / "option_source.json"
    source_path.write_text(json.dumps(last_listing, indent=2, ensure_ascii=True), encoding="ascii", errors="ignore")
    package["listing_source"] = str(source_path)

    drafts = state.get("lease_drafts") or []
    drafts.append(package)
    state["lease_drafts"] = drafts

    summary = [
        "Updated the existing lease draft:",
        *updates_applied,
        f"New text: {package['file_path']}",
        f"New PDF: {package.get('pdf_path', 'n/a')}",
    ]
    return "\n".join(summary)


def answer_with_existing(state: AgentState) -> AgentState:
    """Use existing listings and history to answer follow-ups without re-scraping."""
    messages = state.get("messages") or []
    latest_user_message = messages[-1]["content"] if messages else ""
    prefs = state.get("preferences", {})

    scraped_data = _load_scraped_output(state, force_reload=False)
    lookup_source = scraped_data or state.get("ranked_listings") or state.get("listings") or []
    lookup = state.get("listing_lookup") or _build_listing_lookup(lookup_source)
    state["listing_lookup"] = lookup

    # Check if the user is asking about a specific listing and reload disk data for freshness.
    identity = _identify_listing_from_message(latest_user_message, lookup, lookup_source)
    if identity:
        refreshed_data = _load_scraped_output(state, force_reload=True)
        focus_listing = _find_listing_in_data(refreshed_data or lookup_source, identity)
        if focus_listing:
            state["scraped_listings"] = refreshed_data or lookup_source
            state["listing_lookup"] = _build_listing_lookup(refreshed_data or lookup_source)
            reply, persona_key, streamed = generate_persona_reply(
                state,
                intent="listing_follow_up",
                listing_summaries=[],
                notes="Use only the focused listing data loaded from actor_output.json; be concise and avoid inventing missing details.",
                focused_listing=focus_listing,
            )
            return {
                **state,
                "reply": reply,
                "active_persona": persona_key,
                "focused_listing": focus_listing,
                "listing_lookup": state.get("listing_lookup", {}),
                "reply_streamed": streamed,
            }

    listings = state.get("enriched_listings") or state.get("scraped_listings") or state.get("listings") or []
    if not listings:
        reply, persona_key, streamed = generate_persona_reply(
            state,
            intent="no_cached",
            notes="No cached listings yet; ask for city/state/budget to start a search.",
        )
        return {**state, "reply": reply, "active_persona": persona_key, "reply_streamed": streamed}

    preview_listings = listings[:5]
    listing_summaries: List[Dict[str, Any]] = []
    for idx, item in enumerate(preview_listings, start=1):
        view = _listing_prompt_view(item, prefs)
        view["rank"] = idx
        view["why_match"] = _reason_tags(item, prefs, view.get("amenities", []))
        listing_summaries.append(view)
    notes = "Answer the user's follow-up using cached listings; compare briefly and be transparent about trade-offs."
    reply, persona_key, streamed = generate_persona_reply(
        state, intent="follow_up", listing_summaries=listing_summaries, notes=notes
    )
    return {**state, "reply": reply, "active_persona": persona_key, "reply_streamed": streamed}


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
        lambda s: "end"
        if s.get("off_topic")
        else "clarify"
        if s.get("need_more_info")
        else "answer_existing"
        if s.get("listings") and not s.get("preferences_updated")
        else "build_query",
        {
            "clarify": "clarify",
            "build_query": "build_query",
            "answer_existing": "answer_existing",
            "end": END,
        },
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
        update_reply = handle_lease_update_request(state, user_input)
        if update_reply:
            print(f"Agent: {update_reply}\n")
            continue
        lease_reply = handle_lease_command(state, user_input)
        if lease_reply:
            print(f"Agent: {lease_reply}\n")
            continue

        if not is_real_estate_related(user_input):
            refusal = OFF_TOPIC_REFUSAL
            state_messages = state.get("messages") or []
            state_messages.append({"role": "user", "content": user_input})
            state_messages.append({"role": "assistant", "content": refusal})
            state["messages"] = state_messages
            state["off_topic"] = True
            print(f"Agent: {refusal}\n")
            continue

        state_messages = state.get("messages") or []
        state_messages.append({"role": "user", "content": user_input})
        state["messages"] = state_messages
        state["reply_streamed"] = False
        state["off_topic"] = False

        state = app.invoke(state)
        reply = state.get("reply") or "I didn't catch that—could you rephrase?"
        streamed = state.get("reply_streamed", False)
        if not streamed:
            print(f"Agent: {reply}")
        print()

        # Maintain conversation memory
        state["messages"].append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
