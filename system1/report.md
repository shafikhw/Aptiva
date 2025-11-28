# System 1 (US) Real Estate Agent – Detailed Report

## Purpose and Scope
- Provides a LangGraph-driven, persona-aware rental assistant focused strictly on U.S. residential rentals (search, comparison, viewings, leases).
- Works off Apartments.com data: builds search URLs, scrapes listings via Apify, enriches with Google Maps, ranks, and answers follow-ups using cached results.
- Generates draft residential leases (text + PDF) with basic compliance checks and interactive editing.

## Core Modules
- `real_estate_agent.py`: end-to-end agent logic (conversation guardrails, LangGraph nodes, scraping, enrichment, ranking, lease orchestration, CLI).
- `url_complex.py`: Apartments.com URL builder (location slugs, property type/lifestyle/pet/price/bed-bath filters, near-me paths).
- `scraper.py`: Apify actor wrapper to fetch listings and persist `actor_output.json` fallback data.
- `lease_drafter.py`: Lease input inference, draft assembly, compliance checks, and PDF rendering.
- `session.py`: Web-friendly session wrapper that reuses the compiled graph and handles streaming callbacks.

## Conversation Guardrails and Personas
- Domain guard system message confines answers to U.S. residential rentals; non-housing queries trigger `OFF_TOPIC_REFUSAL`.
- Real-estate detection combines keyword lists, regexes (beds/baths, ZIPs, “City, ST”), and explicit off-topic phrase filters (cars/equipment rentals, commercial leases, travel, etc.).
- Personas:
  - *Neighborhood Naturalist* (warm, vibe-focused, temp 0.9).
  - *Data Whisperer* (concise, metric-driven, temp 0.3).
  - *Deal Navigator* (trade-off and negotiation framing, temp 0.5).
  - *Auto* picks persona heuristically based on user text (deal/data/vibe keywords).
- Replies always injected with guardrail + persona safety instructions; streaming supported via `STREAM_CALLBACK` and stdout labels.

## Preference Extraction and Search Prep
- `analyze_preferences` LLM node extracts structured prefs (city/state/location/near_me, property type, lifestyle, rooms_for_rent, rent/beds/baths ranges, pets, cheap/utilities-included, amenity slugs, FRBO, keyword, pagination, filter_option, tenant name).
- `_normalize_city_state` splits legacy `location`, abbreviates states; drops invalid state input to force clarification.
- `compute_missing_preferences` demands city/state (unless near_me) and at least budget or bed/bath before searching; missing items become clarifying questions.
- `build_query_from_preferences` maps prefs into `ApartmentsSearchQuery`, normalizing enums and numeric fields.

## Apartments.com URL Construction (`url_complex.py`)
- `ApartmentsSearchQuery.build_url` assembles paths using real site patterns:
  - Location slug via `slugify_location(city, state)`.
  - Property type prefixes (houses/condos/townhomes) or lofts trailing slug.
  - Lifestyle pages (`student-housing`, `senior-housing`, `corporate`, `military`, `short-term`).
  - Special pages: rooms-for-rent, cheap, utilities-included, amenity-only slugs.
  - Combined numeric slug builder for beds/baths/price/pet (e.g., `min-2-bedrooms-1-bathrooms-under-1200-pet-friendly-dog`).
  - Near-me paths with cheap/utilities variants; pagination appended as final segment.

## Scraping Pipeline (`scraper.py` + LangGraph `scrape` node)
- Requires `APIFY_API_TOKEN`/`APIFY_TOKEN`; defaults to actor `BvepfniODI2AixuNN`.
- `run_actor_and_save_outputs` runs actor with `{"search_url", "max_pages" (1–5), "filter_option"}` and saves JSON to `actor_output.json`.
- `scrape_listings` node:
  - Skips re-scrape if listings exist and prefs unchanged.
  - Streams a scrape-start signal via callback.
  - Loads scraped items from actor output; gracefully falls back to cached `actor_output.json` if scraping fails or returns empty, adding notices for transparency.

## Listing Enrichment, Ranking, and Formatting
- Maps enrichment (`enrich_with_maps`): optional Google Maps client (`GOOGLE_MAPS_API_KEY`) geocodes listing address/title/url text and collects nearby POIs (gyms, schools, parks, shopping, hospitals, beaches) within 2 km.
- Ranking (`rank_and_format`):
  - Scores listings by budget alignment (min/max rent), beds fit, and pet hints; selects top 5.
  - Builds compact prompt views with price ranges, beds/baths, amenity preview, image URL, and “why it matches” tags.
- Rendering:
  - `_render_listings_markdown` outputs clickable titles with inline thumbnails (if available), bullet details (location, price, beds/baths, amenities, reasons, features, nearby POIs).
  - `_build_listing_lookup` maps option numbers, titles, locations, and URLs to allow free-form references in follow-ups.
- Follow-ups without re-scrape: `answer_with_existing` reloads cached `actor_output.json` when the user mentions a known listing and answers using cached/enriched data.

## LangGraph Workflow
- Nodes: `analyze_preferences` → (clarify | build_query | answer_existing) → `scrape` → `enrich` → `rank_and_format`.
- Conditional edges handle off-topic, missing info, cached listing reuse, and errors; final replies generated via `generate_persona_reply`.
- State (`AgentState`) tracks messages, preferences, search URL/query, scraped/enriched/ranked listings, listing lookup, focused listing, persona state, reply metadata, lease workflow fields, and notices.

## Lease Drafting and Interactive Flow
- Detection: lease trigger phrases (“lease draft”, “draft lease”, “option 2 lease”) or `maybe_schedule_lease_generation` prompts.
- Required context: city/state and budget before auto-collect lease details; otherwise asks for missing fields.
- Multi-stage interactive flow:
  - Floor-plan selection: parses `pricingAndFloorPlans` into options (index, model_name, details, rent range, availability, deposit, per-person flag) and prompts with navigation hints.
  - Unit selection: per-plan units (price, sqft, availability, details) with navigation/back/restart commands.
  - Tenant name capture: first/last/full-name parsing, validation, normalization; supports change-name command and back/restart.
  - Move-in date parsing: multiple date formats; enforces allowable window using listing/plan/unit availability hints with default max `2026-03-31`.
  - Lease term capture: infers duration bounds from listing text/fees and enforces min/max; supports term-only edits.
  - Aliases: users can nickname plans/units in quotes; lookup retained in state.
  - Reasoning assist: `_reason_about_lease_input` uses OpenAI reasoning model (`o4-mini`) to classify intents (select/change option/plan/unit, provide names/dates/duration, navigation, property questions).
  - Property Q&A mid-flow answered from listing summaries while keeping lease context.
- Lease updates: `handle_lease_update_request` edits last draft (move-in date, term, rent, tenant name) with bounds checks and regenerates draft.
- Simple collection path: `maybe_schedule_lease_generation` + `continue_lease_collection` ask through `LEASE_COLLECTION_FIELDS` (tenant name, floor plan, unit, start date, term) when detailed listing options aren’t available.
- Navigation commands recognized: back/previous/restart lease, change plan/unit/name/date/term, compare plans/units, show plans/units, refine search.

## Lease Drafting Engine (`lease_drafter.py`)
- `infer_inputs` merges conversation prefs, selected listing data (landlord name from contact/logo/URL/title, address, rent), and overrides; normalizes names and state.
- `LeaseDraftInputs` dataclass holds all draft knobs (parties, address, rent, deposit, fees, utilities split, pets, plan/unit selections, dates/terms, caps, clauses).
- Draft generation (`generate_lease_text`):
  - Outputs structured ASCII lease patterned after Apartments.com sample, covering mandatory sections (parties, premises, term, rent/fees, late fees, deposit, maintenance, use/occupancy, vehicles, utilities, access, insurance, pets, default, addenda, compliance context).
  - Embeds selected plan/unit details and notes per-person pricing.
- Compliance checks (`compliance_report`):
  - State heuristics for CA/NY/MA (deposit caps, late fee caps, pet rent cap) plus rent cap vs budget and mandatory clause presence.
  - Warns on TBD rent or missing address parts.
- PDF rendering: uses ReportLab; drafts saved to `system1/lease_drafts/{tenant}_{id}_lease_draft.txt` and `.pdf`.
- Helper exports: `render_pdf_from_text`, `text_file_to_pdf`.

## Session and CLI Runtime
- CLI (`main` in `real_estate_agent.py`): prompts for persona, greets, runs REPL with commands (`/persona`, lease update/draft commands, short-negative responses, off-topic refusals), streaming replies when possible.
- Session wrapper (`System1AgentSession`):
  - Reuses compiled graph, keeps JSON-safe state snapshots, handles streaming via `STREAM_CALLBACK`.
  - Routes persona switches, lease updates/commands, lease collection continuation before invoking the graph; always returns updated state and prefs for clients.

## Configuration, Dependencies, and Artifacts
- Environment:
  - `OPENAI_API_KEY` (required), `OPENAI_MODEL` (default `gpt-4o-mini`), `OPENAI_REASONING_MODEL` (default `o4-mini`).
  - `GOOGLE_MAPS_API_KEY` (optional POI enrichment).
  - `APIFY_API_TOKEN` / `APIFY_TOKEN` and optional `APIFY_ACTOR_ID`.
- Files and outputs:
  - `actor_output.json` (latest scrape + fallback listings).
  - `lease_drafts/` (generated text/PDF drafts).
  - Knowledge aids: `Lease_Knowledge_Base.md` and `clean_structured_lease_summary.md` summarize the sample lease used for scaffolding.
- Constants: `DEFAULT_MAX_PAGES`=1 (capped 5), `DEFAULT_FILTER_OPTION`=`all`, `THUMBNAIL_WIDTH_PX`=280 (for inline image sizing), `MAX_DEFAULT_MOVE_IN_DATE`=2026-03-31.

## Data Handling and Safety Notes
- Avoids inventing listing details; if scraping fails, uses cached listings with notices.
- Enforces US rentals-only scope; off-topic requests receive polite refusal.
- Date and term inputs validated against inferred availability; lease drafting blocks without tenant name and required dates/terms.
- Caches scraped data to disk and reuses when preferences unchanged to reduce actor calls.
