# Aptiva – Full System Report

## Overview
- Dual-market, LangGraph-powered rental assistant with two personas-aware agents: **System 1 (US)** and **System 2 (Lebanon)**. Both handle discovery, ranking, follow-ups, scheduling, and lease drafting; delivery via CLI, FastAPI + Supabase backend, and a web UI with streaming.
- OpenAI chat + reasoning models (`gpt-4o-mini` defaults, `o4-mini` for reasoning), optional Google Maps enrichment, persistent Supabase storage for users/sessions/messages/lease drafts, and Zapier MCP (Google Calendar) for scheduling.
- Strong guardrails keep conversations in scope (US residential rentals for System 1; Lebanese Masharee3 Trablos inventory for System 2). Replies are persona-styled, single-listing-focused, avoid fabrication, and surface missing data transparently.
- Outputs: markdown with clickable titles/inline thumbnails, cached listings (`actor_output.json` / `all_listings.json`), PDF/text leases, stored drafts via Supabase, and streaming SSE for the web app (`SCRAPE_SIGNAL` event during scrapes).

## System 1 – United States Real Estate Agent (`system1`)
### Purpose & Guardrails
- US-only residential rentals (search/compare/viewings/leases). Domain guardrail system prompt blocks off-topic requests; refusal message for non-housing queries. Real-estate detection uses keyword lists + regex (beds/baths, ZIPs, “City, ST”) + off-topic filters (cars/equipment rentals, commercial, travel).
- Personas: **Neighborhood Naturalist** (vibe, temp 0.9), **Data Whisperer** (metrics, temp 0.3), **Deal Navigator** (trade-offs, temp 0.5), **Auto** heuristic chooser. Persona commands supported; safety instructions injected into every LLM call.

### Workflow (LangGraph)
- Nodes: `analyze_preferences` → (clarify | build_query | answer_existing) → `scrape` → `enrich` → `rank_and_format`; edges handle off-topic, missing info, cached reuse, and errors.
- State (`AgentState`) tracks messages, preferences, search query/URL, scraped/enriched/ranked listings, listing lookup, focused listing, notices, persona mode, lease workflow fields, and reply metadata.

### Preference Extraction & Query Building
- `analyze_preferences`: LLM extracts structured prefs (city/state/location/near_me, property type, lifestyle, rooms_for_rent, rent/beds/baths ranges, pets, cheap/utilities-included, amenities, FRBO, keyword, pagination, tenant name). Normalizes city/state (`_normalize_city_state`), merges with prior prefs, emits clarifying questions when city/state or budget/beds/baths are missing.
- `build_query_from_preferences` creates `ApartmentsSearchQuery` (numeric/enum normalization). `compute_missing_preferences` gates search until essentials provided.

### URL Construction (`url_complex.py`)
- `ApartmentsSearchQuery.build_url` builds Apartments.com URLs: location slugs, property type prefixes (houses/condos/townhomes/lofts), lifestyle pages (student/senior/corporate/military/short-term), special pages (rooms-for-rent, cheap, utilities-included, amenity-only slugs), combined numeric slug (beds/baths/price/pet), near-me paths, and pagination.

### Scraping & Caching (`scraper.py`)
- Apify actor (default `BvepfniODI2AixuNN`) with `APIFY_API_TOKEN`/`APIFY_TOKEN`. Inputs: `search_url`, `max_pages` (default 1, capped 5), `filter_option`.
- `run_actor_and_save_outputs` writes JSON to `actor_output.json`. `scrape_listings` skips re-scrape when prefs unchanged, streams `SCRAPE_SIGNAL`, loads actor output, and falls back to cached `actor_output.json` with transparency notices if scrape fails/empty.

### Enrichment, Ranking, Rendering
- Optional Google Maps enrichment (`GOOGLE_MAPS_API_KEY`): geocodes addresses/titles/URLs and fetches nearby POIs (gyms, schools, parks, shopping, hospitals, beaches) within 2 km.
- Ranking: budget alignment, beds fit, pet hints; selects top 5. `_render_listings_markdown` builds markdown with clickable titles, inline thumbnails (`THUMBNAIL_WIDTH_PX`=280), bullets for location/price/beds/baths/amenities/reasons/POIs.
- Listing lookup maps option numbers/titles/locations/URLs for free-form follow-ups. `answer_with_existing` reuses cached listings when user references known listing without re-scrape.

### Lease Drafting Flow
- Triggers: lease phrases or `maybe_schedule_lease_generation`. Requires city/state + budget or asks for them.
- Multi-step: floor-plan selection (`pricingAndFloorPlans` parsing, per-person detection), unit selection, tenant name parsing/normalization, move-in date parsing/bounds (default max `2026-03-31`), lease term bounds inferred from listing text/fees. Navigation commands (`back`, `restart`, change plan/unit/name/date/term), aliases for plans/units, comparison commands.
- Reasoning assist (`_reason_about_lease_input`, model `o4-mini`) classifies intents (select/change/show/compare/back/restart/property Q&A).
- Lease updates (`handle_lease_update_request`) edit last draft (date/term/rent/name) with validation.
- Simple collection path (`continue_lease_collection`) walks `LEASE_COLLECTION_FIELDS` when detailed options absent.

### Lease Engine (`lease_drafter.py`)
- `infer_inputs` merges prefs + listing (landlord name via contact/logo/URL/title, address, rent) + overrides. `LeaseDraftInputs` holds parties/address/rent/deposit/fees/utilities/pets/plan-unit info/dates/terms/caps/clauses.
- `generate_lease_text` emits structured ASCII lease mirroring sample template (mandatory sections 1–16). `compliance_report` heuristics for CA/NY/MA (deposit cap, late fee cap, pet rent cap), rent vs budget, missing address.
- PDF via ReportLab; saves to `system1/lease_drafts/{tenant}_{id}_lease_draft.txt/.pdf`. Helpers: `render_pdf_from_text`, `text_file_to_pdf`.

### Runtime Interfaces
- CLI (`real_estate_agent.main`): persona prompt, REPL commands (`/persona`, lease commands, refusals), streaming via stdout labels. `System1AgentSession` wraps compiled graph, supports streaming callbacks, persona switches, lease updates, and state snapshots.
- Config/env: `OPENAI_API_KEY` (required), `OPENAI_MODEL` (`gpt-4o-mini` default), `OPENAI_REASONING_MODEL` (`o4-mini`), `GOOGLE_MAPS_API_KEY` optional, Apify tokens/actor id.
- Artifacts: `actor_output.json`, `lease_drafts/`, knowledge aids (`Lease_Knowledge_Base.md`, `clean_structured_lease_summary.md`), template (`residential_lease_template.txt`). Constants: `DEFAULT_MAX_PAGES`=1, `DEFAULT_FILTER_OPTION`=`all`, `THUMBNAIL_WIDTH_PX`=280, `MAX_DEFAULT_MOVE_IN_DATE`=2026-03-31.

## System 2 – Lebanon Real Estate Agent (`system2`)
### Purpose & Guardrails
- Lebanese residential rental assistant for agency **Masharee3 Trablos**. Uses only local JSON inventory (`all_listings.json`); never references US/external markets. Domain guardrail enforces scope and points to agency contact when data missing. Refusal: “I can only help with Masharee3 Trablos rentals in Lebanon…”
- Real-estate detector: expansive keyword list + regex + large off-topic phrase block (vehicles/equipment rentals, hospitality, storage, tools, party rentals, etc.). Always prepends domain guard system message via `_inject_domain_guardrail`.
- Personas same as System 1 with auto selector; persona commands handled by `handle_persona_command`.

### Data & Preference Handling
- Listings loaded from `all_listings.json`; `_load_fallback_listings` reuses same file. Listing template supports `about`, `amenities`, `pricingAndFloorPlans`, `contact`, `location`, etc. Contact footer uses `contact_info.json` (agency name/phone/email/office).
- `analyze_preferences`: LLM extracts Lebanon-focused prefs (city/area/neighborhood/location, property type, min/max rent/beds/baths, furnished, pet_friendly, amenity_slugs, keyword, tenant_name). Computes missing (city/neighborhood and budget or beds/baths) -> clarifying questions. Normalizes city/state, merges prefs, resets listing offset on updates.
- `build_query_from_preferences` normalizes numeric fields; no external URLs. `build_query_node` sets `search_query`; `compute_missing_preferences` gates searches.

### Search, Filtering, Enrichment, Ranking
- `scrape_listings`: loads local listings; filters via `_filter_local_listings` (score from `_score_listing` + city/area/type matching). If none, retries fallback; otherwise replies with no-results + contact footer.
- Optional Maps enrichment identical to System 1 when `GOOGLE_MAPS_API_KEY` set; attaches `nearby_pois` and geocoded coordinates.
- `rank_and_format`: sorts by score, advances `listing_offset` to rotate options when user says “another”. Shows **one listing at a time** with reason tags, amenities preview, features, POIs, and clickable title if URL. Notes include scheduling offer (Mon–Fri 9:00–17:00).
- `_render_listings_markdown` formats output; `listing_lookup` supports free-form references.

### Response Behavior
- `generate_persona_reply` wraps persona system prompt with safety instructions: Lebanon scope, do not invent, use listing data only, show one listing, include rendered markdown verbatim when provided, mention scheduling option. Streaming supported via `stream_chat_completion` (always streamed, stdout label optional, callback via `STREAM_CALLBACK`).

### Scheduling (Viewings)
- Triggered by viewing intent (`schedule/book/view/tour/showing`), calendar mentions, or follow-up while listing is focused. `_infer_requested_window` parses “today/tomorrow/Mon-Fri/next Wed” to 1-day window.
- `_propose_slots` calls `schedular.backend_get_common_slots` to fetch 30-minute weekday slots between 09:00–17:00 local (`LOCAL_TZ_OFFSET_HOURS`=2). Lists numbered slots with labels; handles window expansion when user rejects options. Stores `scheduling` state (`stage`, `slots`, `listing`, `window_label`). If no slots, prompts for manual times + contact footer.
- `_book_slot` uses `schedular.backend_book_tour` (Zapier MCP Google Calendar) to create events in renter and landlord calendars (configurable IDs/emails, default listing title/address). Confirms booking + invites lease drafting. `_list_upcoming_tours` queries calendars; `_wants_new_listing` handles “next listing” rotation.

### Lease Drafting (Lebanon)
- Mirrors System 1 flow with local listings: command detection, plan/unit prompts, aliases, navigation (`back/restart`, change plan/unit/name/date/term), detail/compare commands, refine search commands. Caches plan/unit options and aliases for references.
- `_reason_about_lease_input` (model `o4-mini`) classifies lease intents; `_handle_global_lease_commands` routes to plan/unit/detail/compare/restart/refine handlers before main flow.
- `_generate_lease_for_listing` collects tenant name, plan, unit, move-in date, term bounds (`_extract_lease_duration_bounds`), overrides rent from unit price when present. Builds lease package via `lease_drafter.build_lease_package`; replies with summary + navigation hints. `maybe_schedule_lease_generation` asks for missing lease fields; `continue_lease_collection` walks `LEASE_COLLECTION_FIELDS` when minimal data.
- `handle_lease_update_request` supports text edits to last draft (date/term/rent/name) with validation and regeneration.

### State & Session
- `AgentState` tracks messages, preferences, persona, clarifications, queries, listings (scraped/enriched/ranked), lookup, focused listing, reply metadata, scheduling, lease caches/aliases, last overrides, duration bounds, off-topic flag. `System2AgentSession` wraps `System2Agent`, returns reply, state snapshot, preferences, and completion flag.
- Defaults: `DEFAULT_OPENAI_MODEL`=`gpt-4o-mini`, `DEFAULT_REASONING_MODEL`=`o4-mini`, `THUMBNAIL_WIDTH_PX`=280, `MAX_DEFAULT_MOVE_IN_DATE`=2026-03-31.

## Scheduler Service (`system2/schedular.py`)
- Uses Zapier MCP Google Calendar tools via `fastmcp` HTTP transport. Config: `OPENAI_API_KEY`, `ZAPIER_MCP_URL` (default provided), `ZAPIER_MCP_TOKEN` (optional), `USER_CAL_ID`, `LANDLORD_CAL_ID`, `USER_EMAIL`, `LISTING_TITLE`, `LISTING_ADDRESS`, `LOCAL_TZ_OFFSET_HOURS` (default +2), `OPENAI_MODEL`=`gpt-4o` for chat.
- Tool layer: `get_common_slots` (weekday 09:00–17:00 intersections, default 30-minute slots, up to 10), `book_tour` (creates events in both calendars, optional attendees), `list_tours` (tour-like events next N days), `cancel_tour`.
- Backend logic: `GoogleCalendarMCP` wraps MCP calls (`find_events_as_busy`, `list_tour_events`, `create_event_detailed`, `delete_event`). Busy blocks inverted to free, intersected, and cut into slots with local-time weekday/hour constraints. Slots labeled for user display (pretty local times).
- CLI runner with system prompt summarizing defaults and rules; supports multi-tool turns with OpenAI tool-calling.

## Shared Platform (CLI, Server, Webapp, Storage)
- **CLI router (`cli/homepage.py`, `cli/router.py`, `main.py`)**: menu to pick US vs Lebanon (`select_system_by_location`), then invokes respective agent `run()` (System 1 or System 2 REPLs).
- **FastAPI backend (`server/app.py`)**:
  - Auth: register/login (password hashed via PBKDF2 `server/security.py`), Google auth, guest sessions, logout, me, forgot-password.
  - Preferences CRUD (`/api/preferences`).
  - Conversations: list/create/get (`system` filter), message append on send/stream; persona mode persisted. History replies when user asks for history.
  - Chat: `/api/chat/send` (non-stream) and `/api/chat/stream` (SSE). Streaming uses background thread + queue; emits `status` on `SCRAPE_SIGNAL`, `token` for streamed text, `final` payload.
  - Lease drafts: list/latest/get/download/pdf, generate (`/api/lease/generate` ties to System 1 state, stores PDF/text via Supabase).
  - Serves web assets (`webapp/`) and static files.
- **Storage (`storage/supabase_store.py`)**: wraps Supabase tables with retry/backoff. Stores users, sessions, conversations (with state + preferences + persona), messages, lease drafts (base64 PDFs, metadata). Conversation messages loaded from table when absent from state.
- **Webapp (`webapp/index.html`, `assets/app.js`, `assets/style.css`)**:
  - UI: dark/light toggle, persona picker, system switcher (US/LB), share-preferences toggles, conversation list, chat feed with markdown rendering (inline images from listings), scrape notice banner, streaming bubbles, stop-stream button, lease drafts panel with PDF download, auth modal (login/signup/forgot), guest session button, status bar.
  - Uses fetch for REST + SSE stream consumption from `/api/chat/stream`; shows scrape status inline when `SCRAPE_SIGNAL` arrives. Renders numbered lists and markdown formatting, strips streaming prefixes, and preserves images/links. Downloads lease PDFs via blob.

## Data Assets & Dependencies
- Data/templates: `system1/actor_output.json` (cached US listings), `system2/all_listings.json` (11 local listings), `listing_template.json` (schema example), sample lease PDF (`Apts_Sample_Residential_Lease_Agreement.pdf`), templates (`residential_lease_template.txt`), knowledge bases (`Lease_Knowledge_Base.md`, `clean_structured_lease_summary.md`), `contact_info.json` for Lebanon agency.
- Dependencies (`requirements.txt`): FastAPI, uvicorn, httpx, Supabase client, OpenAI, LangGraph, Apify client, python-dotenv, pandas, googlemaps, reportlab, fastmcp.

## User–LLM Interaction & Guardrails
- Guardrail system messages always prepended; streaming wrapper enforces injection. Off-topic detection short-circuits with polite refusal and sets `off_topic` flag.
- Responses: single-listing focus, concise persona tone, clickable titles, inline thumbnails, bullets for key facts, reason tags, POIs; offer next listing or preference tweak. Lebanon replies remind scheduling; US replies mention lease drafting and follow-ups.
- Clarifications: missing required prefs prompt targeted questions. Cached listing reuse avoids unnecessary scrapes; notices explain when cached/fallback data used.
- Scheduling guardrails: weekdays 09:00–17:00 local, 30-minute slots, explicit labels; confirms date interpretation and booking, exposes cancel/list operations.
- Lease guardrails: enforce move-in date/term/tenant name before drafting; validate bounds; compliance heuristics flag risky terms; navigation commands prevent dead-ends.

## Environment & Artifacts
- Required env: `OPENAI_API_KEY`; Supabase URL/service key for server; Apify tokens (System 1); optional `GOOGLE_MAPS_API_KEY`; scheduler/env vars for calendars; `OPENAI_MODEL`/`OPENAI_REASONING_MODEL` overrides; `APIFY_ACTOR_ID`, `DEFAULT_MAX_PAGES`, `DEFAULT_FILTER_OPTION`. Outputs: lease drafts folder, cached listing JSONs, Supabase records, SSE chat streams.
