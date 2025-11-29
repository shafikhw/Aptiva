# Aptiva Report

## Problem & Target Users
- **Problem:** Renters struggle to find, compare, and act on apartment options quickly; leasing teams need structured intake and fewer back-and-forth messages.
- **Target users:** Apartment seekers in the US (apartments.com inventory) and Lebanon (Masharee3 Trablos listings), plus leasing agents who want faster scheduling and draft leases.

## Task & Success Criteria
- **Core tasks:** Capture preferences, surface matching listings, answer follow-ups, schedule tours, and generate draft leases.
- **Success criteria:** Relevant listings with reasons, clear scheduling flow (weekday 09:00–17:00 slots), lease drafts only when required inputs are present, fast responses, transparent fallbacks when data/tools are unavailable.

## Architecture (text map)
- **Interfaces:** CLI (`cli/homepage.py`), FastAPI server (`server/app.py`) serving API + web app (`webapp/`), scheduler CLI (`system2/schedular.py`).
- **Agents:** US agent (`system1/real_estate_agent.py`, `system1/session.py`) and Lebanon agent (`system2/real_estate_agent.py`, `system2/session.py`).
- **Data/Tools:** Apify scraping (System 1), Google Maps enrichment (optional), Zapier MCP Google Calendar tools (scheduler), local listings (`system2/all_listings.json`), lease drafter (`system1/lease_drafter.py`, `system2/lease_drafter.py`).
- **Storage:** Supabase store with in-memory fallback (`storage/supabase_store.py`, `storage/memory_store.py`).
- **Telemetry:** Metrics and cost logging (`telemetry/metrics.py`, `metrics/cost_log.csv`), logging utils (`telemetry/logging_utils.py`).
- **Flow:** Clients → FastAPI/CLI → agent state machines → tools (Apify/Maps/MCP) → telemetry + storage.

## Evaluation Plan & Key Results
- **Plan:** Run scripted tasks via `eval/run_eval.py` with task suites in `eval/tasks_us.json`, `eval/tasks_lb.json`, and bias probes in `eval/bias_checks.json`. Baselines defined in `eval/baselines.md`.
- **Outputs:** JSON results at `eval/results.json`; usage/cost rows appended to `metrics/cost_log.csv`; qualitative findings in `docs/experiments.md`.
- **Key results (recent runs):** Tool-enabled “full-agent” runs outperform LLM-only on viewing/lease requests; scheduler robustness depends on Zapier MCP availability; Maps enrichment improves grounding when API key is present. Bias probes monitored but require expansion for broader locales.

## Safety & Ethics
- Domain guardrails refuse off-topic or non-rental queries; Lebanon agent blocks commercial/vehicle/hospitality intents explicitly.
- Prompt-injection detection in sessions (`system1/session.py`, `system2/session.py`) adds reminders and flags state.
- Lease replies include “not legal/financial advice” language; scheduling constrained to working hours with explicit confirmations.
- PII handling limited to user-provided contact names/emails for scheduling; Supabase credentials required for persistence—secure env vars and least-privilege keys.
- Logging: metrics capture token counts and costs; avoid logging sensitive user content beyond what the agent state requires.

## Roadmap & Limitations
- **Roadmap:** 
  - Harden scheduler fallbacks with cached slots/offline queueing.
  - Expand eval coverage (more bias probes, Arabic prompts for Lebanon).
  - Add availability freshness checks and stale-data notices per listing.
  - Optional guardrails for stricter PII redaction in logs.
- **Limitations:** 
  - System 1 depends on Apartments.com/Apify availability; cached data may be stale.
  - Scheduling requires Zapier MCP + Google Calendar connectivity; degrades to manual coordination otherwise.
  - Lease drafts are templates and require human legal review. 
