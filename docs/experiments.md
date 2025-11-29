# Experiments

This file captures how we ran and compared Aptiva variants, which knobs we turned, and where to find metrics/cost data.

## What We Varied
- **Models:** `OPENAI_MODEL` (`gpt-4o-mini` default) vs larger `gpt-4o`; `OPENAI_REASONING_MODEL` (`o4-mini`) for lease reasoning turns.
- **Prompts:** Persona system prompts (naturalist/data/deal) and the domain guardrail injected by `_inject_domain_guardrail` for both systems.
- **Tools:** Apify scraping (System 1), Google Maps enrichment, Zapier MCP calendar calls (scheduler), Supabase persistence vs in-memory fallback, and the lease drafter.
- **Task sets:** US and Lebanon eval suites (`eval/tasks_us.json`, `eval/tasks_lb.json`) plus bias probes (`eval/bias_checks.json`).

## Configs → Evaluation Runs
Use `python -m eval.run_eval --baseline <label> [--system us|lb]` to tag runs. Recommended labels:

| Baseline label | Systems | Tools enabled | Command example | Notes |
| --- | --- | --- | --- | --- |
| `full-agent` | US + LB | Apify (US), Maps (if key), scheduler (Zapier), lease drafter | `python -m eval.run_eval --baseline full-agent` | End-to-end stack; writes results to `eval/results.json` and logs usage to `metrics/cost_log.csv`. |
| `llm-only` | US + LB | No external tools (omit API keys) | `OPENAI_API_KEY=... python -m eval.run_eval --baseline llm-only` | Fast dialogue sanity check; skips scraping/scheduling when keys are absent. |
| `tool-constrained` | LB | Local JSON only; scheduler mocked/offline | `python -m eval.run_eval --baseline tool-constrained --system lb` | Exercises ranking + lease flow without calendar access. |
| `full-agent-us` | US | Apify + Maps + lease drafter | `python -m eval.run_eval --baseline full-agent-us --system us` | Focused on US routing and lease drafting. |
| `full-agent-lb` | LB | Local JSON + scheduler | `python -m eval.run_eval --baseline full-agent-lb --system lb` | Focused on Lebanon inventory and scheduling. |

Each run appends usage/cost rows to `metrics/cost_log.csv` (sample committed). Supabase logging kicks in automatically when `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set.

## Summaries of Key Findings
- **Full agent vs LLM-only:** Tooling materially improves task coverage: scraping yields real listings for US queries and MCP scheduling fulfills viewing requests, while LLM-only runs tend to apologize or stay generic.
- **Prompt/Persona choices:** Auto persona selection performs well for mixed intents; forcing `deal` helps when users negotiate price/terms but can be wordier.
- **Maps enrichment:** When `GOOGLE_MAPS_API_KEY` is set, POI augmentation reduces off-topic answers by anchoring replies to real locations; without it, replies include transparency that POIs are unavailable.
- **Scheduler robustness:** With Zapier MCP reachable, `_propose_slots` surfaces weekday 09:00–17:00 windows and `_book_slot` succeeds; when unreachable, the agent falls back to manual time collection while logging `scheduler_unavailable`.
- **Cost/latency:** Short, single-turn tasks stay in the sub-cent range on `gpt-4o-mini`; leasing flows and scheduler tool calls add modest token usage, visible in `metrics/cost_log.csv` and easy to chart via `metrics/dashboard.md`.

## Mapping Results to Configs
- `eval/results.json` keeps the latest evaluation output; its `run_id` matches the `baseline` label and timestamp. The cost_log rows share that context via `component` (e.g., `system1`, `system2`, `scheduler`).
- For dashboards, join on `conversation_id` or timestamp to correlate eval runs, scheduler calls, and lease drafting turns. The committed `metrics/cost_log.csv` contains synthetic rows in the correct format to validate your pipeline before running live traffic.
