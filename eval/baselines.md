## Evaluation Baselines

These baselines clarify which stack is being exercised for each system during task runs.

- **System 1 (US) – LLM-only, no tools:** Instantiate `System1AgentSession` but stub out scraping/Google calls (or run with empty API keys). Useful for fast sanity checks of dialogue and preference handling without external requests.
- **System 1 (US) – Aptiva full agent:** Default `System1AgentSession` with Apify scraping, Google Maps enrichment, and lease drafter enabled. Measures end-to-end listing retrieval, ranking, and lease drafting.
- **System 2 (Lebanon) – LLM-only, local JSON only:** Default `System2AgentSession` that reads from `all_listings.json` and `contact_info.json` without external tools. Exercises local ranking, scheduling logic, and lease draft prompts.
- **System 2 (Lebanon) – Tool-constrained:** Same as above but with viewing scheduling disabled or mocked to test robustness when calendar access is unavailable.
