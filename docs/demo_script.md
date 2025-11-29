# Live Demo Script

Follow these steps to showcase Aptiva end-to-end with both agents, scheduling, leases, and metrics.

## 1) Start server + webapp
1. In a terminal (venv active, env vars set), run:
   ```bash
   uvicorn server.app:app --reload --port 8000
   ```
2. Open the web UI at http://localhost:8000 to confirm assets load.

## 2) Create guest user (or quick login)
- If the web UI supports guest sessions, click **Continue as guest**.
- Otherwise, hit the API directly:
  ```bash
  curl -X POST http://localhost:8000/api/auth/register \
       -H "Content-Type: application/json" \
       -d '{"email":"demo@example.com","password":"demo123","first_name":"Demo","last_name":"User","username":"demo"}'
  ```
  Copy the returned `token` for authenticated calls, or just use the web login form.

## 3) System 1 (US) flow — search → refine → lease
1. In the web app, select **United States** (System 1) and choose a persona (or Auto).
2. Ask for a search: “Find 2br apartments in Austin under $2400 with in-unit laundry.”
3. Refine: “Show a cheaper option” or “What’s near parks?” to trigger Maps enrichment if configured.
4. Lease draft: “Draft a lease for option 1 starting March 1 for 12 months.” Confirm plan/unit/name/date if prompted; show the generated lease summary and PDF download.

## 4) System 2 (Lebanon) flow — local listing → schedule tour → lease
1. Switch to **Lebanon** (System 2).
2. Ask: “I want a 2br in Tripoli near the corniche.”
3. When a listing appears, request scheduling: “Book a tour next Wednesday afternoon.” Demonstrate:
   - Agent proposes weekday 09:00–17:00 slots.
   - Choose a slot number; confirm booking.
4. Lease: “Generate a lease for this listing starting May 15 for 1 year.” Walk through plan/unit/name/date prompts and show the draft summary.

## 5) Cost/latency dashboard & metrics
1. Tail the local cost log:
   ```bash
   tail -n 10 metrics/cost_log.csv
   ```
2. If Supabase is configured, show the metrics endpoint in a notebook or curl:
   ```bash
   curl http://localhost:8000/api/metrics
   ```
3. Mention `metrics/dashboard.md` for visualization ideas (load the CSV or Supabase `metrics` table into a quick chart). The committed sample `metrics/cost_log.csv` can seed dashboards before running live traffic.

Tips: Keep both agent tabs open; if Apify/Maps/Zapier keys are missing, the agents gracefully fall back and explain why. Use the CLI (`python -m cli.homepage`) if you want to demo terminal streaming.***
