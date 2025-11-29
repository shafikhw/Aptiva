# Pilot Notes

Quick observations from dry-run pilots to guide future iterations.

- **US renter wants cheap 2br near transit (Austin).**
  - Behavior: System 1 asks for city/budget if missing; with Apify + Maps enabled it returns 3–5 listings with POIs. Lease flow triggers cleanly after “draft a lease for option 2”.
  - Friction: When Maps key is absent, POI bullets become generic; users appreciate the fallback notice but still ask for nearby parks—consider a clearer “maps unavailable” banner.

- **US renter negotiates term and rent.**
  - Behavior: Auto persona picks “deal”; responds with trade-offs and suggests shifting move-in or beds to hit price. Lease draft validates term bounds and asks for tenant name/date before drafting.
  - Friction: If user says “shorter than 6 months”, model sometimes proposes 5 months but warns about standard 12-month expectations. Might add explicit short-term disclaimer.

- **Lebanon renter wants Tripoli seafront unit and tour.**
  - Behavior: System 2 filters local JSON, presents one listing at a time, then calls scheduler for 30-minute weekday slots. Booking confirms both calendars and offers lease drafting.
  - Friction: When Zapier MCP is unreachable, fallback to manual times works but users want a clearer CTA to “text the agent” — consider injecting contact_info footer earlier.

- **Lebanon renter asks off-topic (car rental).**
  - Behavior: Domain guardrail refuses and redirects to residential rentals; off-topic flag set in state.
  - Friction: None; refusal is concise and polite.

- **Bias probe (family vs single).**
  - Behavior: Responses stayed neutral; listings unchanged. No observed discriminatory phrasing.
  - Friction: Limited probe set; expand bias tasks for additional demographics and Arabic prompts.
