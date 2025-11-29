# Integration Test Coverage

- search: `tests/test_system1_integration.py::test_system1_scripted_conversation` and `tests/test_system2_scheduling_and_lease` drive listings from local fixtures (apartments_subset.json and system2_listings_subset.json).
- refine: System 1 history/recap branch exercises follow-up behavior using cached listings without rebuilding preferences.
- schedule: `tests/test_system2_scheduling_and_lease` triggers the mocked scheduler path for a held-out local listing.
- lease: Both integration tests request lease drafts (System 1 on Apartments fixture, System 2 on local subset).

Held-out data map:
- Apartments fixture: `tests/fixtures/apartments_subset.json` referenced by `us-heldout-01` in `eval/tasks_us.json`.
- Local listings fixture: `tests/fixtures/system2_listings_subset.json` referenced by `lb-heldout-01` in `eval/tasks_lb.json`.
