import json
import copy
import pytest

from system2 import real_estate_agent as s2_agent
from system2 import session as s2_session


def _load_fixture(name: str):
    with open(f"tests/fixtures/{name}", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def system2_listings():
    return _load_fixture("system2_listings_subset.json")


@pytest.fixture(autouse=True)
def patch_system2(monkeypatch, system2_listings):
    """Stub network/LLM dependencies so conversations stay local."""

    def fake_generate_persona_reply(state, intent, listing_summaries=None, notes=None, focused_listing=None):
        listings = listing_summaries or []
        titles = [item.get("title") or item.get("about", {}).get("title") for item in listings]
        prefix = "Stub results ready" if intent == "results" else "Follow-up ready"
        top = f" Top: {titles[0]}" if titles else ""
        return f"{prefix}.{top}", "test-persona", False

    def fake_analyze_preferences(state):
        user_msg = (state["messages"][-1]["content"] or "").lower()
        prefs = {
            "city": "Tripoli",
            "area": "Dam w Farez",
            "max_rent": 900,
            "min_beds": 2,
        }
        updated = "show" in user_msg or "find" in user_msg
        return {
            **state,
            "preferences": prefs,
            "preferences_updated": updated,
            "clarifying_questions": [],
            "need_more_info": False,
            "off_topic": False,
        }

    def fake_scrape_listings(state):
        listings_copy = copy.deepcopy(system2_listings)
        return {**state, "listings": listings_copy, "scraped_listings": listings_copy}

    def fake_run_scheduler_turn(self, user_input: str, stream_handler=None):
        listing = (self.state.get("ranked_listings") or self.state.get("listings") or [])
        first = listing[0] if listing else {}
        title = (first.get("about", {}) if isinstance(first, dict) else {}).get("title", "listing")
        self.state["scheduling"] = {"listing": title, "status": "booked"}
        self.state["focused_listing"] = first if isinstance(first, dict) else {}
        return f"Tour scheduled for {title} on Friday at 10:00."

    def fake_handle_lease_command(state, user_input: str):
        if "lease" not in user_input.lower():
            return None
        listings = state.get("ranked_listings") or state.get("listings") or []
        title = (listings[0].get("about", {}) if listings else {}).get("title", "listing")
        draft_text = f"Lease draft for {title}."
        state.setdefault("lease_drafts", []).append({"title": title, "text": draft_text})
        state["last_choice_index"] = 0
        state["pending_lease_choice"] = 0
        return s2_agent._reply_with_history(state, user_input, draft_text)

    monkeypatch.setattr(s2_agent, "generate_persona_reply", fake_generate_persona_reply)
    monkeypatch.setattr(s2_agent, "analyze_preferences", fake_analyze_preferences)
    monkeypatch.setattr(s2_agent, "scrape_listings", fake_scrape_listings)
    monkeypatch.setattr(s2_agent.System2Agent, "_run_scheduler_turn", fake_run_scheduler_turn)
    monkeypatch.setattr(s2_agent, "handle_lease_command", fake_handle_lease_command)
    monkeypatch.setattr(s2_agent, "_load_scraped_output", lambda state, force_reload=False: copy.deepcopy(system2_listings))
    monkeypatch.setattr(s2_agent, "is_real_estate_related", lambda text: True)


def test_system2_scheduling_and_lease(monkeypatch):
    session = s2_session.System2AgentSession()

    first = session.send("Show Tripoli options I can rent with 2 bedrooms and parking.")
    assert "Stub results" in first["reply"]
    assert session.state.get("ranked_listings") or session.state.get("listings")

    schedule = session.send("Can you schedule a tour for the first place this Friday?")
    assert "Tour scheduled" in schedule["reply"]
    assert session.state.get("scheduling", {}).get("status") == "booked"

    lease = session.send("Draft a lease for option 1 under Jane Doe.")
    assert "Lease draft" in lease["reply"]
    assert session.state.get("lease_drafts")
