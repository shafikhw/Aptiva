import json
import copy
import pytest

from system1 import real_estate_agent as s1_agent
from system1 import session as s1_session


def _load_fixture(name: str):
    with open(f"tests/fixtures/{name}", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def system1_listings():
    return _load_fixture("apartments_subset.json")


@pytest.fixture(autouse=True)
def patch_system1(monkeypatch, system1_listings):
    """Stub out network/LLM pieces so tests only use local fixtures."""

    def fake_generate_persona_reply(state, intent, listing_summaries=None, notes=None, focused_listing=None):
        listings = listing_summaries or []
        titles = [item.get("title") or item.get("about", {}).get("title") for item in listings]
        reply_prefix = {
            "results": "Stub results ready",
            "follow_up": "Recap from cached history",
        }.get(intent, "Stub response")
        top = f" Top: {titles[0]}" if titles else ""
        return f"{reply_prefix}.{top}", "test-persona", False

    def fake_analyze_preferences(state):
        user_msg = (state["messages"][-1]["content"] or "").lower()
        prefs = {
            "city": "Austin",
            "state": "TX",
            "max_rent": 2500,
            "min_beds": 2,
        }
        # Treat follow-ups (like history recall) as using existing preferences.
        updated = "find" in user_msg or "listing" in user_msg
        return {
            **state,
            "preferences": prefs,
            "preferences_updated": updated,
            "clarifying_questions": [],
            "need_more_info": False,
            "off_topic": False,
        }

    def fake_scrape_listings(state):
        listings_copy = copy.deepcopy(system1_listings)
        return {**state, "listings": listings_copy, "scraped_listings": listings_copy}

    def fake_handle_lease_command(state, user_input: str):
        if "lease" not in user_input.lower():
            return None
        listings = state.get("ranked_listings") or state.get("listings") or []
        title = (listings[0].get("about", {}) if listings else {}).get("title", "listing")
        draft_text = f"Lease draft for {title}."
        state.setdefault("lease_drafts", []).append({"title": title, "text": draft_text})
        state["last_choice_index"] = 0
        state["pending_lease_choice"] = 0
        return s1_agent._reply_with_history(state, user_input, draft_text)

    monkeypatch.setattr(s1_agent, "generate_persona_reply", fake_generate_persona_reply)
    monkeypatch.setattr(s1_agent, "analyze_preferences", fake_analyze_preferences)
    monkeypatch.setattr(s1_agent, "scrape_listings", fake_scrape_listings)
    monkeypatch.setattr(s1_agent, "handle_lease_command", fake_handle_lease_command)
    monkeypatch.setattr(s1_agent, "_load_scraped_output", lambda state, force_reload=False: copy.deepcopy(system1_listings))
    monkeypatch.setattr(s1_agent, "is_real_estate_related", lambda text: True)
    s1_session._GRAPH_CACHE = None
    yield
    s1_session._GRAPH_CACHE = None


def test_system1_scripted_conversation(monkeypatch):
    session = s1_session.System1AgentSession()

    first = session.send("Find me a 2 bedroom place in Austin under $2500.")
    assert "Stub results" in first["reply"]
    assert session.state.get("ranked_listings")

    lease = session.send("Please draft a lease for option 1 with my info.")
    assert "Lease draft" in lease["reply"]
    assert session.state.get("lease_drafts")

    recap = session.send("Can you recap our history so far?")
    assert "Recap" in recap["reply"] or "Stub" in recap["reply"]
    # User and assistant turns for three exchanges -> at least 6 messages stored.
    assert len(session.state.get("messages", [])) >= 6
