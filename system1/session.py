"""Web-friendly session wrapper around the System 1 LangGraph agent."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Dict, Optional, Callable, Any

from .real_estate_agent import (
    AgentState,
    DEFAULT_PERSONA_MODE,
    OFF_TOPIC_REFUSAL,
    build_graph,
    handle_persona_command,
    handle_lease_command,
    handle_lease_update_request,
    is_real_estate_related,
    _normalize_persona_mode,
    maybe_schedule_lease_generation,
    continue_lease_collection,
    STREAM_CALLBACK,
)


def _snapshot(state: AgentState) -> AgentState:
    """Return a JSON-safe deep copy of the agent state."""
    return json.loads(json.dumps(state))


_GRAPH_CACHE = None


def _get_compiled_graph():
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = build_graph()
    return _GRAPH_CACHE


class System1AgentSession:
    """Owns a LangGraph app instance plus mutable AgentState."""

    def __init__(self, persona_mode: Optional[str] = None, state: Optional[AgentState] = None) -> None:
        self._app = _get_compiled_graph()
        if state:
            restored: AgentState = deepcopy(state)
            restored.setdefault("messages", [])
            restored.setdefault("preferences", {})
            restored.setdefault("persona_mode", _normalize_persona_mode(persona_mode) if persona_mode else restored.get("persona_mode", DEFAULT_PERSONA_MODE))
            restored.setdefault("preferences_updated", False)
            self.state = restored
        else:
            self.state = {
                "messages": [],
                "preferences": {},
                "persona_mode": _normalize_persona_mode(persona_mode) if persona_mode else DEFAULT_PERSONA_MODE,
                "preferences_updated": False,
            }

    def send(self, text: str, stream_handler: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
        """Process a single user message and return the assistant reply plus new state."""
        user_input = (text or "").strip()
        if not user_input:
            raise ValueError("Message must not be empty.")

        # Allow persona switches without invoking the full graph.
        command_reply = handle_persona_command(self.state, user_input)
        if command_reply:
            snapshot = _snapshot(self.state)
            return {
                "reply": command_reply,
                "state": snapshot,
                "preferences": snapshot.get("preferences", {}),
                "conversation_complete": False,
            }

        if not is_real_estate_related(user_input):
            msgs = self.state.get("messages") or []
            msgs.append({"role": "user", "content": user_input})
            msgs.append({"role": "assistant", "content": OFF_TOPIC_REFUSAL})
            self.state["messages"] = msgs
            self.state["off_topic"] = True
            snapshot = _snapshot(self.state)
            return {
                "reply": OFF_TOPIC_REFUSAL,
                "state": snapshot,
                "preferences": snapshot.get("preferences", {}),
                "conversation_complete": False,
            }

        state_messages = self.state.get("messages") or []
        state_messages.append({"role": "user", "content": user_input})
        self.state["messages"] = state_messages
        self.state["reply_streamed"] = False
        self.state["off_topic"] = False

        if self.state.get("lease_collection"):
            follow_up = continue_lease_collection(self.state, user_input)
            if follow_up:
                self.state.setdefault("messages", []).append({"role": "assistant", "content": follow_up})
                snapshot = _snapshot(self.state)
                return {
                    "reply": follow_up,
                    "state": snapshot,
                    "preferences": snapshot.get("preferences", {}),
                "conversation_complete": False,
            }

        update_reply = handle_lease_update_request(self.state, user_input)
        if update_reply:
            msgs = self.state.get("messages") or []
            msgs.append({"role": "assistant", "content": update_reply})
            self.state["messages"] = msgs
            snapshot = _snapshot(self.state)
            return {
                "reply": update_reply,
                "state": snapshot,
                "preferences": snapshot.get("preferences", {}),
                "conversation_complete": False,
            }

        lease_cmd_reply = handle_lease_command(self.state, user_input)
        if lease_cmd_reply:
            snapshot = _snapshot(self.state)
            return {
                "reply": lease_cmd_reply,
                "state": snapshot,
                "preferences": snapshot.get("preferences", {}),
                "conversation_complete": False,
            }

        lease_reply, _ = maybe_schedule_lease_generation(self.state, user_input)
        if lease_reply:
            msgs = self.state.get("messages") or []
            msgs.append({"role": "assistant", "content": lease_reply})
            self.state["messages"] = msgs
            snapshot = _snapshot(self.state)
            return {
                "reply": lease_reply,
                "state": snapshot,
                "preferences": snapshot.get("preferences", {}),
                "conversation_complete": False,
            }

        token = None
        if stream_handler:
            token = STREAM_CALLBACK.set(stream_handler)
        try:
            self.state = self._app.invoke(self.state)
        finally:
            if token is not None:
                STREAM_CALLBACK.reset(token)

        reply = self.state.get("reply") or "I didn't catch that—could you rephrase?"
        self.state.setdefault("messages", []).append({"role": "assistant", "content": reply})

        snapshot = _snapshot(self.state)
        return {
            "reply": reply,
            "state": snapshot,
            "preferences": snapshot.get("preferences", {}),
            "conversation_complete": False,
        }
