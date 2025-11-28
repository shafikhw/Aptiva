"""Session wrapper for the Lebanon (System 2) agent."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from .real_estate_agent import System2Agent


def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe deep copy of the agent state."""
    try:
        return json.loads(json.dumps(state))
    except Exception:
        return deepcopy(state)


class System2AgentSession:
    """Owns a System2Agent instance plus mutable conversational state."""

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        self.agent = System2Agent(state=state or {})

    def send(self, text: str, stream_handler: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Process a single user message and return the assistant reply plus new state."""
        reply = self.agent.handle_message(text, stream_handler=stream_handler)
        snapshot = _snapshot(self.agent.state)
        return {
            "reply": reply,
            "state": snapshot,
            "preferences": snapshot.get("preferences", {}),
            "conversation_complete": snapshot.get("conversation_complete", False),
        }
