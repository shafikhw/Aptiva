"""Placeholder session class for System 2 (Lebanon)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable

NOT_IMPLEMENTED_REPLY = (
    "System 2 (Lebanon) isn't implemented yet. "
    "Please switch to the US system or check back later for the Lebanon experience."
)


class System2AgentSession:
    """Simple stub session that just returns a static reply."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "messages": [],
            "persona_mode": "default",
        }

    def send(self, text: str, stream_handler: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        message = (text or "").strip()
        if not message:
            raise ValueError("Message must not be empty.")
        self.state.setdefault("messages", []).append({"role": "user", "content": message})
        self.state["messages"].append({"role": "assistant", "content": NOT_IMPLEMENTED_REPLY})
        if stream_handler:
            try:
                stream_handler(NOT_IMPLEMENTED_REPLY)
            except Exception:
                pass
        return {
            "reply": NOT_IMPLEMENTED_REPLY,
            "state": self.state,
            "preferences": {},
            "conversation_complete": True,
        }
