from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from server.security import hash_password


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryStore:
    """Demo-mode store used when Supabase is unavailable."""

    def __init__(self) -> None:
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.lease_drafts: Dict[str, Dict[str, Any]] = {}

        # Seed a guest user for convenience.
        guest_id = str(uuid.uuid4())
        guest = {
            "id": guest_id,
            "email": "guest@aptiva.local",
            "username": "guest",
            "first_name": "Guest",
            "last_name": "Demo",
            "password_hash": hash_password("guest"),
            "auth_provider": "guest",
            "preferences": {},
            "created_at": _now_iso(),
        }
        self.users[guest_id] = guest

    # User/session methods -------------------------------------------------
    def register_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_id = payload.get("id") or str(uuid.uuid4())
        user = {
            "id": user_id,
            "email": payload["email"],
            "username": payload["username"],
            "first_name": payload.get("first_name"),
            "last_name": payload.get("last_name"),
            "password_hash": payload["password_hash"],
            "auth_provider": payload.get("auth_provider", "password"),
            "preferences": payload.get("preferences") or {},
            "created_at": _now_iso(),
        }
        self.users[user_id] = user
        return user

    def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        for user in self.users.values():
            if user["email"].lower() == email.lower():
                return user
        return None

    def find_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        for user in self.users.values():
            if user["username"].lower() == username.lower():
                return user
        return None

    def find_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.users.get(user_id)

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        if user_id in self.users:
            self.users[user_id]["preferences"] = preferences

    def update_user_password(self, user_id: str, password_hash: str) -> None:
        if user_id in self.users:
            self.users[user_id]["password_hash"] = password_hash

    def create_session(self, user_id: str) -> Dict[str, Any]:
        token = str(uuid.uuid4())
        session = {"token": token, "user_id": user_id}
        self.sessions[token] = session
        return session

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(token)

    def revoke_session(self, token: str) -> None:
        self.sessions.pop(token, None)

    # Conversations/messages -----------------------------------------------
    def create_conversation(
        self,
        user_id: str,
        system: str,
        persona_mode: str,
        *,
        initial_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        convo_id = str(uuid.uuid4())
        convo = {
            "id": convo_id,
            "user_id": user_id,
            "system": system,
            "persona_mode": persona_mode,
            "preferences": initial_preferences or {},
            "state": {},
            "messages": [],
            "updated_at": _now_iso(),
            "created_at": _now_iso(),
        }
        self.conversations[convo_id] = convo
        self.messages[convo_id] = []
        return convo

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        convo = self.conversations.get(conversation_id)
        if convo and convo["user_id"] == user_id:
            convo = {**convo}
            convo["messages"] = list(self.messages.get(conversation_id, []))
            return convo
        return None

    def list_conversations(
        self, user_id: str, *, system: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        convos = [
            c
            for c in self.conversations.values()
            if c["user_id"] == user_id and (system is None or c["system"] == system)
        ]
        convos.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
        if limit:
            convos = convos[:limit]
        return [{k: v for k, v in c.items() if k != "messages"} for c in convos]

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        *,
        role: str,
        content: str,
        state_snapshot: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        persona_mode: Optional[str] = None,
    ) -> None:
        convo = self.conversations.get(conversation_id)
        if not convo or convo["user_id"] != user_id:
            return
        self.messages.setdefault(conversation_id, []).append(
            {"role": role, "content": content, "timestamp": _now_iso()}
        )
        update_payload: Dict[str, Any] = {"updated_at": _now_iso()}
        if state_snapshot is not None:
            update_payload["state"] = state_snapshot
        if preferences is not None:
            update_payload["preferences"] = preferences
        if persona_mode:
            update_payload["persona_mode"] = persona_mode
        convo.update(update_payload)

    # Lease drafts ---------------------------------------------------------
    def save_lease_draft(
        self,
        *,
        user_id: str,
        conversation_id: Optional[str],
        title: str,
        pdf_base64: str,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        draft_id = str(uuid.uuid4())
        draft = {
            "id": draft_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "title": title,
            "pdf_base64": pdf_base64,
            "summary": summary,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        self.lease_drafts[draft_id] = draft
        return draft

    def list_lease_drafts(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        drafts = [d for d in self.lease_drafts.values() if d["user_id"] == user_id]
        drafts.sort(key=lambda d: d.get("created_at") or 0, reverse=True)
        return drafts[:limit] if limit else drafts

    def get_lease_draft(self, draft_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        draft = self.lease_drafts.get(draft_id)
        if draft and draft["user_id"] == user_id:
            return draft
        return None

    def get_latest_lease_draft(self, user_id: str) -> Optional[Dict[str, Any]]:
        drafts = self.list_lease_drafts(user_id, limit=1)
        return drafts[0] if drafts else None

    # Health ---------------------------------------------------------------
    def ping(self) -> bool:
        return True
