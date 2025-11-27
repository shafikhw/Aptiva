from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import uuid

import time
import httpx
from postgrest import APIError

from supabase import Client, create_client


class SupabaseStore:
    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)
        self._max_retries = 3
        self._retry_backoff_seconds = 0.25

    def _table(self, name: str):
        return self.client.table(name)

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        delay = self._retry_backoff_seconds
        for attempt in range(self._max_retries):
            try:
                return fn()
            except (httpx.RemoteProtocolError, httpx.WriteError, APIError):
                if attempt >= self._max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def register_user(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user = {
            "id": payload.get("id") or str(uuid.uuid4()),
            "email": payload["email"],
            "username": payload["username"],
            "first_name": payload.get("first_name"),
            "last_name": payload.get("last_name"),
            "password_hash": payload["password_hash"],
            "auth_provider": payload.get("auth_provider", "password"),
            "preferences": payload.get("preferences") or {},
        }
        resp = self._with_retry(lambda: self._table("users").insert(user).execute())
        if not resp.data:
            raise RuntimeError("Failed to insert user")
        return resp.data[0]

    def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        resp = self._with_retry(
            lambda: self._table("users").select("*").eq("email", email.lower()).maybe_single().execute()
        )
        return resp.data

    def find_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        resp = self._with_retry(
            lambda: self._table("users").select("*").eq("username", username.lower()).maybe_single().execute()
        ) or {}
        return resp.get("data")

    def find_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        resp = self._with_retry(lambda: self._table("users").select("*").eq("id", user_id).maybe_single().execute())
        return resp.data

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self._with_retry(lambda: self._table("users").update({"preferences": preferences}).eq("id", user_id).execute())

    def update_user_password(self, user_id: str, password_hash: str) -> None:
        self._with_retry(
            lambda: self._table("users").update({"password_hash": password_hash}).eq("id", user_id).execute()
        )

    def create_session(self, user_id: str) -> Dict[str, Any]:
        token = str(uuid.uuid4())
        resp = self._with_retry(lambda: self._table("sessions").insert({"token": token, "user_id": user_id}).execute())
        if not resp.data:
            raise RuntimeError("Failed to create session")
        return resp.data[0]

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        resp = self._with_retry(
            lambda: self._table("sessions").select("*").eq("token", token).maybe_single().execute()
        )
        return resp.data

    def revoke_session(self, token: str) -> None:
        self._with_retry(lambda: self._table("sessions").delete().eq("token", token).execute())

    def create_conversation(
        self,
        user_id: str,
        system: str,
        persona_mode: str,
        *,
        initial_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        convo = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "system": system,
            "persona_mode": persona_mode,
            "preferences": initial_preferences or {},
            "state": {},
        }
        resp = self._with_retry(lambda: self._table("conversations").insert(convo).execute())
        if not resp.data:
            raise RuntimeError("Failed to create conversation")
        convo = resp.data[0]
        convo["messages"] = []
        return convo

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        resp = self._with_retry(
            lambda: self._table("conversations")
            .select("*")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        convo = resp.data
        if not convo:
            return None
        state_payload = convo.get("state") or {}
        messages = state_payload.get("messages")
        if not messages:
            messages = self._list_messages(convo["id"])
            if not convo.get("state"):
                convo["state"] = {}
            convo["state"]["messages"] = messages
        convo["messages"] = messages
        return convo

    def list_conversations(
        self, user_id: str, *, system: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        query = (
            self._table("conversations")
            .select("id, system, persona_mode, preferences, state, updated_at, created_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
        )
        if system:
            query = query.eq("system", system)
        if limit:
            query = query.limit(limit)
        resp = self._with_retry(lambda: query.execute())
        return resp.data or []

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
        self._with_retry(
            lambda: self._table("messages").insert(
                {"conversation_id": conversation_id, "role": role, "content": content}
            ).execute()
        )
        update_payload: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if state_snapshot is not None:
            update_payload["state"] = state_snapshot
        if preferences is not None:
            update_payload["preferences"] = preferences
        if persona_mode:
            update_payload["persona_mode"] = persona_mode
        self._with_retry(
            lambda: self._table("conversations")
            .update(update_payload)
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )

    def _list_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        resp = self._with_retry(
            lambda: self._table("messages")
            .select("role, content, timestamp")
            .eq("conversation_id", conversation_id)
            .order("timestamp", desc=False)
            .execute()
        )
        return resp.data or []

    # Lease drafts -----------------------------------------------------------------

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
        payload = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "title": title,
            "pdf_base64": pdf_base64,
            "summary": summary,
            "metadata": metadata or {},
        }
        resp = self._with_retry(lambda: self._table("lease_drafts").insert(payload).execute())
        if not resp.data:
            raise RuntimeError("Failed to store lease draft")
        return resp.data[0]

    def list_lease_drafts(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = (
            self._table("lease_drafts")
            .select("id, title, summary, conversation_id, metadata, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
        )
        if limit:
            query = query.limit(limit)
        resp = self._with_retry(lambda: query.execute())
        return resp.data or []

    def get_lease_draft(self, draft_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        resp = self._with_retry(
            lambda: self._table("lease_drafts")
            .select("*")
            .eq("id", draft_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        return resp.data

    def get_latest_lease_draft(self, user_id: str) -> Optional[Dict[str, Any]]:
        drafts = self.list_lease_drafts(user_id, limit=1)
        return drafts[0] if drafts else None
