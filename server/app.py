from __future__ import annotations

import json
import os
import uuid
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.security import hash_password, verify_password
from storage.supabase_store import SupabaseStore
from system1 import lease_drafter
from system1.real_estate_agent import DEFAULT_PERSONA_MODE, SCRAPE_SIGNAL
from system1.session import System1AgentSession
from system2.session import System2AgentSession

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
WEBAPP_DIR = BASE_DIR / "webapp"
LEASE_OUTPUT_DIR = BASE_DIR / "system1" / "lease_drafts"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Supabase is required. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
store = SupabaseStore(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


class RegisterPayload(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    username: str


class LoginPayload(BaseModel):
    email: str
    password: str


class GooglePayload(BaseModel):
    email: str
    name: str
    google_id: str


class ForgotPasswordPayload(BaseModel):
    email: str
    password: str


class LeaseGeneratePayload(BaseModel):
    conversation_id: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


class ChatPayload(BaseModel):
    message: str
    system: str = "system1"
    conversation_id: Optional[str] = None
    persona_mode: Optional[str] = None
    carry_preferences: Optional[bool] = None


class CreateConversationPayload(BaseModel):
    system: str = "system1"
    persona_mode: Optional[str] = None
    carry_preferences: bool = True


class PreferencesPayload(BaseModel):
    updates: Dict[str, Any]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=WEBAPP_DIR / "assets"), name="assets")


def get_current_user(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("Authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    token = auth_header.split(None, 1)[1].strip()
    session = store.get_session(token)
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    user = store.find_user_by_id(session["user_id"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    request.state.session_token = token
    return user


@app.post("/api/auth/register")
def register_user(payload: RegisterPayload):
    try:
        user = store.register_user(
            {
                "email": payload.email.strip(),
                "username": payload.username.strip(),
                "first_name": payload.first_name.strip(),
                "last_name": payload.last_name.strip(),
                "password_hash": hash_password(payload.password),
                "auth_provider": "password",
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    session = store.create_session(user["id"])
    return {"token": session["token"], "user": _public_user(user)}


@app.post("/api/auth/login")
def login_user(payload: LoginPayload):
    user = store.find_user_by_email(payload.email.strip())
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
    session = store.create_session(user["id"])
    return {"token": session["token"], "user": _public_user(user)}


@app.post("/api/auth/google")
def google_login(payload: GooglePayload):
    user = store.find_user_by_email(payload.email.strip())
    if not user:
        first_name, last_name = _split_name(payload.name.strip())
        username = _derive_unique_username(payload.email.split("@")[0])
        try:
            user = store.register_user(
                {
                    "email": payload.email.strip(),
                    "username": username,
                    "first_name": first_name,
                    "last_name": last_name,
                    "password_hash": hash_password(payload.google_id),
                    "auth_provider": "google",
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    session = store.create_session(user["id"])
    return {"token": session["token"], "user": _public_user(user)}


@app.post("/api/auth/forgot-password")
def forgot_password(payload: ForgotPasswordPayload):
    email = payload.email.strip().lower()
    user = store.find_user_by_email(email)
    if not user or user.get("auth_provider") != "password":
        return {"ok": True}
    store.update_user_password(user["id"], hash_password(payload.password))
    return {"ok": True}


@app.get("/api/lease/drafts")
def list_lease_drafts(user: Dict[str, Any] = Depends(get_current_user)):
    drafts = store.list_lease_drafts(user["id"])
    return {"drafts": [_public_lease_draft(d) for d in drafts]}


@app.get("/api/lease/drafts/latest")
def latest_lease_draft(user: Dict[str, Any] = Depends(get_current_user)):
    draft = store.get_latest_lease_draft(user["id"])
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No lease drafts found.")
    return {"draft": _public_lease_draft(draft)}


@app.get("/api/lease/drafts/{draft_id}")
def get_lease_draft(draft_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    draft = store.get_lease_draft(draft_id, user["id"])
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lease draft not found.")
    return {"draft": _public_lease_draft(draft, include_pdf=True)}


@app.get("/api/lease/drafts/{draft_id}/download")
def download_lease_draft(draft_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    draft = store.get_lease_draft(draft_id, user["id"])
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lease draft not found.")
    pdf_base64 = draft.get("pdf_base64")
    if not pdf_base64:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF not stored for this draft.")
    pdf_bytes = base64.b64decode(pdf_base64)
    filename = (draft.get("title") or "lease_draft").replace(" ", "_") + ".pdf"
    safe_filename = filename.encode("ascii", errors="ignore").decode("ascii") or "lease_draft.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{safe_filename}\""},
    )


@app.post("/api/lease/generate")
def generate_lease(payload: LeaseGeneratePayload, user: Dict[str, Any] = Depends(get_current_user)):
    conversation = None
    if payload.conversation_id:
        conversation = store.get_conversation(payload.conversation_id, user["id"])
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    else:
        conversation = _latest_conversation(user["id"], "system1")
    result = _generate_and_store_lease(user, conversation, overrides=payload.overrides or {})
    return {"draft": result}


@app.post("/api/auth/guest")
def guest_session():
    return _create_guest_account()


@app.post("/api/auth/logout")
def logout_user(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    token = getattr(request.state, "session_token", None)
    if token:
        store.revoke_session(token)
    return {"ok": True}


@app.get("/api/auth/me")
def get_me(user: Dict[str, Any] = Depends(get_current_user)):
    return {"user": _public_user(user)}


@app.get("/api/preferences")
def get_preferences(user: Dict[str, Any] = Depends(get_current_user)):
    return {"preferences": user.get("preferences") or {}}


@app.post("/api/preferences")
def update_preferences(payload: PreferencesPayload, user: Dict[str, Any] = Depends(get_current_user)):
    prefs = user.get("preferences") or {}
    prefs.update(payload.updates or {})
    store.update_user_preferences(user["id"], prefs)
    user["preferences"] = prefs
    return {"preferences": prefs}


@app.get("/api/chat/conversations")
def list_conversations(request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    system = (request.query_params.get("system") or "").strip().lower()
    system_filter = system if system in {"system1", "system2"} else None
    convos = store.list_conversations(user["id"], system=system_filter)
    return {"conversations": [_public_conversation(c, include_messages=False) for c in convos]}


@app.post("/api/chat/conversations")
def create_conversation(payload: CreateConversationPayload, user: Dict[str, Any] = Depends(get_current_user)):
    system = payload.system.strip().lower()
    if system not in {"system1", "system2"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported system.")
    persona_mode = payload.persona_mode or DEFAULT_PERSONA_MODE
    initial_preferences = _initial_preferences(payload.carry_preferences, user)
    convo = store.create_conversation(user["id"], system, persona_mode, initial_preferences=initial_preferences)
    return {"conversation": _public_conversation(convo)}


@app.get("/api/chat/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    convo = store.get_conversation(conversation_id, user["id"])
    if not convo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return {"conversation": _public_conversation(convo)}


@app.post("/api/chat/send")
def chat_send(payload: ChatPayload, user: Dict[str, Any] = Depends(get_current_user)):
    try:
        result = _process_chat_message(user, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return result


@app.post("/api/chat/stream")
def chat_stream(payload: ChatPayload, user: Dict[str, Any] = Depends(get_current_user)):
    queue: "Queue[Tuple[str, Any]]" = Queue()

    def on_token(text: str) -> None:
        if not text:
            return
        if text == SCRAPE_SIGNAL:
            queue.put(("status", "scrape"))
            return
        queue.put(("token", text))

    def worker() -> None:
        try:
            result = _process_chat_message(user, payload.model_dump(), stream_handler=on_token)
            queue.put(("final", result))
        except ValueError as exc:
            queue.put(("error", {"error": str(exc)}))
        except Exception as exc:
            queue.put(("error", {"error": str(exc)}))

    Thread(target=worker, daemon=True).start()

    def generate():
        yield "event:start\ndata:{}\n\n"
        while True:
            kind, data = queue.get()
            if kind == "token":
                yield f"event:token\ndata:{json.dumps(data)}\n\n"
            elif kind == "final":
                yield "event:final\ndata:" + json.dumps(data) + "\n\n"
                break
            elif kind == "status":
                yield f"event:status\ndata:{json.dumps(data)}\n\n"
            else:
                yield "event:error\ndata:" + json.dumps(data) + "\n\n"
                break

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/{path:path}")
def serve_frontend(path: str):
    target = WEBAPP_DIR / path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(WEBAPP_DIR / "index.html")


def _process_chat_message(user: Dict[str, Any], payload: Dict[str, Any], stream_handler=None) -> Dict[str, Any]:
    message = (payload.get("message") or "").strip()
    system_id = (payload.get("system") or "system1").lower()
    conversation_id = payload.get("conversation_id")
    persona_mode = payload.get("persona_mode")

    if not message:
        raise ValueError("Message must not be empty.")
    if system_id not in {"system1", "system2"}:
        raise ValueError("Unsupported system.")

    latest_prior_convo = None
    if not conversation_id:
        latest_prior_convo = _latest_conversation(user["id"], system_id)

    conversation = None
    if conversation_id:
        conversation = store.get_conversation(conversation_id, user["id"])
        if not conversation:
            raise ValueError("Conversation not found.")
    else:
        persona = persona_mode or DEFAULT_PERSONA_MODE
        initial_preferences = _initial_preferences(payload.get("carry_preferences"), user)
        conversation = store.create_conversation(
            user["id"],
            system_id,
            persona,
            initial_preferences=initial_preferences,
        )
        conversation_id = conversation["id"]

    if conversation:
        conversation_state = conversation.get("state") or {}
        conversation_state.setdefault("messages", conversation.get("messages") or [])
        if "preferences" not in conversation_state:
            conversation_state["preferences"] = dict(conversation.get("preferences") or {})
        conversation_state.setdefault("persona_mode", conversation.get("persona_mode") or DEFAULT_PERSONA_MODE)
        conversation["state"] = conversation_state

    history_source = conversation if conversation_id else (latest_prior_convo or conversation)
    if _is_history_request(message):
        reply_text = _build_history_reply(user, system_id, history_source)
        session_result = {
            "reply": reply_text,
            "state": conversation.get("state") or {},
            "preferences": user.get("preferences") or {},
        }
    else:
        session_result = _invoke_system(
            system_id,
            conversation,
            persona_mode,
            message,
            stream_handler=stream_handler,
        )

    lease_plan = session_result["state"].pop("lease_generation_plan", None) if session_result.get("state") else None
    lease_result_payload = None
    if lease_plan and lease_plan.get("status") == "requested":
        ack = lease_plan.get("ack") or "I'll prepare that lease draft for you."
        try:
            lease_result_payload = _generate_and_store_lease(
                user,
                conversation,
                overrides=lease_plan.get("overrides"),
            )
            summary_line = lease_result_payload.get("summary") or "Lease draft saved to your account."
            draft_id = lease_result_payload["id"]
            lease_reply = (
                f"{ack}\n\nLease draft ready. Use the Lease drafts panel to download the PDF "
                f"(draft ID: {draft_id}).\nSummary: {summary_line}"
            )
            session_result["lease_draft"] = lease_result_payload
            session_result["state"].setdefault("latest_lease", lease_result_payload)
        except Exception as exc:
            lease_reply = f"{ack}\n\nHowever, I hit an issue while drafting it: {exc}"
        session_result["reply"] = lease_reply
        msgs = session_result["state"].setdefault("messages", [])
        if msgs and msgs[-1].get("role") == "assistant":
            msgs[-1]["content"] = lease_reply
        else:
            msgs.append({"role": "assistant", "content": lease_reply})

    store.append_message(
        conversation_id,
        user["id"],
        role="user",
        content=message,
        state_snapshot=session_result["state"],
        preferences=session_result.get("preferences") or {},
        persona_mode=session_result["state"].get("persona_mode"),
    )
    clean_reply = _strip_preference_dump(session_result["reply"])
    store.append_message(
        conversation_id,
        user["id"],
        role="assistant",
        content=clean_reply,
        state_snapshot=session_result["state"],
        preferences=session_result.get("preferences") or {},
        persona_mode=session_result["state"].get("persona_mode"),
    )
    _merge_and_save_user_preferences(
        user,
        session_result.get("preferences") or {},
        conversation_id,
        clean_reply,
    )

    updated_conversation = _assemble_conversation_payload(
        conversation,
        conversation_id,
        system_id,
        persona_mode,
        session_result,
    )
    return {
        "conversation_id": conversation_id,
        "reply": clean_reply,
        "state": session_result["state"],
        "preferences": session_result.get("preferences") or {},
        "conversation": _public_conversation(updated_conversation),
    }


def _invoke_system(
    system_id: str,
    conversation: Optional[Dict[str, Any]],
    persona_mode: Optional[str],
    message: str,
    *,
    stream_handler=None,
) -> Dict[str, Any]:
    if system_id == "system1":
        session = System1AgentSession(
            persona_mode=persona_mode or (conversation.get("persona_mode") if conversation else None),
            state=conversation.get("state") if conversation else None,
        )
        return session.send(message, stream_handler=stream_handler)
    session = System2AgentSession(state=conversation.get("state") if conversation else None)
    return session.send(message, stream_handler=stream_handler)


def _public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "username": user["username"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "preferences": user.get("preferences") or {},
        "auth_provider": user.get("auth_provider", "password"),
    }


def _conversation_messages(convo: Dict[str, Any]) -> List[Dict[str, Any]]:
    if convo.get("messages"):
        return convo["messages"]
    state = convo.get("state")
    if isinstance(state, dict):
        msgs = state.get("messages")
        if isinstance(msgs, list):
            return msgs
    return []


def _snippet(text: str, limit: int) -> str:
    clean = " ".join((text or "").strip().split())
    if not clean:
        return ""
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip(" .,;:-") + "…"


def _conversation_topic(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            return _snippet(msg["content"], 80)
    return ""


def _public_conversation(convo: Dict[str, Any], include_messages: bool = True) -> Dict[str, Any]:
    messages = _conversation_messages(convo)
    preview = ""
    if messages:
        for msg in reversed(messages):
            if msg.get("content"):
                preview = _snippet(msg["content"], 160)
                break
    payload = {
        "id": convo["id"],
        "system": convo["system"],
        "persona_mode": convo.get("persona_mode"),
        "preferences": convo.get("preferences", {}),
        "updated_at": _normalize_ts(convo.get("updated_at")),
        "created_at": _normalize_ts(convo.get("created_at")),
        "preview": preview,
        "topic": _conversation_topic(messages),
    }
    if include_messages:
        payload["messages"] = messages
    return payload


def _public_lease_draft(draft: Dict[str, Any], *, include_pdf: bool = False) -> Dict[str, Any]:
    payload = {
        "id": draft["id"],
        "title": draft.get("title"),
        "summary": draft.get("summary"),
        "conversation_id": draft.get("conversation_id"),
        "metadata": draft.get("metadata") or {},
        "created_at": _normalize_ts(draft.get("created_at")),
    }
    if include_pdf:
        payload["pdf_base64"] = draft.get("pdf_base64")
    return payload


def _normalize_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return None
    return None


def _assemble_conversation_payload(
    conversation: Optional[Dict[str, Any]],
    conversation_id: str,
    system_id: str,
    persona_mode: Optional[str],
    session_result: Dict[str, Any],
) -> Dict[str, Any]:
    base = dict(conversation or {})
    base["id"] = conversation_id
    base["system"] = system_id
    state = session_result.get("state") or {}
    preferences = session_result.get("preferences") or {}
    persona = state.get("persona_mode") or persona_mode or base.get("persona_mode") or DEFAULT_PERSONA_MODE
    base["persona_mode"] = persona
    base["preferences"] = preferences
    base["state"] = state
    base["messages"] = state.get("messages", base.get("messages", []))
    now = datetime.now(timezone.utc).isoformat()
    base.setdefault("created_at", now)
    base["updated_at"] = now
    return base


def _collect_preferences(user: Dict[str, Any], conversation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prefs: Dict[str, Any] = dict(user.get("preferences") or {})
    if conversation:
        prefs.update(conversation.get("preferences") or {})
        state = conversation.get("state") or {}
        prefs.update(state.get("preferences") or {})
    return prefs


def _extract_listing_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not state:
        return None
    for key in ("focused_listing", "selected_listing", "active_listing", "current_listing"):
        listing = state.get(key)
        if listing:
            return listing
    for key in ("ranked_listings", "enriched_listings", "scraped_listings", "listings"):
        items = state.get(key)
        if isinstance(items, list) and items:
            return items[0]
    return None


def _summarize_compliance(inputs: lease_drafter.LeaseDraftInputs, compliance: Dict[str, Any]) -> str:
    issues = len((compliance or {}).get("issues") or [])
    warnings = len((compliance or {}).get("warnings") or [])
    location = inputs.location_line or inputs.city or inputs.state or "Property"
    return f"{inputs.tenant_name} – {location} (issues: {issues}, warnings: {warnings})"


def _generate_and_store_lease(
    user: Dict[str, Any],
    conversation: Optional[Dict[str, Any]],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conversation_state = conversation.get("state") if conversation else None
    prefs = _collect_preferences(user, conversation)
    listing = _extract_listing_from_state(conversation_state or {})
    overrides = overrides or {}

    inputs = lease_drafter.infer_inputs(preferences=prefs, listing=listing, overrides=overrides)
    package = lease_drafter.build_lease_package(inputs, output_dir=str(LEASE_OUTPUT_DIR))
    pdf_path = Path(package["pdf_path"])
    pdf_bytes = pdf_path.read_bytes()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    summary = _summarize_compliance(inputs, package.get("compliance") or {})
    inputs_dict = json.loads(json.dumps(asdict(inputs), default=str))
    metadata = {
        "inputs": inputs_dict,
        "compliance": package.get("compliance") or {},
    }
    title = f"Lease Draft – {inputs.city or inputs.state or 'Property'}"
    saved = store.save_lease_draft(
        user_id=user["id"],
        conversation_id=conversation.get("id") if conversation else None,
        title=title,
        pdf_base64=pdf_base64,
        summary=summary,
        metadata=metadata,
    )

    # Cleanup generated files to avoid clutter.
    txt_path = Path(package.get("file_path", ""))
    for path in (pdf_path, txt_path):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    saved["pdf_base64"] = pdf_base64
    return _public_lease_draft(saved, include_pdf=True)


def _strip_preference_dump(text: str) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text
    decoder = json.JSONDecoder()
    try:
        data, idx = decoder.raw_decode(stripped)
    except ValueError:
        return text
    if not isinstance(data, dict) or "preferences" not in data:
        return text
    remainder = stripped[idx:].lstrip()
    return remainder or text


def _should_carry_preferences(flag: Any) -> bool:
    if isinstance(flag, str):
        return flag.strip().lower() not in {"false", "0", "no", "off"}
    if flag is None:
        return True
    return bool(flag)


def _initial_preferences(flag: Any, user: Dict[str, Any]) -> Dict[str, Any]:
    carry = flag
    if carry is None:
        carry = user.get("preferences", {}).get("share_preferences", True)
    if not _should_carry_preferences(carry):
        return {}
    return dict(user.get("preferences") or {})


def _create_guest_account() -> Dict[str, Any]:
    guest_code = uuid.uuid4().hex[:8]
    email = f"guest-{guest_code}@aptiva.local"
    username = _derive_unique_username(f"guest{guest_code}")
    user = store.register_user(
        {
            "email": email,
            "username": username,
            "first_name": "Guest",
            "last_name": guest_code.upper(),
            "password_hash": hash_password(uuid.uuid4().hex),
            "auth_provider": "guest",
        }
    )
    session = store.create_session(user["id"])
    return {"token": session["token"], "user": _public_user(user)}


def _split_name(full_name: str) -> Tuple[str, str]:
    parts = full_name.split()
    if not parts:
        return ("User", "Google")
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], " ".join(parts[1:]))


def _derive_unique_username(base: str) -> str:
    cleaned = "".join(ch for ch in base.lower() if ch.isalnum()) or "user"
    candidate = cleaned
    suffix = 1
    while store.find_user_by_username(candidate):
        candidate = f"{cleaned}{suffix}"
        suffix += 1
    return candidate


def _latest_conversation(user_id: str, system_id: str) -> Optional[Dict[str, Any]]:
    convos = store.list_conversations(user_id, system=system_id, limit=5)
    for convo in convos:
        detailed = store.get_conversation(convo["id"], user_id)
        if detailed and detailed.get("messages"):
            return detailed
    return None


def _build_history_reply(user: Dict[str, Any], system_id: str, latest_convo: Optional[Dict[str, Any]]) -> str:
    prefs = user.get("preferences") or {}
    lines: List[str] = []
    last_message = ""
    if latest_convo and latest_convo.get("messages"):
        for message in reversed(latest_convo["messages"]):
            if message.get("role") == "assistant":
                last_message = message.get("content", "")
                break
    elif prefs.get("_last_reply"):
        last_message = prefs.get("_last_reply", "")
    if last_message:
        lines.append("**Last options I shared**\n" + last_message)

    if not lines:
        return "I don't have previous searches yet. Share your preferences and I'll keep them for future chats."
    return "\n\n".join(lines)


def _is_history_request(message: str) -> bool:
    text = (message or "").lower()
    keywords = [
        "previous search",
        "pervious search",
        "previous preference",
        "pervious preference",
        "previous preferences",
        "pervious preferences",
        "previous option",
        "pervious option",
        "previous options",
        "pervious options",
        "what was i searching",
        "what was i pervious",
        "what did i ask",
        "what did i search",
        "what did i pervious",
        "remind me what i asked",
        "remind me what i searched",
        "last search",
        "last options",
        "earlier search",
        "earlier options",
    ]
    return any(phrase in text for phrase in keywords)


def _merge_and_save_user_preferences(
    user: Dict[str, Any],
    new_prefs: Dict[str, Any],
    conversation_id: str,
    last_reply: str,
) -> None:
    merged = dict(user.get("preferences") or {})
    merged.update(new_prefs or {})
    merged["_last_conversation_id"] = conversation_id
    merged["_last_reply"] = last_reply
    store.update_user_preferences(user["id"], merged)
    user["preferences"] = merged


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
