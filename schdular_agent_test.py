#!/usr/bin/env python3
import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from fastmcp import Client as McpClient
from fastmcp.client.transports import StreamableHttpTransport
import dotenv

dotenv.load_dotenv()

# =========================
# CONFIG
# =========================

# NEVER hard-code your API key in the file.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")
OPENAI_MODEL = "gpt-4o"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

ZAPIER_MCP_URL = os.getenv(
    "ZAPIER_MCP_URL",
    "https://mcp.zapier.com/api/mcp/s/MjZiOWYyM2QtNWE3ZS00ZGYzLWI0MWQtZGE5OGY4MjY4ZjU2OjA3NzczOTdkLWEzMmEtNDJiYi05NzEzLWNiNDVmMGZiMjBhOA==/mcp",
)
ZAPIER_MCP_TOKEN = ""


## change code here
USER_CALENDAR_ID = os.getenv("USER_CAL_ID", "lamamawlawi9@gmail.com")
LANDLORD_CALENDAR_ID = os.getenv("LANDLORD_CAL_ID", "mawlama152003@gmail.com")
USER_EMAIL = USER_CALENDAR_ID

LISTING_TITLE = os.getenv("LISTING_TITLE", "Oskar Luxury Apartments")
LISTING_ADDRESS = os.getenv("LISTING_ADDRESS", "572 11th Ave, New York, NY 10036")

DEFAULT_SLOT_MINUTES = 30

# Local timezone offset from UTC, in hours (e.g. 2 for UTC+2, -5 for UTC-5)
LOCAL_TZ_OFFSET_HOURS = 2
LOCAL_TZ = timezone(timedelta(hours=LOCAL_TZ_OFFSET_HOURS))
# end of change
## ======================================================================

# =========================
# Utility helpers
# =========================

def parse_iso(dt_str: str) -> datetime:
    """Parse ISO string into aware datetime."""
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")
    return datetime.fromisoformat(dt_str)


def to_local(dt: datetime) -> datetime:
    """Convert an aware datetime to the local timezone; assume local if naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def fmt_iso_pretty(dt_str: str) -> str:
    """Human-readable, mainly for instructions to Zapier."""
    dt = parse_iso(dt_str)
    return dt.strftime("%a %Y-%m-%d %H:%M")


def round_up_to_slot(dt: datetime, minutes: int) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    rem = dt.minute % minutes
    if rem == 0:
        return dt
    return dt + timedelta(minutes=(minutes - rem))


def fmt_slot_pretty(start_iso: str, end_iso: str) -> str:
    """User-facing pretty format in LOCAL time."""
    s = to_local(parse_iso(start_iso))
    e = to_local(parse_iso(end_iso))
    return f"{s.strftime('%a %Y-%m-%d %I:%M %p')} -> {e.strftime('%I:%M %p')}"


# =========================
# MCP wrapper (Zapier Google Calendar)
# =========================

def _make_mcp_transport():
    headers = {"Authorization": f"Bearer {ZAPIER_MCP_TOKEN}"} if ZAPIER_MCP_TOKEN else None
    return StreamableHttpTransport(ZAPIER_MCP_URL, headers=headers) if headers else StreamableHttpTransport(ZAPIER_MCP_URL)


class GoogleCalendarMCP:
    """Thin async wrapper around Zapier MCP Google Calendar tools."""

    def __init__(self):
        self.transport = _make_mcp_transport()
        self.client: Optional[McpClient] = None

    async def __aenter__(self):
        self.client = McpClient(transport=self.transport)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.__aexit__(exc_type, exc, tb)
        self.client = None

    async def find_events_as_busy(
        self,
        calendar_id: str,
        start_iso: str,
        end_iso: str,
    ) -> List[Dict[str, Any]]:
        """
        Use google_calendar_find_events and treat each event as a busy block.
        We drive it via 'instructions' to avoid follow-up questions.
        """
        assert self.client is not None, "MCP client not connected"

        start_h = fmt_iso_pretty(start_iso)
        end_h = fmt_iso_pretty(end_iso)

        instructions = (
            f"List all events in calendar '{calendar_id}' between "
            f"{start_h} (ISO {start_iso}) and {end_h} (ISO {end_iso}). "
            "Include each event's id, summary, and start/end dateTime. "
            "Do NOT ask me any follow-up questions."
        )

        res = await self.client.call_tool(
            "google_calendar_find_events",
            {
                "instructions": instructions,
                "calendarid": calendar_id,
                "max_results": 50,
                "ordering": "startTime",
                "showDeleted": False,
            },
        )

        payload = json.loads(res.content[0].text)
        events = payload.get("results", [])
        busy_blocks: List[Dict[str, Any]] = []
        for ev in events:
            s = ev.get("start", {}).get("dateTime")
            e = ev.get("end", {}).get("dateTime")
            if s and e:
                busy_blocks.append({
                    "id": ev.get("id"),
                    "summary": ev.get("summary"),
                    "start": {"dateTime": s},
                    "end": {"dateTime": e},
                })
        return busy_blocks

    async def list_tour_events(
        self,
        calendar_id: str,
        time_min_iso: str,
        time_max_iso: str,
    ) -> List[Dict[str, Any]]:
        """
        List upcoming 'tour' events in this calendar.
        We'll filter by summary containing 'tour' or 'Tour'.
        """
        assert self.client is not None, "MCP client not connected"

        time_min_h = fmt_iso_pretty(time_min_iso)
        time_max_h = fmt_iso_pretty(time_max_iso)

        instructions = (
            f"List all events in calendar '{calendar_id}' between "
            f"{time_min_h} (ISO {time_min_iso}) and {time_max_h} (ISO {time_max_iso}) "
            "whose summary suggests an apartment tour (e.g., contains 'tour' or 'Tour'). "
            "Include each event's id, summary, and start/end dateTime. "
            "Do NOT ask me any follow-up questions."
        )

        res = await self.client.call_tool(
            "google_calendar_find_events",
            {
                "instructions": instructions,
                "calendarid": calendar_id,
                "max_results": 50,
                "ordering": "startTime",
                "showDeleted": False,
            },
        )
        payload = json.loads(res.content[0].text)
        events = payload.get("results", [])
        tours: List[Dict[str, Any]] = []
        for ev in events:
            s = ev.get("start", {}).get("dateTime")
            e = ev.get("end", {}).get("dateTime")
            if s and e:
                tours.append({
                    "id": ev.get("id"),
                    "summary": ev.get("summary"),
                    "start": s,
                    "end": e,
                })
        return tours

    async def create_event_detailed(
        self,
        calendar_id: str,
        summary: str,
        start_iso: str,
        end_iso: str,
        description: str = "",
        location: str = "",
        attendees: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Use google_calendar_create_detailed_event via 'instructions'.
        """
        assert self.client is not None, "MCP client not connected"
        attendees = attendees or []

        start_h = fmt_iso_pretty(start_iso)
        end_h = fmt_iso_pretty(end_iso)
        attendee_emails = ", ".join(a["email"] for a in attendees if "email" in a) or "no extra attendees"

        instructions = (
            f"Create a Google Calendar event in calendar '{calendar_id}' with:\n"
            f"- Title: {summary}\n"
            f"- Start: {start_h} (from ISO {start_iso})\n"
            f"- End: {end_h} (from ISO {end_iso})\n"
            f"- Location: {location}\n"
            f"- Description: {description}\n"
            f"- Attendees: {attendee_emails}\n"
            "If any fields are missing, make reasonable defaults, "
            "but do NOT ask me any follow-up questions."
        )

        res = await self.client.call_tool(
            "google_calendar_create_detailed_event",
            {
                "instructions": instructions,
                "calendarid": calendar_id,
            },
        )
        payload = json.loads(res.content[0].text)
        results = payload.get("results", [])
        return results[0] if results else payload

    async def delete_event(
        self,
        calendar_id: str,
        event_id: str,
    ) -> Dict[str, Any]:
        """
        Use google_calendar_delete_event. We drive it with instructions + calendarid + eventid.
        """
        assert self.client is not None, "MCP client not connected"

        instructions = (
            f"Delete the event with id '{event_id}' from calendar '{calendar_id}'. "
            "Do NOT ask me any follow-up questions."
        )

        res = await self.client.call_tool(
            "google_calendar_delete_event",
            {
                "instructions": instructions,
                "calendarid": calendar_id,
                "eventid": event_id,
            },
        )
        payload = json.loads(res.content[0].text)
        return payload


# =========================
# Scheduling logic (common slots)
# =========================

def invert_busy_to_free(
    start_window: datetime,
    end_window: datetime,
    busy_blocks: List[Dict[str, Any]],
) -> List[Tuple[datetime, datetime]]:
    """Convert busy intervals to free intervals."""
    intervals: List[Tuple[datetime, datetime]] = []
    for block in busy_blocks:
        s_raw = block.get("start", {}).get("dateTime")
        e_raw = block.get("end", {}).get("dateTime")
        if not s_raw or not e_raw:
            continue
        s = parse_iso(s_raw)
        e = parse_iso(e_raw)
        if e <= start_window or s >= end_window:
            continue
        intervals.append((max(s, start_window), min(e, end_window)))

    intervals.sort(key=lambda x: x[0])

    free: List[Tuple[datetime, datetime]] = []
    current = start_window
    for s, e in intervals:
        if s > current:
            free.append((current, s))
        current = max(current, e)
    if current < end_window:
        free.append((current, end_window))
    return free


def intersect_free_slots(
    free_a: List[Tuple[datetime, datetime]],
    free_b: List[Tuple[datetime, datetime]],
    slot_minutes: int = DEFAULT_SLOT_MINUTES,
) -> List[Tuple[datetime, datetime]]:
    """
    Intersect free windows and cut into slots, weekdays only (Mon-Fri)
    between 09:00-17:00 in *local* time.
    """
    out: List[Tuple[datetime, datetime]] = []
    dur = timedelta(minutes=slot_minutes)

    for a_start, a_end in free_a:
        for b_start, b_end in free_b:
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            if end - start < dur:
                continue

            slot_start = round_up_to_slot(start, slot_minutes)
            while slot_start + dur <= end:
                slot_end = slot_start + dur

                # convert to local time before checking weekday & hours
                local_start = to_local(slot_start)
                local_end = to_local(slot_end)

                wd = local_start.weekday()  # 0=Mon..6=Sun (in local time)
                if wd <= 4:  # Mon-Fri
                    sh = local_start.hour + local_start.minute / 60.0
                    eh = local_end.hour + local_end.minute / 60.0
                    # 9:00-17:00 local
                    if 9.0 <= sh and eh <= 17.0:
                        out.append((slot_start, slot_end))

                slot_start += dur

    out.sort(key=lambda x: x[0])
    return out



# =========================
# BACKEND: functions the LLM tools will call
# =========================

async def backend_get_common_slots(
    window_start_iso: str,
    window_end_iso: str,
    duration_minutes: int = DEFAULT_SLOT_MINUTES,
    max_slots: int = 10,
) -> List[Dict[str, str]]:
    """
    Compute common free slots between USER_CALENDAR_ID and LANDLORD_CALENDAR_ID.
    Mon-Fri, 09:00-17:00, duration_minutes each.
    """
    start_dt = parse_iso(window_start_iso)
    end_dt = parse_iso(window_end_iso)

    # If the model supplies a start time later in the day, normalize to 09:00 local
    # so morning slots are not accidentally skipped.
    start_dt_local = to_local(start_dt)
    if start_dt_local.hour > 9 or (start_dt_local.hour == 9 and start_dt_local.minute > 0):
        start_dt = start_dt_local.replace(hour=9, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

    async with GoogleCalendarMCP() as cal:
        user_busy = await cal.find_events_as_busy(USER_CALENDAR_ID, window_start_iso, window_end_iso)
        landlord_busy = await cal.find_events_as_busy(LANDLORD_CALENDAR_ID, window_start_iso, window_end_iso)

    free_user = invert_busy_to_free(start_dt, end_dt, user_busy)
    free_landlord = invert_busy_to_free(start_dt, end_dt, landlord_busy)

    # print("\nDEBUG user_busy blocks (raw):")
    # for b in user_busy:
    #     s = b["start"]["dateTime"]
    #     e = b["end"]["dateTime"]
    #     print("  USER busy:", fmt_slot_pretty(s, e))

    # print("\nDEBUG landlord_busy blocks (raw):")
    # for b in landlord_busy:
    #     s = b["start"]["dateTime"]
    #     e = b["end"]["dateTime"]
    #     print("  LANDLORD busy:", fmt_slot_pretty(s, e))

    raw_slots = intersect_free_slots(free_user, free_landlord, duration_minutes)

    # print("\nDEBUG raw candidate slots (UTC and local):")
    # for s, e in raw_slots:
    #     s_local = to_local(s)
    #     e_local = to_local(e)
    #     print(
    #         "  ", s.isoformat(timespec="minutes"),
    #         "->", e.isoformat(timespec="minutes"),
    #         "| local:", s_local.strftime("%a %Y-%m-%d %I:%M %p"),
    #         "->", e_local.strftime("%I:%M %p"),
    #     )

    slots: List[Dict[str, str]] = []
    for s, e in raw_slots[:max_slots]:
        # Convert to local time for display
        start_local = to_local(s)
        end_local = to_local(e)

        day_label = start_local.strftime("%a %B %d")  # e.g. "Mon November 24"
        start_time = start_local.strftime("%I:%M %p").lstrip("0")  # "1:00 PM"
        end_time = end_local.strftime("%I:%M %p").lstrip("0")      # "1:30 PM"

        slots.append({
            # keep ISO based on real datetimes (s, e)
            "start": s.isoformat(timespec="minutes"),
            "end": e.isoformat(timespec="minutes"),
            # label shown to the user
            "label": f"{day_label} - {start_time} -> {end_time}",
        })

    return slots


async def backend_book_tour(
    slot_start_iso: str,
    slot_end_iso: str,
    listing_title: str = LISTING_TITLE,
    listing_address: str = LISTING_ADDRESS,
) -> Dict[str, Any]:
    """
    Book a tour in both calendars.
    """
    async with GoogleCalendarMCP() as cal:
        user_event = await cal.create_event_detailed(
            calendar_id=USER_CALENDAR_ID,
            summary=f"Apartment tour - {listing_title}",
            start_iso=slot_start_iso,
            end_iso=slot_end_iso,
            location=listing_address,
            description="Scheduled via scheduler_agent",
        )
        landlord_event = await cal.create_event_detailed(
            calendar_id=LANDLORD_CALENDAR_ID,
            summary=f"Tour with {USER_EMAIL} - {listing_title}",
            start_iso=slot_start_iso,
            end_iso=slot_end_iso,
            location=listing_address,
            description="Scheduled via scheduler_agent",
            attendees=[{"email": USER_EMAIL}],
        )

    return {
        "user_event": {
            "id": user_event.get("id"),
            "summary": user_event.get("summary"),
            "start": user_event.get("start", {}),
            "end": user_event.get("end", {}),
        },
        "landlord_event": {
            "id": landlord_event.get("id"),
            "summary": landlord_event.get("summary"),
            "start": landlord_event.get("start", {}),
            "end": landlord_event.get("end", {}),
        },
    }


async def backend_list_tours(
    days_ahead: int = 7,
) -> List[Dict[str, Any]]:
    """
    List upcoming tour-like events from both calendars for the next N days.
    """
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    time_min_iso = now.isoformat(timespec="seconds")
    time_max_iso = end.isoformat(timespec="seconds")

    async with GoogleCalendarMCP() as cal:
        user_tours = await cal.list_tour_events(USER_CALENDAR_ID, time_min_iso, time_max_iso)
        landlord_tours = await cal.list_tour_events(LANDLORD_CALENDAR_ID, time_min_iso, time_max_iso)

    out: List[Dict[str, Any]] = []
    for ev in user_tours:
        out.append({
            "calendar": "user",
            "id": ev["id"],
            "summary": ev["summary"],
            "start": ev["start"],
            "end": ev["end"],
        })
    for ev in landlord_tours:
        out.append({
            "calendar": "landlord",
            "id": ev["id"],
            "summary": ev["summary"],
            "start": ev["start"],
            "end": ev["end"],
        })

    out.sort(key=lambda e: e["start"])
    return out


async def backend_cancel_tour(
    calendar: str,
    event_id: str,
) -> Dict[str, Any]:
    """
    Cancel a tour event from either the user's or landlord's calendar.
    'calendar' should be 'user' or 'landlord'.
    """
    if calendar == "user":
        cal_id = USER_CALENDAR_ID
    elif calendar == "landlord":
        cal_id = LANDLORD_CALENDAR_ID
    else:
        return {"error": f"Unknown calendar: {calendar}"}

    async with GoogleCalendarMCP() as cal:
        res = await cal.delete_event(cal_id, event_id)
    return res


# =========================
# OpenAI tool definitions (what the LLM can call)
# =========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_common_slots",
            "description": (
                "Find up to 10 common free 30-minute weekday (Mon-Fri) slots between 09:00 and 17:00 "
                "for the renter and landlord in a given time window. "
                "Use this to propose tour times."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window_start_iso": {
                        "type": "string",
                        "description": "ISO datetime start of search window, e.g. '2025-11-24T09:00:00+00:00'.",
                    },
                    "window_end_iso": {
                        "type": "string",
                        "description": "ISO datetime end of search window.",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Slot duration in minutes (default 30).",
                        "default": DEFAULT_SLOT_MINUTES,
                    },
                },
                "required": ["window_start_iso", "window_end_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_tour",
            "description": (
                "Book an apartment tour for the selected slot in both the renter's and landlord's calendars. "
                "Use this after the user confirms a slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_start_iso": {
                        "type": "string",
                        "description": "ISO datetime for the chosen slot start.",
                    },
                    "slot_end_iso": {
                        "type": "string",
                        "description": "ISO datetime for the chosen slot end.",
                    },
                    "listing_title": {
                        "type": "string",
                        "description": "Listing title (optional, default configured).",
                    },
                    "listing_address": {
                        "type": "string",
                        "description": "Listing address (optional, default configured).",
                    },
                },
                "required": ["slot_start_iso", "slot_end_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tours",
            "description": (
                "List upcoming apartment tour events for the renter and landlord over the next N days. "
                "Use this when the user asks about their schedule or existing bookings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {
                        "type": "integer",
                        "description": "How many days ahead to look (default 7).",
                        "default": 7,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_tour",
            "description": (
                "Cancel a tour event from either the renter's or landlord's calendar. "
                "Use this when the user wants to cancel or reschedule a specific tour."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calendar": {
                        "type": "string",
                        "enum": ["user", "landlord"],
                        "description": "Which calendar the event is on.",
                    },
                    "event_id": {
                        "type": "string",
                        "description": "ID of the event to cancel (from list_tours).",
                    },
                },
                "required": ["calendar", "event_id"],
            },
        },
    },
]


# =========================
# Tool dispatcher (OpenAI -> Python -> MCP)
# =========================

async def handle_tool_call(tool_call) -> Dict[str, Any]:
    name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}

    if name == "get_common_slots":
        return await backend_get_common_slots(**args)
    elif name == "book_tour":
        return await backend_book_tour(**args)
    elif name == "list_tours":
        return await backend_list_tours(**args)
    elif name == "cancel_tour":
        return await backend_cancel_tour(**args)
    else:
        return {"error": f"Unknown tool: {name}"}


def build_system_prompt() -> str:
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    now_local = datetime.now(LOCAL_TZ).isoformat(timespec="seconds")
    return (
        "You are an apartment tour scheduling assistant.\n"
        f"Current UTC time is {now_utc}.\n"
        f"Current local time (offset {LOCAL_TZ_OFFSET_HOURS:+g}) is {now_local}.\n\n"
        f"The default apartment listing is '{LISTING_TITLE}' at '{LISTING_ADDRESS}'.\n"
        f"The renter's email is {USER_EMAIL} with calendar {USER_CALENDAR_ID}.\n"
        f"The landlord calendar is {LANDLORD_CALENDAR_ID}.\n\n"
        "Your job:\n"
        "- Help the user schedule, view, and cancel apartment tours.\n"
        "- Use get_common_slots to find times when BOTH calendars are free.\n"
        "- Only suggest weekday (Mon-Fri) slots between 09:00 and 17:00.\n"
        "- Present slots in a short, numbered list with local-looking times.\n"
        "- When the user picks a slot, call book_tour.\n"
        "- When the user asks about their schedule, call list_tours.\n"
        "- When they want to cancel, call list_tours to identify the event, then cancel_tour.\n"
        "- If the user doesn't specify dates, default to the next 3 days from now.\n"
        "- If the user names a weekday (e.g., 'Wednesday'), resolve it to the next occurrence in LOCAL time and state the exact date before offering slots. Do NOT silently shift a day due to UTC offsets—confirm the interpreted date if ambiguous.\n"
        "- Each slot from get_common_slots has a 'label' field. "
        "Always show that label exactly as given (do NOT recompute or change the date/time yourself).\n"
        "- Be clear and polite, and confirm actions.\n"
    )


# =========================
# CLI conversation loop
# =========================

def run_scheduler_agent_cli():
    system_prompt = build_system_prompt()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    print("Tour Scheduler Agent ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        # We may need multiple tool->LLM->tool cycles for a single user turn
        while True:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = resp.choices[0].message

            # If the model wants to call tools
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                })

                # Run each tool call
                for tool_call in msg.tool_calls:
                    result = asyncio.run(handle_tool_call(tool_call))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(result),
                    })
                # Loop again so the model can see tool results and answer
                continue

            # No tool calls: this is the final user-facing answer
            print(f"Agent: {msg.content}\n")
            messages.append({"role": "assistant", "content": msg.content})
            break


if __name__ == "__main__":
    run_scheduler_agent_cli()
