from __future__ import annotations

import csv
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - Supabase optional at runtime
    Client = None  # type: ignore
    create_client = None  # type: ignore


METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"
CSV_PATH = METRICS_DIR / "cost_log.csv"
CSV_COLUMNS = [
    "timestamp",
    "component",
    "model_or_tool",
    "tokens_in",
    "tokens_out",
    "latency_ms",
    "cost_usd",
    "conversation_id",
]

# Approximate per-1K token pricing in USD.
MODEL_PRICING_PER_1K = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "o4-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

_csv_lock = threading.Lock()
_supabase_client: Optional[Client] = None


def _get_supabase_client() -> Optional[Client]:
    """Lazily initialize a Supabase client when env vars are present."""
    global _supabase_client
    if _supabase_client or create_client is None:
        return _supabase_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    try:
        _supabase_client = create_client(url, key)
    except Exception:
        _supabase_client = None
    return _supabase_client


def _ensure_csv_header() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        return
    with _csv_lock:
        if CSV_PATH.exists():
            return
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def _coerce_number(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def estimate_openai_cost(model: Optional[str], tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    """Rudimentary USD cost estimate using static per-1K token pricing."""
    if model is None:
        return None
    key = model.lower()
    pricing = MODEL_PRICING_PER_1K.get(key)
    if pricing is None:
        return None
    cost = 0.0
    if tokens_in:
        cost += (tokens_in / 1000.0) * pricing["input"]
    if tokens_out:
        cost += (tokens_out / 1000.0) * pricing["output"]
    return round(cost, 6)


def extract_usage_tokens(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    """Pull token counts from OpenAI responses or usage payloads."""
    usage = getattr(obj, "usage", None)
    if isinstance(obj, dict) and not usage:
        usage = obj.get("usage")
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or (
        usage.get("prompt_tokens") if isinstance(usage, dict) else None
    ) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
    completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or (
        usage.get("completion_tokens") if isinstance(usage, dict) else None
    ) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
    return prompt if prompt is not None else None, completion if completion is not None else None


def log_metric(
    component: str,
    model_or_tool: Optional[str],
    *,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    latency_ms: Optional[float] = None,
    cost_usd: Optional[float] = None,
    conversation_id: Optional[str] = None,
) -> None:
    """Persist a metric row to CSV and Supabase (best effort)."""
    timestamp = datetime.now(timezone.utc).isoformat()
    computed_cost = cost_usd if cost_usd is not None else estimate_openai_cost(model_or_tool, tokens_in, tokens_out)
    row = {
        "timestamp": timestamp,
        "component": component,
        "model_or_tool": model_or_tool or "",
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
        "cost_usd": computed_cost,
        "conversation_id": conversation_id,
    }
    csv_row = {
        k: ("" if v is None else v) for k, v in row.items()
    }

    try:
        _ensure_csv_header()
        with _csv_lock:
            with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(csv_row)
    except Exception:
        pass  # local logging should never crash the app

    client = _get_supabase_client()
    if client is not None:
        try:
            client.table("metrics").insert(row).execute()
        except Exception:
            pass


@dataclass
class MetricTimer:
    component: str
    model_or_tool: Optional[str]
    conversation_id: Optional[str]
    _start: float = field(default_factory=time.perf_counter)

    def done(
        self,
        *,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000
        log_metric(
            self.component,
            self.model_or_tool,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            conversation_id=self.conversation_id,
        )


def start_timer(component: str, model_or_tool: Optional[str], conversation_id: Optional[str] = None) -> MetricTimer:
    """Convenience helper to measure elapsed time + submit a metric."""
    return MetricTimer(component=component, model_or_tool=model_or_tool, conversation_id=conversation_id)


def fetch_metrics(limit: int = 500) -> List[Dict[str, Any]]:
    """Return recent metrics from Supabase if available, otherwise from the local CSV."""
    client = _get_supabase_client()
    if client is not None:
        try:
            resp = client.table("metrics").select("*").order("timestamp", desc=True).limit(limit).execute()
            if resp.data:
                return resp.data
        except Exception:
            pass
    if not CSV_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                rows.append(row)
    except Exception:
        return []
    return rows


def summarize_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute total cost and average latency per component."""
    total_cost = 0.0
    latency_by_component: Dict[str, List[float]] = {}
    for row in records:
        cost = _coerce_number(row.get("cost_usd"))
        if cost:
            total_cost += cost
        latency = _coerce_number(row.get("latency_ms"))
        component = row.get("component") or "unknown"
        if latency is not None:
            latency_by_component.setdefault(component, []).append(latency)
    avg_latency = {
        comp: round(sum(vals) / len(vals), 3) for comp, vals in latency_by_component.items() if vals
    }
    return {
        "total_cost_usd": round(total_cost, 6),
        "average_latency_ms": avg_latency,
        "sample_size": len(records),
    }


@contextmanager
def timed_operation(component: str, model_or_tool: Optional[str], conversation_id: Optional[str] = None):
    """Context manager wrapper to log latency for arbitrary operations."""
    timer = start_timer(component, model_or_tool, conversation_id)
    try:
        yield timer
    finally:
        timer.done()
