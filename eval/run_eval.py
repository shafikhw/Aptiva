"""Run evaluation tasks against System 1 and System 2 agents without HTTP."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from system1.session import System1AgentSession
from system2.session import System2AgentSession


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EVAL_DIR / "results.json"
BIAS_TASKS_PATH = EVAL_DIR / "bias_checks.json"
DEFAULT_BASELINE = "full-agent"


def _load_tasks(path: Path, system_label: str, task_kind: str = "main") -> List[Dict[str, Any]]:
    """Load tasks and tag them with the intended system label."""
    if not path.exists():
        raise FileNotFoundError(f"Missing task file: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks = []
    for idx, task in enumerate(raw_tasks, start=1):
        tasks.append(
            {
                "id": task.get("id") or f"{system_label}-{idx:02d}",
                "system": "us" if system_label == "us" else "lb",
                "user_profile": task.get("user_profile", "").strip(),
                "query": task.get("query", "").strip(),
                "expected_type": task.get("expected_type", "").strip(),
                "task_kind": task_kind,
            }
        )
    return tasks


def _load_bias_tasks(path: Path) -> List[Dict[str, Any]]:
    """Load bias probe tasks; the system must be specified per entry."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks = []
    for idx, task in enumerate(raw_tasks, start=1):
        system = (task.get("system") or "").strip().lower() or "us"
        tasks.append(
            {
                "id": task.get("id") or f"bias-{idx:02d}",
                "system": system,
                "user_profile": task.get("user_profile", "").strip(),
                "query": task.get("query", "").strip(),
                "expected_type": task.get("expected_type", "").strip(),
                "task_kind": "bias",
            }
        )
    return tasks


def _combine_prompt(user_profile: str, query: str) -> str:
    profile_line = f"Profile: {user_profile}" if user_profile else ""
    if profile_line:
        return f"{profile_line}\nRequest: {query}"
    return query


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion of agent outputs to JSON-safe structures."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _extract_locations_from_query(text: str) -> List[str]:
    """Use simple capitalization heuristics to pull likely location names."""
    if not text:
        return []
    matches = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", text)
    return [m.strip() for m in matches if m.strip()]


def _score_accuracy(task: Dict[str, Any], reply: str) -> Dict[str, Any]:
    """Heuristic scoring for whether the reply hit expected intent."""
    checks: Dict[str, bool] = {}
    if not reply:
        return {"score": 0.0, "checks": checks}

    lower_reply = reply.lower()
    lower_query = (task.get("query") or "").lower()
    lower_expect = (task.get("expected_type") or "").lower()

    checks["mentions_listing"] = any(
        token in lower_reply for token in ["listing", "apartment", "unit", "property", "option"]
    )
    checks["has_link_or_price"] = "http" in lower_reply or "$" in reply

    wants_lease = "lease" in lower_query or "lease" in lower_expect
    if wants_lease:
        checks["mentions_lease"] = "lease" in lower_reply

    wants_viewing = any(keyword in lower_query for keyword in ["viewing", "tour", "schedule", "book"])
    if wants_viewing:
        checks["mentions_viewing"] = any(
            token in lower_reply for token in ["viewing", "tour", "schedule", "slot", "book"]
        )

    locations = _extract_locations_from_query(task.get("query", ""))
    if locations:
        checks["mentions_location"] = any(loc.lower() in lower_reply for loc in locations)

    total = len(checks) or 1
    score = sum(1 for hit in checks.values() if hit) / total
    return {"score": round(score, 3), "checks": checks}


def _extract_cost_blob(candidate: Any) -> Dict[str, Any]:
    """Best-effort extraction of cost/usage telemetry from a dict-like object."""
    if not isinstance(candidate, dict):
        return {}
    telemetry = candidate.get("telemetry") or {}
    usage = telemetry.get("usage") or telemetry.get("token_usage") or candidate.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("prompt") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("completion") or usage.get(
        "output_tokens"
    )
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    cost_usd = telemetry.get("cost_usd") or telemetry.get("cost") or candidate.get("cost_usd")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def _merge_cost(cost_a: Dict[str, Any], cost_b: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**cost_a}
    for key, value in cost_b.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
    return merged


def _task_metrics(task: Dict[str, Any], reply: str, latency: float, state: Any, response: Dict[str, Any]) -> Dict[str, Any]:
    accuracy = _score_accuracy(task, reply or "")
    cost_from_response = _extract_cost_blob(response)
    cost_from_state = _extract_cost_blob(state if isinstance(state, dict) else {})
    cost = _merge_cost(cost_from_response, cost_from_state)
    reliability = bool(reply and str(reply).strip())
    return {
        "accuracy": accuracy,
        "latency_seconds": round(latency, 3),
        "reliability": reliability,
        "cost": cost,
    }


def _run_task(task: Dict[str, Any], baseline: str) -> Dict[str, Any]:
    prompt = _combine_prompt(task.get("user_profile", ""), task.get("query", ""))
    agent = System1AgentSession() if task.get("system") == "us" else System2AgentSession()
    start = time.perf_counter()
    try:
        response = agent.send(prompt)
        latency = time.perf_counter() - start
        reply = response.get("reply")
        state = response.get("state")
        return {
            **task,
            "baseline": baseline,
            "input_prompt": prompt,
            "reply": reply,
            "state": _json_safe(state),
            "preferences": _json_safe(response.get("preferences")),
            "conversation_complete": bool(response.get("conversation_complete")),
            "metrics": _task_metrics(task, reply, latency, state, response),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - start
        return {
            **task,
            "baseline": baseline,
            "input_prompt": prompt,
            "reply": None,
            "state": None,
            "preferences": None,
            "conversation_complete": False,
            "metrics": {
                "accuracy": {"score": 0.0, "checks": {}},
                "latency_seconds": round(latency, 3),
                "reliability": False,
                "cost": {},
            },
            "error": str(exc),
        }


def _mean(values: List[float]) -> float:
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _percentile(values: List[float], pct: float) -> float:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return 0.0
    k = max(0, min(len(clean) - 1, int(round(pct * (len(clean) - 1)))))
    return clean[k]


def _summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(group)
    accuracy_scores = [item.get("metrics", {}).get("accuracy", {}).get("score") for item in group]
    reliability_flags = [1 if item.get("metrics", {}).get("reliability") else 0 for item in group]
    latencies = [item.get("metrics", {}).get("latency_seconds") for item in group]
    costs = [
        item.get("metrics", {}).get("cost", {}).get("cost_usd")
        for item in group
        if item.get("metrics", {}).get("cost")
    ]
    errors = [item for item in group if item.get("error")]

    return {
        "count": count,
        "accuracy_avg": round(_mean(accuracy_scores), 3),
        "reliability": round(_mean(reliability_flags), 3),
        "latency_avg_seconds": round(_mean(latencies), 3),
        "latency_p95_seconds": round(_percentile(latencies, 0.95), 3),
        "cost_total_usd": round(sum(c for c in costs if c is not None), 6),
        "cost_avg_usd": round(_mean([c for c in costs if c is not None]), 6),
        "num_errors": len(errors),
    }


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _by(key_fn) -> Dict[str, Any]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in results:
            key = key_fn(item)
            buckets.setdefault(key, []).append(item)
        return {name: _summarize(items) for name, items in buckets.items()}

    return {
        "overall": _summarize(results),
        "by_system": _by(lambda item: item.get("system", "unknown")),
        "by_baseline": _by(lambda item: item.get("baseline", DEFAULT_BASELINE)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aptiva eval tasks locally.")
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help="Label to tag this run (e.g., full-agent, llm-only, tool-constrained).",
    )
    parser.add_argument(
        "--results-path",
        default=str(RESULTS_PATH),
        help="Where to write the JSON results.",
    )
    parser.add_argument(
        "--system",
        choices=["all", "us", "lb"],
        default="all",
        help="Optional filter to run only US or Lebanon tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks: List[Dict[str, Any]] = []
    if args.system in ("all", "us"):
        tasks.extend(_load_tasks(EVAL_DIR / "tasks_us.json", "us"))
    if args.system in ("all", "lb"):
        tasks.extend(_load_tasks(EVAL_DIR / "tasks_lb.json", "lb"))
    bias_tasks = _load_bias_tasks(BIAS_TASKS_PATH)
    if args.system != "all":
        bias_tasks = [task for task in bias_tasks if task.get("system") == args.system]

    results: List[Dict[str, Any]] = []
    for task in tasks:
        results.append(_run_task(task, args.baseline))

    bias_results: List[Dict[str, Any]] = []
    for task in bias_tasks:
        bias_results.append(_run_task(task, args.baseline))

    summary = _aggregate(results)
    bias_summary = _aggregate(bias_results) if bias_results else {}
    payload = {
        "run_id": f"{args.baseline}-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        "baseline": args.baseline,
        "run_started": datetime.utcnow().isoformat() + "Z",
        "num_tasks": len(results),
        "num_bias_tasks": len(bias_results),
        "results": results,
        "bias_results": bias_results,
        "summary": summary,
        "bias_summary": bias_summary,
    }
    results_path = Path(args.results_path)
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {len(results)} main results"
        f"{' and ' + str(len(bias_results)) + ' bias probes' if bias_results else ''} to {results_path}"
    )


if __name__ == "__main__":
    main()
