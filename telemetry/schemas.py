from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from telemetry.logging_utils import get_logger

logger = get_logger(__name__)


class Listing(BaseModel):
    id: Optional[str] = None
    url: Optional[str] = None
    about: Dict[str, Any] = Field(default_factory=dict)
    location: Dict[str, Any] = Field(default_factory=dict)
    pricingAndFloorPlans: Optional[List[Dict[str, Any]]] = None
    feesAndPolicies: Optional[Dict[str, Any]] = None
    contact: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


def normalize_listing_payload(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("items", "listings", "results", "data"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    return []


def validate_listings(raw: Any, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Validate and coerce listing payloads, logging and dropping invalid entries."""
    payload = normalize_listing_payload(raw)
    cleaned: List[Dict[str, Any]] = []
    for entry in payload:
        try:
            model = Listing.model_validate(entry.get("data") if isinstance(entry, dict) and "data" in entry else entry)
            data = model.model_dump()
            if isinstance(entry, dict) and "data" in entry and entry.get("id") and not data.get("_source_id"):
                data["_source_id"] = entry.get("id")
            cleaned.append(data)
        except ValidationError as exc:
            logger.warning("listing_validation_failed", extra={"error": str(exc)[:200]})
            continue
    if limit is not None:
        cleaned = cleaned[:limit]
    return cleaned
