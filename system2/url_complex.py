"""
Placeholder for Lebanon search URL/query builder.

Mirror of System 1's `url_complex.py`, to be customized for Lebanon data
sources and listing providers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ApartmentsSearchQuery:
    """Describe a future Lebanon listing query (fields TBD)."""

    city: Optional[str] = None
    district: Optional[str] = None
    min_rent: Optional[int] = None
    max_rent: Optional[int] = None

    def build_url(self) -> str:
        """TODO: Translate query fields into a provider-specific URL."""
        raise NotImplementedError("System 2 URL builder is pending implementation.")
