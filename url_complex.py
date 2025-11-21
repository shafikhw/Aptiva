from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import math

BASE_URL = "https://www.apartments.com"


class PropertyType(str, Enum):
    """
    How Apartments.com scopes the search.

    APARTMENTS uses the default city page:
        https://www.apartments.com/los-angeles-ca/
    HOUSES / CONDOS / TOWNHOMES use a prefix:
        https://www.apartments.com/houses/los-angeles-ca/
    LOFTS use a trailing slug:
        https://www.apartments.com/los-angeles-ca/lofts/
    """
    APARTMENTS = "apartments"
    HOUSES = "houses"
    CONDOS = "condos"
    TOWNHOMES = "townhomes"
    LOFTS = "lofts"


class Lifestyle(str, Enum):
    """
    Lifestyle specific city pages:
        /city-state/student-housing/
        /city-state/senior-housing/
        /city-state/corporate/
        /city-state/military/
        /city-state/short-term/
    """
    STUDENT = "student-housing"
    SENIOR = "senior-housing"
    CORPORATE = "corporate"
    MILITARY = "military"
    SHORT_TERM = "short-term"


class PetType(str, Enum):
    DOG = "dog"
    CAT = "cat"


def slugify_location(city: str, state: str) -> str:
    """
    Build an Apartments.com slug from a city and state.

    Examples:
        city="Los Angeles", state="CA" -> 'los-angeles-ca'
        city="Midtown Atlanta, Atlanta", state="GA" -> 'midtown-atlanta-atlanta-ga'
    """
    city = city.strip()
    state = state.strip()
    if not city or not state:
        raise ValueError("Both city and state are required to build a location slug.")

    text = f"{city}, {state}".lower()
    # Treat punctuation as spaces
    text = text.replace(",", " ")
    normalized_chars: List[str] = []
    for ch in text:
        if ch.isalnum():
            normalized_chars.append(ch)
        else:
            normalized_chars.append(" ")
    slug = "-".join(part for part in "".join(normalized_chars).split() if part)
    if not slug:
        raise ValueError(f"Could not build slug from location {text!r}")
    return slug


def build_bed_segment(min_beds: Optional[int],
                      max_beds: Optional[int]) -> Optional[str]:
    """
    Known patterns:
        2-bedrooms
        min-2-bedrooms
        max-2-bedrooms
        1-to-2-bedrooms
    """
    if min_beds is None and max_beds is None:
        return None

    if min_beds is not None and max_beds is not None:
        if min_beds == max_beds:
            return f"{min_beds}-bedrooms"
        return f"{min_beds}-to-{max_beds}-bedrooms"

    if min_beds is not None:
        return f"min-{min_beds}-bedrooms"

    return f"max-{max_beds}-bedrooms"


def build_bath_segment(min_baths: Optional[float],
                       max_baths: Optional[float]) -> Optional[str]:
    """
    Known pattern:
        1-bathrooms
        2-bathrooms
    Used as a minimum baths filter (1+, 2+, etc).
    """
    if min_baths is None and max_baths is None:
        return None

    value = min_baths if min_baths is not None else max_baths
    if value is None:
        return None

    value_int = int(math.floor(value))
    if value_int <= 0:
        return None
    return f"{value_int}-bathrooms"


def build_price_segment(min_rent: Optional[int],
                        max_rent: Optional[int]) -> Optional[str]:
    """
    Known patterns:
        /city-state/under-1500/
        /city-state/600-to-1200/
        /houses/city-state/under-3000/
    """
    if min_rent is not None and max_rent is not None:
        if min_rent <= 0 < max_rent:
            # Treat as simple cap; this pattern exists everywhere.
            return f"under-{max_rent}"
        return f"{min_rent}-to-{max_rent}"

    if max_rent is not None:
        return f"under-{max_rent}"

    # No reliable slug for min_rent only
    return None


def build_pet_segment(pet_friendly: bool,
                      pet_type: Optional[PetType]) -> Optional[str]:
    """
    Known patterns:
        pet-friendly
        pet-friendly-dog
        pet-friendly-cat
    """
    if not pet_friendly:
        return None
    if pet_type is None:
        return "pet-friendly"
    return f"pet-friendly-{pet_type.value}"


def build_numeric_filter_slug(
    min_beds: Optional[int],
    max_beds: Optional[int],
    min_baths: Optional[float],
    max_baths: Optional[float],
    min_rent: Optional[int],
    max_rent: Optional[int],
    pet_friendly: bool,
    pet_type: Optional[PetType],
) -> Optional[str]:
    """
    Assemble combined numeric slug in the exact order used by Apartments.com:
        [beds]-[baths]-[price]-[pet]

    Examples that this can recreate:
        min-2-bedrooms-1-bathrooms-under-1200-pet-friendly-dog
        1-bedrooms-1-bathrooms-under-1000-pet-friendly-dog
        1-bedrooms-1-bathrooms-under-1100
        600-to-1200
    """
    parts: List[str] = []

    bed = build_bed_segment(min_beds, max_beds)
    if bed:
        parts.append(bed)

    bath = build_bath_segment(min_baths, max_baths)
    if bath:
        parts.append(bath)

    price = build_price_segment(min_rent, max_rent)
    if price:
        parts.append(price)

    pet = build_pet_segment(pet_friendly, pet_type)
    if pet:
        parts.append(pet)

    if not parts:
        return None

    return "-".join(parts)


@dataclass
class ApartmentsSearchQuery:
    """
    High level description:

    - Uses only path patterns that appear on real Apartments.com pages.
    - Encodes in the URL:
        * city / neighborhood slug built from city and state
        * property type (apartments, houses, condos, townhomes, lofts)
        * beds, baths
        * price min and max (where supported)
        * pet friendly, dog or cat specific
        * lifestyle pages (student, senior, corporate, military, short term)
        * rooms-for-rent pages
        * cheap city pages
        * utilities-included city pages
        * near-me base and cheap / utilities variants

    - Keeps as metadata only (not encoded in URL, because they are handled
      internally by Apartments.com):
        * keyword search
        * FRBO / private landlord toggle
        * ratings
        * commute time
        * drawn map shapes
        * sort order
    """

    # Location vs near-me mode
    city: Optional[str] = None
    state: Optional[str] = None
    near_me: bool = False

    # Property type
    property_type: Optional[PropertyType] = None

    # Beds and baths
    min_beds: Optional[int] = None
    max_beds: Optional[int] = None
    min_baths: Optional[float] = None
    max_baths: Optional[float] = None

    # Price (monthly rent)
    min_rent: Optional[int] = None
    max_rent: Optional[int] = None

    # Lifestyle pages
    lifestyle: Optional[Lifestyle] = None

    # Pets
    pet_friendly: bool = False
    pet_type: Optional[PetType] = None

    # Other city level filters that have their own pages
    cheap_only: bool = False          # /city-state/cheap/
    utilities_included: bool = False  # /city-state/utilities-included/

    # Amenity pages like /city-state/washer-dryer/ or /city-state/yard/
    # Only the first slug is used and only when there are no numeric filters.
    amenity_slugs: List[str] = field(default_factory=list)

    # Rooms for rent page: /city-state/rooms-for-rent/
    rooms_for_rent: bool = False

    # Metadata that is not encoded in the URL (left here for your scraper logic)
    keyword: Optional[str] = None
    frbo_only: bool = False
    rating_min: Optional[float] = None
    commute_minutes_max: Optional[int] = None

    # Pagination (page 1 is the base URL)
    page: int = 1

    def build_url(self) -> str:
        """
        Build a Apartments.com search URL.

        Examples:

        ApartmentsSearchQuery(
            city="Los Angeles",
            state="CA",
            max_rent=1500,
        ).build_url()
          -> https://www.apartments.com/los-angeles-ca/under-1500/

        ApartmentsSearchQuery(
            city="Los Angeles",
            state="CA",
            property_type=PropertyType.HOUSES,
            max_rent=3000,
        ).build_url()
          -> https://www.apartments.com/houses/los-angeles-ca/under-3000/

        ApartmentsSearchQuery(
            city="Kerrville",
            state="TX",
            min_beds=2,
            min_baths=1,
            max_rent=1200,
            pet_friendly=True,
            pet_type=PetType.DOG,
        ).build_url()
          -> https://www.apartments.com/kerrville-tx/min-2-bedrooms-1-bathrooms-under-1200-pet-friendly-dog/

        ApartmentsSearchQuery(
            city="Miami",
            state="FL",
            lifestyle=Lifestyle.STUDENT,
        ).build_url()
          -> https://www.apartments.com/miami-fl/student-housing/

        ApartmentsSearchQuery(
            city="New York",
            state="NY",
            rooms_for_rent=True,
        ).build_url()
          -> https://www.apartments.com/new-york-ny/rooms-for-rent/

        ApartmentsSearchQuery(
            near_me=True,
        ).build_url()
          -> https://www.apartments.com/near-me/apartments-for-rent/

        ApartmentsSearchQuery(
            near_me=True,
            cheap_only=True,
        ).build_url()
          -> https://www.apartments.com/near-me/cheap-apartments-for-rent/
        """
        if self.near_me:
            path = self._build_near_me_path()
        else:
            path = self._build_location_path()
        return BASE_URL + path

    # Internal helpers

    def _build_location_path(self) -> str:
        if not self.city:
            raise ValueError("city is required when near_me is False")
        if not self.state:
            raise ValueError("state is required when near_me is False")

        segments: List[str] = []

        # Houses, condos, townhomes use a prefix; apartments and lofts do not.
        if self.property_type in {
            PropertyType.HOUSES,
            PropertyType.CONDOS,
            PropertyType.TOWNHOMES,
        }:
            segments.append(self.property_type.value)

        loc_slug = slugify_location(self.city, self.state)
        segments.append(loc_slug)

        trailing: Optional[str] = None

        # Lofts use /city-state/lofts/
        if self.property_type is PropertyType.LOFTS:
            trailing = "lofts"

        # Rooms for rent has its own page and does not combine with numeric slug
        if self.rooms_for_rent:
            trailing = "rooms-for-rent"

        # Lifestyle pages also have their own path and are treated as a mode
        elif self.lifestyle is not None:
            trailing = self.lifestyle.value

        # Dedicated city "utilities included" page when not mixing numeric filters
        elif self.utilities_included and not any(
            [
                self.min_beds,
                self.max_beds,
                self.min_baths,
                self.max_baths,
                self.min_rent,
                self.max_rent,
                self.pet_friendly,
                self.cheap_only,
            ]
        ):
            trailing = "utilities-included"

        # Cheap page: /city-state/cheap/
        elif self.cheap_only and not any(
            [
                self.min_beds,
                self.max_beds,
                self.min_baths,
                self.max_baths,
                self.min_rent,
                self.max_rent,
                self.pet_friendly,
            ]
        ):
            trailing = "cheap"

        # Amenity only page like /city-state/washer-dryer/ or /city-state/yard/
        elif self.amenity_slugs and not any(
            [
                self.min_beds,
                self.max_beds,
                self.min_baths,
                self.max_baths,
                self.min_rent,
                self.max_rent,
                self.pet_friendly,
                self.cheap_only,
            ]
        ):
            trailing = self.amenity_slugs[0]

        # Otherwise build numeric slug (beds, baths, price, pet) if possible
        if trailing is None:
            numeric = build_numeric_filter_slug(
                self.min_beds,
                self.max_beds,
                self.min_baths,
                self.max_baths,
                self.min_rent,
                self.max_rent,
                self.pet_friendly,
                self.pet_type,
            )
            if numeric:
                trailing = numeric

        if trailing:
            segments.append(trailing)

        # Pagination segment: /.../2/
        if self.page and self.page > 1:
            segments.append(str(self.page))

        return "/" + "/".join(segments) + "/"

    def _build_near_me_path(self) -> str:
        """
        Known near me pages:
            /near-me/apartments-for-rent/
            /near-me/cheap-apartments-for-rent/
            /near-me/utilities-included-apartments/
        Beds, baths, and price filters for near me are handled by internal
        state on Apartments.com, not by path segments, so they are not encoded.
        """
        segments: List[str] = ["near-me"]

        if self.cheap_only:
            segments.append("cheap-apartments-for-rent")
        elif self.utilities_included:
            segments.append("utilities-included-apartments")
        else:
            segments.append("apartments-for-rent")

        if self.page and self.page > 1:
            segments.append(str(self.page))

        return "/" + "/".join(segments) + "/"


if __name__ == "__main__":
    # A few quick sanity checks against real site patterns.

    cases = [
        ApartmentsSearchQuery(city="Los Angeles", state="CA"),
        ApartmentsSearchQuery(city="Los Angeles", state="CA", max_rent=1500),
        ApartmentsSearchQuery(
            city="Los Angeles",
            state="CA",
            property_type=PropertyType.HOUSES,
            max_rent=3000,
        ),
        ApartmentsSearchQuery(
            city="Kerrville",
            state="TX",
            min_beds=2,
            min_baths=1,
            max_rent=1200,
            pet_friendly=True,
            pet_type=PetType.DOG,
        ),
        ApartmentsSearchQuery(
            city="West Lafayette",
            state="IN",
            min_beds=1,
            min_baths=1,
            max_rent=1000,
            pet_friendly=True,
            pet_type=PetType.DOG,
        ),
        ApartmentsSearchQuery(
            city="Camp Hill",
            state="PA",
            min_beds=1,
            max_beds=2,
            min_baths=1,
            max_rent=950,
        ),
        ApartmentsSearchQuery(
            city="Miami",
            state="FL",
            lifestyle=Lifestyle.STUDENT,
        ),
        ApartmentsSearchQuery(
            city="New York",
            state="NY",
            rooms_for_rent=True,
        ),
        ApartmentsSearchQuery(
            near_me=True,
        ),
        ApartmentsSearchQuery(
            near_me=True,
            cheap_only=True,
        ),
        ApartmentsSearchQuery(
            city="New York",
            state="NY",
            amenity_slugs=["washer-dryer"],
        ),
        ApartmentsSearchQuery(
            city="New York",
            state="NY",
            utilities_included=True,
        ),
    ]

    for q in cases:
        print(q.build_url())
