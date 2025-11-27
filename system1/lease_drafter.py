"""
Utilities for drafting lease agreements and validating compliance rules.

The goal is not to produce a final, attorney-reviewed lease. Instead, this
module assembles a structured draft inspired by ``template.pdf`` and flags
obvious compliance issues (rent caps, deposit limits, missing clauses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import uuid4
from xml.sax.saxutils import escape
import string

MANDATORY_SECTIONS = [
    "1. PARTIES",
    "2. PREMISES",
    "3. TERM AND RENEWAL",
    "4. RENT, ADDITIONAL FEES, AND PAYMENT",
    "5. LATE PAYMENTS AND RETURNED FUNDS",
    "6. SECURITY DEPOSIT",
    "7. CONDITION OF PREMISES",
    "8. USE AND OCCUPANCY",
    "9. VEHICLES AND PARKING",
    "10. UTILITIES AND SERVICES",
    "11. MAINTENANCE, REPAIRS, AND ACCESS",
    "12. INSURANCE AND LIABILITY",
    "13. PETS",
    "14. DEFAULT AND REMEDIES",
    "15. ADDENDA AND RULES",
    "16. COMPLIANCE CONTEXT",
]

STATE_RULES: Dict[str, Dict[str, float]] = {
    # These values are simplified heuristics meant for quick validations.
    "CA": {"security_deposit_max_months": 2, "late_fee_cap_pct": 5, "pet_rent_cap": 75},
    "NY": {"security_deposit_max_months": 1, "late_fee_cap_pct": 5, "pet_rent_cap": 50},
    "MA": {"security_deposit_max_months": 1, "late_fee_cap_pct": 5},
}


def _clean_ascii(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode() if isinstance(value, str) else value


def _normalize_resident_name(value: str) -> str:
    if not value:
        return ""
    collapsed = " ".join(value.split())
    return string.capwords(collapsed)


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _extract_listing_rent(listing: Dict[str, object]) -> Optional[int]:
    """Pull the lowest advertised rent from a listing's pricing info."""
    pricing = listing.get("pricingAndFloorPlans") if isinstance(listing, dict) else None
    if not isinstance(pricing, list):
        return None

    candidates: List[int] = []
    for plan in pricing:
        if not isinstance(plan, dict):
            continue
        rent_label = plan.get("rent_label") or ""
        candidates.extend(int(match.replace(",", "")) for match in re.findall(r"\$?(\d[\d,]*)", rent_label))
        for unit in plan.get("units") or []:
            if isinstance(unit, dict):
                unit_price = unit.get("price")
                if unit_price:
                    matches = re.findall(r"\$?(\d[\d,]*)", str(unit_price))
                    candidates.extend(int(match.replace(",", "")) for match in matches)
    return min(candidates) if candidates else None


def _slug_to_title(slug: str) -> Optional[str]:
    cleaned = slug.strip().strip("/")
    if not cleaned:
        return None
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in re.split(r"[-_/]", cleaned)]
    tokens = [token for token in tokens if token]
    if not tokens:
        return None
    return " ".join(token.capitalize() for token in tokens)


def _city_state_tokens(prefs: Dict[str, object], listing: Dict[str, object]) -> Tuple[set, set]:
    city_tokens: set = set()
    state_tokens: set = set()
    city_pref = prefs.get("city")
    state_pref = prefs.get("state")

    def _add_city(text: Optional[str]) -> None:
        if not text:
            return
        for token in re.findall(r"[A-Za-z]+", text):
            if token:
                city_tokens.add(token.lower())

    def _add_state(text: Optional[str]) -> None:
        if not text:
            return
        cleaned = re.sub(r"[^A-Za-z]", "", text).lower()
        if not cleaned:
            return
        state_tokens.add(cleaned)
        if len(cleaned) == 2:
            state_tokens.add(cleaned.upper())
        else:
            abbr = STATE_ABBREVIATIONS.get(cleaned)
            if abbr:
                state_tokens.add(abbr.lower())
                state_tokens.add(abbr.upper())

    _add_city(str(city_pref) if city_pref else None)
    _add_state(str(state_pref) if state_pref else None)

    if not city_tokens or not state_tokens:
        about = listing.get("about") or {}
        location = about.get("location") or ""
        parts = [part.strip() for part in location.split(",") if part.strip()]
        if len(parts) >= 2:
            if not city_tokens:
                _add_city(parts[-2])
            if not state_tokens:
                _add_state(parts[-1])
    return city_tokens, state_tokens


def _strip_location_tokens(slug_tokens: List[str], city_tokens: set, state_tokens: set) -> List[str]:
    if not slug_tokens:
        return slug_tokens
    cleaned: List[str] = []
    for token in slug_tokens:
        temp = token
        lower = temp.lower()
        for suffix in STOPWORD_SUFFIXES:
            if lower.endswith(suffix) and len(temp) > len(suffix):
                temp = temp[: -len(suffix)]
                lower = temp.lower()
        if not temp:
            continue
        lower = temp.lower()
        if lower in city_tokens or lower in state_tokens or lower in STOPWORD_TOKENS:
            continue
        cleaned.append(temp)
    if cleaned:
        return cleaned
    fallback = [t for t in slug_tokens if t.lower() not in city_tokens and t.lower() not in state_tokens]
    if fallback:
        return fallback
    return slug_tokens


def _name_from_url(url: Optional[str], prefs: Dict[str, object], listing: Dict[str, object]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.netloc or "").split(":")[0].lower()
    host = host.lstrip("www.")
    path = parsed.path or ""
    city_tokens, state_tokens = _city_state_tokens(prefs, listing)
    if path:
        segments = [seg for seg in path.split("/") if seg]
        for segment in segments:
            tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in re.split(r"[-_/]", segment)]
            tokens = [token for token in tokens if token]
            tokens = _strip_location_tokens(tokens, city_tokens, state_tokens)
            if tokens and any(token.isalpha() for token in tokens):
                return " ".join(token.capitalize() for token in tokens)
    if host:
        base = host.split(".")[0]
        tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in re.split(r"[-_/]", base)]
        tokens = [token for token in tokens if token]
        tokens = _strip_location_tokens(tokens, city_tokens, state_tokens)
        if tokens:
            return " ".join(token.capitalize() for token in tokens)
    return None


def _name_from_property_website(website: str, prefs: Dict[str, object], listing: Dict[str, object]) -> Optional[str]:
    return _name_from_url(website, prefs, listing)


def _name_from_listing_url(listing: Dict[str, object], prefs: Dict[str, object]) -> Optional[str]:
    return _name_from_url(listing.get("url"), prefs, listing)


def _extract_landlord_name(listing: Dict[str, object], prefs: Dict[str, object]) -> str:
    """Approximate landlord/owner from property website or listing title."""
    contact = listing.get("contact") if isinstance(listing, dict) else None
    if isinstance(contact, dict):
        logo_url = contact.get("property_management_logo") or ""
        if logo_url:
            path = urlparse(logo_url).path
            stem = Path(path).stem
            stem = re.sub(r"-?logo$", "", stem, flags=re.IGNORECASE)
            company = _slug_to_title(stem)
            if company:
                return company
        website = contact.get("property_website")
        if website:
            company = _name_from_property_website(website, prefs, listing)
            if company:
                return company
    listing_url_name = _name_from_listing_url(listing, prefs)
    if listing_url_name:
        return listing_url_name
    about = listing.get("about") or {}
    fallback = about.get("title") or "Property Owner"
    return fallback


def _render_pdf(draft_text: str, pdf_path: Path) -> str:
    """Render draft text into a PDF using ReportLab."""
    try:
        from reportlab.lib.enums import TA_JUSTIFY
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("The 'reportlab' package is required for PDF export. Install it via `pip install reportlab`.") from exc

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "LeaseTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=24,
        spaceAfter=12,
        alignment=1,
    )
    meta_style = ParagraphStyle(
        "LeaseMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )
    section_style = ParagraphStyle(
        "LeaseSection",
        parent=styles["Heading4"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "LeaseBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
    )

    story = []
    for line in draft_text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 10))
            continue
        if stripped.endswith("LEASE AGREEMENT"):
            style = title_style
        elif stripped.startswith("Prepared on"):
            style = meta_style
        elif re.match(r"^\d+\.", stripped):
            style = section_style
        else:
            style = body_style
        story.append(Paragraph(escape(stripped), style))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="Lease Draft",
    )
    doc.build(story)
    return str(pdf_path)


def render_pdf_from_text(draft_text: str, pdf_path: str) -> str:
    """Public helper to render arbitrary draft text into a PDF."""
    path = Path(pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _render_pdf(draft_text, path)


def text_file_to_pdf(text_path: str, pdf_path: Optional[str] = None) -> str:
    """Convert an existing draft text file into a PDF."""
    source = Path(text_path)
    if not source.exists():
        raise FileNotFoundError(f"No such draft file: {text_path}")
    draft_text = source.read_text(encoding="ascii", errors="ignore")
    target = Path(pdf_path) if pdf_path else source.with_suffix(".pdf")
    return render_pdf_from_text(draft_text, str(target))


@dataclass
class LeaseDraftInputs:
    landlord_name: str = "ABC Properties"
    tenant_name: str = "Prospective Tenant"
    property_address: str = "TBD"
    city: str = ""
    state: str = ""
    zip_code: str = ""
    lease_start: date = field(default_factory=date.today)
    lease_term_months: int = 12
    monthly_rent: int = 0
    grace_period_days: int = 3
    late_fee_initial: int = 25
    late_fee_daily: int = 5
    payment_address: str = "426 Main Street, Anycity, USA"
    security_deposit: Optional[int] = None
    cleaning_fee: int = 200
    vehicle_limit: int = 1
    utilities_landlord: List[str] = field(default_factory=lambda: ["Water and sewer", "Garbage and trash disposal"])
    utilities_tenant: List[str] = field(
        default_factory=lambda: ["Electricity", "Gas", "Heating", "Telephone", "Internet", "All other services"]
    )
    pet_rent: int = 25
    pets_allowed: bool = True
    rent_cap: Optional[int] = None
    local_rent_cap_label: Optional[str] = None
    additional_clauses: Optional[str] = None
    selected_plan_name: Optional[str] = None
    selected_plan_details: Optional[str] = None
    selected_plan_availability: Optional[str] = None
    selected_plan_rent_label: Optional[str] = None
    selected_plan_deposit: Optional[str] = None
    selected_plan_price_per_person: bool = False
    selected_unit_label: Optional[str] = None
    selected_unit_sqft: Optional[str] = None
    selected_unit_price: Optional[int] = None
    selected_unit_availability: Optional[str] = None
    selected_unit_details: Optional[str] = None

    def __post_init__(self) -> None:
        for attr in (
            "landlord_name",
            "tenant_name",
            "property_address",
            "city",
            "state",
            "zip_code",
            "payment_address",
        ):
            val = getattr(self, attr, "")
            if isinstance(val, str):
                setattr(self, attr, _clean_ascii(val))
        if isinstance(self.tenant_name, str):
            self.tenant_name = _normalize_resident_name(self.tenant_name)
        self.utilities_landlord = [_clean_ascii(item) for item in self.utilities_landlord]
        self.utilities_tenant = [_clean_ascii(item) for item in self.utilities_tenant]
        for attr in (
            "selected_plan_name",
            "selected_plan_details",
            "selected_plan_availability",
            "selected_plan_rent_label",
            "selected_plan_deposit",
            "selected_unit_label",
            "selected_unit_sqft",
            "selected_unit_availability",
            "selected_unit_details",
        ):
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, _clean_ascii(val))
        if self.additional_clauses:
            self.additional_clauses = _clean_ascii(self.additional_clauses)
        if self.security_deposit is None:
            self.security_deposit = self.monthly_rent

    @property
    def lease_end(self) -> date:
        return self.lease_start + timedelta(days=30 * self.lease_term_months)

    @property
    def location_line(self) -> str:
        parts = [self.property_address]
        locality = ", ".join(p for p in [self.city, self.state] if p)
        if locality:
            parts.append(locality)
        if self.zip_code:
            parts.append(self.zip_code)
        return ", ".join(parts)


def _format_utilities(label: str, items: List[str]) -> str:
    if not items:
        return f"{label}: None specified."
    bullets = ", ".join(items)
    return f"{label}: {bullets}."


def generate_lease_text(inputs: LeaseDraftInputs) -> str:
    """Assemble a lease draft patterned after the Apartments.com sample lease."""
    execution_date = date.today().strftime("%B %d, %Y")
    lease_start = inputs.lease_start.strftime("%B %d, %Y")
    lease_end = inputs.lease_end.strftime("%B %d, %Y")
    rent_line = f"${inputs.monthly_rent:,}" if inputs.monthly_rent else "TBD"
    deposit_line = f"${inputs.security_deposit:,}" if inputs.security_deposit else "TBD"

    location_header = ", ".join(part for part in [inputs.city, inputs.state] if part)
    location_header = location_header or inputs.state or "City, State"

    additional_fees = [
        f"Cleaning fee (minimum): ${inputs.cleaning_fee:,}",
        f"Pet rent (if approved): ${inputs.pet_rent}/month" if inputs.pets_allowed else "Pets are not permitted.",
        f"Late fee: ${inputs.late_fee_initial} plus ${inputs.late_fee_daily} per day after day {inputs.grace_period_days + 1}",
        "Utility transfer/setup fees: As invoiced",
    ]

    plan_lines = []
    if inputs.selected_plan_name:
        plan_lines.append(f"   Selected floor plan: {inputs.selected_plan_name}.")
    if inputs.selected_plan_details:
        plan_lines.append(f"   Plan details: {inputs.selected_plan_details}.")
    if inputs.selected_plan_availability:
        plan_lines.append(f"   Availability reported by community: {inputs.selected_plan_availability}.")
    if inputs.selected_plan_deposit:
        plan_lines.append(f"   Plan-specific deposit: {inputs.selected_plan_deposit}.")
    if inputs.selected_unit_label:
        unit_line = f"   Selected unit: {inputs.selected_unit_label}"
        if inputs.selected_unit_sqft:
            unit_line += f" ({inputs.selected_unit_sqft} sq ft)"
        if inputs.selected_unit_price:
            unit_line += f" at ${inputs.selected_unit_price:,}/month"
        plan_lines.append(unit_line + ".")
        if inputs.selected_unit_availability:
            plan_lines.append(f"   Unit availability: {inputs.selected_unit_availability}.")
        if inputs.selected_unit_details:
            plan_lines.append(f"   Unit details: {inputs.selected_unit_details}.")

    lines = [
        "RESIDENTIAL LEASE AGREEMENT",
        location_header,
        f"Effective Date: {execution_date}",
        "",
        "THIS RESIDENTIAL LEASE AGREEMENT (\"Lease\") is entered into between Landlord and Resident(s).",
        "Landlord leases the Premises to Resident(s) for the term and consideration described below.",
        "",
        "1. PARTIES",
        f"   Landlord / Property Owner: {inputs.landlord_name}",
        f"   Resident(s): {inputs.tenant_name}",
        "   Additional Occupants: None reported (update if minors or roommates will live in the Premises).",
        "",
        "2. PREMISES",
        f"   Address: {inputs.location_line or 'To be determined'}",
        "   Premises are leased as a private residence; storage or business use is not permitted without written consent.",
        "",
        "3. TERM AND RENEWAL",
        (
            f"   Initial Term: {inputs.lease_term_months} month(s), commencing on {lease_start} and ending on {lease_end}."
            " Upon expiration, the Lease converts to a month-to-month tenancy unless either party provides at least"
            " thirty (30) days' written notice."
        ),
        "   Any renewal for a specific term must be in writing and signed by both parties.",
        "",
        "4. RENT, ADDITIONAL FEES, AND PAYMENT",
        f"   Monthly Rent: {rent_line}, due on or before the first day of each month with a {inputs.grace_period_days}-day grace period.",
        "   Acceptable payment methods include online payments via Apartments.com, personal check, cashier's check, or money order.",
        "   Prorated rent for partial months will be calculated on a daily basis (monthly rent divided by days in the month).",
    ]

    if plan_lines:
        lines.extend(["", "   Floor plan selection:"] + plan_lines)
    if inputs.selected_plan_price_per_person:
        lines.append("   Note: Community pricing for this plan is listed per person.")

    lines.append("   Utilities & Essentials: Additional fees (utilities packages, parking, pet fees, etc.) may apply per community policies.")

    lines.extend(
        [
            "",
            "5. LATE PAYMENTS AND RETURNED FUNDS",
            (
                "   Rent received after the grace period may incur late charges under Landlord's policy. Payments returned for insufficient funds"
                " may require certified funds for future payments."
            ),
            "",
            "6. SECURITY DEPOSIT",
            (
                "   Resident shall pay a refundable security deposit prior to move-in."
                " The deposit cannot be applied to rent without Landlord's consent and may be used to cover unpaid rent,"
                " damages beyond normal wear, unpaid utilities, cleaning, and other charges permitted by law."
                " An itemized disposition will be provided within the statutory time frame after move-out."
            ),
            "",
            "7. CONDITION OF PREMISES",
            (
                "   Resident acknowledges the right to inspect the Premises prior to possession and agrees the Premises"
                " (including appliances and fixtures) are in clean, safe condition unless otherwise noted in writing within 48 hours of move-in."
            ),
            "   Resident must maintain the Premises in a sanitary condition, refrain from unapproved alterations, and promptly report defects.",
            "",
            "8. USE AND OCCUPANCY",
            (
                "   The Premises shall be occupied solely by the Resident(s) and approved Occupants listed above."
                " Commercial activity, subletting, or short-term rentals (e.g., STR platforms) are prohibited without written consent."
            ),
            "   Conduct that disturbs neighbors or violates laws/ordinances constitutes a default.",
            "",
            "9. VEHICLES AND PARKING",
            f"   Resident may keep up to {inputs.vehicle_limit} operable, properly registered vehicle(s) in designated spaces.",
            "   Boats, trailers, or recreational vehicles require prior written permission.",
            "",
            "10. UTILITIES AND SERVICES",
            _format_utilities("   Landlord-provided utilities", inputs.utilities_landlord),
            _format_utilities("   Resident-responsible utilities", inputs.utilities_tenant),
            "   Resident must keep all utility accounts current throughout the Lease term.",
            "",
            "11. MAINTENANCE, REPAIRS, AND ACCESS",
            (
                "   Resident shall promptly notify Landlord of leaks, pest activity, electrical issues, or other conditions that could damage the property."
                " Landlord may access the Premises with reasonable notice for inspections, repairs, or as permitted by law."
            ),
            "",
            "12. INSURANCE AND LIABILITY",
            "   Resident is encouraged to maintain renter's insurance to cover personal property and liability losses.",
            "   Landlord is not responsible for Resident's personal belongings, vehicles, or guests, except as required by law.",
            "",
            "13. PETS",
            (
                "   Pets require prior written approval and a completed pet addendum. When approved, monthly pet rent may apply."
                " Service and support animals will be reasonably accommodated."
            )
            if inputs.pets_allowed
            else "13. PETS\n   Pets are not permitted on the Premises without an approved accommodation request.",
            "",
            "14. DEFAULT AND REMEDIES",
            (
                "   Failure to pay rent, maintain insurance when required, or comply with Lease obligations constitutes a default."
                " Landlord may pursue all remedies available under state law, including termination, eviction, and recovery of damages."
            ),
            "",
            "15. ADDENDA AND RULES",
            (
                "   Community policies, HOA rules, or addenda (parking, pet, mold, lead-based paint, etc.) are incorporated by reference."
                " Resident agrees to follow all published rules and acknowledges receipt of required disclosures."
            ),
            "",
            "16. COMPLIANCE CONTEXT",
            (
                f"   This draft references {inputs.local_rent_cap_label or 'local rent guidance'}."
                " Verify city- and state-specific statutes (rent caps, notice requirements, deposit limits) prior to execution."
            ),
        ]
    )

    if inputs.additional_clauses:
        lines.extend(["", "17. ADDITIONAL CLAUSES", inputs.additional_clauses])

    lines.extend(
        [
            "",
            "Accepted on ______________________",
            "Resident: ________________________   Date: ____________",
            "Landlord/Manager: ________________   Date: ____________",
            "",
            "This draft was generated with AI assistance and should be reviewed by all parties before signing.",
        ]
    )
    return "\n".join(lines)


def compliance_report(inputs: LeaseDraftInputs, draft_text: str) -> Dict[str, List[str]]:
    """Return compliance issues and warnings for the generated draft."""
    issues: List[str] = []
    warnings: List[str] = []
    rules_checked: List[str] = []

    state_rules = STATE_RULES.get(inputs.state.upper())
    if state_rules:
        rules_checked.append(f"{inputs.state.upper()} rental caps")
        max_months = state_rules.get("security_deposit_max_months")
        if max_months and inputs.security_deposit and inputs.monthly_rent:
            if inputs.security_deposit > inputs.monthly_rent * max_months:
                issues.append(
                    f"Security deposit ${inputs.security_deposit:,} exceeds {max_months} month(s) of rent "
                    f"allowed in {inputs.state.upper()}."
                )
        late_fee_cap = state_rules.get("late_fee_cap_pct")
        if late_fee_cap and inputs.monthly_rent:
            total_late_fee = inputs.late_fee_initial + inputs.late_fee_daily
            cap_value = inputs.monthly_rent * (late_fee_cap / 100)
            if total_late_fee > cap_value:
                warnings.append(
                    f"Combined late fees (${total_late_fee}) exceed {late_fee_cap}% of monthly rent "
                    f"(${cap_value:.2f}) referenced by {inputs.state.upper()} guidance."
                )
        pet_cap = state_rules.get("pet_rent_cap")
        if pet_cap and inputs.pet_rent > pet_cap:
            warnings.append(
                f"Monthly pet rent ${inputs.pet_rent} is above the suggested cap (${pet_cap}) for {inputs.state.upper()}."
            )

    if inputs.rent_cap and inputs.monthly_rent and inputs.monthly_rent > inputs.rent_cap:
        issues.append(
            f"Monthly rent ${inputs.monthly_rent:,} exceeds the configured rent cap (${inputs.rent_cap:,})."
        )

    for section in MANDATORY_SECTIONS:
        if section not in draft_text:
            issues.append(f"Mandatory clause '{section}' missing from draft.")

    if not inputs.monthly_rent:
        warnings.append("Monthly rent is set to TBD. Update before sending to tenants.")
    if not inputs.property_address or not inputs.city or not inputs.state:
        warnings.append("Property address is incomplete.")

    return {"issues": issues, "warnings": warnings, "rules_checked": rules_checked}


def build_lease_package(inputs: LeaseDraftInputs, output_dir: str = "system1/lease_drafts") -> Dict[str, object]:
    """Generate the lease draft, run compliance checks, and persist the file."""
    draft_text = generate_lease_text(inputs)
    report = compliance_report(inputs, draft_text)

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    city_slug = inputs.city.lower().replace(" ", "_") if inputs.city else "lease"
    username = inputs.tenant_name or "user"
    username_slug = re.sub(r"[^a-zA-Z0-9]+", "_", username.strip()).strip("_") or "user"
    unique_id = uuid4().hex[:8]
    filename_base = f"{username_slug}_{unique_id}_lease_draft"
    file_path = path / f"{filename_base}.txt"
    file_path.write_text(draft_text, encoding="ascii", errors="strict")
    pdf_path = file_path.with_suffix(".pdf")
    render_pdf_from_text(draft_text, str(pdf_path))

    return {
        "file_path": str(file_path),
        "pdf_path": str(pdf_path),
        "draft": draft_text,
        "compliance": report,
    }


def infer_inputs(
    *,
    preferences: Optional[Dict[str, object]] = None,
    listing: Optional[Dict[str, object]] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> LeaseDraftInputs:
    """
    Build ``LeaseDraftInputs`` from conversation preferences and an optional listing.
    ``overrides`` lets callers inject CLI-provided fields.
    """

    prefs = preferences or {}
    info: Dict[str, object] = {}
    info["city"] = _clean_ascii(str(prefs.get("city"))) if prefs.get("city") else ""
    info["state"] = _clean_ascii(str(prefs.get("state"))) if prefs.get("state") else ""
    monthly_pref = _parse_int(prefs.get("max_rent")) or _parse_int(prefs.get("min_rent")) or 0
    info["monthly_rent"] = monthly_pref or 0
    info["rent_cap"] = _parse_int(prefs.get("max_rent"))
    label_parts = [part for part in [info["city"], info["state"], "rent cap"] if part]
    info["local_rent_cap_label"] = " ".join(label_parts) if label_parts else "local rent guidance"
    first_pref = prefs.get("tenant_first_name")
    last_pref = prefs.get("tenant_last_name")
    tenant_pref_name = prefs.get("tenant_name")
    combined_pref = None
    if first_pref or last_pref:
        combined_pref = " ".join(part for part in [first_pref, last_pref] if part)
    name_source = combined_pref or tenant_pref_name
    if name_source:
        info["tenant_name"] = _normalize_resident_name(_clean_ascii(str(name_source)))

    if listing:
        about = listing.get("about") or {}
        info["landlord_name"] = _extract_landlord_name(listing, prefs)
        location = about.get("location")
        if location:
            info["property_address"] = location
        else:
            breadcrumbs = about.get("breadcrumbs") or []
            if breadcrumbs:
                info["property_address"] = ", ".join(breadcrumbs[-3:])
        listing_rent = _extract_listing_rent(listing)
        if listing_rent:
            info["monthly_rent"] = listing_rent
    lease_start_str = None
    lease_term_override = None
    if overrides:
        if overrides.get("tenant_name"):
            overrides["tenant_name"] = _normalize_resident_name(_clean_ascii(str(overrides["tenant_name"])))
        lease_start_str = overrides.pop("lease_start_date", None)
        lease_term_override = overrides.pop("lease_term_months", None)
    if not lease_start_str:
        lease_start_str = prefs.get("lease_start_date")
    if lease_start_str:
        try:
            info["lease_start"] = date.fromisoformat(str(lease_start_str))
        except Exception:
            pass
    lease_term_value = lease_term_override or prefs.get("lease_duration_months")
    if lease_term_value:
        try:
            info["lease_term_months"] = int(lease_term_value)
        except Exception:
            pass

    if overrides:
        info.update({k: v for k, v in overrides.items() if v is not None})

    return LeaseDraftInputs(**info)
STOPWORD_TOKENS = {
    "apartments",
    "apartment",
    "apts",
    "apt",
    "residences",
    "residence",
    "living",
    "homes",
    "home",
    "property",
    "properties",
    "group",
    "llc",
    "inc",
    "corp",
    "company",
    "co",
    "community",
}

STOPWORD_SUFFIXES = [
    "apartments",
    "apartment",
    "apts",
    "apt",
    "residences",
    "residence",
    "living",
    "homes",
    "home",
    "dtla",
    "tx",
    "ca",
    "az",
    "ny",
    "ga",
]
