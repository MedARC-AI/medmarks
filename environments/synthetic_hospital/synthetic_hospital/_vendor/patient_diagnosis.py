import json
import logging
import re
from enum import Enum
from .diagnosis import normalize_icd10, _strip_markdown
log = logging.getLogger(__name__)

class ParseTier(Enum):
    A = "full_schema_match"
    B = "partial_match"
    C = "unparseable"

ACUITY_NORMALIZATION = {
    "acute": "acute",
    "chronic": "chronic",
    "acute_on_chronic": "acute_on_chronic",
    "subacute": "acute",
    "resolving": "acute",
    "stable": "chronic",
    "exacerbation": "acute_on_chronic",
    "flare": "acute_on_chronic",
    "acute on chronic": "acute_on_chronic",
    "acute-on-chronic": "acute_on_chronic",
}

def parse_output(raw_text: str) -> dict:
    """Parse model output into structured patient diagnosis prediction.

    Returns dict with active_diagnoses, chronic_conditions, and _parse_tier.
    """
    text = _strip_markdown(raw_text)

    data = None
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                pass

    if data is None:
        return {"active_diagnoses": [], "chronic_conditions": [], "_parse_tier": ParseTier.C.value}

    active = data.get("active_diagnoses", [])
    chronic = data.get("chronic_conditions", [])

    # Tier B: flat "diagnoses" list → treat as active
    if not active and not chronic and "diagnoses" in data:
        active = data["diagnoses"]
        log.warning("Tier B parse: flat 'diagnoses' list used as active_diagnoses")

    # Determine tier
    tier = ParseTier.A
    if "active_diagnoses" not in data or "chronic_conditions" not in data:
        tier = ParseTier.B

    # Normalize entries
    norm_active = _normalize_entries(active, default_acuity="acute")
    norm_chronic = _normalize_entries(chronic, default_acuity="chronic")

    # Validate all have icd10 and acuity
    for entry in norm_active + norm_chronic:
        if not entry.get("icd10"):
            tier = ParseTier.B

    # Deduplicate
    norm_active, norm_chronic = _dedup_across_lists(norm_active, norm_chronic)

    return {
        "active_diagnoses": norm_active,
        "chronic_conditions": norm_chronic,
        "_parse_tier": tier.value,
    }

def _normalize_entries(entries: list, default_acuity: str) -> list[dict]:
    """Normalize a list of diagnosis entries."""
    from synthetic_hospital.icd10_validator import is_valid_icd10, nearest_valid_icd10

    result = []
    for dx in entries:
        if not isinstance(dx, dict):
            continue
        icd10 = normalize_icd10(dx.get("icd10", ""))
        name = dx.get("name", dx.get("display_name", ""))
        raw_acuity = dx.get("acuity", default_acuity)
        norm_acuity = ACUITY_NORMALIZATION.get(
            str(raw_acuity).lower().strip(), None
        )
        entry = {
            "icd10": icd10,
            "name": name,
            "acuity": norm_acuity or default_acuity,
        }
        # Validate and correct invalid ICD-10 codes
        if icd10 and not is_valid_icd10(icd10):
            corrected = nearest_valid_icd10(icd10)
            if corrected:
                entry["icd10_original"] = icd10
                entry["icd10"] = corrected
                entry["icd10_corrected"] = True
        result.append(entry)
    return result

def _dedup_across_lists(
    active: list[dict], chronic: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Deduplicate within and across active/chronic lists at 3-char ICD-10 level.

    If same category appears in both, keep only the active entry.
    """
    ACUITY_RANK = {"acute_on_chronic": 3, "acute": 2, "chronic": 1}

    def _dedup_single(entries: list[dict]) -> list[dict]:
        seen: dict[str, dict] = {}
        for e in entries:
            cat = normalize_icd10(e["icd10"])[:3]
            if not cat:
                continue
            # S/T injury: 5-char dedup
            if cat[0] in ("S", "T"):
                cat = normalize_icd10(e["icd10"])[:5]
            existing = seen.get(cat)
            if existing is None:
                seen[cat] = e
            else:
                # Keep higher acuity, more specific code
                if ACUITY_RANK.get(e["acuity"], 0) > ACUITY_RANK.get(existing["acuity"], 0):
                    seen[cat] = e
                elif (ACUITY_RANK.get(e["acuity"], 0) == ACUITY_RANK.get(existing["acuity"], 0)
                      and len(normalize_icd10(e["icd10"])) > len(normalize_icd10(existing["icd10"]))):
                    seen[cat] = e
        return list(seen.values())

    active = _dedup_single(active)
    chronic = _dedup_single(chronic)

    # Cross-list dedup: if same category in both, keep active only
    active_cats = set()
    for e in active:
        cat = normalize_icd10(e["icd10"])[:3]
        if cat and cat[0] in ("S", "T"):
            cat = normalize_icd10(e["icd10"])[:5]
        active_cats.add(cat)

    chronic = [e for e in chronic
               if (normalize_icd10(e["icd10"])[:5] if normalize_icd10(e["icd10"])[:1] in ("S", "T")
                   else normalize_icd10(e["icd10"])[:3]) not in active_cats]

    return active, chronic
