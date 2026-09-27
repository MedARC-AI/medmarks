"""Offline equivalent of the release ICD-10 validator, using its frozen index."""

from .data import asset


def is_valid_icd10(code):
    return bool(code) and code.strip().upper().replace(".", "") in asset("icd10.json.gz")


def nearest_valid_icd10(code):
    codes = asset("icd10.json.gz")
    norm = code.replace(".", "").replace(" ", "").upper()
    if len(norm) < 3:
        return None
    for length in range(len(norm) - 1, 3, -1):
        prefix = norm[:length]
        if prefix in codes and not codes[prefix]:  # False = billable
            return prefix
    return norm[:3] if norm[:3] in codes else None
