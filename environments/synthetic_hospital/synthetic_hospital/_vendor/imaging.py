import json
import logging
import re
from enum import Enum
from .diagnosis import normalize_icd10, _strip_markdown
log = logging.getLogger(__name__)

def parse_output(raw_text: str) -> dict:
    """Parse model output into structured imaging indication prediction."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                return {
                    "clinical_question": raw_text.strip()[:200],
                    "pre_read_summary": "",
                    "must_include_findings": [],
                    "differential": [],
                    "parse_error": "Invalid JSON",
                }
        else:
            return {
                "clinical_question": raw_text.strip()[:200],
                "pre_read_summary": "",
                "must_include_findings": [],
                "differential": [],
                "parse_error": "No JSON found",
            }

    return {
        "clinical_question": data.get("clinical_question", ""),
        "pre_read_summary": data.get("pre_read_summary", ""),
        "must_include_findings": data.get("must_include_findings", []),
        "differential": data.get("differential", []),
    }
