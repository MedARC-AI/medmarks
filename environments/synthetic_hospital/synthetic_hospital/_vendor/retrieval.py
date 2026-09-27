import json
import logging
import re
from enum import Enum
from .diagnosis import normalize_icd10, _strip_markdown
log = logging.getLogger(__name__)

def parse_output(raw_text: str) -> dict:
    """Parse LLM reranking output into ranked results."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                return {"rankings": [], "parse_error": "Invalid JSON"}
        else:
            return {"rankings": [], "parse_error": "No JSON found"}

    rankings = data.get("rankings", [])
    normalized = []
    for item in rankings:
        normalized.append({
            "passage_id": item.get("passage_id", ""),
            "grade": int(item.get("grade", 0)),
        })
    return {"rankings": normalized}
