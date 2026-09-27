import json
import logging
import re
from enum import Enum
from .diagnosis import normalize_icd10, _strip_markdown
log = logging.getLogger(__name__)

def parse_output(raw_text: str) -> dict:
    """Parse model output into structured summarization prediction."""
    text = _strip_markdown(raw_text)

    try:
        data = json.loads(text, strict=False)
        summary = data.get("summary", "")
        if summary:
            return {"summary": summary}
    except json.JSONDecodeError:
        # Try to extract JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(), strict=False)
                summary = data.get("summary", "")
                if summary:
                    return {"summary": summary}
            except json.JSONDecodeError:
                pass

    # Fallback: treat entire text as the summary (skip any reasoning prefix)
    # Look for "summary" keyword and take text after it
    cleaned = raw_text.strip()
    if cleaned:
        return {"summary": cleaned}

    return {"summary": "", "parse_error": "Empty output"}
