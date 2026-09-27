def normalize_icd10(code: str) -> str:
    """Normalize ICD-10 code: strip dots, uppercase."""
    if not code:
        return ""
    return code.replace(".", "").replace(" ", "").upper()

def _strip_markdown(text: str) -> str:
    """Remove markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)
    return text.strip()
