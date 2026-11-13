"""Optional LLM narration. OFF by default (no provider is shipped). A narrator is any ``callable(summary:str)->str``.

Its text is never part of the evidence: it is attached as ``llm_narrative`` only if it passes the guards
below, otherwise it is dropped and the deterministic statements stand alone. An LLM failure never affects
model output.
"""
from __future__ import annotations

import re

BANNED = re.compile(r"tumou?r|cancer|malignan|benign|glioma|diagnos|lesion is|patient has|metasta|carcinoma|"
                    r"probability of|% chance|you have|disease", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")


def safe_narrative(text: str, grounded_text: str, max_chars: int = 1200) -> tuple[str | None, str]:
    """-> (text or None, reason). Rejects banned clinical wording, ungrounded numbers, over-long text."""
    if not text or not text.strip():
        return None, "empty"
    if len(text) > max_chars:
        return None, "too long"
    m = BANNED.search(text)
    if m:
        return None, f"banned wording: {m.group(0)!r}"
    allowed = set(NUM.findall(grounded_text))
    stray = [n for n in NUM.findall(text) if n not in allowed]
    if stray:
        return None, f"numbers not present in the measured evidence: {stray[:3]}"
    return text.strip(), "ok"
