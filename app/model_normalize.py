from __future__ import annotations
import re

_HYPHEN_FIX = re.compile(r"\s*[-–—]\s*")
_SPACES = re.compile(r"\s+")
_BLOCK = re.compile(r"\b(BLK|BLOCK)\s*([0-9]{1,3})\b", re.IGNORECASE)

def normalize_model(s: str) -> str:
    """Normalize model string for matching. Safe, deterministic.
    Examples:
      'f16c blk50' -> 'F-16C BLOCK 50'
      'J 10 C'     -> 'J-10C'
    """
    if not s:
        return ""
    s = s.strip()
    # unify hyphens
    s = _HYPHEN_FIX.sub("-", s)
    # remove extra spaces
    s = _SPACES.sub(" ", s)
    # standardize BLOCK notation
    s = _BLOCK.sub(lambda m: f"BLOCK {m.group(2)}", s)
    # uppercase
    s = s.upper()

    # Heuristic: glue letter+digit sequences like "J 10" -> "J-10"
    # but avoid harming cases with already hyphenated.
    s = re.sub(r"\b([A-Z])\s+([0-9]{1,3})\b", r"\1-\2", s)

    # Remove spaces around hyphen (already normalized)
    s = s.replace(" -", "-").replace("- ", "-")
    return s

def normalize_alias_list(aliases: list[str]) -> list[str]:
    out = []
    for a in aliases or []:
        na = normalize_model(a)
        if na and na not in out:
            out.append(na)
    return out
