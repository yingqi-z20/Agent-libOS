from __future__ import annotations

import re


_TERM_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
SKILL_SEARCH_TEXT_MAX_CHARS = 1_024
SKILL_SEARCH_MAX_TERMS = 16
_LOW_INFORMATION_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "agent",
        "appropriate",
        "can",
        "could",
        "do",
        "for",
        "from",
        "load",
        "matching",
        "need",
        "needed",
        "of",
        "on",
        "or",
        "please",
        "skill",
        "smallest",
        "task",
        "that",
        "the",
        "this",
        "to",
        "tool",
        "use",
        "want",
        "with",
        "would",
    }
)


def skill_search_terms(text: str | None) -> tuple[str, ...]:
    """Return stable, deduplicated metadata-search terms for a natural query."""

    selected_text = str(text or "")[:SKILL_SEARCH_TEXT_MAX_CHARS]
    raw_terms = tuple(
        dict.fromkeys(
            match.group(0).casefold()
            for match in _TERM_PATTERN.finditer(selected_text)
        )
    )[:SKILL_SEARCH_MAX_TERMS]
    informative = tuple(
        term
        for term in raw_terms
        if len(term) > 1 and term not in _LOW_INFORMATION_TERMS
    )
    return informative or raw_terms


def skill_metadata_search_score(
    *,
    skill_id: str,
    name: str,
    description: str,
    text: str | None,
) -> int | None:
    """Score visible Skill metadata using source-independent query semantics.

    Multi-term intent queries may span more than one narrowly owned Skill (for
    example, workspace reading and workspace editing).  Keep those useful
    partial matches while requiring at least two matching terms, so one generic
    word in a longer query does not turn into an unrelated catalog result.
    """

    if not str(text or "").strip():
        return 0
    terms = skill_search_terms(text)
    if not terms:
        return None

    fields = (
        (str(skill_id).casefold(), 12),
        (str(name).casefold(), 10),
        (str(description).casefold(), 4),
    )
    matched_terms = 0
    weighted_hits = 0
    for term in terms:
        term_hits = sum(weight for value, weight in fields if term in value)
        if term_hits:
            matched_terms += 1
            weighted_hits += term_hits
    minimum_matches = 1 if len(terms) == 1 else 2
    if matched_terms < minimum_matches:
        return None

    normalized_query = " ".join(terms)
    normalized_id = " ".join(skill_search_terms(skill_id))
    normalized_name = " ".join(skill_search_terms(name))
    exact_bonus = 10_000 if normalized_query in {normalized_id, normalized_name} else 0
    phrase_bonus = 1_000 if any(normalized_query in value for value, _ in fields) else 0
    return exact_bonus + phrase_bonus + (matched_terms * 100) + weighted_hits


def skill_metadata_exact_match(
    *,
    skill_id: str,
    name: str,
    text: str | None,
) -> bool:
    query = str(text or "").strip().casefold()
    return bool(query) and query in {
        str(skill_id).strip().casefold(),
        str(name).strip().casefold(),
    }
