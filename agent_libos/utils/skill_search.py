from __future__ import annotations

import re
from collections.abc import Iterable


_TERM_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_RAW_TOKEN_PATTERN = re.compile(r"[^\s,;`'\"()\[\]{}<>]+", re.UNICODE)
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


def skill_search_tool_names(text: str | None) -> tuple[str, ...]:
    """Return whitespace-delimited query tokens that may name an exact tool.

    Tool names such as ``run_shell_command`` keep their underscores here,
    unlike ``skill_search_terms`` which splits them into natural-language
    terms.  Surrounding quotes, backticks, and punctuation are stripped so a
    goal fragment like "call `git_status`," still names the tool.
    """

    selected_text = str(text or "")[:SKILL_SEARCH_TEXT_MAX_CHARS]
    return tuple(
        dict.fromkeys(
            token.casefold().strip(".:")
            for match in _RAW_TOKEN_PATTERN.finditer(selected_text)
            if (token := match.group(0))
        )
    )


def _declared_tool_names(tool_names: Iterable[str] | None) -> tuple[str, ...]:
    if not tool_names:
        return ()
    return tuple(
        dict.fromkeys(
            str(name).strip().casefold()
            for name in tool_names
            if isinstance(name, str) and str(name).strip()
        )
    )


def skill_metadata_search_score(
    *,
    skill_id: str,
    name: str,
    description: str,
    text: str | None,
    tool_names: Iterable[str] | None = None,
) -> int | None:
    """Score visible Skill metadata using source-independent query semantics.

    Multi-term intent queries may span more than one narrowly owned Skill (for
    example, workspace reading and workspace editing).  Keep those useful
    partial matches while requiring at least two matching terms, so one generic
    word in a longer query does not turn into an unrelated catalog result.

    Declared tool names are first-class search keys: a query that names an
    exact tool (``human_output``) always finds the Skill that declares it, and
    tool-name terms (``shell``, ``checkpoint``) also count toward term matches.
    A goal that lists the exact tools it requires can therefore be resolved to
    its owning Skills in one discovery call instead of a guessing loop.
    """

    if not str(text or "").strip():
        return 0
    terms = skill_search_terms(text)
    if not terms:
        return None

    declared_tools = _declared_tool_names(tool_names)
    tool_term_text = " ".join(
        " ".join(skill_search_terms(tool_name)) for tool_name in declared_tools
    )
    fields = (
        (str(skill_id).casefold(), 12),
        (str(name).casefold(), 10),
        (str(description).casefold(), 4),
        (tool_term_text, 6),
    )
    matched_terms = 0
    weighted_hits = 0
    for term in terms:
        term_hits = sum(weight for value, weight in fields if value and term in value)
        if term_hits:
            matched_terms += 1
            weighted_hits += term_hits
    exact_tool_hits = sum(
        1
        for token in skill_search_tool_names(text)
        if token in declared_tools
    )
    minimum_matches = 1 if len(terms) == 1 else 2
    if matched_terms < minimum_matches and not exact_tool_hits:
        return None

    normalized_query = " ".join(terms)
    normalized_id = " ".join(skill_search_terms(skill_id))
    normalized_name = " ".join(skill_search_terms(name))
    exact_bonus = 10_000 if normalized_query in {normalized_id, normalized_name} else 0
    phrase_bonus = 1_000 if any(
        value and normalized_query in value for value, _ in fields[:3]
    ) else 0
    tool_bonus = 2_000 * exact_tool_hits
    return exact_bonus + phrase_bonus + tool_bonus + (matched_terms * 100) + weighted_hits


def skill_metadata_exact_match(
    *,
    skill_id: str,
    name: str,
    text: str | None,
    tool_names: Iterable[str] | None = None,
) -> bool:
    """Return whether ``text`` names exactly this Skill or one tool it declares.

    A single-token query equal to a declared tool name is an exact match so
    discovery collapses to the owning Skill instead of listing partial matches
    whose descriptions merely mention similar words.
    """

    query = str(text or "").strip().casefold()
    if not query:
        return False
    if query in {
        str(skill_id).strip().casefold(),
        str(name).strip().casefold(),
    }:
        return True
    tokens = skill_search_tool_names(query)
    if len(tokens) != 1:
        return False
    return tokens[0] in _declared_tool_names(tool_names)
