from agent_libos.utils.skill_search import skill_metadata_search_score


def _score(description: str, query: str) -> int | None:
    return skill_metadata_search_score(
        skill_id="example-skill",
        name="example-skill",
        description=description,
        text=query,
    )


def test_multi_term_skill_search_keeps_only_sufficient_partial_matches() -> None:
    query = "filesystem read write text file directory"

    assert _score("Read bounded workspace text files and directories.", query) is not None
    assert _score("Write workspace text files and directories.", query) is not None
    assert _score("Inspect one directory.", query) is None


def test_skill_search_prefers_more_query_term_coverage() -> None:
    complete = _score("Reconcile a quasar ledger for one intent.", "quasar ledger reconcile intent")
    partial = _score("Use a quasar ledger for one intent.", "quasar ledger reconcile intent")

    assert complete is not None
    assert partial is not None
    assert complete > partial
