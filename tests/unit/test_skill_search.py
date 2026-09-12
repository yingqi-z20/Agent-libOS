from agent_libos.utils.skill_search import (
    skill_metadata_exact_match,
    skill_metadata_search_score,
)


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


def _tool_score(query: str, tool_names: list[str], description: str = "Do work.") -> int | None:
    return skill_metadata_search_score(
        skill_id="example-skill",
        name="example-skill",
        description=description,
        text=query,
        tool_names=tool_names,
    )


def test_exact_tool_name_query_finds_the_declaring_skill() -> None:
    assert _tool_score("human_output", ["ask_human", "human_output"]) is not None
    assert _tool_score("human_output", ["run_shell_command"]) is None
    # Goal fragments often quote or punctuate the tool name.
    assert _tool_score("call `human_output`,", ["human_output"]) is not None


def test_tool_name_terms_count_toward_multi_term_matches() -> None:
    # "create checkpoint" previously matched only ids/names/descriptions;
    # tool-name terms now contribute so the owning Skill is found.
    assert _tool_score("create checkpoint", ["create_checkpoint", "list_checkpoints"]) is not None
    assert _tool_score("create checkpoint", ["git_status"]) is None


def test_exact_tool_hits_outrank_description_mentions() -> None:
    owner = _tool_score(
        "run_shell_command",
        ["run_shell_command"],
        description="Run one approved command.",
    )
    mention = _tool_score(
        "run_shell_command",
        ["write_text_file"],
        description="Edit files; not for shell commands.",
    )
    assert owner is not None and mention is not None
    assert owner > mention


def test_multi_tool_query_scores_every_owning_skill() -> None:
    query = "run_shell_command git_status create_checkpoint human_output"
    for tools in (["run_shell_command"], ["git_status", "git_diff"], ["create_checkpoint"], ["human_output"]):
        assert _tool_score(query, tools) is not None
    assert _tool_score(query, ["propose_jit_tool"]) is None


def test_exact_match_accepts_single_declared_tool_name() -> None:
    assert skill_metadata_exact_match(
        skill_id="example-skill",
        name="example-skill",
        text="Human_Output",
        tool_names=["human_output"],
    )
    assert not skill_metadata_exact_match(
        skill_id="example-skill",
        name="example-skill",
        text="human_output ask_human",
        tool_names=["human_output", "ask_human"],
    )
    assert not skill_metadata_exact_match(
        skill_id="example-skill",
        name="example-skill",
        text="human_output",
        tool_names=["ask_human"],
    )
