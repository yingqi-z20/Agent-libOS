from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_libos.llm.client import LLMCompletion
from agent_libos.models import DataLabels
from agent_libos.models.semantic import (
    SemanticAssessment,
    SemanticAssessmentKind,
    SemanticAssessmentStatus,
)
from agent_libos.semantic.external import (
    ExternalLLMSemanticAssessor,
    SemanticProviderResponseError,
    SemanticUsageTelemetry,
    _extract_usage_telemetry,
)


pytestmark = pytest.mark.providers

_MAX_INTEGER = (1 << 53) - 1
_SECRET = "SEMANTIC_USAGE_SECRET_SENTINEL"


def _completion(usage: Any) -> LLMCompletion:
    completion = LLMCompletion(content="{}", tool_calls=[])
    completion.usage = usage
    return completion


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "cost_microunits": 19,
            },
            SemanticUsageTelemetry(11, 7, 19),
        ),
        (
            {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "cost_microunits": 23,
            },
            SemanticUsageTelemetry(13, 5, 23),
        ),
        (
            {
                "input_tokens": 13,
                "prompt_tokens": 13,
                "output_tokens": 5,
                "completion_tokens": 5,
            },
            SemanticUsageTelemetry(13, 5, None),
        ),
        ({"input_tokens": 0, "output_tokens": 0}, SemanticUsageTelemetry(0, 0)),
        ({"input_tokens": _MAX_INTEGER}, SemanticUsageTelemetry(_MAX_INTEGER)),
    ],
)
def test_external_usage_accepts_only_bounded_exact_integer_counters(
    usage: dict[str, Any],
    expected: SemanticUsageTelemetry,
) -> None:
    assert _extract_usage_telemetry(_completion(usage)) == expected


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True},
        {"output_tokens": 1.5},
        {"cost_microunits": -1},
        {"prompt_tokens": _MAX_INTEGER + 1},
        {"input_tokens": 3, "prompt_tokens": 4},
        {"output_tokens": 2, "completion_tokens": False},
        {"private_provider_field": _SECRET},
        {"input_tokens": _SECRET},
    ],
    ids=(
        "boolean",
        "float",
        "negative",
        "oversize",
        "canonical-alias-conflict",
        "valid-invalid-alias-conflict",
        "unknown-secret-key",
        "secret-counter-value",
    ),
)
def test_external_usage_ignores_untrusted_or_unknown_values(
    usage: dict[str, Any],
) -> None:
    telemetry = _extract_usage_telemetry(_completion(usage))

    assert telemetry is None
    assert _SECRET not in repr(telemetry)


def test_external_usage_ignores_unknown_secret_but_keeps_valid_known_counters() -> None:
    telemetry = _extract_usage_telemetry(
        _completion(
            {
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "private_provider_field": _SECRET,
            }
        )
    )

    assert telemetry == SemanticUsageTelemetry(17, 9, None)
    assert _SECRET not in repr(telemetry)


def test_external_usage_rejects_non_exact_completion_or_usage_container() -> None:
    class CompletionSubclass(LLMCompletion):
        pass

    class UsageSubclass(dict[str, Any]):
        pass

    assert (
        _extract_usage_telemetry(
            CompletionSubclass(content="{}", tool_calls=[], usage={"input_tokens": 1})
        )
        is None
    )
    assert (
        _extract_usage_telemetry(_completion(UsageSubclass(input_tokens=1))) is None
    )
    assert _extract_usage_telemetry(_completion({object(): 1})) is None
    assert (
        _extract_usage_telemetry(
            _completion({f"unknown_{index}": index for index in range(65)})
        )
        is None
    )


def test_usage_consumption_is_one_shot_and_thread_local() -> None:
    assessor = object.__new__(ExternalLLMSemanticAssessor)
    assessor._usage_local = threading.local()
    barrier = threading.Barrier(2)
    observed: dict[str, tuple[SemanticUsageTelemetry | None, SemanticUsageTelemetry | None]] = {}

    def worker(name: str, value: int) -> None:
        assessor._usage_local.value = SemanticUsageTelemetry(input_tokens=value)
        barrier.wait()
        observed[name] = (
            assessor.take_last_usage_telemetry(),
            assessor.take_last_usage_telemetry(),
        )

    first = threading.Thread(target=worker, args=("first", 3))
    second = threading.Thread(target=worker, args=("second", 29))
    first.start()
    second.start()
    first.join()
    second.join()

    assert observed == {
        "first": (SemanticUsageTelemetry(input_tokens=3), None),
        "second": (SemanticUsageTelemetry(input_tokens=29), None),
    }
    assert assessor.take_last_usage_telemetry() is None


def test_usage_survives_invalid_provider_schema_without_retaining_response() -> None:
    class Client:
        def complete_with_metadata(self, **_kwargs: Any) -> LLMCompletion:
            return LLMCompletion(
                content=f'{{"invalid":"{_SECRET}"}}',
                tool_calls=[],
                usage={
                    "prompt_tokens": 31,
                    "completion_tokens": 4,
                    "private_provider_field": _SECRET,
                },
            )

    assessor = object.__new__(ExternalLLMSemanticAssessor)
    assessor._usage_local = threading.local()
    assessor._max_response_bytes = 32 * 1024

    with pytest.raises(SemanticProviderResponseError) as caught:
        assessor._dispatch_and_parse(
            Client(),
            messages=[],
            max_tokens=1,
            labels=DataLabels(),
            kind=SemanticAssessmentKind.APPROVAL,
            projection={"kind": "approval"},
            response_schema={},
        )

    telemetry = assessor.take_last_usage_telemetry()
    assert telemetry == SemanticUsageTelemetry(31, 4, None)
    assert assessor.take_last_usage_telemetry() is None
    assert _SECRET not in str(caught.value)
    assert _SECRET not in repr(telemetry)


def test_external_assessor_rejects_reentrancy_and_clears_its_thread_guard() -> None:
    class ReentrancyProbe(ExternalLLMSemanticAssessor):
        def __init__(self) -> None:
            self._usage_local = threading.local()
            self.calls = 0

        def _assess_once(self, request: Any) -> SemanticAssessment:
            self.calls += 1
            if self.calls == 1:
                with pytest.raises(RuntimeError, match="must not be reentrant"):
                    self.assess(request)
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    assessor = ReentrancyProbe()

    assert assessor.assess(object()).status is SemanticAssessmentStatus.SUCCESS
    assert assessor.assess(object()).status is SemanticAssessmentStatus.SUCCESS
    assert assessor.calls == 2
