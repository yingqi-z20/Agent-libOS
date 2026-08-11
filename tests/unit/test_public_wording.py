from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestPublicWording:
    def test_current_sources_and_docs_do_not_describe_product_as_mvp(self) -> None:
        pattern = re.compile(r"\bMVP\b|current prototype", re.IGNORECASE)
        scanned_roots = [
            ROOT / "agent_libos",
            ROOT / "docs",
            ROOT / "README.md",
        ]
        ignored = {
            ROOT / "agent_libos_design_doc.md",
            ROOT / "plan.md",
        }
        offenders: list[str] = []

        for root in scanned_roots:
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in paths:
                if path in ignored or not path.is_file():
                    continue
                if path.suffix not in {".md", ".py"}:
                    continue
                text = path.read_text(encoding="utf-8")
                if pattern.search(text):
                    offenders.append(path.relative_to(ROOT).as_posix())

        assert offenders == []

    def test_historical_design_archives_are_marked_not_current_behavior(self) -> None:
        design_doc = (ROOT / "agent_libos_design_doc.md").read_text(encoding="utf-8").lower()
        roadmap = (ROOT / "plan.md").read_text(encoding="utf-8").lower()

        assert "historical design archive" in design_doc
        assert "not the source of truth" in design_doc
        assert "current behavior" in design_doc
        assert "not the implementation reference" in roadmap

    def test_provider_response_retention_is_described_as_a_bounded_projection(
        self,
    ) -> None:
        expected_phrases = {
            "README.md": "provider-response evidence is a bounded projection",
            "docs/configuration.md": "bounded\n  provider-response projection",
            "docs/evidence_payload_retention.md": (
                "bounded provider-response\nprojection stored under `raw_response`"
            ),
            "docs/threat_model.md": "bounded provider-response\nprojection",
        }

        for relative_path, phrase in expected_phrases.items():
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            assert phrase in text, relative_path

    def test_long_horizon_docs_assign_retries_to_agent_libos(self) -> None:
        text = (
            ROOT / "benchmarks" / "long_horizon_agent" / "README.md"
        ).read_text(encoding="utf-8")

        assert "Agent libOS's explicit, attempt-traced\ntransport retry loop" in text
        assert "provider-SDK internal retries are disabled" in text
        assert "the SDK already applies" not in text
