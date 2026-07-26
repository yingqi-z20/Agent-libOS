from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import agent_outputs
from scripts.agent_outputs import cleanup_agent_outputs, snapshot_agent_outputs


class TestAgentOutputsCleanup:
    def test_cleanup_removes_only_paths_outside_baseline(self, tmp_path) -> None:
        root = tmp_path / "agent_outputs"
        preserved = root / "existing.txt"
        generated = root / "generated" / "file.txt"
        preserved.parent.mkdir()
        preserved.write_text("keep", encoding="utf-8")
        baseline = snapshot_agent_outputs(root)
        generated.parent.mkdir()
        generated.write_text("remove", encoding="utf-8")

        removed = cleanup_agent_outputs(root, baseline=baseline)

        assert "generated/file.txt" in removed
        assert "generated/" in removed
        assert preserved.read_text(encoding="utf-8") == "keep"
        assert not generated.exists()

    def test_cleanup_dry_run_reports_without_deleting(self, tmp_path) -> None:
        root = tmp_path / "agent_outputs"
        generated = root / "generated.txt"
        root.mkdir()
        generated.write_text("remove", encoding="utf-8")

        removed = cleanup_agent_outputs(root, dry_run=True)

        assert "generated.txt" in removed
        assert generated.exists()

    def test_nested_dry_run_matches_live_cleanup_plan(self, tmp_path: Path) -> None:
        preview_root = tmp_path / "preview" / "agent_outputs"
        live_root = tmp_path / "live" / "agent_outputs"
        for root in (preview_root, live_root):
            generated = root / "nested" / "deeper" / "generated.txt"
            generated.parent.mkdir(parents=True)
            generated.write_text("remove", encoding="utf-8")

        preview = cleanup_agent_outputs(preview_root, dry_run=True)
        removed = cleanup_agent_outputs(live_root)

        assert preview == removed
        assert preview == [
            "nested/deeper/generated.txt",
            "nested/deeper/",
            "nested/",
            ".",
        ]
        assert preview_root.exists()
        assert not live_root.exists()

    def test_cleanup_rejects_symlink_root_without_touching_target(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "outside"
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        root = tmp_path / "agent_outputs"
        try:
            root.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("host does not permit directory symlink creation")

        with pytest.raises(ValueError, match="must not be a symlink"):
            snapshot_agent_outputs(root)
        with pytest.raises(ValueError, match="must not be a symlink"):
            cleanup_agent_outputs(root)

        assert sentinel.read_text(encoding="utf-8") == "keep"
        assert root.is_symlink()

    def test_cleanup_aborts_if_root_identity_changes_before_mutation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if os.name != "posix":
            pytest.skip("descriptor-stable replacement probe is POSIX-specific")
        root = tmp_path / "agent_outputs"
        root.mkdir()
        generated = root / "generated.txt"
        generated.write_text("generated", encoding="utf-8")
        displaced = tmp_path / "displaced"
        outside = tmp_path / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        original_collect = agent_outputs._collect_entries

        def replace_root(root_fd: int) -> dict[str, bool]:
            entries = original_collect(root_fd)
            root.rename(displaced)
            root.symlink_to(outside, target_is_directory=True)
            return entries

        monkeypatch.setattr(agent_outputs, "_collect_entries", replace_root)

        with pytest.raises(RuntimeError, match="root changed"):
            cleanup_agent_outputs(root)

        assert (displaced / "generated.txt").read_text(encoding="utf-8") == "generated"
        assert sentinel.read_text(encoding="utf-8") == "keep"
