"""
Tests for scripts/merge_root_hadd.py

Covers:
- _resolve_group_key grouping logic
- _collect_tasks directory scanning
- _hadd_merge chunked tree-merge for >500 files
- _ByteBudget acquire/release
- _find_recoil_branch name resolution
- dry-run output
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Allow importing the script directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from merge_root_hadd import (
    MergeTask,
    _ByteBudget,
    _HADD_CHUNK_SIZE,
    _collect_tasks,
    _find_recoil_branch,
    _hadd_merge,
    _hadd_one_shot,
    _resolve_group_key,
)


# ---------------------------------------------------------------------------
# _resolve_group_key
# ---------------------------------------------------------------------------

class TestResolveGroupKey:

    def _path(self, name):
        return Path(f"/fake/{name}")

    def test_marker_found(self):
        p = self._path("WtoLNu-2Jets_PTLNu-100to250_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_NANOAODSIM-abc123.root")
        key = _resolve_group_key(p, "NANOAODSIM")
        assert key.endswith("NANOAODSIM")
        assert "abc123" not in key

    def test_marker_not_found_uuid_stripped(self):
        p = self._path("sample-1-12345678-1234-1234-1234-123456789abc.root")
        key = _resolve_group_key(p, "NANOAODSIM")
        assert key == "sample"

    def test_marker_not_found_no_uuid_returns_all_files(self):
        p = self._path("plainfile.root")
        key = _resolve_group_key(p, "NANOAODSIM")
        assert key == "ALL_FILES"

    def test_multiple_files_same_marker_same_key(self):
        files = [
            self._path(f"WtoLNu_NANOAODSIM-chunk{i}.root") for i in range(4)
        ]
        keys = {_resolve_group_key(f, "NANOAODSIM") for f in files}
        assert len(keys) == 1

    def test_different_samples_different_keys(self):
        f1 = self._path("WtoLNu_NANOAODSIM-abc.root")
        f2 = self._path("TTto4Q_NANOAODSIM-abc.root")
        assert _resolve_group_key(f1, "NANOAODSIM") != _resolve_group_key(f2, "NANOAODSIM")


# ---------------------------------------------------------------------------
# _collect_tasks
# ---------------------------------------------------------------------------

class TestCollectTasks:

    def test_groups_files_by_marker(self, tmp_path):
        in_dir = tmp_path / "WtoLNu"
        in_dir.mkdir()
        for i in range(4):
            (in_dir / f"WtoLNu_NANOAODSIM-chunk{i}.root").touch()
        (in_dir / "TTbar_NANOAODSIM-chunk0.root").touch()

        out_dir = tmp_path / "out"
        tasks = _collect_tasks([in_dir], out_dir, "NANOAODSIM")

        labels = {t.group_key for t in tasks}
        assert any("WtoLNu" in k for k in labels)
        assert any("TTbar" in k for k in labels)
        wlnu_task = next(t for t in tasks if "WtoLNu" in t.group_key)
        assert len(wlnu_task.files) == 4

    def test_empty_dir_skipped(self, tmp_path):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        tasks = _collect_tasks([in_dir], tmp_path / "out", "NANOAODSIM")
        assert tasks == []

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _collect_tasks([tmp_path / "no_such_dir"], tmp_path / "out", "NANOAODSIM")

    def test_output_path_is_flat_in_output_dir(self, tmp_path):
        in_dir = tmp_path / "MySample"
        in_dir.mkdir()
        (in_dir / "file_NANOAODSIM-x.root").touch()
        out_dir = tmp_path / "out"
        tasks = _collect_tasks([in_dir], out_dir, "NANOAODSIM")
        assert tasks[0].output_root.parent == out_dir

    def test_input_bytes_summed(self, tmp_path):
        in_dir = tmp_path / "s"
        in_dir.mkdir()
        for i in range(3):
            p = in_dir / f"f_NANOAODSIM-{i}.root"
            p.write_bytes(b"x" * 100)
        tasks = _collect_tasks([in_dir], tmp_path / "out", "NANOAODSIM")
        assert tasks[0].input_bytes == 300


# ---------------------------------------------------------------------------
# _hadd_merge chunked tree-merge
# ---------------------------------------------------------------------------

class TestHaddMergeChunking:

    def _make_task(self, tmp_path, n_files):
        files = []
        for i in range(n_files):
            p = tmp_path / f"f_{i:04d}.root"
            p.touch()
            files.append(p)
        return MergeTask(
            folder=tmp_path,
            group_key="test",
            files=files,
            input_bytes=n_files * 100,
            output_root=tmp_path / "merged.root",
            output_meta_json=tmp_path / "merged.json",
        )

    def test_small_batch_single_hadd_call(self, tmp_path):
        task = self._make_task(tmp_path, 10)
        with patch("merge_root_hadd._hadd_one_shot") as mock_hadd:
            _hadd_merge(task, chunk_size=500)
        mock_hadd.assert_called_once()
        out, inputs = mock_hadd.call_args[0]
        assert out == task.output_root
        assert len(inputs) == 10

    def test_exactly_chunk_size_single_call(self, tmp_path):
        task = self._make_task(tmp_path, _HADD_CHUNK_SIZE)
        with patch("merge_root_hadd._hadd_one_shot") as mock_hadd:
            _hadd_merge(task, chunk_size=_HADD_CHUNK_SIZE)
        mock_hadd.assert_called_once()

    def test_over_chunk_size_uses_intermediates(self, tmp_path):
        n = 1200
        chunk = 500
        task = self._make_task(tmp_path, n)
        calls_log = []

        def fake_hadd(output, inputs):
            calls_log.append((output, list(inputs)))
            output.touch()  # simulate hadd creating the file

        with patch("merge_root_hadd._hadd_one_shot", side_effect=fake_hadd):
            _hadd_merge(task, chunk_size=chunk)

        # 1200 files → 3 chunks (500+500+200) → 3 intermediates → 1 final = 4 calls
        n_chunks = (n + chunk - 1) // chunk
        assert len(calls_log) == n_chunks + 1
        # Last call merges into the real output
        assert calls_log[-1][0] == task.output_root
        # Each chunk call gets ≤ chunk_size inputs
        for out, inputs in calls_log[:-1]:
            assert len(inputs) <= chunk

    def test_1000_files_chunks_correctly(self, tmp_path):
        n = 1000
        chunk = 500
        task = self._make_task(tmp_path, n)
        calls_log = []

        def fake_hadd(output, inputs):
            calls_log.append(len(inputs))
            output.touch()

        with patch("merge_root_hadd._hadd_one_shot", side_effect=fake_hadd):
            _hadd_merge(task, chunk_size=chunk)

        # 1000 → 2 chunks of 500 → final: 3 calls total
        assert len(calls_log) == 3
        assert calls_log[0] == 500
        assert calls_log[1] == 500
        assert calls_log[2] == 2  # 2 intermediates merged into final

    def test_250000_files_two_level_merge(self, tmp_path):
        """Second level needed when n > chunk_size^2 is not tested (too slow),
        but verify the logic path exists by faking chunk_size=3 with 10 files."""
        n = 10
        chunk = 3
        task = self._make_task(tmp_path, n)
        calls_log = []

        def fake_hadd(output, inputs):
            calls_log.append((str(output.name), len(inputs)))
            output.touch()

        with patch("merge_root_hadd._hadd_one_shot", side_effect=fake_hadd):
            _hadd_merge(task, chunk_size=chunk)

        # 10 files, chunk=3 → 4 chunks (3+3+3+1=10) → 4 intermediates
        # 4 intermediates ≤ chunk=3? No (4>3) → second level: 2 chunks of (3+1)
        # → 2 level2 files → final = 4+2+1 = 7 calls
        # Verify all chunk inputs are ≤ chunk
        for _, n_inputs in calls_log:
            assert n_inputs <= chunk

    def test_tmp_chunks_dir_cleaned_up(self, tmp_path):
        task = self._make_task(tmp_path, 600)
        chunk_dir = tmp_path / "_chunks_test"

        def fake_hadd(output, inputs):
            output.touch()

        with patch("merge_root_hadd._hadd_one_shot", side_effect=fake_hadd):
            _hadd_merge(task, chunk_size=500)

        assert not chunk_dir.exists()

    def test_dry_run_prints_chunked_info(self, tmp_path, capsys):
        task = self._make_task(tmp_path, 1200)
        _hadd_merge(task, dry_run=True, chunk_size=500)
        out = capsys.readouterr().out
        assert "chunked" in out
        assert "1200" in out

    def test_dry_run_small_prints_hadd_cmd(self, tmp_path, capsys):
        task = self._make_task(tmp_path, 5)
        _hadd_merge(task, dry_run=True, chunk_size=500)
        out = capsys.readouterr().out
        assert "hadd" in out
        assert "DRY-RUN" in out

    def test_hadd_failure_raises(self, tmp_path):
        task = self._make_task(tmp_path, 3)
        with patch("merge_root_hadd._hadd_one_shot", side_effect=RuntimeError("hadd failed")):
            with pytest.raises(RuntimeError, match="hadd failed"):
                _hadd_merge(task, chunk_size=500)


# ---------------------------------------------------------------------------
# _ByteBudget
# ---------------------------------------------------------------------------

class TestByteBudget:

    def test_acquire_release_cycle(self):
        budget = _ByteBudget(1000)
        used = budget.acquire(400)
        assert used == 400
        budget.release(used)
        used2 = budget.acquire(900)
        assert used2 == 900

    def test_acquire_capped_at_total(self):
        budget = _ByteBudget(500)
        used = budget.acquire(9999)
        assert used == 500

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            _ByteBudget(0)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError):
            _ByteBudget(-1)

    def test_release_does_not_exceed_total(self):
        budget = _ByteBudget(100)
        budget.acquire(100)
        budget.release(200)   # over-release
        used = budget.acquire(100)
        assert used == 100


# ---------------------------------------------------------------------------
# _find_recoil_branch
# ---------------------------------------------------------------------------

class TestFindRecoilBranch:

    def test_exact_match(self):
        assert _find_recoil_branch(["met_pt", "recoil", "n_jets"], "recoil") == "recoil"

    def test_case_insensitive_match(self):
        assert _find_recoil_branch(["MET_PT", "Recoil", "n_jets"], "recoil") == "Recoil"

    def test_alias_fallback_pfmet(self):
        assert _find_recoil_branch(["PFMET_pt", "n_jets"], "recoil") == "PFMET_pt"

    def test_alias_fallback_met_pt(self):
        result = _find_recoil_branch(["met_pt", "n_jets"], "recoil")
        assert result == "met_pt"

    def test_substring_fallback(self):
        result = _find_recoil_branch(["some_recoil_var", "n_jets"], "recoil")
        assert result == "some_recoil_var"

    def test_no_match_returns_none(self):
        assert _find_recoil_branch(["n_jets", "n_bjets"], "recoil") is None

    def test_preferred_takes_priority(self):
        result = _find_recoil_branch(["recoil", "PFMET_pt", "met_pt"], "PFMET_pt")
        assert result == "PFMET_pt"
