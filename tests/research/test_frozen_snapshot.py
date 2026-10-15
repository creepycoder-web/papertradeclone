"""
A research study must replay from an IMMUTABLE, HASH-PINNED input snapshot.

The defect: ``v3search.load_universe()`` built its "frozen" snapshot at call
time - ``frozen_at`` was the current clock, ``files`` was empty (no content
hashes), and the universe came from the live ``var/`` directory the bot keeps
appending to. So the universe could change mid-study and nothing could detect a
file being edited underneath it.

These tests prove snapshot creation, hash registration, replay-from-snapshot,
live-mutation immunity, snapshot-mutation integrity failure, and a non-empty
content-addressed manifest.
"""


from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integrity,
    pytest.mark.research,
]


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "src"))

import v3snapshot as snap_mod  # noqa: E402
from v3snapshot import (  # noqa: E402
    MANIFEST_NAME,
    SnapshotIntegrityError,
    create,
    load,
    open_or_create,
)


def _source(tmp_path: Path) -> dict[str, Path]:
    a = tmp_path / "in" / "a.jsonl"
    b = tmp_path / "in" / "b.txt"
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_text('{"x":1}\n')
    b.write_text("hello\n")
    return {"a.jsonl": a, "b.txt": b}


# ---------------------------------------------------------------------------
# creation + hash registration
# ---------------------------------------------------------------------------


def test_create_writes_a_non_empty_content_addressed_manifest(tmp_path):
    src = _source(tmp_path)
    snap = create(tmp_path / "snap", src, study="unit")
    manifest_path = tmp_path / "snap" / MANIFEST_NAME
    assert manifest_path.exists()
    m = json.loads(manifest_path.read_text())
    assert m["files"], "the manifest must register at least one file"
    assert m["study"] == "unit"
    assert m["content_sha256"]
    assert m["snapshot_dir"] == str(tmp_path / "snap")
    # content-addressed: the declared hash is reproducible from the file hashes
    assert m["content_sha256"] == snap_mod._content_hash(m["files"])


def test_every_input_file_is_sha256_registered(tmp_path):
    src = _source(tmp_path)
    snap = create(tmp_path / "snap", src, study="unit")
    for name in src:
        recorded = snap.files[name]["sha256"]
        actual = hashlib.sha256((tmp_path / "snap" / name).read_bytes()).hexdigest()
        assert recorded == actual, name
        assert snap.files[name]["bytes"] == (tmp_path / "snap" / name).stat().st_size


def test_snapshot_created_at_a_stated_time(tmp_path):
    snap = create(tmp_path / "snap", _source(tmp_path), study="unit")
    assert snap.frozen_at > 0
    assert snap.frozen_at_utc


# ---------------------------------------------------------------------------
# replay uses the snapshot; live mutation cannot change it
# ---------------------------------------------------------------------------


def test_replay_reads_the_snapshot_root(tmp_path):
    snap = create(tmp_path / "snap", _source(tmp_path), study="unit")
    paths_snap = snap.as_paths_snapshot()
    assert paths_snap.root == tmp_path / "snap"
    assert paths_snap.path("a.jsonl").read_text() == '{"x":1}\n'


def test_live_source_mutation_does_not_change_the_replay(tmp_path):
    src = _source(tmp_path)
    snap = create(tmp_path / "snap", src, study="unit")
    before = snap.path("a.jsonl").read_text()
    # the live source keeps appending, as the bot does
    with open(src["a.jsonl"], "a") as fh:
        fh.write('{"x":2}\n')
    # reuse does NOT refresh
    again = open_or_create(tmp_path / "snap", src, study="unit")
    assert again.path("a.jsonl").read_text() == before
    assert again.content_sha256 == snap.content_sha256


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


def test_mutating_a_snapshot_file_causes_integrity_failure(tmp_path):
    create(tmp_path / "snap", _source(tmp_path), study="unit")
    with open(tmp_path / "snap" / "a.jsonl", "a") as fh:
        fh.write("tampered\n")
    with pytest.raises(SnapshotIntegrityError):
        load(tmp_path / "snap", verify=True)


def test_deleting_a_registered_snapshot_file_causes_integrity_failure(tmp_path):
    create(tmp_path / "snap", _source(tmp_path), study="unit")
    (tmp_path / "snap" / "b.txt").unlink()
    with pytest.raises(SnapshotIntegrityError):
        load(tmp_path / "snap", verify=True)


def test_a_manifest_with_no_files_is_not_a_real_snapshot(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    (d / MANIFEST_NAME).write_text(json.dumps({"study": "x", "files": {}}))
    with pytest.raises(SnapshotIntegrityError):
        load(d)


def test_a_missing_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nothing")


def test_open_or_create_is_idempotent_and_verified(tmp_path):
    src = _source(tmp_path)
    first = open_or_create(tmp_path / "snap", src, study="unit")
    second = open_or_create(tmp_path / "snap", src, study="unit")
    assert first.content_sha256 == second.content_sha256
    assert first.frozen_at == second.frozen_at


# ---------------------------------------------------------------------------
# the study itself replays from the snapshot
# ---------------------------------------------------------------------------


@pytest.fixture
def v3search(monkeypatch):
    import v3search

    return v3search


def _temp_source(tmp_path: Path) -> Path:
    """A small live-source dir containing the real ledger and account state."""
    src = tmp_path / "live"
    src.mkdir()
    live_ledger = ROOT / "var" / "ledger.jsonl"
    if not live_ledger.exists():
        pytest.skip("live ledger not present")
    shutil.copyfile(live_ledger, src / "ledger.jsonl")
    acct = ROOT / "var" / "paper_account.json"
    if acct.exists():
        shutil.copyfile(acct, src / "paper_account.json")
    return src


def test_load_universe_creates_and_replays_from_a_snapshot(
    tmp_path, v3search, monkeypatch
):
    src = _temp_source(tmp_path)
    monkeypatch.setattr(v3search, "SNAPSHOT_DIR", tmp_path / "snap")
    monkeypatch.setattr(
        v3search,
        "_snapshot_inputs",
        lambda: {
            "ledger.jsonl": src / "ledger.jsonl",
            "paper_account.json": src / "paper_account.json",
        },
    )
    universe1 = v3search.load_universe()[0]
    manifest = tmp_path / "snap" / MANIFEST_NAME
    assert manifest.exists(), "the study must freeze its inputs"
    assert json.loads(manifest.read_text())["files"]
    assert universe1, "the snapshot universe should not be empty"

    # appending to the LIVE source must not change the replay
    with open(src / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "note", "x": 1}) + "\n")
    universe2 = v3search.load_universe()[0]
    assert len(universe2) == len(universe1), (
        "a live append changed a frozen replay"
    )

    # mutating the SNAPSHOT is an integrity failure, not a silent re-read
    with open(tmp_path / "snap" / "ledger.jsonl", "a") as fh:
        fh.write('{"event":"note"}\n')
    with pytest.raises(SnapshotIntegrityError):
        v3search.load_universe()


def test_load_universe_no_longer_builds_an_empty_snapshot_at_call_time():
    """
    Structural: the old ``frozen_at=time.time(), files={}`` snapshot is gone
    from the default path.
    """
    src = (ROOT / "research" / "v3search.py").read_text()
    # the live fallback may only remain behind the explicit use_snapshot=False
    assert "use_snapshot: bool = True" in src
    assert "v3snapshot.open_or_create" in src
    assert "snap.as_paths_snapshot()" in src
