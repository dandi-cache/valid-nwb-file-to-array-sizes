import json
import pathlib

import h5py
import numpy
import pytest
import update

pytestmark = pytest.mark.ai_generated


def _old_rows(file_path: pathlib.Path, /) -> list[dict]:
    """The rows as the walk computed them before: `visititems`, `h5o.get_info` and both storage calls."""
    rows = []
    with h5py.File(file_path, "r") as h5py_file:

        def _visit(path, node):
            if not isinstance(node, h5py.Dataset):
                return
            layout = update._LAYOUTS[node.id.get_create_plist().get_layout()]
            rows.append(
                {
                    "path": f"/{path}",
                    "storage_bytes": node.id.get_storage_size(),
                    "n_chunks": node.id.get_num_chunks() if layout == "chunked" else None,
                    "address": h5py.h5o.get_info(node.id).addr,
                }
            )

        h5py_file.visititems(_visit)
    return rows


def _write_example(file_path: pathlib.Path, /) -> pathlib.Path:
    with h5py.File(file_path, "w") as h5py_file:
        h5py_file.create_dataset("acquisition/series/data", data=numpy.arange(1000), chunks=(7,), compression="gzip")
        h5py_file.create_dataset("acquisition/series/unwritten", shape=(100,), chunks=(10,), dtype="f4")
        h5py_file.create_dataset("acquisition/series/timestamps", data=numpy.arange(10.0))
        h5py_file.create_dataset("processing/module/table", data=numpy.ones((20, 3)), chunks=(5, 3), shuffle=True)
        h5py_file.create_dataset("general/scalar", data=3)
        h5py_file.create_dataset("general/text", data="hello")
        h5py_file["analysis/same_data"] = h5py_file["acquisition/series/data"]
        h5py_file["analysis/soft"] = h5py.SoftLink("/acquisition/series/data")
        h5py_file["analysis/external"] = h5py.ExternalLink("absent.h5", "/x")
        h5py_file["acquisition/series/loop"] = h5py_file["acquisition"]
    return file_path


@pytest.fixture
def example_file(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_example(tmp_path / "example.h5")


class _FakeClient:
    def head_object(self, *, Bucket, Key):
        return {"ContentLength": 12345}


@pytest.fixture
def local_walk(monkeypatch: pytest.MonkeyPatch, example_file: pathlib.Path) -> pathlib.Path:
    """Point the walk at the local example file instead of the archive."""
    monkeypatch.setattr(update.dandi_cache.s3, "anonymous_client", lambda **_: _FakeClient())
    monkeypatch.setattr(update.dandi_cache.nwb, "open_hdf5", lambda url: (h5py.File(example_file, "r"), None))
    monkeypatch.setattr(update, "s3_range_reader", lambda content_id: _local_reader(example_file))
    return example_file


def test_hard_linked_objects_matches_visititems(example_file: pathlib.Path) -> None:
    expected = []
    with h5py.File(example_file, "r") as h5py_file:
        h5py_file.visititems(lambda path, node: expected.append((path, h5py.h5o.get_info(node.id).addr)))
        objects = update.hard_linked_objects(h5py_file)
    assert objects == expected


def test_walk_matches_the_old_walk(local_walk: pathlib.Path, tmp_path: pathlib.Path) -> None:
    record = update.walk_arrays("content-id", progress_path=str(tmp_path / "progress.jsonl"), measured={})

    rows = [{key: row[key] for key in ("path", "storage_bytes", "n_chunks", "address")} for row in record["arrays"]]
    assert rows == _old_rows(local_walk)
    assert record["n_arrays"] == 6
    assert record["object_size_bytes"] == 12345


def test_walk_records_progress_for_chunked_arrays_only(local_walk: pathlib.Path, tmp_path: pathlib.Path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    record = update.walk_arrays("content-id", progress_path=str(progress_path), measured={})

    chunked = {
        row["path"]: [row["storage_bytes"], row["n_chunks"]] for row in record["arrays"] if row["n_chunks"] is not None
    }
    assert update.read_progress(progress_path) == chunked
    assert set(chunked) == {"/acquisition/series/data", "/acquisition/series/unwritten", "/processing/module/table"}


def test_a_resumed_walk_reads_no_index_already_measured(
    local_walk: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete = update.walk_arrays("content-id", progress_path=str(tmp_path / "first.jsonl"), measured={})
    measured = update.read_progress(tmp_path / "first.jsonl")

    def _refuse(array, /, **_):
        raise AssertionError(f"read the index of {array.name} again")

    monkeypatch.setattr(update, "measure_chunks", _refuse)
    resumed = update.walk_arrays("content-id", progress_path=str(tmp_path / "second.jsonl"), measured=measured)

    assert resumed == complete
    assert update.read_progress(tmp_path / "second.jsonl") == {}


def test_progress_of_an_interrupted_walk_is_kept_and_resumed(
    local_walk: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A walk stopped partway leaves what it measured, and the next attempt measures only the rest."""
    measure_chunks = update.measure_chunks
    calls = []

    def _stop_after_one(array, /, **kwargs):
        if calls:
            raise TimeoutError("exceeded 1200 s")
        calls.append(array.name)
        return measure_chunks(array, **kwargs)

    monkeypatch.setattr(update, "measure_chunks", _stop_after_one)
    dataset = _FakeDataset({})
    progress = update.WalkProgress(dataset)
    with pytest.raises(TimeoutError):
        update.walk_arrays(
            "content-id", progress_path=str(progress.path("content-id")), measured=progress.measured("content-id")
        )

    assert progress.collect("content-id") == 1
    progress.flush()
    assert list(dataset.written["content-id"]) == ["/acquisition/series/data"]

    resumed_calls = []

    def _count(array, /, **kwargs):
        resumed_calls.append(array.name)
        return measure_chunks(array, **kwargs)

    monkeypatch.setattr(update, "measure_chunks", _count)
    next_run = update.WalkProgress(dataset)
    record = update.walk_arrays(
        "content-id", progress_path=str(next_run.path("content-id")), measured=next_run.measured("content-id")
    )
    next_run.finished("content-id")
    next_run.flush()

    assert resumed_calls == ["/acquisition/series/unwritten", "/processing/module/table"]
    assert record["walk_status"] == "ok"
    assert dataset.written["content-id"] is None


class _FakeDataset:
    """The two output calls `WalkProgress` makes, over a plain mapping."""

    def __init__(self, written: dict) -> None:
        self.written = written

    def iter_output(self, name: str, /):
        assert name == update.PROGRESS_OUTPUT
        yield from list(self.written.items())

    def merge_output_lookup(self, records: dict, name: str, /):
        assert name == update.PROGRESS_OUTPUT
        self.written.update(json.loads(json.dumps(records)))


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([], {}),
        (['{"/a": [10, 1]}', '{"/b": [20, 2]}'], {"/a": [10, 1], "/b": [20, 2]}),
        (['{"/a": [10, 1]}', '{"/b": [2'], {"/a": [10, 1]}),
    ],
)
def test_read_progress_ignores_a_line_cut_short(tmp_path: pathlib.Path, lines: list[str], expected: dict) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    assert update.read_progress(path) == expected


def test_progress_is_cleared_only_for_files_that_had_some() -> None:
    dataset = _FakeDataset({"slow": {"/a": [10, 1]}, "done-before": None})
    progress = update.WalkProgress(dataset)
    assert progress.measured("slow") == {"/a": [10, 1]}
    assert progress.measured("done-before") == {}

    progress.finished("slow")
    progress.finished("never-slow")
    progress.flush()

    assert dataset.written == {"slow": None, "done-before": None}


def test_a_failed_walk_with_nothing_new_writes_nothing() -> None:
    dataset = _FakeDataset({"slow": {"/a": [10, 1]}})
    progress = update.WalkProgress(dataset)

    assert progress.collect("slow") == 1
    progress.flush()

    assert dataset.written == {"slow": {"/a": [10, 1]}}


def test_abandoned_walks_are_collected(tmp_path: pathlib.Path) -> None:
    dataset = _FakeDataset({})
    progress = update.WalkProgress(dataset)
    progress.path("abandoned").write_text('{"/a": [10, 1]}\n', encoding="utf-8")
    progress.path("succeeded").write_text('{"/b": [20, 2]}\n', encoding="utf-8")
    progress.finished("succeeded")

    progress.collect_abandoned()
    progress.flush()

    assert dataset.written == {"abandoned": {"/a": [10, 1]}}


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [
        (update.WalkFailed("not_hdf5", "no blob"), "not_hdf5"),
        (TimeoutError("exceeded 1200 s"), "timeout"),
        (OSError("bad read"), "error"),
    ],
)
def test_failure_record_is_unchanged(exception: BaseException, expected_status: str) -> None:
    scope = type("Scope", (), {"exception": exception})()
    assert update.failure_record("content-id", scope)["walk_status"] == expected_status


def _measure_one_then_hang(progress_path: str, /) -> None:
    import time

    with open(progress_path, "a", encoding="utf-8") as progress:
        progress.write('{"/a": [10, 1]}\n')
        progress.flush()
        progress.write('{"/b": [2')
        progress.flush()
        time.sleep(60)


def test_progress_survives_the_child_being_killed_at_its_timeout(tmp_path: pathlib.Path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    with pytest.raises(TimeoutError):
        update.dandi_cache.run_isolated(_measure_one_then_hang, arguments=(str(progress_path),), timeout_seconds=2)
    assert update.read_progress(progress_path) == {"/a": [10, 1]}


def _local_reader(file_path: pathlib.Path, /):
    def read_ranges(byte_ranges):
        with open(file_path, "rb") as file:
            chunks = []
            for offset, length in byte_ranges:
                file.seek(offset)
                chunks.append(file.read(length))
            return chunks

    return read_ranges


@pytest.mark.parametrize(
    ("shape", "chunks", "compression"),
    [
        ((1000,), (7,), "gzip"),
        ((100_000,), (3,), None),  # Several levels of B-tree nodes.
        ((300, 40, 6), (2, 7, 5), "gzip"),
        ((50, 8, 8, 4, 3), (2, 3, 3, 2, 1), "gzip"),
    ],
)
def test_btree_totals_match_hdf5(tmp_path: pathlib.Path, shape: tuple, chunks: tuple, compression: str | None) -> None:
    file_path = tmp_path / "btree.h5"
    random = numpy.random.default_rng(0)
    with h5py.File(file_path, "w") as h5py_file:
        h5py_file.create_dataset(
            "data", data=random.integers(0, 4, size=shape, dtype="int16"), chunks=chunks, compression=compression
        )
    with h5py.File(file_path, "r") as h5py_file:
        array = h5py_file["data"]
        totals = update.btree_chunk_totals(
            _local_reader(file_path), object_address=h5py.h5o.get_info(array.id).addr, file_id=h5py_file.id
        )
        assert totals == (array.id.get_storage_size(), array.id.get_num_chunks())


def test_btree_totals_of_an_unwritten_array_are_zero(tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "empty.h5"
    with h5py.File(file_path, "w") as h5py_file:
        h5py_file.create_dataset("data", shape=(100,), chunks=(10,), dtype="f4")
    with h5py.File(file_path, "r") as h5py_file:
        address = h5py.h5o.get_info(h5py_file["data"].id).addr
        totals = update.btree_chunk_totals(_local_reader(file_path), object_address=address, file_id=h5py_file.id)
    assert totals == (0, 0)


def test_other_indexes_are_left_to_hdf5(tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "latest.h5"
    with h5py.File(file_path, "w", libver="latest") as h5py_file:
        h5py_file.create_dataset("data", data=numpy.arange(1000), chunks=(7,), maxshape=(None,))
    with h5py.File(file_path, "r") as h5py_file:
        array = h5py_file["data"]
        address = h5py.h5o.get_info(array.id).addr
        assert update.btree_chunk_totals(_local_reader(file_path), object_address=address, file_id=h5py_file.id) is None
        measured = update.measure_chunks(array, address=address, read_ranges=_local_reader(file_path))
        assert measured == (array.id.get_storage_size(), array.id.get_num_chunks())
