"""The size, layout and storage of every array in every valid NWB file.

An "array" here is an HDF5 dataset object. The word "dataset" is avoided on purpose, since this
organization uses it in the HDF5 sense and its consumers use it for a dandiset.

Each file is streamed from its content-addressed blob and walked once. The walk visits every
object once by its HDF5 address through hard links and follows neither soft nor external links,
which is the tree `visititems` visits. That is the same tree `valid-nwb-file-to-number-of-datasets`
counts, so this cache's `n_arrays` can be checked against it entry for entry.

The walk never calls `visititems` or `h5o.get_info`, though. Both ask HDF5 for an object's
`meta_size`, and for a chunked array that reads its whole chunk index, which is what the storage
size needs anyway. Reading the index once per array, through `chunk_iter`, is what makes the
heaviest files possible at all: the walk used to read each index five times.

What this cache does that its siblings do not:

- It publishes failures. A file that could not be walked gets a row whose `walk_status` says why,
  rather than being left out, under the `RETRY` policy. A retryable status is selected again by a
  later run, after every content ID that has never been tried, so a slow file cannot hold up the
  rest of the archive.
- It bounds each file with a timeout. The storage size of a chunked array comes from its chunk
  index, read over HTTP, and a file with a large index can take far longer than its neighbours.
  The walk runs through `run_isolated`, in a child process, so a stuck read can be stopped without
  leaving `h5py` in a broken state.
- It keeps the work of a walk that ran out of time. Each chunked array's storage figures are
  written to a local progress file as soon as they are read, and a failed walk's are published in
  a second output, `PROGRESS_OUTPUT`. The next attempt at that file reuses them and reads only the
  indexes it has not read yet, so a file too heavy for one sitting is finished over several runs.

Everything shared with the other caches -- the argument parsing, the logging, the frontier and its
retries, the batch cap, the timeout, the error logs, the output paths and testing mode -- comes
from `dandi_cache_utils`, which the runtime image carries.
"""

import functools
import json
import math
import pathlib
import tempfile
import threading

import dandi_cache_utils as dandi_cache

#: Seconds one file may take, from the HEAD request to the end of the walk.
#:
#: Set from the validation sample, where the slowest successful file took about 10 minutes. A file
#: past it is recorded as `timeout` and retried by a later run.
WALK_TIMEOUT_SECONDS = 20 * 60

#: Top-level NWB groups reported by name. Anything else, including arrays at the root, is `other`.
SECTIONS = frozenset(
    {
        "acquisition",
        "processing",
        "analysis",
        "stimulus",
        "intervals",
        "units",
        "general",
        "scratch",
        "specifications",
    }
)
#: Sections whose second-level name is recorded as `subsection`, since they hold many unrelated
#: things side by side (a processing module, an optophysiology or extracellular ephys group).
SUBSECTIONED = frozenset({"processing", "general"})

#: Files walked at once. A walk spends nearly all of its time waiting on HTTP reads of one file's
#: chunk index, in its own child process, so eight at a time keeps a runner's four cores and 16 GB
#: well within reach while cutting a batch's wall-clock time about eightfold.
WORKERS = 8

#: Statuses a later run selects again. `not_hdf5` is not one: it is a Zarr asset, which this cache
#: does not read, and retrying would only fail the same way.
RETRYABLE = frozenset({"timeout", "error"})

#: The second output: for each file whose walk has not yet finished, the storage figures of the
#: chunked arrays it has already measured, as `{path: [storage_bytes, n_chunks]}`. A file whose walk
#: succeeds has its entry replaced by `null`, and compaction drops the superseded lines.
PROGRESS_OUTPUT = "valid_nwb_file_to_array_sizes_walk_progress.jsonl"

_LAYOUTS = {0: "compact", 1: "contiguous", 2: "chunked", 3: "virtual"}


class WalkFailed(Exception):
    """A file that could not be walked, carrying the status and reason to record for it.

    Every field is passed through to `Exception`, which is what lets it cross back from the child
    process intact: an exception is pickled as its class and its `args`.
    """

    def __init__(self, status: str, reason: str, object_size_bytes: int | None = None) -> None:
        super().__init__(status, reason, object_size_bytes)
        self.status = status
        self.reason = reason
        self.object_size_bytes = object_size_bytes

    def __str__(self) -> str:
        return f"{self.status}: {self.reason}"


def describe_dtype(array) -> str:
    """The element type as a string, naming variable-length and reference types rather than `object`."""
    import h5py

    string_info = h5py.check_string_dtype(array.dtype)
    if string_info is not None and string_info.length is None:
        return f"vlen-str-{string_info.encoding}"
    vlen_base = h5py.check_vlen_dtype(array.dtype)
    if vlen_base is not None:
        return f"vlen-{vlen_base}"
    reference = h5py.check_ref_dtype(array.dtype)
    if reference is not None:
        return "region-reference" if reference is h5py.RegionReference else "object-reference"
    return str(array.dtype)


def describe_filters(creation_properties) -> list[dict]:
    """Every filter in the pipeline, in order, as `{id, name, options}`."""
    filters = []
    for index in range(creation_properties.get_nfilters()):
        filter_id, _flags, options, name = creation_properties.get_filter(index)
        filters.append(
            {
                "id": int(filter_id),
                "name": name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name),
                "options": [int(option) for option in options],
            }
        )
    return filters


def measure_chunks(array, /) -> tuple[int, int]:
    """The bytes and the number of chunks a chunked array has allocated, from one read of its index.

    `get_storage_size()` and `get_num_chunks()` each read the whole index, so asking both reads it
    twice. `chunk_iter` reports every allocated chunk with its stored size in a single pass.
    """
    totals = [0, 0]

    def _count(chunk) -> None:
        totals[0] += chunk.size
        totals[1] += 1

    try:
        array.id.chunk_iter(_count)
    except (AttributeError, NotImplementedError):
        # An h5py built against an HDF5 older than 1.12.3 has no `H5Dchunk_iter`.
        return array.id.get_storage_size(), array.id.get_num_chunks()
    storage_bytes, n_chunks = totals
    return storage_bytes, n_chunks


def describe_array(path: str, array, *, address: int, measured: tuple[int, int] | None = None) -> dict:
    """One array's row: where it sits, its shape and type, and how its bytes are stored.

    `address` is passed in rather than read here, since `h5o.get_info` would read a chunked array's
    whole index for a figure the row does not use. `measured` is a chunked array's
    `(storage_bytes, n_chunks)` from an earlier, interrupted walk of the same file.
    """
    parts = path.split("/")
    section = parts[0] if len(parts) > 1 and parts[0] in SECTIONS else "other"
    subsection = parts[1] if section in SUBSECTIONED and len(parts) > 2 else None

    shape = list(array.shape) if array.shape is not None else None
    # HDF5's own size of one element, the same one `h5ls -v` multiplies out to its "logical bytes".
    # For a variable-length type that is the in-memory handle, while the file stores a larger
    # descriptor per element, so such an array can show more storage than logical bytes. The data the
    # descriptors point at lives in the global heap and is in neither figure.
    itemsize = array.id.get_type().get_size()
    logical_bytes = math.prod(shape) * itemsize if shape is not None else 0

    creation_properties = array.id.get_create_plist()
    layout = _LAYOUTS.get(creation_properties.get_layout(), "unknown")
    chunk_shape = list(array.chunks) if array.chunks is not None else None
    if layout == "chunked":
        storage_bytes, n_chunks = measured if measured is not None else measure_chunks(array)
    else:
        storage_bytes, n_chunks = array.id.get_storage_size(), None

    return {
        "path": f"/{path}",
        "section": section,
        "subsection": subsection,
        "shape": shape,
        "dtype": describe_dtype(array),
        "itemsize": itemsize,
        "logical_bytes": logical_bytes,
        "storage_bytes": storage_bytes,
        "layout": layout,
        "chunk_shape": chunk_shape,
        "n_chunks": n_chunks,
        "filters": describe_filters(creation_properties),
        "address": address,
    }


def walk_arrays(content_id: str, /, *, progress_path: str, measured: dict[str, list[int]]) -> dict:
    """HEAD the blob, then stream it and describe every array. Runs in a child process.

    Each chunked array's storage figures are appended to `progress_path` as soon as they are read,
    so they outlive the child when it is stopped at its timeout. `measured` holds the ones an
    earlier attempt at this file already read, by path, and those indexes are not read again.
    """
    import botocore.exceptions
    import h5py

    client = dandi_cache.s3.anonymous_client(max_pool_connections=2)
    try:
        head = client.head_object(Bucket=dandi_cache.s3.BUCKET, Key=dandi_cache.s3.blob_key(content_id))
    except botocore.exceptions.ClientError as error:
        if error.response.get("Error", {}).get("Code", "") in dandi_cache.s3.ABSENT_ERROR_CODES:
            raise WalkFailed("not_hdf5", "no blob under this content ID; a Zarr asset") from error
        raise
    object_size_bytes = head["ContentLength"]

    arrays = []
    h5py_file, _remote_file = dandi_cache.nwb.open_hdf5(dandi_cache.s3.blob_url(content_id))
    with h5py_file, open(progress_path, "a", encoding="utf-8") as progress:
        for path, address in hard_linked_objects(h5py_file):
            node = h5py_file[path]
            if not isinstance(node, h5py.Dataset):
                continue
            known = measured.get(f"/{path}")
            row = describe_array(path, node, address=address, measured=tuple(known) if known else None)
            if row["layout"] == "chunked" and known is None:
                progress.write(json.dumps({row["path"]: [row["storage_bytes"], row["n_chunks"]]}) + "\n")
                progress.flush()
            arrays.append(row)

    return {
        "walk_status": "ok",
        "object_size_bytes": object_size_bytes,
        "n_arrays": len(arrays),
        "n_zero_byte_arrays": sum(1 for array in arrays if array["storage_bytes"] == 0),
        "total_logical_bytes": sum(array["logical_bytes"] for array in arrays),
        "total_storage_bytes": sum(array["storage_bytes"] for array in arrays),
        "arrays": arrays,
    }


def hard_linked_objects(h5py_file, /) -> list[tuple[str, int]]:
    """Every object reachable through hard links, once, at the first path that reaches it, in walk order.

    This is the order and the set `visititems` gives: both walk the links by name, depth first, and
    neither follows a soft or external link. It is read from the links alone, whose information
    holds the target's address, so no object is opened and no chunk index is read to list them.
    """
    import h5py

    paths: list[str] = []
    h5py_file.visit_links(paths.append)
    objects = []
    seen_addresses: set[int] = set()
    for path in paths:
        link = h5py_file.id.links.get_info(path.encode("utf-8"))
        if link.type != h5py.h5l.TYPE_HARD:
            continue
        address = link.u
        if address in seen_addresses:
            continue
        seen_addresses.add(address)
        objects.append((path, address))
    return objects


class WalkProgress:
    """The storage figures already read for files whose walks have not finished, and their updates.

    Read once at the start of a run. Worker threads only read from it, apart from `finished`; the
    output is written on the calling thread, from `collect` and `flush`, which the runner's
    `on_error` and `on_write` call there.
    """

    def __init__(self, dataset, /) -> None:
        self.dataset = dataset
        self.recorded: dict[str, dict[str, list[int]]] = {
            content_id: value for content_id, value in dataset.iter_output(PROGRESS_OUTPUT) if value
        }
        self.directory = pathlib.Path(tempfile.mkdtemp(prefix="walk_progress_"))
        self.pending: dict[str, dict[str, list[int]] | None] = {}
        self.collected: set[str] = set()
        self._finished: set[str] = set()
        self._lock = threading.Lock()

    def path(self, content_id: str, /) -> pathlib.Path:
        """The local file a walk of this content ID appends its progress to."""
        return self.directory / f"{content_id}.jsonl"

    def measured(self, content_id: str, /) -> dict[str, list[int]]:
        """What earlier attempts at this file already read, by array path."""
        return self.recorded.get(content_id, {})

    def finished(self, content_id: str, /) -> None:
        """Note a walk that succeeded, so its stored progress, if any, is cleared."""
        with self._lock:
            self._finished.add(content_id)

    def collect(self, content_id: str, /) -> int:
        """Queue what a failed walk of this file read, with what earlier attempts read; returns the total."""
        self.collected.add(content_id)
        merged = dict(self.measured(content_id))
        merged.update(read_progress(self.path(content_id)))
        if len(merged) > len(self.measured(content_id)):
            self.pending[content_id] = merged
        return len(merged)

    def collect_abandoned(self) -> None:
        """Queue what walks still running when the batch stopped had read, the runner having left them."""
        with self._lock:
            finished = set(self._finished)
        for path in sorted(self.directory.glob("*.jsonl")):
            content_id = path.stem
            if content_id not in finished and content_id not in self.collected:
                self.collect(content_id)

    def flush(self) -> None:
        """Write what was queued, and clear the stored progress of files whose walks have since succeeded."""
        with self._lock:
            finished = set(self._finished)
        for content_id in finished:
            if content_id in self.recorded:
                self.pending[content_id] = None
        if not self.pending:
            return
        self.dataset.merge_output_lookup(self.pending, PROGRESS_OUTPUT)
        for content_id, value in self.pending.items():
            if value:
                self.recorded[content_id] = value
            else:
                self.recorded.pop(content_id, None)
        self.pending = {}


def read_progress(path: pathlib.Path, /) -> dict[str, list[int]]:
    """The storage figures a walk appended to its progress file, ignoring a last line cut short by its kill."""
    measured: dict[str, list[int]] = {}
    if not path.exists():
        return measured
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            measured.update(json.loads(line))
        except json.JSONDecodeError:
            continue
    return measured


def measure_file(content_id: str, item, *, progress: WalkProgress) -> dict:
    """Walk one file in a child process, stopped at `WALK_TIMEOUT_SECONDS`, resuming from `progress`."""
    item.stage = "reading the NWB file"
    # Spawned rather than forked: with `WORKERS` threads running, a forked child could inherit a lock
    # another thread held at that moment, still held, and hang on it until the timeout.
    walk = functools.partial(
        walk_arrays, progress_path=str(progress.path(content_id)), measured=progress.measured(content_id)
    )
    record = dandi_cache.run_isolated(
        walk,
        arguments=(content_id,),
        timeout_seconds=WALK_TIMEOUT_SECONDS,
        start_method="spawn",
    )
    progress.finished(content_id)
    return record


def failure_record(content_id: str, scope) -> dict:
    """The row published for a file that could not be walked, from what the walk raised."""
    exception = scope.exception
    if isinstance(exception, WalkFailed):
        status, reason, object_size_bytes = exception.status, exception.reason, exception.object_size_bytes
    elif isinstance(exception, TimeoutError):
        # The walk's own limit, or a read that timed out inside it. Either may clear on a later run.
        status, reason, object_size_bytes = "timeout", str(exception), None
    else:
        status, reason, object_size_bytes = "error", f"{type(exception).__name__}: {exception}", None
    return {"walk_status": status, "reason": reason, "object_size_bytes": object_size_bytes}


def main() -> None:
    dataset, arguments = dandi_cache.open_dataset()

    # Only the assets the upstream cache marked valid are measured.
    validity = dataset.read_input()
    valid_content_ids = [content_id for content_id, is_valid in validity.items() if is_valid is True]

    progress = WalkProgress(dataset)

    def _process(content_id: str, item) -> dict:
        return measure_file(content_id, item, progress=progress)

    def _on_error(content_id: str, scope) -> None:
        if isinstance(scope.exception, WalkFailed):
            return
        total = progress.collect(content_id)
        if total:
            dandi_cache.logger.info("%s: kept %d measured chunked arrays for the next attempt.", content_id, total)

    dandi_cache.run_incremental_update(
        dataset,
        candidates=valid_content_ids,
        process=_process,
        limit=dataset.limit(arguments.limit),
        # The failure is published with its reason, and a timeout or an error is tried again once
        # every content ID never tried has had its turn. A Zarr asset is published and left alone.
        on_failure=dandi_cache.RETRY,
        failure_value=failure_record,
        retry_when=lambda record: record["walk_status"] in RETRYABLE,
        # A walk that ran out of time keeps what it measured, and the next attempt starts from there.
        on_error=_on_error,
        on_write=progress.flush,
        stages={"reading the NWB file": "file_read_errors.txt"},
        describe=lambda record: f"{record['n_arrays']} arrays, {record['total_storage_bytes'] / 1e6:.1f} MB stored",
        # A killed run loses at most this many files of work.
        checkpoint_every=200,
        workers=WORKERS,
        # The cache is never loaded whole: at full coverage it would take some 40 GB in memory. Each
        # checkpoint merges what it measured into only the files of the 256 those results belong to.
        in_memory=False,
    )
    # A walk still running when the batch hit its time limit was left by the runner, unrecorded, but
    # what it had measured is still worth keeping.
    progress.collect_abandoned()
    progress.flush()
    dataset.compact_output_if_superseded(PROGRESS_OUTPUT, minimum=100)


if __name__ == "__main__":
    main()
