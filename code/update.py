"""The size, layout and storage of every array in every valid NWB file.

An "array" here is an HDF5 dataset object. The word "dataset" is avoided on purpose, since this
organization uses it in the HDF5 sense and its consumers use it for a dandiset.

Each file is streamed from its content-addressed blob and walked once. The walk visits every
object once by its HDF5 address through hard links and follows neither soft nor external links,
which is the tree `visititems` visits. That is the same tree `valid-nwb-file-to-number-of-datasets`
counts, so this cache's `n_arrays` can be checked against it entry for entry.

The walk never calls `visititems` or `h5o.get_info`, though. Both ask HDF5 for an object's
`meta_size`, and for a chunked array that reads its whole chunk index, so with `get_num_chunks` and
`get_storage_size` the walk used to read each index five times. Now each index is read once. One
kept as a version 1 B-tree, which `h5py` writes by default, is read here, a level of nodes at a time
with many ranged requests in flight. HDF5 reads one node at a time, at about 50 ms each, and an
array with two million chunks has some 30,000 nodes. Any other kind of index is read by HDF5 with
`chunk_iter`.

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
import struct
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

#: Concurrent ranged reads one file's chunk index is read with. A version 1 B-tree is a few levels of
#: nodes, each a few kilobytes, and HDF5 reads them one after another: at about 50 ms a request, the
#: 30,000 leaves of an array with two million chunks take 25 minutes. Read a level at a time, this
#: many at once, they take about a minute.
INDEX_READERS = 16

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


def measure_chunks(array, /, *, address: int | None = None, read_ranges=None) -> tuple[int, int]:
    """The bytes and the number of chunks a chunked array has allocated, from one read of its index.

    An index kept as a version 1 B-tree, which is what `h5py` writes by default, is read directly,
    with `read_ranges` reading many nodes at once. Any other index is read by HDF5 through
    `chunk_iter`, which reports every allocated chunk with its stored size in a single pass, where
    `get_storage_size()` and `get_num_chunks()` would each read the whole index.
    """
    if read_ranges is not None and address is not None:
        totals = btree_chunk_totals(read_ranges, object_address=address, file_id=array.file.id)
        if totals is not None:
            return totals

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


class NotReadable(Exception):
    """The object header or the index is not in a form `btree_chunk_totals` reads, so HDF5 should."""


#: Entries in a full raw-data-chunk B-tree node unless the file chose otherwise, twice HDF5's
#: default `K` of 32. A node is first read at this size, and read again if it holds more.
_DEFAULT_ENTRIES = 64
#: Nodes held in memory at once: a level of 30,000 leaves would otherwise take about 150 MB.
_NODES_PER_READ = 1024
_HEADER_GUESS_BYTES = 4096
_UNDEFINED_ADDRESS = 0xFFFFFFFFFFFFFFFF
_LAYOUT_MESSAGE = 0x0008
_CONTINUATION_MESSAGE = 0x0010


def btree_chunk_totals(read_ranges, /, *, object_address: int, file_id) -> tuple[int, int] | None:
    """`(storage_bytes, n_chunks)` of a chunked array whose index is a version 1 B-tree, or `None`.

    `read_ranges` takes a list of `(offset, length)` and returns their bytes in the same order. The
    tree's leaf keys carry each chunk's stored size, so the sum over them is `get_storage_size()`
    and their number `get_num_chunks()`. `None` means the index is of another kind, or the header
    could not be read here, and HDF5 should read it instead.
    """
    if file_id.get_create_plist().get_sizes() != (8, 8):
        return None
    try:
        layout = _chunk_layout(read_ranges, object_address=object_address)
        if layout is None:
            return None
        btree_address, key_dimensions = layout
        return _sum_btree(read_ranges, root=btree_address, key_dimensions=key_dimensions)
    except (NotReadable, struct.error):
        return None


def _chunk_layout(read_ranges, /, *, object_address: int) -> tuple[int, int] | None:
    """The B-tree address and key dimensionality from a version 3 layout message, else `None`."""
    for message_type, data in _header_messages(read_ranges, object_address=object_address):
        if message_type != _LAYOUT_MESSAGE:
            continue
        version, layout_class = data[0], data[1]
        if version != 3 or layout_class != 2:
            return None
        key_dimensions = data[2]
        (btree_address,) = struct.unpack_from("<Q", data, 3)
        return btree_address, key_dimensions
    return None


def _header_messages(read_ranges, /, *, object_address: int):
    """Yield `(type, data)` for each message of a version 1 object header, following continuations.

    A version 2 header is written only with the newer file format, whose chunked arrays have
    version 4 layouts and other kinds of index, so it is left to HDF5.
    """
    (header,) = read_ranges([(object_address, _HEADER_GUESS_BYTES)])
    if len(header) < 16 or header[0] != 1:
        raise NotReadable("not a version 1 object header")
    (header_size,) = struct.unpack_from("<I", header, 8)
    first = header[16 : 16 + header_size] if 16 + header_size <= len(header) else None
    blocks = [(object_address + 16, header_size, first)]
    while blocks:
        offset, length, block = blocks.pop(0)
        if block is None:
            (block,) = read_ranges([(offset, length)])
        if len(block) < length:
            raise NotReadable("object header cut short")
        position = 0
        while position + 8 <= length:
            message_type, size = struct.unpack_from("<HH", block, position)
            data = block[position + 8 : position + 8 + size]
            if message_type == _CONTINUATION_MESSAGE:
                continuation_offset, continuation_length = struct.unpack_from("<QQ", data, 0)
                blocks.append((continuation_offset, continuation_length, None))
            yield message_type, data
            position += 8 + size


def _sum_btree(read_ranges, /, *, root: int, key_dimensions: int) -> tuple[int, int]:
    """Walk a raw-data-chunk B-tree a level at a time, summing its leaf keys' chunk sizes."""
    if root == _UNDEFINED_ADDRESS:
        return 0, 0
    key_bytes = 8 + 8 * key_dimensions
    # The size of a full node at HDF5's default of 64 entries; a larger one is read again.
    guess = 24 + _DEFAULT_ENTRIES * (key_bytes + 8) + key_bytes
    storage_bytes = n_chunks = 0
    level = [root]
    while level:
        children = []
        for start in range(0, len(level), _NODES_PER_READ):
            addresses = level[start : start + _NODES_PER_READ]
            nodes = read_ranges([(address, guess) for address in addresses])
            short = [index for index, node in enumerate(nodes) if len(node) < _node_bytes(node, key_bytes=key_bytes)]
            rereads = read_ranges(
                [(addresses[index], _node_bytes(nodes[index], key_bytes=key_bytes)) for index in short]
            )
            for index, node in zip(short, rereads):
                nodes[index] = node
            for node in nodes:
                if node[:4] != b"TREE" or node[4] != 1:
                    raise NotReadable("not a raw-data-chunk B-tree node")
                node_level = node[5]
                (entries,) = struct.unpack_from("<H", node, 6)
                for entry in range(entries):
                    key_offset = 24 + entry * (key_bytes + 8)
                    if node_level == 0:
                        (chunk_size,) = struct.unpack_from("<I", node, key_offset)
                        storage_bytes += chunk_size
                        n_chunks += 1
                    else:
                        (child,) = struct.unpack_from("<Q", node, key_offset + key_bytes)
                        children.append(child)
        level = children
    return storage_bytes, n_chunks


def _node_bytes(node: bytes, /, *, key_bytes: int) -> int:
    """The size of a B-tree node from its header: every key and child, and the final key."""
    if len(node) < 8:
        raise NotReadable("B-tree node cut short")
    (entries,) = struct.unpack_from("<H", node, 6)
    return 24 + entries * (key_bytes + 8) + key_bytes


def s3_range_reader(content_id: str, /):
    """A `read_ranges` for one blob: ranged GETs, `INDEX_READERS` at a time."""
    client = dandi_cache.s3.anonymous_client(max_pool_connections=INDEX_READERS)
    key = dandi_cache.s3.blob_key(content_id)

    def _read(byte_range: tuple[int, int]) -> bytes:
        offset, length = byte_range
        response = client.get_object(
            Bucket=dandi_cache.s3.BUCKET, Key=key, Range=f"bytes={offset}-{offset + length - 1}"
        )
        return response["Body"].read()

    def read_ranges(byte_ranges: list[tuple[int, int]]) -> list[bytes]:
        return dandi_cache.s3.concurrent_map(_read, byte_ranges, max_workers=INDEX_READERS)

    return read_ranges


def describe_array(
    path: str, array, *, address: int, measured: tuple[int, int] | None = None, read_ranges=None
) -> dict:
    """One array's row: where it sits, its shape and type, and how its bytes are stored.

    `address` is passed in rather than read here, since `h5o.get_info` would read a chunked array's
    whole index for a figure the row does not use. `measured` is a chunked array's
    `(storage_bytes, n_chunks)` from an earlier, interrupted walk of the same file, and `read_ranges`
    reads the file's bytes directly, for `measure_chunks`.
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
        storage_bytes, n_chunks = (
            measured if measured is not None else measure_chunks(array, address=address, read_ranges=read_ranges)
        )
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
    read_ranges = s3_range_reader(content_id)
    h5py_file, _remote_file = dandi_cache.nwb.open_hdf5(dandi_cache.s3.blob_url(content_id))
    with h5py_file, open(progress_path, "a", encoding="utf-8") as progress:
        for path, address in hard_linked_objects(h5py_file):
            node = h5py_file[path]
            if not isinstance(node, h5py.Dataset):
                continue
            known = measured.get(f"/{path}")
            row = describe_array(
                path, node, address=address, measured=tuple(known) if known else None, read_ranges=read_ranges
            )
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
