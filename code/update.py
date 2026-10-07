"""The size, layout and storage of every array in every valid NWB file.

An "array" here is an HDF5 dataset object. The word "dataset" is avoided on purpose, since this
organization uses it in the HDF5 sense and its consumers use it for a dandiset.

Each file is streamed from its content-addressed blob and walked once. The walk is `visititems`,
which visits every object once by its HDF5 address through hard links and follows neither soft nor
external links. That is the same tree `valid-nwb-file-to-number-of-datasets` counts, so this cache's
`n_arrays` can be checked against it entry for entry.

What this cache does that its siblings do not:

- It records failures. A file that could not be walked gets a row whose `walk_status` says why,
  rather than being left out. A retryable status is selected again by a later run, after every
  content ID that has never been tried, so a slow file cannot hold up the rest of the archive.
- It bounds each file with a timeout. `get_storage_size()` on a chunked array reads the chunk index
  over HTTP, and a file with a large index can take far longer than its neighbours. The walk runs
  in a child process so a stuck read can be stopped without leaving `h5py` in a broken state.

Everything shared with the other caches -- the argument parsing, the logging, the batch cap, the
error logs, the output paths and testing mode -- comes from `dandi_cache_utils`, which the runtime
image carries.
"""

import math
import multiprocessing

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

#: Statuses a later run selects again. `not_hdf5` is not one: it is a Zarr asset, which this cache
#: does not read, and retrying would only fail the same way.
RETRYABLE = frozenset({"timeout", "error"})

_LAYOUTS = {0: "compact", 1: "contiguous", 2: "chunked", 3: "virtual"}


class WalkFailed(Exception):
    """A file that could not be walked, carrying the status and reason to record for it."""

    def __init__(self, status: str, reason: str, /, *, object_size_bytes: int | None = None) -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason
        self.object_size_bytes = object_size_bytes


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


def describe_array(path: str, array) -> dict:
    """One array's row: where it sits, its shape and type, and how its bytes are stored."""
    import h5py

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
    n_chunks = array.id.get_num_chunks() if layout == "chunked" else None

    return {
        "path": f"/{path}",
        "section": section,
        "subsection": subsection,
        "shape": shape,
        "dtype": describe_dtype(array),
        "itemsize": itemsize,
        "logical_bytes": logical_bytes,
        "storage_bytes": array.id.get_storage_size(),
        "layout": layout,
        "chunk_shape": chunk_shape,
        "n_chunks": n_chunks,
        "filters": describe_filters(creation_properties),
        "address": h5py.h5o.get_info(array.id).addr,
    }


def walk_arrays(content_id: str) -> dict:
    """HEAD the blob, then stream it and describe every array. Runs in the child process."""
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
    seen_addresses: set[int] = set()

    def _visit(path: str, node) -> None:
        if not isinstance(node, h5py.Dataset):
            return
        # `visititems` already visits each object once; the guard makes that explicit.
        address = h5py.h5o.get_info(node.id).addr
        if address in seen_addresses:
            return
        seen_addresses.add(address)
        arrays.append(describe_array(path, node))

    h5py_file, _remote_file = dandi_cache.nwb.open_hdf5(dandi_cache.s3.blob_url(content_id))
    with h5py_file:
        h5py_file.visititems(_visit)

    return {
        "walk_status": "ok",
        "object_size_bytes": object_size_bytes,
        "n_arrays": len(arrays),
        "n_zero_byte_arrays": sum(1 for array in arrays if array["storage_bytes"] == 0),
        "total_logical_bytes": sum(array["logical_bytes"] for array in arrays),
        "total_storage_bytes": sum(array["storage_bytes"] for array in arrays),
        "arrays": arrays,
    }


def _walk_in_child(content_id: str, connection) -> None:
    """Child-process entry point: send back the record, or the failure as a status and reason."""
    try:
        connection.send(("ok", walk_arrays(content_id)))
    except WalkFailed as failure:
        connection.send(("failed", (failure.status, failure.reason, failure.object_size_bytes)))
    except BaseException as error:  # noqa: BLE001 -- everything the walk raises becomes a recorded reason
        connection.send(("failed", ("error", f"{type(error).__name__}: {error}", None)))
    finally:
        connection.close()


def measure_file(content_id: str, item) -> dict:
    """Walk one file in a child process, stopping it at `WALK_TIMEOUT_SECONDS`."""
    item.stage = "reading the NWB file"
    # `fork`, so the child inherits the imported library rather than importing it again per file.
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_walk_in_child, args=(content_id, sender), daemon=True)
    child.start()
    sender.close()
    try:
        if not receiver.poll(WALK_TIMEOUT_SECONDS):
            raise WalkFailed("timeout", f"walk exceeded {WALK_TIMEOUT_SECONDS} s")
        try:
            outcome, payload = receiver.recv()
        except EOFError as error:
            child.join(5)
            raise WalkFailed("error", f"walker exited with code {child.exitcode}") from error
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        receiver.close()

    if outcome == "ok":
        return payload
    status, reason, object_size_bytes = payload
    raise WalkFailed(status, reason, object_size_bytes=object_size_bytes)


def failure_record(failure: WalkFailed) -> dict:
    """The row recorded for a file that could not be walked."""
    return {
        "walk_status": failure.status,
        "reason": failure.reason,
        "object_size_bytes": failure.object_size_bytes,
    }


def select_batch(valid_content_ids: list[str], records: dict, limit: int | None) -> list[str]:
    """Never-tried content IDs first, in order, then retryable failures, capped at `limit`."""
    untried = dandi_cache.select_new(valid_content_ids, records)
    retryable = sorted(
        content_id
        for content_id, record in records.items()
        if isinstance(record, dict) and record.get("walk_status") in RETRYABLE
    )
    batch = untried + retryable
    return batch if limit is None else batch[:limit]


def main() -> None:
    dataset, arguments = dandi_cache.open_dataset()

    # Only the assets the upstream cache marked valid are measured.
    validity = dataset.read_input()
    valid_content_ids = [content_id for content_id, is_valid in validity.items() if is_valid is True]

    records = dataset.read_output_lookup()
    batch = select_batch(valid_content_ids, records, dataset.limit(arguments.limit))

    def _record_failure(content_id: str, scope) -> None:
        # The runner leaves a failure unrecorded (`SKIP`), so the batch counts it as failed rather
        # than new and an all-failure batch does not queue another run. The row is written here
        # instead, so the reason is published and a retryable one is selected again later.
        exception = scope.exception
        if isinstance(exception, WalkFailed):
            records[content_id] = failure_record(exception)
        else:
            records[content_id] = failure_record(WalkFailed("error", f"{type(exception).__name__}: {exception}"))

    dandi_cache.run_incremental_update(
        dataset,
        batch=batch,
        recorded=records,
        process=measure_file,
        on_failure=dandi_cache.SKIP,
        on_error=_record_failure,
        stages={"reading the NWB file": "file_read_errors.txt"},
        describe=lambda record: f"{record['n_arrays']} arrays, {record['total_storage_bytes'] / 1e6:.1f} MB stored",
        checkpoint_every=50,
    )


if __name__ == "__main__":
    main()
