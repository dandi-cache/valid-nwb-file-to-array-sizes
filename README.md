# DANDI Cache: `valid-nwb-file-to-array-sizes`

A mapping from the content ID of every valid HDF5 NWB file on the DANDI archive to a description of every array in that file: where it sits, its shape and type, and how its bytes are stored.

An **array** here is an HDF5 dataset object.
This README never uses the bare word "dataset", because it means an HDF5 object in some of this organization's repositories and a dandiset in their consumers.
A **dandiset** is a DANDI collection of files, identified by a six-digit ID.

The set of valid NWB files is taken from the [`content-id-to-valid-nwb-file`](https://github.com/dandi-cache/content-id-to-valid-nwb-file) cache, restricted to the entries it marked `true`.
Each file is streamed from its content-addressed blob on the public DANDI S3 bucket, `https://dandiarchive.s3.amazonaws.com/blobs/<c[:3]>/<c[3:6]>/<content_id>`, with [remfile](https://github.com/flatironinstitute/remfile) and read with [h5py](https://www.h5py.org/).
The DANDI REST API is not used.

The per-file scalars derived from these rows are published by [`valid-nwb-file-to-byte-weighted-structure`](https://github.com/dandi-cache/valid-nwb-file-to-byte-weighted-structure).

## How a file is walked

- Every array is visited once, by its HDF5 object address, through hard links.
  Soft and external links are not followed, so an array reachable by several paths gives one row, at the first path that reaches it.
  This is `h5py`'s `visititems`, and it is the same tree [`valid-nwb-file-to-number-of-datasets`](https://github.com/dandi-cache/valid-nwb-file-to-number-of-datasets) counts, which is what makes `n_arrays` checkable against it.
- The object's size comes from an S3 `HEAD` request on the blob.
- `get_storage_size()` on a chunked array reads its chunk index over HTTP, which for a file with hundreds of thousands of chunks takes many minutes.
  Each file is therefore walked in a child process with a 20-minute timeout.
  A file past it is recorded with `walk_status` `timeout` rather than holding up the run.
- A run works through content IDs never tried before, in order, and then retries the ones recorded with a retryable failure (`timeout` or `error`).
  So a run only does work for content IDs absent from the cache or marked retryable, and a slow file cannot block the rest of the archive.
- Eight files are walked at once, each in its own child process, and a run takes up to 1,000.
  A run checkpoints its results every 200 files, so a killed run loses at most 200 files of work.
- The `Update` workflow runs every six hours, and while a backlog remains each run queues the next one itself.

## What is recorded

Each line of the derivatives is a single-entry mapping:

```json
{"<content_id>": {"walk_status": "ok", "object_size_bytes": 84086336, "n_arrays": 48, "n_zero_byte_arrays": 0, "total_logical_bytes": 83886560, "total_storage_bytes": 83886960, "arrays": [...]}}
```

### Per file

| Field | Meaning |
|---|---|
| `walk_status` | `ok`; `timeout` (the walk exceeded 20 minutes; retried later); `error` (the walk raised; retried later); or `not_hdf5` (no blob exists under the content ID, so it is a Zarr asset; not retried). |
| `reason` | On a failure only: what went wrong. |
| `object_size_bytes` | The blob's size in bytes, from S3 `HEAD`. |
| `n_arrays` | Number of arrays (rows in `arrays`). |
| `n_zero_byte_arrays` | Number of arrays with `storage_bytes` of 0. |
| `total_logical_bytes` | Sum of `logical_bytes` over the arrays. |
| `total_storage_bytes` | Sum of `storage_bytes` over the arrays. |
| `arrays` | One row per array, below. |

A failed file has `walk_status`, `reason` and, when the `HEAD` request got that far, `object_size_bytes`, and nothing else.

The HDF5 library version that wrote a file is not recorded.
HDF5 does not store it.
The nearest thing in the file, the superblock version, is not exposed by `h5py`.

### Per array

| Field | Meaning |
|---|---|
| `path` | The path that first reached the array, e.g. `/acquisition/ElectricalSeries/data`. |
| `section` | The top-level group: `acquisition`, `processing`, `analysis`, `stimulus`, `intervals`, `units`, `general`, `scratch` or `specifications`. An array at the root, or under any other top-level group, is `other`. |
| `subsection` | For `processing` and `general` only, the second-level name (a processing module, `extracellular_ephys`, and so on). Otherwise `null`. |
| `shape` | The array's shape as a list; `[]` for a scalar, `null` for an empty dataspace. |
| `dtype` | The element type as a string. Variable-length types are named `vlen-str-utf-8`, `vlen-str-ascii` or `vlen-<base>`, and references `object-reference` or `region-reference`, rather than numpy's `object`. |
| `itemsize` | HDF5's size of one element, in bytes. |
| `logical_bytes` | `prod(shape) * itemsize`; 0 for an empty dataspace. |
| `storage_bytes` | `get_storage_size()`: the bytes allocated to the array in the file, after compression. |
| `layout` | `contiguous`, `chunked`, `compact` or `virtual`. |
| `chunk_shape` | The chunk shape for a chunked array, otherwise `null`. |
| `n_chunks` | The number of allocated chunks for a chunked array, otherwise `null`. |
| `filters` | The filter pipeline in order, each `{"id", "name", "options"}`; `[]` if none. |
| `address` | The array's HDF5 object address, unique within the file, for deduplicating. |

`logical_bytes` and `itemsize` match what `h5ls -v` reports as "logical bytes".
For a variable-length type, HDF5's element size is the in-memory handle while the file stores a larger descriptor per element, so such an array can show more `storage_bytes` than `logical_bytes`.
The data a variable-length element points at lives in the file's global heap, which neither figure counts.

## Dependencies

Nothing is added to the runtime image.
The walk needs `h5py` and `remfile`, and the `HEAD` request needs `boto3`, all of which the `:nwb` base image from [`dandi-cache-utils`](https://github.com/dandi-cache/dandi-cache-utils) already carries.

## Validation

Checked on a sample of 200 valid NWB files, one from each of 200 dandisets drawn at random, before anything was published archive-wide.

A second sample was checked the same way: the first 200 valid content IDs in sorted order, which span 65 dandisets.
These are exactly the files a run limited to 200 publishes first, so every published row of that run has been checked.
All 200 walked without a failure, with 0 mismatches against the array count, and the slowest took 2.8 minutes.

The results for the random sample:

- **Walks.** 197 succeeded and 3 timed out at 20 minutes, all three large imaging files (from dandisets 000402, 000692 and 000981). There were no other failures.
- **Array count.** `n_arrays` equals the value in [`valid-nwb-file-to-number-of-datasets`](https://github.com/dandi-cache/valid-nwb-file-to-number-of-datasets), which counts arrays despite its name, for all 197 walked files: 0 mismatches.
- **Bytes.** `total_storage_bytes` is at most `object_size_bytes` for every file.
- **Duplicates.** No file has two rows with the same address or the same path.
- **Against `h5ls -rv`.** Two files were downloaded and compared array by array against `h5ls -rv` (HDF5 1.10.10): an intracellular file of 1,347 arrays, mostly chunked with shuffle and deflate, and one of 918 contiguous arrays. Path, address, storage bytes, logical bytes, chunk shape and filter IDs agree for every array. The one exception in each file is an object-reference column, whose storage `h5ls` reports as "information not available".
- **Time.** The median file took 3.5 seconds, but the 90th percentile took 53 and the slowest walk that finished took 19 minutes. Time follows the number of chunks: the files with 130,000 to 256,000 chunks took 7 to 19 minutes.

## Not covered yet

- **Zarr assets.** Their arrays and chunk sizes come from S3 object listings rather than from HDF5, so they need a walk of their own. Today a Zarr asset is recorded as `not_hdf5`.
- **Attribute and object-header bytes per object.** The file-level remainder, `object_size_bytes - total_storage_bytes`, covers them in aggregate, and `valid-nwb-file-to-byte-weighted-structure` reports it as `metadata_fraction`.

## Archive-wide storage

At about 71 KB of JSON per file and about 118,000 valid NWB files, the full cache is about 8.4 GB, and about 360 MB compressed.
GitHub refuses any file over 100 MiB, so the cache is kept as a [Zarr](https://zarr.dev/) store, `derivatives/valid_nwb_file_to_array_sizes.zarr/`, that only grows by appending.
It is one array of JSON Lines text, in chunk files of at most 80 MB, `stream/c/0`, `stream/c/1`, ...
The chunks are plain text and each ends on a whole entry, so git shows a run as the lines it added to the last chunk and any new ones, and any chunk can be read on its own.
An entry that a later run re-records, such as a retried timeout, is a new line, and the last line for a content ID is the current one; the store is rewritten without the older lines once a fifth of them are superseded.
`dist` publishes a copy with the older lines removed and every chunk gzip-compressed, about 3 MB each, so each content ID appears once.
There is no single compressed file of the whole: at 360 MB it is over GitHub's limit.



## One-time use

`dist` publishes each output as a [Zarr](https://zarr.dev/) store: the cache itself is `derivatives/valid_nwb_file_to_array_sizes.zarr/`.
A store is one array of [JSON Lines](https://jsonlines.org/) text cut into chunk files, each holding whole lines and gzip-compressed.
Where a key appears more than once, the last line is its current entry, and the published copy has no repeats.
It can be read with the standard library alone, or opened remotely with any Zarr library.

### Python API (recommended)

The chunk files are gzip-compressed [JSON Lines](https://jsonlines.org/): list them, then read them in order as you would any JSONL files.

```python
import gzip, json, urllib.request

chunks = "https://api.github.com/repos/dandi-cache/valid-nwb-file-to-array-sizes/contents/derivatives/valid_nwb_file_to_array_sizes.zarr/stream/c?ref=dist"
files = sorted((entry["download_url"] for entry in json.load(urllib.request.urlopen(chunks))), key=lambda url: int(url.rsplit("/", 1)[1]))
valid_nwb_file_to_array_sizes = {key: value for url in files for line in gzip.decompress(urllib.request.urlopen(url).read()).decode().splitlines() if line for key, value in json.loads(line).items()}
```

One content ID's entry is in a single file, named by the first two digits of its ID:

```python
content_id = "..."
url = f"{base}/valid_nwb_file_to_array_sizes_{content_id[:2]}.jsonl.gz"
```

An entry is not filed under its key: the chunks hold the entries in the order they were recorded, so looking up one means reading through them.
To look up many, read it once as above, or clone the `derivatives` branch.

### Remote Zarr store

Anything that opens Zarr can open the store over HTTP, reading only the chunks it needs.
The group's `zarr.json` carries the array's metadata, so opening it takes one request:

```bash
pip install "zarr>=3" fsspec aiohttp
```

```python
import json

import zarr

url = "https://raw.githubusercontent.com/dandi-cache/valid-nwb-file-to-array-sizes/refs/heads/dist/derivatives/valid_nwb_file_to_array_sizes.zarr"
stream = zarr.open_group(url, mode="r")["stream"]
print(stream.shape, stream.chunks)  # Bytes in all, and bytes per chunk.

# The first chunk, fetched and decompressed on its own: whole lines, each `{"<key>": <value>}`.
first_chunk = stream[: stream.chunks[0]].tobytes().decode("utf-8")
for line in first_chunk.splitlines():
    if line.strip():
        print(json.loads(line))
        break
```

Reading `stream[:]` returns the whole text; to read it in pieces, step through it by `stream.chunks[0]`, as each chunk is fetched whole.

### Save to file

A raw URL cannot be listed, so to fetch every chunk at once clone the `dist` branch, which holds nothing else:

```bash
git clone --branch dist --depth 1 https://github.com/dandi-cache/valid-nwb-file-to-array-sizes.git
```



## Repeated use

If you plan on using this cache regularly, clone the `derivatives` branch of this repository:

```bash
git clone --branch derivatives https://github.com/dandi-cache/valid-nwb-file-to-array-sizes.git
```

Or, if you prefer [DataLad](https://www.datalad.org/):

```bash
datalad clone https://github.com/dandi-cache/valid-nwb-file-to-array-sizes.git --branch derivatives
```

The `derivatives` branch also keeps the log of every update under `logs/`, next to the results it produced.

Each output is a store on the `derivatives` branch, where it is the whole log, uncompacted, and plain text:

```python
import json, pathlib

chunks = sorted(pathlib.Path("valid-nwb-file-to-array-sizes/derivatives/valid_nwb_file_to_array_sizes.zarr/stream/c").iterdir(), key=lambda path: int(path.name))
valid_nwb_file_to_array_sizes = {key: value for chunk in chunks for line in chunk.read_text().splitlines() if line for key, value in json.loads(line).items()}
```
