"""Audit exact image overlap across and within RVL-CDIP parquet corpora.

Hashes are computed from decoded RGB pixels rather than file names or encoded
bytes, so re-encoding and metadata changes cannot hide an overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def canonical_hash(value) -> str:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path") is not None:
            value = value["path"]
        else:
            raise TypeError(f"unsupported image mapping keys: {sorted(value)}")

    if isinstance(value, str):
        with Image.open(value) as image:
            rgb = image.convert("RGB")
            payload = rgb.tobytes()
            size = rgb.size
    elif isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image:
            rgb = image.convert("RGB")
            payload = rgb.tobytes()
            size = rgb.size
    elif isinstance(value, Image.Image):
        rgb = value.convert("RGB")
        payload = rgb.tobytes()
        size = rgb.size
    else:
        array = np.asarray(value)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.ndim == 3 and array.shape[-1] == 4:
            array = array[..., :3]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"unsupported image shape: {array.shape}")
        payload = np.ascontiguousarray(array).tobytes()
        size = (array.shape[1], array.shape[0])

    digest = hashlib.sha256()
    digest.update(f"{size[0]}x{size[1]}:RGB:".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def split_name(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith("train-"):
        return "train"
    if name.startswith(("validation-", "valid-", "test-")):
        return "test"
    return "other"


def audit_root(name: str, root: str) -> dict:
    occurrences: dict[str, list[dict]] = {}
    files = []
    rows = 0
    for filename in sorted(os.listdir(root)):
        if not filename.endswith(".parquet"):
            continue
        path = os.path.join(root, filename)
        split = split_name(path)
        file_rows = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["image"], batch_size=64):
            for value in batch.column(0).to_pylist():
                digest = canonical_hash(value)
                occurrences.setdefault(digest, []).append(
                    {"split": split, "file": filename, "row": file_rows}
                )
                file_rows += 1
        rows += file_rows
        files.append({"name": filename, "split": split, "rows": file_rows})

    train_hashes = {
        digest
        for digest, items in occurrences.items()
        if any(item["split"] == "train" for item in items)
    }
    test_hashes = {
        digest
        for digest, items in occurrences.items()
        if any(item["split"] == "test" for item in items)
    }
    duplicate_groups = [
        {"sha256": digest, "occurrences": items}
        for digest, items in occurrences.items()
        if len(items) > 1
    ]
    return {
        "name": name,
        "root": root,
        "rows": rows,
        "unique_images": len(occurrences),
        "files": files,
        "split_overlap_count": len(train_hashes & test_hashes),
        "duplicate_groups": duplicate_groups,
        "_hashes": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        action="append",
        nargs=2,
        metavar=("NAME", "ROOT"),
        required=True,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    audits = [audit_root(name, root) for name, root in args.corpus]
    cross = []
    for left_index, left in enumerate(audits):
        for right in audits[left_index + 1 :]:
            shared = set(left["_hashes"]) & set(right["_hashes"])
            cross.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "overlap_count": len(shared),
                    "hashes": sorted(shared),
                }
            )

    for audit in audits:
        audit.pop("_hashes")
    report = {
        "hash_definition": "sha256(size + decoded RGB8 pixel bytes)",
        "corpora": audits,
        "cross_corpus": cross,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
