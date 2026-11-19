import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq
import numpy as np
import torch

from autoencoder.tobacco3482_cache import (
    extract_clip,
    extract_dino,
    extract_siglip,
)


LABELS = [
    "advertisement",
    "budget",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news_article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific_publication",
    "scientific_report",
    "specification",
]


def read_split(path):
    table = pq.read_table(path, columns=["image", "label"]).to_pylist()
    rows = []
    for index, row in enumerate(table):
        image = row["image"]
        entry = {
            "path": os.path.basename(path) + f":{index}",
            "label": int(row["label"]),
        }
        if isinstance(image, dict) and image.get("bytes") is not None:
            entry["bytes"] = image["bytes"]
        else:
            entry["array"] = np.asarray(image, dtype=np.uint8)
        rows.append(entry)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="dataset/rvl_large_balanced/data"
    )
    parser.add_argument("--out-dir", default="outputs/cache/rvl_large")
    parser.add_argument(
        "--siglip-model", default="external_models/siglip_base"
    )
    parser.add_argument(
        "--clip-model", default="openai/clip-vit-large-patch14"
    )
    parser.add_argument(
        "--dino-model", default="facebook/dinov2-base"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    names = sorted(os.listdir(args.data_dir))
    train_paths = [
        os.path.join(args.data_dir, name)
        for name in names
        if name.startswith("train-") and name.endswith(".parquet")
    ]
    test_paths = [
        os.path.join(args.data_dir, name)
        for name in names
        if name.startswith(("valid-", "test-"))
        and name.endswith(".parquet")
    ]
    train_rows = []
    for path in train_paths:
        train_rows.extend(read_split(path))
    test_rows = []
    for path in test_paths:
        test_rows.extend(read_split(path))
    rows = train_rows + test_rows
    train_indices = list(range(len(train_rows)))
    test_indices = list(range(len(train_rows), len(rows)))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    extractors = {
        "clip": (extract_clip, args.clip_model),
        "siglip": (extract_siglip, args.siglip_model),
        "dino": (extract_dino, args.dino_model),
    }
    features = {}
    for name, (extractor, model_path) in extractors.items():
        print(f"extracting {name}", flush=True)
        test_features = extractor(
            rows, test_indices, model_path, device, args.batch_size
        )
        train_features = extractor(
            rows, train_indices, model_path, device, args.batch_size
        )
        features[name] = (train_features, test_features)
    os.makedirs(args.out_dir, exist_ok=True)
    for split, indices in (
        ("train", train_indices),
        ("test", test_indices),
    ):
        cache = {
            name: values[0 if split == "train" else 1]
            for name, values in features.items()
        }
        cache.update(
            {
                "labels": torch.tensor(
                    [rows[index]["label"] for index in indices],
                    dtype=torch.long,
                ),
                "contents": [rows[index]["path"] for index in indices],
                "layouts": [0] * len(indices),
            }
        )
        path = os.path.join(args.out_dir, f"{split}.pt")
        torch.save(cache, path)
        print(split, len(indices), path, flush=True)
    metadata = {
        "dataset": "SBakkali/rvlcdip_subset",
        "labels": LABELS,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "split_overlap": 0,
    }
    with open(
        os.path.join(args.out_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
