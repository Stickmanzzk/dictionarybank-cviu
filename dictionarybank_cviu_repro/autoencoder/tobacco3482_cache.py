import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


LABELS = [
    "ADVE",
    "Email",
    "Form",
    "Letter",
    "Memo",
    "News",
    "Note",
    "Report",
    "Resume",
    "Scientific",
]


def load_rows(parquet_dir):
    paths = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith(".parquet")
    )
    if not paths:
        raise RuntimeError(f"no parquet files in {parquet_dir}")
    rows = []
    for path in paths:
        table = pq.read_table(path, columns=["image", "label"]).to_pylist()
        for row in table:
            rows.append(
                {
                    "path": row["image"]["path"],
                    "bytes": row["image"]["bytes"],
                    "label": int(row["label"]),
                }
            )
    return rows


def split_rows(rows, seed, train_fraction):
    generator = np.random.default_rng(seed)
    train, test = [], []
    for label in sorted({row["label"] for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["label"] == label]
        )
        indices = indices[generator.permutation(len(indices))]
        cut = int(round(len(indices) * train_fraction))
        train.extend(indices[:cut].tolist())
        test.extend(indices[cut:].tolist())
    return np.asarray(train), np.asarray(test)


def decode_images(rows, indices, batch_size, transform=None):
    for start in range(0, len(indices), batch_size):
        batch_rows = [rows[int(index)] for index in indices[start:start + batch_size]]
        images = []
        for row in batch_rows:
            if "array" in row:
                image = Image.fromarray(row["array"]).convert("RGB")
            else:
                image = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
            images.append(image)
        if transform is not None:
            images = [transform(image) for image in images]
        yield images


def extract_siglip(
    rows, indices, model_path, device, batch_size, transform=None
):
    from transformers import AutoProcessor, SiglipModel

    model = SiglipModel.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True
    )
    features = []
    with torch.inference_mode():
        for images in tqdm(
            decode_images(rows, indices, batch_size, transform),
            total=(len(indices) + batch_size - 1) // batch_size,
            desc="siglip",
        ):
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model.vision_model(pixel_values=inputs["pixel_values"])
            features.append(
                F.normalize(output.pooler_output.float(), dim=1).cpu()
            )
    del model
    torch.cuda.empty_cache()
    return torch.cat(features)


def extract_clip(
    rows, indices, model_path, device, batch_size, transform=None
):
    from transformers import AutoProcessor, CLIPVisionModel

    model = CLIPVisionModel.from_pretrained(model_path).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    features = []
    with torch.inference_mode():
        for images in tqdm(
            decode_images(rows, indices, batch_size, transform),
            total=(len(indices) + batch_size - 1) // batch_size,
            desc="clip",
        ):
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            features.append(
                F.normalize(output.pooler_output.float(), dim=1).cpu()
            )
    del model
    torch.cuda.empty_cache()
    return torch.cat(features)


def extract_dino(
    rows, indices, model_path, device, batch_size, transform=None
):
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(model_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_path)
    features = []
    with torch.inference_mode():
        for images in tqdm(
            decode_images(rows, indices, batch_size, transform),
            total=(len(indices) + batch_size - 1) // batch_size,
            desc="dino",
        ):
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            features.append(
                F.normalize(output.pooler_output.float(), dim=1).cpu()
            )
    del model
    torch.cuda.empty_cache()
    return torch.cat(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-dir",
        default="dataset/tobacco3482_hf/data",
    )
    parser.add_argument(
        "--out-dir", default="outputs/cache/tobacco3482"
    )
    parser.add_argument(
        "--siglip-model", default="external_models/siglip_base"
    )
    parser.add_argument(
        "--clip-model", default="openai/clip-vit-large-patch14"
    )
    parser.add_argument(
        "--dino-model", default="facebook/dinov2-base"
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rows = load_rows(args.parquet_dir)
    train_indices, test_indices = split_rows(
        rows, args.seed, args.train_fraction
    )
    if set(train_indices.tolist()) & set(test_indices.tolist()):
        raise RuntimeError("train/test row overlap")
    features = {}
    extractors = {
        "clip": (extract_clip, args.clip_model),
        "siglip": (extract_siglip, args.siglip_model),
        "dino": (extract_dino, args.dino_model),
    }
    for name, (extractor, model_path) in extractors.items():
        print(f"extracting {name} from {model_path}", flush=True)
        test_features = extractor(
            rows, test_indices, model_path, device, args.batch_size
        )
        train_features = extractor(
            rows, train_indices, model_path, device, args.batch_size
        )
        features[name] = (train_features, test_features)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, indices in (
        ("train", train_indices),
        ("test", test_indices),
    ):
        cache = {
            stream: values[0 if name == "train" else 1]
            for stream, values in features.items()
        }
        cache.update(
            {
                "labels": torch.tensor(
                    [rows[int(index)]["label"] for index in indices],
                    dtype=torch.long,
                ),
                "contents": [
                    rows[int(index)]["path"] for index in indices
                ],
                "layouts": [0] * len(indices),
            }
        )
        path = os.path.join(args.out_dir, f"{name}.pt")
        torch.save(cache, path)
        print(name, len(indices), path, flush=True)
    metadata = {
        "dataset": "maveriq/tobacco3482",
        "labels": LABELS,
        "rows": len(rows),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "split_overlap": len(
            set(train_indices.tolist()) & set(test_indices.tolist())
        ),
        "streams": sorted(features),
    }
    with open(
        os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
